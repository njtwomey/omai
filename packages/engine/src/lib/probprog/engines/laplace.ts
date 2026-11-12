/**
 * The Laplace approximation of a program's posterior: find the MAP point of the unconstrained log density with L-BFGS,
 * take the Hessian there, and use the normal with the inverse Hessian as covariance, pushed back onto each site's
 * support. The fit, the approximation and the engine for the planner.
 */

import { hessian } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { multivariateNormal, normal } from '../../dists/index.ts'
import { cholesky, eigvalsh, inv, slogdet } from '../../numerics/linalg/index.ts'
import { lbfgs } from '../../optim/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  arange,
  fromStorage,
  item,
  mul,
  neg,
  primalOf,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import {
  constrainDraws,
  findInitialParams,
  type InitStrategy,
  initToMedian,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** The mode and curvature of an unconstrained log density. */
export interface LaplaceFit {
  /** The MAP point on the unconstrained scale. */
  readonly mode: Tensor
  /** The covariance of the approximation: the inverse of minus the Hessian at the mode. */
  readonly covariance: Tensor
  /** The log density at the mode. */
  readonly logDensity: number
  readonly gradNorm: number
  readonly iterations: number
  readonly converged: boolean
}

/**
 * The MAP point of a program's unconstrained log density (L-BFGS from a valid start) and the Hessian there: the
 * pieces of a Laplace approximation (NumPyro's `AutoLaplaceApproximation` fitted exactly).
 *
 * @param um - The unconstrained model.
 * @param start - The starting point.
 * @param options - The iteration cap, gradient tolerance and tracing.
 * @returns The fit.
 * @throws OmaiError with code `value` when minus the Hessian at the mode is not positive definite (an improper or
 *   degenerate posterior).
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceFit, observe, potentialEnergy, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { toArray } from 'omai-engine/tensor'
 * const um = potentialEnergy(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }, undefined)
 * toArray(laplaceFit(um, [0]).mode).map((v) => Math.round(v * 1e8) / 1e8) // => [1]
 * ```
 */
export function laplaceFit(
  um: UnconstrainedModel,
  start: ArrayLike<number>,
  options: Pick<EngineRunOptions, 'tracer' | 'policy'> & {
    readonly maxIterations?: number
    readonly gradTol?: number
  } = {},
): LaplaceFit {
  const f = (u: Value): Value => neg(um.fn(u))
  const r = lbfgs(f, fromStorage(Float64Array.from(start), [um.dim], 'float64'), {
    maxIterations: options.maxIterations ?? 1000,
    gradTol: options.gradTol ?? 1e-10,
    // Stop on the gradient, not on a small relative decrease: the mode feeds the Hessian.
    ftol: 0,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return laplaceAt(um, toFlat(primalOf(r.x as Value)), {
    logDensity: -r.fun,
    gradNorm: r.gradNorm,
    iterations: r.steps,
    converged: r.converged,
  })
}

/**
 * The curvature stage of {@link laplaceFit}: the Hessian of the unconstrained log density at a point taken as the mode,
 * and the covariance from it. Together with an optimiser that finds the mode step by step it is the Laplace
 * approximation as a method with a fine step.
 *
 * @param um - The unconstrained model.
 * @param point - The mode (unconstrained).
 * @param run - What the optimisation reached: the log density at the point, the gradient norm, the iterations and whether
 *   it converged.
 * @returns The fit.
 * @throws OmaiError with code `value` when minus the Hessian at the point is not positive definite.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceAt, observe, potentialEnergy, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { toArray } from 'omai-engine/tensor'
 * const um = potentialEnergy(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }, undefined)
 * toArray(laplaceAt(um, [1], { logDensity: 0, gradNorm: 0, iterations: 0, converged: true }).covariance).map((v) => Math.round(v * 1e8) / 1e8) // => [0.5]
 * ```
 */
export function laplaceAt(
  um: UnconstrainedModel,
  point: ArrayLike<number>,
  run: Pick<LaplaceFit, 'logDensity' | 'gradNorm' | 'iterations' | 'converged'>,
): LaplaceFit {
  const mode = fromStorage(Float64Array.from(point), [um.dim], 'float64')
  const H = primalOf(hessian((u: Value) => um.fn(u))(mode))
  const precision = primalOf(neg(H))
  try {
    cholesky(precision)
  } catch {
    const boundary = boundaryCoordinates(um, toFlat(mode), run.converged)
    throw new OmaiError(
      'value',
      `laplace: minus the Hessian at the mode is not positive definite${
        boundary.length === 0
          ? ''
          : ` (the mode is on a boundary of the support: ${boundary.join(', ')} ran off to an extreme)`
      }`,
      { gradNorm: run.gradNorm, boundary },
    )
  }
  const covariance = primalOf(inv(precision))
  return {
    mode,
    covariance: primalOf(mul(add(covariance, transpose(covariance)), 0.5)),
    logDensity: run.logDensity,
    gradNorm: run.gradNorm,
    iterations: run.iterations,
    converged: run.converged,
  }
}

/** The unconstrained value beyond which a coordinate of a constrained site is read as having run to its boundary: e^-10 is 4.5e-5. */
const BOUNDARY_LIMIT = 10
/** The condition number of the covariance beyond which the curvature is not a usable description of the posterior. */
const CONDITION_LIMIT = 1e12

/**
 * The coordinates of constrained sites (a scale, a probability, an ordered gap) whose unconstrained value is beyond
 * {@link BOUNDARY_LIMIT} at a point the optimiser stopped at without converging: the signature of a density that keeps
 * increasing towards the boundary of the support (a hierarchical scale going to 0), where the mode is not a point of the
 * posterior. A point where the gradient vanished is a mode, however extreme, and is not reported.
 */
function boundaryCoordinates(um: UnconstrainedModel, point: ArrayLike<number>, converged: boolean): string[] {
  if (converged) return []
  const out: string[] = []
  for (const l of um.layout) {
    if (l.support.kind === 'real') continue
    for (let i = l.offset; i < l.offset + l.size; i++)
      if (Math.abs(point[i] ?? 0) > BOUNDARY_LIMIT)
        out.push(`${um.logDensity.names[i] ?? l.name} (unconstrained ${(point[i] ?? 0).toFixed(1)})`)
  }
  return out
}

/** What {@link diagnoseMode} found about a Laplace fit. */
export interface ModeDiagnosis {
  /** Whether the mode is not a point the normal approximation can describe: any of the findings below. */
  readonly degenerate: boolean
  /** Coordinates that ran off to the boundary of their support, with their unconstrained value (empty when none). */
  readonly boundary: readonly string[]
  /** max / min eigenvalue of the covariance (Infinity when it is not positive definite or not finite). */
  readonly conditionNumber: number
  /** Whether the mode or the covariance has a non-finite entry. */
  readonly nonFinite: boolean
  /** The findings in words, empty for a well-behaved fit. */
  readonly messages: readonly string[]
}

/**
 * Whether a Laplace fit sits at a usable mode, which it does not when the density increases without bound towards the
 * boundary of the support (the centred hierarchical posterior as its scale goes to 0: the density is unbounded there and
 * the optimiser runs to the floor of the scale). Reads the fit only, the maths is unchanged: a coordinate of a constrained
 * site at an extreme unconstrained value that the optimiser did not converge to (a log scale of -14), a covariance whose
 * condition number is beyond 1e12 (the Hessian is singular in some direction, as it is along such a ridge), and a
 * non-finite mode or covariance.
 *
 * @param um - The unconstrained model.
 * @param fit - The fit.
 * @returns The diagnosis.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { diagnoseMode, laplaceFit, observe, potentialEnergy, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const um = potentialEnergy(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }, undefined)
 * diagnoseMode(um, laplaceFit(um, [0])).degenerate // => false
 * ```
 */
export function diagnoseMode(um: UnconstrainedModel, fit: LaplaceFit): ModeDiagnosis {
  const mode = toFlat(fit.mode)
  const cov = toFlat(fit.covariance)
  const nonFinite = mode.some((v) => !Number.isFinite(v)) || cov.some((v) => !Number.isFinite(v))
  let conditionNumber = Number.POSITIVE_INFINITY
  if (!nonFinite) {
    const eig = toFlat(eigvalsh(fit.covariance))
    const low = eig[0] ?? 0
    const high = eig[eig.length - 1] ?? 0
    conditionNumber = low > 0 ? high / low : Number.POSITIVE_INFINITY
  }
  const boundary = boundaryCoordinates(um, mode, fit.converged)
  const messages: string[] = []
  if (boundary.length > 0)
    messages.push(
      `the mode is on a boundary of the support, not a point of the posterior: ${boundary.join(', ')} ran off to an extreme (the density is unbounded towards it, as the centred hierarchical posterior is as its scale goes to 0)`,
    )
  if (nonFinite) messages.push('the mode or the covariance has a non-finite entry')
  else if (!(conditionNumber < CONDITION_LIMIT))
    messages.push(
      `the covariance is ill-conditioned (condition number ${Number.isFinite(conditionNumber) ? conditionNumber.toExponential(1) : 'infinite'}): the Hessian is singular in some direction, so the normal approximation does not describe the posterior`,
    )
  return { degenerate: messages.length > 0, boundary, conditionNumber, nonFinite, messages }
}

/** Options of {@link laplaceApproximation}. */
export interface LaplaceOptions {
  readonly init?: InitStrategy
  readonly maxIterations?: number
  readonly gradTol?: number
}

/**
 * The Laplace approximation of a program's posterior: a normal on the unconstrained scale at the MAP point with the
 * inverse Hessian as covariance, pushed onto each site's support for draws, with the Laplace estimate of the evidence
 * log p(u*) + D/2 log 2 pi - 1/2 log det(-H). Degenerate programs (no valid start, a singular Hessian) end with status
 * `failed` and the reason. The diagnostics say whether the mode is a point of the posterior (see {@link diagnoseMode}):
 * `degenerateMode`, the `boundary` coordinates, the `conditionNumber` of the covariance and the `messages`.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Initialisation, iteration cap, tolerance, key and tracing.
 * @returns The approximate posterior.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceApproximation, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * Math.abs(item(laplaceApproximation(model, undefined, { key: key(0) }).variance('m')) - 0.5) < 1e-8 // => true
 * ```
 */
export function laplaceApproximation<D = void>(
  model: Model<D>,
  data: D,
  options: LaplaceOptions & EngineRunOptions,
): Posterior {
  const fail = (reason: string): Posterior =>
    definePosterior({
      form: 'variational',
      engine: 'laplace',
      sites: [],
      status: 'failed',
      draws: {},
      diagnostics: { reason },
    })
  let um: UnconstrainedModel
  try {
    um = potentialEnergy(model, data, { key: foldIn(options.key, 0) })
  } catch (e) {
    return fail(e instanceof Error ? e.message : String(e))
  }
  const init = findInitialParams(um, model, data, {
    key: foldIn(options.key, 1),
    strategy: options.init ?? initToMedian(),
  })
  if (!init.valid) return fail('no starting point with a finite log density and gradient')
  let fit: LaplaceFit
  try {
    fit = laplaceFit(um, init.unconstrained, {
      ...(options.maxIterations === undefined ? {} : { maxIterations: options.maxIterations }),
      ...(options.gradTol === undefined ? {} : { gradTol: options.gradTol }),
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
    })
  } catch (e) {
    return fail(e instanceof Error ? e.message : String(e))
  }
  return laplacePosterior(um, fit)
}

/**
 * The Laplace posterior of a fit: the normal at the mode with its covariance on the unconstrained scale, pushed through
 * each site's support for draws and marginals, with the Laplace estimate of the evidence log p(u*) + D/2 log 2 pi - 1/2 log
 * det(-H).
 *
 * @param um - The unconstrained model.
 * @param fit - The mode and the covariance ({@link laplaceFit} or {@link laplaceAt}).
 * @returns The posterior.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceFit, laplacePosterior, observe, potentialEnergy, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const um = potentialEnergy(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }, undefined)
 * Math.abs(item(laplacePosterior(um, laplaceFit(um, [0])).variance('m')) - 0.5) < 1e-8 // => true
 * ```
 */
export function laplacePosterior(um: UnconstrainedModel, fit: LaplaceFit): Posterior {
  const q = multivariateNormal(fit.mode, { covariance: fit.covariance })
  const { logabsdet } = slogdet(fit.covariance)
  const logEvidence = fit.logDensity + 0.5 * um.dim * Math.log(2 * Math.PI) + 0.5 * item(logabsdet)
  const marginals: Record<string, ReturnType<typeof normal>> = {}
  for (const l of um.layout) {
    if (l.support.kind !== 'real') continue
    const idx = toFlat(add(arange(l.size), l.offset))
    const cov = toFlat(fit.covariance)
    const m = toFlat(fit.mode)
    const sd = idx.map((i) => Math.sqrt(cov[i * um.dim + i] ?? Number.NaN))
    marginals[l.name] = normal(
      reshape(
        fromStorage(
          Float64Array.from(idx, (i) => m[i] ?? 0),
          [l.size],
          'float64',
        ),
        l.shape,
      ),
      reshape(fromStorage(Float64Array.from(sd), [l.size], 'float64'), l.shape),
    )
  }
  const mode = diagnoseMode(um, fit)
  return definePosterior({
    form: 'variational',
    engine: 'laplace',
    sites: [...um.layout.map((l) => l.name), ...um.enumerated],
    status: fit.converged ? 'converged' : 'completed',
    marginals,
    logEvidence: { value: logEvidence, standardError: Number.NaN, method: 'Laplace' },
    diagnostics: {
      logDensityAtMode: fit.logDensity,
      gradNorm: fit.gradNorm,
      iterations: fit.iterations,
      mode: toFlat(fit.mode),
      degenerateMode: mode.degenerate,
      conditionNumber: Number.isFinite(mode.conditionNumber) ? mode.conditionNumber : null,
      boundary: [...mode.boundary],
      messages: [...mode.messages],
    },
    sampler: (key: Key, n: number) => {
      const u = primalOf(q.sample(key, [n]))
      return constrainDraws(um, Float64Array.from(toFlat(u)))
    },
    logProb: (values) => {
      const u = fromStorage(um.unconstrain(values), [um.dim], 'float64')
      let lp = item(q.logProb(u))
      for (const l of um.layout) {
        const piece = reshape(sliceAxis(u, 0, l.offset, l.offset + l.size), l.unconstrainedShape)
        lp -= item(sum(l.bijector.forwardLogDetJacobian(piece)))
      }
      return lp
    },
  })
}

/**
 * The Laplace engine: a normal approximation at the MAP point (cost: optimisation plus one Hessian, about
 * dimension^2 gradient evaluations).
 *
 * @param options - Initialisation and optimiser options.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { laplaceEngine } from 'omai-engine/probprog'
 * laplaceEngine().name // => 'laplace'
 * ```
 */
export function laplaceEngine(options: LaplaceOptions = {}): InferenceEngine {
  return {
    name: 'laplace',
    form: 'variational',
    description: 'a normal approximation at the MAP point with the inverse Hessian as covariance',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'approximate', 0, 'observed sites have no likelihood')
      if (a.continuous.length === 0) return feasibility(false, 'approximate', 0, 'no continuous latent sites')
      if (!a.differentiable) return feasibility(false, 'approximate', 0, 'the log density is not differentiable')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'approximate', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      const d = Math.max(1, a.continuousDim)
      const enumCost = Math.max(1, Math.min(a.discrete.jointSize, 1e6))
      return feasibility(
        true,
        'approximate',
        (200 + d * d) * Math.max(1, a.observations) * enumCost,
        'differentiable density',
      )
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      laplaceApproximation(model, data, { ...options, ...run }),
  }
}
