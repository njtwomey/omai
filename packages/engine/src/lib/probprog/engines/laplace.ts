import { hessian } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { multivariateNormal, normal } from '../../dists/index.ts'
import { cholesky, inv, slogdet } from '../../numerics/linalg/index.ts'
import { lbfgs } from '../../optim/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
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
 * @throws WhiteboxError with code `value` when minus the Hessian at the mode is not positive definite (an improper or
 *   degenerate posterior).
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceFit, observe, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
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
  const mode = fromStorage(Float64Array.from(toFlat(primalOf(r.x as Value))), [um.dim], 'float64')
  const H = primalOf(hessian((u: Value) => um.fn(u))(mode))
  const precision = primalOf(neg(H))
  try {
    cholesky(precision)
  } catch {
    throw new WhiteboxError('value', 'laplace: minus the Hessian at the mode is not positive definite', {
      gradNorm: r.gradNorm,
    })
  }
  const covariance = primalOf(inv(precision))
  return {
    mode,
    covariance: primalOf(mul(add(covariance, transpose(covariance)), 0.5)),
    logDensity: -r.fun,
    gradNorm: r.gradNorm,
    iterations: r.steps,
    converged: r.converged,
  }
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
 * `failed` and the reason.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Initialisation, iteration cap, tolerance, key and tracing.
 * @returns The approximate posterior.
 * @concept laplace-approximation
 * @example
 * ```ts
 * import { laplaceApproximation, observe, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item } from 'whitebox-engine/tensor'
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
  const q = multivariateNormal(fit.mode, { covariance: fit.covariance })
  const { logabsdet } = slogdet(fit.covariance)
  const logEvidence = fit.logDensity + 0.5 * um.dim * Math.log(2 * Math.PI) + 0.5 * item(logabsdet)
  const marginals: Record<string, ReturnType<typeof normal>> = {}
  for (const l of um.layout) {
    if (l.support.kind !== 'real') continue
    const idx = Array.from({ length: l.size }, (_, i) => l.offset + i)
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
 * import { laplaceEngine } from 'whitebox-engine/probprog'
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
