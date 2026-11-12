/**
 * Pathfinder (Zhang, Carpenter, Gelman and Vehtari 2022): variational inference from the path of a quasi-Newton optimiser.
 * L-BFGS climbs the unconstrained log density from a random start; at every iterate the inverse Hessian estimate of the
 * optimiser gives a normal approximation (the Newton step as its mean, the estimate as its covariance); the ELBO of each is
 * estimated by Monte Carlo and the best one is kept. With several paths from different starts, the paths' best normals form a
 * mixture, from which importance draws are taken and weighted by Pareto-smoothed importance sampling (PSIS), which also
 * reports how far the mixture is from the posterior (the Pareto k-hat). One L-BFGS iteration of every path is one step.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { cholesky, solveTriangular } from '../../numerics/linalg/index.ts'
import { bfgsInverseUpdate, type LbfgsState, lbfgsAlgorithm, twoLoopRecursion } from '../../optim/index.ts'
import { choice, foldIn, type Key, key as makeKey, normal as normalDraws } from '../../random/index.ts'
import { importanceEss, paretoSmoothedWeights } from '../../sampling/index.ts'
import {
  add,
  concat,
  diagonal,
  fromStorage,
  item,
  log,
  logsumexp,
  matmul,
  mul,
  neg,
  primalOf,
  reshape,
  sub,
  sum,
  take,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels } from '../../trace/index.ts'
import { isolated } from '../_stack.ts'
import {
  constrainDraws,
  findInitialParams,
  initToMedian,
  initToUniform,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'

/** The normal approximation at one iterate of a path: the Newton step as its mean and the inverse Hessian estimate. */
export interface PathfinderGaussian {
  /** The path's iteration this approximation is from (0 is the start). */
  readonly iteration: number
  readonly elbo: number
  readonly mean: Float64Array
  /** The covariance, row by row. */
  readonly covariance: Float64Array
}

/** One L-BFGS path and what it has found so far. */
export interface PathfinderPath {
  readonly lbfgs: LbfgsState
  /** Iterations taken (0 right after the start). */
  readonly iteration: number
  /** The ELBO of the normal at the latest iterate (-Infinity when it could not be built). */
  readonly latest: number
  /** The approximation with the highest ELBO so far. */
  readonly best: PathfinderGaussian | null
  readonly finished: boolean
}

/** The state of {@link pathfinderAlgorithm}. */
export interface PathfinderState {
  readonly paths: readonly PathfinderPath[]
  readonly stage: 'path' | 'sample' | 'done'
  /** Importance draws on the unconstrained scale, row by row (`numDraws * D` values), once drawn. */
  readonly draws: Float64Array | null
  /** The Pareto-smoothed (or raw) log weights of the draws, normalised. */
  readonly logWeights: Float64Array | null
  readonly paretoK: number
  readonly ess: number
  /** The best ELBO over paths so far. */
  readonly elbo: number
}

/** Options of {@link pathfinderAlgorithm}. */
export interface PathfinderOptions {
  /** Independent paths from different starts (default 1; with more, the paths' best normals form a mixture). */
  readonly paths?: number
  /** L-BFGS iterations per path (default 50). */
  readonly maxIterations?: number
  /** Curvature pairs the L-BFGS history keeps, which is the rank of the covariance's correction (default 10). */
  readonly memory?: number
  /** Draws for each iterate's ELBO estimate (default 50). */
  readonly elboDraws?: number
  /** Draws taken from the final approximation (per path, then pooled; default 400). */
  readonly numDraws?: number
  /**
   * How the draws are weighted: `psis` keeps Pareto-smoothed importance weights (default), `resample` draws equally weighted
   * values from them with replacement, and `none` returns the draws of the approximation unweighted.
   */
  readonly weights?: 'psis' | 'resample' | 'none'
  /** Where the paths start: uniform on (-2, 2) of the unconstrained scale (default), or the median of the prior draws (the first path only). */
  readonly init?: 'uniform' | 'median'
  /** L-BFGS stops a path when the relative decrease of the objective falls below this (default 1e-9: effectively the iteration cap). */
  readonly ftol?: number
}

const dotted = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0)
  return s
}

/** The inverse Hessian estimate of an L-BFGS history as a dense matrix: the scaled identity updated by each pair, oldest first. */
function inverseHessian(s: readonly Float64Array[], y: readonly Float64Array[], n: number): Float64Array {
  const newest = s.length - 1
  const gamma =
    newest < 0
      ? 1
      : dotted(s[newest] as Float64Array, y[newest] as Float64Array) /
        dotted(y[newest] as Float64Array, y[newest] as Float64Array)
  // banned-ok: kernel: the inverse Hessian of the L-BFGS history, updated in place
  let h: Float64Array = new Float64Array(n * n)
  for (let i = 0; i < n; i++) h[i * n + i] = gamma
  for (let i = 0; i < s.length; i++) h = bfgsInverseUpdate(h, s[i] as Float64Array, y[i] as Float64Array)
  return h
}

/** The lower Cholesky factor of a covariance given row by row. */
const choleskyOf = (cov: Float64Array, D: number): Value => primalOf(cholesky(fromStorage(cov, [D, D], 'float64')))

/** `n` draws `mean + L eps` as rows `[n, D]`. */
const drawsFrom = (mean: Float64Array, L: Value, n: number, key: Key): Value => {
  const D = mean.length
  return add(fromStorage(mean, [D], 'float64'), matmul(normalDraws(key, [n, D]), transpose(L)))
}

/** log N(z; mean, L L^T) for the rows of `z` `[n, D]`, from the Cholesky factor. */
function normalLogDensity(z: Value, mean: Float64Array, L: Value): Value {
  const D = mean.length
  const centred = sub(z, fromStorage(mean, [D], 'float64'))
  const w = solveTriangular(L, transpose(centred), { lower: true })
  const quad = sum(mul(w, w), 0)
  return sub(mul(quad, -0.5), add(sum(log(diagonal(L))), 0.5 * D * Math.log(2 * Math.PI)))
}

interface Parts {
  readonly um: UnconstrainedModel
  readonly objective: (u: Value) => Value
  readonly stepper: Algorithm<LbfgsState>
}

/**
 * Pathfinder as a traced algorithm: every step is one L-BFGS iteration of each unfinished path with the ELBO of the
 * normal at the new iterate; when every path has ended, one step draws from the best normals (the mixture of them, for
 * several paths), weights the draws by Pareto-smoothed importance sampling and records the k-hat and the effective sample
 * size. The state serialises, so a run resumes and a snapshot of the best normal so far can be read at any step.
 *
 * @param model - The model.
 * @param data - Its data.
 * @param options - Paths, iterations, memory, draws and weighting.
 * @returns The algorithm.
 * @throws OmaiError with code `value` when no start has a finite log density.
 * @concept variational-inference
 * @index algorithm Pathfinder: normal approximations along an L-BFGS path, best ELBO, PSIS importance draws, multi-path.
 * @example
 * ```ts
 * import { observe, pathfinderAlgorithm, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * runAlgorithm(pathfinderAlgorithm(model, undefined), { seed: 0, maxSteps: 3 }).steps // => 3
 * ```
 */
export function pathfinderAlgorithm<D = void>(
  model: Model<D>,
  data: D,
  options: PathfinderOptions = {},
): Algorithm<PathfinderState> {
  const nPaths = options.paths ?? 1
  const maxIterations = options.maxIterations ?? 50
  const memory = options.memory ?? 10
  const elboDraws = options.elboDraws ?? 50
  const numDraws = options.numDraws ?? 400
  const mode = options.weights ?? 'psis'
  if (!(Number.isInteger(nPaths) && nPaths >= 1))
    throw new OmaiError('value', 'pathfinderAlgorithm: paths must be positive')
  let cached: Parts | undefined
  const partsOf = (): Parts => {
    if (cached !== undefined) return cached
    const um = isolated(() => potentialEnergy(model, data))
    const objective = (u: Value): Value => neg(um.fn(u))
    // banned-ok: kernel: the starting point handed to the optimiser kernel
    const stepper = lbfgsAlgorithm(objective, new Float64Array(um.dim), {
      memory,
      ftol: options.ftol ?? 1e-9,
      gradTol: 1e-8,
    })
    cached = { um, objective, stepper }
    return cached
  }
  /** The normal at an iterate, with its ELBO from `elboDraws` draws at `key`. */
  const gaussianAt = (st: LbfgsState, iteration: number, key: Key): PathfinderGaussian | null => {
    const { um } = partsOf()
    const D = um.dim
    const H = inverseHessian(st.s, st.y, D)
    // The Newton step of the log density: the objective is its negative, so the mean is x - H grad.
    const step = twoLoopRecursion(st.s, st.y, st.grad)
    const mean = Float64Array.from(st.x, (x, i) => x - (step[i] ?? 0))
    let L: Value
    try {
      L = choleskyOf(H, D)
    } catch {
      return null
    }
    const draws = drawsFrom(mean, L, elboDraws, key)
    const rows = toFlat(primalOf(draws))
    const lq = toFlat(primalOf(normalLogDensity(draws, mean, L)))
    let total = 0
    for (let i = 0; i < elboDraws; i++) {
      const v = item(primalOf(um.fn(fromStorage(Float64Array.from(rows.slice(i * D, (i + 1) * D)), [D], 'float64'))))
      total += (Number.isFinite(v) ? v : Number.NEGATIVE_INFINITY) - (lq[i] ?? 0)
    }
    const elbo = total / elboDraws
    return { iteration, elbo: Number.isFinite(elbo) ? elbo : Number.NEGATIVE_INFINITY, mean, covariance: H }
  }
  const better = (a: PathfinderGaussian | null, b: PathfinderGaussian | null): PathfinderGaussian | null =>
    a === null ? b : b === null ? a : b.elbo > a.elbo ? b : a
  const bestOverall = (paths: readonly PathfinderPath[]): number =>
    paths.reduce((m, p) => Math.max(m, p.best?.elbo ?? Number.NEGATIVE_INFINITY), Number.NEGATIVE_INFINITY)
  return {
    name: 'pathfinder',
    config: { paths: nPaths, maxIterations, memory, elboDraws, numDraws, weights: mode } as Json,
    schema: channels({
      metrics: {
        elbo: { description: 'the best ELBO over every path and iterate so far', direction: 'higher' },
        paretoK: {
          description: 'the Pareto k-hat of the importance weights of the final approximation',
          direction: 'lower',
        },
        ess: { description: 'effective sample size of the importance weights' },
      },
    }),
    init: (key) => {
      const { um, objective } = partsOf()
      const paths: PathfinderPath[] = []
      for (let p = 0; p < nPaths; p++) {
        const start = findInitialParams(um, model, data, {
          key: foldIn(key, p),
          strategy: options.init === 'median' && p === 0 ? initToMedian() : initToUniform(),
        })
        if (!start.valid)
          throw new OmaiError('value', 'pathfinderAlgorithm: no starting point with a finite log density')
        const alg = lbfgsAlgorithm(objective, Float64Array.from(start.unconstrained), {
          memory,
          ftol: options.ftol ?? 1e-9,
          gradTol: 1e-8,
        })
        const lbfgs = alg.init(foldIn(key, p))
        const first = gaussianAt(lbfgs, 0, foldIn(foldIn(key, p), 1 << 20))
        paths.push({
          lbfgs,
          iteration: 0,
          latest: first?.elbo ?? Number.NEGATIVE_INFINITY,
          best: first,
          finished: lbfgs.stop !== undefined,
        })
      }
      return {
        paths,
        stage: 'path',
        draws: null,
        logWeights: null,
        paretoK: Number.NaN,
        ess: Number.NaN,
        elbo: bestOverall(paths),
      }
    },
    step: (s, ctx) => {
      const { um, stepper } = partsOf()
      const D = um.dim
      if (s.stage === 'path') {
        const paths = s.paths.map((path, p): PathfinderPath => {
          if (path.finished) return path
          const lbfgs = stepper.step(path.lbfgs, ctx)
          const iteration = path.iteration + 1
          const g = gaussianAt(lbfgs, iteration, foldIn(foldIn(ctx.key, p), iteration))
          const finished = lbfgs.stop !== undefined || iteration >= maxIterations
          return { lbfgs, iteration, latest: g?.elbo ?? Number.NEGATIVE_INFINITY, best: better(path.best, g), finished }
        })
        const stage = paths.every((p) => p.finished) ? 'sample' : 'path'
        return { ...s, paths, stage, elbo: bestOverall(paths) }
      }
      if (s.stage === 'sample') {
        const components = s.paths.flatMap((p) => (p.best !== null && Number.isFinite(p.best.elbo) ? [p.best] : []))
        if (components.length === 0)
          throw new OmaiError('value', 'pathfinderAlgorithm: no path produced a usable normal approximation')
        const chols = components.map((c) => choleskyOf(c.covariance, D))
        const per = Math.max(1, Math.ceil(numDraws / components.length))
        const total = per * components.length
        const draws = concat(
          components.map((c, i) => drawsFrom(c.mean, chols[i] as Value, per, foldIn(ctx.key, i))),
          0,
        )
        // The proposal is the equal-weight mixture of the paths' best normals; each draw's density is the mixture's.
        const logq = sub(
          logsumexp(
            concat(
              components.map((c, i) => reshape(normalLogDensity(draws, c.mean, chols[i] as Value), [1, total])),
              0,
            ),
            0,
          ),
          Math.log(components.length),
        )
        const rows = toFlat(primalOf(draws))
        const lq = toFlat(primalOf(logq))
        const ratios = Float64Array.from({ length: total }, (_, i) => {
          const v = item(
            primalOf(um.fn(fromStorage(Float64Array.from(rows.slice(i * D, (i + 1) * D)), [D], 'float64'))),
          )
          return (Number.isFinite(v) ? v : Number.NEGATIVE_INFINITY) - (lq[i] ?? 0)
        })
        // banned-ok: kernel: the log-weights, filled by the loop below
        let logWeights = new Float64Array(total)
        let paretoK = Number.NaN
        if (mode !== 'none' && total >= 25 && ratios.every((r) => Number.isFinite(r))) {
          const ps = paretoSmoothedWeights(ratios)
          // Constant ratios (the approximation is the posterior) have no tail to fit: the weights are uniform, k reads 0.
          const spread = Math.max(...ratios) - Math.min(...ratios)
          paretoK = Number.isNaN(ps.paretoK) && spread < 1e-8 ? 0 : ps.paretoK
          logWeights = Float64Array.from(toFlat(primalOf(ps.logWeights)))
        }
        let finalDraws = Float64Array.from(rows)
        if (mode === 'resample' && !Number.isNaN(paretoK)) {
          const idx = choice(foldIn(ctx.key, 1 << 20), total, total, { weights: Array.from(normaliseExp(logWeights)) })
          finalDraws = Float64Array.from(toFlat(primalOf(take(reshape(draws, [total, D]), idx, 0))))
          // banned-ok: kernel: the log-weights, filled by the loop below
          logWeights = new Float64Array(total)
        }
        return { ...s, stage: 'done', draws: finalDraws, logWeights, paretoK, ess: importanceEss(logWeights) }
      }
      return s
    },
    done: (s) => (s.stage === 'done' ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        ...(Number.isFinite(s.elbo) ? { elbo: s.elbo } : {}),
        ...(Number.isFinite(s.paretoK) ? { paretoK: s.paretoK } : {}),
        ...(Number.isFinite(s.ess) ? { ess: s.ess } : {}),
      },
    }),
  }
}

const normaliseExp = (lw: Float64Array): Float64Array => {
  let m = Number.NEGATIVE_INFINITY
  for (const v of lw) if (v > m) m = v
  const w = Float64Array.from(lw, (v) => Math.exp(v - m))
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((v) => v / total)
}

/**
 * The posterior a state of {@link pathfinderAlgorithm} stands for. At the end, the weighted (or resampled) importance draws,
 * constrained to the model's sites, with the k-hat and the effective sample size in the diagnostics. Before that, the normal
 * approximation of the best ELBO so far, as draws from it (an anytime snapshot of the path).
 *
 * @param model - The model.
 * @param data - Its data.
 * @param state - The state.
 * @param options - The weighting and the draws of a partial posterior.
 * @returns The posterior.
 * @throws OmaiError with code `value` when no iterate has a normal approximation yet.
 * @concept variational-inference
 * @example
 * ```ts
 * import { observe, pathfinderAlgorithm, pathfinderPosterior, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const r = runAlgorithm(pathfinderAlgorithm(model, undefined, { maxIterations: 20 }), { seed: 0, maxSteps: 40 })
 * Math.abs(item(pathfinderPosterior(model, undefined, r.state).mean('m')) - 1) < 0.1 // => true
 * ```
 */
export function pathfinderPosterior<D = void>(
  model: Model<D>,
  data: D,
  state: PathfinderState,
  options: { readonly draws?: number } = {},
): Posterior {
  const um = isolated(() => potentialEnergy(model, data))
  const D = um.dim
  const diagnostics: Record<string, Json> = {
    elbo: Number.isFinite(state.elbo) ? state.elbo : null,
    paths: state.paths.length,
    iterations: state.paths.map((p) => p.iteration),
    stage: state.stage,
    ...(Number.isFinite(state.paretoK) ? { paretoK: state.paretoK } : {}),
    ...(Number.isFinite(state.ess) ? { ess: state.ess } : {}),
  }
  if (state.stage === 'done' && state.draws !== null && state.logWeights !== null) {
    const draws = constrainDraws(um, state.draws)
    const weighted = state.logWeights.some((w) => w !== 0)
    return definePosterior({
      form: 'variational',
      engine: 'pathfinder',
      sites: Object.keys(draws).filter((k) => um.layout.some((l) => l.name === k)),
      status: 'completed',
      diagnostics,
      draws,
      ...(weighted ? { logWeights: state.logWeights } : {}),
    })
  }
  const best = state.paths.reduce<PathfinderGaussian | null>(
    (b, p) => (p.best !== null && (b === null || p.best.elbo > b.elbo) ? p.best : b),
    null,
  )
  if (best === null) throw new OmaiError('value', 'pathfinderPosterior: no normal approximation yet')
  const L = choleskyOf(best.covariance, D)
  const n = options.draws ?? 400
  const draws = primalOf(drawsFrom(best.mean, L, n, makeKey(0x7061)))
  return definePosterior({
    form: 'variational',
    engine: 'pathfinder',
    sites: um.layout.map((l) => l.name),
    status: 'stopped',
    diagnostics,
    draws: constrainDraws(um, Float64Array.from(toFlat(draws))),
  })
}
