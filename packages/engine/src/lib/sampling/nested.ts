/**
 * Static nested sampling (Skilling 2006): the evidence and the posterior from a population of live points that is
 * squeezed towards higher likelihood. Each step removes the live point of lowest likelihood, records it with the share of
 * prior volume it stands for, and replaces it by a draw from the prior restricted to the higher-likelihood region, taken
 * by short slice-sampling or random-walk chains started from another live point. The evidence is the sum of the removed
 * points' likelihood times their volume share (with the live points added at the end), with an error bar from the
 * information gained; the removed points with those weights are also the posterior.
 */

import { logSumExpOf, OmaiError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import type { MCMCKernel } from './chains.ts'
import { sliceSampler } from './gibbs.ts'
import { randomWalkMetropolis } from './metropolis.ts'
import { type ChainProposal, numericTarget } from './target.ts'

/** How the constrained prior is sampled: short slice-sampling sweeps or random-walk Metropolis moves from a live point. */
export type NestedInner = 'slice' | 'random-walk'

/** Options of {@link nestedAlgorithm}. */
export interface NestedOptions {
  /** The number of live points (default 200): the resolution in prior volume, and the error of log Z falls as 1 / sqrt(n). */
  readonly numLive?: number
  /** The constrained-prior sampler (default `slice`). */
  readonly inner?: NestedInner
  /** Sweeps (`slice`) or moves (`random-walk`) of the inner chain per replacement (default 5). */
  readonly innerSteps?: number
  /** Stop when the live points could add less than this to log Z (default 0.05). */
  readonly tolerance?: number
}

/** The rows kept per removed point: its coordinates, log likelihood and log volume share. */
const BLOCK_ROWS = 64

/** The state of {@link nestedAlgorithm}. */
export interface NestedState {
  /** Replacements done: the number of removed points. */
  readonly iteration: number
  /** The live points, row-major `[numLive, dim]`, and their log likelihoods. */
  readonly live: Float64Array
  readonly liveLogL: Float64Array
  /**
   * The removed points in blocks of rows (`dim` coordinates, the log likelihood, the log of the prior volume share) and the
   * block being filled, so that a step copies at most one block.
   */
  readonly blocks: readonly Float64Array[]
  readonly tail: Float64Array
  /** The log evidence and the information of the removed points alone (the running sums of the recursion). */
  readonly logZDead: number
  readonly infoDead: number
  /** The log likelihood of the point removed last, which every live point now exceeds. */
  readonly logLStar: number
  /** Log likelihood evaluations by the inner sampler in the last replacement, and in total. */
  readonly evaluations: number
  readonly totalEvaluations: number
  /** Whether the stopping rule has been met (the live points' share is added at the end). */
  readonly finished: boolean
}

/** The paths {@link nestedAlgorithm} logs, as a type. */
export type NestedPath =
  | 'ns/logZ'
  | 'ns/logZError'
  | 'ns/logVolume'
  | 'ns/logL'
  | 'ns/information'
  | 'ns/evaluations'
  | 'ns/live'

/**
 * What {@link nestedAlgorithm} logs, by path, one replacement per step: the log evidence so far (removed points plus the
 * live ones) and its error, the log prior volume reached, the likelihood threshold, the information gained, the likelihood
 * evaluations of the replacement and the live points.
 */
export const nestedContract: Contract<NestedState, NestedPath> = defineContract({
  'ns/logZ': scalar({ description: 'the log evidence so far, the live points included', direction: 'higher' }),
  'ns/logZError': scalar({ description: 'the error of the log evidence, sqrt(H / n)', direction: 'lower' }),
  'ns/logVolume': scalar({ description: 'the log of the prior volume left, -iteration / n', direction: 'lower' }),
  'ns/logL': scalar({
    description: 'the likelihood threshold: the log likelihood of the point just removed',
    direction: 'higher',
  }),
  'ns/information': scalar({ description: 'the information H of the posterior relative to the prior, in nats' }),
  'ns/evaluations': scalar({ description: 'log likelihood evaluations used to replace the point' }),
  'ns/live': tensor({ role: 'state', shape: ['live', 'dim'], description: 'the live points' }),
})

const logAddExp = (a: number, b: number): number => {
  if (a === Number.NEGATIVE_INFINITY) return b
  if (b === Number.NEGATIVE_INFINITY) return a
  const m = Math.max(a, b)
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/** Skilling's recursion for the information when a point of log weight `logWt` and log likelihood `logL` is added. */
function addPoint(logZ: number, info: number, logWt: number, logL: number): { logZ: number; info: number } {
  const next = logAddExp(logZ, logWt + logL)
  const carried = logZ === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logZ - next) * (info + logZ)
  const here = logL === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logWt + logL - next) * logL
  return { logZ: next, info: here + carried - next }
}

/** The log prior volume left after `iteration` removals: each removal shrinks it by a factor `exp(-1 / n)` on average. */
const logVolumeAt = (iteration: number, n: number): number => -iteration / n

function* deadRows(s: NestedState, dim: number): Generator<Float64Array> {
  const width = dim + 2
  for (const block of [...s.blocks, s.tail])
    for (let r = 0; r < block.length / width; r++) yield block.subarray(r * width, (r + 1) * width)
}

/**
 * The evidence of a state, the live points' share included: the live points each stand for an equal part of the volume left
 * (`X_N / n`), so a run read before it ends is still an estimate with the error of the volume it has not resolved.
 *
 * @param s - A state of {@link nestedAlgorithm}.
 * @param n - The number of live points.
 * @returns The log evidence, its error `sqrt(H / n)` and the information `H`.
 * @concept nested-sampling
 * @example
 * ```ts
 * import { nestedEvidence, nestedSampler, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const r = nestedSampler(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { key: key(0), numLive: 100 })
 * Math.abs(nestedEvidence(r.state, 100).logZ - r.logZ) < 1e-12 // => true
 * ```
 */
export function nestedEvidence(
  s: NestedState,
  n: number,
): { readonly logZ: number; readonly logZError: number; readonly information: number } {
  const logWt = logVolumeAt(s.iteration, n) - Math.log(n)
  let logZ = s.logZDead
  let info = s.infoDead
  for (const l of s.liveLogL) ({ logZ, info } = addPoint(logZ, info, logWt, l))
  return { logZ, logZError: info > 0 ? Math.sqrt(info / n) : 0, information: info }
}

/**
 * The points a nested-sampling state holds, with log weights proportional to their share of the posterior: the removed points
 * (volume share times likelihood) and the live ones (an equal share of the volume left times likelihood). The weights are
 * normalised to sum to one.
 *
 * @param s - A state of {@link nestedAlgorithm}.
 * @param dim - The dimension.
 * @param n - The number of live points.
 * @returns The points row-major `[m, dim]` and their normalised log weights.
 * @concept nested-sampling
 * @example
 * ```ts
 * import { nestedPosterior, nestedSampler, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const r = nestedSampler(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { key: key(0), numLive: 50 })
 * nestedPosterior(r.state, 1, 50).logWeights.length === r.state.iteration + 50 // => true
 * ```
 */
export function nestedPosterior(
  s: NestedState,
  dim: number,
  n: number,
): { readonly points: Float64Array; readonly logWeights: Float64Array } {
  const m = s.iteration + n
  const points = new Float64Array(m * dim)
  const lw = new Float64Array(m)
  let at = 0
  for (const row of deadRows(s, dim)) {
    points.set(row.subarray(0, dim), at * dim)
    lw[at] = (row[dim + 1] ?? 0) + (row[dim] ?? 0)
    at += 1
  }
  const logWt = logVolumeAt(s.iteration, n) - Math.log(n)
  for (let i = 0; i < n; i++) {
    points.set(s.live.subarray(i * dim, (i + 1) * dim), at * dim)
    lw[at] = logWt + (s.liveLogL[i] ?? 0)
    at += 1
  }
  const z = logSumExpOf(lw)
  return { points, logWeights: lw.map((v) => v - z) }
}

/**
 * Draws of the log evidence that carry the uncertainty of the prior volumes: the shrinkage of the live set at each removal is
 * `t ~ Beta(n, 1)`, so `-log t` is exponential with rate `n`, and the volumes are cumulative products. The spread of the draws
 * is the honest error of log Z; for a smooth likelihood it is close to the `sqrt(H / n)` of the state.
 *
 * @param s - A state of {@link nestedAlgorithm}.
 * @param dim - The dimension.
 * @param n - The number of live points.
 * @param key - The key of the draws.
 * @param draws - How many (default 100).
 * @returns The sampled log evidences.
 * @concept nested-sampling
 * @example
 * ```ts
 * import { nestedLogZDraws, nestedSampler, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const r = nestedSampler(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { key: key(0), numLive: 50 })
 * nestedLogZDraws(r.state, 1, 50, key(1), 20).length // => 20
 * ```
 */
export function nestedLogZDraws(s: NestedState, dim: number, n: number, key: Key, draws = 100): Float64Array {
  const logL: number[] = []
  for (const row of deadRows(s, dim)) logL.push(row[dim] ?? 0)
  const k = logL.length
  const u = uniform(key, [draws * (k + 1)]).data
  const out = new Float64Array(draws)
  for (let d = 0; d < draws; d++) {
    let logX = 0
    let logZ = Number.NEGATIVE_INFINITY
    for (let i = 0; i < k; i++) {
      // X_i = X_{i-1} t_i with -log t_i ~ Exp(n); the share of the point is X_{i-1} - X_i.
      const shrink = -Math.log(1 - (u[d * (k + 1) + i] ?? 0)) / n
      const next = logX - shrink
      logZ = logAddExp(logZ, logX + Math.log(-Math.expm1(next - logX)) + (logL[i] ?? 0))
      logX = next
    }
    // The live points share the volume left equally.
    const live = logSumExpOf(s.liveLogL) - Math.log(n)
    out[d] = logAddExp(logZ, logX + live)
  }
  return out
}

/**
 * Static nested sampling as a steppable algorithm: `init` draws the live points from the prior; a step removes the live point
 * of lowest likelihood, adds its share to the evidence (`X_{i-1} - X_i` with `X_i = exp(-i / n)`, the expected volume), and
 * replaces it by a draw from the prior restricted to the likelihood above that point's, taken by `innerSteps` slice sweeps or
 * random-walk moves from another live point (the width or covariance from the live points). It ends `converged` when the
 * live points could add less than `tolerance` to log Z. The state serialises, and every step's randomness comes from the
 * step's key, so a resumed run is the uninterrupted one.
 *
 * @param prior - The prior (its draws and log density).
 * @param logLikelihood - log L(x) (NaN reads as -Infinity).
 * @param options - Live points, inner sampler and steps, tolerance.
 * @returns The algorithm; its state is a {@link NestedState}.
 * @throws OmaiError with code `value` for fewer than 2 live points or a non-positive tolerance or step count.
 * @concept nested-sampling
 * @reference J. Skilling, Nested sampling for general Bayesian computation (Bayesian Analysis 2006).
 * @index algorithm Nested sampling, one live-point replacement per step, with evidence and error.
 * @example
 * ```ts
 * import { nestedAlgorithm, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = nestedAlgorithm(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { numLive: 50 })
 * runAlgorithm(alg, { seed: 0, maxSteps: 1000 }).status // => 'converged'
 * ```
 */
export function nestedAlgorithm(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: NestedOptions = {},
): Algorithm<NestedState, NestedPath> {
  const n = options.numLive ?? 200
  const inner: NestedInner = options.inner ?? 'slice'
  const steps = options.innerSteps ?? 5
  const tolerance = options.tolerance ?? 0.05
  if (!(Number.isInteger(n) && n >= 2)) throw new OmaiError('value', 'nestedAlgorithm: need at least 2 live points')
  if (!(Number.isInteger(steps) && steps >= 1))
    throw new OmaiError('value', 'nestedAlgorithm: innerSteps must be at least 1')
  if (!(tolerance > 0)) throw new OmaiError('value', 'nestedAlgorithm: the tolerance must be positive')
  const dim = prior.dim
  const width = dim + 2
  const ll = (x: Float64Array): number => {
    const v = logLikelihood(x)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const spread = (live: Float64Array): { mean: Float64Array; sd: Float64Array } => {
    const mean = new Float64Array(dim)
    for (let i = 0; i < n; i++) for (let a = 0; a < dim; a++) mean[a] = (mean[a] ?? 0) + (live[i * dim + a] ?? 0) / n
    const sd = new Float64Array(dim)
    for (let i = 0; i < n; i++)
      for (let a = 0; a < dim; a++) sd[a] = (sd[a] ?? 0) + ((live[i * dim + a] ?? 0) - (mean[a] ?? 0)) ** 2 / (n - 1)
    return { mean, sd: sd.map((v) => (v > 0 ? Math.sqrt(v) : 1)) }
  }
  return {
    name: 'nested-sampling',
    config: { numLive: n, inner, innerSteps: steps, tolerance, dim },
    init: (key) => {
      const live = prior.sampleMany(foldIn(key, 0), n)
      const liveLogL = new Float64Array(n)
      for (let i = 0; i < n; i++) liveLogL[i] = ll(live.subarray(i * dim, (i + 1) * dim))
      return {
        iteration: 0,
        live,
        liveLogL,
        blocks: [],
        tail: new Float64Array(0),
        logZDead: Number.NEGATIVE_INFINITY,
        infoDead: 0,
        logLStar: Number.NEGATIVE_INFINITY,
        evaluations: n,
        totalEvaluations: n,
        finished: false,
      }
    },
    step: (s, ctx) => {
      let worst = 0
      for (let i = 1; i < n; i++) if ((s.liveLogL[i] ?? 0) < (s.liveLogL[worst] ?? 0)) worst = i
      const logLStar = s.liveLogL[worst] ?? Number.NEGATIVE_INFINITY
      const iteration = s.iteration + 1
      const logWt = logVolumeAt(iteration - 1, n) + Math.log(-Math.expm1(-1 / n))
      const dead = addPoint(s.logZDead, s.infoDead, logWt, logLStar)
      const row = new Float64Array(width)
      row.set(s.live.subarray(worst * dim, (worst + 1) * dim), 0)
      row[dim] = logLStar
      row[dim + 1] = logWt
      let tail = new Float64Array(s.tail.length + width)
      tail.set(s.tail)
      tail.set(row, s.tail.length)
      let blocks = s.blocks
      if (tail.length >= BLOCK_ROWS * width) {
        blocks = [...blocks, tail]
        tail = new Float64Array(0)
      }
      // The replacement: the prior restricted to log L >= logLStar, sampled from another live point.
      const constrained = numericTarget((x) => {
        const lp = prior.logProb(x)
        if (lp === Number.NEGATIVE_INFINITY) return lp
        return ll(x) >= logLStar ? lp : Number.NEGATIVE_INFINITY
      }, dim)
      const { sd } = spread(s.live)
      // Another live point: a uniform index among the n - 1 that are not the removed one.
      const pick = Math.min(n - 2, Math.floor((uniform(foldIn(ctx.key, 0), [1]).data[0] ?? 0) * (n - 1)))
      const from = pick >= worst ? pick + 1 : pick
      let evaluations = 0
      let kernel: MCMCKernel
      if (inner === 'slice') {
        kernel = sliceSampler(constrained, { width: sd })
      } else {
        const cov = Array.from({ length: dim }, (_, a) =>
          Array.from({ length: dim }, (_, b) => {
            let c = 0
            let ma = 0
            let mb = 0
            for (let i = 0; i < n; i++) {
              ma += (s.live[i * dim + a] ?? 0) / n
              mb += (s.live[i * dim + b] ?? 0) / n
            }
            for (let i = 0; i < n; i++)
              c += (((s.live[i * dim + a] ?? 0) - ma) * ((s.live[i * dim + b] ?? 0) - mb)) / (n - 1)
            return ((2.38 * 2.38) / dim) * c + (a === b ? 1e-12 : 0)
          }),
        )
        kernel = randomWalkMetropolis(constrained, { covariance: cov })
      }
      let state = kernel.init(s.live.subarray(from * dim, (from + 1) * dim))
      for (let k = 0; k < steps; k++) {
        const r = kernel.step(foldIn(ctx.key, 1 + k), state)
        state = r.state
        evaluations += r.info.numSteps ?? 1
      }
      const live = Float64Array.from(s.live)
      live.set(state.position, worst * dim)
      const liveLogL = Float64Array.from(s.liveLogL)
      liveLogL[worst] = ll(state.position)
      evaluations += 2
      let top = Number.NEGATIVE_INFINITY
      for (const v of liveLogL) if (v > top) top = v
      const logX = logVolumeAt(iteration, n)
      const remain = logAddExp(dead.logZ, top + logX) - dead.logZ
      const next: NestedState = {
        iteration,
        live,
        liveLogL,
        blocks,
        tail,
        logZDead: dead.logZ,
        infoDead: dead.info,
        logLStar,
        evaluations,
        totalEvaluations: s.totalEvaluations + evaluations,
        finished: remain < tolerance,
      }
      if (ctx.wants('ns/logZ') || ctx.wants('ns/logZError') || ctx.wants('ns/information')) {
        const e = nestedEvidence(next, n)
        ctx.log('ns/logZ', e.logZ)
        ctx.log('ns/logZError', e.logZError)
        ctx.log('ns/information', e.information)
      }
      ctx.log('ns/logVolume', logX)
      ctx.log('ns/logL', logLStar)
      ctx.log('ns/evaluations', evaluations)
      if (ctx.wants('ns/live')) ctx.log('ns/live', live)
      return next
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NestedState,
  }
}

/** The result of {@link nestedSampler}. */
export interface NestedResult {
  /** The log evidence (the live points included) and its error `sqrt(H / n)`. */
  readonly logZ: number
  readonly logZError: number
  /** The information gained from the prior, in nats. */
  readonly information: number
  /** Removals done and likelihood evaluations in all. */
  readonly iterations: number
  readonly evaluations: number
  /** The posterior sample: points row-major `[m, dim]` with normalised log weights. */
  readonly points: Float64Array
  readonly logWeights: Float64Array
  /** Whether the stopping rule was met within the iteration cap. */
  readonly converged: boolean
  /** The final state. */
  readonly state: NestedState
}

/**
 * Run nested sampling ({@link nestedAlgorithm}) until the live points could add less than the tolerance to log Z, and return
 * the evidence with its error, the information, and the weighted posterior sample.
 *
 * @param prior - The prior (a proper density: nested sampling's volumes are prior mass).
 * @param logLikelihood - log L(x).
 * @param options - Nested options, a cap on removals (default 1,000,000), key or seed, tracer and policy.
 * @returns The evidence, error, posterior sample and cost.
 * @concept nested-sampling
 * @index function Nested sampling: log evidence with error bar and weighted posterior sample.
 * @example
 * ```ts
 * import { nestedSampler, proposalFromDistribution } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * // Prior N(0, 3^2), likelihood exp(-(1 - x)^2 / 2): Z = sqrt(2 pi) N(1; 0, 10) on the natural scale.
 * const r = nestedSampler(proposalFromDistribution(normal(0, 3)), (x) => -0.5 * (1 - (x[0] ?? 0)) ** 2, { key: key(0), numLive: 200 })
 * Math.abs(r.logZ - (0.5 * Math.log(2 * Math.PI) - 0.5 * Math.log(2 * Math.PI * 10) - 0.05)) < 4 * r.logZError // => true
 * ```
 */
export function nestedSampler(
  prior: ChainProposal,
  logLikelihood: (x: Float64Array) => number,
  options: NestedOptions & { readonly maxIterations?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NestedResult {
  const { maxIterations, numLive, inner, innerSteps, tolerance, ...run } = options
  const n = numLive ?? 200
  const algorithm = nestedAlgorithm(prior, logLikelihood, {
    ...(numLive === undefined ? {} : { numLive }),
    ...(inner === undefined ? {} : { inner }),
    ...(innerSteps === undefined ? {} : { innerSteps }),
    ...(tolerance === undefined ? {} : { tolerance }),
  })
  const r = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? 1_000_000 })
  const e = nestedEvidence(r.state, n)
  const p = nestedPosterior(r.state, prior.dim, n)
  return {
    logZ: e.logZ,
    logZError: e.logZError,
    information: e.information,
    iterations: r.state.iteration,
    evaluations: r.state.totalEvaluations,
    points: p.points,
    logWeights: p.logWeights,
    converged: r.status === 'converged',
    state: r.state,
  }
}
