/**
 * Rejection samplers that produce independent exact draws from an unnormalised density: classic rejection sampling
 * under an envelope M q(x), adaptive rejection sampling for log-concave densities (Gilks and Wild), whose
 * piecewise-exponential envelope tightens with each rejection, and the ratio-of-uniforms method. Each is a steppable
 * algorithm whose step is one proposal (`rejectionAlgorithm`, `adaptiveRejectionAlgorithm`,
 * `ratioOfUniformsAlgorithm`; the state, with the draws so far and the number of uniforms used, serialises, and a run
 * resumes the same stream), and a function that runs it until it has the draws or the proposal budget is spent.
 */

import { logAddExp, OmaiError } from '../base/index.ts'
import { foldIn, type Key, uniformStream } from '../random/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunStatus,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import type { ChainProposal, LogDensity } from './target.ts'

/** The result of the rejection samplers. */
export interface RejectionResult {
  /** The accepted draws `[n, dim]` (fewer than asked when the proposal budget ran out). */
  readonly samples: Tensor<'float64'>
  /** Proposals made. */
  readonly proposals: number
  /** Accepted / proposed. */
  readonly acceptanceRate: number
  /** Whether all n draws were obtained within the budget. */
  readonly complete: boolean
}

/** Options of {@link rejectionSample}. */
export interface RejectionOptions {
  readonly key: Key
  readonly n: number
  /** log M with pi(x) <= M q(x) everywhere (pi may be unnormalised). */
  readonly logEnvelope: number
  /** The most proposals (default 1000 n). */
  readonly maxProposals?: number
}

/** The paths the rejection samplers log, as a type. */
export type RejectionPath =
  | 'reject/proposal'
  | 'reject/accepted'
  | 'reject/count'
  | 'reject/rate'
  | 'reject/log-ratio'
  | 'reject/threshold'
  | 'reject/violations'
  | 'reject/envelope'
  | 'reject/squeezed'
  | 'reject/evaluations'
  | 'reject/point'

/**
 * What the rejection samplers log, by path, one proposal per step: the proposal, whether it was accepted, the draws so
 * far and the acceptance rate so far, the log of the target over the envelope at the proposal (accepted when the
 * log of the uniform is below it), that log uniform, the envelope violations seen (rejection sampling), the abscissae of
 * the envelope, whether the squeeze accepted without evaluating the target and the target evaluations so far
 * (adaptive rejection), and the point (u, v) in the rectangle (ratio of uniforms).
 */
export const rejectionContract: Contract<RejectionBaseState, RejectionPath> = defineContract({
  'reject/proposal': tensor({ role: 'output', shape: ['dim'], description: 'the proposed point' }),
  'reject/accepted': scalar({ description: '1 when the proposal was accepted, 0 when rejected' }),
  'reject/count': scalar({ description: 'draws accepted so far', direction: 'higher' }),
  'reject/rate': scalar({ description: 'accepted / proposed so far' }),
  'reject/log-ratio': scalar({ description: 'log of the target over the envelope at the proposal' }),
  'reject/threshold': scalar({ description: 'the log of the uniform the ratio is compared with' }),
  'reject/violations': scalar({ description: 'proposals where the target exceeded the envelope (rejection sampling)' }),
  'reject/envelope': tensor({
    role: 'state',
    shape: ['abscissae'],
    description: 'the abscissae of the hull (adaptive)',
  }),
  'reject/squeezed': scalar({ description: '1 when the squeeze accepted without evaluating the target (adaptive)' }),
  'reject/evaluations': scalar({ description: 'target evaluations so far (adaptive)' }),
  'reject/point': tensor({
    role: 'output',
    shape: [2],
    description: 'the point (u, v) in the rectangle (ratio of uniforms)',
  }),
})

/**
 * What every rejection sampler's state holds: the proposals made, the draws accepted (flat, `dim` numbers each), the
 * number of uniforms used (the stream is a pure function of the key, so this is all a resume needs) and the last
 * proposal with whether it was accepted and the log of target over envelope there.
 */
export interface RejectionBaseState {
  readonly proposals: number
  readonly accepted: number
  readonly draws: number
  readonly samples: readonly number[]
  readonly lastProposal: readonly number[]
  readonly lastAccepted: boolean
  readonly lastLogRatio: number
}

/** A log value that is safe in JSON: infinities clamp, NaN reads as 0. */
const finiteLog = (v: number): number => (Number.isNaN(v) ? 0 : Math.max(-1e300, Math.min(1e300, v)))

/** A stream of uniforms that resumes after `used` draws, and counts what it hands out. */
function counted(k: Key, used: number): { readonly next: () => number; readonly used: () => number } {
  const stream = uniformStream(k, 64, used)
  let n = used
  return {
    next: () => {
      n += 1
      return stream()
    },
    used: () => n,
  }
}

const sampleTensor = (flat: readonly number[], accepted: number, dim: number): Tensor<'float64'> =>
  fromStorage(Float64Array.from(flat).slice(0, accepted * dim), [accepted, dim], 'float64')

function logBase(ctx: StepContext<RejectionPath>, s: RejectionBaseState, threshold: number | undefined): void {
  if (ctx.wants('reject/proposal')) ctx.log('reject/proposal', Float64Array.from(s.lastProposal))
  if (ctx.wants('reject/accepted')) ctx.log('reject/accepted', s.lastAccepted ? 1 : 0)
  if (ctx.wants('reject/count')) ctx.log('reject/count', s.accepted)
  if (ctx.wants('reject/rate')) ctx.log('reject/rate', s.proposals === 0 ? 0 : s.accepted / s.proposals)
  if (ctx.wants('reject/log-ratio')) ctx.log('reject/log-ratio', s.lastLogRatio)
  if (threshold !== undefined && ctx.wants('reject/threshold')) ctx.log('reject/threshold', finiteLog(threshold))
}

function statusOf(accepted: number, n: number, proposals: number, budget: number): RunStatus | undefined {
  return accepted >= n ? 'converged' : proposals >= budget ? 'stopped' : undefined
}

/** The state of {@link rejectionAlgorithm}: the base, the batch of proposals being worked through and the violations. */
export interface RejectionState extends RejectionBaseState {
  /** The index of the next batch of proposals to draw. */
  readonly batch: number
  /** The batch in hand: `dim` numbers per proposal. */
  readonly pending: readonly number[]
  /** log q of every proposal of the batch. */
  readonly pendingLogQ: readonly number[]
  /** The next proposal of the batch. */
  readonly cursor: number
  readonly violations: number
}

/**
 * Rejection sampling (von Neumann 1951) as a steppable algorithm: one step takes the next proposal x ~ q (drawn in
 * batches, as `rejectionSample` always did) and a uniform u, and keeps x when u M q(x) <= pi(x). Accepted draws are
 * exact samples of pi, and the acceptance rate is Z / M. The run ends with the draws asked for or the proposal budget.
 *
 * @param target - The target (unnormalised is fine).
 * @param q - The proposal.
 * @param options - Key, number of draws, log M and the proposal budget.
 * @returns The algorithm; its state is a {@link RejectionState}.
 * @throws OmaiError with code `shape` when dimensions differ, and `value` for a bad number of draws.
 * @concept rejection-sampling
 * @example
 * ```ts
 * import { numericTarget, proposalFromDistribution, rejectionAlgorithm } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const alg = rejectionAlgorithm(t, proposalFromDistribution(normal(0, 2)), { key: key(0), n: 5, logEnvelope: 1.6 })
 * runAlgorithm(alg, { maxSteps: 1000 }).state.accepted // => 5
 * ```
 */
export function rejectionAlgorithm(
  target: LogDensity,
  q: ChainProposal,
  options: RejectionOptions,
): Algorithm<RejectionState, RejectionPath> {
  if (q.dim !== target.dim) throw new OmaiError('shape', 'rejectionSample: proposal and target dimensions differ')
  const { n } = options
  if (!(Number.isInteger(n) && n >= 0))
    throw new OmaiError('value', 'rejectionSample: n must be a non-negative integer')
  const dim = target.dim
  const budget = options.maxProposals ?? 1000 * Math.max(1, n)
  return {
    name: 'rejection',
    config: { n, logEnvelope: options.logEnvelope, maxProposals: budget, dim },
    init: () => ({
      proposals: 0,
      accepted: 0,
      draws: 0,
      samples: [],
      lastProposal: [],
      lastAccepted: false,
      lastLogRatio: 0,
      batch: 0,
      pending: [],
      pendingLogQ: [],
      cursor: 0,
      violations: 0,
    }),
    step: (s, ctx) => {
      let { batch, pending, pendingLogQ, cursor } = s
      if (cursor >= pendingLogQ.length) {
        const size = Math.min(budget - s.proposals, Math.max(16, 2 * (n - s.accepted)))
        const xs = q.sampleMany(foldIn(foldIn(options.key, 0), batch), size)
        pending = Array.from(xs)
        pendingLogQ = Array.from(q.logProbMany(xs))
        batch += 1
        cursor = 0
      }
      const rand = counted(foldIn(options.key, 1), s.draws)
      const x = pending.slice(cursor * dim, (cursor + 1) * dim)
      const logRatio = target.logDensity(x) - options.logEnvelope - (pendingLogQ[cursor] ?? 0)
      const u = Math.log(1 - rand.next())
      const accept = u <= logRatio
      const nextState: RejectionState = {
        proposals: s.proposals + 1,
        accepted: s.accepted + (accept ? 1 : 0),
        draws: rand.used(),
        samples: accept ? [...s.samples, ...x] : s.samples,
        lastProposal: x,
        lastAccepted: accept,
        lastLogRatio: finiteLog(logRatio),
        batch,
        pending,
        pendingLogQ,
        cursor: cursor + 1,
        violations: s.violations + (logRatio > 1e-12 ? 1 : 0),
      }
      logBase(ctx, nextState, u)
      if (ctx.wants('reject/violations')) ctx.log('reject/violations', nextState.violations)
      return nextState
    },
    done: (s) => statusOf(s.accepted, n, s.proposals, budget),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RejectionState,
  }
}

/**
 * Rejection sampling (von Neumann 1951): draw x ~ q and u ~ U(0, 1), and keep x when u M q(x) <= pi(x). Accepted
 * draws are exact samples of pi, and the acceptance rate is Z / M. Points where pi(x) > M q(x) break the guarantee;
 * they are counted in `envelopeViolations`. {@link rejectionAlgorithm} run until it has the draws or the budget.
 *
 * @param target - The target (unnormalised is fine).
 * @param q - The proposal.
 * @param options - Key, number of draws, log M and the proposal budget.
 * @returns The draws, proposals, acceptance rate, completeness and envelope violations.
 * @throws OmaiError with code `shape` when dimensions differ, and `value` for a bad number of draws.
 * @concept rejection-sampling
 * @reference J. von Neumann, Various techniques used in connection with random digits (1951).
 * @example
 * ```ts
 * import { numericTarget, proposalFromDistribution, rejectionSample } from 'omai-engine/sampling'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * // pi(x) = exp(-x^2 / 2) and q = N(0, 2^2): pi / q <= 2 sqrt(2 pi).
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const r = rejectionSample(t, proposalFromDistribution(normal(0, 2)), { key: key(0), n: 500, logEnvelope: Math.log(2 * Math.sqrt(2 * Math.PI)) })
 * Math.abs(r.acceptanceRate - 0.5) < 0.06 // => true
 * ```
 */
export function rejectionSample(
  target: LogDensity,
  q: ChainProposal,
  options: RejectionOptions,
): RejectionResult & { readonly envelopeViolations: number } {
  const algorithm = rejectionAlgorithm(target, q, options)
  const budget = options.maxProposals ?? 1000 * Math.max(1, options.n)
  const s = runAlgorithm(algorithm, { maxSteps: budget }).state
  return {
    samples: sampleTensor(s.samples, s.accepted, target.dim),
    proposals: s.proposals,
    acceptanceRate: s.proposals === 0 ? Number.NaN : s.accepted / s.proposals,
    complete: s.accepted === options.n,
    envelopeViolations: s.violations,
  }
}

/** Options of {@link adaptiveRejectionSample}. */
export interface AdaptiveRejectionOptions {
  readonly key: Key
  readonly n: number
  /** Starting abscissae (at least two; default -1 and 1). With an infinite bound the outermost must slope the right way. */
  readonly initialPoints?: readonly number[]
  /** The support [low, high] (default the real line). */
  readonly domain?: readonly [number, number]
  /** The most abscissae in the hull (default 50). */
  readonly maxPoints?: number
  /** The most proposals (default 100 n). */
  readonly maxProposals?: number
}

/** The state of {@link adaptiveRejectionAlgorithm}: the base and the hull, the abscissae with h = log f and its slope there. */
export interface AdaptiveRejectionState extends RejectionBaseState {
  readonly abscissae: readonly number[]
  readonly logDensities: readonly number[]
  readonly slopes: readonly number[]
  /** Target evaluations (the gradient at an inserted abscissa counts one). */
  readonly evaluations: number
  /** Draws the squeeze accepted without an evaluation. */
  readonly squeezed: number
}

/** Insert an abscissa into the hull (a sorted copy), checking that the slopes still decrease. */
function insertAbscissa(
  target: LogDensity,
  hull: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'>,
  x: number,
): Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'> {
  const r = target.valueAndGrad([x])
  const xs = [...hull.abscissae]
  const hs = [...hull.logDensities]
  const ds = [...hull.slopes]
  let i = 0
  while (i < xs.length && (xs[i] ?? 0) < x) i++
  if (xs[i] === x) return hull
  xs.splice(i, 0, x)
  hs.splice(i, 0, r.value)
  ds.splice(i, 0, r.grad[0] ?? 0)
  for (let j = 1; j < ds.length; j++) {
    if ((ds[j] ?? 0) > (ds[j - 1] ?? 0) + 1e-9 * (1 + Math.abs(ds[j - 1] ?? 0))) {
      throw new OmaiError('value', 'adaptiveRejectionSample: the density is not log-concave')
    }
  }
  return { abscissae: xs, logDensities: hs, slopes: ds }
}

/** The envelope: breakpoints z_0..z_k and the log masses of the k tangent pieces. */
function hullOf(
  h: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'>,
  lo: number,
  hi: number,
): { readonly z: number[]; readonly logMass: number[] } {
  const { abscissae: xs, logDensities: hs, slopes: ds } = h
  const k = xs.length
  const z: number[] = [lo]
  for (let j = 0; j < k - 1; j++) {
    const [x0, x1, h0, h1, d0, d1] = [
      xs[j] ?? 0,
      xs[j + 1] ?? 0,
      hs[j] ?? 0,
      hs[j + 1] ?? 0,
      ds[j] ?? 0,
      ds[j + 1] ?? 0,
    ]
    // Where the tangents at x0 and x1 meet (the midpoint when they are parallel).
    z.push(Math.abs(d0 - d1) < 1e-12 ? 0.5 * (x0 + x1) : (h1 - h0 - x1 * d1 + x0 * d0) / (d0 - d1))
  }
  z.push(hi)
  const logMass = xs.map((x, j) => logPieceMass(hs[j] ?? 0, ds[j] ?? 0, x, z[j] ?? lo, z[j + 1] ?? hi))
  return { z, logMass }
}

/** The envelope (the tangent that applies) at x. */
function upperAt(
  h: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'>,
  x: number,
  z: readonly number[],
): number {
  let j = 0
  while (j < h.abscissae.length - 1 && x > (z[j + 1] ?? 0)) j++
  return (h.logDensities[j] ?? 0) + (h.slopes[j] ?? 0) * (x - (h.abscissae[j] ?? 0))
}

/** The squeeze (the chord) at x, minus infinity outside the outermost abscissae. */
function lowerAt(h: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities'>, x: number): number {
  const xs = h.abscissae
  const hs = h.logDensities
  if (x < (xs[0] ?? 0) || x > (xs[xs.length - 1] ?? 0)) return Number.NEGATIVE_INFINITY
  let j = 0
  while (j < xs.length - 2 && x > (xs[j + 1] ?? 0)) j++
  const [x0, x1, h0, h1] = [xs[j] ?? 0, xs[j + 1] ?? 0, hs[j] ?? 0, hs[j + 1] ?? 0]
  return ((x1 - x) * h0 + (x - x0) * h1) / (x1 - x0)
}

/** log of the integral of exp(h + d (x - x0)) over [a, b]. */
function logPieceMass(h: number, d: number, x0: number, a: number, b: number): number {
  if (!(b > a)) return Number.NEGATIVE_INFINITY
  if (Math.abs(d) < 1e-12) return h + Math.log(b - a)
  // log (exp(d (b - x0)) - exp(d (a - x0))) / |d|, with the larger exponent factored out.
  const ea = d * (a - x0)
  const eb = d * (b - x0)
  const hiE = Math.max(ea, eb)
  const loE = Math.min(ea, eb)
  return h + hiE + Math.log1p(-Math.exp(loE - hiE)) - Math.log(Math.abs(d))
}

/** Invert the CDF of exp(d (x - x0)) on [a, b] at u. */
function samplePiece(d: number, x0: number, a: number, b: number, u: number): number {
  if (Math.abs(d) < 1e-12) return a + u * (b - a)
  const ea = d * (a - x0)
  const eb = d * (b - x0)
  // x = x0 + log(exp(ea) + u (exp(eb) - exp(ea))) / d, stably in log space.
  const m = Math.max(ea, eb)
  const v = Math.exp(ea - m) + u * (Math.exp(eb - m) - Math.exp(ea - m))
  const x = x0 + (m + Math.log(v)) / d
  return Math.min(Math.max(x, a), b)
}

/**
 * Adaptive rejection sampling (Gilks and Wild 1992) as a steppable algorithm: one step draws one point from the
 * piecewise-exponential envelope made of the tangents of h = log f at the abscissae, accepts it by the squeeze (the
 * chords) when it can without evaluating f, otherwise by evaluating h; every evaluation becomes a new abscissa, so the
 * envelope tightens and the acceptance rate approaches 1. The state holds the abscissae, h and its slope there.
 *
 * @param target - A one-dimensional log-concave target with a gradient.
 * @param options - Key, number of draws, starting points, domain, hull size and proposal budget.
 * @returns The algorithm; its state is an {@link AdaptiveRejectionState}.
 * @throws OmaiError with code `value` for a target that is not one-dimensional with a gradient, fewer than two
 *   starting points, points outside the domain, an envelope that cannot be normalised (wrong slopes at infinite
 *   bounds), or a density found not to be log-concave.
 * @concept adaptive-rejection-sampling
 * @example
 * ```ts
 * import { adaptiveRejectionAlgorithm, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const s = runAlgorithm(adaptiveRejectionAlgorithm(t, { key: key(0), n: 100 }), { maxSteps: 1000 }).state
 * s.abscissae.length > 2 // => true
 * ```
 */
export function adaptiveRejectionAlgorithm(
  target: LogDensity,
  options: AdaptiveRejectionOptions,
): Algorithm<AdaptiveRejectionState, RejectionPath> {
  if (target.dim !== 1 || !target.hasGradient) {
    throw new OmaiError('value', 'adaptiveRejectionSample: need a one-dimensional target with a gradient')
  }
  const { n } = options
  const [lo, hi] = options.domain ?? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
  const maxPoints = options.maxPoints ?? 50
  const budget = options.maxProposals ?? 100 * Math.max(1, n)
  const init = [...(options.initialPoints ?? [-1, 1])].sort((a, b) => a - b)
  if (init.length < 2 || init.some((x) => !(x > lo && x < hi))) {
    throw new OmaiError('value', 'adaptiveRejectionSample: need two starting points inside the domain')
  }
  return {
    name: 'adaptive-rejection',
    config: { n, maxPoints, maxProposals: budget, initialPoints: init },
    init: () => {
      let h: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'> = {
        abscissae: [],
        logDensities: [],
        slopes: [],
      }
      for (const x of init) h = insertAbscissa(target, h, x)
      if (!h.logDensities.every(Number.isFinite))
        throw new OmaiError('value', 'adaptiveRejectionSample: the log density must be finite at the starting points')
      const d0 = h.slopes[0] ?? 0
      const d1 = h.slopes[h.slopes.length - 1] ?? 0
      if ((lo === Number.NEGATIVE_INFINITY && !(d0 > 0)) || (hi === Number.POSITIVE_INFINITY && !(d1 < 0))) {
        throw new OmaiError(
          'value',
          'adaptiveRejectionSample: with an infinite bound the outer tangents must point down',
        )
      }
      return {
        proposals: 0,
        accepted: 0,
        draws: 0,
        samples: [],
        lastProposal: [],
        lastAccepted: false,
        lastLogRatio: 0,
        ...h,
        evaluations: init.length,
        squeezed: 0,
      }
    },
    step: (s, ctx) => {
      const rand = counted(options.key, s.draws)
      const env = hullOf(s, lo, hi)
      const total = env.logMass.reduce((a, b) => logAddExp(a, b), Number.NEGATIVE_INFINITY)
      if (!Number.isFinite(total))
        throw new OmaiError('value', 'adaptiveRejectionSample: the envelope cannot be normalised')
      // Choose a piece by its mass, then invert its exponential CDF.
      let u = rand.next()
      let j = 0
      for (; j < env.logMass.length - 1; j++) {
        const p = Math.exp((env.logMass[j] ?? Number.NEGATIVE_INFINITY) - total)
        if (u < p) break
        u -= p
      }
      const x = samplePiece(s.slopes[j] ?? 0, s.abscissae[j] ?? 0, env.z[j] ?? lo, env.z[j + 1] ?? hi, rand.next())
      const w = Math.log(1 - rand.next())
      const ux = upperAt(s, x, env.z)
      const lx = lowerAt(s, x)
      let accept = false
      let squeezed = false
      let hull: Pick<AdaptiveRejectionState, 'abscissae' | 'logDensities' | 'slopes'> = s
      let evaluations = s.evaluations
      let logRatio = lx - ux
      if (w <= lx - ux) {
        accept = true
        squeezed = true
      } else {
        const hx = target.logDensity([x])
        evaluations += 1
        logRatio = hx - ux
        if (w <= hx - ux) accept = true
        if (s.abscissae.length < maxPoints && Number.isFinite(hx)) {
          hull = insertAbscissa(target, s, x)
          evaluations += 1
        }
      }
      const nextState: AdaptiveRejectionState = {
        proposals: s.proposals + 1,
        accepted: s.accepted + (accept ? 1 : 0),
        draws: rand.used(),
        samples: accept ? [...s.samples, x] : s.samples,
        lastProposal: [x],
        lastAccepted: accept,
        lastLogRatio: finiteLog(logRatio),
        abscissae: hull.abscissae,
        logDensities: hull.logDensities,
        slopes: hull.slopes,
        evaluations,
        squeezed: s.squeezed + (squeezed ? 1 : 0),
      }
      logBase(ctx, nextState, w)
      if (ctx.wants('reject/envelope')) ctx.log('reject/envelope', Float64Array.from(nextState.abscissae))
      if (ctx.wants('reject/squeezed')) ctx.log('reject/squeezed', squeezed ? 1 : 0)
      if (ctx.wants('reject/evaluations')) ctx.log('reject/evaluations', evaluations)
      return nextState
    },
    done: (s) => statusOf(s.accepted, n, s.proposals, budget),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AdaptiveRejectionState,
  }
}

/**
 * Adaptive rejection sampling (Gilks and Wild 1992) for a log-concave density on an interval: the tangents of
 * h = log f at the abscissae form a piecewise-exponential envelope and the chords a squeeze. Draws from the envelope
 * are accepted by the squeeze test when possible (no evaluation), otherwise by evaluating h; every evaluation
 * becomes a new abscissa, so the envelope tightens and the acceptance rate approaches 1. {@link adaptiveRejectionAlgorithm}
 * run until it has the draws or the budget.
 *
 * @param target - A one-dimensional log-concave target with a gradient.
 * @param options - Key, number of draws, starting points, domain, hull size and proposal budget.
 * @returns The draws `[n, 1]`, proposals, acceptance rate, completeness and the final abscissae.
 * @throws OmaiError with code `value` for a target that is not one-dimensional with a gradient, fewer than two
 *   starting points, points outside the domain, an envelope that cannot be normalised (wrong slopes at infinite
 *   bounds), or a density found not to be log-concave.
 * @concept adaptive-rejection-sampling
 * @reference W. R. Gilks and P. Wild, Adaptive rejection sampling for Gibbs sampling (Applied Statistics 1992).
 * @example
 * ```ts
 * import { adaptiveRejectionSample, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const r = adaptiveRejectionSample(t, { key: key(0), n: 1000 })
 * r.acceptanceRate > 0.9 // => true
 * ```
 */
export function adaptiveRejectionSample(
  target: LogDensity,
  options: AdaptiveRejectionOptions,
): RejectionResult & { readonly abscissae: readonly number[] } {
  const budget = options.maxProposals ?? 100 * Math.max(1, options.n)
  const s = runAlgorithm(adaptiveRejectionAlgorithm(target, options), { maxSteps: budget }).state
  return {
    samples: sampleTensor(s.samples, s.accepted, 1),
    proposals: s.proposals,
    acceptanceRate: s.proposals === 0 ? Number.NaN : s.accepted / s.proposals,
    complete: s.accepted === options.n,
    abscissae: [...s.abscissae],
  }
}

/** Options of {@link ratioOfUniforms}. */
export interface RatioOfUniformsOptions {
  readonly key: Key
  readonly n: number
  /**
   * The bounding rectangle [0, uMax] x [vMin, vMax] of {(u, v): 0 < u <= sqrt(f(v / u))}: uMax = sup sqrt f, vMin and
   * vMax the inf and sup of x sqrt f(x). Default: found by a grid search over `range`, enlarged by 1%.
   */
  readonly bounds?: { readonly uMax: number; readonly vMin: number; readonly vMax: number }
  /** The interval searched for the bounds (default [-30, 30]). */
  readonly range?: readonly [number, number]
  /** The most proposals (default 1000 n). */
  readonly maxProposals?: number
}

/** The rectangle {@link ratioOfUniforms} proposes from: u in [0, uMax], v in [vMin, vMax]. */
export interface RatioOfUniformsBounds {
  readonly uMax: number
  readonly vMin: number
  readonly vMax: number
}

/**
 * The bounding rectangle of the ratio-of-uniforms region by a grid search: uMax = sup sqrt f, vMin and vMax the inf and
 * sup of x sqrt f(x), each enlarged by 1%. The options' `bounds` when given.
 *
 * @param density - f(x) >= 0, unnormalised.
 * @param options - The bounds to use as they are, or the interval to search (default [-30, 30]).
 * @returns The rectangle.
 * @throws OmaiError with code `value` for a density that is zero on the search range or bounds that are not finite.
 * @example
 * ```ts
 * import { ratioOfUniformsBounds } from 'omai-engine/sampling'
 * ratioOfUniformsBounds((x) => Math.exp(-0.5 * x * x), {}).uMax // => 1.01
 * ```
 */
export function ratioOfUniformsBounds(
  density: (x: number) => number,
  options: Pick<RatioOfUniformsOptions, 'bounds' | 'range'>,
): RatioOfUniformsBounds {
  let bounds = options.bounds
  if (bounds === undefined) {
    const f = safeDensity(density)
    const [a, b] = options.range ?? [-30, 30]
    let uMax = 0
    let vMin = 0
    let vMax = 0
    const grid = 20001
    for (let i = 0; i < grid; i++) {
      const x = a + ((b - a) * i) / (grid - 1)
      const s = Math.sqrt(f(x))
      uMax = Math.max(uMax, s)
      vMin = Math.min(vMin, x * s)
      vMax = Math.max(vMax, x * s)
    }
    if (!(uMax > 0)) throw new OmaiError('value', 'ratioOfUniforms: the density is zero on the search range')
    bounds = { uMax: 1.01 * uMax, vMin: 1.01 * vMin, vMax: 1.01 * vMax }
  }
  if (![bounds.uMax, bounds.vMin, bounds.vMax].every(Number.isFinite) || !(bounds.uMax > 0)) {
    throw new OmaiError('value', 'ratioOfUniforms: the bounds must be finite with uMax > 0')
  }
  return bounds
}

const safeDensity =
  (density: (x: number) => number) =>
  (x: number): number => {
    const v = density(x)
    return v > 0 && Number.isFinite(v) ? v : 0
  }

/**
 * The ratio-of-uniforms method (Kinderman and Monahan 1977) as a steppable algorithm: one step draws (u, v) uniform on
 * the bounding rectangle and keeps x = v / u when (u, v) falls in C = {(u, v): 0 < u <= sqrt(f(v / u))}. The state is
 * the base state of the rejection samplers; the rectangle is in the configuration.
 *
 * @param density - f(x) >= 0, unnormalised (log densities: pass exp of them).
 * @param options - Key, number of draws, bounds or search range, and the proposal budget.
 * @returns The algorithm; its state is a {@link RejectionBaseState}.
 * @throws OmaiError with code `value` for a density that is zero on the search range or bounds that are not finite.
 * @concept ratio-of-uniforms
 * @example
 * ```ts
 * import { ratioOfUniformsAlgorithm } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = ratioOfUniformsAlgorithm((x) => Math.exp(-0.5 * x * x), { key: key(0), n: 10 })
 * runAlgorithm(alg, { maxSteps: 1000 }).state.accepted // => 10
 * ```
 */
export function ratioOfUniformsAlgorithm(
  density: (x: number) => number,
  options: RatioOfUniformsOptions,
): Algorithm<RejectionBaseState, RejectionPath> {
  const { n } = options
  const f = safeDensity(density)
  const bounds = ratioOfUniformsBounds(density, options)
  const budget = options.maxProposals ?? 1000 * Math.max(1, n)
  return {
    name: 'ratio-of-uniforms',
    config: { n, maxProposals: budget, uMax: bounds.uMax, vMin: bounds.vMin, vMax: bounds.vMax },
    init: () => ({
      proposals: 0,
      accepted: 0,
      draws: 0,
      samples: [],
      lastProposal: [],
      lastAccepted: false,
      lastLogRatio: 0,
    }),
    step: (s, ctx) => {
      const rand = counted(options.key, s.draws)
      const u = bounds.uMax * (1 - rand.next())
      const v = bounds.vMin + (bounds.vMax - bounds.vMin) * rand.next()
      const x = v / u
      const fx = f(x)
      const accept = u * u <= fx
      const nextState: RejectionBaseState = {
        proposals: s.proposals + 1,
        accepted: s.accepted + (accept ? 1 : 0),
        draws: rand.used(),
        samples: accept ? [...s.samples, x] : s.samples,
        lastProposal: [x],
        lastAccepted: accept,
        lastLogRatio: finiteLog(Math.log(fx) - 2 * Math.log(u)),
      }
      logBase(ctx, nextState, undefined)
      if (ctx.wants('reject/point')) ctx.log('reject/point', Float64Array.of(u, v))
      return nextState
    },
    done: (s) => statusOf(s.accepted, n, s.proposals, budget),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RejectionBaseState,
  }
}

/**
 * The ratio-of-uniforms method (Kinderman and Monahan 1977): for an unnormalised density f, (u, v) uniform on
 * C = {(u, v): 0 < u <= sqrt(f(v / u))} gives x = v / u ~ f. C is sampled by rejection from its bounding rectangle,
 * found here by a grid search when not given (densities with tails heavier than 1/x^2 have no bounded C).
 * {@link ratioOfUniformsAlgorithm} run until it has the draws or the budget.
 *
 * @param density - f(x) >= 0, unnormalised (log densities: pass exp of them).
 * @param options - Key, number of draws, bounds or search range, and the proposal budget.
 * @returns The draws `[n, 1]`, proposals, acceptance rate, completeness and the rectangle used.
 * @throws OmaiError with code `value` for a density that is zero on the search range or bounds that are not
 *   finite.
 * @concept ratio-of-uniforms
 * @reference A. J. Kinderman and J. F. Monahan, Computer generation of random variables using the ratio of uniform
 *   deviates (ACM TOMS 1977).
 * @example
 * ```ts
 * import { ratioOfUniforms } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * // The standard normal: the region has area sqrt(2 pi) / 2 inside a 1 x 2 sqrt(2 / e) box.
 * const r = ratioOfUniforms((x) => Math.exp(-0.5 * x * x), { key: key(0), n: 2000 })
 * Math.abs(r.acceptanceRate - Math.sqrt(Math.PI * Math.E) / 4) < 0.03 // => true
 * ```
 */
export function ratioOfUniforms(
  density: (x: number) => number,
  options: RatioOfUniformsOptions,
): RejectionResult & { readonly bounds: RatioOfUniformsBounds } {
  const bounds = ratioOfUniformsBounds(density, options)
  const budget = options.maxProposals ?? 1000 * Math.max(1, options.n)
  const s = runAlgorithm(ratioOfUniformsAlgorithm(density, { ...options, bounds }), { maxSteps: budget }).state
  return {
    samples: sampleTensor(s.samples, s.accepted, 1),
    proposals: s.proposals,
    acceptanceRate: s.proposals === 0 ? Number.NaN : s.accepted / s.proposals,
    complete: s.accepted === options.n,
    bounds,
  }
}
