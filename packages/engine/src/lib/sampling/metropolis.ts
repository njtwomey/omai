import { type Json, WhiteboxError } from '../base/index.ts'
import { dgemv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { lowerCholesky, squareMatrix, vectorOf } from './_linear.ts'
import { acceptWith, initState, type Welford, welfordCovariance, welfordInit, welfordUpdate } from './_mcmc.ts'
import type { Adaptation, MCMCKernel, MCMCState, Transition, TransitionInfo, Warmup } from './chains.ts'
import type { LogDensity, Proposal } from './target.ts'

/**
 * The Metropolis-Hastings acceptance probability min(1, pi(x') q(x | x') / (pi(x) q(x' | x))) from log terms. A NaN
 * ratio (both points outside the support) reads as 0, so a chain never moves on an undefined ratio.
 *
 * @param logTarget - log pi(x) at the current point.
 * @param logTargetProposal - log pi(x') at the proposal.
 * @param logForward - log q(x' | x) (0 for symmetric proposals).
 * @param logBackward - log q(x | x') (0 for symmetric proposals).
 * @returns The acceptance probability in [0, 1].
 * @concept metropolis-hastings
 * @example
 * ```ts
 * import { metropolisAcceptance } from 'whitebox-engine/sampling'
 * metropolisAcceptance(0, Math.log(0.25), 0, 0) // => 0.25
 * metropolisAcceptance(-1, 0, 0, 0) // => 1
 * ```
 */
export function metropolisAcceptance(
  logTarget: number,
  logTargetProposal: number,
  logForward: number,
  logBackward: number,
): number {
  const r = logTargetProposal - logTarget + logBackward - logForward
  if (Number.isNaN(r)) return 0
  return r >= 0 ? 1 : Math.exp(r)
}

function transition(
  state: MCMCState,
  proposal: Float64Array,
  logTargetProposal: number,
  alpha: number,
  key: Key,
): Transition<MCMCState> {
  const accepted = acceptWith(key, alpha)
  const info: TransitionInfo = { acceptanceRate: alpha, accepted }
  return { state: accepted ? { position: proposal, logDensity: logTargetProposal } : state, info }
}

/** A proposal and the log densities of the forward and backward moves (equal for symmetric proposals). */
export interface MoveProposal {
  readonly proposal: ArrayLike<number>
  /** log q(x' | x). */
  readonly logForward: number
  /** log q(x | x'). */
  readonly logBackward: number
}

/**
 * General Metropolis-Hastings with any proposal mechanism: `propose(key, x)` returns x' with log q(x' | x) and
 * log q(x | x'), and the move is accepted with {@link metropolisAcceptance}. Discrete targets work too: positions are
 * just numbers (state indices, counts).
 *
 * @param target - The target.
 * @param propose - The proposal mechanism.
 * @param options - A name for traces.
 * @returns The kernel.
 * @concept metropolis-hastings
 * @reference W. K. Hastings, Monte Carlo sampling methods using Markov chains and their applications (Biometrika 1970).
 * @example
 * ```ts
 * import { metropolisHastings, numericTarget, sampleChain } from 'whitebox-engine/sampling'
 * import { key, uniform } from 'whitebox-engine/random'
 * // A walk on {0, 1, 2} with target weights 1 : 2 : 1.
 * const t = numericTarget((x) => Math.log([1, 2, 1][x[0] ?? 0] ?? 0), 1)
 * const k = metropolisHastings(t, (k, x) => ({
 *   proposal: [((x[0] ?? 0) + ((uniform(k, [1]).data[0] ?? 0) < 0.5 ? 1 : 2)) % 3],
 *   logForward: 0,
 *   logBackward: 0,
 * }))
 * sampleChain(k, [0], { key: key(0), numDraws: 10 }).draws.every((v) => v >= 0 && v <= 2) // => true
 * ```
 */
export function metropolisHastings(
  target: LogDensity,
  propose: (key: Key, x: Float64Array) => MoveProposal,
  options: { readonly name?: string } = {},
): MCMCKernel {
  return {
    name: options.name ?? 'metropolisHastings',
    dim: target.dim,
    config: { dim: target.dim },
    init: (position) => initState(target, position, 'metropolisHastings'),
    step: (key, state) => {
      const m = propose(foldIn(key, 0), state.position)
      const proposal = Float64Array.from(m.proposal)
      if (proposal.length !== target.dim) {
        throw new WhiteboxError('shape', `metropolisHastings: a proposal has ${proposal.length} values`)
      }
      const lp = target.logDensity(proposal)
      const alpha = metropolisAcceptance(state.logDensity, lp, m.logForward, m.logBackward)
      return transition(state, proposal, lp, alpha, foldIn(key, 1))
    },
  }
}

/** Options of {@link randomWalkMetropolis}. */
export interface RandomWalkOptions {
  /** The proposal standard deviation, one for all coordinates or one each (default 2.38 / sqrt(dim)). */
  readonly scale?: number | TensorLike
  /** A full proposal covariance `[dim, dim]` (overrides `scale`). */
  readonly covariance?: TensorLike
}

/**
 * Random-walk Metropolis: x' = x + L z with z standard normal and L L^T the proposal covariance (diagonal `scale^2` or
 * a full `covariance`); the proposal is symmetric, so the acceptance ratio is pi(x') / pi(x).
 *
 * @param target - The target.
 * @param options - Proposal scale or covariance.
 * @returns The kernel.
 * @throws WhiteboxError with code `value` for non-positive scales or a covariance that is not positive definite.
 * @concept random-walk-metropolis
 * @reference N. Metropolis et al., Equation of state calculations by fast computing machines (J. Chem. Phys. 1953).
 * @example
 * ```ts
 * import { randomWalkMetropolis, numericTarget } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = randomWalkMetropolis(t, { scale: 1 })
 * k.step(key(0), k.init([0])).info.acceptanceRate <= 1 // => true
 * ```
 */
export function randomWalkMetropolis(target: LogDensity, options: RandomWalkOptions = {}): MCMCKernel {
  const dim = target.dim
  const factor =
    options.covariance === undefined
      ? undefined
      : lowerCholesky(squareMatrix(options.covariance, dim, 'randomWalkMetropolis'), dim, 'randomWalkMetropolis')
  const scale = vectorOf(options.scale ?? 2.38 / Math.sqrt(dim), dim, 'randomWalkMetropolis')
  if (!scale.every((s) => s > 0 && Number.isFinite(s))) {
    throw new WhiteboxError('value', 'randomWalkMetropolis: scales must be positive and finite')
  }
  return rwmKernel(target, factor, scale, 'randomWalkMetropolis')
}

function rwmKernel(
  target: LogDensity,
  factor: Float64Array | undefined,
  scale: Float64Array,
  name: string,
  extra: Json = {},
): MCMCKernel {
  const dim = target.dim
  const step = (z: Float64Array): Float64Array =>
    factor === undefined ? z.map((v, i) => v * (scale[i] ?? 1)) : dgemv(factor, dim, dim, z)
  return {
    name,
    dim,
    config: {
      dim,
      ...(factor === undefined ? { scale: Array.from(scale) } : { covarianceFactor: Array.from(factor) }),
      ...(extra as Record<string, Json>),
    },
    init: (position) => initState(target, position, name),
    step: (key, state) => {
      const dz = step(normal(foldIn(key, 0), [dim]).data)
      const proposal = state.position.map((x, i) => x + (dz[i] ?? 0))
      const lp = target.logDensity(proposal)
      return transition(state, proposal, lp, metropolisAcceptance(state.logDensity, lp, 0, 0), foldIn(key, 1))
    },
  }
}

/**
 * Independence Metropolis-Hastings: proposals x' ~ q drawn regardless of the current point, accepted with
 * min(1, w(x') / w(x)) for importance weights w = pi / q. It mixes well exactly when importance sampling with q would.
 *
 * @param target - The target.
 * @param q - The proposal (see `proposalFromDistribution`).
 * @returns The kernel.
 * @throws WhiteboxError with code `shape` when the proposal's dimension differs from the target's.
 * @concept independence-sampler
 * @reference L. Tierney, Markov chains for exploring posterior distributions (Annals of Statistics 1994).
 * @example
 * ```ts
 * import { independenceMetropolis, numericTarget, proposalFromDistribution } from 'whitebox-engine/sampling'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * // A proposal equal to the target accepts every move.
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = independenceMetropolis(t, proposalFromDistribution(normal(0, 1)))
 * Math.abs(k.step(key(0), k.init([0.3])).info.acceptanceRate - 1) < 1e-12 // => true
 * ```
 */
export function independenceMetropolis(target: LogDensity, q: Proposal): MCMCKernel {
  if (q.dim !== target.dim) {
    throw new WhiteboxError(
      'shape',
      `independenceMetropolis: the proposal has dimension ${q.dim}, the target ${target.dim}`,
    )
  }
  return {
    name: 'independenceMetropolis',
    dim: target.dim,
    config: { dim: target.dim },
    init: (position) => initState(target, position, 'independenceMetropolis'),
    step: (key, state) => {
      const proposal = q.sample(foldIn(key, 0))
      const lp = target.logDensity(proposal)
      const alpha = metropolisAcceptance(state.logDensity, lp, q.logProb(proposal), q.logProb(state.position))
      return transition(state, proposal, lp, alpha, foldIn(key, 1))
    },
  }
}

/** The state of {@link adaptiveMetropolis}: the chain's running mean and covariance accumulator. */
export interface AdaptiveMetropolisState extends MCMCState {
  /** Points seen so far (including the start). */
  readonly count: number
  readonly mean: Float64Array
  /** The sum of outer products of deviations, row-major `[dim, dim]` (Welford). */
  readonly m2: Float64Array
}

/** Options of {@link adaptiveMetropolis}. */
export interface AdaptiveMetropolisOptions {
  /** The proposal standard deviation before adaptation starts (default 0.1). */
  readonly initialScale?: number
  /** Iterations before the empirical covariance is used (default 2 dim, at least 10). */
  readonly adaptAfter?: number
  /** The scale s_d of the empirical covariance (default 2.38^2 / dim, Gelman, Roberts and Gilks). */
  readonly scaleFactor?: number
  /** The regulariser eps added as s_d eps I (default 1e-6). */
  readonly epsilon?: number
}

/**
 * The adaptive Metropolis algorithm of Haario, Saksman and Tamminen: a random walk whose proposal covariance is
 * s_d (Cov(x_0, ..., x_{t-1}) + eps I), the running covariance of the chain's own history (updated in O(dim^2) per
 * step), after `adaptAfter` iterations of a fixed isotropic proposal. The adaptation never stops, yet the chain is
 * ergodic for the target (the adaptation diminishes as 1/t).
 *
 * @param target - The target.
 * @param options - Initial scale, adaptation start, scale factor and regulariser.
 * @returns The kernel (its state carries the running mean and covariance).
 * @throws WhiteboxError with code `value` for non-positive scales or a negative regulariser.
 * @concept adaptive-metropolis
 * @reference H. Haario, E. Saksman and J. Tamminen, An adaptive Metropolis algorithm (Bernoulli 2001).
 * @example
 * ```ts
 * import { adaptiveMetropolis, banana, sampleChain } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const r = sampleChain(adaptiveMetropolis(banana()), [0, 0], { key: key(0), numDraws: 200 })
 * r.state.count // => 201
 * ```
 */
export function adaptiveMetropolis(
  target: LogDensity,
  options: AdaptiveMetropolisOptions = {},
): MCMCKernel<AdaptiveMetropolisState> {
  const dim = target.dim
  const initialScale = options.initialScale ?? 0.1
  const adaptAfter = options.adaptAfter ?? Math.max(10, 2 * dim)
  const sd = options.scaleFactor ?? (2.38 * 2.38) / dim
  const eps = options.epsilon ?? 1e-6
  if (!(initialScale > 0 && sd > 0 && eps >= 0)) {
    throw new WhiteboxError('value', 'adaptiveMetropolis: scales must be positive and epsilon non-negative')
  }
  return {
    name: 'adaptiveMetropolis',
    dim,
    config: { dim, initialScale, adaptAfter, scaleFactor: sd, epsilon: eps },
    init: (position) => {
      const s = initState(target, position, 'adaptiveMetropolis')
      return { ...s, ...welfordUpdate(welfordInit(dim), s.position) }
    },
    step: (key, state) => {
      const z = normal(foldIn(key, 0), [dim]).data
      let dz: Float64Array
      if (state.count <= adaptAfter) dz = z.map((v) => initialScale * v)
      else {
        const cov = welfordCovariance(state).map((v, k) => sd * v + (k % (dim + 1) === 0 ? sd * eps : 0))
        dz = dgemv(lowerCholesky(cov, dim, 'adaptiveMetropolis'), dim, dim, z)
      }
      const proposal = state.position.map((x, i) => x + (dz[i] ?? 0))
      const lp = target.logDensity(proposal)
      const alpha = metropolisAcceptance(state.logDensity, lp, 0, 0)
      const accepted = acceptWith(foldIn(key, 1), alpha)
      const position = accepted ? proposal : state.position
      return {
        state: { position, logDensity: accepted ? lp : state.logDensity, ...welfordUpdate(state, position) },
        info: { acceptanceRate: alpha, accepted },
      }
    },
  }
}

/** Options of {@link randomWalkAdaptation}. */
export interface RandomWalkAdaptationOptions {
  /** The acceptance rate to aim for (default 0.234, optimal for high-dimensional Gaussian targets). */
  readonly targetAcceptance?: number
  /** Learn the proposal's shape from the covariance of the second half of warm-up (default true). */
  readonly adaptCovariance?: boolean
  /** The starting proposal standard deviation (default 2.38 / sqrt(dim)). */
  readonly initialScale?: number
}

interface RwmWarmupState {
  readonly chain: MCMCState
  readonly logScale: number
  readonly history: Welford
  /** The Cholesky factor of the learned shape, once enough of the second half has been seen. */
  readonly shape: Float64Array | undefined
}

function scaledKernel(target: LogDensity, logScale: number, shape: Float64Array | undefined): MCMCKernel {
  const s = Math.exp(logScale)
  return rwmKernel(
    target,
    shape?.map((v) => v * s),
    new Float64Array(target.dim).fill(s),
    'randomWalkMetropolis',
  )
}

/**
 * Warm-up for random-walk Metropolis with vanishing adaptation: the log proposal scale follows a Robbins-Monro
 * recursion (gain t^-0.6) towards the target acceptance rate, and, in the second half of warm-up, the proposal's shape
 * becomes the Cholesky factor of the running covariance of that half (the scale restarting at the optimal
 * 2.38 / sqrt(dim)). The tuned kernel is a plain random-walk Metropolis kernel, so sampling is exactly Markov.
 *
 * @param target - The target.
 * @param options - Target acceptance, shape learning and starting scale.
 * @returns The adaptation (pass to `sampleChains`); its parameters are the scale and the shape factor.
 * @throws WhiteboxError with code `value` unless 0 < targetAcceptance < 1.
 * @concept adaptive-metropolis
 * @reference C. Andrieu and J. Thoms, A tutorial on adaptive MCMC (Statistics and Computing 2008).
 * @example
 * ```ts
 * import { banana, randomWalkAdaptation, sampleChains } from 'whitebox-engine/sampling'
 * import { key } from 'whitebox-engine/random'
 * const c = sampleChains(randomWalkAdaptation(banana()), { key: key(0), numChains: 1, numWarmup: 200, numDraws: 50 })
 * typeof (c.parameters[0] as { scale: number }).scale // => 'number'
 * ```
 */
export function randomWalkAdaptation(target: LogDensity, options: RandomWalkAdaptationOptions = {}): Adaptation {
  const dim = target.dim
  const goal = options.targetAcceptance ?? 0.234
  if (!(goal > 0 && goal < 1)) {
    throw new WhiteboxError('value', 'randomWalkAdaptation: targetAcceptance must lie in (0, 1)')
  }
  const learnShape = options.adaptCovariance ?? true
  const initialScale = options.initialScale ?? 2.38 / Math.sqrt(dim)
  return {
    name: 'randomWalkAdaptation',
    dim,
    run: (position, run) => {
      const half = Math.floor(run.numWarmup / 2)
      const algorithm: Algorithm<RwmWarmupState> = {
        name: 'randomWalkAdaptation',
        config: { dim, targetAcceptance: goal, adaptCovariance: learnShape },
        schema: channels({
          metrics: {
            acceptanceRate: { description: 'acceptance probability of the transition', direction: 'higher' },
            scale: { description: 'proposal scale' },
            logDensity: { description: 'log density at the current position' },
          },
        }),
        init: () => ({
          chain: initState(target, position, 'randomWalkAdaptation'),
          logScale: Math.log(initialScale),
          history: welfordInit(dim),
          shape: undefined,
        }),
        step: (s, ctx) => {
          const tr = scaledKernel(target, s.logScale, s.shape).step(ctx.key, s.chain)
          run.onTransition?.(tr.state, tr.info)
          const t = ctx.t + 1
          let logScale = s.logScale + (tr.info.acceptanceRate - goal) / t ** 0.6
          let { history, shape } = s
          if (learnShape && t > half) {
            history = welfordUpdate(history, tr.state.position)
            if (history.count > 2 * dim + 2) {
              const cov = welfordCovariance(history).map((v, k) => v + (k % (dim + 1) === 0 ? 1e-10 : 0))
              if (shape === undefined) logScale = Math.log(2.38 / Math.sqrt(dim))
              shape = lowerCholesky(cov, dim, 'randomWalkAdaptation')
            }
          }
          return { chain: tr.state, logScale, history, shape }
        },
        observe: (s) => ({ metrics: { scale: Math.exp(s.logScale), logDensity: s.chain.logDensity } }),
      }
      const result = runAlgorithm(algorithm, {
        key: run.key,
        maxSteps: run.numWarmup,
        ...(run.parent === undefined ? {} : { parent: run.parent }),
        ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
        ...(run.policy === undefined ? {} : { policy: run.policy }),
      })
      const s = result.state
      const warmup: Warmup = {
        kernel: scaledKernel(target, s.logScale, s.shape),
        state: s.chain,
        parameters: {
          scale: Math.exp(s.logScale),
          ...(s.shape === undefined ? {} : { shapeFactor: Array.from(s.shape) }),
        },
      }
      return warmup
    },
  }
}
