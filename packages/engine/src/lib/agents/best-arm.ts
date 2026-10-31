/**
 * Best-arm identification in the fixed-confidence setting: instead of maximising reward, a bandit algorithm samples arms
 * until it can name the best one with probability at least `1 - delta`, and the figure of merit is how many pulls that
 * took (the stopping time). Successive elimination (Even-Dar et al. 2006), median elimination (Even-Dar et al. 2002,
 * the PAC form), LUCB (Kalyanakrishnan et al. 2012) and track-and-stop for Gaussian arms (Garivier and Kaufmann 2016),
 * each an Algorithm with a stopping rule and serialisable state that logs every pull, the estimates and intervals, the
 * eliminations and the decision; and the quantities the sample complexity is measured against (the gaps, the hardness
 * `H1` and the characteristic time of the lower bound).
 */

import { argmaxOf, MdmlError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { type Env, isBandit } from '../rl/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  event,
  type RunStatus,
  runAlgorithm,
  scalar,
  tensor,
} from '../trace/index.ts'

/** The paths the algorithms of this module log, for a method's contract. */
export type BestArmPath =
  | 'pull/arm'
  | 'pull/reward'
  | 'arms/active'
  | 'arms/estimates'
  | 'arms/counts'
  | 'arms/lower'
  | 'arms/upper'
  | 'arm/eliminated'
  | 'stop/statistic'
  | 'stop/threshold'
  | 'stop/decision'

/** The state of a best-arm algorithm: statistics per arm, the active set and where the algorithm is in its schedule. */
export interface BestArmState {
  /** The state of the bandit environment. */
  readonly envState: unknown
  /** Pulls and reward sums per arm. */
  readonly counts: Float64Array
  readonly sums: Float64Array
  /** Pulls and reward sums within the current stage (median elimination). */
  readonly stageCounts: Float64Array
  readonly stageSums: Float64Array
  /** Whether each arm is still in play. */
  readonly active: Uint8Array
  /** The total number of pulls, which is the stopping time once the algorithm has stopped. */
  readonly pulls: number
  /** Completed rounds (elimination: pulls per active arm; LUCB: pairs sampled). */
  readonly round: number
  /** The stage of median elimination, with its accuracy and confidence. */
  readonly stage: number
  readonly stageEpsilon: number
  readonly stageDelta: number
  /** LUCB: the pair being sampled this round and the sub-step within it. */
  readonly pair: readonly [number, number]
  readonly phase: number
  /** The last value of the stopping statistic and its threshold. */
  readonly statistic: number
  readonly threshold: number
  /** The arm recommended once stopped, else -1. */
  readonly recommendation: number
  /** The arms eliminated, in order. */
  readonly eliminated: readonly number[]
}

/** What a finished (or budget-stopped) run reports. */
export interface BestArmResult {
  /** The recommended arm (the empirical best among those left if the budget ran out first). */
  readonly recommendation: number
  /** The pulls made. */
  readonly stoppingTime: number
  /** Whether the algorithm stopped by its own rule rather than the budget. */
  readonly stopped: boolean
  /** Pulls and empirical means per arm. */
  readonly counts: readonly number[]
  readonly means: readonly number[]
  /** The arms eliminated, in order. */
  readonly eliminated: readonly number[]
  /** Whether the recommendation is the arm with the largest true mean, by the environment's oracle. */
  readonly correct: boolean
  readonly status: RunStatus
}

/** Options shared by the best-arm algorithms. */
export interface BestArmOptions {
  /** The confidence: the algorithm names a wrong arm with probability at most `delta` (default 0.05). */
  readonly delta?: number
  /** The sub-Gaussian scale of the rewards: 0.5 for rewards in [0, 1] (default), the standard deviation for Gaussian arms. */
  readonly sigma?: number
}

const requireDelta = (delta: number, name: string): void => {
  if (!(delta > 0 && delta < 1)) throw new MdmlError('value', `${name}: delta ${delta} must be in (0, 1)`)
}

function armsOf<S>(env: Env<S, unknown, number>, name: string): number {
  if (env.actionSpace.kind !== 'discrete') throw new MdmlError('value', `${name}: needs a bandit with discrete arms`)
  return env.actionSpace.n
}

const meanOf = (s: BestArmState, i: number): number =>
  (s.counts[i] ?? 0) === 0 ? 0 : (s.sums[i] ?? 0) / (s.counts[i] ?? 1)

/** The arm with the largest empirical mean among those in play (the first on ties). */
function leader(s: BestArmState): number {
  let best = -1
  for (let i = 0; i < s.counts.length; i++) {
    if ((s.active[i] ?? 0) === 0) continue
    if (best < 0 || meanOf(s, i) > meanOf(s, best)) best = i
  }
  return best
}

function initialState<S, O>(env: Env<S, O, number>, k: Key, arms: number): BestArmState {
  return {
    envState: env.reset(foldIn(k, 0)).state,
    counts: new Float64Array(arms),
    sums: new Float64Array(arms),
    stageCounts: new Float64Array(arms),
    stageSums: new Float64Array(arms),
    active: new Uint8Array(arms).fill(1),
    pulls: 0,
    round: 0,
    stage: 0,
    stageEpsilon: 0,
    stageDelta: 0,
    pair: [0, 0],
    phase: 0,
    statistic: 0,
    threshold: 0,
    recommendation: -1,
    eliminated: [],
  }
}

/** Pull an arm: advance the environment and add the reward to the statistics. */
function pullArm<S, O>(
  env: Env<S, O, number>,
  s: BestArmState,
  arm: number,
  k: Key,
): BestArmState & { reward: number } {
  const step = env.step(s.envState as S, arm, foldIn(k, 1))
  const counts = s.counts.slice()
  const sums = s.sums.slice()
  const stageCounts = s.stageCounts.slice()
  const stageSums = s.stageSums.slice()
  counts[arm] = (counts[arm] ?? 0) + 1
  sums[arm] = (sums[arm] ?? 0) + step.reward
  stageCounts[arm] = (stageCounts[arm] ?? 0) + 1
  stageSums[arm] = (stageSums[arm] ?? 0) + step.reward
  return { ...s, envState: step.state, counts, sums, stageCounts, stageSums, pulls: s.pulls + 1, reward: step.reward }
}

type Ctx = Parameters<Algorithm<BestArmState, BestArmPath>['step']>[1]

function logPull(ctx: Ctx, arm: number, reward: number, s: BestArmState): void {
  ctx.log('pull/arm', arm)
  ctx.log('pull/reward', reward)
  ctx.log(
    'arms/active',
    s.active.reduce((a, b) => a + b, 0),
  )
  if (ctx.wants('arms/estimates'))
    ctx.log(
      'arms/estimates',
      Array.from(s.counts, (_, i) => meanOf(s, i)),
    )
  if (ctx.wants('arms/counts')) ctx.log('arms/counts', s.counts)
}

/** The contract fragment of the best-arm algorithms: what each path is, for a method to spread into its own. */
export const bestArmContract: Contract<BestArmState, BestArmPath> = defineContract({
  'pull/arm': scalar({ description: 'the arm pulled' }),
  'pull/reward': scalar({ description: 'the reward it paid' }),
  'arms/active': scalar({ description: 'the number of arms still in play' }),
  'arms/estimates': tensor({ role: 'state', description: 'the empirical mean of every arm', shape: ['arms'] }),
  'arms/counts': tensor({ role: 'state', description: 'the pulls of every arm', shape: ['arms'] }),
  'arms/lower': tensor({ role: 'state', description: 'the lower confidence bound of every arm', shape: ['arms'] }),
  'arms/upper': tensor({ role: 'state', description: 'the upper confidence bound of every arm', shape: ['arms'] }),
  'arm/eliminated': event({
    context: { arm: 'the arm dropped', pulls: 'the pulls so far', mean: 'its empirical mean' },
  }),
  'stop/statistic': scalar({ description: 'the stopping statistic (the gap between the leaders in confidence units)' }),
  'stop/threshold': scalar({ description: 'what the statistic must exceed to stop' }),
  'stop/decision': event({ context: { arm: 'the arm recommended', pulls: 'the stopping time' } }),
})

function drop(ctx: Ctx, s: BestArmState, arm: number): BestArmState {
  const active = s.active.slice()
  active[arm] = 0
  ctx.emit('arm/eliminated', { arm, pulls: s.pulls, mean: meanOf(s, arm) })
  return { ...s, active, eliminated: [...s.eliminated, arm] }
}

function stop(ctx: Ctx, s: BestArmState, arm: number): BestArmState {
  ctx.emit('stop/decision', { arm, pulls: s.pulls })
  return { ...s, recommendation: arm }
}

const doneOf =
  (budget: number) =>
  (s: BestArmState): RunStatus | undefined =>
    s.recommendation >= 0 ? 'completed' : s.pulls >= budget ? 'stopped' : undefined

// ----------------------------------------------------------------------------------------- successive elimination

/** Options of {@link successiveEliminationAlgorithm}. */
export interface SuccessiveEliminationOptions extends BestArmOptions {
  /** The most pulls to make (default 1,000,000). */
  readonly maxPulls?: number
}

/**
 * Successive elimination (Even-Dar, Mannor and Mansour 2006): pull every arm still in play once per round; after `n`
 * pulls each, with the confidence radius `c_n = sigma sqrt(2 log(4 K n^2 / delta) / n)` that holds for every arm and
 * every `n` at once with probability `1 - delta`, drop each arm whose empirical mean lies more than `2 c_n` below the
 * leader's, and stop when one arm remains. An arm with gap `Delta` is dropped after about `log(K / (delta Delta)) / Delta^2`
 * pulls.
 *
 * @param env - The bandit (its oracle is used only to score the result).
 * @param options - The confidence, the sub-Gaussian scale and the budget.
 * @returns The algorithm; its state's `recommendation` is the arm named once it stops.
 * @throws MdmlError with code `value` for `delta` outside (0, 1) or fewer than two arms.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { successiveEliminationAlgorithm } from 'mdml-engine/agents'
 * import { bernoulliBandit } from 'mdml-engine/environments'
 * successiveEliminationAlgorithm(bernoulliBandit.create({ probs: [0.2, 0.8] })).name // => 'successive-elimination'
 * ```
 */
export function successiveEliminationAlgorithm<S, O>(
  env: Env<S, O, number>,
  options: SuccessiveEliminationOptions = {},
): Algorithm<BestArmState, BestArmPath> {
  const arms = armsOf(env, 'successiveElimination')
  const delta = options.delta ?? 0.05
  const sigma = options.sigma ?? 0.5
  requireDelta(delta, 'successiveElimination')
  if (arms < 2) throw new MdmlError('value', 'successiveElimination: needs at least two arms')
  const radius = (n: number): number => sigma * Math.sqrt((2 * Math.log((4 * arms * n * n) / delta)) / n)
  return {
    name: 'successive-elimination',
    config: { arms, delta, sigma },
    init: (k) => initialState(env, k, arms),
    step: (s0, ctx) => {
      // The next pull goes to the active arm with the fewest pulls (round-robin in index order).
      let arm = -1
      for (let i = 0; i < arms; i++) {
        if ((s0.active[i] ?? 0) === 1 && (arm < 0 || (s0.counts[i] ?? 0) < (s0.counts[arm] ?? 0))) arm = i
      }
      const pulled = pullArm(env, s0, arm, ctx.key)
      let s: BestArmState = pulled
      logPull(ctx, arm, pulled.reward, s)
      const n = Math.min(...Array.from(s.counts, (c, i) => ((s.active[i] ?? 0) === 1 ? c : Number.POSITIVE_INFINITY)))
      if (n > s.round && Number.isFinite(n)) {
        // A round has just completed: every active arm has n pulls.
        s = { ...s, round: n }
        const c = radius(n)
        const top = meanOf(s, leader(s))
        if (ctx.wants('arms/lower') || ctx.wants('arms/upper')) {
          ctx.log(
            'arms/lower',
            Array.from(s.counts, (_, i) => meanOf(s, i) - c),
          )
          ctx.log(
            'arms/upper',
            Array.from(s.counts, (_, i) => meanOf(s, i) + c),
          )
        }
        ctx.log('stop/threshold', 2 * c)
        ctx.log(
          'stop/statistic',
          top - Math.min(...Array.from(s.counts, (_, i) => ((s.active[i] ?? 0) === 1 ? meanOf(s, i) : top))),
        )
        for (let i = 0; i < arms; i++) {
          if ((s.active[i] ?? 0) === 1 && top - meanOf(s, i) > 2 * c) s = drop(ctx, s, i)
        }
        if ([...s.active].reduce((a, b) => a + b, 0) === 1) s = stop(ctx, s, leader(s))
      }
      return s
    },
    done: doneOf(options.maxPulls ?? 1_000_000),
  }
}

// -------------------------------------------------------------------------------------------- median elimination

/** Options of {@link medianEliminationAlgorithm}. */
export interface MedianEliminationOptions extends BestArmOptions {
  /** The accuracy: the arm returned is within `epsilon` of the best with probability `1 - delta` (default 0.1). */
  readonly epsilon?: number
  /** The most pulls to make (default 10,000,000). */
  readonly maxPulls?: number
}

/**
 * Median elimination (Even-Dar, Mannor and Mansour 2002): an (epsilon, delta)-PAC algorithm that needs no gap. In stage
 * `l`, with `epsilon_l` and `delta_l` starting at `epsilon / 4` and `delta / 2`, it pulls every arm in play
 * `(2 sigma)^2 * 4 log(3 / delta_l) / epsilon_l^2` times, estimates the means from this stage's pulls alone, and drops the
 * worse half; then `epsilon_{l+1} = 3 epsilon_l / 4` and `delta_{l+1} = delta_l / 2`. The arm left is epsilon-optimal with
 * probability `1 - delta`, after `O(K log(1/delta) / epsilon^2)` pulls whatever the gaps.
 *
 * @param env - The bandit.
 * @param options - The accuracy, confidence, sub-Gaussian scale and budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for `delta` outside (0, 1), a non-positive `epsilon`, or fewer than two arms.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { medianEliminationAlgorithm } from 'mdml-engine/agents'
 * import { bernoulliBandit } from 'mdml-engine/environments'
 * medianEliminationAlgorithm(bernoulliBandit.create({ probs: [0.2, 0.8] })).name // => 'median-elimination'
 * ```
 */
export function medianEliminationAlgorithm<S, O>(
  env: Env<S, O, number>,
  options: MedianEliminationOptions = {},
): Algorithm<BestArmState, BestArmPath> {
  const arms = armsOf(env, 'medianElimination')
  const delta = options.delta ?? 0.05
  const epsilon = options.epsilon ?? 0.1
  const sigma = options.sigma ?? 0.5
  requireDelta(delta, 'medianElimination')
  if (!(epsilon > 0)) throw new MdmlError('value', 'medianElimination: epsilon must be positive')
  if (arms < 2) throw new MdmlError('value', 'medianElimination: needs at least two arms')
  const stagePulls = (e: number, d: number): number => Math.ceil(((2 * sigma) ** 2 * 4 * Math.log(3 / d)) / (e * e))
  return {
    name: 'median-elimination',
    config: { arms, delta, epsilon, sigma },
    init: (k) => ({ ...initialState(env, k, arms), stage: 1, stageEpsilon: epsilon / 4, stageDelta: delta / 2 }),
    step: (s0, ctx) => {
      const need = stagePulls(s0.stageEpsilon, s0.stageDelta)
      let arm = -1
      for (let i = 0; i < arms; i++) {
        if ((s0.active[i] ?? 0) === 1 && (arm < 0 || (s0.stageCounts[i] ?? 0) < (s0.stageCounts[arm] ?? 0))) arm = i
      }
      const pulled = pullArm(env, s0, arm, ctx.key)
      let s: BestArmState = pulled
      logPull(ctx, arm, pulled.reward, s)
      const done = Array.from(s.active).every((a, i) => a === 0 || (s.stageCounts[i] ?? 0) >= need)
      if (!done) return s
      // The stage is complete: drop the arms below the median of this stage's means, keeping the better half.
      const live = Array.from(s.active, (a, i) => (a === 1 ? i : -1)).filter((i) => i >= 0)
      const mean = (i: number): number => (s.stageSums[i] ?? 0) / Math.max(1, s.stageCounts[i] ?? 1)
      const ranked = [...live].sort((a, b) => mean(b) - mean(a) || a - b)
      const keep = Math.ceil(live.length / 2)
      ctx.log('stop/statistic', mean(ranked[0] ?? 0) - mean(ranked[Math.min(keep, ranked.length - 1)] ?? 0))
      ctx.log('stop/threshold', s.stageEpsilon)
      for (const i of ranked.slice(keep)) s = drop(ctx, s, i)
      s = {
        ...s,
        round: s.round + 1,
        stage: s.stage + 1,
        stageEpsilon: (3 * s.stageEpsilon) / 4,
        stageDelta: s.stageDelta / 2,
        stageCounts: new Float64Array(arms),
        stageSums: new Float64Array(arms),
      }
      if (keep === 1) s = stop(ctx, s, ranked[0] ?? 0)
      return s
    },
    done: doneOf(options.maxPulls ?? 10_000_000),
  }
}

// ------------------------------------------------------------------------------------------------------- LUCB

/** Options of {@link lucbAlgorithm}. */
export interface LucbOptions extends BestArmOptions {
  /** Stop when the leader's lower bound is within `epsilon` of the challenger's upper bound (default 0: exact). */
  readonly epsilon?: number
  /** The most pulls to make (default 1,000,000). */
  readonly maxPulls?: number
}

/**
 * LUCB (Kalyanakrishnan, Tewari, Auer and Stone 2012) for the single best arm: after pulling every arm once, each round
 * pulls the empirical leader `h` and its most dangerous challenger `l`, the other arm with the highest upper confidence
 * bound, using `beta(u, t) = sigma sqrt(2 log(5 K t^4 / (4 delta)) / u)` as the radius of an arm pulled `u` times in
 * round `t`; it stops when the leader's lower bound exceeds the challenger's upper bound (less `epsilon`). The pulls
 * concentrate on the arms that are hard to tell apart.
 *
 * @param env - The bandit.
 * @param options - The confidence, the sub-Gaussian scale, the slack `epsilon` and the budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for `delta` outside (0, 1) or fewer than two arms.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { lucbAlgorithm } from 'mdml-engine/agents'
 * import { bernoulliBandit } from 'mdml-engine/environments'
 * lucbAlgorithm(bernoulliBandit.create({ probs: [0.2, 0.8] })).name // => 'lucb'
 * ```
 */
export function lucbAlgorithm<S, O>(
  env: Env<S, O, number>,
  options: LucbOptions = {},
): Algorithm<BestArmState, BestArmPath> {
  const arms = armsOf(env, 'lucb')
  const delta = options.delta ?? 0.05
  const sigma = options.sigma ?? 0.5
  const slack = options.epsilon ?? 0
  requireDelta(delta, 'lucb')
  if (arms < 2) throw new MdmlError('value', 'lucb: needs at least two arms')
  const beta = (u: number, t: number): number =>
    sigma * Math.sqrt((2 * Math.log((5 * arms * t ** 4) / (4 * delta))) / u)
  return {
    name: 'lucb',
    config: { arms, delta, sigma, epsilon: slack },
    init: (k) => initialState(env, k, arms),
    step: (s0, ctx) => {
      // The first K steps pull each arm once; afterwards a round is two steps, the leader then the challenger.
      let arm: number
      if (s0.pulls < arms) arm = s0.pulls
      else if (s0.phase === 0) {
        const t = s0.round + 1
        const h = leader(s0)
        let l = -1
        let best = Number.NEGATIVE_INFINITY
        for (let i = 0; i < arms; i++) {
          if (i === h) continue
          const ucb = meanOf(s0, i) + beta(s0.counts[i] ?? 1, t)
          if (ucb > best) {
            best = ucb
            l = i
          }
        }
        s0 = { ...s0, pair: [h, l] }
        arm = h
      } else arm = s0.pair[1]
      const pulled = pullArm(env, s0, arm, ctx.key)
      let s: BestArmState = pulled
      logPull(ctx, arm, pulled.reward, s)
      if (s.pulls <= arms) return s
      if (s.phase === 0) return { ...s, phase: 1 }
      const t = s.round + 1
      const [h, l] = s.pair
      const lower = meanOf(s, h) - beta(s.counts[h] ?? 1, t)
      const upper = meanOf(s, l) + beta(s.counts[l] ?? 1, t)
      s = { ...s, phase: 0, round: t, statistic: lower - upper, threshold: -slack }
      ctx.log('stop/statistic', lower - upper)
      ctx.log('stop/threshold', -slack)
      if (ctx.wants('arms/lower') || ctx.wants('arms/upper')) {
        ctx.log(
          'arms/lower',
          Array.from(s.counts, (c, i) => meanOf(s, i) - beta(c || 1, t)),
        )
        ctx.log(
          'arms/upper',
          Array.from(s.counts, (c, i) => meanOf(s, i) + beta(c || 1, t)),
        )
      }
      if (lower > upper - slack) s = stop(ctx, s, h)
      return s
    },
    done: doneOf(options.maxPulls ?? 1_000_000),
  }
}

// ------------------------------------------------------------------------------------------------ track-and-stop

/** The optimal allocation of pulls for identifying the best arm of Gaussian arms, and the rate it achieves. */
export interface CharacteristicTime {
  /** The fraction of pulls each arm should get. */
  readonly weights: readonly number[]
  /** `T*(mu)^-1`: the expected stopping time is at least this times `log(1 / (2.4 delta))` (asymptotically equal). */
  readonly time: number
}

/**
 * The characteristic time of best-arm identification for Gaussian arms of known variance (Garivier and Kaufmann 2016):
 * the value `T*(mu)` of `max_w min_{a != best} w_best w_a / (w_best + w_a) * Delta_a^2 / (2 sigma^2)` over the simplex,
 * whose inverse multiplies `log(1/delta)` in the lower bound on any delta-correct algorithm's expected stopping time, and
 * the weights `w*` achieving it. The optimum equalises the terms and satisfies `sum_a (w_a / w_best)^2 = 1`, which a
 * bisection on the common value finds.
 *
 * @param means - The arms' means (the best one is the largest).
 * @param sigma - The standard deviation of the rewards (default 1).
 * @returns The weights and `1 / T*`.
 * @throws MdmlError with code `value` for fewer than two arms or a best arm that is not unique.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { characteristicTime } from 'mdml-engine/agents'
 * const { weights, time } = characteristicTime([1, 0, 0])
 * [weights[1] === weights[2], Math.round(time * 100) / 100] // => [true, 11.66]
 * ```
 */
export function characteristicTime(means: readonly number[], sigma = 1): CharacteristicTime {
  if (means.length < 2) throw new MdmlError('value', 'characteristicTime: needs at least two arms')
  const best = argmaxOf(means)
  const gaps = means.map((m) => (means[best] ?? 0) - m)
  const others = gaps.flatMap((g, i) => (i === best ? [] : [g]))
  if (others.some((g) => !(g > 0))) throw new MdmlError('value', 'characteristicTime: the best arm must be unique')
  // With the best arm's weight 1, the others have x_a = 1 / (Delta_a^2 / (2 sigma^2 T) - 1) for the common value T.
  const total = (t: number): number =>
    others.reduce((a, g) => a + (1 / ((g * g) / (2 * sigma * sigma * t) - 1)) ** 2, 0)
  let lo = 0
  let hi = Math.min(...others) ** 2 / (2 * sigma * sigma)
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2
    if (total(mid) < 1) lo = mid
    else hi = mid
  }
  const t = (lo + hi) / 2
  const x = gaps.map((g, i) => (i === best ? 1 : 1 / ((g * g) / (2 * sigma * sigma * t) - 1)))
  const sum = x.reduce((a, b) => a + b, 0)
  const weights = x.map((v) => v / sum)
  const w1 = weights[best] ?? 0
  const value = Math.min(
    ...gaps.flatMap((g, i) =>
      i === best ? [] : [(w1 * (weights[i] ?? 0) * g * g) / ((w1 + (weights[i] ?? 0)) * 2 * sigma * sigma)],
    ),
  )
  return { weights, time: 1 / value }
}

/**
 * The hardness `H1 = sum_a 1 / Delta_a^2` of a bandit (with the best arm's own gap taken as the gap to the runner-up):
 * the quantity successive elimination's and LUCB's sample complexity is proportional to, up to logarithms.
 *
 * @param means - The arms' means.
 * @returns `H1`.
 * @throws MdmlError with code `value` for fewer than two arms or a tie for best.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { hardness } from 'mdml-engine/agents'
 * hardness([0.9, 0.5, 0.5]) // => 18.75
 * ```
 */
export function hardness(means: readonly number[]): number {
  if (means.length < 2) throw new MdmlError('value', 'hardness: needs at least two arms')
  const sorted = [...means].sort((a, b) => b - a)
  const top = sorted[0] ?? 0
  const second = sorted[1] ?? 0
  if (!(top > second)) throw new MdmlError('value', 'hardness: the best arm must be unique')
  let h = 1 / (top - second) ** 2
  for (const m of sorted.slice(1)) h += 1 / (top - m) ** 2
  return h
}

/** Options of {@link trackAndStopAlgorithm}. */
export interface TrackAndStopOptions {
  /** The confidence (default 0.05). */
  readonly delta?: number
  /** The known standard deviation of the Gaussian rewards (default 1). */
  readonly sigma?: number
  /**
   * The stopping threshold `beta(t, delta)` the generalised likelihood ratio must exceed (default
   * `log((K - 1) (1 + log t) / delta)`, the heuristic of the paper's experiments made conservative by the factor `K - 1`).
   */
  readonly threshold?: (t: number, delta: number, arms: number) => number
  /** The most pulls to make (default 1,000,000). */
  readonly maxPulls?: number
}

/**
 * Track-and-stop (Garivier and Kaufmann 2016) for Gaussian arms of known variance: sample so that the pulls track the
 * optimal allocation of {@link characteristicTime} computed at the current estimates (D-tracking: an arm with fewer than
 * `sqrt(t) - K/2` pulls is forced, otherwise the pull goes to the arm furthest below its target share), and stop when
 * the generalised likelihood ratio `min_{b != a} N_a N_b / (N_a + N_b) (mu_a - mu_b)^2 / (2 sigma^2)` of the empirical
 * leader `a` exceeds the threshold. Its expected stopping time approaches the lower bound as `delta` shrinks.
 *
 * @param env - The bandit with Gaussian arms.
 * @param options - The confidence, standard deviation, threshold and budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for `delta` outside (0, 1) or fewer than two arms.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { trackAndStopAlgorithm } from 'mdml-engine/agents'
 * import { gaussianBandit } from 'mdml-engine/environments'
 * trackAndStopAlgorithm(gaussianBandit.create({ means: [1, 0] })).name // => 'track-and-stop'
 * ```
 */
export function trackAndStopAlgorithm<S, O>(
  env: Env<S, O, number>,
  options: TrackAndStopOptions = {},
): Algorithm<BestArmState, BestArmPath> {
  const arms = armsOf(env, 'trackAndStop')
  const delta = options.delta ?? 0.05
  const sigma = options.sigma ?? 1
  requireDelta(delta, 'trackAndStop')
  if (arms < 2) throw new MdmlError('value', 'trackAndStop: needs at least two arms')
  const threshold = options.threshold ?? ((t, d, k) => Math.log(((k - 1) * (1 + Math.log(Math.max(t, 1)))) / d))
  const statistic = (s: BestArmState): number => {
    const a = leader(s)
    let z = Number.POSITIVE_INFINITY
    for (let b = 0; b < arms; b++) {
      if (b === a) continue
      const na = s.counts[a] ?? 0
      const nb = s.counts[b] ?? 0
      const gap = meanOf(s, a) - meanOf(s, b)
      z = Math.min(z, (((na * nb) / (na + nb)) * (gap * gap)) / (2 * sigma * sigma))
    }
    return z
  }
  const target = (s: BestArmState): readonly number[] => {
    const means = Array.from(s.counts, (_, i) => meanOf(s, i))
    // Ties in the estimates would make the allocation singular: separate them by a hair, in index order.
    const top = argmaxOf(means)
    const guarded = means.map((m, i) => (i === top ? m : Math.min(m, (means[top] ?? 0) - 1e-3)))
    return characteristicTime(guarded, sigma).weights
  }
  return {
    name: 'track-and-stop',
    config: { arms, delta, sigma },
    init: (k) => initialState(env, k, arms),
    step: (s0, ctx) => {
      const t = s0.pulls
      let arm: number
      if (t < arms) arm = t
      else {
        const forced = Math.sqrt(t) - arms / 2
        const starved = argmaxOf(Array.from(s0.counts, (c) => -c))
        if ((s0.counts[starved] ?? 0) < forced) arm = starved
        else {
          const w = target(s0)
          arm = argmaxOf(Array.from(s0.counts, (c, i) => t * (w[i] ?? 0) - c))
        }
      }
      const pulled = pullArm(env, s0, arm, ctx.key)
      let s: BestArmState = pulled
      logPull(ctx, arm, pulled.reward, s)
      if (s.pulls < arms) return s
      const z = statistic(s)
      const beta = threshold(s.pulls, delta, arms)
      ctx.log('stop/statistic', z)
      ctx.log('stop/threshold', beta)
      s = { ...s, statistic: z, threshold: beta }
      if (z > beta) s = stop(ctx, s, leader(s))
      return s
    },
    done: doneOf(options.maxPulls ?? 1_000_000),
  }
}

// ------------------------------------------------------------------------------------------------------ running

/**
 * Run a best-arm algorithm to its stopping rule (or its budget) and score the recommendation against the bandit's oracle.
 *
 * @param env - The bandit the algorithm was built for.
 * @param algorithm - The algorithm.
 * @param options - The seed or key and a cap on algorithm steps.
 * @returns The recommendation, the stopping time, the counts and means, and whether the arm named is the best.
 * @concept best-arm-identification
 * @example
 * ```ts
 * import { lucbAlgorithm, runBestArm } from 'mdml-engine/agents'
 * import { bernoulliBandit } from 'mdml-engine/environments'
 * const env = bernoulliBandit.create({ probs: [0.1, 0.9] })
 * runBestArm(env, lucbAlgorithm(env), { seed: 1 }).recommendation // => 1
 * ```
 */
export function runBestArm<S, O>(
  env: Env<S, O, number>,
  algorithm: Algorithm<BestArmState, BestArmPath>,
  options: { readonly seed?: number; readonly key?: Key; readonly maxSteps?: number } = {},
): BestArmResult {
  const result = runAlgorithm(algorithm, {
    maxSteps: options.maxSteps ?? 10_000_000,
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const s = result.state
  const recommendation = s.recommendation >= 0 ? s.recommendation : leader(s)
  if (!isBandit(env)) throw new MdmlError('value', `runBestArm: ${env.spec.id} has no oracle to score the arm by`)
  const best = env.oracle.optimalAction?.(s.envState as S)
  const truth = env.oracle
  const armsCount = s.counts.length
  const optimal =
    best ?? argmaxOf(Array.from({ length: armsCount }, (_, a) => truth.expectedReward(s.envState as S, a)))
  return {
    recommendation,
    stoppingTime: s.pulls,
    stopped: s.recommendation >= 0,
    counts: Array.from(s.counts),
    means: Array.from(s.counts, (_, i) => meanOf(s, i)),
    eliminated: s.eliminated,
    correct: recommendation === optimal,
    status: result.status,
  }
}
