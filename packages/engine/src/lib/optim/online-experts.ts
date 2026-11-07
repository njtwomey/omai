/**
 * Prediction with expert advice and online convex optimisation beyond `online.ts`, each as a stepped algorithm with the
 * regret on the tape: weighted majority (deterministic and randomised, with the mistake bounds), follow the leader (which
 * fails on an alternating sequence), follow the perturbed leader (exponential perturbations), Hedge, fixed share and
 * variable share (for tracking a switching best expert, with the best k-switching comparator found by dynamic
 * programming), Exp4 under bandit feedback with its regret bound, the p-norm online mirror descent, AdaGrad with box
 * constraints, and the comparator in hindsight for convex losses. `online.ts` has the Euclidean, entropic (exponentiated
 * gradient) and Newton-step learners that these sit beside.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { categorical, exponential, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { add, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { lbfgsb } from './minimize/lbfgsb.ts'
import type { OnlineLearner, OnlineLoss } from './online.ts'

/** The rounds of a prediction-with-expert-advice problem. */
export interface ExpertsProblem {
  /** T x N losses in [0, 1]: `losses[t][i]` is expert i's loss in round t. */
  readonly losses: readonly (readonly number[])[]
  /** For binary prediction (weighted majority): T x N predictions in {0, 1}. */
  readonly predictions?: readonly (readonly number[])[]
  /** For binary prediction: the T outcomes in {0, 1}. */
  readonly outcomes?: readonly number[]
}

/** What a rule plays in a round: the distribution over experts and the loss it suffers. */
export interface RoundPlay {
  readonly probabilities: Float64Array
  /** The learner's loss: `probabilities . losses` unless the rule predicts (weighted majority) and errs or not. */
  readonly loss: number
}

/** A rule for choosing among experts. Its state must be plain data. */
export interface ExpertsRule<S> {
  readonly name: string
  readonly config: Json
  init(n: number, key: Key): S
  /** Decide, given the state and the round (the losses are given so that predictions can be scored). */
  play(
    state: S,
    round: {
      readonly t: number
      readonly losses: ArrayLike<number>
      readonly predictions?: ArrayLike<number>
      readonly outcome?: number
    },
    key: Key,
  ): RoundPlay
  /** Fold in the round's losses. */
  update(state: S, round: { readonly t: number; readonly losses: ArrayLike<number> }, key: Key): S
}

const normalise = (w: Float64Array): Float64Array => {
  let s = 0
  for (const v of w) s += v
  return s > 0 ? w.map((v) => v / s) : w.fill(1 / w.length)
}

/** The state shared by the loss-vector rules: cumulative losses and (for the multiplicative rules) the weights. */
export interface WeightState {
  readonly cumulative: readonly number[]
  readonly weights: readonly number[]
}

const initWeights = (n: number): WeightState => ({
  cumulative: new Array<number>(n).fill(0),
  weights: new Array<number>(n).fill(1 / n),
})
const addLoss = (cum: readonly number[], l: ArrayLike<number>): number[] => cum.map((c, i) => c + (l[i] ?? 0))

/**
 * Follow the leader: play the expert with the smallest cumulative loss so far (the lowest index on ties). Optimal on
 * stochastic sequences and with regret linear in T on an adversarial one, such as losses alternating between (1/2, 0), (0, 1),
 * (1, 0), ... where the leader changes every round and is always wrong.
 *
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { followTheLeader } from 'mdml-engine/optim'
 * followTheLeader().name // => 'follow-the-leader'
 * ```
 */
export function followTheLeader(): ExpertsRule<WeightState> {
  return {
    name: 'follow-the-leader',
    config: {},
    init: (n) => initWeights(n),
    play: (s, round) => {
      let best = 0
      s.cumulative.forEach((c, i) => {
        if (c < (s.cumulative[best] ?? 0)) best = i
      })
      const p = new Float64Array(s.cumulative.length)
      p[best] = 1
      return { probabilities: p, loss: round.losses[best] ?? 0 }
    },
    update: (s, round) => ({ ...s, cumulative: addLoss(s.cumulative, round.losses) }),
  }
}

/**
 * Follow the perturbed leader (Kalai and Vempala 2005; Hutter and Poland 2005): each round, perturb every expert's cumulative
 * loss by an independent exponential variable of mean 1/eta, subtracted, and play the minimiser. The randomisation defeats
 * the alternating sequence that breaks follow the leader; the expected regret is of order sqrt(T ln N) at eta of order
 * sqrt(ln N / T), and larger perturbations (smaller eta) trade regret for stability. The loss recorded is the realised
 * loss of the sampled expert.
 *
 * @param eta - The inverse perturbation scale; the perturbation is exponential with mean 1/eta.
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { followThePerturbedLeader } from 'mdml-engine/optim'
 * followThePerturbedLeader(0.1).config // => { eta: 0.1 }
 * ```
 */
export function followThePerturbedLeader(eta: number): ExpertsRule<WeightState> {
  if (!(eta > 0)) throw new MdmlError('value', 'followThePerturbedLeader: eta must be positive')
  return {
    name: 'follow-the-perturbed-leader',
    config: { eta },
    init: (n) => initWeights(n),
    play: (s, round, key) => {
      const n = s.cumulative.length
      const z = toFlat(exponential(key, [n], eta))
      let best = 0
      let bestVal = Number.POSITIVE_INFINITY
      for (let i = 0; i < n; i++) {
        const v = (s.cumulative[i] ?? 0) - (z[i] ?? 0)
        if (v < bestVal) {
          bestVal = v
          best = i
        }
      }
      const p = new Float64Array(n)
      p[best] = 1
      return { probabilities: p, loss: round.losses[best] ?? 0 }
    },
    update: (s, round) => ({ ...s, cumulative: addLoss(s.cumulative, round.losses) }),
  }
}

/**
 * Hedge, the exponential weights algorithm: play the distribution proportional to exp(-eta L_i) over the cumulative
 * losses L; with losses in [0, 1] the regret against the best expert is at most ln N / eta + eta T / 8
 * (`hedgeRegretBound` in `learn/theory.ts`).
 *
 * @param eta - The learning rate.
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { hedgeRule } from 'mdml-engine/optim'
 * hedgeRule(0.5).name // => 'hedge'
 * ```
 */
export function hedgeRule(eta: number): ExpertsRule<WeightState> {
  if (!(eta > 0)) throw new MdmlError('value', 'hedgeRule: eta must be positive')
  return {
    name: 'hedge',
    config: { eta },
    init: (n) => initWeights(n),
    play: (s, round) => {
      const lo = Math.min(...s.cumulative)
      const p = normalise(Float64Array.from(s.cumulative, (c) => Math.exp(-eta * (c - lo))))
      return { probabilities: p, loss: ddot(p, round.losses) }
    },
    update: (s, round) => ({ ...s, cumulative: addLoss(s.cumulative, round.losses) }),
  }
}

/** Options of {@link weightedMajority}. */
export interface WeightedMajorityOptions {
  /** The penalty factor in (0, 1) applied to the weight of an expert that errs (default 1/2). */
  readonly beta?: number
  /** Predict by sampling an expert in proportion to its weight (randomised weighted majority) rather than by the weighted vote (default false). */
  readonly randomised?: boolean
}

/**
 * The weighted majority algorithm (Littlestone and Warmuth 1994) for binary prediction: each expert has a weight, starting at
 * one, multiplied by beta whenever it errs. The deterministic version predicts the weighted vote and makes at most
 * (ln N + m ln(1/beta)) / ln(2 / (1 + beta)) mistakes when the best of N experts makes m; the randomised version predicts
 * with an expert drawn in proportion to the weights and has expected mistakes at most (ln N + m ln(1/beta)) / (1 - beta).
 * Needs `predictions` and `outcomes` in the problem.
 *
 * @param options - beta and randomised.
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { weightedMajority } from 'mdml-engine/optim'
 * weightedMajority({ beta: 0.5 }).name // => 'weighted-majority'
 * ```
 */
export function weightedMajority(options: WeightedMajorityOptions = {}): ExpertsRule<WeightState> {
  const beta = options.beta ?? 0.5
  if (!(beta > 0 && beta < 1)) throw new MdmlError('value', 'weightedMajority: beta must be in (0, 1)')
  const randomised = options.randomised ?? false
  return {
    name: randomised ? 'randomised-weighted-majority' : 'weighted-majority',
    config: { beta, randomised },
    init: (n) => ({ cumulative: new Array<number>(n).fill(0), weights: new Array<number>(n).fill(1) }),
    play: (s, round) => {
      const p = normalise(Float64Array.from(s.weights))
      if (round.predictions === undefined || round.outcome === undefined)
        throw new MdmlError('value', 'weightedMajority: the problem needs predictions and outcomes')
      // The randomised rule draws an expert in proportion to the weights; the loss recorded is its expectation p . l, which
      // is what the bound is on and keeps the regret curve free of sampling noise.
      if (randomised) return { probabilities: p, loss: ddot(p, round.losses) }
      let vote = 0
      for (let i = 0; i < p.length; i++) vote += (p[i] ?? 0) * (round.predictions[i] ?? 0)
      const prediction = vote >= 0.5 ? 1 : 0
      return { probabilities: p, loss: prediction === round.outcome ? 0 : 1 }
    },
    update: (s, round) => ({
      cumulative: addLoss(s.cumulative, round.losses),
      weights: s.weights.map((w, i) => ((round.losses[i] ?? 0) > 0 ? w * beta : w)),
    }),
  }
}

/**
 * The mistake bound of deterministic weighted majority: if the best of N experts makes m mistakes, the algorithm makes at
 * most (ln N + m ln(1/beta)) / ln(2 / (1 + beta)).
 *
 * @param n - The number of experts.
 * @param m - The mistakes of the best expert.
 * @param beta - The penalty factor.
 * @returns The bound on the mistakes.
 * @concept online-learning
 * @example
 * ```ts
 * import { weightedMajorityMistakeBound } from 'mdml-engine/optim'
 * weightedMajorityMistakeBound(4, 10, 0.5) // => 28.9
 * ```
 */
export function weightedMajorityMistakeBound(n: number, m: number, beta: number): number {
  return (Math.log(n) + m * Math.log(1 / beta)) / Math.log(2 / (1 + beta))
}

/**
 * The expected-mistake bound of randomised weighted majority: (ln N + m ln(1/beta)) / (1 - beta).
 *
 * @param n - The number of experts.
 * @param m - The mistakes of the best expert.
 * @param beta - The penalty factor.
 * @returns The bound on the expected mistakes.
 * @concept online-learning
 * @example
 * ```ts
 * import { randomisedMajorityBound } from 'mdml-engine/optim'
 * randomisedMajorityBound(4, 10, 0.5) // => 16.6
 * ```
 */
export function randomisedMajorityBound(n: number, m: number, beta: number): number {
  return (Math.log(n) + m * Math.log(1 / beta)) / (1 - beta)
}

/** Options of {@link fixedShare} and {@link variableShare}. */
export interface ShareOptions {
  /** The learning rate of the exponential update. */
  readonly eta: number
  /** The share alpha in (0, 1): the fraction of weight spread over the other experts each round. */
  readonly alpha: number
}

/**
 * Fixed share (Herbster and Warmuth 1998), for tracking a best expert that changes over time: after the exponential
 * update the weights are mixed, w_i <- (1 - alpha) w_i + alpha (1 - w_i) / (N - 1), so no expert's weight falls so low that it
 * cannot recover when it becomes the best. Its regret against the best sequence with k switches is
 * O((k + 1) ln N + k ln T) / eta, where plain Hedge's grows linearly in T.
 *
 * @param options - eta and alpha.
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { fixedShare } from 'mdml-engine/optim'
 * fixedShare({ eta: 1, alpha: 0.01 }).name // => 'fixed-share'
 * ```
 */
export function fixedShare(options: ShareOptions): ExpertsRule<WeightState> {
  const { eta, alpha } = options
  if (!(eta > 0 && alpha > 0 && alpha < 1))
    throw new MdmlError('value', 'fixedShare: needs eta > 0 and alpha in (0, 1)')
  return {
    name: 'fixed-share',
    config: { eta, alpha },
    init: (n) => initWeights(n),
    play: (s, round) => {
      const p = Float64Array.from(s.weights)
      return { probabilities: p, loss: ddot(p, round.losses) }
    },
    update: (s, round) => {
      const n = s.weights.length
      const v = normalise(Float64Array.from(s.weights, (w, i) => w * Math.exp(-eta * (round.losses[i] ?? 0))))
      return {
        cumulative: addLoss(s.cumulative, round.losses),
        weights: Array.from(v, (x) => (1 - alpha) * x + (alpha * (1 - x)) / (n - 1)),
      }
    },
  }
}

/**
 * Variable share (Herbster and Warmuth 1998): like fixed share, but the weight an expert gives away depends on its loss,
 * v_i = w_i exp(-eta l_i), w_i <- (1 - alpha)^{l_i} v_i + (1 / (N - 1)) sum_{j != i} (1 - (1 - alpha)^{l_j}) v_j, so an expert
 * that did well shares almost nothing and one that did badly shares a lot. It tracks as well as fixed share when the best
 * expert's loss is small, with better constants when it is.
 *
 * @param options - eta and alpha.
 * @returns The rule.
 * @concept online-learning
 * @example
 * ```ts
 * import { variableShare } from 'mdml-engine/optim'
 * variableShare({ eta: 1, alpha: 0.05 }).name // => 'variable-share'
 * ```
 */
export function variableShare(options: ShareOptions): ExpertsRule<WeightState> {
  const { eta, alpha } = options
  if (!(eta > 0 && alpha > 0 && alpha < 1))
    throw new MdmlError('value', 'variableShare: needs eta > 0 and alpha in (0, 1)')
  return {
    name: 'variable-share',
    config: { eta, alpha },
    init: (n) => initWeights(n),
    play: (s, round) => {
      const p = Float64Array.from(s.weights)
      return { probabilities: p, loss: ddot(p, round.losses) }
    },
    update: (s, round) => {
      const n = s.weights.length
      const v = s.weights.map((w, i) => w * Math.exp(-eta * (round.losses[i] ?? 0)))
      const keep = v.map((x, i) => (1 - alpha) ** (round.losses[i] ?? 0) * x)
      const pool = v.map((x, i) => x - (keep[i] ?? 0))
      const total = pool.reduce((a, b) => a + b, 0)
      const next = normalise(Float64Array.from(keep, (k, i) => k + (total - (pool[i] ?? 0)) / (n - 1)))
      return { cumulative: addLoss(s.cumulative, round.losses), weights: Array.from(next) }
    },
  }
}

/**
 * The smallest total loss of any sequence of experts with at most `switches` changes, by dynamic programming over (round,
 * switches used, current expert). Used as the comparator for tracking.
 *
 * @param losses - T x N losses.
 * @param switches - The number of times the chosen expert may change.
 * @returns For each prefix length t = 1..T, the best loss of a sequence with at most `switches` switches over the first t rounds.
 * @concept online-learning
 * @example
 * ```ts
 * import { bestSwitchingLoss } from 'mdml-engine/optim'
 * bestSwitchingLoss([[0, 1], [0, 1], [1, 0], [1, 0]], 1) // => [0, 0, 0, 0]
 * ```
 */
export function bestSwitchingLoss(losses: readonly (readonly number[])[], switches: number): number[] {
  const n = losses[0]?.length ?? 0
  const out: number[] = []
  // dp[s][i]: best loss so far with s switches used, currently on expert i.
  let dp: number[][] = Array.from({ length: switches + 1 }, () => new Array<number>(n).fill(Number.POSITIVE_INFINITY))
  losses.forEach((row, t) => {
    const next = Array.from({ length: switches + 1 }, () => new Array<number>(n).fill(Number.POSITIVE_INFINITY))
    for (let s = 0; s <= switches; s++) {
      const bestPrev = s > 0 ? Math.min(...(dp[s - 1] as number[])) : Number.POSITIVE_INFINITY
      for (let i = 0; i < n; i++) {
        const stay = t === 0 ? 0 : ((dp[s] as number[])[i] ?? Number.POSITIVE_INFINITY)
        const move = t === 0 ? Number.POSITIVE_INFINITY : bestPrev
        ;(next[s] as number[])[i] = Math.min(stay, move) + (row[i] ?? 0)
      }
    }
    dp = next
    out.push(Math.min(...dp.map((r) => Math.min(...r))))
  })
  return out
}

/** State of {@link expertsAlgorithm}: the rule's state and the regret record. */
export interface ExpertsRunState<S> {
  readonly t: number
  readonly rule: S
  readonly probabilities: readonly number[]
  readonly loss: number
  readonly cumulativeLoss: number
  readonly expertLosses: readonly number[]
}

/** Options of {@link expertsAlgorithm}. */
export interface ExpertsRunOptions {
  /** Also track the regret against the best sequence of this many switches (default none). */
  readonly switches?: number
}

/**
 * A rule facing a problem, as a stepped algorithm, one round per step: the rule plays, suffers the loss, and updates.
 * Records the loss, the cumulative loss, the regret against the best single expert in hindsight (and, with `switches`,
 * against the best sequence with that many switches), and the distribution played.
 *
 * @param rule - The rule.
 * @param problem - The losses (and predictions and outcomes for weighted majority).
 * @param options - The switching comparator.
 * @returns The algorithm over {@link ExpertsRunState}.
 * @concept online-learning
 * @example
 * ```ts
 * import { expertsAlgorithm, hedgeRule } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = expertsAlgorithm(hedgeRule(1), { losses: [[0, 1], [0, 1]] })
 * runAlgorithm(alg, { maxSteps: 2 }).state.expertLosses // => [0, 2]
 * ```
 */
export function expertsAlgorithm<S>(
  rule: ExpertsRule<S>,
  problem: ExpertsProblem,
  options: ExpertsRunOptions = {},
): Algorithm<ExpertsRunState<S>> {
  const rounds = problem.losses.length
  const n = problem.losses[0]?.length ?? 0
  if (rounds === 0 || n < 2) throw new MdmlError('value', 'expertsAlgorithm: needs at least one round and two experts')
  const comparator = options.switches === undefined ? undefined : bestSwitchingLoss(problem.losses, options.switches)
  return {
    name: `experts(${rule.name})`,
    config: {
      rule: rule.name,
      ...(rule.config as Record<string, Json>),
      rounds,
      experts: n,
      switches: options.switches ?? null,
    } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'the learner loss in the round', direction: 'lower' },
        cumulativeLoss: { description: 'the learner loss so far', direction: 'lower' },
        bestExpertLoss: { description: 'the loss of the best single expert so far', direction: 'lower' },
        regret: { description: 'cumulative loss minus the best single expert in hindsight', direction: 'lower' },
        ...(comparator === undefined
          ? {}
          : {
              switchingRegret: {
                description: 'cumulative loss minus the best sequence with k switches',
                direction: 'lower' as const,
              },
            }),
      },
      tensors: { probabilities: { role: 'state', description: 'the distribution over experts played in the round' } },
    }),
    init: (key) => ({
      t: 0,
      rule: rule.init(n, key),
      probabilities: new Array<number>(n).fill(1 / n),
      loss: 0,
      cumulativeLoss: 0,
      expertLosses: new Array<number>(n).fill(0),
    }),
    step: (s, ctx) => {
      const losses = problem.losses[s.t] as readonly number[]
      const play = rule.play(
        s.rule,
        {
          t: s.t + 1,
          losses,
          ...(problem.predictions === undefined ? {} : { predictions: problem.predictions[s.t] as readonly number[] }),
          ...(problem.outcomes === undefined ? {} : { outcome: problem.outcomes[s.t] as number }),
        },
        foldIn(ctx.key, 0),
      )
      return {
        t: s.t + 1,
        rule: rule.update(s.rule, { t: s.t + 1, losses }, foldIn(ctx.key, 1)),
        probabilities: Array.from(play.probabilities),
        loss: play.loss,
        cumulativeLoss: s.cumulativeLoss + play.loss,
        expertLosses: s.expertLosses.map((l, i) => l + (losses[i] ?? 0)),
      }
    },
    done: (s) => (s.t >= rounds ? 'completed' : undefined),
    observe: (s) => {
      const best = Math.min(...s.expertLosses)
      return {
        metrics: {
          loss: s.loss,
          cumulativeLoss: s.cumulativeLoss,
          bestExpertLoss: best,
          regret: s.cumulativeLoss - best,
          ...(comparator === undefined ? {} : { switchingRegret: s.cumulativeLoss - (comparator[s.t - 1] ?? 0) }),
        },
        tensors: { probabilities: { role: 'state', value: s.probabilities } },
      }
    },
    serialise: (s) => JSON.parse(JSON.stringify(s)) as Json,
    deserialise: (j) => j as unknown as ExpertsRunState<S>,
  }
}

/** The record of a finished experts run. */
export interface ExpertsResult {
  readonly cumulativeLoss: number
  readonly bestExpertLoss: number
  /** Cumulative loss minus the best expert's, at the end. */
  readonly regret: number
  /** Regret after each round. */
  readonly regretCurve: readonly number[]
  readonly expertLosses: readonly number[]
  readonly finalProbabilities: readonly number[]
}

/**
 * Run a rule over a problem and return its regret record ({@link expertsAlgorithm} run to the end, one key per run).
 *
 * @param rule - The rule.
 * @param problem - The problem.
 * @param key - The key for randomised rules (default `key(0)`).
 * @returns Loss, regret and the regret after each round.
 * @concept online-learning
 * @example
 * ```ts
 * import { followTheLeader, runExperts } from 'mdml-engine/optim'
 * runExperts(followTheLeader(), { losses: [[0.5, 0], [0, 1], [1, 0], [0, 1]] }).regret // => 2
 * ```
 */
export function runExperts<S>(rule: ExpertsRule<S>, problem: ExpertsProblem, key: Key = makeKey(0)): ExpertsResult {
  const curve: number[] = []
  const alg = expertsAlgorithm(rule, problem)
  const recording: Algorithm<ExpertsRunState<S>> = {
    ...alg,
    step: (s, ctx) => {
      const next = alg.step(s, ctx)
      curve.push(next.cumulativeLoss - Math.min(...next.expertLosses))
      return next
    },
  }
  const r = runAlgorithm(recording, { maxSteps: problem.losses.length, key })
  const best = Math.min(...r.state.expertLosses)
  return {
    cumulativeLoss: r.state.cumulativeLoss,
    bestExpertLoss: best,
    regret: r.state.cumulativeLoss - best,
    regretCurve: curve,
    expertLosses: r.state.expertLosses,
    finalProbabilities: r.state.probabilities,
  }
}

/** The state of {@link exp4Algorithm}. */
export interface Exp4State {
  readonly t: number
  /** Cumulative importance-weighted loss estimate of each expert. */
  readonly estimated: readonly number[]
  /** The probability of each arm in the latest round. */
  readonly armProbabilities: readonly number[]
  /** The arm pulled in the latest round (-1 before the first). */
  readonly arm: number
  /** The sum over rounds of the expected loss sum_k p_k l_k of the learner. */
  readonly expectedLoss: number
  /** The realised loss. */
  readonly realisedLoss: number
}

/** Options of {@link exp4Algorithm}. */
export interface Exp4Options {
  /** The learning rate (default sqrt(2 ln N / (K T)), the value giving regret sqrt(2 K T ln N)). */
  readonly eta?: number
  /** Uniform exploration mixed into the arm distribution (default 0). */
  readonly gamma?: number
}

/**
 * Exp4 (Auer, Cesa-Bianchi, Freund and Schapire 2002) for K arms with N experts under bandit feedback: each round every
 * expert gives a distribution over arms, the learner plays arm k with probability p_k = (1 - gamma) sum_i q_i xi_i(k) + gamma / K
 * for expert weights q, sees only the loss of the arm it pulled, and updates each expert's cumulative loss by the importance
 * weighted estimate xi_i(a) l_a / p_a. The pseudo-regret against the best expert is at most sqrt(2 K T ln N) at
 * eta = sqrt(2 ln N / (K T)) (gamma = 0, losses in [0, 1]). Exp3 is the case where expert i always recommends arm i
 * (`advice` the identity). The tape records the expected loss and the pseudo-regret.
 *
 * @param advice - Either `[N][K]` advice that is the same every round, or `[T][N][K]`.
 * @param losses - The T x K loss of every arm in every round (the learner sees only the pulled arm's).
 * @param options - eta and gamma.
 * @returns The algorithm over {@link Exp4State}.
 * @concept online-learning
 * @example
 * ```ts
 * import { exp4Algorithm } from 'mdml-engine/optim'
 * exp4Algorithm([[1, 0], [0, 1]], [[0, 1], [0, 1]]).name // => 'exp4'
 * ```
 */
export function exp4Algorithm(
  advice: readonly (readonly number[])[] | readonly (readonly (readonly number[])[])[],
  losses: readonly (readonly number[])[],
  options: Exp4Options = {},
): Algorithm<Exp4State> {
  const rounds = losses.length
  const k = losses[0]?.length ?? 0
  const perRound = Array.isArray((advice[0] as unknown[] | undefined)?.[0])
  const adviceAt = (t: number): readonly (readonly number[])[] =>
    (perRound
      ? (advice as readonly (readonly (readonly number[])[])[])[t]
      : (advice as readonly (readonly number[])[])) as readonly (readonly number[])[]
  const n = adviceAt(0).length
  if (rounds === 0 || k < 2 || n < 1)
    throw new MdmlError('value', 'exp4: needs rounds, at least two arms and an expert')
  const eta = options.eta ?? Math.sqrt((2 * Math.log(Math.max(n, 2))) / (k * rounds))
  const gamma = options.gamma ?? 0
  return {
    name: 'exp4',
    config: { arms: k, experts: n, rounds, eta, gamma } as Json,
    schema: channels({
      metrics: {
        expectedLoss: { description: 'expected loss of the learner in the round, sum_k p_k l_k', direction: 'lower' },
        pseudoRegret: {
          description: "cumulative expected loss minus the best expert's cumulative expected loss",
          direction: 'lower',
        },
      },
      tensors: { armProbabilities: { role: 'state', description: 'the arm distribution of the round' } },
    }),
    init: () => ({
      t: 0,
      estimated: new Array<number>(n).fill(0),
      armProbabilities: new Array<number>(k).fill(1 / k),
      arm: -1,
      expectedLoss: 0,
      realisedLoss: 0,
    }),
    step: (s, ctx) => {
      const a = adviceAt(s.t)
      const lo = Math.min(...s.estimated)
      const q = normalise(Float64Array.from(s.estimated, (e) => Math.exp(-eta * (e - lo))))
      const p = new Float64Array(k)
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) p[j] = (p[j] ?? 0) + (q[i] ?? 0) * (a[i]?.[j] ?? 0)
      for (let j = 0; j < k; j++) p[j] = (1 - gamma) * (p[j] ?? 0) + gamma / k
      const arm = Number(toFlat(categorical(ctx.key, Array.from(p), [1]))[0] ?? 0)
      const row = losses[s.t] as readonly number[]
      const lossArm = row[arm] ?? 0
      const estimated = s.estimated.map((e, i) => e + ((a[i]?.[arm] ?? 0) * lossArm) / (p[arm] ?? 1))
      return {
        t: s.t + 1,
        estimated,
        armProbabilities: Array.from(p),
        arm,
        expectedLoss: s.expectedLoss + ddot(p, row),
        realisedLoss: s.realisedLoss + lossArm,
      }
    },
    done: (s) => (s.t >= rounds ? 'completed' : undefined),
    observe: (s) => {
      // The best expert's cumulative expected loss: sum_t sum_k xi_i(k) l_k(t).
      let best = Number.POSITIVE_INFINITY
      for (let i = 0; i < n; i++) {
        let total = 0
        for (let t = 0; t < s.t; t++) total += ddot(adviceAt(t)[i] ?? [], losses[t] ?? [])
        best = Math.min(best, total)
      }
      return {
        metrics: { expectedLoss: s.expectedLoss, pseudoRegret: s.expectedLoss - best },
        tensors: { armProbabilities: { role: 'state', value: s.armProbabilities } },
      }
    },
    serialise: (s) => JSON.parse(JSON.stringify(s)) as Json,
    deserialise: (j) => j as unknown as Exp4State,
  }
}

/**
 * The pseudo-regret bound of Exp4: at learning rate eta and no exploration, E[regret] <= ln N / eta + eta K T / 2 (losses
 * in [0, 1]), which is sqrt(2 K T ln N) at eta = sqrt(2 ln N / (K T)).
 *
 * @param T - The number of rounds.
 * @param K - The number of arms.
 * @param N - The number of experts.
 * @param eta - The learning rate (default the optimal one).
 * @returns The bound.
 * @concept online-learning
 * @example
 * ```ts
 * import { exp4RegretBound } from 'mdml-engine/optim'
 * exp4RegretBound(1000, 4, 10) // => 135.7228
 * ```
 */
export function exp4RegretBound(T: number, K: number, N: number, eta?: number): number {
  const e = eta ?? Math.sqrt((2 * Math.log(N)) / (K * T))
  return Math.log(N) / e + (e * K * T) / 2
}

/** Options of {@link pNormMirrorDescent}. */
export interface PNormOptions {
  /** The norm q > 1 of the potential psi(x) = |x|_q^2 / 2 (default 2: online gradient descent; 2 ln d for sparse targets). */
  readonly q?: number
  /** The step size (a constant or a function of the round from 1; default 1 / sqrt(t)). */
  readonly learningRate?: number | ((t: number) => number)
}

/**
 * Online mirror descent with the p-norm potential (Gentile 2003; Shalev-Shwartz 2012, section 2.6): the dual iterate is the
 * negative sum of the gradients, theta_t = -eta sum g, and the played point is the gradient of the conjugate potential,
 * x_i = sign(theta_i) |theta_i|^{p-1} / |theta|_p^{p-2} for 1/p + 1/q = 1. With q = 2 it is online gradient descent; with q near
 * 2 ln d and sparse comparators in the l1 ball it has regret of order sqrt(T ln d) as the entropic mirror does, but needs no
 * simplex. Unconstrained.
 *
 * @param options - q and the step size.
 * @returns A learner for `onlineAlgorithm` / `runOnline`.
 * @concept online-learning
 * @example
 * ```ts
 * import { pNormMirrorDescent } from 'mdml-engine/optim'
 * const l = pNormMirrorDescent({ q: 2, learningRate: 0.5 })
 * Array.from(l.play(l.update(l.init(2), new Float64Array([2, -4]), 1))) // => [-1, 2]
 * ```
 */
export function pNormMirrorDescent(
  options: PNormOptions = {},
): OnlineLearner<{ readonly theta: Float64Array; readonly x: Float64Array }> {
  const q = options.q ?? 2
  if (!(q > 1)) throw new MdmlError('value', 'pNormMirrorDescent: q must exceed 1')
  const p = q / (q - 1)
  const lr = options.learningRate ?? ((t: number) => 1 / Math.sqrt(t))
  const dual = (theta: Float64Array): Float64Array => {
    let norm = 0
    for (const v of theta) norm += Math.abs(v) ** p
    norm = norm ** (1 / p)
    if (norm === 0) return new Float64Array(theta.length)
    return theta.map((v) => (Math.sign(v) * Math.abs(v) ** (p - 1)) / norm ** (p - 2))
  }
  return {
    name: 'pNormMirrorDescent',
    init: (dim) => ({ theta: new Float64Array(dim), x: new Float64Array(dim) }),
    play: (s) => s.x,
    update: (s, g, t) => {
      const eta = typeof lr === 'number' ? lr : lr(t)
      const theta = Float64Array.from(s.theta, (v, i) => v - eta * (g[i] ?? 0))
      return { theta, x: dual(theta) }
    },
  }
}

/** Options of {@link onlineAdagrad}. */
export interface OnlineAdagradOptions {
  /** The base step eta (default radius / sqrt(2) for the box of half-width radius, the value of the headline bound). */
  readonly eta?: number
  /** The box [-radius, radius]^d the plays are projected onto (default 1). */
  readonly radius?: number
  /** Added to the accumulated squared gradients to keep the first steps finite (default 1e-12). */
  readonly epsilon?: number
}

/**
 * AdaGrad (Duchi, Hazan and Singer 2011), diagonal, for online convex optimisation over a box: x_{t+1} = clip(x_t - eta g_t /
 * sqrt(sum_{s <= t} g_s^2)), a step per coordinate that shrinks with the gradients that coordinate has seen, so rare
 * coordinates move fast. Projection in the diagonal metric onto a box is clipping. Its regret against the best point of the
 * box is at most sqrt(2) D_inf sum_j |g_{1:T,j}|_2 at eta = D_inf / sqrt(2), D_inf = 2 radius the width of the box (see
 * {@link adagradRegretBound}).
 *
 * @param options - The step, box and epsilon.
 * @returns A learner for `onlineAlgorithm` / `runOnline`.
 * @concept online-learning
 * @example
 * ```ts
 * import { onlineAdagrad } from 'mdml-engine/optim'
 * const l = onlineAdagrad({ eta: 0.5, radius: 10 })
 * Array.from(l.play(l.update(l.init(1), new Float64Array([3]), 1))) // => [-0.5]
 * ```
 */
export function onlineAdagrad(
  options: OnlineAdagradOptions = {},
): OnlineLearner<{ readonly x: Float64Array; readonly sumSq: Float64Array }> {
  const radius = options.radius ?? 1
  const eta = options.eta ?? (2 * radius) / Math.SQRT2
  const eps = options.epsilon ?? 1e-12
  return {
    name: 'onlineAdagrad',
    init: (dim) => ({ x: new Float64Array(dim), sumSq: new Float64Array(dim) }),
    play: (s) => s.x,
    update: (s, g) => {
      const sumSq = Float64Array.from(s.sumSq, (v, i) => v + (g[i] ?? 0) ** 2)
      const x = Float64Array.from(s.x, (v, i) => {
        const step = (eta * (g[i] ?? 0)) / Math.sqrt((sumSq[i] ?? 0) + eps)
        return Math.max(-radius, Math.min(radius, v - step))
      })
      return { x, sumSq }
    },
  }
}

/**
 * AdaGrad's regret bound for a box of width D_inf (twice the radius) at eta = D_inf / sqrt(2): sqrt(2) D_inf sum_j |g_{1:T,j}|_2,
 * the sum over coordinates of the Euclidean norm of that coordinate's gradients over the rounds. It is data dependent and
 * much smaller than the worst case D G sqrt(T) when the gradients are sparse.
 *
 * @param gradients - The T gradients, each of dimension d.
 * @param width - D_inf, the width of the box in each coordinate (2 radius).
 * @returns The bound.
 * @concept online-learning
 * @example
 * ```ts
 * import { adagradRegretBound } from 'mdml-engine/optim'
 * adagradRegretBound([[3, 0], [4, 0]], 2) // => 14.1421
 * ```
 */
export function adagradRegretBound(gradients: readonly (readonly number[])[], width: number): number {
  const d = gradients[0]?.length ?? 0
  let total = 0
  for (let j = 0; j < d; j++) {
    let ss = 0
    for (const g of gradients) ss += (g[j] ?? 0) ** 2
    total += Math.sqrt(ss)
  }
  return Math.SQRT2 * width * total
}

/** Options of {@link hindsightComparator}. */
export interface HindsightOptions {
  /** Constrain the point to the box [-box, box]^d (default unconstrained). */
  readonly box?: number
}

/**
 * The best fixed point in hindsight, argmin_u sum_t f_t(u), for smooth convex losses, by L-BFGS(-B) on the summed loss
 * (gradients by automatic differentiation): the comparator that an online learner's regret is measured against when it is
 * not available in closed form. For linear losses over a ball or box the comparator is explicit and cheaper.
 *
 * @param losses - The T losses, each a function of the point.
 * @param dim - The dimension.
 * @param options - An optional box.
 * @returns The minimiser and the total loss there.
 * @concept online-learning
 * @example
 * ```ts
 * import { hindsightComparator } from 'mdml-engine/optim'
 * import { square, sub, sum, toArray } from 'mdml-engine/tensor'
 * const f = (c: number) => (x: any) => sum(square(sub(x, c)))
 * Math.round(hindsightComparator([f(1), f(3)], 1).point[0] * 1e6) / 1e6 // => 2
 * ```
 */
export function hindsightComparator(
  losses: readonly OnlineLoss[],
  dim: number,
  options: HindsightOptions = {},
): { readonly point: Float64Array; readonly loss: number } {
  const total = (x: Value): Value => losses.map((f) => f(x)).reduce((a, b) => add(a, b))
  const x0 = new Float64Array(dim)
  const bounds = options.box === undefined ? {} : { bounds: { lower: -options.box, upper: options.box } }
  const r = lbfgsb(total, x0, { ...bounds, gradTol: 1e-10, ftol: 1e-15 })
  return { point: Float64Array.from(toFlat(r.x as Value)), loss: r.fun }
}
