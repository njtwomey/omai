/**
 * Fairness during training by reduction, ports of fairlearn's `reductions`: fair classification becomes a sequence of
 * cost-sensitive problems under a demographic-parity or equalised-odds constraint. Exponentiated gradient (a game
 * between a learner and a constraint player, giving a randomised classifier; golden against fairlearn's final mixture)
 * and the grid search over multipliers that traces the accuracy-fairness trade-off.
 */

import { MdmlError } from '../base/index.ts'
import { type Estimator, type Fitted, requireVerb } from '../learn/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { linprog } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { asTensor, fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/**
 * The moment a reduction constrains: demographic parity, equalised odds, or one half of it, true-positive rate parity
 * (equal opportunity) or false-positive rate parity.
 */
export type ReductionConstraint =
  | 'demographicParity'
  | 'equalisedOdds'
  | 'truePositiveRateParity'
  | 'falsePositiveRateParity'

/**
 * fairlearn's utility-parity moments, laid out as fairlearn lays them out: constraints indexed by sign (+ then -), then
 * event (`all`, or each label for equalised odds), then group; `gamma` of decisions is each group's mean decision in
 * an event minus the event's; `signedWeights` of multipliers is each row's `U . lambda`.
 */
interface Moments {
  /** Constraints per sign: events x groups. */
  readonly half: number
  readonly k: number
  readonly events: number
  readonly n: number
  gamma(h: readonly number[]): number[]
  signedWeights(lambda: readonly number[]): number[]
  project(lambda: readonly number[]): number[]
  /** The non-reference basis of the grid search: for each event, every group but the last to appear. */
  readonly basis: readonly number[]
}

function moments(constraint: ReductionConstraint, groups: readonly number[], labels: readonly number[]): Moments {
  const n = groups.length
  const k = Math.max(...groups) + 1
  const events = constraint === 'equalisedOdds' ? 2 : 1
  // Each row's event (-1 for rows outside every event: the negatives for TPR parity, the positives for FPR parity).
  const eventOf = (i: number): number => {
    const y = labels[i] === 1 ? 1 : 0
    if (constraint === 'demographicParity') return 0
    if (constraint === 'equalisedOdds') return y
    if (constraint === 'truePositiveRateParity') return y === 1 ? 0 : -1
    return y === 0 ? 0 : -1
  }
  const half = events * k
  const cell = (e: number, g: number): number => e * k + g
  const pEvent = new Array<number>(events).fill(0)
  const pCell = new Array<number>(half).fill(0)
  for (let i = 0; i < n; i++) {
    const e = eventOf(i)
    if (e < 0) continue
    pEvent[e] = (pEvent[e] ?? 0) + 1 / n
    pCell[cell(e, groups[i] ?? 0)] = (pCell[cell(e, groups[i] ?? 0)] ?? 0) + 1 / n
  }
  // Groups in order of appearance (fairlearn's `unique()`); the last is left out of the grid's basis.
  const seen: number[] = []
  for (const g of groups) if (!seen.includes(g)) seen.push(g)
  const basis: number[] = []
  for (let e = 0; e < events; e++) for (const g of seen.slice(0, -1)) basis.push(cell(e, g))
  return {
    half,
    k,
    events,
    n,
    basis,
    gamma: (h) => {
      const sumEvent = new Array<number>(events).fill(0)
      const sumCell = new Array<number>(half).fill(0)
      for (let i = 0; i < n; i++) {
        const e = eventOf(i)
        if (e < 0) continue
        sumEvent[e] = (sumEvent[e] ?? 0) + (h[i] ?? 0)
        sumCell[cell(e, groups[i] ?? 0)] = (sumCell[cell(e, groups[i] ?? 0)] ?? 0) + (h[i] ?? 0)
      }
      const plus = sumCell.map((s, c) => {
        const e = Math.floor(c / k)
        const pc = pCell[c] ?? 0
        if (pc === 0) return 0
        return s / n / pc - (sumEvent[e] ?? 0) / n / (pEvent[e] ?? 1)
      })
      return [...plus, ...plus.map((v) => -v)]
    },
    signedWeights: (lambda) =>
      Array.from({ length: n }, (_, i) => {
        const e = eventOf(i)
        if (e < 0) return 0
        const own = cell(e, groups[i] ?? 0)
        let s = 0
        for (let g = 0; g < k; g++) {
          const c = cell(e, g)
          const pc = pCell[c] ?? 0
          if (pc === 0) continue
          const net = (lambda[c] ?? 0) - (lambda[c + half] ?? 0)
          // U[+, e, g] = 1{e} / P(e) - 1{e, g} / P(e, g); U[-] its negative.
          s += net * (1 / (pEvent[e] ?? 1) - (c === own ? 1 / pc : 0))
        }
        return s
      }),
    project: (lambda) => {
      const signed = Array.from({ length: half }, (_, c) => (lambda[c] ?? 0) - (lambda[c + half] ?? 0))
      for (let e = 0; e < events; e++) {
        const cells = Array.from({ length: k }, (_, g) => cell(e, g)).filter((c) => (pCell[c] ?? 0) > 0)
        const prob = cells.map((c) => (pCell[c] ?? 0) / (pEvent[e] ?? 1))
        const normalised = cells.map((c, j) => (signed[c] ?? 0) / (prob[j] ?? 1))
        const order = normalised.map((_, j) => j).sort((a, b) => (normalised[a] ?? 0) - (normalised[b] ?? 0) || a - b)
        const half_ = prob.reduce((s, v) => s + v, 0) / 2
        let acc = 0
        let pos = 0
        // numpy searchsorted (left) of half the mass in the cumulative probabilities.
        for (; pos < order.length; pos++) {
          acc += prob[order[pos] ?? 0] ?? 0
          if (acc >= half_) break
        }
        const shift = normalised[order[Math.min(pos, order.length - 1)] ?? 0] ?? 0
        cells.forEach((c, j) => {
          signed[c] = (signed[c] ?? 0) - (prob[j] ?? 0) * shift
        })
      }
      return [...signed.map((v) => Math.max(0, v)), ...signed.map((v) => Math.max(0, -v))]
    },
  }
}

/** The constant classifier fairlearn substitutes (a DummyClassifier) when the relabelled target has one class. */
function constantFitted(value: number): Fitted {
  return {
    estimator: 'constant',
    predict: (x: TensorLike) => {
      const rows = matrixRows(x)
      return fromStorage(new Float64Array(rows).fill(value), [rows], 'float64')
    },
  } as unknown as Fitted
}

function matrixRows(x: TensorLike): number {
  const t = asTensor(x)
  return t.shape[0] ?? 0
}

/** The cost-sensitive oracle: relabel rows by the sign of their signed weight, weight them by its size, and fit. */
function callOracle(
  estimator: Estimator,
  x: TensorLike,
  labels: readonly number[],
  signed: readonly number[],
  normalise: boolean,
  key: Key,
): Fitted {
  const n = labels.length
  const target = signed.map((w) => (w > 0 ? 1 : 0))
  const total = signed.reduce((s, w) => s + Math.abs(w), 0)
  const weights = signed.map((w) => (normalise ? (n * Math.abs(w)) / total : Math.abs(w)))
  if (target.every((t) => t === target[0])) return constantFitted(target[0] ?? 0)
  return estimator.fit({ x, y: target, weights }, { key, explain: false })
}

const errorOf = (labels: readonly number[], h: readonly number[]): number =>
  labels.reduce((s, y, i) => s + Math.abs(y - (h[i] ?? 0)), 0) / Math.max(1, labels.length)

/** The rows a reduction learns from: features, binary labels and each row's group index. */
export interface ReductionData {
  readonly x: TensorLike
  readonly labels: readonly number[]
  readonly groups: readonly number[]
}

/** Options of {@link exponentiatedGradientReduction}. */
export interface ExponentiatedGradientReductionOptions {
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** fairlearn's eps (default 0.01): the multipliers' bound is 1 / epsilon. */
  readonly epsilon?: number
  /** The allowed violation of each moment (default 0.01, the constraint's difference_bound in fairlearn). */
  readonly differenceBound?: number
  /** The iteration cap (default 50, fairlearn's max_iter). */
  readonly iterations?: number
  /** The step size times the bound (default 2, fairlearn's `eta0`). */
  readonly eta?: number
  /** The duality-gap tolerance (default fairlearn's: half the spread of the first classifier's errors over sqrt(n)). */
  readonly nu?: number
  readonly key?: Key
}

/** One iteration of {@link exponentiatedGradientReduction}. */
export interface ReductionStep {
  readonly iteration: number
  /** The error and violation of this iteration's best response. */
  readonly error: number
  readonly violation: number
  /** The error and violation of this iteration's mixture (the randomised classifier). */
  readonly averageError: number
  readonly averageViolation: number
  /** The duality gap of this iteration's mixture. */
  readonly gap: number
  /** The largest net price on a constraint, how hard the constraints push. */
  readonly lambdaTotal: number
}

/** The result of {@link exponentiatedGradientReduction}. */
export interface ExponentiatedGradientReduction {
  /** The best responses; the randomised classifier picks member k with probability `weights[k]`. */
  readonly members: readonly Fitted[]
  readonly weights: readonly number[]
  /** P(positive decision) for each row under the randomised classifier. */
  readonly decisions: readonly number[]
  readonly steps: readonly ReductionStep[]
  /** The iteration whose mixture was kept (the last with the smallest gap) and its gap. */
  readonly bestIteration: number
  readonly bestGap: number
  /** The unconstrained classifier's positive-class probabilities, error and violation, for comparison. */
  readonly baseline: { readonly scores: readonly number[]; readonly error: number; readonly violation: number }
}

// fairlearn's constants (`_exponentiated_gradient/_constants.py`).
const PRECISION = 1e-8
const ACCURACY_MUL = 0.5
const REGRET_CHECK_START_T = 5
const REGRET_CHECK_INCREASE_T = 1.6
const SHRINK_REGRET = 0.8
const SHRINK_ETA = 0.8
const MIN_ITER = 5

/** The paths {@link exponentiatedGradientAlgorithm} logs. */
export type ExponentiatedGradientPath =
  | 'eg/classifier'
  | 'eg/error'
  | 'eg/violation'
  | 'eg/lambda'
  | 'eg/gap'
  | 'eg/mixture'

/** What a stepped exponentiated-gradient reduction logs, one iteration of the game per step. */
export const exponentiatedGradientContract: Contract<unknown, ExponentiatedGradientPath> = defineContract({
  'eg/classifier': scalar({
    description: 'the index of the best response used this iteration (a new one when the learner found a better)',
  }),
  'eg/error': scalar({
    description: 'the error of the mixture (the randomised classifier) this iteration',
    direction: 'lower',
  }),
  'eg/violation': scalar({
    description: 'the largest constraint violation of the mixture this iteration',
    direction: 'lower',
  }),
  'eg/lambda': scalar({ description: 'the largest net price on a constraint' }),
  'eg/gap': scalar({ description: 'the duality gap of the mixture this iteration', direction: 'lower' }),
  'eg/mixture': scalar({ description: 'classifiers kept so far (the mixture draws from these)' }),
})

/** A best response kept by the game: its decisions on the training rows, error and moments (the classifier is held apart). */
export interface ReductionHypothesis {
  readonly decisions: readonly number[]
  readonly error: number
  readonly gamma: readonly number[]
}

/** The linear-programme mixture of the classifiers so far, valid while their number is `n`. */
export interface ReductionMixture {
  readonly n: number
  readonly q: readonly number[]
  readonly lambda: readonly number[]
  readonly gap: number
}

/**
 * The state of {@link exponentiatedGradientAlgorithm}: the iteration, the constraint prices' logits `theta`, the best
 * responses kept, how often each was chosen, the running sums the averages need, each iteration's gap and mixture, the
 * steps recorded, and the tolerance and step size once set. The classifiers themselves are not in the state (they
 * are not JSON); {@link exponentiatedGradientReduction} keeps them.
 */
export interface ExponentiatedGradientState {
  readonly t: number
  readonly theta: readonly number[]
  readonly hs: readonly ReductionHypothesis[]
  readonly calls: number
  /** `[classifier, times chosen]`. */
  readonly qsum: readonly (readonly [number, number])[]
  readonly lambdaSum: readonly number[]
  readonly gapsEG: readonly number[]
  readonly gaps: readonly number[]
  readonly qs: readonly (readonly number[])[]
  readonly steps: readonly ReductionStep[]
  readonly nu: number | null
  readonly eta: number
  readonly lastRegretChecked: number
  /** The best averaged gap at the last regret check (null: infinity). */
  readonly lastGap: number | null
  readonly mixture: ReductionMixture | null
  readonly stopped: boolean
}

interface ExponentiatedGradientRun {
  readonly algorithm: Algorithm<ExponentiatedGradientState, ExponentiatedGradientPath>
  /** The classifiers fitted so far, aligned with the state's `hs` (held here because they are not JSON). */
  readonly fits: Fitted[]
  readonly maxIter: number
  readonly finish: (s: ExponentiatedGradientState) => ExponentiatedGradientReduction
}

function exponentiatedGradientRun(
  estimator: Estimator,
  data: ReductionData,
  options: ExponentiatedGradientReductionOptions,
): ExponentiatedGradientRun {
  const { x, labels, groups } = data
  const constraint = options.constraint ?? 'demographicParity'
  const epsilon = options.epsilon ?? 0.01
  if (!(epsilon > 0)) throw new MdmlError('value', 'exponentiatedGradientReduction: epsilon must be positive')
  const maxIter = options.iterations ?? 50
  const B = 1 / epsilon
  const bound = options.differenceBound ?? 0.01
  const key = options.key ?? makeKey(0)
  const m = moments(constraint, groups, labels)
  const n = labels.length
  const objectiveWeights = labels.map((y) => 2 * (y === 1 ? 1 : 0) - 1)
  const fits: Fitted[] = []
  const predict = (f: Fitted): number[] =>
    Array.from(toFlat(requireVerb(f, 'predict', 'exponentiatedGradientReduction')(x)))
  const empty: ExponentiatedGradientState = {
    t: 0,
    theta: new Array<number>(2 * m.half).fill(0),
    hs: [],
    calls: 0,
    qsum: [],
    lambdaSum: new Array<number>(2 * m.half).fill(0),
    gapsEG: [],
    gaps: [],
    qs: [],
    steps: [],
    nu: options.nu ?? null,
    eta: 0,
    lastRegretChecked: REGRET_CHECK_START_T,
    lastGap: null,
    mixture: null,
    stopped: false,
  }
  const algorithm: Algorithm<ExponentiatedGradientState, ExponentiatedGradientPath> = {
    name: 'exponentiated-gradient-reduction',
    config: { constraint, epsilon, iterations: maxIter, rows: n },
    schema: channels({ metrics: { gap: { description: 'the duality gap of the mixture', direction: 'lower' } } }),
    init: () => empty,
    step: (s, ctx) => {
      const t = s.t
      // Everything below works on copies of the state's arrays and is returned as the next state.
      const hs: ReductionHypothesis[] = [...s.hs]
      let calls = s.calls
      let mixture = s.mixture
      const bestH = (lambda: readonly number[]): number => {
        const sw = m.signedWeights(lambda)
        const fitted = callOracle(
          estimator,
          x,
          labels,
          objectiveWeights.map((w, i) => w + (sw[i] ?? 0)),
          true,
          foldIn(key, ++calls),
        )
        const decisions = predict(fitted)
        const error = errorOf(labels, decisions)
        const gamma = m.gamma(decisions)
        const value = error + ddot(gamma, lambda)
        let best = -1
        let bestValue = Number.POSITIVE_INFINITY
        for (const [j, h] of hs.entries()) {
          const v = h.error + ddot(h.gamma, lambda)
          if (v < bestValue) {
            bestValue = v
            best = j
          }
        }
        if (value < bestValue - PRECISION) {
          hs.push({ decisions, error, gamma })
          fits[hs.length - 1] = fitted
          return hs.length - 1
        }
        return best
      }
      // The Lagrangian of a mixture (weights over hs) at multipliers: (L, L_high, gamma, error).
      const evaluate = (q: readonly number[], lambda: readonly number[]) => {
        const error = hs.reduce((a, h, j) => a + h.error * (q[j] ?? 0), 0)
        const gamma = Array.from({ length: 2 * m.half }, (_, c) =>
          hs.reduce((a, h, j) => a + (h.gamma[c] ?? 0) * (q[j] ?? 0), 0),
        )
        const projected = m.project(lambda)
        const L = error + projected.reduce((a, l, c) => a + l * ((gamma[c] ?? 0) - bound), 0)
        const maxConstraint = Math.max(...gamma.map((g) => g - bound))
        return { L, high: error + (maxConstraint > 0 ? B * maxConstraint : 0), gamma, error }
      }
      const evalGap = (q: readonly number[], lambda: readonly number[], nuValue: number) => {
        const e = evaluate(q, lambda)
        let low = e.L
        const gap = (): number => Math.max(e.L - low, e.high - e.L)
        for (const mul of [1, 2, 5, 10]) {
          const idx = bestH(lambda.map((l) => mul * l))
          const single = hs.map((_, j) => (j === idx ? 1 : 0))
          const lm = evaluate(single, lambda).L
          if (lm < low) low = lm
          if (gap() > nuValue + PRECISION) break
        }
        return { ...e, gap: gap() }
      }
      const solveLinprog = (nuValue: number): ReductionMixture => {
        if (mixture !== null && mixture.n === hs.length) return mixture
        const k = hs.length
        const c = [...hs.map((h) => h.error), B]
        const aUb = Array.from({ length: 2 * m.half }, (_, j) => [...hs.map((h) => (h.gamma[j] ?? 0) - bound), -1])
        const primal = linprog({ c, aUb, bUb: aUb.map(() => 0), aEq: [[...hs.map(() => 1), 0]], bEq: [1] })
        const q = Array.from(toFlat(primal.x)).slice(0, k)
        // The dual: maximise over lambda >= 0 and a free w, as fairlearn's second linprog.
        const rows = 2 * m.half
        const dualC = [...new Array<number>(rows).fill(0), -1]
        const dualA = Array.from({ length: k + 1 }, (_, j) => [
          ...Array.from({ length: rows }, (_, i) => -(aUb[i]?.[j] ?? 0)),
          j < k ? 1 : 0,
        ])
        const dual = linprog({
          c: dualC,
          aUb: dualA,
          bUb: c,
          bounds: [...new Array<[number, null]>(rows).fill([0, null]), [null, null]],
        })
        const lambda = Array.from(toFlat(dual.x)).slice(0, rows)
        // The count is taken before the gap is evaluated, which may find a better response and grow the set.
        const gap = evalGap(q, lambda, nuValue).gap
        mixture = { n: k, q, lambda, gap }
        return mixture
      }
      const ex = s.theta.map(Math.exp)
      const z = 1 + ex.reduce((a, v) => a + v, 0)
      const lambda = ex.map((v) => (B * v) / z)
      const lambdaSum = s.lambdaSum.map((v, c) => v + (lambda[c] ?? 0))
      const lambdaEG = lambdaSum.map((v) => v / (t + 1))
      const hIdx = bestH(lambda)
      const h = hs[hIdx] as ReductionHypothesis
      let nu = s.nu
      let eta = s.eta
      if (t === 0) {
        if (nu === null) {
          const absErr = h.decisions.map((d, i) => Math.abs(d - (labels[i] ?? 0)))
          const mean = absErr.reduce((a, v) => a + v, 0) / n
          const sd = Math.sqrt(absErr.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, n - 1))
          nu = (ACCURACY_MUL * sd) / Math.sqrt(n)
        }
        eta = (options.eta ?? 2) / B
      }
      const nuValue = nu as number
      const qsumMap = new Map(s.qsum)
      qsumMap.set(hIdx, (qsumMap.get(hIdx) ?? 0) + 1)
      const total = [...qsumMap.values()].reduce((a, v) => a + v, 0)
      const qEG = hs.map((_, j) => (qsumMap.get(j) ?? 0) / total)
      const resultEG = evalGap(qEG, lambdaEG, nuValue)
      const gapsEG = [...s.gapsEG, resultEG.gap]
      const lp = t === 0 ? undefined : solveLinprog(nuValue)
      const useLP = lp !== undefined && !(resultEG.gap < lp.gap)
      const q = useLP ? lp.q : qEG
      const gap = useLP ? lp.gap : resultEG.gap
      const mix = labels.map((_, i) => hs.reduce((a, hh, j) => a + (q[j] ?? 0) * (hh.decisions[i] ?? 0), 0))
      const step: ReductionStep = {
        iteration: t + 1,
        error: h.error,
        violation: Math.max(0, ...h.gamma),
        averageError: errorOf(labels, mix),
        averageViolation: Math.max(0, ...m.gamma(mix)),
        gap,
        lambdaTotal: Math.max(...lambda.slice(0, m.half).map((v, c) => Math.abs(v - (lambda[c + m.half] ?? 0)))),
      }
      ctx.log('eg/classifier', hIdx)
      ctx.log('eg/error', step.averageError)
      ctx.log('eg/violation', step.averageViolation)
      ctx.log('eg/lambda', step.lambdaTotal)
      ctx.log('eg/gap', gap)
      ctx.log('eg/mixture', hs.length)
      let theta = s.theta
      let eta2 = eta
      let lastRegretChecked = s.lastRegretChecked
      let lastGap = s.lastGap
      const stopped = gap < nuValue && t >= MIN_ITER
      if (!stopped) {
        if (t >= lastRegretChecked * REGRET_CHECK_INCREASE_T) {
          const bestGap = Math.min(...gapsEG)
          if (bestGap > (lastGap ?? Number.POSITIVE_INFINITY) * SHRINK_REGRET) eta2 *= SHRINK_ETA
          lastRegretChecked = t
          lastGap = bestGap
        }
        theta = s.theta.map((v, c) => v + eta2 * ((h.gamma[c] ?? 0) - bound))
      }
      return {
        t: t + 1,
        theta,
        hs,
        calls,
        qsum: [...qsumMap.entries()],
        lambdaSum,
        gapsEG,
        gaps: [...s.gaps, gap],
        qs: [...s.qs, q],
        steps: [...s.steps, step],
        nu,
        eta: eta2,
        lastRegretChecked,
        lastGap,
        mixture,
        stopped,
      }
    },
    done: (s) => (s.stopped ? 'converged' : s.t >= maxIter ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.gaps.length === 0 ? {} : { gap: s.gaps[s.gaps.length - 1] ?? 0 } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ExponentiatedGradientState,
  }
  const finish = (s: ExponentiatedGradientState): ExponentiatedGradientReduction => {
    if (fits.length < s.hs.length || fits.slice(0, s.hs.length).some((f) => f === undefined))
      throw new MdmlError('value', 'exponentiatedGradientReduction: the classifiers of a resumed run were not kept')
    const minGap = Math.min(...s.gaps)
    let bestIteration = 0
    for (const [t, g] of s.gaps.entries()) if (g <= minGap + PRECISION) bestIteration = t
    const chosen = s.qs[bestIteration] ?? []
    const weights = s.hs.map((_, j) => chosen[j] ?? 0)
    const decisions = labels.map((_, i) => s.hs.reduce((a, h, j) => a + (weights[j] ?? 0) * (h.decisions[i] ?? 0), 0))
    // The unconstrained classifier, for comparison.
    const base = estimator.fit({ x, y: labels }, { key: foldIn(key, 0), explain: false })
    const proba = typeof base.predictProba === 'function' ? Array.from(toFlat(base.predictProba(x))) : undefined
    const baseDecisions = predict(base)
    const scores = proba === undefined ? baseDecisions : labels.map((_, i) => proba[2 * i + 1] ?? 0)
    return {
      members: fits.slice(0, s.hs.length),
      weights,
      decisions,
      steps: s.steps,
      bestIteration: bestIteration + 1,
      bestGap: s.gaps[bestIteration] ?? Number.NaN,
      baseline: {
        scores,
        error: errorOf(labels, baseDecisions),
        violation: Math.max(0, ...m.gamma(baseDecisions)),
      },
    }
  }
  return { algorithm, fits, maxIter, finish }
}

/**
 * Exponentiated-gradient fair classification (Agarwal et al. 2018; fairlearn's `ExponentiatedGradient`) as a stepped
 * algorithm: each step is one round of the game between the learner (the best classifier for the current prices of the
 * constraint violations) and the auditor (which raises the price of violated moments), keeping the better of the
 * averaged and the linear-programme mixture, and stopping once the duality gap is under nu. The state serialises; the
 * fitted classifiers are not part of it, so only a run from the start gives the final randomised classifier (see
 * {@link exponentiatedGradientReduction}).
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, epsilon, difference bound, iterations, step size, nu and key.
 * @returns The algorithm, at most `iterations` steps.
 * @throws MdmlError `value` for an epsilon that is not positive.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { exponentiatedGradientAlgorithm } from 'mdml-engine/fairness'
 * import { defineEstimator, matrixView } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { asTensor } from 'mdml-engine/tensor'
 * const stump = defineEstimator({
 *   name: 'stump', kind: 'classifier',
 *   fit: (data) => { const m = matrixView(data.x, 'stump'); let s = 0; for (let i = 0; i < m.n; i++) s += m.data[i * m.d] ?? 0; return { t: s / m.n } },
 *   methods: (p) => ({ classes: [0, 1], predict: (x) => { const m = matrixView(x, 'stump'); return asTensor(Array.from({ length: m.n }, (_, i) => ((m.data[i * m.d] ?? 0) > p.t ? 1 : 0))) } }),
 * }, {})
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * const data = { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }
 * runAlgorithm(exponentiatedGradientAlgorithm(stump, data, { iterations: 3 }), { maxSteps: 3 }).state.steps.length // => 3
 * ```
 */
export function exponentiatedGradientAlgorithm(
  estimator: Estimator,
  data: ReductionData,
  options: ExponentiatedGradientReductionOptions = {},
): Algorithm<ExponentiatedGradientState, ExponentiatedGradientPath> {
  return exponentiatedGradientRun(estimator, data, options).algorithm
}

/**
 * Fair classification by reduction to a sequence of cost-sensitive problems (Agarwal et al., 2018), a port of fairlearn's
 * `ExponentiatedGradient`: a game between a learner, which fits the best classifier for the current prices of the
 * constraint violations (rows relabelled by the sign of their cost and weighted by its size), and an auditor, which
 * raises the price of every violated moment by exponentiated gradient. Each iteration keeps the better (smaller duality
 * gap) of the average of the best responses so far and the best mixture of them by linear programme, stops once the gap
 * falls below nu, and returns the mixture of the iteration with the smallest gap: a randomised classifier whose
 * violation is near the difference bound. As fairlearn, `epsilon` sets the multipliers' bound (1 / epsilon) and the
 * constraint's own difference bound (default 0.01) the violation allowed. This is {@link exponentiatedGradientAlgorithm} run
 * to the end.
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, epsilon, iterations, step size, nu and key.
 * @returns The members, their weights, the decisions, the steps and the unconstrained baseline.
 * @throws MdmlError `value` for an epsilon that is not positive.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { exponentiatedGradientReduction } from 'mdml-engine/fairness'
 * import { defineEstimator, matrixView } from 'mdml-engine/learn'
 * import { asTensor } from 'mdml-engine/tensor'
 * // A stump on the first feature: the simplest classifier an estimator can be.
 * const stump = defineEstimator({
 *   name: 'stump', kind: 'classifier',
 *   fit: (data) => { const m = matrixView(data.x, 'stump'); let s = 0; for (let i = 0; i < m.n; i++) s += m.data[i * m.d] ?? 0; return { t: s / m.n } },
 *   methods: (p) => ({ classes: [0, 1], predict: (x) => { const m = matrixView(x, 'stump'); return asTensor(Array.from({ length: m.n }, (_, i) => ((m.data[i * m.d] ?? 0) > p.t ? 1 : 0))) } }),
 * }, {})
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * const r = exponentiatedGradientReduction(stump, { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }, { iterations: 3 })
 * r.steps.length // => 3
 * ```
 */
export function exponentiatedGradientReduction(
  estimator: Estimator,
  data: ReductionData,
  options: ExponentiatedGradientReductionOptions = {},
): ExponentiatedGradientReduction {
  const { algorithm, maxIter, finish } = exponentiatedGradientRun(estimator, data, options)
  return finish(runAlgorithm(algorithm, { maxSteps: maxIter, seed: 0 }).state)
}

/** Options of {@link gridReduction}. */
export interface GridReductionOptions {
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** Multiplier vectors tried (default 10, fairlearn's grid_size). */
  readonly gridSize?: number
  /** The largest multiplier (default 2, fairlearn's grid_limit). */
  readonly bound?: number
  /** The weight of the violation against the error when picking the best point (default 0.5). */
  readonly constraintWeight?: number
  readonly key?: Key
}

/** One classifier of {@link gridReduction}. */
export interface GridPoint {
  /** The signed multiplier on each basis constraint (one per event and non-final group). */
  readonly lambda: number
  readonly lambdas: readonly number[]
  readonly error: number
  /** The largest moment (fairlearn's `gamma.max()`): the worst gap of a group's mean decision over everyone's. */
  readonly violation: number
  readonly fitted: Fitted
  /** Whether it is the point fairlearn's tradeoff rule picks. */
  readonly best: boolean
}

/** fairlearn's integer grid: every vector of `dim` integers with |entries| summing to at most `units`, in its order. */
function integerGrid(dim: number, units: number): number[][] {
  const out: number[][] = []
  const entry = new Array<number>(dim).fill(0)
  const accumulate = (index: number, max: number): void => {
    if (index === dim) {
      out.push([...entry])
      return
    }
    for (let v = -max; v <= max; v++) {
      entry[index] = v
      accumulate(index + 1, max - Math.abs(v))
    }
  }
  accumulate(0, units)
  return out
}

/** The paths {@link gridReductionAlgorithm} logs. */
export type GridReductionPath = 'grid/point' | 'grid/error' | 'grid/violation' | 'grid/loss' | 'grid/best'

/** What a stepped grid reduction logs: the point fitted, its error and violation, its trade-off loss and the best so far. */
export const gridReductionContract: Contract<unknown, GridReductionPath> = defineContract({
  'grid/point': scalar({ description: 'the grid point just fitted (from 0)' }),
  'grid/error': scalar({ description: 'the classifier’s error', direction: 'lower' }),
  'grid/violation': scalar({ description: 'its largest moment (the worst gap of a group)', direction: 'lower' }),
  'grid/loss': scalar({ description: '(1 - w) error + w violation', direction: 'lower' }),
  'grid/best': scalar({ description: 'the index of the point with the smallest loss so far' }),
})

/** One evaluated grid point of {@link GridReductionState} (the classifier is held apart). */
export interface GridPointScores {
  readonly lambda: number
  readonly lambdas: readonly number[]
  readonly error: number
  readonly violation: number
}

/** The state of {@link gridReductionAlgorithm}: the grid points fitted so far, in order. */
export interface GridReductionState {
  readonly points: readonly GridPointScores[]
}

function gridReductionRun(
  estimator: Estimator,
  data: ReductionData,
  options: GridReductionOptions,
): {
  readonly algorithm: Algorithm<GridReductionState, GridReductionPath>
  readonly fits: Fitted[]
  readonly size: number
  readonly finish: (s: GridReductionState) => GridPoint[]
} {
  const { x, labels, groups } = data
  const m = moments(options.constraint ?? 'demographicParity', groups, labels)
  const size = options.gridSize ?? 10
  const limit = options.bound ?? 2
  const w = options.constraintWeight ?? 0.5
  const key = options.key ?? makeKey(0)
  const dim = m.basis.length
  let units = Math.max(0, Math.floor((size / 2 ** dim) ** (1 / dim) - 1))
  let grid = integerGrid(dim, units)
  while (grid.length < size) grid = integerGrid(dim, ++units)
  const vectors = grid.slice(0, size).map((v) => v.map((c) => (c * limit) / units))
  const objective = labels.map((y) => 2 * (y === 1 ? 1 : 0) - 1)
  const fits: Fitted[] = []
  const lossOf = (p: GridPointScores): number => (1 - w) * p.error + w * p.violation
  const algorithm: Algorithm<GridReductionState, GridReductionPath> = {
    name: 'grid-reduction',
    config: { constraint: options.constraint ?? 'demographicParity', gridSize: size, bound: limit },
    schema: channels({ metrics: { loss: { description: 'the trade-off loss of the point', direction: 'lower' } } }),
    init: () => ({ points: [] }),
    step: (s, ctx) => {
      const index = s.points.length
      const coefs = vectors[index] as number[]
      const lambda = new Array<number>(2 * m.half).fill(0)
      coefs.forEach((c, j) => {
        const cell = m.basis[j] ?? 0
        if (c > 0) lambda[cell] = c
        else lambda[cell + m.half] = -c
      })
      const sw = m.signedWeights(lambda)
      const fitted = callOracle(
        estimator,
        x,
        labels,
        objective.map((o, i) => o + (sw[i] ?? 0)),
        false,
        foldIn(key, index),
      )
      const decisions = Array.from(toFlat(requireVerb(fitted, 'predict', 'gridReduction')(x)))
      const point: GridPointScores = {
        lambda: coefs[0] ?? 0,
        lambdas: coefs,
        error: errorOf(labels, decisions),
        violation: Math.max(...m.gamma(decisions)),
      }
      fits[index] = fitted
      const points = [...s.points, point]
      const losses = points.map(lossOf)
      ctx.log('grid/point', index)
      ctx.log('grid/error', point.error)
      ctx.log('grid/violation', point.violation)
      ctx.log('grid/loss', lossOf(point))
      ctx.log('grid/best', losses.indexOf(Math.min(...losses)))
      return { points }
    },
    done: (s) => (s.points.length >= size ? 'completed' : undefined),
    observe: (s) => {
      const last = s.points[s.points.length - 1]
      return { metrics: last === undefined ? {} : { loss: lossOf(last) } }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GridReductionState,
  }
  const finish = (s: GridReductionState): GridPoint[] => {
    if (fits.length < s.points.length)
      throw new MdmlError('value', 'gridReduction: the classifiers of a resumed run were not kept')
    const losses = s.points.map(lossOf)
    const best = losses.indexOf(Math.min(...losses))
    return s.points.map((p, i) => ({ ...p, fitted: fits[i] as Fitted, best: i === best }))
  }
  return { algorithm, fits, size, finish }
}

/**
 * Fair classification by a grid of multipliers (fairlearn's `GridSearch`) as a stepped algorithm: each step fits the
 * classifier best for one multiplier vector of the grid and records its error and largest moment. The state (the
 * scores of the points so far) serialises; the classifiers are kept by {@link gridReduction}, not in the state.
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, grid size, bound, constraint weight and key.
 * @returns The algorithm, one step per grid point.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { gridReductionAlgorithm } from 'mdml-engine/fairness'
 * import { defineEstimator, matrixView } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { asTensor } from 'mdml-engine/tensor'
 * const stump = defineEstimator({
 *   name: 'stump', kind: 'classifier',
 *   fit: (data) => { const m = matrixView(data.x, 'stump'); let s = 0; for (let i = 0; i < m.n; i++) s += m.data[i * m.d] ?? 0; return { t: s / m.n } },
 *   methods: (p) => ({ classes: [0, 1], predict: (x) => { const m = matrixView(x, 'stump'); return asTensor(Array.from({ length: m.n }, (_, i) => ((m.data[i * m.d] ?? 0) > p.t ? 1 : 0))) } }),
 * }, {})
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * const data = { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }
 * runAlgorithm(gridReductionAlgorithm(stump, data, { gridSize: 3 }), { maxSteps: 5 }).state.points.length // => 3
 * ```
 */
export function gridReductionAlgorithm(
  estimator: Estimator,
  data: ReductionData,
  options: GridReductionOptions = {},
): Algorithm<GridReductionState, GridReductionPath> {
  return gridReductionRun(estimator, data, options).algorithm
}

/**
 * Fair classification by a grid of multipliers, a port of fairlearn's `GridSearch`: fairlearn's grid of multiplier
 * vectors (integer points of the cross-polytope on the basis of non-redundant constraints, scaled to `bound`), for each
 * the classifier best for those prices (rows relabelled by the sign of their cost and weighted by its size), each
 * point's error and largest moment, and the point minimising `(1 - w) error + w violation`. This is
 * {@link gridReductionAlgorithm} run to the end. With two groups the grid is
 * one multiplier, the price on the first group's gap.
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, grid size, bound, constraint weight and key.
 * @returns The classifiers with their multipliers, error and violation, the best marked.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { gridReduction } from 'mdml-engine/fairness'
 * import { defineEstimator, matrixView } from 'mdml-engine/learn'
 * import { asTensor } from 'mdml-engine/tensor'
 * // A stump on the first feature: the simplest classifier an estimator can be.
 * const stump = defineEstimator({
 *   name: 'stump', kind: 'classifier',
 *   fit: (data) => { const m = matrixView(data.x, 'stump'); let s = 0; for (let i = 0; i < m.n; i++) s += m.data[i * m.d] ?? 0; return { t: s / m.n } },
 *   methods: (p) => ({ classes: [0, 1], predict: (x) => { const m = matrixView(x, 'stump'); return asTensor(Array.from({ length: m.n }, (_, i) => ((m.data[i * m.d] ?? 0) > p.t ? 1 : 0))) } }),
 * }, {})
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * gridReduction(stump, { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }, { gridSize: 3 }).length // => 3
 * ```
 */
export function gridReduction(
  estimator: Estimator,
  data: ReductionData,
  options: GridReductionOptions = {},
): GridPoint[] {
  const { algorithm, size, finish } = gridReductionRun(estimator, data, options)
  return finish(runAlgorithm(algorithm, { maxSteps: size, seed: 0 }).state)
}
