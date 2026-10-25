/**
 * Fairness during training by reduction, ports of fairlearn's `reductions`: fair classification becomes a sequence of
 * cost-sensitive problems under a demographic-parity or equalised-odds constraint. Exponentiated gradient (a game
 * between a learner and a constraint player, giving a randomised classifier; golden against fairlearn's final mixture)
 * and the grid search over multipliers that traces the accuracy-fairness trade-off.
 */

import { WhiteboxError } from '../base/index.ts'
import { type Estimator, type Fitted, requireVerb } from '../learn/index.ts'
import { linprog } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { asTensor, fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'

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

/** A fitted best response: the classifier, its decisions on the training rows, error and moments. */
interface Hypothesis {
  readonly fitted: Fitted
  readonly decisions: number[]
  readonly error: number
  readonly gamma: number[]
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

const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)

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

/**
 * Fair classification by reduction to a sequence of cost-sensitive problems (Agarwal et al., 2018), a port of fairlearn's
 * `ExponentiatedGradient`: a game between a learner, which fits the best classifier for the current prices of the
 * constraint violations (rows relabelled by the sign of their cost and weighted by its size), and an auditor, which
 * raises the price of every violated moment by exponentiated gradient. Each iteration keeps the better (smaller duality
 * gap) of the average of the best responses so far and the best mixture of them by linear programme, stops once the gap
 * falls below nu, and returns the mixture of the iteration with the smallest gap: a randomised classifier whose
 * violation is near the difference bound. As fairlearn, `epsilon` sets the multipliers' bound (1 / epsilon) and the
 * constraint's own difference bound (default 0.01) the violation allowed.
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, epsilon, iterations, step size, nu and key.
 * @returns The members, their weights, the decisions, the steps and the unconstrained baseline.
 * @throws WhiteboxError `value` for an epsilon that is not positive.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { exponentiatedGradientReduction } from 'whitebox-engine/fairness'
 * import { logisticRegression } from 'whitebox-engine/methods/linear'
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * const r = exponentiatedGradientReduction(logisticRegression(), { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }, { iterations: 3 })
 * r.steps.length // => 3
 * ```
 */
export function exponentiatedGradientReduction(
  estimator: Estimator,
  data: ReductionData,
  options: ExponentiatedGradientReductionOptions = {},
): ExponentiatedGradientReduction {
  const { x, labels, groups } = data
  const constraint = options.constraint ?? 'demographicParity'
  const epsilon = options.epsilon ?? 0.01
  if (!(epsilon > 0)) throw new WhiteboxError('value', 'exponentiatedGradientReduction: epsilon must be positive')
  const maxIter = options.iterations ?? 50
  const B = 1 / epsilon
  const bound = options.differenceBound ?? 0.01
  const key = options.key ?? makeKey(0)
  const m = moments(constraint, groups, labels)
  const n = labels.length
  const objectiveWeights = labels.map((y) => 2 * (y === 1 ? 1 : 0) - 1)
  const hs: Hypothesis[] = []
  let calls = 0
  const predict = (f: Fitted): number[] =>
    Array.from(toFlat(requireVerb(f, 'predict', 'exponentiatedGradientReduction')(x)))
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
    const value = error + dot(gamma, lambda)
    let best = -1
    let bestValue = Number.POSITIVE_INFINITY
    for (const [j, h] of hs.entries()) {
      const v = h.error + dot(h.gamma, lambda)
      if (v < bestValue) {
        bestValue = v
        best = j
      }
    }
    if (value < bestValue - PRECISION) {
      hs.push({ fitted, decisions, error, gamma })
      return hs.length - 1
    }
    return best
  }
  // The Lagrangian of a mixture (weights over hs) at multipliers: (L, L_high, gamma, error).
  const evaluate = (q: readonly number[], lambda: readonly number[]) => {
    const error = hs.reduce((s, h, j) => s + h.error * (q[j] ?? 0), 0)
    const gamma = Array.from({ length: 2 * m.half }, (_, c) =>
      hs.reduce((s, h, j) => s + (h.gamma[c] ?? 0) * (q[j] ?? 0), 0),
    )
    const projected = m.project(lambda)
    const L = error + projected.reduce((s, l, c) => s + l * ((gamma[c] ?? 0) - bound), 0)
    const maxConstraint = Math.max(...gamma.map((g) => g - bound))
    return { L, high: error + (maxConstraint > 0 ? B * maxConstraint : 0), gamma, error }
  }
  const evalGap = (q: readonly number[], lambda: readonly number[], nu: number) => {
    const e = evaluate(q, lambda)
    let low = e.L
    const gap = (): number => Math.max(e.L - low, e.high - e.L)
    for (const mul of [1, 2, 5, 10]) {
      const idx = bestH(lambda.map((l) => mul * l))
      const single = hs.map((_, j) => (j === idx ? 1 : 0))
      const lm = evaluate(single, lambda).L
      if (lm < low) low = lm
      if (gap() > nu + PRECISION) break
    }
    return { ...e, gap: gap() }
  }
  let linprogCache: { n: number; q: number[]; gap: number } | undefined
  const solveLinprog = (nu: number) => {
    if (linprogCache !== undefined && linprogCache.n === hs.length) return linprogCache
    const t = hs.length
    const c = [...hs.map((h) => h.error), B]
    const aUb = Array.from({ length: 2 * m.half }, (_, j) => [...hs.map((h) => (h.gamma[j] ?? 0) - bound), -1])
    const primal = linprog({ c, aUb, bUb: aUb.map(() => 0), aEq: [[...hs.map(() => 1), 0]], bEq: [1] })
    const q = Array.from(toFlat(primal.x)).slice(0, t)
    // The dual: maximise over lambda >= 0 and a free w, as fairlearn's second linprog.
    const rows = 2 * m.half
    const dualC = [...new Array<number>(rows).fill(0), -1]
    const dualA = Array.from({ length: t + 1 }, (_, j) => [
      ...Array.from({ length: rows }, (_, i) => -(aUb[i]?.[j] ?? 0)),
      j < t ? 1 : 0,
    ])
    const dual = linprog({
      c: dualC,
      aUb: dualA,
      bUb: c,
      bounds: [...new Array<[number, null]>(rows).fill([0, null]), [null, null]],
    })
    const lambda = Array.from(toFlat(dual.x)).slice(0, rows)
    const result = { n: hs.length, q, lambda, gap: 0 }
    result.gap = evalGap(q, lambda, nu).gap
    linprogCache = result
    return result
  }
  let theta = new Array<number>(2 * m.half).fill(0)
  const qsum = new Map<number, number>()
  const lambdaSum = new Array<number>(2 * m.half).fill(0)
  const gapsEG: number[] = []
  const gaps: number[] = []
  const qs: number[][] = []
  const steps: ReductionStep[] = []
  let nu = options.nu
  let eta = 0
  let lastRegretChecked = REGRET_CHECK_START_T
  let lastGap = Number.POSITIVE_INFINITY
  for (let t = 0; t < maxIter; t++) {
    const ex = theta.map(Math.exp)
    const z = 1 + ex.reduce((s, v) => s + v, 0)
    const lambda = ex.map((v) => (B * v) / z)
    for (let c = 0; c < lambda.length; c++) lambdaSum[c] = (lambdaSum[c] ?? 0) + (lambda[c] ?? 0)
    const lambdaEG = lambdaSum.map((v) => v / (t + 1))
    const hIdx = bestH(lambda)
    const h = hs[hIdx] as Hypothesis
    if (t === 0) {
      if (nu === undefined) {
        const absErr = h.decisions.map((d, i) => Math.abs(d - (labels[i] ?? 0)))
        const mean = absErr.reduce((s, v) => s + v, 0) / n
        const sd = Math.sqrt(absErr.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, n - 1))
        nu = (ACCURACY_MUL * sd) / Math.sqrt(n)
      }
      eta = (options.eta ?? 2) / B
    }
    qsum.set(hIdx, (qsum.get(hIdx) ?? 0) + 1)
    const total = [...qsum.values()].reduce((s, v) => s + v, 0)
    const qEG = hs.map((_, j) => (qsum.get(j) ?? 0) / total)
    const resultEG = evalGap(qEG, lambdaEG, nu as number)
    gapsEG.push(resultEG.gap)
    const lp = t === 0 ? undefined : solveLinprog(nu as number)
    const useLP = lp !== undefined && !(resultEG.gap < lp.gap)
    const q = useLP ? lp.q : qEG
    qs.push(q)
    gaps.push(useLP ? lp.gap : resultEG.gap)
    const mixture = labels.map((_, i) => hs.reduce((s, hh, j) => s + (q[j] ?? 0) * (hh.decisions[i] ?? 0), 0))
    steps.push({
      iteration: t + 1,
      error: h.error,
      violation: Math.max(0, ...h.gamma),
      averageError: errorOf(labels, mixture),
      averageViolation: Math.max(0, ...m.gamma(mixture)),
      gap: gaps[t] ?? Number.NaN,
      lambdaTotal: Math.max(...lambda.slice(0, m.half).map((v, c) => Math.abs(v - (lambda[c + m.half] ?? 0)))),
    })
    if ((gaps[t] ?? 0) < (nu as number) && t >= MIN_ITER) break
    if (t >= lastRegretChecked * REGRET_CHECK_INCREASE_T) {
      const bestGap = Math.min(...gapsEG)
      if (bestGap > lastGap * SHRINK_REGRET) eta *= SHRINK_ETA
      lastRegretChecked = t
      lastGap = bestGap
    }
    theta = theta.map((v, c) => v + eta * ((h.gamma[c] ?? 0) - bound))
  }
  const minGap = Math.min(...gaps)
  let bestIteration = 0
  for (const [t, g] of gaps.entries()) if (g <= minGap + PRECISION) bestIteration = t
  const chosen = qs[bestIteration] ?? []
  const weights = hs.map((_, j) => chosen[j] ?? 0)
  const decisions = labels.map((_, i) => hs.reduce((s, h, j) => s + (weights[j] ?? 0) * (h.decisions[i] ?? 0), 0))
  // The unconstrained classifier, for comparison.
  const base = estimator.fit({ x, y: labels }, { key: foldIn(key, 0), explain: false })
  const proba = typeof base.predictProba === 'function' ? Array.from(toFlat(base.predictProba(x))) : undefined
  const baseDecisions = predict(base)
  const scores = proba === undefined ? baseDecisions : labels.map((_, i) => proba[2 * i + 1] ?? 0)
  return {
    members: hs.map((h) => h.fitted),
    weights,
    decisions,
    steps,
    bestIteration: bestIteration + 1,
    bestGap: gaps[bestIteration] ?? Number.NaN,
    baseline: {
      scores,
      error: errorOf(labels, baseDecisions),
      violation: Math.max(0, ...m.gamma(baseDecisions)),
    },
  }
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

/**
 * Fair classification by a grid of multipliers, a port of fairlearn's `GridSearch`: fairlearn's grid of multiplier
 * vectors (integer points of the cross-polytope on the basis of non-redundant constraints, scaled to `bound`), for each
 * the classifier best for those prices (rows relabelled by the sign of their cost and weighted by its size), each
 * point's error and largest moment, and the point minimising `(1 - w) error + w violation`. With two groups the grid is
 * one multiplier, the price on the first group's gap.
 *
 * @param estimator - A classifier that accepts sample weights.
 * @param data - Features, binary labels and group indices.
 * @param options - Constraint, grid size, bound, constraint weight and key.
 * @returns The classifiers with their multipliers, error and violation, the best marked.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { gridReduction } from 'whitebox-engine/fairness'
 * import { logisticRegression } from 'whitebox-engine/methods/linear'
 * const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5]]
 * gridReduction(logisticRegression(), { x, labels: [0, 0, 1, 1, 0, 1, 1, 1], groups: [0, 0, 0, 0, 1, 1, 1, 1] }, { gridSize: 3 }).length // => 3
 * ```
 */
export function gridReduction(
  estimator: Estimator,
  data: ReductionData,
  options: GridReductionOptions = {},
): GridPoint[] {
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
  const points = vectors.map((coefs, s) => {
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
      foldIn(key, s),
    )
    const decisions = Array.from(toFlat(requireVerb(fitted, 'predict', 'gridReduction')(x)))
    return {
      lambda: coefs[0] ?? 0,
      lambdas: coefs,
      error: errorOf(labels, decisions),
      violation: Math.max(...m.gamma(decisions)),
      fitted,
    }
  })
  const losses = points.map((p) => (1 - w) * p.error + w * p.violation)
  const best = losses.indexOf(Math.min(...losses))
  return points.map((p, i) => ({ ...p, best: i === best }))
}
