import { WhiteboxError } from '../../lib/base/index.ts'
import { curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import { type Estimator, type Fitted, requireVerb } from '../../lib/learn/index.ts'
import { linprog } from '../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { buildGroupReport, groupReportReference } from './audit.ts'
import { binaryLabels, type FairnessData, type SensitiveSpec, sensitiveGroups, withoutColumns } from './groups.ts'

/** The moment a reduction constrains: demographic parity or equalised odds. */
export type ReductionConstraint = 'demographicParity' | 'equalisedOdds'

/**
 * The constraint moments of a reduction (Agarwal et al., 2018): for each group (and, for equalised odds, each label)
 * the gap between the group's mean decision and everyone's, each with a positive and a negative side. `gamma` gives
 * the moments of decisions; `gradient` the derivative of `sum_j lambda_j gamma_j` with respect to each row's decision.
 */
interface Moments {
  readonly count: number
  gamma(h: readonly number[]): number[]
  gradient(lambda: readonly number[]): Float64Array
}

function moments(
  constraint: ReductionConstraint,
  groups: readonly number[],
  labels: readonly number[],
  k: number,
): Moments {
  const n = groups.length
  // Cells: a group (demographic parity) or a group and a label (equalised odds); each compared with its label's rows.
  const cells = constraint === 'demographicParity' ? k : 2 * k
  const cellOf = (i: number): number =>
    constraint === 'demographicParity' ? (groups[i] ?? 0) : 2 * (groups[i] ?? 0) + (labels[i] ?? 0)
  const baseOf = (i: number): number => (constraint === 'demographicParity' ? 0 : (labels[i] ?? 0))
  const cellSize = new Array<number>(cells).fill(0)
  const baseSize = [0, 0]
  for (let i = 0; i < n; i++) {
    cellSize[cellOf(i)] = (cellSize[cellOf(i)] ?? 0) + 1
    baseSize[baseOf(i)] = (baseSize[baseOf(i)] ?? 0) + 1
  }
  const baseOfCell = (c: number): number => (constraint === 'demographicParity' ? 0 : c % 2)
  return {
    count: 2 * cells,
    gamma: (h) => {
      const cellSum = new Array<number>(cells).fill(0)
      const baseSum = [0, 0]
      for (let i = 0; i < n; i++) {
        cellSum[cellOf(i)] = (cellSum[cellOf(i)] ?? 0) + (h[i] ?? 0)
        baseSum[baseOf(i)] = (baseSum[baseOf(i)] ?? 0) + (h[i] ?? 0)
      }
      const gap = cellSum.map((s, c) => {
        const size = cellSize[c] ?? 0
        const b = baseOfCell(c)
        return size === 0 ? 0 : s / size - (baseSum[b] ?? 0) / Math.max(1, baseSize[b] ?? 0)
      })
      return [...gap, ...gap.map((v) => -v)]
    },
    gradient: (lambda) => {
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const c = cellOf(i)
        const b = baseOf(i)
        let s = 0
        for (let d = 0; d < cells; d++) {
          if (baseOfCell(d) !== b) continue
          const net = (lambda[d] ?? 0) - (lambda[d + cells] ?? 0)
          s += net * ((d === c ? 1 / Math.max(1, cellSize[d] ?? 0) : 0) - 1 / Math.max(1, baseSize[b] ?? 0))
        }
        out[i] = s
      }
      return out
    },
  }
}

/** The largest constraint violation of decisions: the worst |gap| of any group's mean decision. */
function violation(m: Moments, h: readonly number[]): number {
  return Math.max(0, ...m.gamma(h))
}

/** The cost-sensitive best response: a classifier fitted to the signed costs of predicting positive. */
function bestResponse(
  estimator: Estimator,
  x: FairnessData['x'],
  labels: readonly number[],
  m: Moments,
  lambda: readonly number[],
  key: Key,
): { readonly fitted: Fitted; readonly decisions: number[] } {
  const n = labels.length
  const g = m.gradient(lambda)
  // Cost of predicting 1 minus predicting 0: the error's (1 for a negative, -1 for a positive) plus the moments'.
  const cost = labels.map((y, i) => (y === 1 ? -1 : 1) / n + (g[i] ?? 0))
  const total = cost.reduce((s, c) => s + Math.abs(c), 0)
  const weights = cost.map((c) => (total === 0 ? 1 : (Math.abs(c) * n) / total))
  const target = cost.map((c) => (c < 0 ? 1 : 0))
  // A degenerate target (every row wants the same decision) is a constant classifier.
  if (target.every((t) => t === target[0])) {
    const constant = target[0] ?? 0
    return { fitted: { estimator: 'constant' } as unknown as Fitted, decisions: labels.map(() => constant) }
  }
  const fitted = estimator.fit({ x, y: target, weights }, { key, explain: false })
  return { fitted, decisions: Array.from(toFlat(requireVerb(fitted, 'predict', 'exponentiatedGradient')(x))) }
}

/**
 * The mixture of hypotheses with the least error whose moments exceed epsilon by at most a slack priced at the bound
 * (fairlearn's `solve_linprog`): minimise sum_h w_h err_h + B s subject to sum_h w_h (gamma_jh - epsilon) <= s for
 * every moment j, the weights a distribution and s >= 0.
 */
function bestMixture(
  errors: readonly number[],
  gammas: readonly (readonly number[])[],
  epsilon: number,
  bound: number,
): number[] {
  const t = errors.length
  if (t === 1) return [1]
  const moments = gammas[0]?.length ?? 0
  const aUb = Array.from({ length: moments }, (_, j) => [...gammas.map((g) => (g[j] ?? 0) - epsilon), -1])
  const r = linprog({
    c: [...errors, bound],
    aUb,
    bUb: new Array<number>(moments).fill(0),
    aEq: [[...new Array<number>(t).fill(1), 0]],
    bEq: [1],
  })
  if (!r.success) return errors.map(() => 1 / t)
  const w = Array.from(toFlat(r.x))
    .slice(0, t)
    .map((v) => Math.max(0, v))
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((v) => v / total)
}

/** Options of {@link exponentiatedGradient}. */
export interface ExponentiatedGradientOptions {
  readonly sensitive?: SensitiveSpec
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** The allowed violation of each moment (default 0.01). */
  readonly epsilon?: number
  /** Iterations (default 30). */
  readonly iterations?: number
  /** The bound on the multipliers' total (default 1 / epsilon, at most 100). */
  readonly bound?: number
  /** The step size times the bound (default 2, fairlearn's `eta0`). */
  readonly eta?: number
  /** The classifier fitted in each best response (default logistic regression); it must accept sample weights. */
  readonly estimator?: Estimator
  /** Fit without the sensitive column (default false). */
  readonly dropSensitive?: boolean
  readonly reference?: number | string
  readonly key?: Key
}

/** One iteration of {@link exponentiatedGradient}. */
export interface ReductionStep {
  readonly iteration: number
  /** The error and violation of this iteration's best response. */
  readonly error: number
  readonly violation: number
  /** The error and violation of the best mixture of the best responses so far (the randomised classifier). */
  readonly averageError: number
  readonly averageViolation: number
  /** The largest net price on a constraint, how hard the constraints push. */
  readonly lambdaTotal: number
}

/** The result of {@link exponentiatedGradient}. */
export interface ExponentiatedGradient {
  readonly groups: readonly string[]
  /** The best responses; the randomised classifier picks member k with probability `weights[k]`. */
  readonly members: readonly Fitted[]
  readonly weights: readonly number[]
  /** P(positive decision) for each training row under the randomised classifier. */
  readonly decisions: readonly number[]
  readonly steps: readonly ReductionStep[]
  /** The unconstrained classifier's error and violation, for comparison. */
  readonly baseline: { readonly error: number; readonly violation: number }
}

const errorOf = (labels: readonly number[], h: readonly number[]): number =>
  labels.reduce((s, y, i) => s + Math.abs(y - (h[i] ?? 0)), 0) / Math.max(1, labels.length)

/**
 * Fair classification by reduction to a sequence of cost-sensitive problems (Agarwal et al., 2018; fairlearn's
 * `ExponentiatedGradient`): a game between a learner, which fits the best classifier for the current prices of the
 * constraint violations (a weighted fit with relabelled rows), and an auditor, which raises the price of every
 * violated moment by exponentiated gradient. The best mixture of the learner's classifiers (a small linear program
 * over their errors and moments, as fairlearn) is a randomised classifier whose violation shrinks towards `epsilon` as
 * error rises a little. Exposition: the trade-off moving over the
 * iterations (error and violation of the randomised classifier and of each best response, the price on the violation), the path in
 * (violation, error) space from the unconstrained classifier, and the group reports before and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Constraint, epsilon, iterations, bound, step size, estimator and key.
 * @returns The members, decisions and steps, with the records.
 */
export function exponentiatedGradient(
  data: FairnessData,
  options: ExponentiatedGradientOptions = {},
): Explained<ExponentiatedGradient> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'exponentiatedGradient')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const constraint = options.constraint ?? 'demographicParity'
  const epsilon = options.epsilon ?? 0.01
  if (!(epsilon >= 0)) throw new WhiteboxError('value', 'exponentiatedGradient: epsilon must be at least 0')
  const iterations = options.iterations ?? 30
  const bound = options.bound ?? Math.min(100, 1 / Math.max(epsilon, 1e-9))
  const eta = options.eta ?? 2
  const estimator = options.estimator ?? logisticRegression()
  const key = options.key ?? makeKey(0)
  const m = moments(constraint, groups.index, labels, groups.names.length)
  const base = estimator.fit({ x, y: labels }, { key: foldIn(key, 0), explain: false })
  const baseProba = Array.from(toFlat(requireVerb(base, 'predictProba', 'exponentiatedGradient')(x)))
  const baseScores = labels.map((_, i) => baseProba[2 * i + 1] ?? 0)
  const baseDecisions = baseScores.map((s) => (s >= 0.5 ? 1 : 0))
  const theta = new Array<number>(m.count).fill(0)
  const members: Fitted[] = []
  const hs: number[][] = []
  const errors: number[] = []
  const gammas: number[][] = []
  const steps: ReductionStep[] = []
  // fairlearn's step: eta0 / B, constant.
  const step = eta / bound
  let weights: number[] = []
  for (let t = 0; t < iterations; t++) {
    const ex = theta.map(Math.exp)
    const z = 1 + ex.reduce((s, v) => s + v, 0)
    const lambda = ex.map((v) => (bound * v) / z)
    const r = bestResponse(estimator, x, labels, m, lambda, foldIn(key, t + 1))
    members.push(r.fitted)
    hs.push(r.decisions)
    errors.push(errorOf(labels, r.decisions))
    const gamma = m.gamma(r.decisions)
    gammas.push(gamma)
    weights = bestMixture(errors, gammas, epsilon, bound)
    const mixture = labels.map((_, i) => hs.reduce((s, h, k) => s + (weights[k] ?? 0) * (h[i] ?? 0), 0))
    const half = m.count / 2
    steps.push({
      iteration: t + 1,
      error: errors[t] ?? 0,
      violation: violation(m, r.decisions),
      averageError: errorOf(labels, mixture),
      averageViolation: violation(m, mixture),
      lambdaTotal: Math.max(...lambda.slice(0, half).map((v, j) => Math.abs(v - (lambda[j + half] ?? 0)))),
    })
    for (let j = 0; j < theta.length; j++) theta[j] = (theta[j] ?? 0) + step * ((gamma[j] ?? 0) - epsilon)
  }
  const decisions = labels.map((_, i) => hs.reduce((s, h, k) => s + (weights[k] ?? 0) * (h[i] ?? 0), 0))
  const baseline = { error: errorOf(labels, baseDecisions), violation: violation(m, baseDecisions) }
  const reference = groupReportReference(groups, options.reference)
  const classes = data.classes?.length === 2 ? ([data.classes[0], data.classes[1]] as [string, string]) : undefined
  const thresholds = groups.names.map(() => 0.5)
  const common = { labels, groups, reference, thresholds, ...(classes === undefined ? {} : { classes }) }
  const it = steps.map((s) => s.iteration)
  const exposition: Exposition[] = [
    curveSet({
      x: it,
      xLabel: 'iteration',
      yLabel: 'rate',
      y2Label: 'price on the violation',
      series: [
        { name: 'error (randomised classifier)', y: steps.map((s) => s.averageError), group: 'error' },
        { name: 'error (best response)', y: steps.map((s) => s.error), group: 'error' },
        { name: 'violation (randomised classifier)', y: steps.map((s) => s.averageViolation), group: 'violation' },
        { name: 'violation (best response)', y: steps.map((s) => s.violation), group: 'violation' },
        { name: 'price', y: steps.map((s) => s.lambdaTotal), axis: 'right', group: 'price' },
      ],
    }),
    curveSet({
      xLabel: 'constraint violation',
      yLabel: 'error',
      series: [
        {
          name: 'randomised classifier, iteration by iteration',
          x: [baseline.violation, ...steps.map((s) => s.averageViolation)],
          y: [baseline.error, ...steps.map((s) => s.averageError)],
        },
      ],
      marks: [{ x: epsilon, label: 'ε' }],
    }),
    buildGroupReport({ ...common, scores: baseScores, label: 'unconstrained' }),
    buildGroupReport({
      ...common,
      scores: decisions,
      decisions,
      label: `exponentiated gradient (${constraint})`,
      sweep: 0,
    }),
  ]
  return { groups: groups.names, members, weights, decisions, steps, baseline, exposition }
}

/** Options of {@link gridSearchReduction}. */
export interface GridSearchReductionOptions {
  readonly sensitive?: SensitiveSpec
  /** The constraint (default `demographicParity`). */
  readonly constraint?: ReductionConstraint
  /** Multipliers tried, from -bound to bound (default 21). */
  readonly gridSize?: number
  /**
   * The largest multiplier (default 0.5). A multiplier prices a unit of a group's gap against a unit of mean error, so a
   * small group's rows flip once it passes about P(group) / (1 - P(group)).
   */
  readonly bound?: number
  readonly estimator?: Estimator
  readonly dropSensitive?: boolean
  readonly key?: Key
}

/** One classifier of {@link gridSearchReduction}. */
export interface GridPoint {
  readonly lambda: number
  readonly error: number
  readonly violation: number
  readonly fitted: Fitted
}

/**
 * Fair classification by a grid of multipliers (fairlearn's `GridSearch`): for each multiplier, the best classifier for
 * that price on the gap between the non-reference groups and the reference, giving a family of deterministic
 * classifiers that trace the trade-off between error and violation. Exposition: a `curves` record of error against
 * violation over the grid.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - Constraint, grid size, bound, estimator and key.
 * @returns The classifiers with their error and violation, with the record.
 */
export function gridSearchReduction(
  data: FairnessData,
  options: GridSearchReductionOptions = {},
): Explained<{ readonly points: readonly GridPoint[] }> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'gridSearchReduction')
  const x =
    options.dropSensitive === true && groups.column !== undefined ? withoutColumns(data, [groups.column]).x : data.x
  const constraint = options.constraint ?? 'demographicParity'
  const m = moments(constraint, groups.index, labels, groups.names.length)
  const size = options.gridSize ?? 21
  const bound = options.bound ?? 0.5
  const estimator = options.estimator ?? logisticRegression()
  const key = options.key ?? makeKey(0)
  const reference = groupReportReference(groups)
  const cells = m.count / 2
  const points: GridPoint[] = Array.from({ length: size }, (_, s) => {
    const lam = size === 1 ? 0 : -bound + (2 * bound * s) / (size - 1)
    // Price every non-reference cell's gap at lam (positive side) or -lam (negative side).
    const lambda = new Array<number>(m.count).fill(0)
    for (let c = 0; c < cells; c++) {
      const g = constraint === 'demographicParity' ? c : Math.floor(c / 2)
      if (g === reference) continue
      if (lam >= 0) lambda[c] = lam
      else lambda[c + cells] = -lam
    }
    const r = bestResponse(estimator, x, labels, m, lambda, foldIn(key, s))
    return { lambda: lam, error: errorOf(labels, r.decisions), violation: violation(m, r.decisions), fitted: r.fitted }
  })
  const sorted = [...points].sort((a, b) => a.violation - b.violation)
  return {
    points,
    exposition: [
      curveSet({
        xLabel: 'constraint violation',
        yLabel: 'error',
        series: [{ name: 'grid of multipliers', x: sorted.map((p) => p.violation), y: sorted.map((p) => p.error) }],
      }),
    ],
  }
}
