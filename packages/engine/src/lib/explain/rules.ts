/**
 * Rule-based explanations: the conjunctive rules a tree's paths define, RuleFit's sparse linear model over rules and
 * features (Friedman and Popescu 2008) with each rule's support and importance, and alibi's linearity measure of how
 * far a model departs from linear around a point.
 */
import { MdmlError } from '../base/index.ts'
import { keyFor } from '../learn/_shared.ts'
import type { LearnData } from '../learn/estimator.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import { lassoCoordinateDescent } from '../optim/index.ts'
import { foldIn, type Key, subsample, uniform } from '../random/index.ts'
import { asTensor, fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import { type PredictFunction, type TreeModel, treePredict } from './shapley.ts'
import { surrogateTree } from './surrogates.ts'

/** One condition of a rule: feature j <= threshold, or > threshold. */
export interface RuleCondition {
  readonly feature: number
  readonly op: '<=' | '>'
  readonly threshold: number
}

/** A conjunction of conditions. */
export type Rule = readonly RuleCondition[]

/**
 * The rules of a tree (RuleFit's rule generation): one per non-root node, the conjunction of the splits on the path to
 * it, with redundant conditions on the same feature and side merged to the tightest.
 *
 * @param tree - The tree.
 * @returns The rules, in node order.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const stump = { childrenLeft: [1, -1, -1], childrenRight: [2, -1, -1], feature: [0, -2, -2], threshold: [0.5, 0, 0], value: [0.5, 0, 1], cover: [2, 1, 1] }
 * explain.treeRules(stump) // => [[{ feature: 0, op: '<=', threshold: 0.5 }], [{ feature: 0, op: '>', threshold: 0.5 }]]
 * ```
 */
export function treeRules(tree: TreeModel): Rule[] {
  const out: Rule[] = []
  const walk = (node: number, path: RuleCondition[], depth: number): void => {
    if (depth > tree.value.length) throw new MdmlError('value', 'treeRules: the tree has a cycle')
    const left = tree.childrenLeft[node] ?? -1
    if (left < 0) return
    const right = tree.childrenRight[node] ?? -1
    const f = tree.feature[node] ?? 0
    const t = tree.threshold[node] ?? 0
    for (const [child, op] of [
      [left, '<='],
      [right, '>'],
    ] as const) {
      const next = simplify([...path, { feature: f, op, threshold: t }])
      out.push(next)
      walk(child, next, depth + 1)
    }
  }
  walk(0, [], 0)
  return out
}

function simplify(rule: readonly RuleCondition[]): RuleCondition[] {
  const tight = new Map<string, RuleCondition>()
  for (const c of rule) {
    const k = `${c.feature}${c.op}`
    const prev = tight.get(k)
    if (prev === undefined || (c.op === '<=' ? c.threshold < prev.threshold : c.threshold > prev.threshold))
      tight.set(k, c)
  }
  return [...tight.values()].sort((a, b) => a.feature - b.feature || (a.op === '<=' ? -1 : 1))
}

/**
 * Whether each row satisfies a rule.
 *
 * @param rule - The rule.
 * @param x - Rows `[n, d]`.
 * @returns 1 or 0 per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * explain.ruleMask([{ feature: 0, op: '>', threshold: 1 }], [[0], [2]]) // => [0, 1]
 * ```
 */
export function ruleMask(rule: Rule, x: TensorLike): number[] {
  const m = matrixView(x, 'ruleMask')
  return Array.from({ length: m.n }, (_, i) =>
    rule.every((c) => {
      const v = m.data[i * m.d + c.feature] ?? 0
      return c.op === '<=' ? v <= c.threshold : v > c.threshold
    })
      ? 1
      : 0,
  )
}

/** Options of {@link ruleFit}. */
export interface RuleFitOptions {
  /** Trees grown (default 30), each on a subsample (default half the rows) of the current residuals. */
  readonly nTrees?: number
  readonly subsample?: number
  /** The trees' largest depth (default 3). */
  readonly maxDepth?: number
  /** The lasso's L1 weight on the standardised design (default 0.01 times the rows). */
  readonly lambda?: number
  /** Include the (winsorised) features as linear terms (default true). */
  readonly linear?: boolean
  readonly key?: Key
}

/** A RuleFit term: a rule or a linear feature, with its coefficient, support and importance. */
export interface RuleFitTerm {
  /** The rule, or undefined for a linear term. */
  readonly rule?: Rule
  /** The feature of a linear term. */
  readonly feature?: number
  readonly coefficient: number
  /** The share of rows the rule covers (1 for a linear term). */
  readonly support: number
  /** Friedman and Popescu's importance: |coefficient| sqrt(s (1 - s)) for a rule, |coefficient| std for a feature. */
  readonly importance: number
}

/** A RuleFit model. */
export interface RuleFitModel {
  readonly intercept: number
  /** The terms with non-zero coefficients, by importance. */
  readonly terms: readonly RuleFitTerm[]
  /** The winsorising bounds of each feature's linear term. */
  readonly bounds: readonly (readonly [number, number])[]
}

/**
 * RuleFit (Friedman and Popescu 2008) for regression: generate rules from an ensemble of shallow trees grown on
 * subsamples of the boosting residuals, add the winsorised features as linear terms, and fit a lasso on the
 * standardised rule indicators and features; the surviving terms, ranked by importance, are the explanation.
 *
 * @param data - Rows and targets.
 * @param options - nTrees, subsample, maxDepth, lambda, linear and key.
 * @returns The model.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 10, (i % 7) / 7])
 * const y = x.map(([a]) => (a > 3 ? 2 : 0))
 * const r = explain.ruleFit({ x, y }, { nTrees: 5, linear: false })
 * r.terms[0].rule.some((c) => c.feature === 0 && Math.abs(c.threshold - 3) < 0.1) // => true
 * ```
 */
export function ruleFit(data: LearnData, options: RuleFitOptions = {}): RuleFitModel {
  const m = matrixView(data.x, 'ruleFit')
  if (data.y === undefined) throw new MdmlError('value', 'ruleFit: needs targets')
  const y = targetVector(data.y, 'ruleFit', m.n)
  const key = options.key ?? keyFor({}, 'ruleFit')
  const nTrees = options.nTrees ?? 30
  const size = Math.max(2, Math.floor((options.subsample ?? 0.5) * m.n))
  const mean = y.reduce((a, b) => a + b, 0) / m.n
  const f = new Float64Array(m.n).fill(mean)
  const seen = new Set<string>()
  const rules: Rule[] = []
  for (let t = 0; t < nTrees; t++) {
    const idx = toFlat(subsample(foldIn(key, t), m.n, size))
    const xs = Float64Array.from(
      { length: size * m.d },
      (_, p) => m.data[(idx[Math.floor(p / m.d)] ?? 0) * m.d + (p % m.d)] ?? 0,
    )
    const r = idx.map((i) => (y[i] ?? 0) - (f[i] ?? 0))
    const depth = Math.max(1, (options.maxDepth ?? 3) - (t % 2))
    const tree = surrogateTree(fromStorage(xs, [size, m.d], 'float64'), r, { maxDepth: depth, minSamplesLeaf: 2 })
    const all = fromStorage(m.data, [m.n, m.d], 'float64')
    const pred = toFlat(treePredict(tree, all))
    for (let i = 0; i < m.n; i++) f[i] = (f[i] ?? 0) + 0.1 * (pred[i] ?? 0)
    for (const rule of treeRules(tree)) {
      const k = JSON.stringify(rule)
      if (!seen.has(k)) {
        seen.add(k)
        rules.push(rule)
      }
    }
  }
  const all = fromStorage(m.data, [m.n, m.d], 'float64')
  const columns: { kind: 'rule' | 'linear'; rule?: Rule; feature?: number; values: number[] }[] = []
  for (const rule of rules) {
    const mask = ruleMask(rule, all)
    const s = mask.reduce((a, b) => a + b, 0) / m.n
    if (s > 0 && s < 1) columns.push({ kind: 'rule', rule, values: mask })
  }
  const bounds = Array.from({ length: m.d }, (_, j) => {
    const col = Float64Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0).sort()
    return [col[Math.floor(0.025 * (m.n - 1))] ?? 0, col[Math.ceil(0.975 * (m.n - 1))] ?? 0] as [number, number]
  })
  if (options.linear ?? true)
    for (let j = 0; j < m.d; j++) {
      const [lo, hi] = bounds[j] ?? [0, 0]
      columns.push({
        kind: 'linear',
        feature: j,
        values: Array.from({ length: m.n }, (_, i) => Math.min(hi, Math.max(lo, m.data[i * m.d + j] ?? 0))),
      })
    }
  if (columns.length === 0) return { intercept: mean, terms: [], bounds }
  // Standardise the columns, fit the lasso to the centred target, and map back.
  const stats = columns.map((c) => {
    const mu = c.values.reduce((a, b) => a + b, 0) / m.n
    const sd = Math.sqrt(c.values.reduce((a, b) => a + (b - mu) ** 2, 0) / m.n) || 1
    return { mu, sd }
  })
  const k = columns.length
  const design = new Float64Array(m.n * k)
  columns.forEach((c, j) => {
    const { mu, sd } = stats[j] ?? { mu: 0, sd: 1 }
    for (let i = 0; i < m.n; i++) design[i * k + j] = ((c.values[i] ?? 0) - mu) / sd
  })
  const lambda = options.lambda ?? 0.01 * m.n
  const beta = toFlat(
    lassoCoordinateDescent(
      fromStorage(design, [m.n, k], 'float64'),
      Float64Array.from(y, (v) => v - mean),
      lambda,
      { maxIterations: 2000 },
    ).coef,
  )
  let intercept = mean
  const terms: RuleFitTerm[] = []
  columns.forEach((c, j) => {
    const b = beta[j] ?? 0
    if (Math.abs(b) < 1e-12) return
    const { mu, sd } = stats[j] ?? { mu: 0, sd: 1 }
    const coefficient = b / sd
    intercept -= coefficient * mu
    const support = c.kind === 'rule' ? c.values.reduce((a, v) => a + v, 0) / m.n : 1
    terms.push({
      ...(c.rule === undefined ? {} : { rule: c.rule }),
      ...(c.feature === undefined ? {} : { feature: c.feature }),
      coefficient,
      support,
      importance:
        c.kind === 'rule' ? Math.abs(coefficient) * Math.sqrt(support * (1 - support)) : Math.abs(coefficient) * sd,
    })
  })
  terms.sort((a, b) => b.importance - a.importance)
  return { intercept, terms, bounds }
}

/**
 * A RuleFit model's predictions: the intercept plus each term's coefficient times its rule indicator or winsorised
 * feature.
 *
 * @param model - The model.
 * @param x - Rows `[n, d]`.
 * @returns One prediction per row.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const model = { intercept: 1, terms: [{ rule: [{ feature: 0, op: '>', threshold: 0 }], coefficient: 2, support: 0.5, importance: 1 }], bounds: [[-9, 9]] }
 * explain.ruleFitPredict(model, [[-1], [1]]) // => [1, 3]
 * ```
 */
export function ruleFitPredict(model: RuleFitModel, x: TensorLike): number[] {
  const m = matrixView(x, 'ruleFitPredict')
  const out = new Array<number>(m.n).fill(model.intercept)
  for (const t of model.terms) {
    if (t.rule !== undefined) {
      const mask = ruleMask(t.rule, x)
      for (let i = 0; i < m.n; i++) out[i] = (out[i] ?? 0) + t.coefficient * (mask[i] ?? 0)
    } else if (t.feature !== undefined) {
      const [lo, hi] = model.bounds[t.feature] ?? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
      for (let i = 0; i < m.n; i++)
        out[i] = (out[i] ?? 0) + t.coefficient * Math.min(hi, Math.max(lo, m.data[i * m.d + t.feature] ?? 0))
    }
  }
  return out
}

/** Options of {@link linearityMeasure}. */
export interface LinearityOptions {
  /** The half-width of the box sampled around x, per feature in its scale (default 0.04). */
  readonly epsilon?: number
  /** Feature scales (default ones). */
  readonly scales?: readonly number[]
  /** Points sampled (default 50). */
  readonly nSamples?: number
  readonly key?: Key
}

/**
 * The linearity measure (Oberst et al. 2019; alibi's linearity_measure): how far a model departs from linear around a
 * point, the distance between the model's output at the mean of nearby points and the mean of its outputs there,
 * |f(mean z) - mean f(z)|, for points z drawn uniformly from a small box around x (summed over outputs). It is 0 for a
 * model linear on the box.
 *
 * @param predict - The model.
 * @param x - The point `[d]`.
 * @param options - epsilon, scales, nSamples and key.
 * @returns The measure (non-negative).
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { tensor } from 'mdml-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 3 * x.data[i] + 1))
 * explain.linearityMeasure(linear, [2]) < 1e-12 // => true
 * ```
 */
export function linearityMeasure(predict: PredictFunction, x: TensorLike, options: LinearityOptions = {}): number {
  const xv = toFlat(asTensor(x))
  const d = xv.length
  const n = options.nSamples ?? 50
  const eps = options.epsilon ?? 0.04
  const u = toFlat(uniform(options.key ?? keyFor({}, 'linearityMeasure'), [n, d]))
  const z = Float64Array.from(
    { length: n * d },
    (_, p) => (xv[p % d] ?? 0) + (2 * (u[p] ?? 0) - 1) * eps * (options.scales?.[p % d] ?? 1),
  )
  const mean = Float64Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += z[i * d + j] ?? 0
    return s / n
  })
  const out = predict(fromStorage(z, [n, d], 'float64'))
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const ov = toFlat(out)
  const atMean = toFlat(predict(fromStorage(mean, [1, d], 'float64')))
  let total = 0
  for (let c = 0; c < k; c++) {
    let s = 0
    for (let i = 0; i < n; i++) s += (ov[i * k + c] ?? 0) / n
    total += Math.abs((atMean[c] ?? 0) - s)
  }
  return total
}
