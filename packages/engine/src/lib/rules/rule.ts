/**
 * The representation shared by the rule learners: a literal (a test on one feature), a rule (a conjunction of literals
 * with the class counts of the training rows it covers), a rule model (an ordered list, an unordered set voting, or a
 * table), prediction by first match, by voting and by default rule, the rule quality measures (support, confidence,
 * lift, Laplace accuracy, m-estimate, weighted relative accuracy, entropy, FOIL gain) and the significance of a rule's
 * class distribution, and the rule-set record that views draw.
 */

import { OmaiError } from '../base/index.ts'
import { type RuleSet as RuleSetRecord, ruleSet as ruleSetRecord } from '../exposition/index.ts'
import { chdtrc } from '../numerics/special/index.ts'
import { item } from '../tensor/index.ts'

/** A test on one feature: equal to a (category) value, at most a threshold, or above a threshold. */
export type Literal = { readonly feature: number; readonly op: '==' | '<=' | '>'; readonly value: number }

/** A conjunction of literals predicting a class, with the class counts of the training rows it covers. */
export type Rule = {
  readonly literals: Literal[]
  /** The index (into the model's classes) of the class the rule predicts. */
  readonly label: number
  /** Training rows covered, per class, in the context the rule was learned (after earlier rules for a list). */
  readonly counts: number[]
}

/**
 * A fitted rule model. A `list` predicts with the first rule that fires; a `set` sums the class counts of every rule
 * that fires (CN2's unordered voting); both fall back to the default counts when nothing fires.
 */
export type RuleModel = {
  readonly kind: 'list' | 'set'
  /** The class labels, ascending. */
  readonly classes: number[]
  readonly nFeatures: number
  readonly rules: Rule[]
  /** Class counts of the default rule (the rows no rule covers, or the whole training set for a set). */
  readonly defaultCounts: number[]
}

/**
 * Whether a row satisfies a literal.
 *
 * @param literal - The literal.
 * @param value - The row's value of the literal's feature.
 * @returns True when it holds (a missing value never does).
 * @example
 * ```ts
 * import { literalHolds } from 'omai-engine/rules'
 * [literalHolds({ feature: 0, op: '<=', value: 2 }, 1.5), literalHolds({ feature: 0, op: '>', value: 2 }, 1.5)] // => [true, false]
 * ```
 */
export function literalHolds(literal: Literal, value: number): boolean {
  if (Number.isNaN(value)) return false
  return literal.op === '=='
    ? value === literal.value
    : literal.op === '<='
      ? value <= literal.value
      : value > literal.value
}

/**
 * Whether each row of a row-major matrix satisfies every literal of a rule.
 *
 * @param literals - The conjunction (empty covers everything).
 * @param data - Row-major values.
 * @param d - The number of columns.
 * @returns 1 for covered rows, 0 otherwise.
 * @example
 * ```ts
 * import { coverageOf } from 'omai-engine/rules'
 * Array.from(coverageOf([{ feature: 1, op: '>', value: 0 }], Float64Array.of(0, 1, 0, -1), 2)) // => [1, 0]
 * ```
 */
export function coverageOf(literals: readonly Literal[], data: ArrayLike<number>, d: number): Uint8Array {
  const n = Math.floor(data.length / Math.max(1, d))
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    let ok = 1
    for (const l of literals) {
      if (!literalHolds(l, data[i * d + l.feature] ?? Number.NaN)) {
        ok = 0
        break
      }
    }
    out[i] = ok
  }
  return out
}

/**
 * The class counts a model assigns to each row: the first firing rule's counts (a list) or the sum of the firing
 * rules' (a set), the default counts when none fires.
 *
 * @param model - The model.
 * @param data - Row-major values.
 * @returns `n * k` counts, row-major.
 * @example
 * ```ts
 * import { predictCounts } from 'omai-engine/rules'
 * const model = { kind: 'list', classes: [0, 1], nFeatures: 1, rules: [{ literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [0, 3] }], defaultCounts: [4, 0] } as const
 * Array.from(predictCounts(model, Float64Array.of(0, 2))) // => [4, 0, 0, 3]
 * ```
 */
export function predictCounts(model: RuleModel, data: ArrayLike<number>): Float64Array {
  const k = model.classes.length
  const d = model.nFeatures
  const n = Math.floor(data.length / Math.max(1, d))
  const out = new Float64Array(n * k)
  const cover = model.rules.map((r) => coverageOf(r.literals, data, d))
  for (let i = 0; i < n; i++) {
    let fired = false
    for (let r = 0; r < model.rules.length; r++) {
      if (cover[r]?.[i] !== 1) continue
      fired = true
      for (let c = 0; c < k; c++) out[i * k + c] = (out[i * k + c] ?? 0) + (model.rules[r]?.counts[c] ?? 0)
      if (model.kind === 'list') break
    }
    if (!fired) for (let c = 0; c < k; c++) out[i * k + c] = model.defaultCounts[c] ?? 0
  }
  return out
}

/**
 * The index of the rule that decides each row of a list (the first that fires), a set's first firing rule, or -1 for
 * the default rule.
 *
 * @param model - The model.
 * @param data - Row-major values.
 * @returns One rule index per row.
 * @example
 * ```ts
 * import { firingRule } from 'omai-engine/rules'
 * const model = { kind: 'list', classes: [0, 1], nFeatures: 1, rules: [{ literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [0, 3] }], defaultCounts: [4, 0] } as const
 * firingRule(model, Float64Array.of(0, 2)) // => [-1, 0]
 * ```
 */
export function firingRule(model: RuleModel, data: ArrayLike<number>): number[] {
  const d = model.nFeatures
  const n = Math.floor(data.length / Math.max(1, d))
  const cover = model.rules.map((r) => coverageOf(r.literals, data, d))
  return Array.from({ length: n }, (_, i) => cover.findIndex((c) => c[i] === 1))
}

/**
 * Laplace-smoothed class probabilities from counts: `(c + 1) / (sum + k)`.
 *
 * @param counts - Class counts.
 * @returns Probabilities summing to one.
 * @example
 * ```ts
 * import { smoothedProbabilities } from 'omai-engine/rules'
 * smoothedProbabilities([3, 0]) // => [0.8, 0.2]
 * ```
 */
export function smoothedProbabilities(counts: ArrayLike<number>): number[] {
  const k = counts.length
  let total = 0
  for (let c = 0; c < k; c++) total += counts[c] ?? 0
  return Array.from({ length: k }, (_, c) => ((counts[c] ?? 0) + 1) / (total + k))
}

/** The rule quality measures a covering search can maximise. */
export type RuleHeuristic = 'entropy' | 'laplace' | 'accuracy' | 'm-estimate' | 'wracc'

/**
 * A rule's quality from the class counts of the rows it covers (higher is better): `entropy` is minus the entropy of
 * the covered distribution in bits (CN2's original), `laplace` the Laplace accuracy `(p + 1) / (n + k)` of the target
 * class, `accuracy` the precision `p / n`, `m-estimate` `(p + m * prior) / (n + m)`, and `wracc` weighted relative
 * accuracy `(n / N) (p / n - P / N)`.
 *
 * @param measure - Which measure.
 * @param covered - Class counts among the covered rows.
 * @param target - The class the rule predicts.
 * @param prior - Class counts among all rows in play.
 * @param m - The m of the m-estimate (default 2).
 * @returns The quality, or -Infinity for a rule covering nothing.
 * @example
 * ```ts
 * import { ruleQuality } from 'omai-engine/rules'
 * [ruleQuality('laplace', [8, 0], 0, [10, 10]), ruleQuality('accuracy', [8, 2], 0, [10, 10])] // => [0.9, 0.8]
 * ```
 */
export function ruleQuality(
  measure: RuleHeuristic,
  covered: ArrayLike<number>,
  target: number,
  prior: ArrayLike<number>,
  m = 2,
): number {
  const k = covered.length
  let n = 0
  let N = 0
  for (let c = 0; c < k; c++) {
    n += covered[c] ?? 0
    N += prior[c] ?? 0
  }
  if (n <= 0) return Number.NEGATIVE_INFINITY
  const p = covered[target] ?? 0
  const P = prior[target] ?? 0
  switch (measure) {
    case 'entropy': {
      let h = 0
      for (let c = 0; c < k; c++) {
        const q = (covered[c] ?? 0) / n
        if (q > 0) h -= q * Math.log2(q)
      }
      return -h
    }
    case 'laplace':
      return (p + 1) / (n + k)
    case 'accuracy':
      return p / n
    case 'm-estimate':
      return (p + (m * P) / N) / (n + m)
    case 'wracc':
      return (n / N) * (p / n - P / N)
  }
}

/**
 * The likelihood-ratio statistic of CN2: `2 sum f_i ln(f_i / e_i)` of the covered class counts `f` against the counts
 * `e` expected if the rule covered a random sample of the rows in play. It is chi-squared with k - 1 degrees of freedom
 * under the null of no association between the rule and the class.
 *
 * @param covered - Class counts among the covered rows.
 * @param prior - Class counts among all rows in play.
 * @returns The statistic.
 * @example
 * ```ts
 * import { likelihoodRatio } from 'omai-engine/rules'
 * likelihoodRatio([10, 0], [10, 10]) // => 13.862943611198906
 * ```
 */
export function likelihoodRatio(covered: ArrayLike<number>, prior: ArrayLike<number>): number {
  let n = 0
  let N = 0
  for (let c = 0; c < covered.length; c++) {
    n += covered[c] ?? 0
    N += prior[c] ?? 0
  }
  let lrs = 0
  for (let c = 0; c < covered.length; c++) {
    const f = covered[c] ?? 0
    const e = (n * (prior[c] ?? 0)) / N
    if (f > 0 && e > 0) lrs += f * Math.log(f / e)
  }
  return 2 * lrs
}

/**
 * The p-value of a rule's class distribution under CN2's significance test.
 *
 * @param covered - Class counts among the covered rows.
 * @param prior - Class counts among all rows in play.
 * @returns The probability of a statistic at least as large under chance (1 for one class).
 * @example
 * ```ts
 * import { ruleSignificance } from 'omai-engine/rules'
 * ruleSignificance([10, 0], [10, 10]) < 0.001 // => true
 * ```
 */
export function ruleSignificance(covered: ArrayLike<number>, prior: ArrayLike<number>): number {
  if (covered.length < 2) return 1
  return item(chdtrc(covered.length - 1, Math.max(0, likelihoodRatio(covered, prior))))
}

/** Support, confidence, lift and friends of one rule against a labelled data set. */
export type RuleStatistics = {
  /** Rows covered. */
  readonly covered: number
  /** Covered rows of the rule's class. */
  readonly correct: number
  /** Covered rows over all rows. */
  readonly support: number
  /** Correct over covered. */
  readonly confidence: number
  /** Confidence over the class's share of all rows. */
  readonly lift: number
  /** `(correct + 1) / (covered + k)`. */
  readonly laplace: number
  /** Weighted relative accuracy. */
  readonly wracc: number
}

/**
 * The quality statistics of a rule on data: how many rows it covers and gets right, its support, confidence, lift,
 * Laplace accuracy and weighted relative accuracy.
 *
 * @param rule - The rule.
 * @param data - Row-major values.
 * @param d - The number of columns.
 * @param labels - The class index of each row.
 * @param k - The number of classes.
 * @returns The statistics.
 * @example
 * ```ts
 * import { ruleStatistics } from 'omai-engine/rules'
 * const s = ruleStatistics({ literals: [{ feature: 0, op: '>', value: 0 }], label: 1, counts: [] }, Float64Array.of(0, 1, 2, -1), 1, [0, 1, 1, 0], 2)
 * [s.covered, s.correct, s.confidence, s.lift] // => [2, 2, 1, 2]
 * ```
 */
export function ruleStatistics(
  rule: Pick<Rule, 'literals' | 'label'>,
  data: ArrayLike<number>,
  d: number,
  labels: ArrayLike<number>,
  k: number,
): RuleStatistics {
  const cover = coverageOf(rule.literals, data, d)
  const n = labels.length
  if (cover.length !== n) throw new OmaiError('shape', 'ruleStatistics: labels and rows differ in length')
  let covered = 0
  let correct = 0
  let classRows = 0
  for (let i = 0; i < n; i++) {
    if (labels[i] === rule.label) classRows++
    if (cover[i] === 1) {
      covered++
      if (labels[i] === rule.label) correct++
    }
  }
  const confidence = covered === 0 ? 0 : correct / covered
  const share = classRows / Math.max(1, n)
  return {
    covered,
    correct,
    support: covered / Math.max(1, n),
    confidence,
    lift: share === 0 ? 0 : confidence / share,
    laplace: (correct + 1) / (covered + k),
    wracc: (covered / Math.max(1, n)) * (confidence - share),
  }
}

/**
 * A literal as text, `x0 <= 2.5`, `colour = 1`.
 *
 * @param literal - The literal.
 * @param features - Feature names (default `x0`, `x1`, ...).
 * @returns The text.
 * @example
 * ```ts
 * import { literalText } from 'omai-engine/rules'
 * literalText({ feature: 1, op: '<=', value: 2.5 }, ['age', 'height']) // => 'height <= 2.5'
 * ```
 */
export function literalText(literal: Literal, features: readonly string[] = []): string {
  const name = features[literal.feature] ?? `x${literal.feature}`
  return `${name} ${literal.op === '==' ? '=' : literal.op} ${Number(literal.value.toPrecision(6))}`
}

/**
 * A rule as an if-then line: `IF a > 1 AND b = 0 THEN class 1`, with `IF true` for the empty rule.
 *
 * @param rule - The rule.
 * @param classes - Class labels or names indexed by the rule's label.
 * @param features - Feature names.
 * @returns The text.
 * @example
 * ```ts
 * import { ruleLine } from 'omai-engine/rules'
 * ruleLine({ literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [] }, ['no', 'yes'], ['age']) // => 'IF age > 1 THEN yes'
 * ```
 */
export function ruleLine(rule: Rule, classes: readonly (string | number)[], features: readonly string[] = []): string {
  const body = rule.literals.length === 0 ? 'true' : rule.literals.map((l) => literalText(l, features)).join(' AND ')
  return `IF ${body} THEN ${String(classes[rule.label] ?? rule.label)}`
}

/**
 * The model as text, one rule per line and the default last.
 *
 * @param model - The model.
 * @param features - Feature names.
 * @param classNames - Class names (default the labels).
 * @returns The text.
 * @example
 * ```ts
 * import { modelText } from 'omai-engine/rules'
 * const model = { kind: 'list', classes: [0, 1], nFeatures: 1, rules: [{ literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [0, 3] }], defaultCounts: [4, 0] } as const
 * modelText(model, ['age']) // => 'IF age > 1 THEN 1\nELSE 0'
 * ```
 */
export function modelText(model: RuleModel, features: readonly string[] = [], classNames?: readonly string[]): string {
  const names = classNames ?? model.classes
  const lines = model.rules.map((r) => ruleLine(r, names, features))
  const fallback = argmaxOf(model.defaultCounts)
  lines.push(`ELSE ${String(names[fallback] ?? fallback)}`)
  return lines.join('\n')
}

/**
 * The index of the largest entry (the first on ties).
 *
 * @param values - The values.
 * @returns The index (0 for an empty list).
 * @example
 * ```ts
 * import { argmaxOf } from 'omai-engine/rules'
 * argmaxOf([1, 3, 3]) // => 1
 * ```
 */
export function argmaxOf(values: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < values.length; i++)
    if ((values[i] ?? Number.NEGATIVE_INFINITY) > (values[best] ?? Number.NEGATIVE_INFINITY)) best = i
  return best
}

/**
 * The model as a rule-set record for the views: each rule with its precision (Laplace-free confidence among the rows it
 * covered when learned) and coverage (the share of the training rows).
 *
 * @param model - The model.
 * @param features - Feature names.
 * @param classNames - Class names.
 * @param method - The learner's name.
 * @returns The record.
 * @example
 * ```ts
 * import { modelRecord } from 'omai-engine/rules'
 * const model = { kind: 'list', classes: [0, 1], nFeatures: 1, rules: [{ literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [0, 3] }], defaultCounts: [4, 0] } as const
 * modelRecord(model, ['age'], ['no', 'yes'], 'demo').rules[0]?.precision // => 1
 * ```
 */
export function modelRecord(
  model: RuleModel,
  features: readonly string[],
  classNames: readonly string[],
  method: string,
): RuleSetRecord {
  const total =
    model.defaultCounts.reduce((s, v) => s + v, 0) +
    model.rules.reduce((s, r) => s + r.counts.reduce((a, v) => a + v, 0), 0)
  return ruleSetRecord({
    method,
    features,
    output: `class (${classNames.join(' / ')})`,
    rules: model.rules.map((r) => {
      const n = r.counts.reduce((s, v) => s + v, 0)
      return {
        clauses: r.literals.map((l) => ({
          feature: l.feature,
          op: l.op === '==' ? ('=' as const) : l.op,
          threshold: l.value,
        })),
        weight: r.label,
        precision: n === 0 ? 0 : (r.counts[r.label] ?? 0) / n,
        coverage: total === 0 ? 0 : n / total,
      }
    }),
  })
}
