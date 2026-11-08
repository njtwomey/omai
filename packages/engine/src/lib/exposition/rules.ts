/**
 * The record of rule-based explanations: a list of conjunctive rules (an anchor, a RuleFit model's terms, a surrogate's
 * rules), each with the statistics that qualify it (precision and coverage, coefficient, support and importance).
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'

/** One condition of a rule: feature `op` value, or the feature within [lower, upper]. */
export interface RuleClause {
  readonly feature: number
  readonly op: '<=' | '>' | 'in' | '='
  readonly threshold?: number
  readonly lower?: number
  readonly upper?: number
}

/** A rule and its statistics. */
export interface RuleEntry {
  /** The conditions, all of which hold (empty for a linear term). */
  readonly clauses: readonly RuleClause[]
  /** A linear term's feature instead of a rule. */
  readonly linear?: number
  /** The rule's prediction or coefficient. */
  readonly weight?: number
  /** Anchors: the precision (agreement with the model's prediction) and coverage (share of data covered). */
  readonly precision?: number
  readonly coverage?: number
  /** RuleFit: the share of rows covered and the importance. */
  readonly support?: number
  readonly importance?: number
}

/** A list of rules explaining a model, globally (RuleFit) or around one row (anchors). */
export interface RuleSet {
  readonly kind: 'rules'
  readonly method: string
  readonly features: readonly string[]
  /** What the rules predict or explain (`prediction`, `P(malignant)`). */
  readonly output: string
  readonly rules: readonly RuleEntry[]
  /** The intercept of an additive rule model. */
  readonly intercept?: number
  /** The explained row's values, for anchors. */
  readonly row?: readonly number[]
}

/**
 * A rule-set record, with its features checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `value` when a clause or linear term names a feature that is not in `features`.
 * @example
 * ```ts
 * import { ruleSet } from 'omai-engine/exposition'
 * const r = ruleSet({ method: 'anchors', features: ['age'], output: 'P(y)', rules: [{ clauses: [{ feature: 0, op: '>', threshold: 40 }], precision: 0.97, coverage: 0.3 }] })
 * r.rules[0].precision // => 0.97
 * ```
 */
export function ruleSet(record: Omit<RuleSet, 'kind'>): RuleSet {
  const d = record.features.length
  const named = (j: number): void => {
    if (!(Number.isInteger(j) && j >= 0 && j < d)) throw new OmaiError('value', `rules: feature ${j} is not named`)
  }
  for (const r of record.rules) {
    for (const c of r.clauses) named(c.feature)
    if (r.linear !== undefined) named(r.linear)
  }
  return seal({ kind: 'rules', ...record })
}

/**
 * A rule as text: its clauses joined by "and", or the linear term's feature name.
 *
 * @param rule - The rule.
 * @param features - The feature names.
 * @param format - How numbers are written (default 3 significant digits).
 * @returns The text.
 * @example
 * ```ts
 * import { ruleText } from 'omai-engine/exposition'
 * ruleText({ clauses: [{ feature: 0, op: '>', threshold: 40 }, { feature: 1, op: 'in', lower: 1, upper: 2 }] }, ['age', 'kids']) // => 'age > 40 and 1 ≤ kids ≤ 2'
 * ```
 */
export function ruleText(
  rule: RuleEntry,
  features: readonly string[],
  format: (v: number) => string = (v) => String(Number(v.toPrecision(3))),
): string {
  if (rule.clauses.length === 0)
    return rule.linear === undefined ? '(always)' : (features[rule.linear] ?? `x${rule.linear}`)
  return rule.clauses
    .map((c) => {
      const name = features[c.feature] ?? `x${c.feature}`
      if (c.op === 'in')
        return `${format(c.lower ?? Number.NEGATIVE_INFINITY)} ≤ ${name} ≤ ${format(c.upper ?? Number.POSITIVE_INFINITY)}`
      return `${name} ${c.op === '<=' ? '≤' : c.op} ${format(c.threshold ?? 0)}`
    })
    .join(' and ')
}
