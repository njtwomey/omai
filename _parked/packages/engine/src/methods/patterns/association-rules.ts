/**
 * Association rules from frequent itemsets, as in mlxtend's `association_rules`: every rule "X implies Y" between
 * disjoint frequent itemsets whose confidence passes a threshold, with its support, confidence, lift, leverage and
 * conviction, the measures of how much more often Y appears with X than without it.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type CoefficientTable, coefficientTable, type Explained } from '../../lib/exposition/index.ts'
import type { Itemset } from './itemsets.ts'

/** An association rule: antecedent implies consequent, with its measures. */
export interface AssociationRule {
  readonly antecedent: readonly string[]
  readonly consequent: readonly string[]
  /** The share of transactions holding both sides. */
  readonly support: number
  /** support(both) / support(antecedent). */
  readonly confidence: number
  /** confidence / support(consequent): above 1, the sides appear together more than chance. */
  readonly lift: number
  /** support(both) - support(antecedent) support(consequent). */
  readonly leverage: number
  /** (1 - support(consequent)) / (1 - confidence) (Infinity for exact rules). */
  readonly conviction: number
}

/** Options of {@link associationRules}. */
export interface RuleOptions {
  /** The smallest confidence kept (default 0.6). */
  readonly minConfidence?: number
  /** The smallest lift kept (default 0). */
  readonly minLift?: number
  /** Rules shown in the record, by lift (default 15). */
  readonly shown?: number
}

const keyOf = (items: readonly string[]): string => [...items].sort().join('\u0001')

/**
 * Every rule X ⇒ Y with X and Y non-empty and disjoint, X ∪ Y frequent, and confidence at least the threshold, with
 * support, confidence, lift, leverage and conviction (each side's support from the itemsets, which are closed under
 * subsets when they come from Apriori or FP-growth).
 *
 * @param itemsets - Frequent itemsets with their supports.
 * @param options - Minimum confidence and lift, and how many rules the record shows.
 * @returns The rules by lift, with a `coefficientTable` record of the top rules' lifts.
 * @throws MdmlError `value` when a subset of a frequent itemset is missing.
 */
export function associationRules(
  itemsets: readonly Itemset[],
  options: RuleOptions = {},
): Explained<{ readonly rules: readonly AssociationRule[] }> {
  const minConfidence = options.minConfidence ?? 0.6
  const minLift = options.minLift ?? 0
  const support = new Map(itemsets.map((s) => [keyOf(s.items), s.support]))
  const of = (items: readonly string[]): number => {
    const s = support.get(keyOf(items))
    if (s === undefined) throw new MdmlError('value', `associationRules: no support for {${items.join(', ')}}`)
    return s
  }
  const rules: AssociationRule[] = []
  for (const set of itemsets) {
    const n = set.items.length
    if (n < 2) continue
    for (let mask = 1; mask < (1 << n) - 1; mask++) {
      const antecedent = set.items.filter((_, i) => (mask >> i) & 1)
      const consequent = set.items.filter((_, i) => !((mask >> i) & 1))
      const sa = of(antecedent)
      const sc = of(consequent)
      const confidence = set.support / sa
      const lift = confidence / sc
      if (confidence < minConfidence || lift < minLift) continue
      rules.push({
        antecedent,
        consequent,
        support: set.support,
        confidence,
        lift,
        leverage: set.support - sa * sc,
        conviction: confidence >= 1 ? Number.POSITIVE_INFINITY : (1 - sc) / (1 - confidence),
      })
    }
  }
  rules.sort((a, b) => b.lift - a.lift || b.confidence - a.confidence || b.support - a.support)
  const top = rules.slice(0, options.shown ?? 15)
  const fmt = (x: number): string => x.toFixed(2)
  const record: CoefficientTable = coefficientTable({
    names: top.map(
      (r) => `{${r.antecedent.join(', ')}} ⇒ {${r.consequent.join(', ')}}  s ${fmt(r.support)} c ${fmt(r.confidence)}`,
    ),
    estimates: top.map((r) => r.lift),
  })
  return { rules, exposition: [record] }
}
