/**
 * Association rules from frequent itemsets: every split of a frequent itemset into an antecedent and a consequent,
 * with the interestingness measures of mlxtend (support, confidence, lift, leverage, conviction, Zhang's metric) from
 * the three counts that define them, filtered and ranked by any one of them.
 */

import { MdmlError } from '../base/index.ts'
import type { FrequentItemset } from './itemsets.ts'

/** The counts that define a rule's measures. */
export interface RuleCounts {
  /** The number of baskets. */
  readonly n: number
  /** Baskets containing the antecedent and the consequent. */
  readonly both: number
  /** Baskets containing the antecedent. */
  readonly antecedent: number
  /** Baskets containing the consequent. */
  readonly consequent: number
}

/** A rule's interestingness measures. */
export interface RuleMeasures {
  readonly antecedentSupport: number
  readonly consequentSupport: number
  readonly support: number
  readonly confidence: number
  /** Confidence over the consequent's support: 1 when the two are independent. */
  readonly lift: number
  /** Support minus the support expected under independence (Piatetsky-Shapiro). */
  readonly leverage: number
  /** How much more often the rule would be wrong if antecedent and consequent were independent; infinite at confidence 1. */
  readonly conviction: number
  /** Zhang's metric, in [-1, 1]: leverage scaled by its largest possible magnitude. */
  readonly zhangs: number
}

/**
 * The measures of a rule from its counts (mlxtend's definitions).
 *
 * @param counts - The basket counts.
 * @returns The measures.
 * @throws MdmlError with code `value` when `n` is not positive or a count exceeds its bounds.
 * @example
 * ```ts
 * import { ruleMeasures } from 'mdml-engine/patterns'
 * const m = ruleMeasures({ n: 10, both: 3, antecedent: 4, consequent: 5 })
 * [m.support, m.confidence, m.lift, m.leverage] // => [0.3, 0.75, 1.5, 0.1]
 * ```
 */
export function ruleMeasures(counts: RuleCounts): RuleMeasures {
  const { n, both, antecedent, consequent } = counts
  if (!(n > 0) || both > antecedent || both > consequent || both < 0) {
    throw new MdmlError('value', 'ruleMeasures: counts are inconsistent')
  }
  const sA = antecedent / n
  const sC = consequent / n
  const sAC = both / n
  const confidence = antecedent === 0 ? 0 : both / antecedent
  const leverage = sAC - sA * sC
  const denominator = Math.max(sAC * (1 - sA), sA * (sC - sAC))
  return {
    antecedentSupport: sA,
    consequentSupport: sC,
    support: sAC,
    confidence,
    lift: sC === 0 ? 0 : confidence / sC,
    leverage,
    conviction: confidence === 1 ? Number.POSITIVE_INFINITY : (1 - sC) / (1 - confidence),
    zhangs: denominator === 0 ? 0 : leverage / denominator,
  }
}

/** An association rule `antecedent => consequent` with its counts and measures. */
export interface AssociationRule extends RuleMeasures {
  readonly antecedent: readonly number[]
  readonly consequent: readonly number[]
  /** Baskets containing both sides. */
  readonly count: number
}

/** The measures a set of rules can be filtered and ranked by. */
export type RuleMeasure = 'support' | 'confidence' | 'lift' | 'leverage' | 'conviction' | 'zhangs'

/** Options of {@link associationRules}. */
export interface AssociationRuleOptions {
  /** The measure the threshold applies to and the ranking uses (default confidence). */
  readonly metric?: RuleMeasure
  /** The smallest value of the metric (default 0.8 for confidence, 0 otherwise). */
  readonly minThreshold?: number
  /** Only rules with at most this many items in the consequent (default no limit). */
  readonly maxConsequent?: number
}

/**
 * Association rules (Agrawal, Imielinski and Swami 1993) from a complete list of frequent itemsets and the number of
 * baskets: each frequent itemset of two or more items is split every way into antecedent and consequent, the measures
 * are computed from the counts of the three itemsets (all frequent, by the downward closure), and the rules whose
 * metric meets the threshold are returned, best first (ties in itemset order).
 *
 * @param itemsets - Every frequent itemset with its count (from any miner, with no `maxLength` cut of the subsets).
 * @param n - The number of baskets.
 * @param options - The metric, its threshold and a limit on the consequent's size.
 * @returns The rules, ranked by the metric.
 * @throws MdmlError with code `value` when a subset of a frequent itemset is missing from the list.
 * @example
 * ```ts
 * import { apriori, associationRules } from 'mdml-engine/patterns'
 * const tx = [[0, 1], [0, 1], [0, 1], [0, 2], [1]]
 * const rules = associationRules(apriori(tx, { minSupport: 0.4 }), tx.length, { minThreshold: 0.7 })
 * rules.map((r) => `${r.antecedent}=>${r.consequent}:${r.confidence}`) // => ['0=>1:0.75', '1=>0:0.75']
 * ```
 */
export function associationRules(
  itemsets: readonly FrequentItemset[],
  n: number,
  options: AssociationRuleOptions = {},
): AssociationRule[] {
  const metric = options.metric ?? 'confidence'
  const threshold = options.minThreshold ?? (metric === 'confidence' ? 0.8 : 0)
  const maxConsequent = options.maxConsequent ?? Number.POSITIVE_INFINITY
  const counts = new Map(itemsets.map((s) => [s.items.join(','), s.count]))
  const out: AssociationRule[] = []
  for (const s of itemsets) {
    const m = s.items.length
    if (m < 2) continue
    for (let mask = 1; mask < (1 << m) - 1; mask++) {
      const antecedent = s.items.filter((_, b) => (mask >> b) & 1)
      const consequent = s.items.filter((_, b) => !((mask >> b) & 1))
      if (consequent.length > maxConsequent) continue
      const a = counts.get(antecedent.join(','))
      const c = counts.get(consequent.join(','))
      if (a === undefined || c === undefined)
        throw new MdmlError('value', 'associationRules: itemsets are not closed downward')
      const measures = ruleMeasures({ n, both: s.count, antecedent: a, consequent: c })
      if (measures[metric] >= threshold - 1e-12) out.push({ antecedent, consequent, count: s.count, ...measures })
    }
  }
  return out.sort((x, y) => y[metric] - x[metric])
}
