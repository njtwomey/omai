import { allAssignments, type Factor, factor } from '../factor.ts'
import { type BayesNet, bayesNode, jointProbability } from '../models/bayes-net.ts'

/**
 * Exact inference by brute-force enumeration of every joint assignment (exponential; the reference other inference
 * algorithms are checked against).
 *
 * @param bn - The network.
 * @param query - The query variables.
 * @param evidence - Observed state indices by variable.
 * @returns The normalised posterior over the query.
 * @example
 * ```ts
 * import { bayesNet, enumerate, factorValues } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.3, 0.7] }])
 * factorValues(enumerate(bn, ['a'])) // => [0.3, 0.7]
 * ```
 */
export function enumerate(
  bn: BayesNet,
  query: readonly string[],
  evidence: Readonly<Record<string, number>> = {},
): Factor {
  const names = bn.nodes.map((n) => n.name)
  const card = bn.nodes.map((n) => n.states.length)
  const qcard = query.map((q) => bayesNode(bn, q).states.length)
  const table = new Float64Array(qcard.reduce((a, b) => a * b, 1))
  for (const a of allAssignments(names, card)) {
    if (Object.entries(evidence).some(([k, v]) => a[k] !== v)) continue
    let idx = 0
    query.forEach((q, k) => {
      idx = idx * (qcard[k] ?? 1) + (a[q] ?? 0)
    })
    table[idx] = (table[idx] ?? 0) + jointProbability(bn, a)
  }
  const total = table.reduce((s, v) => s + v, 0)
  return factor(
    query,
    qcard,
    table.map((v) => v / total),
  )
}
