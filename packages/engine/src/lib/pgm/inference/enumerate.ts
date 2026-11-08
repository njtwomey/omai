/**
 * Exact inference by brute-force enumeration of every joint assignment: exponential in the number of variables, and the
 * reference the other inference algorithms are tested against.
 */

import { logSumExpOf } from '../../base/index.ts'
import { allAssignments, type Factor, logFactor } from '../factor.ts'
import {
  type DiscreteModel,
  evidenceIndices,
  modelVariable,
  modelVariables,
  unnormalisedLogProbability,
} from '../models/factor-graph.ts'

/**
 * Exact inference by brute-force enumeration of every joint assignment (exponential; the reference other inference
 * algorithms are checked against). Works for any discrete model.
 *
 * @param model - The model.
 * @param query - The query variables.
 * @param evidence - Observed states (names or indices) by variable.
 * @returns The normalised posterior over the query, in query order.
 * @concept exact-inference
 * @example
 * ```ts
 * import { bayesNet, enumerate, factorValues } from 'omai-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.3, 0.7] }])
 * factorValues(enumerate(bn, ['a'])).map((p) => Math.round(p * 10) / 10) // => [0.3, 0.7]
 * ```
 */
export function enumerate(
  model: DiscreteModel,
  query: readonly string[],
  evidence: Readonly<Record<string, string | number>> = {},
): Factor {
  const ev = evidenceIndices(model, evidence)
  const variables = modelVariables(model)
  const qcard = query.map((q) => modelVariable(model, q).states.length)
  const cells: number[][] = Array.from({ length: qcard.reduce((a, b) => a * b, 1) }, () => [])
  for (const a of allAssignments(
    variables.map((v) => v.name),
    variables.map((v) => v.states.length),
  )) {
    if (Object.entries(ev).some(([k, v]) => a[k] !== v)) continue
    let idx = 0
    for (const [k, q] of query.entries()) idx = idx * (qcard[k] ?? 1) + (a[q] ?? 0)
    cells[idx]?.push(unnormalisedLogProbability(model, a))
  }
  const logs = cells.map(logSumExpOf)
  const total = logSumExpOf(logs)
  return logFactor(
    query,
    qcard,
    logs.map((l) => l - total),
  )
}
