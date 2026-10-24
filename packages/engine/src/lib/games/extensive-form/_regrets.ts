// Regret matching, policy averages and the trace channels shared by the CFR family.
import { channels, type Observation, type TraceSchema } from '../../trace/index.ts'
import { tablePolicy } from './_tree.ts'
import { exploitability } from './best-response.ts'
import type { GameTree } from './tree.ts'

/** Regret matching on cumulative regrets: proportional to the positive part, uniform when none is positive. */
export function regretMatch(r: Float64Array): Float64Array {
  let total = 0
  for (const v of r) if (v > 0) total += v
  return total > 0 ? r.map((v) => (v > 0 ? v / total : 0)) : new Float64Array(r.length).fill(1 / r.length)
}

/** The average policy: the policy sums normalised, uniform where they are zero. */
export function averageOf(sums: readonly Float64Array[]): Float64Array[] {
  return sums.map((s) => {
    let total = 0
    for (const v of s) total += v
    return total > 0 ? s.map((v) => v / total) : new Float64Array(s.length).fill(1 / s.length)
  })
}

/** The channels of the CFR family. */
export const CFR_SCHEMA: TraceSchema = channels({
  metrics: {
    exploitability: { description: 'exploitability of the average policy', direction: 'lower' },
    averageRegret: {
      description: 'sum over information sets of the largest positive cumulative regret, over iterations',
      direction: 'lower',
    },
  },
  tensors: {
    regrets: { role: 'state', description: 'cumulative regrets, information sets in tree order', shape: ['actions'] },
    average: { role: 'state', description: 'the average policy, information sets in tree order', shape: ['actions'] },
  },
})

/** Exploitability, average regret, and the regrets and average policy flattened in tree order. */
export function cfrObservation(
  tree: GameTree,
  t: number,
  regrets: readonly Float64Array[],
  sums: readonly Float64Array[],
): Observation {
  const average = averageOf(sums)
  let bound = 0
  for (const r of regrets) {
    let top = 0
    for (const v of r) top = Math.max(top, v)
    bound += top
  }
  const flat = (xs: readonly Float64Array[]): number[] => xs.flatMap((x) => Array.from(x))
  return {
    metrics: { exploitability: exploitability(tree, tablePolicy(tree, average)), averageRegret: t > 0 ? bound / t : 0 },
    tensors: { regrets: { role: 'state', value: flat(regrets) }, average: { role: 'state', value: flat(average) } },
  }
}
