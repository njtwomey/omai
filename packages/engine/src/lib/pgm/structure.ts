import { type Structure, structure } from '../scene/index.ts'
import type { BayesNet } from './models/bayes-net.ts'

/**
 * A Bayes network as a {@link Structure}: one node per variable (with its states and CPT size) and an edge from each
 * parent to its child; observed variables (by evidence) take the `observed` role.
 *
 * @param bn - The network.
 * @param observed - Variables to show as observed.
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { bayesNet, toStructure } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * toStructure(bn).edges // => [{ from: 'a', to: 'b', directed: true, kind: 'dependency' }]
 * ```
 */
export function toStructure(bn: BayesNet, observed: readonly string[] = []): Structure {
  return structure(
    'bayes-net',
    bn.nodes.map((n) => ({
      id: n.name,
      label: n.name,
      role: observed.includes(n.name) ? ('observed' as const) : ('latent' as const),
      detail: n.states.join(', '),
      badges: { states: n.states.length, cpt: n.cpt.length },
    })),
    bn.nodes.flatMap((n) =>
      n.parents.map((p) => ({ from: p, to: n.name, directed: true, kind: 'dependency' as const })),
    ),
    { title: 'Bayes network', direction: 'down' },
  )
}
