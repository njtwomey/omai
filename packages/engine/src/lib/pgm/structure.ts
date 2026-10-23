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

/**
 * The moral graph of a Bayes network: undirected edges between each node and its parents, and between co-parents.
 *
 * @param bn - The network.
 * @returns The neighbours of each variable.
 * @example
 * ```ts
 * import { bayesNet, moralGraph } from 'whitebox-engine/pgm'
 * const p = [0.5, 0.5]
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: p }, { name: 'b', states: ['0', '1'], parents: [], cpt: p }, { name: 'c', states: ['0', '1'], parents: ['a', 'b'], cpt: [1, 0, 1, 0, 1, 0, 1, 0] }])
 * [...(moralGraph(bn).get('a') ?? [])].sort() // => ['b', 'c']
 * ```
 */
export function moralGraph(bn: BayesNet): Map<string, Set<string>> {
  const g = new Map<string, Set<string>>(bn.nodes.map((n) => [n.name, new Set<string>()]))
  const link = (a: string, b: string): void => {
    g.get(a)?.add(b)
    g.get(b)?.add(a)
  }
  for (const n of bn.nodes) {
    for (const p of n.parents) link(p, n.name)
    for (let i = 0; i < n.parents.length; i++)
      for (let j = i + 1; j < n.parents.length; j++) link(n.parents[i] as string, n.parents[j] as string)
  }
  return g
}

/**
 * A greedy elimination ordering on an undirected graph: repeatedly eliminate the variable with the fewest neighbours
 * (`minDegree`) or the fewest fill-in edges (`minFill`), ties broken by name.
 *
 * @param graph - Neighbours of each variable (for example a moral graph).
 * @param variables - The variables to order.
 * @param heuristic - The greedy criterion.
 * @returns The order.
 * @concept elimination-orderings
 * @example
 * ```ts
 * import { eliminationOrder } from 'whitebox-engine/pgm'
 * const g = new Map([['a', new Set(['b'])], ['b', new Set(['a', 'c'])], ['c', new Set(['b'])]])
 * eliminationOrder(g, ['a', 'b', 'c'], 'minDegree') // => ['a', 'b', 'c']
 * ```
 */
export function eliminationOrder(
  graph: ReadonlyMap<string, ReadonlySet<string>>,
  variables: readonly string[],
  heuristic: 'minDegree' | 'minFill' = 'minFill',
): string[] {
  const g = new Map([...graph].map(([k, v]) => [k, new Set(v)]))
  const remaining = new Set(variables)
  const order: string[] = []
  const cost = (v: string): number => {
    const ns = [...(g.get(v) ?? [])]
    if (heuristic === 'minDegree') return ns.length
    let fill = 0
    for (let i = 0; i < ns.length; i++)
      for (let j = i + 1; j < ns.length; j++) if (!g.get(ns[i] as string)?.has(ns[j] as string)) fill++
    return fill
  }
  while (remaining.size > 0) {
    const v = [...remaining].sort((a, b) => cost(a) - cost(b) || a.localeCompare(b))[0] as string
    const ns = [...(g.get(v) ?? [])]
    for (let i = 0; i < ns.length; i++) {
      for (let j = i + 1; j < ns.length; j++) {
        g.get(ns[i] as string)?.add(ns[j] as string)
        g.get(ns[j] as string)?.add(ns[i] as string)
      }
    }
    for (const w of ns) g.get(w)?.delete(v)
    g.delete(v)
    remaining.delete(v)
    order.push(v)
  }
  return order
}
