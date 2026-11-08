/**
 * Influence diagrams (Howard and Matheson 1984), Bayes networks extended with decision nodes and utility nodes for
 * decision making under uncertainty: construction and validation, information sets, the network a policy defines,
 * expected utility, the optimal policy by backward induction, and the value of information of observing a variable
 * before deciding.
 */

import { OmaiError } from '../../base/index.ts'
import { allAssignments, factorValues } from '../factor.ts'
import { descendants, topologicalOrder } from '../graphs.ts'
import { logPartition, variableElimination } from '../inference/elimination.ts'
import { type BayesNet, type BayesNode, bayesNet } from './bayes-net.ts'

/** A decision node: its options and its informational parents (what is known when it is made). */
export interface DecisionNode {
  readonly name: string
  readonly states: readonly string[]
  readonly parents: readonly string[]
}

/** A utility node: a table of utilities over its parents (row-major, the first parent slowest). */
export interface UtilityNode {
  readonly name: string
  readonly parents: readonly string[]
  readonly values: readonly number[]
}

/**
 * An influence diagram (Howard and Matheson 1984): chance nodes with CPTs, decision nodes in their temporal order with
 * informational parents, and additive utility nodes. Decisions are taken with no forgetting: each decision also knows
 * the earlier decisions and everything they knew.
 *
 * @concept influence-diagrams
 */
export interface InfluenceDiagram {
  readonly kind: 'influenceDiagram'
  readonly chance: readonly BayesNode[]
  /** Decisions in the order they are made. */
  readonly decisions: readonly DecisionNode[]
  readonly utilities: readonly UtilityNode[]
}

/** A policy: for each decision, the chosen state (index) for each configuration of its information set. */
export type Policy = Readonly<Record<string, readonly number[]>>

/**
 * Build and validate an influence diagram: parents must exist, utilities may not have children, the chance and decision
 * nodes must form a DAG, CPT and utility tables must have the right sizes, and no decision may be an ancestor of an
 * earlier one.
 *
 * @param chance - The chance nodes.
 * @param decisions - The decisions, in temporal order.
 * @param utilities - The utility nodes.
 * @returns The diagram.
 * @throws OmaiError with code `value` for invalid structure or tables.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram } from 'omai-engine/pgm'
 * const id = influenceDiagram(
 *   [{ name: 'weather', states: ['dry', 'rain'], parents: [], cpt: [0.7, 0.3] }],
 *   [{ name: 'umbrella', states: ['leave', 'take'], parents: [] }],
 *   [{ name: 'u', parents: ['weather', 'umbrella'], values: [100, 70, 0, 80] }],
 * )
 * id.decisions.length // => 1
 * ```
 */
export function influenceDiagram(
  chance: readonly BayesNode[],
  decisions: readonly DecisionNode[],
  utilities: readonly UtilityNode[],
): InfluenceDiagram {
  const states = new Map<string, readonly string[]>([...chance, ...decisions].map((n) => [n.name, n.states]))
  const all = [...chance.map((n) => n.name), ...decisions.map((n) => n.name), ...utilities.map((n) => n.name)]
  if (new Set(all).size !== all.length) throw new OmaiError('value', 'influenceDiagram: node names must be distinct')
  for (const n of [...chance, ...decisions, ...utilities]) {
    for (const p of n.parents) {
      if (!states.has(p))
        throw new OmaiError('value', `influenceDiagram: ${n.name} has unknown parent ${p} (utilities have no children)`)
    }
  }
  for (const u of utilities) {
    const size = u.parents.reduce((k, p) => k * (states.get(p)?.length ?? 1), 1)
    if (u.values.length !== size)
      throw new OmaiError('value', `influenceDiagram: utility ${u.name} needs ${size} values`)
  }
  const net: InfluenceDiagram = { kind: 'influenceDiagram', chance, decisions, utilities }
  try {
    topologicalOrder([...chance, ...decisions])
  } catch (e) {
    throw new OmaiError('value', `influenceDiagram: ${(e as Error).message}`)
  }
  // Every decision's information set must precede it.
  for (const d of decisions) {
    const after = new Set(descendants({ nodes: [...chance, ...decisions] }, d.name))
    for (const v of informationSet(net, d.name)) {
      if (after.has(v)) throw new OmaiError('value', `influenceDiagram: ${v} is known at ${d.name} but depends on it`)
    }
  }
  policyNetwork(net, 'uniform') // validates the CPTs
  return net
}

function decisionIndex(id: InfluenceDiagram, name: string): number {
  const k = id.decisions.findIndex((d) => d.name === name)
  if (k < 0) throw new OmaiError('value', `unknown decision ${name}`)
  return k
}

/**
 * The information set of a decision under no forgetting: the parents of every decision up to and including it, and
 * the earlier decisions themselves, in temporal order.
 *
 * @param id - The diagram.
 * @param decision - The decision.
 * @returns The variables known when it is made.
 * @throws OmaiError with code `value` for an unknown decision.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram, informationSet } from 'omai-engine/pgm'
 * const id = influenceDiagram(
 *   [{ name: 'test', states: ['neg', 'pos'], parents: [], cpt: [0.5, 0.5] }],
 *   [{ name: 'd1', states: ['a', 'b'], parents: ['test'] }, { name: 'd2', states: ['a', 'b'], parents: [] }],
 *   [],
 * )
 * informationSet(id, 'd2') // => ['test', 'd1']
 * ```
 */
export function informationSet(id: InfluenceDiagram, decision: string): string[] {
  const k = decisionIndex(id, decision)
  const out: string[] = []
  for (let j = 0; j <= k; j++) {
    const d = id.decisions[j] as DecisionNode
    for (const p of d.parents) if (!out.includes(p)) out.push(p)
    if (j < k && !out.includes(d.name)) out.push(d.name)
  }
  return out
}

function statesOf(id: InfluenceDiagram, name: string): readonly string[] {
  return [...id.chance, ...id.decisions].find((n) => n.name === name)?.states ?? []
}

/**
 * The Bayes network obtained by fixing a policy: each decision becomes a node whose parents are its information set,
 * with a deterministic CPT choosing the policy's state (or a uniform CPT for `'uniform'` or decisions the policy omits).
 *
 * @param id - The diagram.
 * @param policy - The policy, or `'uniform'` for randomised decisions.
 * @returns The network.
 * @throws OmaiError with code `value` for policy tables of the wrong size.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram, policyNetwork } from 'omai-engine/pgm'
 * const id = influenceDiagram([], [{ name: 'd', states: ['a', 'b'], parents: [] }], [])
 * policyNetwork(id, { d: [1] }).nodes[0]?.cpt // => [0, 1]
 * ```
 */
export function policyNetwork(id: InfluenceDiagram, policy: Policy | 'uniform'): BayesNet {
  const decisions: BayesNode[] = id.decisions.map((d) => {
    const info = informationSet(id, d.name)
    const configs = info.reduce((k, v) => k * statesOf(id, v).length, 1)
    const k = d.states.length
    const choice = policy === 'uniform' ? undefined : policy[d.name]
    if (choice !== undefined && choice.length !== configs)
      throw new OmaiError('value', `policy for ${d.name} needs ${configs} entries`)
    const cpt = Array.from({ length: configs }, (_, c) =>
      Array.from({ length: k }, (_, s) => (choice === undefined ? 1 / k : (choice[c] ?? 0) === s ? 1 : 0)),
    ).flat()
    return { name: d.name, states: d.states, parents: info, cpt }
  })
  return bayesNet([...id.chance, ...decisions])
}

/** E[U | evidence] for every utility node, summed, or NaN when the evidence is impossible. */
function conditionalUtility(bn: BayesNet, id: InfluenceDiagram, evidence: Readonly<Record<string, number>>): number {
  if (Object.keys(evidence).length > 0 && logPartition(bn, { evidence }) === Number.NEGATIVE_INFINITY) return Number.NaN
  let total = 0
  for (const u of id.utilities) {
    const free = u.parents.filter((p) => !(p in evidence))
    let posterior: number[] = [1]
    if (free.length > 0) {
      posterior = factorValues(variableElimination(bn, free, { evidence }).posterior)
    }
    const cards = u.parents.map((p) => statesOf(id, p).length)
    for (const [e, a] of allAssignments(
      free,
      free.map((p) => statesOf(id, p).length),
    ).entries()) {
      let idx = 0
      for (const [m, p] of u.parents.entries())
        idx = idx * (cards[m] ?? 1) + (p in evidence ? (evidence[p] ?? 0) : (a[p] ?? 0))
      total += (posterior[e] ?? 0) * (u.values[idx] ?? 0)
    }
  }
  return total
}

/**
 * The expected utility of a policy: the sum over utility nodes of E[U] under the Bayes network the policy defines.
 *
 * @param id - The diagram.
 * @param policy - The policy.
 * @returns The expected utility.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram, expectedUtility } from 'omai-engine/pgm'
 * const id = influenceDiagram(
 *   [{ name: 'weather', states: ['dry', 'rain'], parents: [], cpt: [0.7, 0.3] }],
 *   [{ name: 'umbrella', states: ['leave', 'take'], parents: [] }],
 *   [{ name: 'u', parents: ['weather', 'umbrella'], values: [100, 70, 0, 80] }],
 * )
 * expectedUtility(id, { umbrella: [0] }) // => 70
 * ```
 */
export function expectedUtility(id: InfluenceDiagram, policy: Policy): number {
  return conditionalUtility(policyNetwork(id, policy), id, {})
}

/**
 * Solve an influence diagram: the optimal policy by backward induction over the decisions (the last first; for each
 * configuration of a decision's information set, the option of largest conditional expected utility given optimal later
 * decisions) and its maximum expected utility (MEU).
 *
 * @param id - The diagram.
 * @returns The optimal policy, the MEU, and the decision rules by state name.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram, solveInfluenceDiagram } from 'omai-engine/pgm'
 * const id = influenceDiagram(
 *   [{ name: 'weather', states: ['dry', 'rain'], parents: [], cpt: [0.7, 0.3] }],
 *   [{ name: 'umbrella', states: ['leave', 'take'], parents: [] }],
 *   [{ name: 'u', parents: ['weather', 'umbrella'], values: [100, 70, 0, 80] }],
 * )
 * const r = solveInfluenceDiagram(id)
 * [r.policy.umbrella, r.meu] // => [[1], 73]
 * ```
 */
export function solveInfluenceDiagram(id: InfluenceDiagram): {
  readonly policy: Policy
  readonly meu: number
  readonly rules: Readonly<
    Record<string, readonly { readonly when: Record<string, string>; readonly choose: string }[]>
  >
} {
  const policy: Record<string, number[]> = {}
  const rules: Record<string, { when: Record<string, string>; choose: string }[]> = {}
  for (let k = id.decisions.length - 1; k >= 0; k--) {
    const d = id.decisions[k] as DecisionNode
    const info = informationSet(id, d.name)
    // Earlier decisions randomised (keeps every configuration possible), later ones optimal, this one free.
    const bn = policyNetwork(id, { ...policy })
    const configs = allAssignments(
      info,
      info.map((v) => statesOf(id, v).length),
    )
    const choice: number[] = []
    rules[d.name] = []
    for (const c of configs) {
      let best = 0
      let bestValue = Number.NEGATIVE_INFINITY
      for (let s = 0; s < d.states.length; s++) {
        const v = conditionalUtility(bn, id, { ...c, [d.name]: s })
        if (v > bestValue) {
          best = s
          bestValue = v
        }
      }
      choice.push(best)
      rules[d.name]?.push({
        when: Object.fromEntries(info.map((v) => [v, statesOf(id, v)[c[v] ?? 0] ?? ''])),
        choose: d.states[best] ?? '',
      })
    }
    policy[d.name] = choice
  }
  return { policy, meu: expectedUtility(id, policy), rules }
}

/**
 * The value of information of observing some chance variables before a decision: the MEU when they are added to the
 * decision's informational parents minus the MEU without them (never negative).
 *
 * @param id - The diagram.
 * @param decision - The decision.
 * @param variables - The variables that would be observed.
 * @returns The value of the information.
 * @throws OmaiError with code `value` when a variable depends on the decision.
 * @concept influence-diagrams
 * @example
 * ```ts
 * import { influenceDiagram, valueOfInformation } from 'omai-engine/pgm'
 * const id = influenceDiagram(
 *   [{ name: 'weather', states: ['dry', 'rain'], parents: [], cpt: [0.7, 0.3] }],
 *   [{ name: 'umbrella', states: ['leave', 'take'], parents: [] }],
 *   [{ name: 'u', parents: ['weather', 'umbrella'], values: [100, 70, 0, 80] }],
 * )
 * valueOfInformation(id, 'umbrella', ['weather']) // => 21
 * ```
 */
export function valueOfInformation(id: InfluenceDiagram, decision: string, variables: readonly string[]): number {
  const k = decisionIndex(id, decision)
  const informed = id.decisions.map((d, j) =>
    j === k ? { ...d, parents: [...d.parents, ...variables.filter((v) => !d.parents.includes(v))] } : d,
  )
  const with_ = solveInfluenceDiagram(influenceDiagram(id.chance, informed, id.utilities)).meu
  return with_ - solveInfluenceDiagram(id).meu
}
