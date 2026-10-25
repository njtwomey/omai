/**
 * Discrete Bayes networks as plain data: nodes with states, parents and conditional probability tables, validated and
 * in topological order. Their factors and joint probabilities, ancestral sampling, Markov blankets, d-separation and
 * active trails (Bayes-ball), minimal d-separators, and the local and pairwise independencies a network implies, as in
 * pgmpy.
 */

import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { type Factor, factor } from '../factor.ts'
import {
  ancestors,
  type DagNode,
  type DirectedModel,
  descendants,
  moralGraph,
  separated,
  topologicalOrder,
} from '../graphs.ts'

/** One node of a Bayes network: its states, parents and conditional probability table. */
export interface BayesNode {
  readonly name: string
  readonly states: readonly string[]
  readonly parents: readonly string[]
  /**
   * P(node | parents) in row-major order over [...parents, node]: one row of probabilities over the node's states for
   * each joint parent assignment (the first parent varies slowest).
   */
  readonly cpt: readonly number[]
}

/**
 * A discrete Bayes network: plain data (nodes with states, parents and CPTs) in a topological order.
 *
 * @concept bayes-networks
 */
export interface BayesNet {
  readonly kind: 'bayesNet'
  readonly nodes: readonly BayesNode[]
}

/**
 * Build and validate a Bayes network: parents must exist, the graph must be acyclic, and every CPT row must be a
 * probability distribution. Nodes are reordered topologically.
 *
 * @param nodes - The nodes, in any order.
 * @returns The network.
 * @throws WhiteboxError with code `value` for unknown parents, cycles or invalid CPTs.
 * @concept bayes-networks
 * @example
 * ```ts
 * import { bayesNet } from 'whitebox-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * bn.nodes.map((n) => n.name) // => ['rain', 'wet']
 * ```
 */
export function bayesNet(nodes: readonly BayesNode[]): BayesNet {
  const byName = new Map(nodes.map((n) => [n.name, n]))
  if (byName.size !== nodes.length) throw new WhiteboxError('value', 'bayesNet: node names must be distinct')
  for (const n of nodes) {
    if (n.states.length === 0) throw new WhiteboxError('value', `bayesNet: ${n.name} has no states`)
    for (const p of n.parents)
      if (!byName.has(p)) throw new WhiteboxError('value', `bayesNet: ${n.name} has unknown parent ${p}`)
    const rows = n.parents.reduce((k, p) => k * (byName.get(p)?.states.length ?? 1), 1)
    const k = n.states.length
    if (n.cpt.length !== rows * k)
      throw new WhiteboxError('shape', `bayesNet: ${n.name} needs ${rows * k} CPT entries, got ${n.cpt.length}`)
    for (let r = 0; r < rows; r++) {
      const row = n.cpt.slice(r * k, (r + 1) * k)
      if (row.some((v) => !(v >= 0)) || Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-9) {
        throw new WhiteboxError('value', `bayesNet: row ${r} of ${n.name}'s CPT is not a distribution`)
      }
    }
  }
  try {
    return { kind: 'bayesNet', nodes: topologicalOrder(nodes) }
  } catch (e) {
    throw new WhiteboxError('value', `bayesNet: ${(e as Error).message}`)
  }
}

/**
 * A node of a network by name.
 *
 * @param bn - The network.
 * @param name - The node.
 * @returns The node.
 * @throws WhiteboxError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { bayesNet, bayesNode } from 'whitebox-engine/pgm'
 * bayesNode(bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }]), 'a').states.length // => 2
 * ```
 */
export function bayesNode(bn: BayesNet, name: string): BayesNode {
  const n = bn.nodes.find((x) => x.name === name)
  if (n === undefined) throw new WhiteboxError('value', `unknown variable ${name}`)
  return n
}

/**
 * The CPT of each node as a factor over [...parents, node].
 *
 * @param bn - The network.
 * @returns One factor per node, in topological order.
 * @concept factors
 * @example
 * ```ts
 * import { bayesNet, bayesFactors } from 'whitebox-engine/pgm'
 * bayesFactors(bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }]))[0]?.variables // => ['a']
 * ```
 */
export function bayesFactors(bn: BayesNet): Factor[] {
  return bn.nodes.map((n) =>
    factor([...n.parents, n.name], [...n.parents.map((p) => bayesNode(bn, p).states.length), n.states.length], n.cpt),
  )
}

/** The CPT row index for a node given its parents' states. */
function rowOf(bn: BayesNet, n: BayesNode, assignment: Readonly<Record<string, number>>): number {
  return n.parents.reduce((r, p) => r * bayesNode(bn, p).states.length + (assignment[p] ?? 0), 0)
}

/**
 * The conditional distribution of a node given a full assignment of its parents (one row of its CPT).
 *
 * @param bn - The network.
 * @param name - The node.
 * @param assignment - State indices of (at least) its parents.
 * @returns P(node | parents), over its states.
 * @example
 * ```ts
 * import { bayesNet, cptRow } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.3, 0.7] }])
 * cptRow(bn, 'b', { a: 1 }) // => [0.3, 0.7]
 * ```
 */
export function cptRow(bn: BayesNet, name: string, assignment: Readonly<Record<string, number>>): number[] {
  const n = bayesNode(bn, name)
  const k = n.states.length
  const r = rowOf(bn, n, assignment)
  return n.cpt.slice(r * k, (r + 1) * k)
}

/**
 * The joint probability of a full assignment: the product of each node's CPT entry.
 *
 * @param bn - The network.
 * @param assignment - A state index for every node.
 * @returns The probability.
 * @example
 * ```ts
 * import { bayesNet, jointProbability } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.25, 0.75] }])
 * jointProbability(bn, { a: 1 }) // => 0.75
 * ```
 */
export function jointProbability(bn: BayesNet, assignment: Readonly<Record<string, number>>): number {
  return bn.nodes.reduce(
    (p, n) => p * (n.cpt[rowOf(bn, n, assignment) * n.states.length + (assignment[n.name] ?? 0)] ?? 0),
    1,
  )
}

/** Draw a state from a probability row with a uniform number. */
function drawRow(row: ArrayLike<number>, u: number): number {
  let acc = 0
  for (let k = 0; k < row.length; k++) {
    acc += row[k] ?? 0
    if (u < acc) return k
  }
  return row.length - 1
}

/**
 * Ancestral (forward) sampling: draw each node from its CPT given its sampled parents, in topological order.
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of samples.
 * @returns The samples, as state indices by variable.
 * @concept sampling
 * @example
 * ```ts
 * import { bayesNet, ancestralSample } from 'whitebox-engine/pgm'
 * import { key } from 'whitebox-engine/random'
 * ancestralSample(bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0, 1] }]), key(0), 3) // => [{ a: 1 }, { a: 1 }, { a: 1 }]
 * ```
 */
export function ancestralSample(bn: BayesNet, key: Key, n: number): Record<string, number>[] {
  return Array.from({ length: n }, (_, i) => {
    const u = uniform(foldIn(key, i), [bn.nodes.length]).data
    const s: Record<string, number> = {}
    for (const [j, node] of bn.nodes.entries()) {
      const base = rowOf(bn, node, s) * node.states.length
      s[node.name] = drawRow(node.cpt.slice(base, base + node.states.length), u[j] ?? 0)
    }
    return s
  })
}

function dagNode(g: DirectedModel, name: string): DagNode {
  const n = g.nodes.find((x) => x.name === name)
  if (n === undefined) throw new WhiteboxError('value', `unknown variable ${name}`)
  return n
}

/**
 * The children of a node.
 *
 * @param g - The network (or any DAG).
 * @param name - The node.
 * @returns Its children, in node order.
 * @example
 * ```ts
 * import { dag, children } from 'whitebox-engine/pgm'
 * children(dag([['a', 'b'], ['a', 'c']]), 'a') // => ['b', 'c']
 * ```
 */
export function children(g: DirectedModel, name: string): string[] {
  dagNode(g, name)
  return g.nodes.filter((n) => n.parents.includes(name)).map((n) => n.name)
}

/**
 * The Markov blanket of a node: its parents, children and the children's other parents.
 *
 * @param g - The network (or any DAG).
 * @param name - The node.
 * @returns The blanket, sorted.
 * @concept bayes-networks
 * @example
 * ```ts
 * import { dag, markovBlanket } from 'whitebox-engine/pgm'
 * markovBlanket(dag([['a', 'b'], ['c', 'b'], ['d', 'a']]), 'a') // => ['b', 'c', 'd']
 * ```
 */
export function markovBlanket(g: DirectedModel, name: string): string[] {
  const node = dagNode(g, name)
  const out = new Set<string>(node.parents)
  for (const c of children(g, name)) {
    out.add(c)
    for (const p of dagNode(g, c).parents) if (p !== name) out.add(p)
  }
  return [...out].sort()
}

/**
 * Whether two sets of variables are d-separated given a third (the ancestral moral graph criterion: x and y are
 * d-separated given z when they are disconnected after restricting to the ancestors of x, y and z, moralising, and
 * deleting z).
 *
 * @param g - The network (or any DAG).
 * @param x - First set.
 * @param y - Second set.
 * @param z - The conditioning set.
 * @returns `true` when d-separated.
 * @concept d-separation
 * @example
 * ```ts
 * import { dag, dSeparated } from 'whitebox-engine/pgm'
 * const collider = dag([['a', 'c'], ['b', 'c']])
 * dSeparated(collider, ['a'], ['b'], []) // => true
 * dSeparated(collider, ['a'], ['b'], ['c']) // => false
 * ```
 */
export function dSeparated(
  g: DirectedModel,
  x: readonly string[],
  y: readonly string[],
  z: readonly string[],
): boolean {
  const keep = new Set(ancestors(g, [...x, ...y, ...z]))
  const sub = { nodes: g.nodes.filter((n) => keep.has(n.name)) }
  return separated(moralGraph(sub), x, y, z)
}

/**
 * The nodes reachable from a start node by an active trail given observations (the Bayes-ball reachability algorithm,
 * Koller and Friedman algorithm 3.1; pgmpy's `active_trail_nodes`). A node is d-connected to the start exactly when it
 * is reachable.
 *
 * @param g - The network (or any DAG).
 * @param start - The start node.
 * @param observed - The observed nodes.
 * @returns The reachable unobserved nodes (including the start), sorted.
 * @concept d-separation
 * @example
 * ```ts
 * import { dag, activeTrailNodes } from 'whitebox-engine/pgm'
 * const g = dag([['a', 'c'], ['b', 'c'], ['c', 'd']])
 * activeTrailNodes(g, 'a', []) // => ['a', 'c', 'd']
 * activeTrailNodes(g, 'a', ['d']) // => ['a', 'b', 'c']
 * ```
 */
export function activeTrailNodes(g: DirectedModel, start: string, observed: readonly string[] = []): string[] {
  dagNode(g, start)
  const obs = new Set(observed)
  const ancestorsOfObserved = new Set(observed.length === 0 ? [] : ancestors(g, observed))
  const visited = new Set<string>()
  const reachable = new Set<string>()
  const queue: [string, 'up' | 'down'][] = [[start, 'up']]
  while (queue.length > 0) {
    const [v, dir] = queue.shift() as [string, 'up' | 'down']
    const tag = `${v}|${dir}`
    if (visited.has(tag)) continue
    visited.add(tag)
    if (!obs.has(v)) reachable.add(v)
    if (dir === 'up' && !obs.has(v)) {
      for (const p of dagNode(g, v).parents) queue.push([p, 'up'])
      for (const c of children(g, v)) queue.push([c, 'down'])
    } else if (dir === 'down') {
      if (!obs.has(v)) for (const c of children(g, v)) queue.push([c, 'down'])
      if (ancestorsOfObserved.has(v)) for (const p of dagNode(g, v).parents) queue.push([p, 'up'])
    }
  }
  return [...reachable].sort()
}

/**
 * A minimal d-separating set of two non-adjacent nodes (Tian, Paz and Pearl 1998, algorithm 4, as pgmpy's
 * `minimal_dseparator`): start from the parents of both and drop members, in sorted order, while separation holds.
 *
 * @param g - The network (or any DAG).
 * @param a - First node.
 * @param b - Second node.
 * @returns The separator (sorted), or `null` when the nodes are adjacent.
 * @concept d-separation
 * @example
 * ```ts
 * import { dag, minimalDSeparator } from 'whitebox-engine/pgm'
 * minimalDSeparator(dag([['a', 'b'], ['b', 'c'], ['a', 'd'], ['d', 'c']]), 'a', 'c') // => ['b', 'd']
 * ```
 */
export function minimalDSeparator(g: DirectedModel, a: string, b: string): string[] | null {
  const na = dagNode(g, a)
  const nb = dagNode(g, b)
  if (na.parents.includes(b) || nb.parents.includes(a)) return null
  const keep = new Set(ancestors(g, [a, b]))
  const sub = { nodes: g.nodes.filter((n) => keep.has(n.name)) }
  const sep = [...new Set([...na.parents, ...nb.parents])].filter((v) => v !== a && v !== b).sort()
  if (!dSeparated(sub, [a], [b], sep)) return null
  let current = sep
  for (const u of sep) {
    const without = current.filter((v) => v !== u)
    if (dSeparated(sub, [a], [b], without)) current = without
  }
  return current
}

/** A conditional independence statement: the variables of `x` are independent of those of `y` given `given`. */
export interface Independence {
  readonly x: readonly string[]
  readonly y: readonly string[]
  readonly given: readonly string[]
}

/**
 * The local (Markov) independencies of a network: each node is independent of its non-descendants other than its
 * parents, given its parents (pgmpy's `local_independencies`).
 *
 * @param g - The network (or any DAG).
 * @returns One statement per node with a non-empty right-hand side, in sorted node order.
 * @concept d-separation
 * @example
 * ```ts
 * import { dag, localIndependencies } from 'whitebox-engine/pgm'
 * localIndependencies(dag([['a', 'b'], ['b', 'c']])) // => [{ x: ['c'], y: ['a'], given: ['b'] }]
 * ```
 */
export function localIndependencies(g: DirectedModel): Independence[] {
  const out: Independence[] = []
  for (const v of g.nodes.map((n) => n.name).sort()) {
    const desc = new Set(descendants(g, v))
    const parents = [...dagNode(g, v).parents].sort()
    const rest = g.nodes
      .map((n) => n.name)
      .filter((w) => !desc.has(w) && !parents.includes(w))
      .sort()
    if (rest.length > 0) out.push({ x: [v], y: rest, given: parents })
  }
  return out
}

/**
 * The pairwise independencies implied by a DAG: for every pair of non-adjacent nodes, the statement with their minimal
 * d-separator (pgmpy's `get_independencies` before its reduction step).
 *
 * @param g - The network (or any DAG).
 * @returns The statements, for pairs in sorted order.
 * @concept d-separation
 * @example
 * ```ts
 * import { dag, independencies } from 'whitebox-engine/pgm'
 * independencies(dag([['a', 'b'], ['b', 'c']])) // => [{ x: ['a'], y: ['c'], given: ['b'] }]
 * ```
 */
export function independencies(g: DirectedModel): Independence[] {
  const names = g.nodes.map((n) => n.name).sort()
  const out: Independence[] = []
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = names[i] as string
      const b = names[j] as string
      const sep = minimalDSeparator(g, a, b)
      if (sep !== null) out.push({ x: [a], y: [b], given: sep })
    }
  }
  return out
}
