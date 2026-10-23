import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { type Factor, factor } from '../factor.ts'

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
  // Kahn's algorithm, keeping the given order among ready nodes.
  const ordered: BayesNode[] = []
  const placed = new Set<string>()
  while (ordered.length < nodes.length) {
    const next = nodes.find((n) => !placed.has(n.name) && n.parents.every((p) => placed.has(p)))
    if (next === undefined) throw new WhiteboxError('value', 'bayesNet: the graph has a cycle')
    ordered.push(next)
    placed.add(next.name)
  }
  return { kind: 'bayesNet', nodes: ordered }
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

/**
 * Ancestral (forward) sampling: draw each node from its CPT given its sampled parents, in topological order.
 *
 * @param bn - The network.
 * @param key - The random key.
 * @param n - Number of samples.
 * @returns The samples, as state indices by variable.
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
    bn.nodes.forEach((node, j) => {
      const base = rowOf(bn, node, s) * node.states.length
      let acc = 0
      let state = node.states.length - 1
      for (let k = 0; k < node.states.length; k++) {
        acc += node.cpt[base + k] ?? 0
        if ((u[j] ?? 0) < acc) {
          state = k
          break
        }
      }
      s[node.name] = state
    })
    return s
  })
}

/**
 * The children of a node.
 *
 * @param bn - The network.
 * @param name - The node.
 * @returns Its children.
 * @example
 * ```ts
 * import { bayesNet, children } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * children(bn, 'a') // => ['b']
 * ```
 */
export function children(bn: BayesNet, name: string): string[] {
  return bn.nodes.filter((n) => n.parents.includes(name)).map((n) => n.name)
}

/**
 * The Markov blanket of a node: its parents, children and the children's other parents.
 *
 * @param bn - The network.
 * @param name - The node.
 * @returns The blanket, sorted.
 * @concept bayes-networks
 * @example
 * ```ts
 * import { bayesNet, markovBlanket } from 'whitebox-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * markovBlanket(bn, 'a') // => ['b']
 * ```
 */
export function markovBlanket(bn: BayesNet, name: string): string[] {
  const node = bayesNode(bn, name)
  const out = new Set<string>(node.parents)
  for (const c of children(bn, name)) {
    out.add(c)
    for (const p of bayesNode(bn, c).parents) if (p !== name) out.add(p)
  }
  return [...out].sort()
}

/**
 * Whether two sets of variables are d-separated given a third (the ancestral moral graph criterion: x and y are
 * d-separated given z when they are disconnected after restricting to the ancestors of x, y and z, moralising, and
 * deleting z).
 *
 * @param bn - The network.
 * @param x - First set.
 * @param y - Second set.
 * @param z - The conditioning set.
 * @returns `true` when d-separated.
 * @concept d-separation
 * @example
 * ```ts
 * import { bayesNet, dSeparated } from 'whitebox-engine/pgm'
 * const p = [0.5, 0.5]
 * const bn = bayesNet([
 *   { name: 'a', states: ['0', '1'], parents: [], cpt: p },
 *   { name: 'b', states: ['0', '1'], parents: [], cpt: p },
 *   { name: 'c', states: ['0', '1'], parents: ['a', 'b'], cpt: [1, 0, 0.5, 0.5, 0.5, 0.5, 0, 1] },
 * ])
 * dSeparated(bn, ['a'], ['b'], []) // => true
 * dSeparated(bn, ['a'], ['b'], ['c']) // => false
 * ```
 */
export function dSeparated(bn: BayesNet, x: readonly string[], y: readonly string[], z: readonly string[]): boolean {
  const ancestors = new Set<string>()
  const stack = [...x, ...y, ...z]
  while (stack.length > 0) {
    const v = stack.pop() as string
    if (ancestors.has(v)) continue
    ancestors.add(v)
    stack.push(...bayesNode(bn, v).parents)
  }
  const adjacency = new Map<string, Set<string>>([...ancestors].map((v) => [v, new Set<string>()]))
  const link = (a: string, b: string): void => {
    adjacency.get(a)?.add(b)
    adjacency.get(b)?.add(a)
  }
  for (const v of ancestors) {
    const ps = bayesNode(bn, v).parents
    for (const p of ps) link(p, v)
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) link(ps[i] as string, ps[j] as string)
  }
  const blocked = new Set(z)
  const seen = new Set<string>(x.filter((v) => !blocked.has(v)))
  const queue = [...seen]
  while (queue.length > 0) {
    const v = queue.shift() as string
    if (y.includes(v)) return false
    for (const w of adjacency.get(v) ?? []) {
      if (!seen.has(w) && !blocked.has(w)) {
        seen.add(w)
        queue.push(w)
      }
    }
  }
  return true
}
