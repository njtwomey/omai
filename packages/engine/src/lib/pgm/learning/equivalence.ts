/**
 * Markov equivalence of DAGs, the classes of graphs that encode the same conditional independencies and so cannot be
 * told apart from observational data: PDAGs, skeletons and immoralities, the equivalence test, Meek's orientation
 * rules, the CPDAG (essential graph) of a DAG, a DAG extension of a PDAG, and the structural Hamming distance between
 * graphs.
 */

import { OmaiError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { type Dag, type DirectedModel, dag, dagEdges, type UndirectedGraph, undirectedGraph } from '../graphs.ts'

/**
 * A partially directed acyclic graph: some edges directed, the rest undirected. A CPDAG (essential graph) represents a
 * Markov equivalence class of DAGs this way.
 *
 * @concept markov-equivalence
 */
export interface Pdag {
  readonly kind: 'pdag'
  readonly nodes: readonly string[]
  /** Directed edges [from, to]. */
  readonly directed: readonly (readonly [string, string])[]
  /** Undirected edges, each once as a sorted pair. */
  readonly undirected: readonly (readonly [string, string])[]
}

const pair = (a: string, b: string): [string, string] => (a < b ? [a, b] : [b, a])
const sortEdges = (es: Iterable<readonly [string, string]>): [string, string][] =>
  [...es].map(([a, b]): [string, string] => [a, b]).sort((x, y) => x[0].localeCompare(y[0]) || x[1].localeCompare(y[1]))

/**
 * Build a PDAG from directed and undirected edges.
 *
 * @param nodes - The nodes (others are added from the edges).
 * @param directed - Directed edges.
 * @param undirected - Undirected edges.
 * @returns The PDAG, edges sorted.
 * @throws OmaiError with code `value` when a pair has both a directed and an undirected edge.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { pdag } from 'omai-engine/pgm/learning'
 * pdag(['a'], [['a', 'b']], [['c', 'b']]).undirected // => [['b', 'c']]
 * ```
 */
export function pdag(
  nodes: readonly string[],
  directed: readonly (readonly [string, string])[],
  undirected: readonly (readonly [string, string])[] = [],
): Pdag {
  const all = new Set(nodes)
  for (const [a, b] of [...directed, ...undirected]) {
    all.add(a)
    all.add(b)
  }
  const u = new Map(undirected.map(([a, b]) => [pair(a, b).join('\u0000'), pair(a, b)]))
  for (const [a, b] of directed) {
    if (u.has(pair(a, b).join('\u0000')))
      throw new OmaiError('value', `pdag: ${a}-${b} is both directed and undirected`)
  }
  return {
    kind: 'pdag',
    nodes: [...all].sort(),
    directed: sortEdges(new Map(directed.map(([a, b]) => [`${a}\u0000${b}`, [a, b] as const])).values()),
    undirected: sortEdges(u.values()),
  }
}

function adjacency(g: DirectedModel | Pdag): UndirectedGraph {
  if ('kind' in g && g.kind === 'pdag') return undirectedGraph([...g.directed, ...g.undirected], g.nodes)
  const d = g as DirectedModel
  return undirectedGraph(
    dagEdges(d),
    d.nodes.map((n) => n.name),
  )
}

/**
 * The skeleton of a DAG or PDAG: its edges without directions.
 *
 * @param g - The graph.
 * @returns The undirected graph.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dag, graphEdges } from 'omai-engine/pgm'
 * import { skeleton } from 'omai-engine/pgm/learning'
 * graphEdges(skeleton(dag([['b', 'a']]))) // => [['a', 'b']]
 * ```
 */
export function skeleton(g: DirectedModel | Pdag): UndirectedGraph {
  return adjacency(g)
}

/**
 * The immoralities (unshielded colliders) of a DAG: triples a -> c <- b with a and b not adjacent (pgmpy's
 * `get_immoralities`).
 *
 * @param g - The DAG.
 * @returns Triples [a, c, b] with a < b, sorted.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dag } from 'omai-engine/pgm'
 * import { immoralities } from 'omai-engine/pgm/learning'
 * immoralities(dag([['a', 'c'], ['b', 'c'], ['c', 'd']])) // => [['a', 'c', 'b']]
 * ```
 */
export function immoralities(g: DirectedModel): [string, string, string][] {
  const adj = adjacency(g)
  const out: [string, string, string][] = []
  for (const n of g.nodes) {
    const ps = [...n.parents].sort()
    for (let i = 0; i < ps.length; i++) {
      for (let j = i + 1; j < ps.length; j++) {
        const a = ps[i] as string
        const b = ps[j] as string
        if (!adj.get(a)?.has(b)) out.push([a, n.name, b])
      }
    }
  }
  return out.sort((x, y) => x.join('\u0000').localeCompare(y.join('\u0000')))
}

/**
 * Whether two DAGs are Markov equivalent (encode the same independencies): the same skeleton and the same immoralities
 * (Verma and Pearl 1990; pgmpy's `is_iequivalent`).
 *
 * @param a - First DAG.
 * @param b - Second DAG.
 * @returns `true` when equivalent.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dag } from 'omai-engine/pgm'
 * import { isMarkovEquivalent } from 'omai-engine/pgm/learning'
 * [isMarkovEquivalent(dag([['a', 'b'], ['b', 'c']]), dag([['c', 'b'], ['b', 'a']])), isMarkovEquivalent(dag([['a', 'b'], ['c', 'b']]), dag([['b', 'a'], ['b', 'c']]))] // => [true, false]
 * ```
 */
export function isMarkovEquivalent(a: DirectedModel, b: DirectedModel): boolean {
  const ea = JSON.stringify(sortEdges(dagEdges(a).map(([x, y]) => pair(x, y))))
  const eb = JSON.stringify(sortEdges(dagEdges(b).map(([x, y]) => pair(x, y))))
  return ea === eb && JSON.stringify(immoralities(a)) === JSON.stringify(immoralities(b))
}

/** Mutable PDAG state for the orientation rules. */
class Mutable {
  readonly directed = new Set<string>()
  readonly undirected = new Set<string>()
  readonly nodes: readonly string[]
  constructor(nodes: readonly string[]) {
    this.nodes = nodes
  }
  key(a: string, b: string): string {
    return `${a}\u0000${b}`
  }
  ukey(a: string, b: string): string {
    return pair(a, b).join('\u0000')
  }
  hasDirected(a: string, b: string): boolean {
    return this.directed.has(this.key(a, b))
  }
  hasUndirected(a: string, b: string): boolean {
    return this.undirected.has(this.ukey(a, b))
  }
  adjacent(a: string, b: string): boolean {
    return this.hasDirected(a, b) || this.hasDirected(b, a) || this.hasUndirected(a, b)
  }
  orient(a: string, b: string): void {
    this.undirected.delete(this.ukey(a, b))
    this.directed.add(this.key(a, b))
  }
  undirectedNeighbours(a: string): string[] {
    return this.nodes.filter((b) => b !== a && this.hasUndirected(a, b))
  }
  parents(a: string): string[] {
    return this.nodes.filter((b) => this.hasDirected(b, a))
  }
  children(a: string): string[] {
    return this.nodes.filter((b) => this.hasDirected(a, b))
  }
  freeze(): Pdag {
    const split = (k: string): [string, string] => k.split('\u0000') as [string, string]
    return pdag(this.nodes, [...this.directed].map(split), [...this.undirected].map(split))
  }
}

const splitKey = (k: string): [string, string] => k.split('\u0000') as [string, string]

/** The edges of a mutable PDAG as plain pairs, in insertion order. */
function partsOf(m: Mutable): { directed: [string, string][]; undirected: [string, string][] } {
  return { directed: [...m.directed].map(splitKey), undirected: [...m.undirected].map(splitKey) }
}

/** A mutable PDAG from plain pairs. */
function fromParts(
  nodes: readonly string[],
  directed: readonly (readonly [string, string])[],
  undirected: readonly (readonly [string, string])[],
): Mutable {
  const m = new Mutable(nodes)
  for (const [a, b] of directed) m.directed.add(m.key(a, b))
  for (const [a, b] of undirected) m.undirected.add(m.ukey(a, b))
  return m
}

/** The paths the orientation algorithms of this module log, as a type. */
export type MeekPath = 'meek/rule' | 'meek/oriented' | 'meek/undirected' | 'meek/node'

/**
 * What Meek's rules and the DAG extension log, by path: the rule just swept (1 to 4), the edges it oriented as flat
 * `[from, to]` pairs of node indices (in the sorted node order), how many undirected edges remain, and, for the DAG
 * extension, the index of the node it removed.
 */
export const meekContract: Contract<unknown, MeekPath> = defineContract({
  'meek/rule': scalar({
    description: 'the rule of Meek swept at this step, 1 to 4 (R4 only with background knowledge)',
  }),
  'meek/oriented': tensor({
    role: 'state',
    shape: ['pairs'],
    description: 'the edges the step oriented, as flat [from, to] pairs of node indices',
  }),
  'meek/undirected': scalar({ description: 'undirected edges still in the graph after the step' }),
  'meek/node': scalar({ description: 'the node removed at this step of the DAG extension, as its index' }),
})

/** The state of {@link meekRulesAlgorithm}. */
export interface MeekState {
  readonly nodes: readonly string[]
  /** Directed edges so far. */
  readonly directed: readonly (readonly [string, string])[]
  /** Undirected edges so far. */
  readonly undirected: readonly (readonly [string, string])[]
  /** The rule the next step sweeps (1 to 4). */
  readonly rule: number
  /** Whether a rule fired earlier in the current cycle of rules. */
  readonly changed: boolean
  /** The rule swept at the last step (0 before the first). */
  readonly lastRule: number
  /** The edges the last step oriented. */
  readonly lastOriented: readonly (readonly [string, string])[]
  /** Sweeps made. */
  readonly sweeps: number
  /** True once a whole cycle of rules oriented nothing. */
  readonly finished: boolean
}

/** One sweep of a Meek rule over the graph; returns the edges it oriented (as from, to). */
type MeekSweep = (m: Mutable) => [string, string][]

const meekSweeps: readonly MeekSweep[] = [
  // R1: a -> b - c, a and c not adjacent: b -> c.
  (m) => {
    const out: [string, string][] = []
    for (const b of m.nodes) {
      for (const a of m.parents(b)) {
        for (const c of m.undirectedNeighbours(b)) {
          if (c !== a && !m.adjacent(a, c) && m.hasUndirected(b, c)) {
            m.orient(b, c)
            out.push([b, c])
          }
        }
      }
    }
    return out
  },
  // R2: a -> c -> b and a - b: a -> b.
  (m) => {
    const out: [string, string][] = []
    for (const c of m.nodes) {
      for (const a of m.parents(c)) {
        for (const b of m.children(c)) {
          if (m.hasUndirected(a, b)) {
            m.orient(a, b)
            out.push([a, b])
          }
        }
      }
    }
    return out
  },
  // R3: a - c -> b, a - d -> b with c, d not adjacent and a - b: a -> b.
  (m) => {
    const out: [string, string][] = []
    for (const a of m.nodes) {
      const nbs = m.undirectedNeighbours(a)
      for (const b of nbs) {
        const via = nbs.filter((c) => c !== b && m.hasDirected(c, b))
        let applies = false
        for (let i = 0; i < via.length && !applies; i++)
          for (let j = i + 1; j < via.length && !applies; j++)
            if (!m.adjacent(via[i] as string, via[j] as string)) applies = true
        if (applies && m.hasUndirected(a, b)) {
          m.orient(a, b)
          out.push([a, b])
        }
      }
    }
    return out
  },
  // R4: d -> c -> b, a - b, a - d, a adjacent to c, b and d not adjacent: a -> b.
  (m) => {
    const out: [string, string][] = []
    for (const c of m.nodes) {
      for (const b of m.children(c)) {
        for (const d of m.parents(c)) {
          if (b === d || m.adjacent(b, d)) continue
          for (const a of m.undirectedNeighbours(b)) {
            if (m.hasUndirected(a, d) && m.adjacent(a, c) && m.hasUndirected(a, b)) {
              m.orient(a, b)
              out.push([a, b])
            }
          }
        }
      }
    }
    return out
  },
]

/**
 * Meek's orientation rules as a steppable algorithm: each step sweeps one rule (R1, R2, R3 and, with `r4`, R4, in
 * turn) over the whole graph, orienting every edge it can, and logs the rule, the edges it oriented and the undirected
 * edges left. It is done when a whole cycle of rules orients nothing.
 *
 * @param p - The PDAG.
 * @param options - `r4` to include rule 4 (needed only with background knowledge).
 * @returns The algorithm.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { meekRulesAlgorithm, pdag } from 'omai-engine/pgm/learning'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(meekRulesAlgorithm(pdag([], [['a', 'b']], [['b', 'c']])), { maxSteps: 20 })
 * [run.state.directed, run.state.finished] // => [[['a', 'b'], ['b', 'c']], true]
 * ```
 */
export function meekRulesAlgorithm(p: Pdag, options: { readonly r4?: boolean } = {}): Algorithm<MeekState, MeekPath> {
  const rules = options.r4 === true ? 4 : 3
  const index = new Map(p.nodes.map((v, i) => [v, i]))
  return {
    name: 'pgm/meek-rules',
    config: { nodes: p.nodes.length, r4: options.r4 === true },
    init: () => ({
      nodes: p.nodes,
      directed: p.directed,
      undirected: p.undirected,
      rule: 1,
      changed: false,
      lastRule: 0,
      lastOriented: [],
      sweeps: 0,
      finished: false,
    }),
    step: (s, ctx) => {
      if (s.finished) return s
      const m = fromParts(s.nodes, s.directed, s.undirected)
      const oriented = (meekSweeps[s.rule - 1] as MeekSweep)(m)
      const changed = s.changed || oriented.length > 0
      const wrap = s.rule === rules
      const parts = partsOf(m)
      if (ctx.wants('meek/rule')) ctx.log('meek/rule', s.rule)
      if (ctx.wants('meek/oriented'))
        ctx.log(
          'meek/oriented',
          Int32Array.from(oriented.flatMap(([a, b]) => [index.get(a) ?? -1, index.get(b) ?? -1])),
        )
      if (ctx.wants('meek/undirected')) ctx.log('meek/undirected', parts.undirected.length)
      return {
        nodes: s.nodes,
        directed: parts.directed,
        undirected: parts.undirected,
        rule: wrap ? 1 : s.rule + 1,
        changed: wrap ? false : changed,
        lastRule: s.rule,
        lastOriented: oriented,
        sweeps: s.sweeps + 1,
        finished: wrap && !changed,
      }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MeekState,
  }
}

/**
 * Apply Meek's orientation rules to a PDAG until none applies (Meek 1995): R1 a -> b - c, a and c not adjacent
 * gives b -> c; R2 a -> c -> b and a - b gives a -> b; R3 a - c -> b, a - d -> b with c, d not adjacent and a - b
 * gives a -> b; and optionally R4 d -> c -> b, a - b, a - d, a adjacent to c, b and d not adjacent gives a -> b
 * (needed only with background knowledge). This is {@link meekRulesAlgorithm} run to the end.
 *
 * @param p - The PDAG.
 * @param options - `r4` to include rule 4.
 * @returns The maximally oriented PDAG.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { pdag, meekRules } from 'omai-engine/pgm/learning'
 * meekRules(pdag([], [['a', 'b']], [['b', 'c']])).directed // => [['a', 'b'], ['b', 'c']]
 * ```
 */
export function meekRules(p: Pdag, options: { readonly r4?: boolean } = {}): Pdag {
  const { state } = runAlgorithm(meekRulesAlgorithm(p, options), {
    maxSteps: 4 * (p.undirected.length + 2) + 4,
  })
  return pdag(state.nodes, state.directed, state.undirected)
}

/** A DAG's skeleton with its immoralities oriented: the PDAG Meek's rules complete into the CPDAG. */
function immoralityPdag(g: DirectedModel): Pdag {
  const names = g.nodes.map((n) => n.name)
  const im = immoralities(g)
  const oriented = new Set(im.flatMap(([a, c, b]) => [`${a}\u0000${c}`, `${b}\u0000${c}`]))
  const directed: [string, string][] = []
  const undirected: [string, string][] = []
  for (const [a, b] of dagEdges(g)) {
    if (oriented.has(`${a}\u0000${b}`)) directed.push([a, b])
    else undirected.push(pair(a, b))
  }
  return pdag(names, directed, undirected)
}

/**
 * The CPDAG construction as a steppable algorithm: the DAG's skeleton with its immoralities oriented, completed by
 * Meek's rules one sweep per step (see {@link meekRulesAlgorithm}).
 *
 * @param g - The DAG.
 * @returns The algorithm; its state's `directed` and `undirected` are the CPDAG once it is done.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dag } from 'omai-engine/pgm'
 * import { cpdagAlgorithm } from 'omai-engine/pgm/learning'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(cpdagAlgorithm(dag([['a', 'b'], ['b', 'c']])), { maxSteps: 20 }).state.undirected // => [['a', 'b'], ['b', 'c']]
 * ```
 */
export function cpdagAlgorithm(g: DirectedModel): Algorithm<MeekState, MeekPath> {
  return meekRulesAlgorithm(immoralityPdag(g))
}

/**
 * The CPDAG (essential graph) of a DAG: its skeleton with the immoralities oriented and Meek's rules applied, so the
 * directed edges are exactly those shared by every DAG in its Markov equivalence class (Chickering 1995; pgmpy's
 * `to_pdag`).
 *
 * @param g - The DAG.
 * @returns The CPDAG.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dag } from 'omai-engine/pgm'
 * import { cpdag } from 'omai-engine/pgm/learning'
 * const c = cpdag(dag([['a', 'c'], ['b', 'c'], ['c', 'd']]))
 * [c.directed, c.undirected] // => [[['a', 'c'], ['b', 'c'], ['c', 'd']], []]
 * ```
 */
export function cpdag(g: DirectedModel): Pdag {
  return meekRules(immoralityPdag(g))
}

/** The state of {@link pdagToDagAlgorithm}. */
export interface DagExtensionState {
  readonly nodes: readonly string[]
  /** The nodes not yet removed, ascending. */
  readonly remaining: readonly string[]
  /** The directed edges so far: the PDAG's and those oriented into removed nodes. */
  readonly directed: readonly (readonly [string, string])[]
  /** The undirected edges still to orient. */
  readonly undirected: readonly (readonly [string, string])[]
  /** The node removed at the last step ('' before the first). */
  readonly node: string
}

/**
 * The consistent DAG extension of a PDAG (Dor and Tarsi 1992) as a steppable algorithm: each step removes a node with
 * no outgoing directed edge whose undirected neighbours are adjacent to all its other neighbours, orienting its
 * undirected edges into it. A PDAG with no consistent extension throws while stepping.
 *
 * @param p - The PDAG.
 * @returns The algorithm; its state's `directed` is the DAG's edge list once it is done.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { pdag, pdagToDagAlgorithm } from 'omai-engine/pgm/learning'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(pdagToDagAlgorithm(pdag([], [], [['a', 'b']])), { maxSteps: 5 }).state.directed // => [['b', 'a']]
 * ```
 */
export function pdagToDagAlgorithm(p: Pdag): Algorithm<DagExtensionState, MeekPath> {
  const index = new Map(p.nodes.map((v, i) => [v, i]))
  return {
    name: 'pgm/pdag-to-dag',
    config: { nodes: p.nodes.length },
    init: () => ({
      nodes: p.nodes,
      remaining: [...p.nodes].sort(),
      directed: p.directed,
      undirected: p.undirected,
      node: '',
    }),
    step: (s, ctx) => {
      if (s.remaining.length === 0) return s
      const m = fromParts(s.nodes, s.directed, s.undirected)
      const remaining = new Set(s.remaining)
      const alive = (v: string): boolean => remaining.has(v)
      let pick: string | undefined
      for (const x of s.remaining) {
        if (m.children(x).some(alive)) continue
        const nbs = m.nodes.filter((y) => y !== x && alive(y) && m.adjacent(x, y))
        const und = m.undirectedNeighbours(x).filter(alive)
        if (und.every((y) => nbs.every((z) => z === y || m.adjacent(y, z)))) {
          pick = x
          break
        }
      }
      if (pick === undefined) throw new OmaiError('value', 'pdagToDag: the PDAG has no consistent extension')
      const oriented: [string, string][] = []
      for (const y of m.undirectedNeighbours(pick).filter(alive)) {
        m.orient(y, pick)
        oriented.push([y, pick])
      }
      const parts = partsOf(m)
      if (ctx.wants('meek/node')) ctx.log('meek/node', index.get(pick) ?? -1)
      if (ctx.wants('meek/oriented'))
        ctx.log(
          'meek/oriented',
          Int32Array.from(oriented.flatMap(([a, b]) => [index.get(a) ?? -1, index.get(b) ?? -1])),
        )
      if (ctx.wants('meek/undirected')) ctx.log('meek/undirected', parts.undirected.length)
      return {
        nodes: s.nodes,
        remaining: s.remaining.filter((v) => v !== pick),
        directed: parts.directed,
        undirected: parts.undirected,
        node: pick,
      }
    },
    done: (s) => (s.remaining.length === 0 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DagExtensionState,
  }
}

/**
 * A DAG in the equivalence class of a PDAG (a consistent extension; Dor and Tarsi 1992): repeatedly remove a node that
 * has no outgoing directed edge and whose undirected neighbours are adjacent to all its other neighbours, orienting
 * its undirected edges into it. This is {@link pdagToDagAlgorithm} run to the end.
 *
 * @param p - The PDAG.
 * @returns A DAG with the same skeleton, the same orientations and no new immoralities.
 * @throws OmaiError with code `value` when the PDAG has no consistent extension.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dagEdges } from 'omai-engine/pgm'
 * import { pdag, pdagToDag } from 'omai-engine/pgm/learning'
 * dagEdges(pdagToDag(pdag([], [], [['a', 'b'], ['b', 'c']]))).length // => 2
 * ```
 */
export function pdagToDag(p: Pdag): Dag {
  const { state } = runAlgorithm(pdagToDagAlgorithm(p), { maxSteps: p.nodes.length + 1 })
  return dag(
    state.directed.map(([a, b]): [string, string] => [a, b]),
    state.nodes,
  )
}

/**
 * The structural Hamming distance between two graphs (DAGs or PDAGs): the number of node pairs whose connection
 * differs (missing, extra, or differently oriented, an undirected edge counting as one orientation).
 *
 * @param a - First graph.
 * @param b - Second graph.
 * @returns The distance.
 * @concept structure-learning
 * @example
 * ```ts
 * import { dag } from 'omai-engine/pgm'
 * import { structuralHammingDistance } from 'omai-engine/pgm/learning'
 * structuralHammingDistance(dag([['a', 'b'], ['b', 'c']]), dag([['b', 'a'], ['a', 'c']], ['b'])) // => 3
 * ```
 */
export function structuralHammingDistance(a: DirectedModel | Pdag, b: DirectedModel | Pdag): number {
  const marks = (g: DirectedModel | Pdag): Map<string, string> => {
    const out = new Map<string, string>()
    const directed = 'kind' in g && g.kind === 'pdag' ? g.directed : dagEdges(g as DirectedModel)
    for (const [x, y] of directed) out.set(pair(x, y).join('\u0000'), `${x}>${y}`)
    if ('kind' in g && g.kind === 'pdag') for (const [x, y] of g.undirected) out.set(pair(x, y).join('\u0000'), '-')
    return out
  }
  const ma = marks(a)
  const mb = marks(b)
  let d = 0
  for (const k of new Set([...ma.keys(), ...mb.keys()])) if (ma.get(k) !== mb.get(k)) d++
  return d
}
