/**
 * Markov equivalence of DAGs, the classes of graphs that encode the same conditional independencies and so cannot be
 * told apart from observational data: PDAGs, skeletons and immoralities, the equivalence test, Meek's orientation
 * rules, the CPDAG (essential graph) of a DAG, a DAG extension of a PDAG, and the structural Hamming distance between
 * graphs.
 */

import { WhiteboxError } from '../../base/index.ts'
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
 * @throws WhiteboxError with code `value` when a pair has both a directed and an undirected edge.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { pdag } from 'whitebox-engine/pgm/learning'
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
      throw new WhiteboxError('value', `pdag: ${a}-${b} is both directed and undirected`)
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
 * import { dag, graphEdges } from 'whitebox-engine/pgm'
 * import { skeleton } from 'whitebox-engine/pgm/learning'
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
 * import { dag } from 'whitebox-engine/pgm'
 * import { immoralities } from 'whitebox-engine/pgm/learning'
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
 * import { dag } from 'whitebox-engine/pgm'
 * import { isMarkovEquivalent } from 'whitebox-engine/pgm/learning'
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

function mutable(p: Pdag): Mutable {
  const m = new Mutable(p.nodes)
  for (const [a, b] of p.directed) m.directed.add(m.key(a, b))
  for (const [a, b] of p.undirected) m.undirected.add(m.ukey(a, b))
  return m
}

/**
 * Apply Meek's orientation rules to a PDAG until none applies (Meek 1995): R1 a -> b - c, a and c not adjacent
 * gives b -> c; R2 a -> c -> b and a - b gives a -> b; R3 a - c -> b, a - d -> b with c, d not adjacent and a - b
 * gives a -> b; and optionally R4 d -> c -> b, a - b, a - d, a adjacent to c, b and d not adjacent gives a -> b
 * (needed only with background knowledge).
 *
 * @param p - The PDAG.
 * @param options - `r4` to include rule 4.
 * @returns The maximally oriented PDAG.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { pdag, meekRules } from 'whitebox-engine/pgm/learning'
 * meekRules(pdag([], [['a', 'b']], [['b', 'c']])).directed // => [['a', 'b'], ['b', 'c']]
 * ```
 */
export function meekRules(p: Pdag, options: { readonly r4?: boolean } = {}): Pdag {
  const m = mutable(p)
  let changed = true
  while (changed) {
    changed = false
    for (const b of m.nodes) {
      for (const a of m.parents(b)) {
        for (const c of m.undirectedNeighbours(b)) {
          if (c !== a && !m.adjacent(a, c)) {
            m.orient(b, c)
            changed = true
          }
        }
      }
    }
    for (const c of m.nodes) {
      for (const a of m.parents(c)) {
        for (const b of m.children(c)) {
          if (m.hasUndirected(a, b)) {
            m.orient(a, b)
            changed = true
          }
        }
      }
    }
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
          changed = true
        }
      }
    }
    if (options.r4 === true) {
      for (const c of m.nodes) {
        for (const b of m.children(c)) {
          for (const d of m.parents(c)) {
            if (b === d || m.adjacent(b, d)) continue
            for (const a of m.undirectedNeighbours(b)) {
              if (m.hasUndirected(a, d) && m.adjacent(a, c)) {
                m.orient(a, b)
                changed = true
              }
            }
          }
        }
      }
    }
  }
  return m.freeze()
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
 * import { dag } from 'whitebox-engine/pgm'
 * import { cpdag } from 'whitebox-engine/pgm/learning'
 * const c = cpdag(dag([['a', 'c'], ['b', 'c'], ['c', 'd']]))
 * [c.directed, c.undirected] // => [[['a', 'c'], ['b', 'c'], ['c', 'd']], []]
 * ```
 */
export function cpdag(g: DirectedModel): Pdag {
  const names = g.nodes.map((n) => n.name)
  const im = immoralities(g)
  const oriented = new Set(im.flatMap(([a, c, b]) => [`${a}\u0000${c}`, `${b}\u0000${c}`]))
  const directed: [string, string][] = []
  const undirected: [string, string][] = []
  for (const [a, b] of dagEdges(g)) {
    if (oriented.has(`${a}\u0000${b}`)) directed.push([a, b])
    else undirected.push(pair(a, b))
  }
  return meekRules(pdag(names, directed, undirected))
}

/**
 * A DAG in the equivalence class of a PDAG (a consistent extension; Dor and Tarsi 1992): repeatedly remove a node that
 * has no outgoing directed edge and whose undirected neighbours are adjacent to all its other neighbours, orienting
 * its undirected edges into it.
 *
 * @param p - The PDAG.
 * @returns A DAG with the same skeleton, the same orientations and no new immoralities.
 * @throws WhiteboxError with code `value` when the PDAG has no consistent extension.
 * @concept markov-equivalence
 * @example
 * ```ts
 * import { dagEdges } from 'whitebox-engine/pgm'
 * import { pdag, pdagToDag } from 'whitebox-engine/pgm/learning'
 * dagEdges(pdagToDag(pdag([], [], [['a', 'b'], ['b', 'c']]))).length // => 2
 * ```
 */
export function pdagToDag(p: Pdag): Dag {
  const m = mutable(p)
  const remaining = new Set(m.nodes)
  const edges: [string, string][] = [...p.directed].map(([a, b]): [string, string] => [a, b])
  const alive = (v: string): boolean => remaining.has(v)
  while (remaining.size > 0) {
    let pick: string | undefined
    for (const x of [...remaining].sort()) {
      if (m.children(x).some(alive)) continue
      const nbs = m.nodes.filter((y) => y !== x && alive(y) && m.adjacent(x, y))
      const und = m.undirectedNeighbours(x).filter(alive)
      if (und.every((y) => nbs.every((z) => z === y || m.adjacent(y, z)))) {
        pick = x
        break
      }
    }
    if (pick === undefined) throw new WhiteboxError('value', 'pdagToDag: the PDAG has no consistent extension')
    for (const y of m.undirectedNeighbours(pick).filter(alive)) {
      edges.push([y, pick])
      m.orient(y, pick)
    }
    remaining.delete(pick)
  }
  return dag(edges, m.nodes)
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
 * import { dag } from 'whitebox-engine/pgm'
 * import { structuralHammingDistance } from 'whitebox-engine/pgm/learning'
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
