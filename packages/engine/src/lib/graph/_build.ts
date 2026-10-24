// Graph assembly shared by the constructors and transformations of the graph domain (not re-exported).
import type { Json } from '../base/index.ts'
import type { Components } from './components.ts'
import type { Adjacency, Edge, Graph } from './graph.ts'

/** Sorted adjacency of one side ('both' for undirected graphs). */
export function adjacency(n: number, edges: readonly Edge[], side: 'out' | 'in' | 'both'): Adjacency {
  const count = new Int32Array(n + 1)
  const ends = (e: Edge): [number, number][] => {
    if (side === 'out') return [[e.source, e.target]]
    if (side === 'in') return [[e.target, e.source]]
    return e.source === e.target
      ? [[e.source, e.target]]
      : [
          [e.source, e.target],
          [e.target, e.source],
        ]
  }
  for (const e of edges) for (const [a] of ends(e)) count[a + 1] = (count[a + 1] ?? 0) + 1
  for (let v = 0; v < n; v++) count[v + 1] = (count[v + 1] ?? 0) + (count[v] ?? 0)
  const fill = Int32Array.from(count)
  const m = count[n] ?? 0
  const indices = new Int32Array(m)
  const ids = new Int32Array(m)
  for (const [k, e] of edges.entries()) {
    for (const [a, b] of ends(e)) {
      const at = fill[a] ?? 0
      indices[at] = b
      ids[at] = k
      fill[a] = at + 1
    }
  }
  const weights = new Float64Array(m)
  for (let v = 0; v < n; v++) {
    const lo = count[v] ?? 0
    const hi = count[v + 1] ?? 0
    const row = Array.from({ length: hi - lo }, (_, i) => lo + i).sort(
      (p, q) => (indices[p] ?? 0) - (indices[q] ?? 0) || (ids[p] ?? 0) - (ids[q] ?? 0),
    )
    const nb = row.map((p) => indices[p] ?? 0)
    const id = row.map((p) => ids[p] ?? 0)
    for (let i = 0; i < row.length; i++) {
      indices[lo + i] = nb[i] ?? 0
      ids[lo + i] = id[i] ?? 0
      weights[lo + i] = edges[id[i] ?? 0]?.weight ?? 1
    }
  }
  return { indptr: count, indices, weights, edges: ids }
}

/** Assemble a graph from resolved parts (shared by the constructors and transformations). */
export function assemble(
  directed: boolean,
  labels: readonly string[],
  edges: readonly Edge[],
  edgeData: readonly Readonly<Record<string, Json>>[],
  attributes: Readonly<Record<string, Json>>,
  nodeData: readonly Readonly<Record<string, Json>>[],
): Graph {
  const n = labels.length
  const out = adjacency(n, edges, directed ? 'out' : 'both')
  const index = new Map<string, number>()
  for (const [i, l] of labels.entries()) if (!index.has(l)) index.set(l, i)
  return {
    kind: 'graph',
    directed,
    n,
    labels,
    edges,
    out,
    in: directed ? adjacency(n, edges, 'in') : out,
    attributes,
    nodeData: Array.from({ length: n }, (_, i) => nodeData[i] ?? {}),
    edgeData,
    index,
  }
}

/** The canonical partition (components numbered by their smallest node, members sorted) of any labelling. */
export function partitionFromLabels(raw: ArrayLike<number>): Components {
  const n = raw.length
  const relabel = new Map<number, number>()
  const labels = new Int32Array(n)
  const components: number[][] = []
  for (let v = 0; v < n; v++) {
    const r = raw[v] ?? 0
    let c = relabel.get(r)
    if (c === undefined) {
      c = components.length
      relabel.set(r, c)
      components.push([])
    }
    labels[v] = c
    components[c]?.push(v)
  }
  return { count: components.length, labels, components }
}
