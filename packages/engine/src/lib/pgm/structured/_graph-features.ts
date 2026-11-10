// The features of a graph CRF as visitors: which parameters a node with a label, an edge with a label pair or a clique
// with a label tuple switches on, and with what value. The score, the feature counts, the potentials, the expected
// counts and the pseudo-likelihood are all sums over these visits, so they cannot disagree (not re-exported).
import { OmaiError } from '../../base/index.ts'
import type { ItemFeatures } from './crf.ts'
import type { GraphClique, GraphCrf, GraphEdge, GraphInput, GraphLabelling } from './graph-crf.ts'

/** Receives (parameter index, feature value). */
export type Visit = (index: number, value: number) => void

interface Lookups {
  readonly nodes: ReadonlyMap<string, number>
  readonly types: ReadonlyMap<string, number>
  readonly attributes: ReadonlyMap<string, number>
  readonly cliques: ReadonlyMap<string, number>
  readonly symmetric: readonly boolean[]
}

const cache = new WeakMap<object, Lookups>()

function lookups(model: GraphCrf): Lookups {
  const hit = cache.get(model.layout)
  if (hit !== undefined) return hit
  const shift = model.bias ? 1 : 0
  const made: Lookups = {
    nodes: new Map(model.nodeAttributes.map((a, i) => [a, i + shift])),
    types: new Map(model.edgeTypes.map((t, i) => [t, i])),
    attributes: new Map(model.edgeAttributes.map((a, i) => [a, i])),
    cliques: new Map(model.cliqueTypes.map((c, i) => [c.name, i])),
    symmetric: model.edgeTypes.map((t) => model.symmetric.includes(t)),
  }
  cache.set(model.layout, made)
  return made
}

/** The index of an upper-triangle entry (i <= j) in a row-by-row packing. */
function packed(K: number, i: number, j: number): number {
  const a = Math.min(i, j)
  const b = Math.max(i, j)
  return a * K - (a * (a - 1)) / 2 + (b - a)
}

/** The edge type's index in the model (the default type is `'edge'`). */
export function edgeTypeIndex(model: GraphCrf, edge: GraphEdge): number {
  const t = lookups(model).types.get(edge.type ?? 'edge')
  if (t === undefined) throw new OmaiError('value', `graph CRF: unknown edge type ${edge.type ?? 'edge'}`)
  return t
}

/** The clique type's index in the model. */
export function cliqueTypeIndex(model: GraphCrf, clique: GraphClique): number {
  const t = lookups(model).cliques.get(clique.type)
  if (t === undefined) throw new OmaiError('value', `graph CRF: unknown clique type ${clique.type}`)
  const arity = model.cliqueTypes[t]?.arity ?? 0
  if (clique.nodes.length !== arity)
    throw new OmaiError('shape', `graph CRF: a ${clique.type} clique has ${arity} nodes, got ${clique.nodes.length}`)
  return t
}

/** Check an input against the model: node indices in range, distinct ends and clique members, known types. */
export function checkGraphInput(model: GraphCrf, x: GraphInput): void {
  const n = x.nodes.length
  const inRange = (i: number): boolean => Number.isInteger(i) && i >= 0 && i < n
  for (const e of x.edges) {
    if (!inRange(e.from) || !inRange(e.to))
      throw new OmaiError('value', `graph CRF: edge ${e.from}-${e.to} is out of range`)
    if (e.from === e.to) throw new OmaiError('value', `graph CRF: self-loop at node ${e.from}`)
    edgeTypeIndex(model, e)
  }
  for (const c of x.cliques ?? []) {
    cliqueTypeIndex(model, c)
    if (!c.nodes.every(inRange) || new Set(c.nodes).size !== c.nodes.length)
      throw new OmaiError('value', `graph CRF: clique [${c.nodes.join(', ')}] needs distinct nodes in range`)
  }
}

/** Label indices of a labelling given as names or indices, one per node. */
export function graphLabelIndices(model: GraphCrf, y: GraphLabelling, n: number): number[] {
  if (y.length !== n) throw new OmaiError('shape', `graph CRF: ${y.length} labels for ${n} nodes`)
  const K = model.labels.length
  return y.map((l) => {
    const i = typeof l === 'number' ? l : model.labels.indexOf(l)
    if (!(Number.isInteger(i) && i >= 0 && i < K)) throw new OmaiError('value', `graph CRF: unknown label ${String(l)}`)
    return i
  })
}

/** The node features of a node with attribute record `item` and label `k`. */
export function visitNode(model: GraphCrf, item: ItemFeatures, k: number, visit: Visit): void {
  const K = model.layout.labels
  if (model.bias) visit(k, 1)
  const index = lookups(model).nodes
  for (const [a, v] of Object.entries(item)) {
    const i = index.get(a)
    if (i !== undefined) visit(i * K + k, v)
  }
}

/** The features of an edge whose ends carry labels `yf` and `yt`: its type's weight and its attributes'. */
export function visitEdge(model: GraphCrf, edge: GraphEdge, yf: number, yt: number, visit: Visit): void {
  const K = model.layout.labels
  const look = lookups(model)
  const t = edgeTypeIndex(model, edge)
  const base = model.layout.pairOffsets[t] ?? 0
  visit(base + (look.symmetric[t] ? packed(K, yf, yt) : yf * K + yt), 1)
  if (edge.attributes === undefined) return
  const block = model.symmetricAttributes ? (K * (K + 1)) / 2 : K * K
  for (const [b, v] of Object.entries(edge.attributes)) {
    const a = look.attributes.get(b)
    if (a === undefined) continue
    visit(model.layout.attributeOffset + a * block + (model.symmetricAttributes ? packed(K, yf, yt) : yf * K + yt), v)
  }
}

/** The index of a label tuple of a clique in its type's block (row-major in the clique's node order). */
export function cliqueEntry(K: number, labels: readonly number[]): number {
  let idx = 0
  for (const l of labels) idx = idx * K + l
  return idx
}

/** The feature of a clique whose nodes carry the labels `ys` (in the clique's node order). */
export function visitClique(model: GraphCrf, clique: GraphClique, ys: readonly number[], visit: Visit): void {
  const t = cliqueTypeIndex(model, clique)
  visit((model.layout.cliqueOffsets[t] ?? 0) + cliqueEntry(model.layout.labels, ys), 1)
}
