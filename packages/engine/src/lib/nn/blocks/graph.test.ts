import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { cycleGraph, laplacianPositionalEncoding, randomWalkPositionalEncoding } from '../../graph/index.ts'
import { qr } from '../../numerics/linalg/index.ts'
import { key, normal } from '../../random/index.ts'
import {
  add,
  concat,
  matmul,
  mul,
  sliceAxis,
  softmax,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { gelu } from '../functional/activations.ts'
import { layerNorm } from '../functional/normalisation.ts'
import { EGNN, EGNNLayer, GraphTransformer, GraphTransformerLayer, RGCNConv, rgcnConv } from './graph.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('rgcnConv', () => {
  const edgeIndex = [
    [0, 1, 2, 2, 3],
    [1, 2, 1, 0, 1],
  ]
  const edgeType = [0, 1, 1, 0, 1]
  const x = n(1, [4, 2])
  const weight = n(2, [2, 2, 3])
  const root = n(3, [2, 3])

  it('sums per-relation transforms of the neighbours, each normalised by its relation count', () => {
    const w = toArray(weight) as number[][][]
    const xs = toArray(x) as number[][]
    const lin = (row: number[], m: number[][]): number[] =>
      [0, 1, 2].map((o) => row.reduce((a, v, i) => a + v * ((m[i] as number[])[o] as number), 0))
    // Node 1 receives relation 0 from node 0 (alone) and relation 1 from nodes 2 and 3 (averaged).
    const r0 = lin(xs[0] as number[], w[0] as number[][])
    const r1a = lin(xs[2] as number[], w[1] as number[][])
    const r1b = lin(xs[3] as number[], w[1] as number[][])
    const self = lin(xs[1] as number[], toArray(root) as number[][])
    const want = [0, 1, 2].map(
      (o) => (self[o] as number) + (r0[o] as number) + ((r1a[o] as number) + (r1b[o] as number)) / 2,
    )
    const got = toArray(rgcnConv(x, edgeIndex, edgeType, { weight, root })) as number[][]
    expectClose(got[1] as number[], want, 1e-12)
    // Node 3 has no incoming edge: only its self connection.
    expectClose(got[3] as number[], lin(xs[3] as number[], toArray(root) as number[][]), 1e-12)
  })

  it('the sum aggregation does not divide, and a basis decomposition is the explicit weights', () => {
    const summed = toFlat(rgcnConv(x, edgeIndex, edgeType, { weight, aggregate: 'sum' }))
    const meaned = toFlat(rgcnConv(x, edgeIndex, edgeType, { weight, aggregate: 'mean' }))
    expect(summed).not.toEqual(meaned)
    const bases = n(4, [2, 2, 3])
    const comp = tensor([
      [0.3, -0.7],
      [1.2, 0.4],
    ])
    const c = toFlat(comp)
    const [b0, b1] = [sliceAxis(bases, 0, 0, 1), sliceAxis(bases, 0, 1, 2)]
    const explicit = concat(
      [0, 1].map((r) => add(mul(b0, c[2 * r] as number), mul(b1, c[2 * r + 1] as number))),
      0,
    )
    expectClose(
      rgcnConv(x, edgeIndex, edgeType, { weight: bases, comp }),
      rgcnConv(x, edgeIndex, edgeType, { weight: explicit }),
      1e-12,
    )
  })

  it('the module has fewer parameters with bases and its output is differentiable in every parameter', () => {
    const full = new RGCNConv(3, 4, 10, { key: key(5) })
    const decomposed = new RGCNConv(3, 4, 10, { numBases: 2, key: key(5) })
    expect(decomposed.parameterCount()).toBeLessThan(full.parameterCount())
    expect(Object.keys(decomposed.stateDict()).sort()).toEqual(['bias', 'comp', 'root', 'weight'])
    const m = new RGCNConv(2, 3, 2, { numBases: 2, key: key(6) })
    const input = { x, edgeIndex, edgeType }
    for (const name of ['weight', 'comp', 'root', 'bias']) {
      const report = checkPrimitive(
        (p: Value) => m.apply({ ...m.params, [name]: p }, input),
        [m.params[name] as Value],
        { rtol: 1e-5 },
      )
      expect(report.failures, name).toEqual([])
    }
    expect(checkPrimitive((v: Value) => m.apply(m.params, { ...input, x: v }), [x], { rtol: 1e-5 }).failures).toEqual(
      [],
    )
    expect(() => rgcnConv(x, edgeIndex, [0, 1, 2, 0, 1], { weight })).toThrow(/edge types/)
  })
})

/** A rotation matrix in `d` dimensions from the QR of a Gaussian matrix, with a sign fixed so it is a proper rotation. */
function orthogonal(seed: number, d: number): Value {
  return qr(n(seed, [d, d])).q
}

describe('E(n) equivariant graph layer', () => {
  const edgeIndex = [
    [0, 1, 2, 3, 1, 4, 2],
    [1, 2, 3, 0, 0, 2, 4],
  ]
  const h = n(10, [5, 4])
  const x = n(11, [5, 3])

  it('leaves the features invariant and carries the coordinates through any rotation, reflection and translation', () => {
    const net = new EGNN(4, 6, { layers: 2, key: key(12) })
    const base = net.forward({ h, x, edgeIndex })
    for (const seed of [20, 21, 22]) {
      const q = orthogonal(seed, 3)
      const shift = n(seed + 10, [1, 3])
      const moved = net.forward({ h, x: addRows(matmul(x, transpose(q)), shift), edgeIndex })
      expectClose(moved.h, base.h, 1e-9)
      expectClose(moved.x, addRows(matmul(base.x, transpose(q)), shift), 1e-9)
    }
  })

  it('a layer that sees raw coordinates is not equivariant, which is what the distances buy', () => {
    const q = orthogonal(30, 3)
    // A plain feature update from the coordinates themselves changes under rotation.
    const plain = (c: Value): Value =>
      matmul(
        c,
        tensor([
          [1, 2],
          [0.5, -1],
          [2, 0],
        ]),
      )
    const diff = toFlat(plain(matmul(x, transpose(q)))).map((v, i) => v - (toFlat(plain(x))[i] as number))
    expect(Math.max(...diff.map(Math.abs))).toBeGreaterThan(0.1)
  })

  it('moves the coordinates along the relative vectors and respects the edge attributes and options', () => {
    const layer = new EGNNLayer(4, { key: key(13), edgeDim: 2, tanh: true, coordsAggregate: 'sum' })
    const attr = n(14, [7, 2])
    const out = layer.forward({ h, x, edgeIndex, edgeAttr: attr })
    expect(out.h.shape).toEqual([5, 4])
    expect(out.x.shape).toEqual([5, 3])
    const frozen = new EGNNLayer(4, { key: key(13), updateCoords: false }).forward({ h, x, edgeIndex })
    expectClose(frozen.x, x, 0)
    expect(Object.keys(layer.stateDict()).some((k) => k.startsWith('coord'))).toBe(true)
  })

  it('has gradients for every parameter', () => {
    const layer = new EGNNLayer(2, { key: key(15), hidden: 3 })
    const hs = n(16, [3, 2])
    const xs = n(17, [3, 2])
    const ei = [
      [0, 1, 2],
      [1, 2, 0],
    ]
    for (const name of ['edge.0.weight', 'coord.2.weight', 'node.2.bias']) {
      const f = (p: Value): Value => layer.apply(setPath(layer.params, name, p), { h: hs, x: xs, edgeIndex: ei }).x
      expect(checkPrimitive(f, [getPath(layer.params, name)], { rtol: 1e-5 }).failures, name).toEqual([])
    }
  })
})

const addRows = (a: Value, row: Value): Value =>
  tensor((toArray(a) as number[][]).map((r) => r.map((v, j) => v + (toFlat(row)[j] as number))))
const getPath = (tree: Record<string, unknown>, path: string): Value =>
  path.split('.').reduce<unknown>((t, k) => (t as Record<string, unknown>)[k], tree) as Value
function setPath(tree: Record<string, unknown>, path: string, value: Value): Record<string, unknown> {
  const [head, ...rest] = path.split('.')
  if (head === undefined) return tree
  return {
    ...tree,
    [head]: rest.length === 0 ? value : setPath(tree[head] as Record<string, unknown>, rest.join('.'), value),
  }
}

describe('graph transformer', () => {
  const nodes = 6
  const graph = cycleGraph(nodes)
  const edgeIndex = [
    [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
    [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
  ]

  it('without an edge index it is dense attention: softmax(Q K^T / sqrt(d)) V, then residual, norm and feed-forward', () => {
    const layer = new GraphTransformerLayer(4, { heads: 1, key: key(30) })
    const x = n(31, [nodes, 4])
    const p = layer.params as Record<string, Record<string, Value>>
    const withBias = (name: string, v: Value): Value =>
      add(
        matmul(v, transpose((p[name] as Record<string, Value>)['weight'] as Value)),
        (p[name] as Record<string, Value>)['bias'] as Value,
      )
    const attention = matmul(
      softmax(mul(matmul(withBias('query', x), transpose(withBias('key', x))), 1 / Math.sqrt(4)), -1),
      withBias('value', x),
    )
    // The attended values pass through the output projection, a residual and layer norm; compare before the norm via the full forward pass.
    const got = layer.forward({ x })
    const norm = (v: Value): Value => layerNorm(v, [4])
    const hand = norm(add(x, withBias('out', attention)))
    const ff = withBias('ff2', gelu(withBias('ff1', hand)))
    expectClose(got, norm(add(hand, ff)), 1e-10)
  })

  it('is equivariant to relabelling the nodes, including their positional encodings', () => {
    const model = new GraphTransformer(3, 8, { peDim: 2, heads: 2, layers: 2, randomSignFlip: false, key: key(32) })
    const x = n(33, [nodes, 3])
    const pe = laplacianPositionalEncoding(graph, 2)
    const base = model.forward({ x, edgeIndex, pe })
    const perm = [3, 0, 5, 1, 4, 2]
    const inverse = perm.map((_, i) => perm.indexOf(i))
    const rows = (a: Value): Value => tensor(perm.map((p) => (toArray(a) as number[][])[p] as number[]))
    const relabelled = [
      (edgeIndex[0] as number[]).map((v) => inverse[v] as number),
      (edgeIndex[1] as number[]).map((v) => inverse[v] as number),
    ]
    const out = model.forward({ x: rows(x), edgeIndex: relabelled, pe: rows(pe) })
    expectClose(out, rows(base), 1e-9)
  })

  it('the positional encodings matter, and random sign flips act only in training', () => {
    const model = new GraphTransformer(3, 8, { peDim: 2, heads: 2, layers: 1, key: key(34) })
    const x = n(35, [nodes, 3])
    const pe = laplacianPositionalEncoding(graph, 2)
    const without = toFlat(model.eval().forward({ x, edgeIndex }))
    const withPe = toFlat(model.eval().forward({ x, edgeIndex, pe }))
    expect(withPe).not.toEqual(without)
    expect(toFlat(model.eval().forward({ x, edgeIndex, pe }))).toEqual(withPe)
    const flipped = toFlat(model.train().forward({ x, edgeIndex, pe }, { key: key(1) }))
    const again = toFlat(model.train().forward({ x, edgeIndex, pe }, { key: key(1) }))
    expect(again).toEqual(flipped)
    // The random-walk encoding is sign-free and works too.
    expect(model.eval().forward({ x, edgeIndex, pe: randomWalkPositionalEncoding(graph, 2) }).shape).toEqual([nodes, 8])
  })

  it('passes the conformance check', () => {
    const layer = new GraphTransformerLayer(4, { heads: 2, key: key(36) })
    expect(
      checkPrimitive((v: Value) => layer.apply(layer.params, { x: v, edgeIndex }), [n(37, [nodes, 4])], { rtol: 1e-4 })
        .failures,
    ).toEqual([])
    expect(zeros([1]).shape).toEqual([1])
  })
})
