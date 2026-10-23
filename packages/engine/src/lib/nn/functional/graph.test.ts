import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { sub, tensor, toArray, toFlat, type Value, zeros } from '../../tensor/index.ts'
import {
  addSelfLoops,
  aggregate,
  edgeSoftmax,
  gatConv,
  gcnConv,
  ginConv,
  globalPool,
  messagePassing,
  sageConv,
  topKPool,
} from './graph.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: number[][] | number[], digits = 12): void => {
  const x = toFlat(a)
  const y = (b as (number | number[])[]).flat()
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

// A directed graph on 4 nodes (j -> i): 0->1, 1->2, 2->0, 3->0, 0->3, 1->3.
const EI = [
  [0, 1, 2, 3, 0, 1],
  [1, 2, 0, 0, 3, 3],
]
const N = 4
const W = [0.5, 2, 1, 1.5, 0.3, 0.7]
/** Dense A[i][j] = weight of edge j -> i. */
const dense = (weights: readonly number[] = new Array(6).fill(1), loops = 0): number[][] => {
  const a = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i === j ? loops : 0)))
  for (const [e, j] of (EI[0] as number[]).entries()) {
    const i = (EI[1] as number[])[e] as number
    ;(a[i] as number[])[j] = (a[i]?.[j] ?? 0) + (weights[e] ?? 0)
  }
  return a
}
const mm = (a: number[][], b: number[][]): number[][] =>
  a.map((row) => (b[0] ?? []).map((_, k) => row.reduce((s, v, j) => s + v * (b[j]?.[k] ?? 0), 0)))
const tr = (a: number[][]): number[][] => (a[0] ?? []).map((_, j) => a.map((r) => r[j] ?? 0))

describe('message passing', () => {
  it('sum, mean and max aggregation match the dense adjacency', () => {
    const x = n(0, [N, 3])
    const X = toArray(x) as number[][]
    const A = dense(W)
    close(messagePassing(x, EI, { edgeWeight: W }), mm(A, X))
    const deg = dense().map((r) => r.reduce((a, b) => a + b, 0))
    close(
      messagePassing(x, EI, { aggregate: 'mean' }),
      mm(dense(), X).map((r, i) => r.map((v) => v / Math.max(deg[i] ?? 1, 1))),
    )
    const maxed = dense().map((row) =>
      [0, 1, 2].map((k) => {
        const vals = row.flatMap((a, j) => (a > 0 ? [X[j]?.[k] ?? 0] : []))
        return vals.length === 0 ? 0 : vals.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY)
      }),
    )
    close(messagePassing(x, EI, { aggregate: 'max' }), maxed)
    expect(aggregate(zeros([0, 3]), [], 2).shape).toEqual([2, 3])
  })

  it('custom messages see the source and target features', () => {
    const x = tensor([[1], [10], [100], [1000]])
    const y = messagePassing(x, EI, { message: (s, t) => sub(s, t) })
    expect(toFlat(y)).toEqual([100 - 1 + (1000 - 1), 1 - 10, 10 - 100, 1 - 1000 + (10 - 1000)])
  })

  it('edge softmax normalises over incoming edges', () => {
    const s = n(1, [6, 2])
    const w = toArray(edgeSoftmax(s, EI[1] as number[], N)) as number[][]
    for (let i = 0; i < N; i++) {
      const incoming = (EI[1] as number[]).flatMap((t, e) => (t === i ? [e] : []))
      for (const h of [0, 1]) {
        const total = incoming.reduce((a, e) => a + (w[e]?.[h] ?? 0), 0)
        expect(total).toBeCloseTo(incoming.length === 0 ? 0 : 1, 12)
      }
    }
    ok((v) => edgeSoftmax(v, EI[1] as number[], N), [s])
  })

  it('aggregation and message passing are differentiable', () => {
    for (const reduce of ['sum', 'mean', 'max'] as const)
      ok((x) => messagePassing(x, EI, { aggregate: reduce }), [n(2, [N, 2])])
    ok((x, w) => messagePassing(x, EI, { edgeWeight: w }), [n(3, [N, 2]), tensor(W)])
  })
})

describe('graph convolutions', () => {
  it('GCN equals D^-1/2 (A + I) D^-1/2 X W^T + b with weighted edges', () => {
    const x = n(4, [N, 3])
    const w = n(5, [2, 3])
    const b = [0.1, -0.2]
    const A = dense(W, 1)
    const d = A.map((r) => 1 / Math.sqrt(r.reduce((s, v) => s + v, 0)))
    const norm = A.map((r, i) => r.map((v, j) => (d[i] ?? 0) * v * (d[j] ?? 0)))
    const want = mm(mm(norm, toArray(x) as number[][]), tr(toArray(w) as number[][])).map((r) =>
      r.map((v, k) => v + (b[k] ?? 0)),
    )
    close(gcnConv(x, EI, w, { bias: b, edgeWeight: W }), want)
    // Without normalisation: (A + I) X W^T.
    close(
      gcnConv(x, EI, w, { normalize: false }),
      mm(mm(dense(undefined, 1), toArray(x) as number[][]), tr(toArray(w) as number[][])),
    )
    ok((xx, ww) => gcnConv(xx, EI, ww), [x, w])
  })

  it('GAT equals dense masked attention per head', () => {
    const x = n(6, [N, 3])
    const heads = 2
    const c = 2
    const params = {
      weight: n(7, [heads * c, 3]),
      attSrc: n(8, [heads, c]),
      attDst: n(9, [heads, c]),
      bias: n(10, [heads * c]),
    }
    const r = gatConv(x, EI, params, { heads })
    const H = mm(toArray(x) as number[][], tr(toArray(params.weight) as number[][])) // [N, heads c]
    const as = toArray(params.attSrc) as number[][]
    const ad = toArray(params.attDst) as number[][]
    const mask = dense(undefined, 1)
    const want = Array.from({ length: N }, (_, i) =>
      Array.from({ length: heads }, (_, h) => {
        const hv = (k: number): number[] => (H[k] ?? []).slice(h * c, h * c + c)
        const score = (j: number): number => {
          const s =
            hv(j).reduce((a, v, q) => a + v * (as[h]?.[q] ?? 0), 0) +
            hv(i).reduce((a, v, q) => a + v * (ad[h]?.[q] ?? 0), 0)
          return s > 0 ? s : 0.2 * s
        }
        const js = Array.from({ length: N }, (_, j) => j).filter((j) => (mask[i]?.[j] ?? 0) > 0)
        const e = js.map((j) => Math.exp(score(j)))
        const z = e.reduce((a, v) => a + v, 0)
        return Array.from({ length: c }, (_, q) =>
          js.reduce((acc, j, t) => acc + ((e[t] ?? 0) / z) * (hv(j)[q] ?? 0), 0),
        )
      }).flat(),
    ).map((row) => row.map((v, k) => v + (toFlat(params.bias)[k] ?? 0)))
    close(r.output, want)
    expect(r.attention.shape).toEqual([10, heads])
    expect(gatConv(x, EI, { ...params, bias: n(11, [c]) }, { heads, concat: false }).output.shape).toEqual([N, c])
    ok(
      (xx, ww, a1, a2) => gatConv(xx, EI, { weight: ww, attSrc: a1, attDst: a2 }, { heads }).output,
      [x, params.weight, params.attSrc, params.attDst],
    )
  })

  it('GraphSAGE and GIN match their dense forms', () => {
    const x = n(12, [N, 3])
    const X = toArray(x) as number[][]
    const wl = n(13, [2, 3])
    const wr = n(14, [2, 3])
    const bl = [0.5, -0.5]
    const deg = dense().map((r) => r.reduce((a, b) => a + b, 0))
    const meanAgg = mm(dense(), X).map((r, i) => r.map((v) => v / Math.max(deg[i] ?? 1, 1)))
    const want = mm(meanAgg, tr(toArray(wl) as number[][])).map((r, i) =>
      r.map((v, k) => v + (bl[k] ?? 0) + (mm(X, tr(toArray(wr) as number[][]))[i]?.[k] ?? 0)),
    )
    close(sageConv(x, EI, { weightNeighbour: wl, biasNeighbour: bl, weightRoot: wr }), want)
    const normed = toArray(
      sageConv(x, EI, { weightNeighbour: wl }, { aggregate: 'sum', normalize: true }),
    ) as number[][]
    for (const row of normed) expect(Math.hypot(...row)).toBeCloseTo(1, 10)
    close(
      ginConv(x, EI, (h) => h, 0.25),
      mm(dense(undefined, 1.25), X),
    )
    ok((xx, a, b) => sageConv(xx, EI, { weightNeighbour: a, weightRoot: b }, { aggregate: 'max' }), [x, wl, wr])
    ok((xx, eps) => ginConv(xx, EI, (h) => h, eps), [x, tensor(0.3)])
  })

  it('global pooling by batch and self loops', () => {
    const x = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ])
    close(globalPool(x, [0, 0, 1, 1], 'mean'), [
      [2, 3],
      [6, 7],
    ])
    close(globalPool(x, [0, 1, 1, 1], 'max'), [
      [1, 2],
      [7, 8],
    ])
    close(globalPool(x), [[4, 5]])
    expect(globalPool(x, [0, 0, 0, 0], 'sum', 2).shape).toEqual([2, 2])
    const looped = addSelfLoops(
      [
        [0, 1],
        [1, 1],
      ],
      2,
      [2, 3],
      5,
    )
    expect(toArray(looped.edgeIndex)).toEqual([
      [0, 1, 0],
      [1, 1, 0],
    ])
    expect(toFlat(looped.edgeWeight)).toEqual([2, 3, 5])
    ok((v) => globalPool(v, [0, 1, 1, 0], 'max'), [n(15, [4, 2])])
  })
})

describe('top-k pooling', () => {
  it('keeps the best-scored nodes of each graph, renumbers their edges and gates by the score', () => {
    const x = tensor([
      [1, 0],
      [3, 1],
      [2, 0],
      [0, 1],
      [5, 0],
    ])
    const edges = [
      [0, 1, 2, 3, 1],
      [1, 2, 3, 4, 3],
    ]
    const r = topKPool(x, edges, [1, 0], { ratio: 0.5, batch: [0, 0, 0, 1, 1] })
    expect(r.perm).toEqual([1, 2, 4])
    expect(r.batch).toEqual([0, 0, 1])
    expect(toArray(r.edgeIndex)).toEqual([[0], [1]])
    expect(toFlat(r.x)[0]).toBeCloseTo(3 * Math.tanh(3), 12)
    ok((v, w) => topKPool(v, edges, w, { batch: [0, 0, 0, 1, 1] }).x, [x, tensor([0.6, -0.8])])
  })
})
