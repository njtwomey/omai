import { describe, expect, it } from 'vitest'
import { eigvalsh, solve } from '../numerics/linalg/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  commuteTime,
  completeGraph,
  cycleGraph,
  digraph,
  edgeEffectiveResistances,
  effectiveResistance,
  effectiveResistanceBetween,
  graph,
  karateClubGraph,
  kirchhoffIndex,
  laplacian,
  pathGraph,
  petersenGraph,
  spanningTreeCount,
  starGraph,
} from './index.ts'

const weightedKarate = (): ReturnType<typeof graph> => {
  const k = karateClubGraph()
  return graph(
    k.edges.map((e, i) => [e.source, e.target, 1 + (i % 4)] as [number, number, number]),
    { nodes: k.n },
  )
}

describe('effective resistance: closed forms', () => {
  it('is the hop distance on a tree', () => {
    const path = toFlat(effectiveResistance(pathGraph(4)))
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) expect(path[a * 4 + b]).toBeCloseTo(Math.abs(a - b), 12)
    const star = toFlat(effectiveResistance(starGraph(4)))
    expect(star[1 * 5 + 2]).toBeCloseTo(2, 12)
    expect(star[0 * 5 + 3]).toBeCloseTo(1, 12)
  })

  it('on a cycle R_ij = d (n - d) / n and on a complete graph 2 / n', () => {
    const n = 9
    const c = toFlat(effectiveResistance(cycleGraph(n)))
    for (let d = 1; d < n; d++) expect(c[d]).toBeCloseTo((d * (n - d)) / n, 12)
    const k = toFlat(effectiveResistance(completeGraph(6)))
    expect(k[1]).toBeCloseTo(2 / 6, 12)
  })

  it('combines in series and in parallel, and a conductance is the reciprocal of a resistance', () => {
    expect(effectiveResistanceBetween(graph([[0, 1, 2]]), 0, 1)).toBeCloseTo(0.5, 14)
    expect(
      effectiveResistanceBetween(
        graph([
          [0, 1, 2],
          [1, 2, 4],
        ]),
        0,
        2,
      ),
    ).toBeCloseTo(0.75, 14)
    // A direct edge of conductance 1 in parallel with a two-edge path of resistance 2: 1 / (1 + 1/2).
    const tri = graph([
      [0, 1],
      [0, 2],
      [2, 1],
    ])
    expect(effectiveResistanceBetween(tri, 0, 1)).toBeCloseTo(2 / 3, 14)
    expect(toFlat(effectiveResistance(tri, { weighted: false }))[1]).toBeCloseTo(2 / 3, 14)
  })

  it('unweighted ignores the weights', () => {
    const w = graph([
      [0, 1, 5],
      [1, 2, 5],
    ])
    expect(effectiveResistanceBetween(w, 0, 2, { weighted: false })).toBeCloseTo(2, 14)
    expect(effectiveResistanceBetween(w, 0, 2)).toBeCloseTo(0.4, 14)
  })
})

describe('effective resistance: two computations and the theorems', () => {
  const g = weightedKarate()

  it('the pseudoinverse and the grounded solve agree on every pair', () => {
    const r = toFlat(effectiveResistance(g))
    for (const [u, v] of [
      [0, 33],
      [4, 10],
      [16, 5],
      [2, 31],
      [11, 12],
    ] as const) {
      expect(effectiveResistanceBetween(g, u, v)).toBeCloseTo(r[u * g.n + v] ?? Number.NaN, 10)
    }
    expect(r[7 * g.n + 7]).toBe(0)
  })

  it("Foster's theorem: the conductance-weighted edge resistances sum to n - 1", () => {
    const r = edgeEffectiveResistances(g)
    const total = g.edges.reduce((a, e, i) => a + e.weight * (r[i] ?? 0), 0)
    expect(total).toBeCloseTo(g.n - 1, 9)
    const plain = edgeEffectiveResistances(karateClubGraph())
    expect(plain.reduce((a, b) => a + b, 0)).toBeCloseTo(33, 9)
  })

  it('an unweighted edge resistance is the chance the edge is in a uniform spanning tree', () => {
    const p = petersenGraph()
    const total = spanningTreeCount(p)
    expect(Math.round(total)).toBe(2000)
    const r = edgeEffectiveResistances(p)
    p.edges.forEach((_, i) => {
      const rest = graph(
        p.edges.filter((_, j) => j !== i).map((f) => [f.source, f.target] as [number, number]),
        { nodes: p.n },
      )
      expect(r[i]).toBeCloseTo(1 - spanningTreeCount(rest) / total, 8)
    })
    // Every edge of Petersen is equivalent: n - 1 over m.
    expect(r[0]).toBeCloseTo(9 / 15, 10)
  })

  it('counts the spanning trees of the complete graph (Cayley) and a weighted cycle', () => {
    expect(Math.round(spanningTreeCount(completeGraph(6)))).toBe(1296)
    expect(
      spanningTreeCount(
        graph([
          [0, 1, 2],
          [1, 2, 3],
          [2, 0, 5],
        ]),
        { weighted: true },
      ),
    ).toBeCloseTo(2 * 3 + 3 * 5 + 2 * 5, 10)
    expect(spanningTreeCount(graph([[0, 1]], { nodes: 3 }))).toBe(0)
  })

  it('the Kirchhoff index is n times the sum of the reciprocal nonzero Laplacian eigenvalues', () => {
    expect(kirchhoffIndex(pathGraph(4))).toBeCloseTo(10, 12)
    const lam = toFlat(eigvalsh(laplacian(g)))
    const want =
      g.n *
      Array.from(lam)
        .filter((x) => x > 1e-9)
        .reduce((a, x) => a + 1 / x, 0)
    expect(kirchhoffIndex(g)).toBeCloseTo(want, 8)
  })

  it('is infinite across components', () => {
    const two = graph([
      [0, 1],
      [2, 3],
    ])
    expect(effectiveResistanceBetween(two, 0, 3)).toBe(Number.POSITIVE_INFINITY)
    expect(toFlat(effectiveResistance(two))[3]).toBe(Number.POSITIVE_INFINITY)
    expect(toFlat(effectiveResistance(two))[1]).toBeCloseTo(1, 14)
    expect(kirchhoffIndex(two)).toBe(Number.POSITIVE_INFINITY)
  })

  it('rejects directed graphs and non-positive conductances', () => {
    expect(() => effectiveResistance(digraph([[0, 1]]))).toThrow(/undirected/)
    expect(() => effectiveResistance(graph([[0, 1, 0]]))).toThrow(/positive/)
  })
})

describe('commute time', () => {
  /** The expected steps for the random walk (probabilities proportional to weight) to first reach `target`. */
  const hitting = (g: ReturnType<typeof graph>, target: number): number[] => {
    const others = Array.from({ length: g.n }, (_, i) => i).filter((i) => i !== target)
    const a = others.map((i) => others.map((j) => (i === j ? 1 : 0)))
    const strength = new Array<number>(g.n).fill(0)
    for (const e of g.edges) {
      strength[e.source] = (strength[e.source] ?? 0) + e.weight
      strength[e.target] = (strength[e.target] ?? 0) + e.weight
    }
    for (const e of g.edges) {
      for (const [x, y] of [
        [e.source, e.target],
        [e.target, e.source],
      ] as const) {
        const r = others.indexOf(x)
        const c = others.indexOf(y)
        if (r >= 0 && c >= 0) (a[r] as number[])[c] = ((a[r] as number[])[c] ?? 0) - e.weight / (strength[x] ?? 1)
      }
    }
    const h = toFlat(
      solve(
        a,
        others.map(() => 1),
      ),
    )
    const out = new Array<number>(g.n).fill(0)
    others.forEach((i, k) => {
      out[i] = h[k] ?? 0
    })
    return out
  }

  it('equals the sum of the two hitting times, computed by solving for them', () => {
    const g = graph([
      [0, 1, 1],
      [1, 2, 2],
      [2, 3, 1],
      [3, 0, 3],
      [1, 3, 1],
      [3, 4, 2],
    ])
    for (const [u, v] of [
      [0, 2],
      [4, 1],
      [3, 4],
    ] as const) {
      const there = hitting(g, v)[u] ?? 0
      const back = hitting(g, u)[v] ?? 0
      expect(commuteTime(g, u, v)).toBeCloseTo(there + back, 9)
    }
  })

  it('has the closed forms of a path and a complete graph', () => {
    expect(commuteTime(pathGraph(3), 0, 2)).toBe(8)
    expect(commuteTime(completeGraph(5), 0, 1)).toBeCloseTo(2 * 10 * (2 / 5), 12)
    expect(commuteTime(pathGraph(2), 1, 1)).toBe(0)
  })
})
