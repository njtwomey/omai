import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../random/index.ts'
import {
  completeGraph,
  digraph,
  graph,
  karateClubGraph,
  kHopNeighbourhood,
  type LayerSample,
  layerwiseImportanceSample,
  neighbourSample,
  neighbourSampleVariance,
  node2vecTransition,
  node2vecWalks,
  pathGraph,
  randomWalkSubgraph,
  randomWalks,
  sampledAggregate,
  starGraph,
} from './index.ts'

const karate = karateClubGraph()
const mean = (x: readonly number[]): number => x.reduce((a, b) => a + b, 0) / x.length
const variance = (x: readonly number[]): number => mean(x.map((v) => (v - mean(x)) ** 2))

describe('neighbour sampling', () => {
  it('with unbounded fan-outs the receptive field is the k-hop neighbourhood', () => {
    for (const hops of [1, 2, 3]) {
      const s = neighbourSample(karate, [5], Array<number>(hops).fill(-1), key(0))
      expect([...s.nodes].sort((a, b) => a - b)).toEqual(kHopNeighbourhood(karate, 5, hops))
      expect(s.hops[0]?.weight.every((w) => w === 1)).toBe(true)
      expect(s.layers).toHaveLength(hops + 1)
    }
  })

  it('a fan-out caps the neighbours drawn and grows the field no faster than the product of the fan-outs', () => {
    const s = neighbourSample(karate, [0, 33], [5, 3], key(1))
    const bySrc = (h: number, v: number): number => (s.hops[h]?.dst ?? []).filter((d) => d === v).length
    expect(bySrc(0, 0)).toBe(5)
    expect(bySrc(0, 33)).toBe(5)
    expect(s.hops[0]?.weight[0]).toBeCloseTo(16 / 5, 12)
    for (const v of s.layers[1] ?? []) expect(bySrc(1, v)).toBeLessThanOrEqual(3)
    expect((s.layers[1] ?? []).length).toBeLessThanOrEqual(2 + 2 * 5)
    expect((s.layers[2] ?? []).length).toBeLessThanOrEqual((s.layers[1] ?? []).length * 4)
    // Layers nest, and every sampled edge is a real edge.
    expect(s.layers[0]).toEqual([0, 33])
    for (let h = 1; h < s.layers.length; h++) for (const v of s.layers[h - 1] ?? []) expect(s.layers[h]).toContain(v)
    for (const hop of s.hops) {
      for (let i = 0; i < hop.src.length; i++) {
        const row = karate.out.indices.subarray(
          karate.out.indptr[hop.dst[i] ?? 0],
          karate.out.indptr[(hop.dst[i] ?? 0) + 1],
        )
        expect(row).toContain(hop.src[i])
      }
    }
    expect(s.subgraph.n).toBe(s.nodes.length)
    expect(s.subgraph.labels).toEqual(s.nodes.map(String))
  })

  it('is unbiased, and its variance is the finite-population formula', () => {
    const x = Array.from({ length: 34 }, (_, v) => ((v * 37) % 11) - 4)
    const nb = Array.from(karate.out.indices.subarray(karate.out.indptr[0], karate.out.indptr[1]))
    const total = nb.reduce((a, u) => a + (x[u] ?? 0), 0)
    for (const replace of [false, true]) {
      const est: number[] = []
      for (let i = 0; i < 8000; i++) {
        const s = neighbourSample(karate, [0], [4], foldIn(key(5), i), { replace })
        est.push(s.hops[0]?.src.reduce((a, u, j) => a + (s.hops[0]?.weight[j] ?? 0) * (x[u] ?? 0), 0) ?? 0)
      }
      const want = neighbourSampleVariance(
        nb.map((u) => x[u] ?? 0),
        4,
        replace,
      )
      expect(Math.abs(mean(est) - total)).toBeLessThan(4 * Math.sqrt(want / 8000))
      expect(Math.abs(variance(est) / want - 1)).toBeLessThan(0.08)
    }
    expect(neighbourSampleVariance([1, 2, 3, 4], 4)).toBe(0)
  })

  it('draws for a node depend on the key, the hop and the node only', () => {
    const a = neighbourSample(karate, [0], [4], key(2))
    const b = neighbourSample(karate, [0, 33], [4], key(2))
    const from0 = (s: ReturnType<typeof neighbourSample>): number[] =>
      (s.hops[0]?.src ?? []).filter((_, i) => s.hops[0]?.dst[i] === 0)
    expect(from0(b)).toEqual(from0(a))
    expect(neighbourSample(karate, [0], [4], key(3)).hops[0]?.src).not.toEqual(a.hops[0]?.src)
  })

  it('follows the direction of the edges', () => {
    const g = digraph([
      [1, 0],
      [2, 0],
      [0, 3],
    ])
    expect([...neighbourSample(g, [0], [-1], key(0)).nodes].sort()).toEqual([0, 1, 2])
    expect([...neighbourSample(g, [0], [-1], key(0), { direction: 'out' }).nodes].sort()).toEqual([0, 3])
    expect([...neighbourSample(g, [0], [-1], key(0), { direction: 'both' }).nodes].sort()).toEqual([0, 1, 2, 3])
  })

  it('rejects a fractional fan-out and a variance fan-out beyond the degree', () => {
    expect(() => neighbourSample(karate, [0], [2.5], key(0))).toThrow(/integers/)
    expect(() => neighbourSampleVariance([1, 2], 3)).toThrow(/fan-out/)
  })
})

describe('random walks', () => {
  it('step along edges and end at a dead end', () => {
    expect(randomWalks(pathGraph(2), [0], 3, key(0))[0]).toEqual([0, 1, 0, 1])
    const dag = digraph([[0, 1]])
    expect(randomWalks(dag, [0], 5, key(0))[0]).toEqual([0, 1])
    expect(randomWalks(karate, [0, 1], 4, key(1), { walksPerNode: 3 })).toHaveLength(6)
    expect(() => randomWalks(karate, [0], -1, key(0))).toThrow(/length/)
  })

  it('a long walk spends time in proportion to the degree', () => {
    const walk = randomWalks(karate, [0], 60000, key(2))[0] ?? []
    const visits = new Array<number>(34).fill(0)
    for (const v of walk) visits[v] = (visits[v] ?? 0) + 1
    for (const v of [0, 33, 11]) {
      const d = (karate.out.indptr[v + 1] ?? 0) - (karate.out.indptr[v] ?? 0)
      expect(Math.abs((visits[v] ?? 0) / walk.length - d / 156)).toBeLessThan(0.012)
    }
  })

  it('weights bias the steps', () => {
    const g = graph([
      [0, 1, 9],
      [0, 2, 1],
    ])
    const next = randomWalks(g, [0], 1, key(3), { weighted: true, walksPerNode: 2000 }).map((w) => w[1])
    expect(Math.abs(next.filter((v) => v === 1).length / 2000 - 0.9)).toBeLessThan(0.025)
  })
})

describe('node2vec', () => {
  const g = graph([
    [0, 1],
    [1, 2],
    [1, 3],
    [2, 3],
    [1, 4],
  ])

  it('has the exact second-order transition probabilities', () => {
    // Came from 2 to 1: back to 2 (1 / p = 0.5), to 3 which is adjacent to 2 (1), to 0 and 4 further out (1 / q = 4).
    const t = node2vecTransition(g, 2, 1, { p: 2, q: 0.25 })
    expect(t.nodes).toEqual([0, 2, 3, 4])
    const total = 4 + 0.5 + 1 + 4
    expect(t.probabilities.map((x) => x * total)).toEqual([4, 0.5, 1, 4])
    expect(node2vecTransition(g, 2, 1).probabilities).toEqual([0.25, 0.25, 0.25, 0.25])
    expect(() => node2vecTransition(g, 2, 1, { p: 0 })).toThrow(/positive/)
  })

  it('the walks follow them', () => {
    // From node 0 the first step is forced to 1; the second then has the transition from 0 to 1.
    const p = 2
    const q = 0.5
    const t = node2vecTransition(g, 0, 1, { p, q })
    const walks = node2vecWalks(g, [0], 2, key(4), { p, q, walksPerNode: 8000 })
    expect(walks.every((w) => w[1] === 1)).toBe(true)
    t.nodes.forEach((x, i) => {
      const freq = walks.filter((w) => w[2] === x).length / walks.length
      expect(Math.abs(freq - (t.probabilities[i] ?? 0))).toBeLessThan(0.02)
    })
  })

  it('a tiny p returns to the previous node, and a tiny q runs away from it', () => {
    expect(node2vecWalks(pathGraph(3), [1], 4, key(0), { p: 1e-9 })[0]?.slice(2)).toEqual([1, 0, 1])
    const star = graph([
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ])
    const far = node2vecWalks(star, [2], 3, key(1), { q: 1e-9, walksPerNode: 20 })
    for (const w of far) expect(Math.abs((w[3] ?? 0) - (w[2] ?? 0))).toBe(1)
  })
})

describe('random-walk subgraphs', () => {
  it('keep the roots and the induced edges', () => {
    const s = randomWalkSubgraph(karate, [0, 33], 3, key(0))
    expect(s.nodes).toContain(0)
    expect(s.nodes).toContain(33)
    expect(s.subgraph.n).toBe(s.nodes.length)
    expect(s.subgraph.labels).toEqual(s.nodes.map(String))
    expect(randomWalkSubgraph(completeGraph(6), [2], 0, key(0)).nodes).toEqual([2])
    expect(randomWalkSubgraph(karate, 5, 2, key(1)).nodes.length).toBeGreaterThanOrEqual(5)
    expect(() => randomWalkSubgraph(karate, 40, 2, key(0))).toThrow(/roots/)
  })
})

describe('layer-wise importance sampling', () => {
  const x = Array.from({ length: 34 }, (_, v) => ((v * 13) % 7) + 1)
  const exact = (targets: readonly number[]): number[] =>
    targets.map((v) => {
      const nb = Array.from(karate.out.indices.subarray(karate.out.indptr[v], karate.out.indptr[v + 1]))
      return nb.reduce((a, u) => a + (x[u] ?? 0), 0) / nb.length
    })

  it('draws a layer of the stated width whose weights make the draw probabilities add up to one', () => {
    const layers = layerwiseImportanceSample(karate, [0, 33], [8, 6], key(0))
    expect(layers).toHaveLength(2)
    for (const layer of layers) {
      expect(layer.nodes.length).toBeLessThanOrEqual(8)
      const mass = layer.weights.reduce((a, w, i) => a + w * (layer.probabilities[i] ?? 0), 0)
      expect(mass).toBeCloseTo(1, 12)
    }
    expect(layers[0]?.nodes.every((u) => u >= 0 && u < 34)).toBe(true)
  })

  it('gives unbiased aggregates whose error variance is the multinomial formula', () => {
    const targets = [0, 33, 2, 5]
    const want = exact(targets)
    const draws = 10
    for (const mode of ['importance', 'uniform'] as const) {
      // The exact variance of the estimate for one target: (sum_u (P_vu x_u)^2 / p_u - (P x)_v^2) / draws.
      const score = new Map<number, number>()
      for (const v of targets) {
        const nb = Array.from(karate.out.indices.subarray(karate.out.indptr[v], karate.out.indptr[v + 1]))
        for (const u of nb) score.set(u, (score.get(u) ?? 0) + (mode === 'importance' ? 1 / nb.length ** 2 : 1))
      }
      const norm = [...score.values()].reduce((a, b) => a + b, 0)
      const variances = targets.map((v, j) => {
        const nb = Array.from(karate.out.indices.subarray(karate.out.indptr[v], karate.out.indptr[v + 1]))
        const second = nb.reduce((a, u) => a + (x[u] ?? 0) ** 2 / nb.length ** 2 / ((score.get(u) ?? 1) / norm), 0)
        return (second - (want[j] ?? 0) ** 2) / draws
      })
      const trials = 4000
      const sum = new Array<number>(targets.length).fill(0)
      const sq = new Array<number>(targets.length).fill(0)
      for (let i = 0; i < trials; i++) {
        const layer = layerwiseImportanceSample(karate, targets, [draws], foldIn(key(7), i), { mode })[0] as LayerSample
        sampledAggregate(karate, targets, layer, x).forEach((e, j) => {
          sum[j] = (sum[j] ?? 0) + e / trials
          sq[j] = (sq[j] ?? 0) + (e - (want[j] ?? 0)) ** 2 / trials
        })
      }
      targets.forEach((_, j) => {
        expect(Math.abs((sum[j] ?? 0) - (want[j] ?? 0))).toBeLessThan(4 * Math.sqrt((variances[j] ?? 0) / trials))
        expect(Math.abs((sq[j] ?? 0) / (variances[j] ?? 1) - 1)).toBeLessThan(0.12)
      })
    }
  })

  it('rejects empty layers and bad sizes', () => {
    expect(() => layerwiseImportanceSample(karate, [0], [0], key(0))).toThrow(/positive/)
    expect(() => layerwiseImportanceSample(graph([], { nodes: 2 }), [0], [3], key(0))).toThrow(/no neighbours/)
    expect(
      sampledAggregate(starGraph(4), [0], { nodes: [1], probabilities: [0.25], weights: [4] }, [0, 1, 2, 3, 4]),
    ).toEqual([1])
  })
})
