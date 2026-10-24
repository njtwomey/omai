import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import type { Tensor } from '../tensor/index.ts'
import { bipartiteMatching, type Graph, maxFlow, maxFlowMinCost, minCostFlow, minCut } from './index.ts'

/** Net outflow of each node under a flow on the edges. */
function netOut(g: Graph, flow: ArrayLike<number>): number[] {
  const out = new Array<number>(g.n).fill(0)
  g.edges.forEach((e, k) => {
    out[e.source] = (out[e.source] ?? 0) + (flow[k] ?? 0)
    out[e.target] = (out[e.target] ?? 0) - (flow[k] ?? 0)
  })
  return out
}

describe('flows against networkx and scipy.sparse.csgraph', () => {
  for (const c of loadFixture('graph/flows').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      const want = Array.from((decode(c.output) as Tensor).data)
      if (c.fn === 'maxFlow') {
        const s = p['source'] as number
        const t = p['sink'] as number
        const [value, cutValue, scipyValue] = want
        for (const method of ['dinic', 'edmonds-karp'] as const) {
          const f = maxFlow(g, s, t, { method })
          expect(f.value).toBeCloseTo(value ?? 0, 10)
          expect(f.value).toBeCloseTo(scipyValue ?? 0, 10)
          // Capacity and conservation.
          for (const [k, e] of g.edges.entries()) {
            expect(Math.abs(f.flow[k] ?? 0)).toBeLessThanOrEqual(e.weight + 1e-12)
            if (g.directed) expect(f.flow[k] ?? 0).toBeGreaterThanOrEqual(-1e-12)
          }
          netOut(g, f.flow).forEach((x, v) => {
            if (v !== s && v !== t) expect(Math.abs(x)).toBeLessThan(1e-9)
          })
        }
        const cut = minCut(g, s, t)
        expect(cut.value).toBeCloseTo(cutValue ?? 0, 10)
        let crossing = 0
        for (const [u, v] of cut.cutEdges) {
          const e = g.edges.find(
            (x) => (x.source === u && x.target === v) || (!g.directed && x.source === v && x.target === u),
          )
          crossing += e?.weight ?? 0
        }
        expect(crossing).toBeCloseTo(cutValue ?? 0, 10)
      } else if (c.fn === 'minCostFlow') {
        const [cost, mfmcCost, flowValue] = want
        const costs = p['cost'] as number[]
        const demand = p['demand'] as number[]
        const r = minCostFlow(g, demand, { cost: costs })
        expect(r.cost).toBeCloseTo(cost ?? 0, 9)
        for (const [v, x] of netOut(g, r.flow).entries()) expect(x).toBeCloseTo(-(demand[v] ?? 0), 9)
        g.edges.forEach((e, k) => {
          expect(r.flow[k] ?? 0).toBeGreaterThanOrEqual(-1e-12)
          expect(r.flow[k] ?? 0).toBeLessThanOrEqual(e.weight + 1e-12)
        })
        const m = maxFlowMinCost(g, 0, g.n - 1, { cost: costs })
        expect(m.value).toBeCloseTo(flowValue ?? 0, 10)
        expect(m.cost).toBeCloseTo(mfmcCost ?? 0, 9)
      } else {
        const [size, scipySize] = want
        const m = bipartiteMatching(g, { left: p['left'] as number[] })
        expect(m.size).toBe(size)
        expect(m.size).toBe(scipySize)
        const used = new Set<number>()
        for (const [a, b] of m.pairs) {
          expect(used.has(a) || used.has(b)).toBe(false)
          used.add(a)
          used.add(b)
          expect(g.edges.some((e) => (e.source === a && e.target === b) || (e.source === b && e.target === a))).toBe(
            true,
          )
        }
        expect(bipartiteMatching(g).size).toBe(size)
      }
    })
  }
})
