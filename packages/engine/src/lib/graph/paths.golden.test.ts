import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { aStar, bellmanFord, dijkstra, floydWarshall, johnson, kShortestPaths } from './index.ts'

function rows(g: ReturnType<typeof graphFromCase>, f: (s: number) => Float64Array): Tensor {
  const out = new Float64Array(g.n * g.n)
  for (let s = 0; s < g.n; s++) out.set(f(s), s * g.n)
  return fromStorage(out, [g.n, g.n], 'float64')
}

describe('shortest paths against scipy.sparse.csgraph and networkx', () => {
  for (const c of loadFixture('graph/paths').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      switch (c.fn) {
        case 'dijkstra':
          expectMatches(
            rows(g, (s) => dijkstra(g, s).distance),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          break
        case 'unweighted':
          expectMatches(
            rows(g, (s) => dijkstra(g, s, { weighted: false }).distance),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          break
        case 'floydWarshall':
          expectMatches(floydWarshall(g).distance, c.output, c.rtol, c.atol, c.name)
          break
        case 'johnson':
          expectMatches(johnson(g).distance, c.output, c.rtol, c.atol, c.name)
          break
        case 'bellmanFord':
          expectMatches(
            rows(g, (s) => bellmanFord(g, s).distance),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          break
        case 'negativeCycle':
          expect(() => bellmanFord(g, 0)).toThrow()
          expect(() => johnson(g)).toThrow()
          expect(() => floydWarshall(g)).toThrow()
          expect(p['negative']).toBe(true)
          break
        case 'pairs': {
          const want = Array.from((decode(c.output) as Tensor).data)
          const pairs = p['pairs'] as [number, number][]
          const ksp = p['kShortest'] as number[][]
          pairs.forEach(([s, t], i) => {
            expect(aStar(g, s, t).distance).toBe(want[i])
            expect(kShortestPaths(g, s, t, 4).map((x) => x.cost)).toEqual(ksp[i])
          })
          break
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
