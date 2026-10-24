import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { maxOf } from '../base/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { girvanNewman, labelPropagation, leiden, louvain, modularity, spectralBisection } from './index.ts'

const lexical = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return (a[i] ?? 0) - (b[i] ?? 0)
  return a.length - b.length
}
const sortedSets = (sets: readonly (readonly number[])[]): number[][] =>
  sets.map((s) => [...s].sort((a, b) => a - b)).sort(lexical)

describe('communities against networkx', () => {
  for (const c of loadFixture('graph/community').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      switch (c.fn) {
        case 'modularity': {
          const labels = p['labels'] as number[]
          const q = [1, 0.5, 2].map((resolution) => modularity(g, labels, { resolution }))
          expectMatches(fromStorage(Float64Array.from(q), [3], 'float64'), c.output, c.rtol, c.atol, c.name)
          break
        }
        case 'girvanNewman': {
          const levels = p['levels'] as number[][][]
          const got = girvanNewman(g, { levels: levels.length }).map((x) => sortedSets(x.components))
          expect(got).toEqual(levels)
          break
        }
        case 'clear':
          expect(sortedSets(labelPropagation(g, { seed: 1 }).components)).toEqual(p['labelPropagation'])
          expect(sortedSets(louvain(g, { seed: 1 }).components)).toEqual(p['louvain'])
          expect(sortedSets(leiden(g, { seed: 1 }).components)).toEqual(p['leiden'])
          if (p['bisection'] !== null) expect(sortedSets(spectralBisection(g))).toEqual(p['bisection'])
          break
        case 'multilevel': {
          // Over the same number of seeds, the best modularity reached is at least the best NetworkX reaches.
          const [louvainMin, louvainMax, leidenMin, leidenMax] = Array.from((decode(c.output) as Tensor).data)
          const best = (f: (seed: number) => number): number => maxOf(Array.from({ length: 10 }, (_, s) => f(s)))
          const lv = best((seed) => modularity(g, louvain(g, { seed }).labels))
          const ld = best((seed) => modularity(g, leiden(g, { seed }).labels))
          expect(lv).toBeGreaterThanOrEqual((louvainMin ?? 0) - 1e-12)
          expect(lv).toBeGreaterThanOrEqual((louvainMax ?? 0) - 1e-12)
          expect(ld).toBeGreaterThanOrEqual((leidenMin ?? 0) - 1e-12)
          expect(ld).toBeGreaterThanOrEqual((leidenMax ?? 0) - 1e-12)
          break
        }
        case 'bisection':
          expect(sortedSets(spectralBisection(g))).toEqual(p['bisection'])
          expect(sortedSets(spectralBisection(g, { normalised: true }))).toEqual(p['normalised'])
          break
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
