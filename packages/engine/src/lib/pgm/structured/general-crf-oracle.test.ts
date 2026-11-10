import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { generalCrf, generalCrfObjective } from './crf.ts'
import { generalCrfOracle, graphAsGeneralCrf } from './general-crf-oracle.ts'
import { graphCrf } from './graph-crf.ts'
import { graphOracle } from './graph-oracle.ts'
import { gridCrf, gridInput, higherOrderChainCrf, higherOrderChainInput } from './graph-structures.ts'

const vector = (seed: number, n: number, scale = 0.7): Float64Array =>
  Float64Array.from(toFlat(normal(key(seed), [n])), (v) => v * scale)

const items = (seed: number, n: number): Record<string, number>[] => {
  const v = vector(seed, n, 1)
  return Array.from({ length: n }, (_, i) => ({ f: v[i] ?? 0 }))
}

const cases = [
  {
    name: 'a directional grid',
    model: gridCrf(['a', 'b', 'c'], ['f'], { directional: true }),
    x: gridInput(items(1, 6), 2, 3, { directional: true }),
  },
  {
    name: 'a second-order chain',
    model: higherOrderChainCrf(['a', 'b'], ['f'], 2),
    x: higherOrderChainInput(items(2, 5), 2),
  },
]

describe('GeneralCrf as a graph CRF', () => {
  for (const c of cases) {
    it(`${c.name}: the general form of an input has the same features, objective and gradient as the graph oracle`, () => {
      const w = vector(3, c.model.layout.size)
      const oracle = graphOracle(c.model)
      const g = graphAsGeneralCrf(c.model, c.x)
      const y = c.x.nodes.map((_, i) => c.model.labels[(i * 5 + 1) % c.model.labels.length] ?? '')
      const general = generalCrfOracle(generalCrf(g.crf.variables, g.crf.features))
      expect(Array.from(general.features(g.instance, g.labelling(y)))).toEqual(Array.from(oracle.features(c.x, y)))
      const e = oracle.expectations(w, c.x)
      const ge = general.expectations(w, g.instance)
      expect(ge.logPartition).toBeCloseTo(e.logPartition, 9)
      for (let i = 0; i < w.length; i++) expect(ge.counts[i]).toBeCloseTo(e.counts[i] ?? 0, 9)
      // the older training objective on the same instance
      const old = generalCrfObjective({ ...g.crf, weights: w }, [{ instance: g.instance, y: g.labelling(y) }])
      const phi = oracle.features(c.x, y)
      expect(old.value).toBeCloseTo(e.logPartition - phi.reduce((s, v, i) => s + v * (w[i] ?? 0), 0), 9)
      for (let i = 0; i < w.length; i++) expect(old.gradient[i]).toBeCloseTo((e.counts[i] ?? 0) - (phi[i] ?? 0), 9)
    })

    it(`${c.name}: decoding, with and without cost augmentation, and the loss agree with the graph oracle`, () => {
      const w = vector(4, c.model.layout.size, 1.2)
      const oracle = graphOracle(c.model)
      const g = graphAsGeneralCrf(c.model, c.x)
      const general = generalCrfOracle(generalCrf(g.crf.variables, g.crf.features))
      const gold = c.x.nodes.map((_, i) => c.model.labels[i % c.model.labels.length] ?? '')
      expect(general.decode(w, g.instance).score).toBeCloseTo(oracle.decode(w, c.x).score, 9)
      const against = general.decode(w, g.instance, { against: g.labelling(gold) })
      expect(against.score).toBeCloseTo(oracle.decode(w, c.x, { against: gold }).score, 9)
      expect(general.loss(g.labelling(gold), g.labelling(gold))).toBe(0)
      expect(general.loss(g.labelling(gold), against.y)).toBe(
        oracle.loss(
          gold,
          c.x.nodes.map((_, i) => against.y[`y${i}`] ?? ''),
        ),
      )
    })
  }

  it('loopy belief propagation and a custom MAP function are accepted', () => {
    const model = graphCrf(['a', 'b'], [])
    const g = graphAsGeneralCrf(model, { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] })
    const w = vector(5, model.layout.size)
    const bp = generalCrfOracle(g.crf, { inference: 'bp', map: 'maxProduct' }).expectations(w, g.instance)
    expect(bp.logPartition).toBeCloseTo(
      graphOracle(model).expectations(w, { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] }).logPartition,
      8,
    )
    const picked = generalCrfOracle(g.crf, {
      map: () => ({ assignment: {}, indices: { y0: 1, y1: 0 }, logValue: 0 }),
    }).decode(w, g.instance)
    expect(picked.y).toEqual({ y0: 'b', y1: 'a' })
  })
})
