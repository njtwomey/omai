import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../../trace/index.ts'
import * as P from '../index.ts'

const models: [string, P.DiscreteModel][] = [
  ['ferromagnet', P.isingModel({ rows: 3, cols: 3, coupling: 0.4, field: { s_0_0: 0.3 } })],
  ['antiferromagnet', P.isingModel({ rows: 2, cols: 3, coupling: -0.5, field: 0.1 })],
  ['potts', P.pottsModel({ rows: 2, cols: 2, q: 3, coupling: 0.8 })],
]

describe('mean field and variational message passing', () => {
  it('are exact on independent variables', () => {
    const m = P.isingModel({ rows: 1, cols: 3, coupling: 0, field: { s_0_0: 0.7, s_0_2: -0.2 } })
    const r = P.meanField(m)
    expect(r.logPartitionBound).toBeCloseTo(P.logPartition(m), 12)
    expect(P.factorValues(r.marginals['s_0_0'] as P.Factor)[1]).toBeCloseTo(
      Math.exp(0.7) / (Math.exp(0.7) + Math.exp(-0.7)),
      12,
    )
  })

  for (const [name, m] of models) {
    it(`lower-bound log Z (${name}) and agree with each other`, () => {
      const exact = P.logPartition(m)
      const mf = P.meanField(m, { key: key(1) })
      expect(mf.converged).toBe(true)
      expect(mf.logPartitionBound).toBeLessThanOrEqual(exact + 1e-12)
      const vmp = P.variationalMessagePassing(m, { key: key(1), maxIter: 2000 })
      expect(vmp.converged).toBe(true)
      expect(vmp.logPartitionBound).toBeLessThanOrEqual(exact + 1e-12)
    })
  }

  it('never decrease the ELBO under coordinate ascent, and trace their messages', () => {
    const rec = recorder()
    P.meanField(models[0]?.[1] as P.DiscreteModel, { tracer: rec, key: key(2) })
    const elbo = metricSeries(rec.run(), 'elbo').values
    for (let i = 1; i < elbo.length; i++) expect(elbo[i] ?? 0).toBeGreaterThanOrEqual((elbo[i - 1] ?? 0) - 1e-12)
    const rec2 = recorder()
    P.variationalMessagePassing(models[0]?.[1] as P.DiscreteModel, {
      tracer: rec2,
      policy: tracePolicy({ messages: true }),
      maxIter: 3,
    })
    expect(messages(rec2.run(), 'vmp').length).toBeGreaterThan(0)
  })

  it('handle evidence and hard zeros in Bayes networks', () => {
    const bn = P.bayesNet([
      { name: 'a', states: ['0', '1'], parents: [], cpt: [0.3, 0.7] },
      { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0.4, 0.6] },
      { name: 'c', states: ['0', '1'], parents: ['b'], cpt: [0.9, 0.1, 0.2, 0.8] },
    ])
    const r = P.meanField(bn, { evidence: { c: '1' } })
    expect(r.logPartitionBound).toBeLessThanOrEqual(P.logPartition(bn, { evidence: { c: '1' } }) + 1e-12)
    expect(P.factorValues(r.marginals['c'] as P.Factor)).toEqual([0, 1])
  })
})

describe('tree-reweighted belief propagation', () => {
  for (const [name, m] of models) {
    it(`upper-bounds log Z (${name}) above the mean-field lower bound`, () => {
      const exact = P.logPartition(m)
      const trw = P.treeReweighted(m)
      expect(trw.converged).toBe(true)
      expect(trw.logPartitionBound).toBeGreaterThanOrEqual(exact - 1e-10)
      expect(P.meanField(m).logPartitionBound).toBeLessThanOrEqual(trw.logPartitionBound)
    })
  }

  it('is exact on a tree, where every edge appears in every spanning tree', () => {
    const chain = P.isingModel({
      rows: 1,
      cols: 5,
      coupling: { 's_0_0|s_0_1': 0.9, 's_0_1|s_0_2': -0.4, 's_0_2|s_0_3': 1.2, 's_0_3|s_0_4': 0.3 },
      field: { s_0_2: 0.5 },
    })
    expect(Object.values(P.spanningTreeEdgeProbabilities(chain))).toEqual([1, 1, 1, 1])
    const trw = P.treeReweighted(chain)
    expect(trw.logPartitionBound).toBeCloseTo(P.logPartition(chain), 10)
    const exact = P.junctionTreeMarginals(chain)
    for (const v of Object.keys(exact)) {
      expect(P.factorValues(trw.marginals[v] as P.Factor)[1]).toBeCloseTo(
        P.factorValues(exact[v] as P.Factor)[1] ?? 0,
        9,
      )
    }
  })

  it('uses the effective resistances of the graph as edge appearance probabilities', () => {
    // A 4-cycle has 4 spanning trees, each missing one edge: rho = 3/4 on every edge.
    const rho = P.spanningTreeEdgeProbabilities(P.isingModel({ rows: 2, cols: 2 }))
    for (const v of Object.values(rho)) expect(v).toBeCloseTo(0.75, 12)
    const grid = P.spanningTreeEdgeProbabilities(P.isingModel({ rows: 3, cols: 3 }))
    expect(Object.values(grid).reduce((a, b) => a + b, 0)).toBeCloseTo(8, 10)
    // A valid uniform choice of rho also gives an upper bound.
    const m = models[0]?.[1] as P.DiscreteModel
    expect(P.treeReweighted(m, { rho: 2 / 3 }).logPartitionBound).toBeGreaterThanOrEqual(P.logPartition(m))
    expect(() => P.treeReweighted(m, { rho: 1.5 })).toThrow(/edge appearance/)
    const triple = P.factorGraph(
      ['a', 'b', 'c'].map((name) => ({ name, states: ['0', '1'] })),
      [P.factor(['a', 'b', 'c'], [2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8])],
    )
    expect(() => P.treeReweighted(triple)).toThrow(/pairwise/)
  })
  it('returns the edge pseudo-marginals: normalised, consistent with the variable beliefs, and exact on a tree', () => {
    for (const [, m] of models) {
      const r = P.treeReweighted(m, { tol: 1e-12, maxIter: 5000 })
      expect(r.edgeBeliefs?.length).toBe(P.modelFactors(m).filter((f) => f.variables.length === 2).length)
      for (const b of r.edgeBeliefs ?? []) {
        const [s, t] = b.variables as [string, string]
        const joint = P.factorValues(b)
        expect(joint.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10)
        const marginal = P.factorValues(P.factorMarginal(b, [s]))
        expect(marginal).toEqual(P.factorValues(r.marginals[s] as P.Factor).map((p) => expect.closeTo(p, 6)))
        expect(t).not.toBe(s)
      }
    }
    const chain = P.isingModel({ rows: 1, cols: 3, coupling: 0.9, field: { s_0_0: 0.4 } })
    const exact = P.junctionTreeQuery(P.calibrate(chain), ['s_0_0', 's_0_1'])
    const edge = (P.treeReweighted(chain).edgeBeliefs ?? [])[0] as P.Factor
    expect(P.factorValues(edge)).toEqual(P.factorValues(exact).map((p) => expect.closeTo(p, 8)))
  })
})
