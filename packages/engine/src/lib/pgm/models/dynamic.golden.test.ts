import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import * as P from '../index.ts'

describe('dynamic Bayes networks against pgmpy', () => {
  for (const c of loadFixture('pgm/models/dynamic').cases) {
    it(c.name, () => {
      const dbn = P.dynamicBayesNet(c.params['initial'] as P.BayesNode[], c.params['transition'] as P.TemplateNode[])
      const evidence = c.params['evidence'] as Record<string, string>[]
      const filtered = c.params['filtered'] as Record<string, number[]>[]
      const r = P.dbnFilter(dbn, evidence)
      for (const [t, expected] of filtered.entries()) {
        for (const [v, values] of Object.entries(expected)) {
          const got = P.factorValues(r.marginals[t]?.[v] as P.Factor)
          for (const [i, x] of values.entries()) expect(got[i]).toBeCloseTo(x, 12)
        }
      }
      // Smoothing on the unrolled network, and the evidence likelihood.
      const T = evidence.length
      const bn = P.unroll(dbn, T)
      const flat = Object.fromEntries(evidence.flatMap((ev, t) => Object.entries(ev).map(([k, v]) => [`${k}_${t}`, v])))
      for (const [v, values] of Object.entries(c.params['smoothed'] as Record<string, number[]>)) {
        const got = P.factorValues(P.variableElimination(bn, [`${v}_0`], { evidence: flat }).posterior)
        for (const [i, x] of values.entries()) expect(got[i]).toBeCloseTo(x, 12)
      }
      expect(r.logLikelihood).toBeCloseTo(P.logPartition(bn, { evidence: flat }), 12)
    })
  }
})

describe('dynamic Bayes networks', () => {
  it('validate their slices and describe their interface', () => {
    const z = { name: 'z', states: ['a', 'b'], parents: [], cpt: [0.5, 0.5] }
    expect(() => P.dynamicBayesNet([z], [])).toThrow(/same variables/)
    expect(() =>
      P.dynamicBayesNet([z], [{ name: 'z', states: ['a', 'c'], parents: [], previous: ['z'], cpt: [1, 0, 0, 1] }]),
    ).toThrow(/differs/)
    expect(() =>
      P.dynamicBayesNet([z], [{ name: 'z', states: ['a', 'b'], parents: [], previous: ['z'], cpt: [1, 0, 0.5, 0.6] }]),
    ).toThrow(/distribution/)
    expect(() => P.unroll(P.dynamicBayesNet([z], [{ ...z, previous: [] }]), 0)).toThrow(/positive/)
    expect(P.interfaceVariables(P.dynamicBayesNet([z], [{ ...z, previous: [] }]))).toEqual([])
  })
})
