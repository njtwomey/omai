import { describe, expect, it } from 'vitest'
import { randomPairwise } from '../../../testing/pairwise-models.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'

const exact = (model: P.DiscreteModel): number => P.maxProductElimination(model).logValue

describe('lpRelaxationMap', () => {
  it('is exact on trees: integral, the optimum, a zero gap', () => {
    for (const seed of [1, 2, 3]) {
      const model = randomPairwise({ seed, shape: 'chain', size: 6, labels: 3, kind: 'any' })
      const r = P.lpRelaxationMap(model)
      expect(r.integral).toBe(true)
      expect(r.logValue).toBeCloseTo(exact(model), 8)
      expect(r.dualBound).toBeCloseTo(exact(model), 8)
      expect(Math.abs(r.integralityGap)).toBeLessThan(1e-8)
    }
  })

  it('bounds the optimum from above on loopy models and sits at or below MPLP-free bounds', () => {
    for (const seed of [1, 2, 3, 4]) {
      const model = randomPairwise({ seed, shape: 'grid', size: 3, cols: 3, kind: 'any' })
      const r = P.lpRelaxationMap(model)
      const optimum = exact(model)
      expect(r.dualBound).toBeGreaterThanOrEqual(optimum - 1e-8)
      expect(r.logValue).toBeLessThanOrEqual(optimum + 1e-8)
      expect(r.integralityGap).toBeCloseTo(r.dualBound - r.logValue, 12)
      if (r.integral) expect(r.logValue).toBeCloseTo(optimum, 8)
    }
  })

  it('is tight on submodular binary models, where the cut is the MAP', () => {
    const model = randomPairwise({ seed: 2, shape: 'grid', size: 3, cols: 3, kind: 'submodular' })
    const r = P.lpRelaxationMap(model)
    expect(r.integral).toBe(true)
    expect(r.logValue).toBeCloseTo(P.graphCutMap(model).logValue, 8)
  })

  it('handles evidence and rejects infinite log values', () => {
    const model = randomPairwise({ seed: 2, shape: 'chain', size: 4, kind: 'any' })
    const r = P.lpRelaxationMap(model, { evidence: { v1: '1' } })
    expect(Object.keys(r.marginals)).not.toContain('v1')
    const hard = P.factorGraph(
      [{ name: 'a', states: ['0', '1'] }],
      [P.logFactor(['a'], [2], [0, Number.NEGATIVE_INFINITY])],
    )
    expect(() => P.lpRelaxationMap(hard)).toThrow(/finite/)
  })
})

describe('ad3Map', () => {
  it('reaches the MAP of tree models, as MPLP does', () => {
    for (const seed of [1, 2, 3]) {
      const model = randomPairwise({ seed, shape: 'chain', size: 6, labels: 3, kind: 'any' })
      const a = P.ad3Map(model)
      const m = P.mplp(model)
      expect(a.status).toBe('converged')
      expect(a.integral).toBe(true)
      expect(a.logValue).toBeCloseTo(exact(model), 8)
      expect(a.logValue).toBeCloseTo(m.logValue, 8)
      expect(Math.abs(a.integralityGap)).toBeLessThan(1e-6)
    }
  })

  for (const seed of [1, 2, 3, 4]) {
    it(`reaches the LP bound that MPLP approaches, on a loopy model (seed ${seed})`, () => {
      const model = randomPairwise({ seed, shape: 'grid', size: 3, cols: 3, kind: 'any' })
      const lp = P.lpRelaxationMap(model)
      const a = P.ad3Map(model, { maxIter: 3000, tol: 1e-8 })
      const m = P.mplp(model, { maxIter: 3000 })
      const optimum = exact(model)
      // The LP bound is below MPLP's (a fixed point of block coordinate descent is an upper bound of the LP optimum).
      expect(a.dualBound).toBeGreaterThanOrEqual(lp.dualBound - 1e-6)
      expect(a.dualBound).toBeLessThanOrEqual(lp.dualBound + 1e-3)
      expect(m.dualBound).toBeGreaterThanOrEqual(lp.dualBound - 1e-6)
      // bound >= optimum >= decoded value
      expect(a.dualBound).toBeGreaterThanOrEqual(optimum - 1e-8)
      expect(a.logValue).toBeLessThanOrEqual(optimum + 1e-8)
      expect(a.integralityGap).toBeCloseTo(a.dualBound - a.logValue, 12)
      if (a.integral) expect(a.logValue).toBeCloseTo(optimum, 6)
    })
  }

  it('reports the integrality gap of a frustrated triangle: LP, AD3 and MPLP bounds agree', () => {
    // Antiferromagnetic triangle: every pair can disagree in the relaxation (value 3) but not in a labelling (value 1).
    const tri = P.markovNetwork(
      ['a', 'b', 'c'].map((name) => ({ name, states: ['0', '1'] })),
      [
        P.logFactor(['a', 'b'], [2, 2], [-1, 1, 1, -1]),
        P.logFactor(['b', 'c'], [2, 2], [-1, 1, 1, -1]),
        P.logFactor(['a', 'c'], [2, 2], [-1, 1, 1, -1]),
      ],
    )
    const lp = P.lpRelaxationMap(tri)
    const a = P.ad3Map(tri, { maxIter: 5000, tol: 1e-9 })
    expect(lp.integral).toBe(false)
    expect(lp.dualBound).toBeCloseTo(3, 8)
    expect(exact(tri)).toBeCloseTo(1, 12)
    expect(a.integral).toBe(false)
    expect(a.dualBound).toBeCloseTo(3, 4)
    expect(a.integralityGap).toBeCloseTo(2, 4)
    expect(P.mplp(tri).dualBound).toBeCloseTo(3, 6)
    for (const v of ['a', 'b', 'c']) expect(a.marginals[v]?.[0]).toBeCloseTo(0.5, 3)
  })

  it('handles factors over three variables and evidence', () => {
    const vars = ['a', 'b', 'c'].map((name) => ({ name, states: ['0', '1'] }))
    const model = P.factorGraph(vars, [
      P.logFactor(['a'], [2], [0, 0.3]),
      P.logFactor(['a', 'b', 'c'], [2, 2, 2], [0.1, -0.2, 0.5, 1.5, -1, 0.2, 0.4, 0.9]),
      P.logFactor(['b'], [2], [0.2, 0]),
    ])
    const a = P.ad3Map(model)
    expect(a.status).toBe('converged')
    expect(a.logValue).toBeCloseTo(exact(model), 8)
    const ev = P.ad3Map(model, { evidence: { c: '0' } })
    expect(Object.keys(ev.indices)).toEqual(['a', 'b'])
  })

  it('traces the bound, the primal value, the gap and the residuals; the bound never undercuts the primal', () => {
    const model = randomPairwise({ seed: 3, shape: 'grid', size: 2, cols: 3, kind: 'any' })
    const rec = recorder()
    const a = P.ad3Map(model, { tracer: rec, maxIter: 300 })
    const bound = metricSeries(rec.run(), 'bound').values
    const primal = metricSeries(rec.run(), 'primal').values
    expect(bound.length).toBe(a.iterations + 1)
    for (let i = 0; i < bound.length; i++) expect(bound[i] ?? 0).toBeGreaterThanOrEqual((primal[i] ?? 0) - 1e-9)
    expect(metricSeries(rec.run(), 'primalResidual').values.length).toBeGreaterThan(1)
  })

  it('rejects a penalty that is not positive', () => {
    expect(() => P.ad3Map(randomPairwise({ seed: 1, shape: 'chain', size: 3 }), { rho: 0 })).toThrow(/rho/)
  })
})
