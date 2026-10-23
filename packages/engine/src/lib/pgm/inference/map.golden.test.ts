import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { key } from '../../random/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'

interface NetworkJson {
  readonly variables: P.DiscreteVariable[]
  readonly potentials: { variables: string[]; cardinality: number[]; theta: number[] }[]
}
const networkOf = (p: Readonly<Record<string, unknown>>): P.MarkovNetwork => {
  const n = p['network'] as NetworkJson
  return P.markovNetwork(
    n.variables,
    n.potentials.map((f) => P.logFactor(f.variables, f.cardinality, f.theta)),
  )
}

describe('MAP by LP relaxation against pgmpy', () => {
  for (const c of loadFixture('pgm/inference/map').cases) {
    it(c.name, () => {
      const mn = networkOf(c.params)
      const r = P.mplp(mn)
      const theirs = c.params['assignment'] as Record<string, number>
      // Where pgmpy's decoding is the optimum the assignments agree; elsewhere ours is at least as good.
      if (c.params['optimal'] === true) expect(r.indices).toEqual(theirs)
      else expect(r.logValue).toBeGreaterThan(P.unnormalisedLogProbability(mn, theirs))
      // The dual bound is never below the exact MAP value; when the gap closes the assignment is the MPE.
      const exact = P.maxProductElimination(mn)
      expect(r.dualBound).toBeGreaterThanOrEqual(exact.logValue - 1e-9)
      if (r.integralityGap < 1e-9) expect(r.indices).toEqual(exact.indices)
    })
  }
})

describe('MAP inference', () => {
  const ferro = P.isingModel({ rows: 3, cols: 3, coupling: 0.6, field: { s_0_0: 0.3, s_2_2: -0.1, s_1_1: 0.2 } })
  const exact = P.maxProductElimination(ferro)

  it('agrees across max-product elimination, junction trees, max-product BP, MPLP and local search', () => {
    expect(P.junctionTreeMap(ferro).assignment).toEqual(exact.assignment)
    const bp = P.maxProductMap(ferro)
    expect(bp.converged).toBe(true)
    expect(bp.assignment).toEqual(exact.assignment)
    const lp = P.mplp(ferro)
    expect(lp.integralityGap).toBeLessThan(1e-9)
    expect(lp.assignment).toEqual(exact.assignment)
    expect(lp.logValue).toBeCloseTo(exact.logValue, 10)
    expect(P.localSearchMap(ferro, { key: key(1), restarts: 20 }).logValue).toBeCloseTo(exact.logValue, 10)
  })

  it('MPLP never increases its dual bound, and local search stops at local maxima', () => {
    const frustrated = P.isingModel({ rows: 3, cols: 3, coupling: -0.8, field: { s_0_0: 0.1 } })
    const rec = recorder()
    const r = P.mplp(frustrated, { tracer: rec })
    const dual = metricSeries(rec.run(), 'dual').values
    for (let i = 1; i < dual.length; i++) expect(dual[i] ?? 0).toBeLessThanOrEqual((dual[i - 1] ?? 0) + 1e-9)
    expect(r.dualBound).toBeGreaterThanOrEqual(P.maxProductElimination(frustrated).logValue - 1e-9)
    const icm = P.localSearchMap(frustrated, { init: { s_0_0: 0 } })
    expect(icm.logValue).toBeLessThanOrEqual(P.maxProductElimination(frustrated).logValue + 1e-12)
  })

  it('respects evidence', () => {
    const r = P.mplp(ferro, { evidence: { s_1_1: '-1' } })
    const e = P.maxProductElimination(ferro, { evidence: { s_1_1: '-1' } })
    expect(r.logValue).toBeLessThanOrEqual(e.logValue + 1e-12)
    expect(r.assignment['s_1_1']).toBeUndefined()
    expect(P.maxProductMap(ferro, { evidence: { s_1_1: '-1' } }).assignment).toEqual(e.assignment)
  })
})
