import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectDistribution, normalCdf } from '../../testing/sampling.ts'
import { exponential } from '../dists/index.ts'
import { aliasTable, key } from '../random/index.ts'
import {
  aliasMethod,
  boxMuller,
  discreteInverseTransform,
  inverseTransformFromCdf,
  inverseTransformSample,
  marsagliaPolar,
} from './exact.ts'

describe('inverse-transform sampling', () => {
  it('draws from a distribution through its quantile function', () => {
    expectDistribution(inverseTransformSample(key(1), 2000, exponential(2)).samples.data, (x) => 1 - Math.exp(-2 * x))
    const f = inverseTransformSample(key(2), 1000, (u) => -Math.log(1 - u) / 2)
    expectDistribution(f.samples.data, (x) => 1 - Math.exp(-2 * x))
    expect(f.steps).toBeUndefined()
    expect(() => inverseTransformSample(key(0), -1, (u) => u)).toThrow()
  })

  it('inverts a CDF numerically, growing the bracket to far-away mass', () => {
    const r = inverseTransformFromCdf(key(3), 1000, (x) => normalCdf(x - 100), { steps: true })
    expectDistribution(r.samples.data, (x) => normalCdf(x - 100))
    expect(r.steps?.every((s) => s.iterations <= 200)).toBe(true)
  })

  it('searches discrete categories in order', () => {
    const p = [0.2, 0.5, 0.3]
    const r = discreteInverseTransform(key(4), 6000, p, { steps: true })
    const counts = [0, 0, 0]
    for (const v of r.samples.data) counts[v] = (counts[v] ?? 0) + 1
    counts.forEach((c, i) => {
      expect(Math.abs(c / 6000 - (p[i] ?? 0))).toBeLessThan(0.025)
    })
    expect(r.steps?.every((s) => s.comparisons === s.index + 1)).toBe(true)
    expect(() => discreteInverseTransform(key(0), 1, [0, 0])).toThrow()
  })
})

describe('Gaussian samplers', () => {
  it('Box-Muller produces standard normals, including an odd count', () => {
    const r = boxMuller(key(5), 2001, { steps: true })
    expect(r.samples.shape).toEqual([2001])
    expectDistribution(r.samples.data, normalCdf)
    const s = r.steps?.[0]
    expect(s?.z1).toBeCloseTo((s?.r ?? 0) * Math.cos(s?.theta ?? 0), 14)
  })

  it('the polar method produces standard normals and accepts pi / 4 of the pairs', () => {
    const r = marsagliaPolar(key(6), 2000, { steps: true })
    expectDistribution(r.samples.data, normalCdf)
    const steps = r.steps ?? []
    expect(steps.filter((s) => s.accepted).every((s) => s.s < 1)).toBe(true)
  })
})

describe('aliasMethod', () => {
  it('builds the same table as random.aliasTable for any weights', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: 0, max: 10, noNaN: true }), { minLength: 1, maxLength: 12 }), (w) => {
        fc.pre(w.reduce((a, b) => a + b, 0) > 0)
        const ours = aliasMethod(key(0), 1, w).table
        const theirs = aliasTable(w)
        expect(Array.from(ours.alias.data)).toEqual(Array.from(theirs.alias.data))
        expect(Array.from(ours.probability.data)).toEqual(Array.from(theirs.probability.data))
      }),
      { numRuns: RUNS },
    )
  })

  it('draws in proportion to the weights and records each step', () => {
    const w = [1, 2, 3, 4]
    const r = aliasMethod(key(7), 10000, w, { steps: true })
    const counts = [0, 0, 0, 0]
    for (const v of r.samples.data) counts[v] = (counts[v] ?? 0) + 1
    counts.forEach((c, i) => {
      expect(Math.abs(c / 10000 - (w[i] ?? 0) / 10)).toBeLessThan(0.02)
    })
    expect(r.steps?.length).toBeLessThanOrEqual(3)
    expect(() => aliasMethod(key(0), 1, [])).toThrow()
  })
})
