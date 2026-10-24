import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { exp, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const lambdas = fc.integer({ min: -20, max: 30 }).map((v) => v / 10)

describe('power transforms', () => {
  it('invert exactly', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.integer({ min: 1, max: 500 }).map((v) => v / 50),
          { minLength: 2, maxLength: 20 },
        ),
        lambdas,
        (x, lambda) => {
          const y = S.boxcox(x, { lambda }).transformed
          toFlat(S.invBoxcox(y, lambda)).forEach((v, i) => {
            expect(v).toBeCloseTo(x[i] ?? 0, 8)
          })
          const z = x.map((v) => v - 5)
          const w = S.yeojohnson(z, { lambda }).transformed
          toFlat(S.invYeojohnson(w, lambda)).forEach((v, i) => {
            expect(v).toBeCloseTo(z[i] ?? 0, 8)
          })
        },
      ),
    )
  })

  it('the fitted lambda maximises the log-likelihood and recovers log-normal data', () => {
    const x = toFlat(exp(normal(key(9), [400])))
    const lambda = S.boxcoxNormmax(x)
    expect(Math.abs(lambda)).toBeLessThan(0.15)
    for (const d of [-0.05, 0.05]) expect(S.boxcoxLlf(lambda + d, x)).toBeLessThan(S.boxcoxLlf(lambda, x))
    const r = S.boxcox(x, { level: 0.95 })
    expect(r.interval?.[0]).toBeLessThan(lambda)
    expect(r.interval?.[1]).toBeGreaterThan(lambda)
    const ly = S.yeojohnsonNormmax(x)
    for (const d of [-0.05, 0.05]) expect(S.yeojohnsonLlf(ly + d, x)).toBeLessThan(S.yeojohnsonLlf(ly, x))
  })

  it('standardise centres and scales each column', () => {
    const s = S.standardise([
      [1, 5],
      [2, 5],
      [3, 5],
    ])
    expect(toFlat(s.transformed)).toEqual([-1.224744871391589, 0, 0, 0, 1.224744871391589, 0])
    expect(toFlat(s.scale)).toEqual([0.816496580927726, 1])
    expect(() => S.boxcox([1, -1])).toThrow()
  })
})
