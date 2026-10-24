import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { add, mul, stack, toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

const n = 400
const x = normal(key(1), [n])
const noise = normal(key(2), [n])
const rho = 0.8
// Y = rho X + sqrt(1 - rho^2) N: a standard bivariate normal with correlation rho.
const y = add(mul(x, rho), mul(noise, Math.sqrt(1 - rho * rho)))
const trueMi = -0.5 * Math.log(1 - rho * rho)

describe('k-nearest-neighbour estimators', () => {
  it('Kozachenko-Leonenko recovers the entropy of a normal with either norm', () => {
    const h = 0.5 * Math.log(2 * Math.PI * Math.E)
    expect(Math.abs(I.kozachenkoLeonenkoEntropy(x) - h)).toBeLessThan(0.1)
    expect(Math.abs(I.kozachenkoLeonenkoEntropy(x, { metric: 'euclidean' }) - h)).toBeLessThan(0.1)
    const xy = stack([x, noise], 1)
    expect(Math.abs(I.kozachenkoLeonenkoEntropy(xy, { metric: 'euclidean', k: 5 }) - 2 * h)).toBeLessThan(0.15)
    expect(() => I.kozachenkoLeonenkoEntropy([1, 2, 3], { k: 3 })).toThrow()
    expect(() => I.kozachenkoLeonenkoEntropy([[[1]]])).toThrow()
  })

  it('KSG algorithms 1 and 2 recover the mutual information of a bivariate normal', () => {
    expect(Math.abs(I.ksgMutualInformation(x, y) - trueMi)).toBeLessThan(0.08)
    expect(Math.abs(I.ksgMutualInformation(x, y, { algorithm: 2 }) - trueMi)).toBeLessThan(0.08)
    expect(Math.abs(I.ksgMutualInformation(x, noise))).toBeLessThan(0.05)
    expect(I.ksgMutualInformation(x, y, { base: 2 })).toBeCloseTo(I.ksgMutualInformation(x, y) / Math.LN2, 12)
    expect(() => I.ksgMutualInformation(x, [1, 2])).toThrow()
  })

  it('conditional MI vanishes for conditionally independent variables', () => {
    const z = x
    const a = add(z, mul(normal(key(3), [n]), 0.5))
    const b = add(z, mul(normal(key(4), [n]), 0.5))
    expect(Math.abs(I.ksgConditionalMutualInformation(a, b, z))).toBeLessThan(0.05)
    expect(I.ksgMutualInformation(a, b)).toBeGreaterThan(0.3)
  })

  it('the k-NN KL divergence estimate is close to the Gaussian value', () => {
    const shifted = add(normal(key(5), [n]), 1)
    expect(Math.abs(I.knnKlDivergence(x, shifted) - 0.5)).toBeLessThan(0.12)
    expect(() => I.knnKlDivergence([[1, 2]], [[1]])).toThrow()
  })

  it("Ross's mixed estimator recovers a bit for separated classes", () => {
    const labels = Array.from({ length: n }, (_, i) => i % 2)
    const c = toFlat(x).map((v, i) => v * 0.1 + 10 * (labels[i] ?? 0))
    expect(Math.abs(I.mixedMutualInformation(c, labels) - Math.LN2)).toBeLessThan(0.02)
    // A label that occurs once is ignored.
    expect(Number.isFinite(I.mixedMutualInformation([...c, 3], [...labels, 7]))).toBe(true)
    expect(() => I.mixedMutualInformation([1, 2], [0, 1])).toThrow()
    expect(() => I.mixedMutualInformation([1, 2], [0])).toThrow()
  })
})
