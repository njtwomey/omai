import { describe, expect, it } from 'vitest'
import { isMdmlError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { toArray, zeros } from '../tensor/index.ts'
import {
  additiveChi2Sampler,
  constantKernel,
  gram,
  hyperparameters,
  linearKernel,
  locallyPeriodicKernel,
  nystroem,
  orthogonalRandomFeatures,
  periodicKernel,
  polynomialCountSketch,
  powerKernel,
  randomFourierFeatures,
  rationalQuadraticKernel,
  rbfKernel,
  resolveKernel,
  scaleKernel,
  skewedChi2Sampler,
  tensorSketch,
  whiteNoiseKernel,
} from './index.ts'

/** The error code a call fails with, or `undefined` when it returns. */
function codeOf(f: () => unknown): string | undefined {
  try {
    f()
  } catch (e) {
    return isMdmlError(e) ? e.code : `not an MdmlError: ${String(e)}`
  }
  return undefined
}

describe('kernel entry points refuse arguments that would hang, wrap or give NaN', () => {
  const k = key(0)
  it('the random-feature constructors take positive integer sizes and positive scales', () => {
    const rows: [string, () => unknown][] = [
      ['orthogonal inputDim 0 (used to hang)', () => orthogonalRandomFeatures({ inputDim: 0, key: k })],
      ['orthogonal inputDim -1', () => orthogonalRandomFeatures({ inputDim: -1, key: k })],
      ['orthogonal nComponents 0', () => orthogonalRandomFeatures({ inputDim: 2, nComponents: 0, key: k })],
      ['orthogonal lengthScale 0', () => orthogonalRandomFeatures({ inputDim: 2, lengthScale: 0, key: k })],
      ['orthogonal lengthScale NaN', () => orthogonalRandomFeatures({ inputDim: 2, lengthScale: Number.NaN, key: k })],
      ['fourier inputDim -1 (a RangeError)', () => randomFourierFeatures({ inputDim: -1, key: k })],
      ['fourier inputDim 2.5', () => randomFourierFeatures({ inputDim: 2.5, key: k })],
      ['fourier nComponents 0', () => randomFourierFeatures({ inputDim: 2, nComponents: 0, key: k })],
      ['fourier nComponents NaN', () => randomFourierFeatures({ inputDim: 2, nComponents: Number.NaN, key: k })],
      ['skewed inputDim 0', () => skewedChi2Sampler({ inputDim: 0, key: k })],
      ['skewed nComponents 0', () => skewedChi2Sampler({ inputDim: 2, nComponents: 0, key: k })],
      ['skewed skewedness NaN', () => skewedChi2Sampler({ inputDim: 2, skewedness: Number.NaN, key: k })],
      ['additive chi2 interval 0', () => additiveChi2Sampler({ sampleSteps: 2, sampleInterval: 0 })],
      ['additive chi2 interval NaN', () => additiveChi2Sampler({ sampleSteps: 2, sampleInterval: Number.NaN })],
      ['additive chi2 steps NaN', () => additiveChi2Sampler({ sampleSteps: Number.NaN })],
      ['tensor sketch nComponents 0', () => tensorSketch({ inputDims: [2], nComponents: 0, key: k })],
      ['tensor sketch dimension 0', () => tensorSketch({ inputDims: [0], nComponents: 4, key: k })],
      [
        'tensor sketch hash out of range',
        () => tensorSketch({ indexHash: [[0, 4]], bitHash: [[1, 1]], nComponents: 4 }),
      ],
      ['count sketch gamma -1 (NaN features)', () => polynomialCountSketch({ inputDim: 2, gamma: -1, key: k })],
      ['count sketch coef0 -1', () => polynomialCountSketch({ inputDim: 2, coef0: -1, key: k })],
      ['count sketch degree 0 (ragged array)', () => polynomialCountSketch({ inputDim: 2, degree: 0, key: k })],
      ['count sketch nComponents 0', () => polynomialCountSketch({ inputDim: 2, nComponents: 0, key: k })],
      ['count sketch inputDim 0', () => polynomialCountSketch({ inputDim: 0, key: k })],
      ['nystroem nComponents 0', () => nystroem(linearKernel(), [[1], [2]], { nComponents: 0 })],
      ['nystroem landmark out of range', () => nystroem(linearKernel(), [[1], [2]], { indices: [2] })],
    ]
    for (const [name, f] of rows) expect(codeOf(f), name).toBe('value')
    // The defaults and valid sizes still build.
    expect(orthogonalRandomFeatures({ inputDim: 3, nComponents: 7, key: k }).weights.shape).toEqual([3, 7])
    expect(polynomialCountSketch({ inputDim: 2, degree: 1, coef0: 0, key: k }).indexHash.length).toBe(1)
  })
  it('a NaN feature gives NaN features in the additive chi2 map and a zero stays zero', () => {
    const z = toArray(additiveChi2Sampler({ sampleSteps: 2 }).transform([[0, Number.NaN, 2]])) as number[][]
    const row = z[0] as number[]
    // Features are laid out [plain x d, cos x d, sin x d]: the zero column is 0 in each block, the NaN column is NaN.
    expect([0, 3, 6].map((j) => row[j])).toEqual([0, 0, 0])
    expect([1, 4, 7].map((j) => Number.isNaN(row[j]))).toEqual([true, true, true])
    expect([2, 5, 8].every((j) => Number.isFinite(row[j]))).toBe(true)
  })
  it('hyperparameters are finite, and a scale, period, alpha or decay is positive', () => {
    const inf = Number.POSITIVE_INFINITY
    const rows: [string, () => unknown][] = [
      ['rbf lengthScale 0 (an all-NaN Gram)', () => rbfKernel({ lengthScale: 0 })],
      ['rbf lengthScale Infinity', () => rbfKernel({ lengthScale: inf })],
      ['rbf lengthScale NaN', () => rbfKernel({ lengthScale: Number.NaN })],
      ['rbf ard with a zero', () => rbfKernel({ lengthScale: [1, 0] })],
      ['rbf negative', () => rbfKernel({ lengthScale: -1 })],
      ['periodic period 0', () => periodicKernel({ period: 0 })],
      ['locally periodic decay 0', () => locallyPeriodicKernel({ decay: 0 })],
      ['rational quadratic alpha 0 (the wrong limit)', () => rationalQuadraticKernel({ alpha: 0 })],
      ['white noise Infinity', () => whiteNoiseKernel({ noiseLevel: inf })],
      ['scale by Infinity', () => scaleKernel(rbfKernel(), inf)],
      ['power NaN', () => powerKernel(linearKernel(), Number.NaN)],
      ['power Infinity', () => powerKernel(linearKernel(), inf)],
    ]
    for (const [name, f] of rows) expect(codeOf(f), name).toBe('domain')
    // A zero variance-like setting is still allowed and held fixed (log 0 never appears).
    for (const kernel of [
      whiteNoiseKernel({ noiseLevel: 0 }),
      constantKernel({ value: 0 }),
      scaleKernel(rbfKernel(), 0),
    ])
      expect(hyperparameters(kernel).some((h) => h.constraint === 'fixed' && h.value.every((v) => v === 0))).toBe(true)
  })
  it('gram is [n, n] for one point and for none, and refuses a jitter that is not a small non-negative number', () => {
    const kernel = rbfKernel()
    expect(toArray(gram(kernel, [1, 2]))).toEqual([[1]])
    expect(toArray(gram(kernel, [1, 2], { jitter: 0.1 }))).toEqual([[1.1]])
    expect(toArray(gram(kernel, 3, { jitter: 0.5 }))).toEqual([[1.5]])
    expect(gram(kernel, [[1, 2]]).shape).toEqual([1, 1])
    expect(gram(kernel, zeros([0, 2]), { jitter: 0.1 }).shape).toEqual([0, 0])
    for (const jitter of [Number.NaN, -1, Number.POSITIVE_INFINITY])
      expect(
        codeOf(() => gram(kernel, [[1], [2]], { jitter })),
        `jitter ${jitter}`,
      ).toBe('value')
  })
  it("gamma 'scale' is 1 / (d Var(X)) for data far from the origin, 1 for constant data, and needs finite rows", () => {
    const choice = { kernel: 'rbf', gamma: 'scale', degree: 3, coef0: 1 } as const
    // 50 rows of spread about 3 around 1e8: the single-pass formula cancelled to 0 and returned the fallback 1.
    const shifted = Array.from({ length: 50 }, (_, i) => [1e8 + ((i * 7) % 11)])
    const mean = shifted.reduce((a, r) => a + (r[0] ?? 0), 0) / 50
    const variance = shifted.reduce((a, r) => a + ((r[0] ?? 0) - mean) ** 2, 0) / 50
    expect(resolveKernel(choice, shifted).gamma).toBeCloseTo(1 / variance, 10)
    expect(resolveKernel(choice, [[3], [3], [3]]).gamma).toBe(1)
    expect(codeOf(() => resolveKernel(choice, [[1], [Number.NaN]]))).toBe('value')
    expect(resolveKernel({ ...choice, gamma: 'auto' }, [[1, 2]]).gamma).toBe(0.5)
  })
})
