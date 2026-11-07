import { describe, expect, it } from 'vitest'
import { isMdmlError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import {
  additiveChi2Sampler,
  linearKernel,
  nystroem,
  orthogonalRandomFeatures,
  polynomialCountSketch,
  randomFourierFeatures,
  skewedChi2Sampler,
  tensorSketch,
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
})
