import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { item, mean, toArray, toFlat, type Value, variance } from '../../tensor/index.ts'
import {
  batchNorm,
  groupNorm,
  instanceNorm,
  layerNorm,
  localResponseNorm,
  normalize,
  rmsNorm,
  spectralNorm,
  weightNorm,
} from './normalisation.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('normalisation gradients', () => {
  it('batch normalisation in training and evaluation', () => {
    ok((x, w, b) => batchNorm(x, { weight: w, bias: b }).output, [n(0, [4, 3, 2]), n(1, [3]), n(2, [3])])
    const rv = uniform(key(3), [3], { low: 0.5, high: 2 })
    ok(
      (x, w) => batchNorm(x, { weight: w, runningMean: n(4, [3]), runningVar: rv, training: false }).output,
      [n(5, [4, 3]), n(6, [3])],
    )
  })
  it('layer, RMS, group and instance normalisation', () => {
    ok((x, w, b) => layerNorm(x, [3], { weight: w, bias: b }), [n(7, [2, 3]), n(8, [3]), n(9, [3])])
    ok((x, w) => rmsNorm(x, [2, 3], { weight: w }), [n(10, [2, 2, 3]), n(11, [2, 3])])
    ok((x, w, b) => groupNorm(x, 2, { weight: w, bias: b }), [n(12, [2, 4, 3]), n(13, [4]), n(14, [4])])
    ok((x) => instanceNorm(x).output, [n(15, [2, 2, 4])])
  })
  it('local response normalisation and normalize', () => {
    ok((x) => localResponseNorm(x, 3, 0.1), [n(16, [1, 4, 2])])
    ok((x) => normalize(x), [n(17, [2, 3])])
  })
  it('weight and spectral normalisation', () => {
    ok((v, g) => weightNorm(v, g), [n(18, [3, 2]), n(19, [3, 1])])
    ok((v) => weightNorm(v, [2], null), [n(20, [2, 2])])
    const u = n(21, [3])
    const v = n(22, [2])
    // The power iteration is detached (as in PyTorch), so the exact derivative is that of W / (u^T W v) at fixed vectors.
    ok((w) => spectralNorm(w, u, v, { training: false }).weight, [n(23, [3, 2])])
  })
})

describe('normalisation behaviour', () => {
  it('batch normalisation gives zero mean and unit variance per channel', () => {
    const y = batchNorm(n(0, [64, 2]), { eps: 0 }).output
    expect(Math.abs(item(mean(y)))).toBeLessThan(1e-12)
    expect(toFlat(variance(y, 0)).map((v) => Math.round(v * 1e9) / 1e9)).toEqual([1, 1])
  })
  it('a cumulative average weights each batch equally', () => {
    const r = batchNorm([[1], [3]], { runningMean: [10], runningVar: [1], momentum: null, numBatchesTracked: 1 })
    expect(toArray(r.runningMean as Value)).toEqual([6])
  })
  it('spectral normalisation converges to the largest singular value', () => {
    let u: Value = n(1, [2])
    let v: Value = n(2, [2])
    let sigma = 0
    for (let i = 0; i < 30; i++) {
      const r = spectralNorm(
        [
          [3, 1],
          [0, 2],
        ],
        u,
        v,
      )
      u = r.u
      v = r.v
      sigma = item(r.sigma)
    }
    // The singular values of [[3, 1], [0, 2]] are sqrt(7 +- sqrt(13)).
    expect(sigma).toBeCloseTo(Math.sqrt(7 + Math.sqrt(13)), 10)
  })
  it('rejects mismatched shapes', () => {
    expect(() => layerNorm(n(0, [2, 3]), [4])).toThrow()
    expect(() => groupNorm(n(0, [1, 3, 2]), 2)).toThrow()
    expect(() => instanceNorm(n(0, [1, 3, 2]), { useInputStats: false })).toThrow()
  })
})
