import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { points } from '../../testing/kernels.ts'
import { key, normal, permutation } from '../random/index.ts'
import { add, matmul, mul, sin, sum, swapaxes, take, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  centerKernel,
  effectiveDimension,
  hsic,
  kernelAlignment,
  kernelPca,
  linearKernel,
  mmdSquared,
  pairwiseKernels,
  rbfKernel,
  targetAlignment,
} from './index.ts'

const rbf = rbfKernel({ lengthScale: 0.8 })
const X = normal(key(1), [12, 2])
const K = rbf.k(X)

describe('centring and alignment', () => {
  it('centred Gram matrices have zero row and column sums', () => {
    fc.assert(
      fc.property(points(6, 2), (x) => {
        const c = centerKernel(rbf.k(x))
        for (const v of toFlat(sum(c, 0))) expect(Math.abs(v)).toBeLessThan(1e-12)
        for (const v of toFlat(sum(c, 1))) expect(Math.abs(v)).toBeLessThan(1e-12)
      }),
      { numRuns: RUNS },
    )
    // Centring a training cross-kernel with itself as reference is plain centring.
    expect(toArray(centerKernel(K, K))).toEqual(toArray(centerKernel(K)))
    expect(() => centerKernel([[1, 2, 3]])).toThrow()
    expect(() => centerKernel([[1, 2, 3]], K)).toThrow()
  })
  it('alignment is a cosine in [-1, 1], 1 for proportional kernels', () => {
    fc.assert(
      fc.property(points(6, 2), fc.double({ min: 0.1, max: 5, noNaN: true }), (x, s) => {
        const a = rbf.k(x)
        const b = linearKernel({ sigma0: 1 }).k(x)
        const v = toFlat(kernelAlignment(a, b))[0] ?? Number.NaN
        expect(Math.abs(v)).toBeLessThanOrEqual(1 + 1e-12)
        expect(toFlat(kernelAlignment(a, mul(a, s)))[0]).toBeCloseTo(1, 12)
      }),
      { numRuns: RUNS },
    )
    expect(
      toFlat(
        targetAlignment(
          [
            [1, 1],
            [1, 1],
          ],
          [1, 1],
        ),
      )[0],
    ).toBeCloseTo(1, 14)
    expect(() =>
      kernelAlignment(
        [[1]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow()
  })
  it('alignment, HSIC, MMD, centring and the effective dimension are differentiable in the Gram matrices', () => {
    const L = linearKernel().k(X)
    expectConformant((a: Value) => kernelAlignment(a, L, { centred: true }), [K])
    expectConformant((a: Value) => hsic(a, L, { estimator: 'unbiased' }), [K])
    expectConformant((a: Value) => hsic(a, L, { normalise: true }), [K])
    expectConformant((a: Value) => centerKernel(a), [K])
    expectConformant((a: Value) => effectiveDimension(a, 0.3), [K])
    const sub = take(K, tensor([0, 1, 2, 3], { dtype: 'int32' }), 0)
    expectConformant(
      (a: Value) => mmdSquared(a, K, sub, { estimator: 'unbiased' }),
      [rbf.k(take(X, tensor([0, 1, 2, 3], { dtype: 'int32' }), 0))],
    )
  })
})

describe('HSIC and MMD', () => {
  it('HSIC is larger for dependent than for independent variables, and invariant to a joint permutation', () => {
    const x = normal(key(2), [60, 1])
    const noise = normal(key(3), [60, 1])
    const dependent = add(sin(mul(x, 2)), mul(noise, 0.1))
    const independent = normal(key(4), [60, 1])
    const kx = rbf.k(x)
    const dep = toFlat(hsic(kx, rbf.k(dependent), { estimator: 'unbiased' }))[0] ?? 0
    const ind = toFlat(hsic(kx, rbf.k(independent), { estimator: 'unbiased' }))[0] ?? 0
    expect(dep).toBeGreaterThan(10 * Math.abs(ind))
    const perm = permutation(key(5), 60)
    const px = take(x, perm, 0)
    const pd = take(dependent, perm, 0)
    expect(toFlat(hsic(rbf.k(px), rbf.k(pd)))[0]).toBeCloseTo(toFlat(hsic(kx, rbf.k(dependent)))[0] ?? 0, 12)
    expect(toFlat(hsic(kx, kx))[0]).toBeGreaterThan(0)
    expect(() =>
      hsic(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 0],
          [0, 1],
        ],
        { estimator: 'unbiased' },
      ),
    ).toThrow()
    expect(() =>
      hsic(
        [[1]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow()
  })
  it('MMD vanishes between a sample and itself and separates shifted samples', () => {
    const a = normal(key(6), [40, 2])
    const b = add(normal(key(7), [40, 2]), 1.5)
    const kaa = rbf.k(a)
    expect(toFlat(mmdSquared(kaa, kaa, kaa))[0]).toBeCloseTo(0, 12)
    const shifted = toFlat(mmdSquared(kaa, rbf.k(b), rbf.k(a, b), { estimator: 'unbiased' }))[0] ?? 0
    const same =
      toFlat(
        mmdSquared(kaa, rbf.k(normal(key(8), [40, 2])), rbf.k(a, normal(key(8), [40, 2])), { estimator: 'unbiased' }),
      )[0] ?? 0
    expect(shifted).toBeGreaterThan(10 * Math.abs(same))
    const linear = toFlat(mmdSquared(kaa, rbf.k(b), rbf.k(a, b), { estimator: 'linear' }))[0] ?? 0
    expect(linear).toBeGreaterThan(0)
    // Swapping the samples leaves the statistic unchanged.
    const ba = toFlat(mmdSquared(rbf.k(b), kaa, rbf.k(b, a)))[0] ?? 0
    expect(ba).toBeCloseTo(toFlat(mmdSquared(kaa, rbf.k(b), rbf.k(a, b)))[0] ?? 0, 12)
    expect(() => mmdSquared([[1]], [[1]], [[1, 2]])).toThrow()
    expect(() => mmdSquared([[1]], [[1]], [[1]], { estimator: 'unbiased' })).toThrow()
    expect(() =>
      mmdSquared(
        kaa,
        [[1]],
        toFlat(rbf.k(a, [[0, 0]])).map((v) => [v]),
        { estimator: 'linear' },
      ),
    ).toThrow()
  })
})

describe('effective dimension and kernel PCA', () => {
  it('the effective dimension lies in [0, n], falls with lambda and tends to the rank', () => {
    const lowRank = linearKernel().k(X)
    expect(toFlat(effectiveDimension(lowRank, 1e-6))[0]).toBeCloseTo(2, 5)
    const small = toFlat(effectiveDimension(K, 0.01))[0] ?? 0
    const large = toFlat(effectiveDimension(K, 10))[0] ?? 0
    expect(small).toBeGreaterThan(large)
    expect(small).toBeLessThanOrEqual(12)
    expect(large).toBeGreaterThan(0)
  })
  it('kernel PCA projections reproduce the centred Gram matrix and project the training points consistently', () => {
    const pca = kernelPca(K)
    const p = pca.projections
    const rebuilt = toFlat(matmul(p, swapaxes(p, 0, 1)))
    toFlat(centerKernel(K)).forEach((v, i) => {
      expect(Math.abs(v - (rebuilt[i] ?? 0))).toBeLessThan(1e-9)
    })
    const values = toFlat(pca.eigenvalues)
    for (let i = 1; i < values.length; i++) expect(values[i]).toBeLessThanOrEqual((values[i - 1] ?? 0) + 1e-12)
    expect(values.every((v) => v >= 0)).toBe(true)
    const two = kernelPca(K, { nComponents: 2 })
    const t = toFlat(two.transform(K))
    toFlat(two.projections).forEach((v, i) => {
      expect(Math.abs(v - (t[i] ?? 0))).toBeLessThan(1e-9)
    })
    expect(() => kernelPca(K, { nComponents: 13 })).toThrow()
  })
})

describe('pairwiseKernels', () => {
  it('accepts kernels as well as metric names', () => {
    expect(toArray(pairwiseKernels(X, undefined, rbf))).toEqual(toArray(K))
    expect(toArray(pairwiseKernels(X, X, rbf))).toEqual(toArray(rbf.k(X, X)))
    expect(toArray(pairwiseKernels([[1, 2]], [[3, 4]], 'linear'))).toEqual([[11]])
  })
})
