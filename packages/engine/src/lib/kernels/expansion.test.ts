import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { createContext } from '../context/index.ts'
import { key, normal, uniform } from '../random/index.ts'
import { matmul, mul, swapaxes, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  additiveChi2Kernel,
  additiveChi2Sampler,
  approximateGram,
  approximationError,
  exponentialKernel,
  type FeatureMap,
  type Kernel,
  laplacianKernel,
  linearKernel,
  maternKernel,
  nystroem,
  orthogonalRandomFeatures,
  periodicKernel,
  polynomialCountSketch,
  polynomialKernel,
  randomFourierFeatures,
  rationalQuadraticKernel,
  rbfKernel,
  scaleKernel,
  skewedChi2Kernel,
  skewedChi2Sampler,
  tensorSketch,
} from './index.ts'

const X = mul(normal(key(5), [20, 3]), 0.8)
const H = uniform(key(6), [15, 4], { high: 2 })

function maxError(kernel: Kernel, map: FeatureMap, x: Value = X): number {
  return approximationError(kernel.k(x), approximateGram(map, x)).max
}

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('random Fourier features', () => {
  const kernels: readonly (readonly [string, Kernel])[] = [
    ['rbf', rbfKernel({ lengthScale: 0.9 })],
    ['rbf ard', rbfKernel({ lengthScale: [0.5, 1, 2] })],
    ['matern 1/2', maternKernel({ nu: 0.5, lengthScale: 1.2 })],
    ['matern 3/2', maternKernel({ nu: 1.5 })],
    ['matern 5/2', maternKernel({ nu: 2.5, lengthScale: [0.8, 1.2, 1.6] })],
    ['matern inf', maternKernel({ nu: Number.POSITIVE_INFINITY })],
    ['matern 1.2', maternKernel({ nu: 1.2, lengthScale: 1.1 })],
    ['exponential', exponentialKernel({ lengthScale: 1.4 })],
    ['laplacian', laplacianKernel({ lengthScale: [1, 2, 1.5] })],
    ['rational quadratic', rationalQuadraticKernel({ alpha: 1.5, lengthScale: 0.9 })],
    ['scaled rbf', scaleKernel(rbfKernel({ lengthScale: 1.1 }), 2.5)],
  ]
  for (const [name, kernel] of kernels) {
    it(`${name}: the approximation error shrinks with the number of features`, () => {
      const error = (n: number): number =>
        mean(
          [0, 1, 2].map((s) =>
            maxError(kernel, randomFourierFeatures({ inputDim: 3, nComponents: n, kernel, key: key(s) })),
          ),
        )
      const few = error(32)
      const many = error(4096)
      expect(many).toBeLessThan(few)
      expect(many).toBeLessThan(0.1 * Math.sqrt(name === 'scaled rbf' ? 2.5 : 1))
    })
  }
  it('uses given draws exactly, and draws reproducibly from a key or a context', () => {
    const map = randomFourierFeatures({ inputDim: 3, nComponents: 10, key: key(1) })
    expect(map.weights.shape).toEqual([3, 10])
    expect(toFlat(map.offset).every((b) => b >= 0 && b < 2 * Math.PI)).toBe(true)
    const again = randomFourierFeatures({ weights: map.weights, offset: map.offset })
    expect(toArray(again.transform(X))).toEqual(toArray(map.transform(X)))
    const ctx = createContext({ seed: 4 })
    expect(toArray(randomFourierFeatures({ inputDim: 3, nComponents: 5, ctx }).weights)).toEqual(
      toArray(randomFourierFeatures({ inputDim: 3, nComponents: 5, ctx }).weights),
    )
    expect(randomFourierFeatures({ inputDim: 3, key: key(0) }).transform([1, 2, 3]).shape).toEqual([100])
    expect(() => randomFourierFeatures({ inputDim: 3, kernel: periodicKernel(), key: key(0) })).toThrow()
    expect(() =>
      randomFourierFeatures({ inputDim: 3, kernel: exponentialKernel({ power: 1.5 }), key: key(0) }),
    ).toThrow()
    expect(() => randomFourierFeatures({ weights: [[1, 2]], offset: [1] })).toThrow()
  })
  it('orthogonal random features are unbiased with lower error than independent ones', () => {
    const d = 8
    const x = mul(normal(key(9), [24, d]), 0.5)
    const kernel = rbfKernel({ lengthScale: 2 })
    const err = (map: FeatureMap): number => approximationError(kernel.k(x), approximateGram(map, x)).relative
    const seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    const iid = mean(
      seeds.map((s) => err(randomFourierFeatures({ inputDim: d, nComponents: 32, kernel, key: key(s) }))),
    )
    const orf = mean(
      seeds.map((s) => err(orthogonalRandomFeatures({ inputDim: d, nComponents: 32, lengthScale: 2, key: key(s) }))),
    )
    expect(orf).toBeLessThan(iid)
    const many = orthogonalRandomFeatures({ inputDim: d, nComponents: 4000, lengthScale: 2, key: key(11) })
    expect(err(many)).toBeLessThan(0.05)
    // Rows of each block are orthogonal.
    const w = orthogonalRandomFeatures({ inputDim: 3, nComponents: 3, key: key(2) }).weights
    const gramW = toArray(matmul(swapaxes(w, 0, 1), w)) as number[][]
    expect(Math.abs(gramW[0]?.[1] ?? 1)).toBeLessThan(1e-12)
  })
  it('feature maps are differentiable in the inputs', () => {
    const p = tensor([
      [0.2, -0.3, 0.5],
      [1, 0.4, -0.7],
    ])
    expectConformant(
      (x: Value) => randomFourierFeatures({ inputDim: 3, nComponents: 6, key: key(0) }).transform(x),
      [p],
    )
    expectConformant(
      (x: Value) => orthogonalRandomFeatures({ inputDim: 3, nComponents: 5, key: key(0) }).transform(x),
      [p],
    )
    expectConformant((x: Value) => nystroem(rbfKernel(), X, { nComponents: 4, key: key(0) }).transform(x), [p])
    expectConformant(
      (x: Value) =>
        polynomialCountSketch({ inputDim: 3, degree: 2, coef0: 1, nComponents: 8, key: key(0) }).transform(x),
      [p],
    )
    const q = tensor([[0.2, 0.3, 1.5, 0.9]])
    expectConformant((x: Value) => additiveChi2Sampler({ sampleSteps: 2 }).transform(x), [q])
    expectConformant((x: Value) => skewedChi2Sampler({ inputDim: 4, nComponents: 6, key: key(0) }).transform(x), [q])
  })
})

describe('Nyström', () => {
  it('is exact with every point as a landmark and improves with more landmarks', () => {
    const kernel = maternKernel({ nu: 2.5 })
    const all = nystroem(kernel, X, { indices: Array.from({ length: 20 }, (_, i) => i) })
    expect(approximationError(kernel.k(X), approximateGram(all, X)).max).toBeLessThan(1e-8)
    const error = (m: number): number =>
      mean([0, 1, 2].map((s) => maxError(kernel, nystroem(kernel, X, { nComponents: m, key: key(s) }))))
    expect(error(12)).toBeLessThan(error(3))
    const landmarks = nystroem(kernel, X, { nComponents: 5, key: key(0) })
    expect(new Set(landmarks.componentIndices).size).toBe(5)
    expect(landmarks.transform([0, 0, 0]).shape).toEqual([5])
  })
  it('reproduces a low-rank kernel exactly from enough landmarks', () => {
    const kernel = linearKernel()
    const map = nystroem(kernel, X, { nComponents: 6, key: key(1) })
    expect(approximationError(kernel.k(X), approximateGram(map, X)).relative).toBeLessThan(1e-9)
  })
})

describe('histogram feature maps', () => {
  it('the additive chi-square sampler approximates the additive chi-square kernel, better with more steps', () => {
    const kernel = additiveChi2Kernel()
    const rel = (steps: number): number =>
      approximationError(kernel.k(H), approximateGram(additiveChi2Sampler({ sampleSteps: steps }), H)).relative
    expect(rel(3)).toBeLessThan(rel(1))
    expect(rel(3)).toBeLessThan(0.05)
    expect(() => additiveChi2Sampler({ sampleSteps: 4 })).toThrow()
    expect(() => additiveChi2Sampler({ sampleSteps: 0, sampleInterval: 1 })).toThrow()
    expect(() => additiveChi2Sampler().transform([[-1, 2]])).toThrow()
  })
  it('skewed chi-square features approximate the skewed chi-square kernel', () => {
    const kernel = skewedChi2Kernel({ skewedness: 0.5 })
    const error = (n: number): number =>
      mean(
        [0, 1, 2].map((s) =>
          maxError(kernel, skewedChi2Sampler({ inputDim: 4, nComponents: n, skewedness: 0.5, key: key(s) }), H),
        ),
      )
    expect(error(4096)).toBeLessThan(error(32))
    expect(error(4096)).toBeLessThan(0.1)
    expect(() => skewedChi2Sampler({ inputDim: 4, key: key(0) }).transform([[-2, 0, 0, 0]])).toThrow()
  })
})

describe('sketches', () => {
  it('the polynomial count sketch is unbiased for the polynomial kernel and sharpens with its size', () => {
    const x = mul(X, 0.7)
    const kernel = polynomialKernel({ degree: 2, gamma: 0.6, coef0: 0.5 })
    const exact = toFlat(kernel.k(x))
    const sketchGram = (n: number, s: number): number[] =>
      toFlat(
        approximateGram(
          polynomialCountSketch({ inputDim: 3, degree: 2, gamma: 0.6, coef0: 0.5, nComponents: n, key: key(s) }),
          x,
        ),
      )
    const seeds = Array.from({ length: 60 }, (_, s) => s)
    const average = exact.map(() => 0)
    for (const s of seeds)
      sketchGram(256, s).forEach((v, i) => {
        average[i] = (average[i] ?? 0) + v / seeds.length
      })
    // The average of 60 independent sketches is within a few standard errors of the kernel.
    const bias = average.reduce((m, v, i) => Math.max(m, Math.abs(v - (exact[i] ?? 0)) / Math.max(1, exact[i] ?? 0)), 0)
    expect(bias).toBeLessThan(0.1)
    const err = (n: number): number =>
      mean([0, 1, 2, 3].map((s) => sketchGram(n, s).reduce((m, v, i) => Math.max(m, Math.abs(v - (exact[i] ?? 0))), 0)))
    expect(err(2048)).toBeLessThan(err(16))
  })
  it('a tensor sketch of two feature vectors estimates the product of their inner products', () => {
    const a = mul(normal(key(20), [6, 3]), 0.6)
    const b = mul(normal(key(21), [6, 2]), 0.6)
    const exact = toFlat(mul(linearKernel().k(a), linearKernel().k(b)))
    const sketched = (s: number): number[] => {
      const t = tensorSketch({ inputDims: [3, 2], nComponents: 512, key: key(s) })
      const z = t.sketch(a, b)
      return toFlat(matmul(z, swapaxes(z, 0, 1)))
    }
    const avg = exact.map(() => 0)
    for (let s = 0; s < 20; s++)
      sketched(s).forEach((v, i) => {
        avg[i] = (avg[i] ?? 0) + v / 20
      })
    avg.forEach((v, i) => {
      expect(Math.abs(v - (exact[i] ?? 0))).toBeLessThan(0.1)
    })
    expect(() => tensorSketch({ indexHash: [[0]], bitHash: [[1, 1]], nComponents: 2 })).toThrow()
    expect(() => tensorSketch({ inputDims: [2], nComponents: 4, key: key(0) }).sketch([[1, 2]], [[1]])).toThrow()
  })
})

describe('approximation errors', () => {
  it('measures Frobenius, relative, spectral and maximum differences', () => {
    const e = approximationError(
      [
        [2, 0],
        [0, 2],
      ],
      [
        [1, 0],
        [0, 2],
      ],
    )
    expect(e.frobenius).toBe(1)
    expect(e.spectral).toBeCloseTo(1, 12)
    expect(e.max).toBe(1)
    expect(approximationError([[0]], [[0]]).relative).toBe(0)
    expect(approximationError([[0]], [[1]]).relative).toBe(Number.POSITIVE_INFINITY)
    expect(() => approximationError([[1]], [[1, 2]])).toThrow()
  })
})
