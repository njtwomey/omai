/**
 * Tests for kernel mean embeddings: the embedding algebra, the MMD witness against the library's MMD, the conditional
 * mean embedding (equal to kernel ridge regression, and recovering a planted conditional expectation), the kernel Bayes
 * rule (a Gaussian posterior and a tilted prior), and the Stein kernel (against its closed form for a Gaussian kernel)
 * with the goodness-of-fit test.
 */

import { describe, expect, it } from 'vitest'
import { gram, linearKernel, mmdSquared, rbfKernel } from '../kernels/index.ts'
import { key as makeKey } from '../random/index.ts'
import { item, neg, toFlat } from '../tensor/index.ts'
import { normals } from './_testdata.ts'
import {
  conditionalMeanEmbedding,
  embeddingInnerProduct,
  kernelBayesRule,
  kernelRidge,
  kernelSteinDiscrepancy,
  meanEmbedding,
  mmdWitness,
  steinGoodnessOfFit,
  steinKernelGram,
} from './index.ts'

const sample = (n: number, seed: number, mean = 0, sd = 1): number[][] => normals(n, seed).map((v) => [mean + sd * v])

describe('mean embeddings', () => {
  it('has a squared norm |mean|^2 for the linear kernel, and a symmetric inner product', () => {
    const a = meanEmbedding(linearKernel(), [
      [1, 2],
      [3, 4],
    ])
    expect(a.normSquared()).toBeCloseTo(2 ** 2 + 3 ** 2, 12)
    const b = meanEmbedding(linearKernel(), [[0, 1]])
    expect(embeddingInnerProduct(a, b)).toBeCloseTo(embeddingInnerProduct(b, a), 12)
    expect(embeddingInnerProduct(a, b)).toBeCloseTo(3, 12)
  })

  it('has a witness whose squared norm is the biased MMD of the library, positive where the first sample lives', () => {
    const kernel = rbfKernel({ lengthScale: 1 })
    const x = sample(60, 1, 0)
    const y = sample(60, 2, 2)
    const w = mmdWitness(kernel, x, y)
    const reference = item(mmdSquared(gram(kernel, x), gram(kernel, y), kernel.k(x, y)))
    expect(w.mmd2).toBeCloseTo(reference, 10)
    const f = toFlat(w.witness([[-0.5], [0], [2.5], [3]]))
    expect(f[0]).toBeGreaterThan(0)
    expect(f[3]).toBeLessThan(0)
    // Two samples of one distribution are closer than two distributions.
    expect(mmdWitness(kernel, x, sample(60, 3, 0)).mmd2).toBeLessThan(w.mmd2 / 3)
  })
})

describe('the conditional mean embedding', () => {
  const n = 150
  const x = normals(n, 11).map((v) => [2 * v])
  const noise = normals(n, 12)
  const y = x.map((r, i) => [Math.sin(r[0] ?? 0) + 0.1 * (noise[i] ?? 0)])
  const kx = rbfKernel({ lengthScale: 1 })
  const ky = rbfKernel({ lengthScale: 1 })

  it('is kernel ridge regression with the ridge n lambda for the identity', () => {
    const lambda = 0.01
    const cme = conditionalMeanEmbedding(kx, ky, x, y, lambda)
    const q = [[-1], [0.3], [1.7]]
    const a = toFlat(
      cme.expectation(
        y.map((r) => r[0] ?? 0),
        q,
      ),
    )
    const b = toFlat(
      kernelRidge
        .with({ kernel: kx, alpha: n * lambda })
        .fit({ x, y: y.map((r) => r[0] ?? 0) })
        .predict(q),
    )
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 8)
  })

  it('recovers the planted conditional expectations of y and y^2', () => {
    const cme = conditionalMeanEmbedding(kx, ky, x, y, 0.005)
    const q = [[-2], [-0.5], [0.5], [2]]
    const mean = toFlat(
      cme.expectation(
        y.map((r) => r[0] ?? 0),
        q,
      ),
    )
    const second = toFlat(
      cme.expectation(
        y.map((r) => (r[0] ?? 0) ** 2),
        q,
      ),
    )
    q.forEach((r, i) => {
      expect(Math.abs((mean[i] ?? 0) - Math.sin(r[0] ?? 0))).toBeLessThan(0.15)
      expect(Math.abs((second[i] ?? 0) - (Math.sin(r[0] ?? 0) ** 2 + 0.01))).toBeLessThan(0.15)
    })
  })

  it('has weights that sum to nearly one and an embedding equal to their sum of kernel columns', () => {
    const cme = conditionalMeanEmbedding(kx, ky, x, y, 0.005)
    const w = toFlat(cme.weights([[0.4]]))
    expect(w.reduce((a, b) => a + b, 0)).toBeGreaterThan(0.8)
    const z = [[0], [0.5]]
    const e = toFlat(cme.evaluate([[0.4]], z))
    const l = toFlat(ky.k(y, z))
    for (let c = 0; c < 2; c++) {
      let s = 0
      for (let i = 0; i < n; i++) s += (w[i] ?? 0) * (l[i * 2 + c] ?? 0)
      expect(e[c]).toBeCloseTo(s, 10)
    }
  })

  it('refuses a non-positive lambda and mismatched samples', () => {
    expect(() => conditionalMeanEmbedding(kx, ky, x, y, 0)).toThrow()
    expect(() => conditionalMeanEmbedding(kx, ky, x, y.slice(1))).toThrow()
  })
})

describe('the kernel Bayes rule', () => {
  const n = 220
  const x = normals(n, 21).map((v) => [v])
  const e = normals(n, 22)
  const sigma = 0.5
  const y = x.map((r, i) => [(r[0] ?? 0) + sigma * (e[i] ?? 0)])
  const kernel = rbfKernel({ lengthScale: 0.8 })
  const identity = x.map((r) => r[0] ?? 0)

  it('has the Gaussian posterior mean y / (1 + sigma^2) under the sample prior', () => {
    const post = kernelBayesRule(kernel, kernel, x, y, { epsilon: 1e-3, delta: 1e-2 })
    for (const obs of [-1, 0.5, 1]) {
      const m = item(post.expectation(identity, [[obs]]))
      expect(Math.abs(m - obs / (1 + sigma ** 2))).toBeLessThan(0.2)
    }
  })

  it('moves the posterior with the prior: a tilt towards positive inputs raises the posterior mean', () => {
    const flat = kernelBayesRule(kernel, kernel, x, y)
    const prior = x.map((r) => Math.exp(r[0] ?? 0))
    const total = prior.reduce((a, b) => a + b, 0)
    const tilted = kernelBayesRule(kernel, kernel, x, y, { prior: prior.map((v) => v / total) })
    // N(0, 1) tilted by exp(x) is N(1, 1): the posterior mean at y = 0 is 0.2 instead of 0.
    expect(item(tilted.expectation(identity, [[0]])) - item(flat.expectation(identity, [[0]]))).toBeGreaterThan(0.1)
  })

  it('refuses a prior of the wrong length', () => {
    expect(() => kernelBayesRule(kernel, kernel, x, y, { prior: [1] })).toThrow()
  })
})

describe('the kernel Stein discrepancy', () => {
  const l = 1.3
  const kernel = rbfKernel({ lengthScale: l })
  const score = (p: Parameters<typeof neg>[0]): ReturnType<typeof neg> => neg(p)

  it('builds the Stein kernel of a Gaussian kernel by autodiff, equal to its closed form', () => {
    const pts = [
      [0.1, 0.2],
      [1, -0.5],
      [-0.7, 0.4],
    ]
    const u = toFlat(steinKernelGram(kernel, score, pts))
    const dot = (a: number[], b: number[]): number => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)
    pts.forEach((a, i) => {
      pts.forEach((b, j) => {
        const diff = a.map((v, k) => v - (b[k] ?? 0))
        const r2 = dot(diff, diff)
        const sa = a.map((v) => -v)
        const sb = b.map((v) => -v)
        const expected =
          Math.exp(-r2 / (2 * l * l)) *
          (dot(sa, sb) + dot(sa, diff) / l ** 2 - dot(sb, diff) / l ** 2 + 2 / l ** 2 - r2 / l ** 4)
        expect(u[i * 3 + j]).toBeCloseTo(expected, 10)
      })
    })
  })

  it('is near zero for a sample of the model and large for a shifted one, and the test says so', () => {
    const good = sample(60, 31)
    const shifted = sample(60, 32, 1.2)
    expect(kernelSteinDiscrepancy(kernel, score, shifted, true)).toBeGreaterThan(
      5 * Math.abs(kernelSteinDiscrepancy(kernel, score, good, true)),
    )
    expect(steinGoodnessOfFit(kernel, score, good, { nBootstrap: 199, key: makeKey(1) }).pValue).toBeGreaterThan(0.05)
    expect(steinGoodnessOfFit(kernel, score, shifted, { nBootstrap: 199, key: makeKey(1) }).pValue).toBeLessThan(0.05)
  })
})
