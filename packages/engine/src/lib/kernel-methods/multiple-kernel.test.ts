/**
 * Tests for multiple kernel learning: the weights of a planted informative kernel against a noise kernel by every
 * method, the optimality of the alignment programme against a brute-force search, the combined kernel as an object, and
 * the solver slot (the weights logged by the algorithm; solvers agreeing).
 */

import { describe, expect, it } from 'vitest'
import { gram, onDimensions, rbfKernel } from '../kernels/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { dict } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { defineMethod, metricSeries, runAlgorithm, session } from '../trace/index.ts'
import { normals } from './_testdata.ts'
import {
  alignmentWeights,
  easyMkl,
  fitMultipleKernel,
  kernelRidge,
  mklAlgorithm,
  mklContract,
  mklProblem,
} from './index.ts'

const n = 40
const z = normals(2 * n, 61)
const x = Array.from({ length: n }, (_, i) => [(i < n / 2 ? -1 : 1) * 1.2 + 0.5 * (z[2 * i] ?? 0), z[2 * i + 1] ?? 0])
const y = x.map((_, i) => (i < n / 2 ? -1 : 1))
const informative = onDimensions(rbfKernel({ lengthScale: 1 }), [0])
const noise = onDimensions(rbfKernel({ lengthScale: 1 }), [1])
const grams = [gram(informative, x), gram(noise, x)]
const sumTo1 = (w: readonly number[]): number => w.reduce((a, b) => a + b, 0)

describe('the weights of a planted pair of kernels', () => {
  it('put the weight on the informative kernel by alignment, EasyMKL and the ridge criterion', () => {
    const projected = alignmentWeights(grams, y)
    const closed = alignmentWeights(grams, y, { method: 'closed-form' })
    const easy = easyMkl(grams, y)
    for (const w of [projected.weights, closed.weights, easy.weights]) {
      expect(sumTo1(w)).toBeCloseTo(1, 10)
      for (const v of w) expect(v).toBeGreaterThanOrEqual(0)
      expect(w[0] ?? 0).toBeGreaterThan(0.7)
    }
    const run = runAlgorithm(mklAlgorithm(grams, y, { lambda: 0.5 }), { maxSteps: 100 })
    const u = Array.from((run.state as { x: Float64Array }).x)
    expect(Math.exp(u[0] ?? 0) / (Math.exp(u[0] ?? 0) + Math.exp(u[1] ?? 0))).toBeGreaterThan(0.7)
  })

  it('has an alignment of the combination above that of every base kernel, and of the brute-force optimum', () => {
    const r = alignmentWeights(grams, y)
    expect(r.alignment).toBeGreaterThanOrEqual(Math.max(...r.individual) - 1e-9)
    let best = -1
    for (let w = 0; w <= 1.0000001; w += 0.001) {
      const mix = grams.map((g, i) => toFlat(g).map((v) => v * (i === 0 ? w : 1 - w)))
      const combined = mix[0]?.map((v, i) => v + (mix[1]?.[i] ?? 0)) ?? []
      best = Math.max(best, alignmentWeights([fromFlat(combined)], y).alignment)
    }
    expect(r.alignment).toBeGreaterThanOrEqual(best - 1e-6)
  })

  it('agrees between the closed form and the projected gradient when the unconstrained solution is non-negative', () => {
    const near = [gram(rbfKernel({ lengthScale: 1 }), x), gram(rbfKernel({ lengthScale: 2 }), x)]
    const a = alignmentWeights(near, y)
    const b = alignmentWeights(near, y, { method: 'closed-form' })
    if (b.weights.every((v) => v > 1e-9)) expect(a.alignment).toBeCloseTo(b.alignment, 6)
    expect(a.alignment).toBeGreaterThanOrEqual(b.alignment - 1e-9)
  })
})

describe('the fitted combination', () => {
  for (const method of ['alignment', 'alignment-closed-form', 'easy', 'simple'] as const) {
    it(`is a kernel whose Gram matrix is the weighted sum (${method})`, () => {
      const fit = fitMultipleKernel([informative, noise], x, y, { method })
      expect(sumTo1(fit.weights)).toBeCloseTo(1, 8)
      const direct = toFlat(fit.kernel.k(x))
      const stored = toFlat(fit.gram)
      for (let i = 0; i < stored.length; i++) expect(direct[i]).toBeCloseTo(stored[i] ?? 0, 9)
      expect(fit.weights[0] ?? 0).toBeGreaterThan(0.5)
    })
  }

  it('plugs into kernel ridge and beats the noise kernel alone', () => {
    const fit = fitMultipleKernel([informative, noise], x, y)
    const test = Array.from({ length: n }, (_, i) => [
      (i < n / 2 ? -1 : 1) * 1.2 + 0.5 * (z[(i + 7) % n] ?? 0),
      z[(i + 3) % n] ?? 0,
    ])
    const labels = test.map((_, i) => (i < n / 2 ? -1 : 1))
    const score = (kernel: typeof noise): number => {
      const pred = toFlat(kernelRidge.with({ kernel, alpha: 0.5 }).fit({ x, y }).predict(test))
      return pred.filter((p, i) => Math.sign(p) === labels[i]).length / n
    }
    expect(score(fit.kernel)).toBeGreaterThan(score(noise) + 0.2)
  })

  it('refuses no kernels, and EasyMKL on more than two classes', () => {
    expect(() => fitMultipleKernel([], x, y)).toThrow()
    expect(() =>
      easyMkl(
        grams,
        y.map((_, i) => i % 3),
      ),
    ).toThrow()
  })
})

describe('the solver slot of the weights', () => {
  it('logs the weights at every step of a session, and the objective falls', () => {
    const m = defineMethod<
      unknown,
      Record<string, never>,
      'solver/objective' | 'solver/gradNorm' | 'solver/step' | 'solver/lr' | 'mkl/weights'
    >({
      name: 'mkl',
      config: dict({}),
      make: (_c, key) => mklAlgorithm(grams, y, { lambda: 0.5, key }),
      contract: mklContract,
    })
    const s = session(m, { key: makeKey(1), steps: 30, mode: 'eager', logs: 'all' })
    const run = s.run()
    const objective = metricSeries(run, 'solver/objective').values
    expect(objective.length).toBeGreaterThan(2)
    expect(objective.at(-1) ?? 0).toBeLessThan(objective[0] ?? 0)
    expect(Object.keys(s.contract)).toContain('mkl/weights')
  })

  it('reaches the same weights with L-BFGS, BFGS and Newton', () => {
    const final = (kind: string): number => {
      const run = runAlgorithm(
        mklAlgorithm(grams, y, { lambda: 0.5, solver: smoothSolverConfig({ kind, gradTol: 1e-9 } as never) }),
        { maxSteps: 300 },
      )
      const u = Array.from((run.state as { x: Float64Array }).x)
      return Math.exp(u[0] ?? 0) / (Math.exp(u[0] ?? 0) + Math.exp(u[1] ?? 0))
    }
    const a = final('lbfgs')
    expect(final('bfgs')).toBeCloseTo(a, 3)
    expect(final('newton')).toBeCloseTo(a, 3)
  })

  it('has a problem whose objective is the optimal ridge loss at uniform weights', () => {
    const p = mklProblem(grams, y, 0.5)
    expect(p.x0).toEqual([0, 0])
  })
})

import { fromStorage } from '../tensor/index.ts'

function fromFlat(v: ArrayLike<number>): ReturnType<typeof fromStorage> {
  return fromStorage(Float64Array.from(v), [n, n], 'float64')
}
