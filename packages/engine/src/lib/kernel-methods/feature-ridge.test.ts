/**
 * Tests for ridge regression in explicit feature spaces: the closed form, the slot (every solver reaches the closed
 * form), the identity with exact kernel ridge when the features span the data, and the approximation bound that shrinks
 * as the number of features grows.
 */

import { describe, expect, it } from 'vitest'
import { gram, linearKernel, nystroem, randomFourierFeatures, rbfKernel } from '../kernels/index.ts'
import { flatObjective, smoothSolverConfig } from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { normals, uniformRows } from './_testdata.ts'
import {
  approximateKernelRidge,
  featureRidge,
  featureRidgeProblem,
  kernelRidge,
  ridgeGap,
  ridgeSolve,
} from './index.ts'

const x = uniformRows(60, 2, 11)
const noise = normals(60, 12)
const f = (r: readonly number[]): number => Math.sin(1.5 * (r[0] ?? 0)) + 0.5 * Math.cos(2 * (r[1] ?? 0))
const y = x.map((r, i) => f(r) + 0.05 * (noise[i] ?? 0))
const xt = uniformRows(40, 2, 13)
const yt = xt.map(f)
const rmse = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return Math.sqrt(s / a.length)
}

describe('ridgeSolve', () => {
  it('shrinks an orthonormal design by 1 / (1 + alpha)', () => {
    const w = ridgeSolve(
      [
        [1, 0],
        [0, 1],
      ],
      [2, 4],
      1,
    )
    expect(Array.from(w.weights)).toEqual([1, 2])
  })

  it('has the fitted values of the dual solve when the features are those of the linear kernel', () => {
    const alpha = 0.7
    const w = ridgeSolve(x, y, alpha).weights
    const k = toFlat(gram(linearKernel(), x))
    const exact = kernelRidge.with({ kernel: 'linear', alpha }).fit({ x, y })
    const fitted = toFlat(exact.predict(x))
    for (let i = 0; i < 60; i++) {
      const primal = (x[i]?.[0] ?? 0) * (w[0] ?? 0) + (x[i]?.[1] ?? 0) * (w[1] ?? 0)
      expect(primal).toBeCloseTo(fitted[i] ?? 0, 9)
    }
    expect(k.length).toBe(3600)
  })

  it('fits an unpenalised intercept by centring', () => {
    const z = [[0], [1], [2], [3]]
    const t = [5, 7, 9, 11]
    const s = ridgeSolve(z, t, 1e-12, true)
    expect(s.weights[0]).toBeCloseTo(2, 6)
    expect(s.intercept[0]).toBeCloseTo(5, 6)
  })

  it('rejects a bad alpha and mismatched rows', () => {
    expect(() => ridgeSolve([[1]], [1], 0)).toThrow()
    expect(() => ridgeSolve([[1], [2]], [1], 1)).toThrow()
  })
})

describe('the solver slot of the feature ridge', () => {
  const z = toFlat(nystroem(rbfKernel({ lengthScale: 1 }), x, { nComponents: 12, key: makeKey(5) }).transform(x))
  const zt = Array.from({ length: 60 }, (_, i) => z.slice(i * 12, (i + 1) * 12))
  it('is minimised by the closed form: the gradient there vanishes', () => {
    const problem = featureRidgeProblem(zt, y, 0.5)
    const w = ridgeSolve(zt, y, 0.5).weights
    const flat = flatObjective(problem.objective, problem.x0)
    const g = flat.valueAndGrad(Array.from(w)).grad
    expect(Math.max(...Array.from(g).map(Math.abs))).toBeLessThan(1e-8)
    // And it is not a stationary point elsewhere.
    expect(Math.max(...Array.from(flat.valueAndGrad(problem.x0 as number[]).grad).map(Math.abs))).toBeGreaterThan(0.1)
  })

  for (const solver of [
    { kind: 'newton' },
    { kind: 'lbfgs', ftol: 1e-15 },
    { kind: 'bfgs' },
    { kind: 'conjugateGradient' },
  ] as const) {
    it(`reaches the closed form with ${solver.kind}`, () => {
      const base = { features: 'nystroem', nComponents: 12, alpha: 0.5, fitIntercept: true } as const
      const closed = approximateKernelRidge(base).fit({ x, y })
      const other = approximateKernelRidge({
        ...base,
        solver: smoothSolverConfig({ ...solver, gradTol: 1e-10 }),
        maxIter: 2000,
      }).fit({ x, y })
      const a = toFlat(closed.predict(xt))
      const b = toFlat(other.predict(xt))
      expect(rmse(a, b)).toBeLessThan(1e-5)
    })
  }

  it('logs the objective and the gradient on the tape, and serialises', () => {
    const est = approximateKernelRidge({
      nComponents: 8,
      alpha: 0.5,
      solver: smoothSolverConfig({ kind: 'lbfgs' }),
      maxIter: 50,
    })
    const fit = est.fit({ x, y })
    expect(fit.run.events.length).toBeGreaterThan(0)
    const revived = est.revive(JSON.parse(JSON.stringify(fit.params)))
    expect(rmse(toFlat(revived.predict(xt)), toFlat(fit.predict(xt)))).toBeLessThan(1e-12)
  })
})

describe('approximation of exact kernel ridge', () => {
  const kernel = rbfKernel({ lengthScale: 1.2 })
  const alpha = 0.2
  const k = gram(kernel, x)

  it('is exact when every training row is a Nystrom landmark', () => {
    const exact = kernelRidge.with({ kernel, alpha }).fit({ x, y })
    const approx = featureRidge(nystroem(kernel, x, { indices: Array.from({ length: 60 }, (_, i) => i) }), x, y, alpha)
    expect(rmse(toFlat(approx.predict(xt)), toFlat(exact.predict(xt)))).toBeLessThan(1e-4)
  })

  it('has a fitted-value gap below |K - K~| |y| / alpha that falls with nested Nystrom landmarks', () => {
    const order = Array.from({ length: 60 }, (_, i) => (i * 37) % 60)
    const gaps = [4, 8, 16, 32].map((m) =>
      ridgeGap(k, nystroem(kernel, x, { indices: order.slice(0, m) }), x, y, alpha),
    )
    for (const g of gaps) expect(g.fittedGap).toBeLessThanOrEqual(g.bound * (1 + 1e-9))
    for (let i = 1; i < gaps.length; i++) {
      expect(gaps[i]?.spectral ?? 0).toBeLessThanOrEqual((gaps[i - 1]?.spectral ?? 0) + 1e-9)
      expect(gaps[i]?.fittedGap ?? 0).toBeLessThanOrEqual((gaps[i - 1]?.fittedGap ?? 0) + 1e-9)
    }
    expect(gaps[3]?.fittedGap ?? 1).toBeLessThan((gaps[0]?.fittedGap ?? 0) / 3)
  })

  it('has a random Fourier gap within the bound that falls with the number of features', () => {
    const mean = (d: number): number => {
      let s = 0
      for (let seed = 0; seed < 6; seed++) {
        const map = randomFourierFeatures({ inputDim: 2, nComponents: d, kernel, key: makeKey(100 + seed) })
        const g = ridgeGap(k, map, x, y, alpha)
        expect(g.fittedGap).toBeLessThanOrEqual(g.bound * (1 + 1e-9))
        s += g.fittedGap / 6
      }
      return s
    }
    expect(mean(800)).toBeLessThan(mean(25) / 2)
  })

  it('recovers a function of the kernel space with Nystrom, better with more landmarks', () => {
    const error = (m: number): number => {
      const fit = approximateKernelRidge({ features: 'nystroem', nComponents: m, kernel, alpha: 1e-2 }).fit({ x, y })
      return rmse(toFlat(fit.predict(xt)), yt)
    }
    const small = error(6)
    const large = error(40)
    expect(large).toBeLessThan(0.1)
    expect(large).toBeLessThan(small)
  })

  it('recovers the function with random Fourier and orthogonal features', () => {
    for (const features of ['fourier', 'orthogonal-fourier'] as const) {
      const fit = approximateKernelRidge({ features, nComponents: 300, kernel, alpha: 1e-2 }).fit({ x, y })
      expect(rmse(toFlat(fit.predict(xt)), yt)).toBeLessThan(0.15)
    }
  })

  it('handles several outputs', () => {
    const y2 = x.map((r, i) => [y[i] ?? 0, (r[0] ?? 0) * (r[1] ?? 0)])
    const fit = approximateKernelRidge({ nComponents: 20, kernel, alpha: 0.1 }).fit({ x, y: y2 })
    expect(fit.predict(xt).shape).toEqual([40, 2])
  })
})
