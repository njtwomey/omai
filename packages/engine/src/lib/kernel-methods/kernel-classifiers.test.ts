/**
 * Tests for the kernel classifiers: kernel logistic regression (stationarity, the solvers of the slot agreeing, the
 * linear-kernel case against the primal), the import vector machine (monotone objective, sparsity, the full-span limit),
 * the least-squares support vector machine (its optimality conditions, the closed form as a mode) and kernel ridge
 * classification (equal to kernel ridge on plus and minus one targets).
 */

import { describe, expect, it } from 'vitest'
import { flatObjective, smoothSolverConfig, smoothSolvers } from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm } from '../trace/index.ts'
import { sigmoid, softplus } from './_classifier.ts'
import { problemOf } from './_slot.ts'
import { blobs, normals, uniformRows } from './_testdata.ts'
import {
  importVectorMachine,
  kernelLogisticProblem,
  kernelLogisticRegression,
  kernelRidge,
  kernelRidgeClassifier,
  lssvm,
} from './index.ts'

const { x, y } = blobs(25, 2.5, 31)
const test = blobs(40, 2.5, 32)
const f = (v: ArrayLike<number>, r: readonly number[]): number =>
  (v[0] ?? 0) * (r[0] ?? 0) + (v[1] ?? 0) * (r[1] ?? 0) + (v[2] ?? 0)
/** The Bayes accuracy of the blobs: Phi(gap / 2) = Phi(1.25). */
const BAYES = 0.894
const accuracy = (pred: ArrayLike<number>, truth: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < truth.length; i++) s += pred[i] === truth[i] ? 1 : 0
  return s / truth.length
}
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)))
  return m
}

describe('kernel logistic regression', () => {
  const base = { gamma: 0.4, lambda: 0.5 } as const

  it('stops where the gradient of its objective vanishes, and classifies the blobs', () => {
    const fit = kernelLogisticRegression(base).fit({ x, y })
    expect(kernelLogisticRegression(base).session({ x, y }, { mode: 'eager' }).status).toBe('converged')
    const gram = toFlat(fit.decisionFunction(x)).length
    expect(gram).toBe(50)
    expect(accuracy(toFlat(fit.predict(test.x)), test.y)).toBeGreaterThan(BAYES - 0.07)
    const p = toFlat(fit.predictProba(test.x))
    for (let i = 0; i < 80; i++) expect((p[2 * i] ?? 0) + (p[2 * i + 1] ?? 0)).toBeCloseTo(1, 12)
  })

  it('has the stationarity of the problem at its fit', () => {
    const fit = kernelLogisticRegression(base).fit({ x, y })
    const k = toFlat(kernelMatrixOf(x, 0.4))
    const problem = kernelLogisticProblem(fromRows(k, 50), y, 0.5)
    const theta = [...fit.params.coef, fit.params.intercept]
    const g = flatObjective(problem.objective, problem.x0).valueAndGrad(theta).grad
    expect(Math.max(...Array.from(g).map(Math.abs))).toBeLessThan(1e-6)
  })

  for (const solver of [
    { kind: 'newton' },
    { kind: 'lbfgs', ftol: 1e-16 },
    { kind: 'bfgs' },
    { kind: 'conjugateGradient' },
  ] as const) {
    it(`reaches the IRLS optimum with ${solver.kind}`, () => {
      const irls = kernelLogisticRegression(base).fit({ x, y })
      const other = kernelLogisticRegression({
        ...base,
        solver: smoothSolverConfig({ ...solver, gradTol: 1e-9 }),
        maxIter: 3000,
      }).fit({ x, y })
      expect(maxDiff(toFlat(other.decisionFunction(test.x)), toFlat(irls.decisionFunction(test.x)))).toBeLessThan(2e-3)
    })
  }

  it('is the L2-penalised primal logistic regression for the linear kernel', () => {
    const lambda = 0.5
    const kernelFit = kernelLogisticRegression({ kernel: 'linear', lambda }).fit({ x, y })
    // The primal problem on (w, b): sum softplus(f) - t f + lambda/2 |w|^2, minimised by L-BFGS.
    const primal = problemOf(
      {
        value: (v) =>
          x.reduce((s, r, i) => s + softplus(f(v, r)) - (y[i] ?? 0) * f(v, r), 0) +
          0.5 * lambda * ((v[0] ?? 0) ** 2 + (v[1] ?? 0) ** 2),
        gradient: (v) => {
          const g = [lambda * (v[0] ?? 0), lambda * (v[1] ?? 0), 0]
          x.forEach((r, i) => {
            const e = sigmoid(f(v, r)) - (y[i] ?? 0)
            g[0] = (g[0] ?? 0) + e * (r[0] ?? 0)
            g[1] = (g[1] ?? 0) + e * (r[1] ?? 0)
            g[2] = (g[2] ?? 0) + e
          })
          return g
        },
      },
      [0, 0, 0],
    )
    const run = runAlgorithm(
      smoothSolvers.make(primal, smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 }), makeKey(0)),
      { maxSteps: 500 },
    )
    const w = Array.from((run.state as { x: Float64Array }).x)
    const expected = test.x.map((r) => f(w, r))
    expect(maxDiff(toFlat(kernelFit.decisionFunction(test.x)), expected)).toBeLessThan(1e-4)
  })

  it('refuses more than two classes and revives from its parameters', () => {
    expect(() => kernelLogisticRegression().fit({ x: [[0], [1], [2]], y: [0, 1, 2] })).toThrow()
    const est = kernelLogisticRegression(base)
    const fit = est.fit({ x, y })
    const back = est.revive(JSON.parse(JSON.stringify(fit.params)))
    expect(maxDiff(toFlat(back.decisionFunction(test.x)), toFlat(fit.decisionFunction(test.x)))).toBeLessThan(1e-12)
  })
})

describe('the import vector machine', () => {
  it('lowers the objective at every step and keeps at most nImport vectors', () => {
    const fit = importVectorMachine({ gamma: 0.4, lambda: 0.5, nImport: 8 }).fit({ x, y })
    expect(fit.params.centres.length).toBeLessThanOrEqual(8)
    const values = metricSeries(fit.run, 'solver/objective').values
    for (let i = 1; i < values.length; i++) expect(values[i] ?? 0).toBeLessThanOrEqual((values[i - 1] ?? 0) + 1e-9)
  })

  it('is nearly as accurate as the full model with a few import vectors', () => {
    const sparse = importVectorMachine({ gamma: 0.4, lambda: 0.5, nImport: 6 }).fit({ x, y })
    const full = kernelLogisticRegression({ gamma: 0.4, lambda: 0.5 }).fit({ x, y })
    const a = accuracy(toFlat(sparse.predict(test.x)), test.y)
    expect(sparse.params.centres.length).toBeLessThanOrEqual(6)
    expect(a).toBeGreaterThan(accuracy(toFlat(full.predict(test.x)), test.y) - 0.06)
  })

  it('has the objective of kernel logistic regression when every row is an import vector', () => {
    const xs = uniformRows(10, 2, 41)
    const ys = xs.map((r) => ((r[0] ?? 0) + 0.3 * (r[1] ?? 0) > 0 ? 1 : 0))
    const ivm = importVectorMachine({ gamma: 0.8, lambda: 1, nImport: 10, tol: -1 }).fit({ x: xs, y: ys })
    const klr = kernelLogisticRegression({ gamma: 0.8, lambda: 1 }).fit({ x: xs, y: ys })
    expect(ivm.params.centres.length).toBe(10)
    expect(maxDiff(toFlat(ivm.decisionFunction(xs)), toFlat(klr.decisionFunction(xs)))).toBeLessThan(1e-3)
  })
})

describe('the least-squares classifiers', () => {
  it('satisfies the optimality conditions of the LS-SVM: alpha = C (y - f) and sum alpha = 0', () => {
    const c = 5
    const fit = lssvm({ gamma: 0.4, regularisation: c }).fit({ x, y })
    const f = toFlat(fit.decisionFunction(x))
    const alpha = fit.params.dualCoef.map((r) => r[0] ?? 0)
    let total = 0
    for (let i = 0; i < 50; i++) {
      expect(alpha[i] ?? 0).toBeCloseTo(c * ((y[i] === 1 ? 1 : -1) - (f[i] ?? 0)), 8)
      total += alpha[i] ?? 0
    }
    expect(total).toBeCloseTo(0, 8)
    expect(accuracy(toFlat(fit.predict(test.x)), test.y)).toBeGreaterThan(BAYES - 0.07)
  })

  for (const solver of [{ kind: 'newton' }, { kind: 'lbfgs', ftol: 1e-16 }, { kind: 'bfgs' }] as const) {
    it(`reaches the closed form with ${solver.kind}`, () => {
      const closed = lssvm({ gamma: 0.4, regularisation: 2 }).fit({ x, y })
      const other = lssvm({
        gamma: 0.4,
        regularisation: 2,
        solver: smoothSolverConfig({ ...solver, gradTol: 1e-9 }),
        maxIter: 3000,
      }).fit({ x, y })
      expect(maxDiff(toFlat(other.decisionFunction(test.x)), toFlat(closed.decisionFunction(test.x)))).toBeLessThan(
        2e-3,
      )
    })
  }

  it('is kernel ridge on plus and minus one targets for the kernel ridge classifier', () => {
    const alpha = 0.7
    const clf = kernelRidgeClassifier({ gamma: 0.4, alpha }).fit({ x, y })
    const reg = kernelRidge.with({ kernel: 'rbf', gamma: 0.4, alpha }).fit({ x, y: y.map((v) => (v === 1 ? 1 : -1)) })
    expect(maxDiff(toFlat(clf.decisionFunction(test.x)), toFlat(reg.predict(test.x)))).toBeLessThan(1e-9)
  })

  it('classifies several classes one-versus-rest', () => {
    const z = normals(120, 51)
    const centres = [
      [0, 0],
      [4, 0],
      [0, 4],
    ]
    const xs = Array.from({ length: 60 }, (_, i) => [
      (centres[i % 3]?.[0] ?? 0) + 0.6 * (z[2 * i] ?? 0),
      (centres[i % 3]?.[1] ?? 0) + 0.6 * (z[2 * i + 1] ?? 0),
    ])
    const ys = Array.from({ length: 60 }, (_, i) => i % 3)
    const fit = kernelRidgeClassifier({ gamma: 0.2, alpha: 0.1 }).fit({ x: xs, y: ys })
    expect(accuracy(toFlat(fit.predict(xs)), ys)).toBeGreaterThan(0.95)
    expect(fit.decisionFunction(xs).shape).toEqual([60, 3])
  })
})

function kernelMatrixOf(rows: number[][], gamma: number): ReturnType<typeof gramOf> {
  return gramOf(rows, gamma)
}

import { pairwiseKernels } from '../kernels/index.ts'
import { fromStorage } from '../tensor/index.ts'

function gramOf(rows: number[][], gamma: number) {
  return pairwiseKernels(rows, undefined, 'rbf', { gamma })
}
function fromRows(k: ArrayLike<number>, n: number) {
  return fromStorage(Float64Array.from(k), [n, n], 'float64')
}
