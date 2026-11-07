/**
 * Tests for kernel selection: the criteria against their definitions (the Gaussian-process evidence, refitting for
 * leave-one-out, the explicit smoother for GCV), their automatic gradients against the analytic one and finite
 * differences, and the tuner (a planted noise level recovered, solvers agreeing, gradient search beating a grid).
 */

import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { logMarginalLikelihood } from '../gp/index.ts'
import { fixHyperparameters, gram, rbfKernel, scaleKernel, theta, withTheta } from '../kernels/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { item, tensor, toFlat } from '../tensor/index.ts'
import { normals, uniformRows } from './_testdata.ts'
import {
  criterionValue,
  gcvScore,
  gridKernelRidge,
  hatDiagonal,
  kernelCriterionProblem,
  kernelRidge,
  logEvidence,
  looScore,
  tuneKernelRidge,
} from './index.ts'

const x = uniformRows(40, 1, 21, 0, 6)
const noise = normals(40, 22)
const sigma = 0.2
const y = x.map((r, i) => Math.sin(2 * (r[0] ?? 0)) + sigma * (noise[i] ?? 0))

describe('the selection criteria', () => {
  const kernel = rbfKernel({ lengthScale: 0.8 })
  const k = gram(kernel, x)

  it('has an evidence equal to the Gaussian-process log marginal likelihood', () => {
    const gp = logMarginalLikelihood(kernel, x, y, { alpha: 0.3 })
    expect(item(logEvidence(k, y, 0.3))).toBeCloseTo(gp.value, 9)
  })

  it('has an automatic gradient of the evidence equal to the analytic one', () => {
    const problem = kernelCriterionProblem(kernel, x, y, { alpha: 0.3, tuneAlpha: false })
    const g = toFlat(grad(problem.objective as (t: never) => never)(tensor(problem.x0 as number[]) as never))
    const gp = logMarginalLikelihood(kernel, x, y, { alpha: 0.3 })
    expect(g[0]).toBeCloseTo(-(gp.gradient[0] ?? 0), 7)
  })

  it('has a leave-one-out score equal to refitting n times', () => {
    const alpha = 0.15
    let s = 0
    for (let i = 0; i < 40; i++) {
      const fit = kernelRidge
        .with({ kernel, alpha })
        .fit({ x: x.filter((_, j) => j !== i), y: y.filter((_, j) => j !== i) })
      s += ((y[i] ?? 0) - (toFlat(fit.predict([x[i] as number[]]))[0] ?? 0)) ** 2
    }
    expect(item(looScore(k, y, alpha))).toBeCloseTo(s / 40, 8)
  })

  it('has a generalised cross-validation score from the explicit smoother', () => {
    const alpha = 0.15
    const fitted = toFlat(kernelRidge.with({ kernel, alpha }).fit({ x, y }).predict(x))
    const h = toFlat(hatDiagonal(k, alpha))
    let rss = 0
    for (let i = 0; i < 40; i++) rss += ((y[i] ?? 0) - (fitted[i] ?? 0)) ** 2
    const trace = h.reduce((a, b) => a + b, 0)
    expect(item(gcvScore(k, y, alpha))).toBeCloseTo((40 * rss) / (40 - trace) ** 2, 8)
  })

  it('has gradients of each criterion in log alpha matching finite differences', () => {
    for (const criterion of ['evidence', 'loo', 'gcv'] as const) {
      const f = (la: number): number => item(criterionValue(criterion, k, y, Math.exp(la)))
      const problem = kernelCriterionProblem(fixHyperparameters(kernel, ['lengthScale']), x, y, {
        criterion,
        alpha: 0.4,
      })
      const g = toFlat(grad(problem.objective as (t: never) => never)(tensor(problem.x0 as number[]) as never))[0] ?? 0
      const h = 1e-5
      const la = Math.log(0.4)
      expect(g).toBeCloseTo((f(la + h) - f(la - h)) / (2 * h), 4)
    }
  })

  it('rejects a bad alpha and shapes', () => {
    expect(() => kernelCriterionProblem(kernel, x, y, { alpha: 0 })).toThrow()
    expect(() => looScore(k, y.slice(1), 1)).toThrow()
  })
})

describe('the tuners', () => {
  const base = scaleKernel(rbfKernel({ lengthScale: 2 }), 1)

  it('recovers the planted noise level and a sensible length scale by the evidence', () => {
    const tuned = tuneKernelRidge(fixHyperparameters(base, ['outputScale']), x, y, { alpha: 1 })
    expect(tuned.converged).toBe(true)
    // The noise variance is sigma^2 = 0.04 (the prior scale is fixed at 1).
    expect(tuned.alpha).toBeGreaterThan(0.015)
    expect(tuned.alpha).toBeLessThan(0.12)
    // The signal sin(2x) has period pi: the length scale is of the order of a quarter of it.
    const length = Math.exp(tuned.theta[0] ?? 0)
    expect(length).toBeGreaterThan(0.3)
    expect(length).toBeLessThan(1.6)
  })

  it('improves on its start, and polishes the best point of a coarse grid', () => {
    for (const criterion of ['evidence', 'loo'] as const) {
      const start = item(criterionValue(criterion, gram(base, x), y, 1))
      expect(tuneKernelRidge(base, x, y, { criterion, alpha: 1 }).value).toBeLessThan(start)
      const grid = gridKernelRidge(
        base,
        x,
        y,
        {
          theta: [
            [Math.log(0.5), Math.log(1), Math.log(2)],
            [-1, 0, 1],
          ],
          alpha: [0.01, 0.1, 1],
        },
        criterion,
      )
      expect(grid.table.length).toBe(27)
      const polished = tuneKernelRidge(withTheta(base, grid.best.theta), x, y, { criterion, alpha: grid.best.alpha })
      expect(polished.value).toBeLessThanOrEqual(grid.best.value + 1e-9)
    }
  })

  it('reaches the same optimum with different solvers of the slot', () => {
    const fixed = fixHyperparameters(base, ['outputScale'])
    const values = [{ kind: 'lbfgs', ftol: 1e-15 }, { kind: 'bfgs' }, { kind: 'newton' }].map((s) =>
      tuneKernelRidge(fixed, x, y, {
        alpha: 0.5,
        solver: smoothSolverConfig({ ...s, gradTol: 1e-8 } as never),
        // The solvers get within 1e-8 of the optimum in at most 20 steps; gradTol 1e-8 is the noise floor of the
        // objective (about sqrt(eps)), so a solver whose rounding falls the wrong way would otherwise walk to the cap.
        maxSteps: 60,
      }),
    )
    for (const t of values) expect(t.value).toBeCloseTo(values[0]?.value ?? 0, 4)
    expect(theta(values[1]?.kernel ?? fixed).shape).toEqual([1])
  })

  it('tunes the three criteria to finite optima', () => {
    for (const criterion of ['evidence', 'loo', 'gcv'] as const) {
      const t = tuneKernelRidge(fixHyperparameters(base, ['outputScale']), x, y, { criterion, alpha: 0.5 })
      expect(Number.isFinite(t.value)).toBe(true)
      expect(t.alpha).toBeGreaterThan(0)
    }
  })

  it('checks the grid against the kernel', () => {
    expect(() => gridKernelRidge(base, x, y, { theta: [[0]], alpha: [1] })).toThrow()
  })
})
