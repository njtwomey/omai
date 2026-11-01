/**
 * The solver of the hyperparameter fit is a slot: L-BFGS-B is the default and keeps today's fit, and the other solvers
 * of `boundedSolvers` (SLSQP and the generic smooth solvers on a reparametrised box) reach the same marginal likelihood.
 */

import { describe, expect, it } from 'vitest'
import { constantKernel, productKernel, rbfKernel, sumKernel, whiteNoiseKernel } from '../kernels/index.ts'
import { boundedSolvers, solverConfigOf } from '../optim/index.ts'
import { key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { gpClassifier } from './classification.ts'
import { gpRegression } from './regression.ts'
import { sparseGp } from './sparse.ts'

const x = Array.from({ length: 18 }, (_, i) => [i / 3])
const data = { x, y: x.map(([v], i) => Math.sin(v as number) + 0.08 * Math.sin(37 * i)) }
const kernel = sumKernel(
  productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 1 })),
  whiteNoiseKernel({ noiseLevel: 0.05 }),
)
const slot = (kind: string): { solver: SpaceValue } => ({ solver: solverConfigOf(boundedSolvers, { kind }) })
const fit = (hyper: Parameters<typeof gpRegression.with>[0]): { lml: number; theta: number[] } => {
  const f = gpRegression.with({ kernel, ...hyper }).fit(data, { explain: false, key: key(0) })
  return { lml: f.params.logMarginalLikelihood, theta: f.params.theta }
}

describe('the solver slot of the Gaussian-process hyperparameter fit', () => {
  const reference = fit({})

  it('names L-BFGS-B explicitly and reproduces the default fit', () => {
    const named = fit(slot('lbfgsb'))
    expect(named.lml).toBeCloseTo(reference.lml, 9)
    named.theta.forEach((t, i) => {
      expect(t).toBeCloseTo(reference.theta[i] ?? 0, 8)
    })
  })

  it('reaches the same marginal likelihood with SLSQP, BFGS, L-BFGS and conjugate gradient', () => {
    for (const kind of ['slsqp', 'bfgs', 'lbfgs', 'conjugateGradient']) {
      const other = fit({ ...slot(kind), gradTol: 1e-6 })
      expect(other.lml, kind).toBeCloseTo(reference.lml, 2)
    }
  }, 60_000)

  it('keeps the hyperparameters inside the bounds with a generic solver', () => {
    const other = fit({ ...slot('bfgs'), bounds: [1e-2, 1e1] })
    for (const t of other.theta) {
      expect(t).toBeGreaterThanOrEqual(Math.log(1e-2) - 1e-9)
      expect(t).toBeLessThanOrEqual(Math.log(1e1) + 1e-9)
    }
  }, 60_000)

  it('serves the classifier and the sparse regressor too', () => {
    const cx = [[-2.1], [-1.6], [-1.1], [-0.6], [-0.2], [0.3], [0.7], [1.2], [1.7], [2.2]]
    const cy = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    const c = (hyper: Parameters<typeof gpClassifier.with>[0]): number =>
      gpClassifier.with(hyper).fit({ x: cx, y: cy }, { explain: false, key: key(0) }).params.logMarginalLikelihood
    expect(c(slot('bfgs'))).toBeCloseTo(c({}), 2)
    const sx = Array.from({ length: 36 }, (_, i) => [i / 6])
    const sy = sx.map(([v], i) => Math.sin(v as number) + 0.1 * Math.sin(23 * i))
    const sp = (hyper: Parameters<typeof sparseGp.with>[0]): number =>
      sparseGp.with({ inducing: 8, ...hyper }).fit({ x: sx, y: sy }, { explain: false, key: key(0) }).params.bound
    expect(sp(slot('lbfgs'))).toBeCloseTo(sp({}), 1)
  }, 60_000)
})
