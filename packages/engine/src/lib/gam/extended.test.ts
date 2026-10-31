/**
 * Tests for the extended-family machinery: the analytic gradient of the Laplace criterion against differences of the
 * criterion, the solver slot of the inner fit, and the fitted models on planted data (beta regression first).
 */

import { describe, expect, it } from 'vitest'
import { beta as betaDist } from '../dists/continuous/index.ts'
import { families } from '../dists/index.ts'
import { smoothSolvers, solverConfigOf } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import {
  assemblePenalty,
  evaluateExtended,
  extendedModel,
  fitExtended,
  innerExtendedFit,
  predictExtended,
  s,
} from './index.ts'

const logistic = (v: number): number => 1 / (1 + Math.exp(-v))
const truthEta = (t: number): number => -0.2 + 1.3 * Math.sin(2 * Math.PI * t)

function betaData(seed: number, n: number, phi: number): { x: number[][]; y: number[] } {
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
  const mu = x.map((r) => logistic(truthEta(r[0] ?? 0)))
  const draw = betaDist(tensor(mu.map((m) => m * phi)), tensor(mu.map((m) => (1 - m) * phi))).sample(key(seed))
  return { x, y: Array.from(toFlat(draw), (v) => Math.min(Math.max(v, 1e-6), 1 - 1e-6)) }
}

describe('the criterion of the extended outer iteration', () => {
  const { x, y } = betaData(1, 120, 15)
  const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 8 })]])

  it('has an analytic gradient in log lambda and in the aux parameter equal to differences of the criterion', () => {
    const at = [3.2, Math.log(12)] as const
    const ev = evaluateExtended(model, [at[0]], [at[1]])
    const g = ev.gradient as Float64Array
    const h = 1e-4
    const value = (l: number, a: number): number => evaluateExtended(model, [l], [a], { gradient: false }).value
    expect(g[0] ?? Number.NaN).toBeCloseTo((value(at[0] + h, at[1]) - value(at[0] - h, at[1])) / (2 * h), 5)
    expect(g[1] ?? Number.NaN).toBeCloseTo((value(at[0], at[1] + h) - value(at[0], at[1] - h)) / (2 * h), 4)
  })

  it('reaches the same penalised optimum with Newton, BFGS and L-BFGS from the slot as with the safeguarded Newton', () => {
    const S = assemblePenalty(model.structure, [3])
    const reference = innerExtendedFit(model, S, [Math.log(15)])
    expect(reference.gradNorm).toBeLessThan(1e-8)
    for (const solver of [
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'bfgs', gradTol: 1e-9 },
      { kind: 'lbfgs', gradTol: 1e-9, ftol: 1e-15 },
    ]) {
      const fit = innerExtendedFit(model, S, [Math.log(15)], {
        solver: solverConfigOf(smoothSolvers, solver),
        maxIter: 600,
      })
      fit.beta.forEach((b, j) => {
        expect(b, `${solver.kind} ${j}`).toBeCloseTo(reference.beta[j] ?? 0, 4)
      })
    }
  })
})

describe('beta regression on planted data', () => {
  it('recovers the mean curve and the precision', () => {
    const phi = 20
    const { x, y } = betaData(11, 300, phi)
    const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 10 })]])
    const fit = fitExtended(model)
    expect(fit.converged).toBe(true)
    expect(fit.state.gradNorm).toBeLessThan(1e-3)
    const grid = Array.from({ length: 50 }, (_, i) => [(i + 0.5) / 50])
    const pred = predictExtended(model, fit, grid)
    let sq = 0
    grid.forEach((r, i) => {
      sq += ((pred.eta[0]?.[i] ?? 0) - truthEta(r[0] ?? 0)) ** 2
    })
    expect(Math.sqrt(sq / grid.length)).toBeLessThan(0.15)
    expect(Math.exp(fit.state.aux[0] ?? 0) / phi).toBeGreaterThan(0.75)
    expect(Math.exp(fit.state.aux[0] ?? 0) / phi).toBeLessThan(1.3)
  }, 120000)
})
