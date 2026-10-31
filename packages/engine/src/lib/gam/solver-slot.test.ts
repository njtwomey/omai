/**
 * The inner solver of the penalised IRLS is a slot: at fixed smoothing parameters every mode of `glmSolvers` reaches the
 * penalised-likelihood optimum the reweighted least-squares step does, with the smoothing parameters still chosen by the
 * outer criterion; and the state of such a fit serialises and resumes.
 */

import { describe, expect, it } from 'vitest'
import { families } from '../dists/index.ts'
import { glmSolvers, solverConfigOf } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { fitGam, GAM_DEFAULTS, gamAlgorithm, gamProblem, s } from './index.ts'

const n = 60
const x = Array.from({ length: n }, (_, i) => [i / (n - 1)])
const counts = x.map((r, i) => Math.round(Math.exp(0.5 + Math.sin(5 * (r[0] ?? 0))) + 0.6 * Math.sin(i * 12.9898)))
const base = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })], family: families.poisson(), maxIter: 60 }

describe('GAM with a solver of the slot', () => {
  it('reaches the IRLS optimum at fixed smoothing parameters with Newton, BFGS, L-BFGS and Adam', () => {
    const fixed = { ...base, method: 'fixed' as const, sp: [0.05] }
    const problem = gamProblem(x, counts, fixed)
    const reference = fitGam(problem, fixed).state
    for (const solver of [
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'bfgs', gradTol: 1e-9 },
      { kind: 'lbfgs', gradTol: 1e-9, ftol: 1e-15 },
      { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 0.05, gradTol: 1e-7 },
    ]) {
      const config = solverConfigOf(glmSolvers, solver)
      const state = fitGam(problem, { ...fixed, solver: config }).state
      state.beta.forEach((b, j) => {
        expect(b, `${solver.kind} ${j}`).toBeCloseTo(reference.beta[j] ?? 0, 3)
      })
      expect(state.penalisedDeviance, solver.kind).toBeCloseTo(reference.penalisedDeviance, 6)
    }
  })

  it('keeps choosing the smoothing parameters by REML around the inner solver', () => {
    const problem = gamProblem(x, counts, base)
    const reference = fitGam(problem, base).state
    const state = fitGam(problem, {
      ...base,
      solver: solverConfigOf(glmSolvers, { kind: 'newton', gradTol: 1e-10 }),
    }).state
    expect(state.logLambda[0]).toBeCloseTo(reference.logLambda[0] ?? 0, 2)
    expect(state.edf).toBeCloseTo(reference.edf, 2)
  })

  it('serialises the state of a fit with another inner solver and resumes', () => {
    const hyper = { ...base, solver: solverConfigOf(glmSolvers, { kind: 'lbfgs', gradTol: 1e-9 }) }
    const problem = gamProblem(x, counts, hyper)
    const algorithm = gamAlgorithm(problem, hyper, key(0))
    const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: 6 })
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 3 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(0),
      maxSteps: 6 - head.steps,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.beta).toEqual(straight.state.beta)
  })
})
