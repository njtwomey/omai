/**
 * Tests for the outer iteration: the analytic gradient of the Laplace-approximate REML score against differences of the
 * score itself (canonical and non-canonical links, known and unknown scale), the optimum against a grid search, the
 * agreement with the performance iteration where it is exact (Gaussian), the criteria other than REML, the solver
 * slot, and the state of the algorithm.
 */

import { describe, expect, it } from 'vitest'
import { binomial, gamma, log, poisson, probit } from '../dists/families/index.ts'
import { glmSolvers, solverConfigOf } from '../optim/index.ts'
import { foldIn, key, normal, poisson as poissonDraw } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  evaluateOuter,
  fitGam,
  fitGamOuter,
  GAM_DEFAULTS,
  type GamHyper,
  gam,
  gamOuterAlgorithm,
  gamProblem,
  s,
} from './index.ts'

const n = 120
const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
const z = normal(key(21), [n]).data as Float64Array
const mean = (t: number): number => Math.exp(1 + Math.sin(2 * Math.PI * t))
const counts = x.map((r, i) => poissonDraw(foldIn(key(5), i), mean(r[0] ?? 0), []).data[0] ?? 0)
const positive = x.map((r, i) => mean(r[0] ?? 0) * Math.exp(0.25 * (z[i] ?? 0)))
const binary = x.map((r, i) => ((r[0] ?? 0) + 0.3 * (z[i] ?? 0) > 0.5 ? 1 : 0))
const gaussianY = x.map((r, i) => Math.sin(2 * Math.PI * (r[0] ?? 0)) + 0.3 * (z[i] ?? 0))

const outer = (extra: Partial<GamHyper>): GamHyper => ({
  ...GAM_DEFAULTS,
  terms: [s(0, { k: 10 })],
  iteration: 'outer',
  ...extra,
})

describe('the gradient of the Laplace criterion', () => {
  const cases: [string, GamHyper, number[]][] = [
    ['poisson-log (canonical, known scale)', outer({ family: poisson() }), counts],
    ['gamma-log (unknown scale)', outer({ family: gamma(log()) }), positive],
    ['binomial-probit (non-canonical)', outer({ family: binomial(probit()) }), binary],
    ['gaussian', outer({}), gaussianY],
  ]
  for (const [name, hyper, y] of cases) {
    for (const method of ['reml', 'ml'] as const) {
      it(`matches central differences of the score: ${name}, ${method}`, () => {
        const h = { ...hyper, method }
        const problem = gamProblem(x, y, h)
        const setup = { problem, hyper: h, key: key(0) }
        const at: number[] = [3]
        const ev = evaluateOuter(setup, at)
        const step = 1e-3
        const up = evaluateOuter(setup, [(at[0] ?? 0) + step], { warm: ev.fit, gradient: false }).value
        const down = evaluateOuter(setup, [(at[0] ?? 0) - step], { warm: ev.fit, gradient: false }).value
        const numeric = (up - down) / (2 * step)
        expect(ev.gradient?.[0]).toBeCloseTo(numeric, 5)
      })
    }
  }
})

describe('the optimum of the criterion', () => {
  const grid = Array.from({ length: 49 }, (_, i) => -10 + 0.5 * i)
  for (const [name, family, y] of [
    ['poisson', poisson(), counts],
    ['gamma', gamma(log()), positive],
    ['binomial', binomial(), binary],
  ] as const) {
    it(`is the minimiser a grid search finds: ${name}`, { timeout: 60000 }, () => {
      const hyper = outer({ family })
      const problem = gamProblem(x, y, hyper)
      const setup = { problem, hyper, key: key(0) }
      const values = grid.map((l) => evaluateOuter(setup, [l], { gradient: false }).value)
      const best = Math.min(...values)
      const atBest = grid[values.indexOf(best)] ?? 0
      const { state, converged } = fitGamOuter(problem, hyper)
      expect(converged).toBe(true)
      // The binomial score is flat towards a straight line (lambda -> infinity): equal to a few parts in 1e7.
      expect(state.criterion).toBeLessThanOrEqual(best + 1e-7 * (1 + Math.abs(best)))
      if (name !== 'binomial') expect(Math.abs((state.logLambda[0] ?? 0) - atBest)).toBeLessThan(0.6)
      expect(state.gradNorm).toBeLessThan(1e-4)
    })
  }

  it('agrees with the performance iteration for a Gaussian model, where both are exact', () => {
    const o = outer({})
    const p = { ...o, iteration: 'performance' as const }
    const a = fitGamOuter(gamProblem(x, gaussianY, o), o).state
    const b = fitGam(gamProblem(x, gaussianY, p), p).state
    expect(a.logLambda[0]).toBeCloseTo(b.logLambda[0] ?? 0, 1)
    expect(a.edf).toBeCloseTo(b.edf, 1)
  })

  it('selects by ML, GCV and UBRE too, at the minimum of their own scores', () => {
    for (const [method, family, y] of [
      ['ml', gamma(log()), positive],
      ['gcv', poisson(), counts],
      ['ubre', poisson(), counts],
    ] as const) {
      const hyper = outer({ family, method })
      const problem = gamProblem(x, y, hyper)
      const setup = { problem, hyper, key: key(0) }
      const values = [-8, -4, -2, 0, 2, 4, 6, 8].map((l) => evaluateOuter(setup, [l], { gradient: false }).value)
      const { state } = fitGamOuter(problem, hyper)
      expect(state.criterion).toBeLessThanOrEqual(Math.min(...values) + 1e-6)
    }
  })

  it('has two smooths with their own smoothing parameters, one of them heavily penalised', () => {
    const w = x.map((_, i) => [((i * 37) % n) / n])
    const two = x.map((r, i) => [r[0] ?? 0, w[i]?.[0] ?? 0])
    const hyper = outer({ family: poisson(), terms: [s(0, { k: 8 }), s(1, { k: 8 })] })
    const problem = gamProblem(two, counts, hyper)
    const { state, converged } = fitGamOuter(problem, hyper)
    expect(converged).toBe(true)
    expect(state.logLambda.length).toBe(2)
    expect(state.edfTerms[0]).toBeGreaterThan(state.edfTerms[1] ?? 0)
  })
})

describe('the slot, the state and the estimator', () => {
  it('reaches the same optimum with another inner solver', () => {
    const hyper = outer({ family: poisson() })
    const problem = gamProblem(x, counts, hyper)
    const a = fitGamOuter(problem, hyper).state
    const newton = { ...hyper, solver: solverConfigOf(glmSolvers, { kind: 'newton', gradTol: 1e-11 }) }
    const b = fitGamOuter(problem, newton).state
    expect(b.logLambda[0]).toBeCloseTo(a.logLambda[0] ?? 0, 2)
  })

  it('serialises its state and resumes to the same result', () => {
    const hyper = outer({ family: poisson() })
    const problem = gamProblem(x, counts, hyper)
    const algorithm = gamOuterAlgorithm(problem, hyper)
    const straight = runAlgorithm(algorithm, { maxSteps: 4 })
    const head = runAlgorithm(algorithm, { maxSteps: 2 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      maxSteps: 4 - head.steps,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.logLambda[0]).toBeCloseTo(straight.state.logLambda[0] ?? 0, 10)
    expect(resumed.state.beta).toEqual(straight.state.beta)
  })

  it('is selectable on the estimator and gives the posterior band and summary of the outer fit', () => {
    const fitted = gam.with({ family: poisson(), terms: [s(0, { k: 10 })], iteration: 'outer' }).fit({ x, y: counts })
    expect(fitted.smoothingParameters.length).toBe(1)
    expect(fitted.summary().smooth[0]?.pValue).toBeLessThan(1e-6)
    expect(fitted.predictionBand(x).lower.length).toBe(n)
  })

  it('refuses a fixed method and shape-constrained terms', () => {
    const fixed = outer({ method: 'fixed', sp: [1] })
    expect(() => gamOuterAlgorithm(gamProblem(x, gaussianY, fixed), fixed)).toThrow(/fixed/)
    const shaped = outer({ terms: [s(0, { basis: 'ps', k: 8, shape: 'increasing' })] })
    expect(() => fitGamOuter(gamProblem(x, gaussianY, shaped), shaped)).toThrow(/shape/)
  })
})
