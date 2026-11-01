/**
 * Tests for generalised additive models: the penalised solution against an independent dense solve, its limits in the
 * smoothing parameter, effective degrees of freedom, the criteria and their minimiser, recovery of a known truth,
 * backfitting against the joint solve, the term types, and the session.
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { poisson } from '../dists/families/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { runAlgorithm, session } from '../trace/index.ts'
import {
  assemblePenalty,
  backfittingAlgorithm,
  buildDesign,
  evaluateDesign,
  factor,
  fitGam,
  GAM_DEFAULTS,
  gam,
  gamAlgorithm,
  gamClassifier,
  gamProblem,
  gamSolution,
  gcvScore,
  gram,
  influenceDiagonal,
  linear,
  penaltyStructure,
  remlScore,
  s,
  selectLogLambda,
  solvePenalised,
  te,
  termEdf,
  ti,
  ubreScore,
} from './index.ts'

/** A deterministic pseudo-noise in [-1, 1] (no random source in tests either). */
const wobble = (i: number): number =>
  Math.sin(12.9898 * i + 78.233) * 0.999 - Math.floor(Math.sin(12.9898 * i + 78.233))

const grid = (n: number): number[][] => Array.from({ length: n }, (_, i) => [i / (n - 1)])
const truth = (t: number): number => Math.sin(2 * Math.PI * t)
const noisy = (n: number, sd: number): { x: number[][]; y: number[] } => {
  const x = grid(n)
  return { x, y: x.map((r, i) => truth(r[0] ?? 0) + sd * 2 * (wobble(i) - 0.5)) }
}

const dense = (A: Float64Array, S: Float64Array, b: Float64Array, p: number): number[] => {
  const M = Float64Array.from(A, (v, i) => v + (S[i] ?? 0))
  return toFlat(solve(fromStorage(M, [p, p], 'float64'), fromStorage(b, [p], 'float64')) as never)
}

describe('penalised least squares', () => {
  const { x, y } = noisy(60, 0.2)
  const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })], method: 'fixed' as const }

  it("matches an independent dense solve of (X'X + lambda S) beta = X'y", () => {
    for (const lambda of [1e-3, 0.3, 40]) {
      const problem = gamProblem(x, y, { ...hyper, sp: [lambda] })
      const { design, structure } = problem
      const g = gram(design.X, 60, design.p, new Float64Array(60).fill(1), Float64Array.from(y))
      const expected = dense(g.xtwx, assemblePenalty(structure, [Math.log(lambda)]), g.xtwz, design.p)
      const { state } = fitGam(problem, { ...hyper, sp: [lambda] })
      state.beta.forEach((b, i) => {
        expect(b).toBeCloseTo(expected[i] ?? 0, 9)
      })
    }
  })

  it('approaches least squares as lambda goes to zero and the best straight line as it goes to infinity', () => {
    const problem = (lambda: number) => gamProblem(x, y, { ...hyper, sp: [lambda] })
    const fit = (lambda: number) => fitGam(problem(lambda), { ...hyper, sp: [lambda] }).state
    const p0 = problem(1)
    const g = gram(p0.design.X, 60, p0.design.p, new Float64Array(60).fill(1), Float64Array.from(y))
    const ols = dense(g.xtwx, new Float64Array(p0.design.p ** 2), g.xtwz, p0.design.p)
    fit(1e-12).beta.forEach((b, i) => {
      expect(b).toBeCloseTo(ols[i] ?? 0, 5)
    })
    // lambda -> infinity: the null space of the cubic regression spline penalty is the straight lines.
    const stiff = fit(1e8)
    const xs = x.map((r) => r[0] ?? 0)
    const xm = xs.reduce((a, b) => a + b, 0) / 60
    const ym = y.reduce((a, b) => a + b, 0) / 60
    const slope =
      xs.reduce((a, v, i) => a + (v - xm) * ((y[i] ?? 0) - ym), 0) / xs.reduce((a, v) => a + (v - xm) ** 2, 0)
    stiff.eta.forEach((e, i) => {
      expect(e).toBeCloseTo(ym + slope * ((xs[i] ?? 0) - xm), 4)
    })
    expect(stiff.edf).toBeCloseTo(2, 4)
  })

  it('has term degrees of freedom between the null space and the basis size, falling as lambda grows', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -8, max: 8, noNaN: true }),
        fc.double({ min: 0.1, max: 4, noNaN: true }),
        (logLam, step) => {
          const edfAt = (l: number): number => {
            const problem = gamProblem(x, y, { ...hyper, sp: [Math.exp(l)] })
            const { state } = fitGam(problem, { ...hyper, sp: [Math.exp(l)] })
            return state.edfTerms[0] ?? 0
          }
          const a = edfAt(logLam)
          const b = edfAt(logLam + step)
          // The centred cubic regression spline has 7 columns and a null space of dimension 1 (the line, less the constant).
          expect(a).toBeGreaterThanOrEqual(1 - 1e-8)
          expect(a).toBeLessThanOrEqual(7 + 1e-8)
          expect(b).toBeLessThanOrEqual(a + 1e-9)
        },
      ),
      { numRuns: 25 },
    )
  })
})

describe('smoothing parameter selection', () => {
  const { x, y } = noisy(80, 0.3)
  const problem = gamProblem(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 10 })] })
  const { design, structure } = problem
  const g = gram(design.X, 80, design.p, new Float64Array(80).fill(1), Float64Array.from(y))
  const lambdas = Array.from({ length: 81 }, (_, i) => -14 + i * 0.25)
  const solved = (l: number) => solvePenalised(g, assemblePenalty(structure, [l]))

  it('finds the minimiser of the REML, GCV and UBRE scores that a grid search finds', () => {
    const scale = 0.09
    const scores = {
      reml: (l: number) => remlScore(solved(l), { n: 80, scale: undefined, structure, logLambda: [l] }),
      gcv: (l: number) => gcvScore(solved(l), 80),
      ubre: (l: number) => ubreScore(solved(l), 80, scale),
    }
    for (const method of ['reml', 'gcv', 'ubre'] as const) {
      const values = lambdas.map(scores[method])
      const best = Math.min(...values)
      const at = lambdas[values.indexOf(best)] ?? 0
      const chosen = selectLogLambda(g, structure, [0], method, method === 'ubre' ? { scale } : {})
      expect(chosen.value).toBeLessThanOrEqual(best + 1e-6)
      expect(at).toBeGreaterThan(lambdas[0] ?? 0)
      expect(at).toBeLessThan(lambdas.at(-1) ?? 0)
      expect(Math.abs((chosen.logLambda[0] ?? 0) - at)).toBeLessThanOrEqual(0.3)
    }
  })

  it('has a GCV score equal to n rss / (n - edf)^2 and a UBRE score equal to Mallows Cp', () => {
    const sol = solved(1)
    expect(gcvScore(sol, 80)).toBeCloseTo((80 * sol.rss) / (80 - sol.edfTotal) ** 2, 12)
    expect(ubreScore(sol, 80, 0.5)).toBeCloseTo(sol.rss / 80 - 0.5 + (2 * 0.5 * sol.edfTotal) / 80, 12)
  })

  it('smooths better than an unpenalised fit of the same basis: closer to the truth', () => {
    const big = noisy(150, 0.3)
    const err = (fitted: { predict: (x: number[][]) => unknown }) => {
      const p = toFlat(fitted.predict(big.x) as never)
      return Math.sqrt(p.reduce((a, v, i) => a + (v - truth(big.x[i]?.[0] ?? 0)) ** 2, 0) / p.length)
    }
    const smooth = gam.with({ terms: [s(0, { k: 20 })] }).fit(big)
    const wiggly = gam.with({ terms: [s(0, { k: 20 })], method: 'fixed', sp: [1e-10] }).fit(big)
    expect(err(smooth)).toBeLessThan(0.1)
    expect(err(smooth)).toBeLessThan(err(wiggly))
    expect(smooth.edf).toBeLessThan(wiggly.edf)
  })
})

describe('families and terms', () => {
  it('fits a Poisson additive model that recovers the mean', () => {
    const n = 200
    const x = grid(n)
    const mu = x.map((r) => Math.exp(1 + Math.sin(2 * Math.PI * (r[0] ?? 0))))
    // Expected counts rounded to integers keep the data deterministic and the truth known.
    const y = mu.map((m) => Math.round(m))
    const fitted = gam.with({ family: poisson(), terms: [s(0, { k: 12 })] }).fit({ x, y })
    const p = toFlat(fitted.predict(x))
    const rmse = Math.sqrt(p.reduce((a, v, i) => a + (v - (mu[i] ?? 0)) ** 2, 0) / n)
    expect(rmse).toBeLessThan(0.25)
    expect(fitted.scale).toBe(1)
    expect(fitted.edf).toBeGreaterThan(4)
  })

  it('classifies with the logit of a smooth effect', () => {
    const n = 120
    const x = grid(n)
    const y = x.map((r, i) => ((r[0] ?? 0) + 0.25 * (wobble(i) - 0.5) > 0.5 ? 3 : 7))
    const fitted = gamClassifier.with({ terms: [s(0, { k: 6 })] }).fit({ x, y })
    expect(fitted.classes).toEqual([3, 7])
    expect(fitted.score({ x, y })).toBeGreaterThan(0.9)
    const proba = toFlat(fitted.predictProba?.(x) as never)
    expect((proba[0] ?? 0) + (proba[1] ?? 0)).toBeCloseTo(1, 12)
  })

  it('has partial effects that add up to the linear predictor', () => {
    const n = 90
    const x = Array.from({ length: n }, (_, i) => [i / (n - 1), wobble(i), i % 3])
    const y = x.map((r) => Math.sin(5 * (r[0] ?? 0)) + 0.8 * (r[1] ?? 0) + (r[2] ?? 0))
    const fitted = gam.with({ terms: [s(0, { k: 7 }), linear(1), factor(2)] }).fit({ x, y })
    const terms = toFlat(fitted.predictTerms(x))
    const eta = toFlat(fitted.linearPredictor(x))
    const intercept = fitted.coef[0] ?? 0
    for (let i = 0; i < n; i++)
      expect(intercept + (terms[3 * i] ?? 0) + (terms[3 * i + 1] ?? 0) + (terms[3 * i + 2] ?? 0)).toBeCloseTo(
        eta[i] ?? 0,
        10,
      )
    expect(Object.keys(fitted.edfTerms)).toEqual(['s(x0)', 'x1', 'factor(x2)'])
    expect(fitted.edfTerms['x1']).toBeCloseTo(1, 8)
    expect(fitted.score({ x, y })).toBeGreaterThan(0.99)
  })

  it('fits a tensor-product surface and its interaction', () => {
    const n = 100
    const x = Array.from({ length: n }, (_, i) => [wobble(i), wobble(i + 500)])
    const y = x.map((r) => (r[0] ?? 0) * (r[1] ?? 0) + Math.sin(4 * (r[0] ?? 0)) * 0.3)
    const surface = gam.with({ terms: [te([0, 1], { k: 5 })] }).fit({ x, y })
    expect(surface.score({ x, y })).toBeGreaterThan(0.95)
    const separate = gam.with({ terms: [s(0, { k: 6 }), s(1, { k: 6 }), ti([0, 1], { k: 4 })] }).fit({ x, y })
    expect(separate.score({ x, y })).toBeGreaterThan(0.95)
    expect(Object.keys(separate.edfTerms)).toEqual(['s(x0)', 's(x1)', 'ti(x0,x1)'])
  }, 60_000)

  it('treats a random-effect factor as a shrunken intercept per level', () => {
    const levels = 6
    const x = Array.from({ length: 60 }, (_, i) => [i % levels])
    const y = x.map((r, i) => 2 * (r[0] ?? 0) + 0.2 * wobble(i))
    const free = gam.with({ terms: [factor(0)] }).fit({ x, y })
    const shrunk = gam.with({ terms: [factor(0, { randomEffect: true })] }).fit({ x, y })
    expect(free.edf).toBeCloseTo(levels, 8)
    expect(shrunk.edf).toBeLessThanOrEqual(levels + 1e-8)
    expect(shrunk.smoothingParameters.length).toBe(1)
  })

  it('has standard errors equal to the classical ones for an unpenalised model', () => {
    const n = 40
    const x = grid(n)
    const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) + 0.3 * (wobble(i) - 0.5))
    const fitted = gam.with({ terms: [linear(0)] }).fit({ x, y })
    const xs = x.map((r) => r[0] ?? 0)
    const xm = xs.reduce((a, b) => a + b, 0) / n
    const sxx = xs.reduce((a, v) => a + (v - xm) ** 2, 0)
    const pred = toFlat(fitted.predict(x))
    const sigma2 = y.reduce((a, v, i) => a + (v - (pred[i] ?? 0)) ** 2, 0) / (n - 2)
    const at = 0.8
    const expected = Math.sqrt(sigma2 * (1 / n + (at - xm) ** 2 / sxx))
    expect(toFlat(fitted.standardErrors([[at]]))[0]).toBeCloseTo(expected, 10)
  })

  it('refuses inconsistent models', () => {
    const { x, y } = noisy(20, 0.1)
    expect(() => gam.with({ terms: [s(0, { k: 10 }), s(0, { k: 10 }), s(0, { k: 10 })] }).fit({ x, y })).toThrow()
    expect(() => gam.with({ method: 'ubre' }).fit({ x, y })).toThrow(/scale/)
    expect(() => gam.with({ method: 'fixed' }).fit({ x, y })).toThrow(/sp/)
    expect(() => gam.with({ terms: [s(3)] }).fit({ x, y })).toThrow(/column/)
    const fitted = gam.with({ terms: [factor(0)] }).fit({ x: [[0], [1], [2], [0], [1], [2]], y: [1, 2, 3, 1, 2, 3] })
    expect(() => fitted.predict([[5]])).toThrow(/level/)
  })
})

describe('backfitting and the session', () => {
  const n = 80
  const x = Array.from({ length: n }, (_, i) => [i / (n - 1), wobble(i), ((i * 7) % n) / n])
  const y = x.map((r) => Math.sin(5 * (r[0] ?? 0)) + 0.5 * (r[1] ?? 0) ** 2 + 2 * (r[2] ?? 0))
  const hyper = {
    ...GAM_DEFAULTS,
    terms: [s(0, { k: 6 }), s(1, { k: 6 }), linear(2)],
    method: 'fixed' as const,
    sp: [0.5, 2],
    tol: 1e-12,
  }

  it('converges to the joint penalised solution at fixed smoothing parameters', () => {
    const problem = gamProblem(x, y, hyper)
    const joint = fitGam(problem, hyper).state
    const result = runAlgorithm(backfittingAlgorithm(problem, hyper), { maxSteps: 2000 })
    expect(result.state.converged).toBe(true)
    result.state.eta.forEach((e, i) => {
      expect(e).toBeCloseTo(joint.eta[i] ?? 0, 7)
    })
  })

  it('backfits an exponential family by local scoring to the joint solution', () => {
    const counts = x.map((r) => Math.round(Math.exp(0.5 + (r[0] ?? 0))))
    const h = { ...hyper, family: poisson(), terms: [s(0, { k: 6 }), linear(2)], sp: [1] }
    const problem = gamProblem(x, counts, h)
    const joint = fitGam(problem, h).state
    const result = runAlgorithm(backfittingAlgorithm(problem, h), { maxSteps: 3000 })
    expect(result.state.converged).toBe(true)
    result.state.eta.forEach((e, i) => {
      expect(e).toBeCloseTo(joint.eta[i] ?? 0, 5)
    })
  })

  it('has leverages that sum to the effective degrees of freedom', () => {
    const problem = gamProblem(x, y, hyper)
    const { state } = fitGam(problem, hyper)
    const h = influenceDiagonal(problem, state)
    expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(state.edf, 9)
    for (const v of h) {
      expect(v).toBeGreaterThan(0)
      expect(v).toBeLessThan(1)
    }
    const { solution } = gamSolution(problem, state)
    expect(termEdf(problem.design, solution.edf).reduce((a, b) => a + b, 1)).toBeCloseTo(state.edf, 9)
  })

  it('serialises its state, and the session reaches every step and revives from parameters', () => {
    const data = { x, y }
    const est = gam.with({ terms: [s(0, { k: 6 }), s(1, { k: 6 })] })
    const problem = gamProblem(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 6 }), s(1, { k: 6 })] })
    const alg = gamAlgorithm(problem, { ...GAM_DEFAULTS, terms: [s(0, { k: 6 }), s(1, { k: 6 })] })
    const run = runAlgorithm(alg, { maxSteps: 50 })
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(run.state))))).toEqual(run.state)
    const session = est.session(data)
    session.advance({ steps: 3 })
    expect(session.fittedAt(1).params.coef.length).toBe(problem.design.p)
    const fitted = est.fit(data)
    const revived = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    const a = toFlat(fitted.predict(x))
    const b = toFlat(revived.predict(x))
    a.forEach((v, i) => {
      expect(b[i]).toBeCloseTo(v, 10)
    })
    expect(buildDesign(x, [s(0, { k: 6 })]).p).toBe(evaluateDesign(buildDesign(x, [s(0, { k: 6 })]), x).X.length / n)
    expect(penaltyStructure(problem.design).blocks.length).toBe(2)
  })
})

describe('the session of a GAM', () => {
  it('probes the fitted mean with its credible band at any step, the last equal to the fitted value’s', () => {
    const x = Array.from({ length: 60 }, (_, i) => [i / 59])
    const y = x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.1 * Math.cos(37 * i))
    const model = gam.with({ terms: [s(0, { k: 8 })] })
    const live = session(model.method({ x, y }), { mode: 'eager', steps: 30 })
    const last = live.t
    const probed = live.probe(last, 'model/band') as { fit: number[]; lower: number[]; upper: number[] }
    const band = model.fit({ x, y }).predictionBand(x)
    expect(probed.fit).toHaveLength(60)
    for (let i = 0; i < 60; i++) {
      expect(probed.fit[i]).toBeCloseTo(band.fit[i] ?? Number.NaN, 8)
      expect(probed.lower[i]).toBeCloseTo(band.lower[i] ?? Number.NaN, 8)
      expect(probed.upper[i]).toBeCloseTo(band.upper[i] ?? Number.NaN, 8)
      expect(probed.upper[i] ?? 0).toBeGreaterThan(probed.lower[i] ?? 0)
    }
    // At rows of the caller's choosing: a dense grid for drawing.
    const grid = Array.from({ length: 301 }, (_, i) => [i / 300])
    const dense = live.probe(last, 'model/band', { x: grid } as never) as { fit: number[] }
    const direct = model.fit({ x, y }).predictionBand(grid)
    expect(dense.fit).toHaveLength(301)
    expect(dense.fit[150]).toBeCloseTo(direct.fit[150] ?? Number.NaN, 8)
    // An earlier step is a different (earlier) model, probed without rerunning the fit.
    expect(live.probe(0, 'model/band')).toBeDefined()
  })
})
