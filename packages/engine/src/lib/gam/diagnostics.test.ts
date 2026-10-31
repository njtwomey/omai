/**
 * Tests for model selection and diagnostics: the double penalty shrinks an irrelevant term out, the information
 * criteria and the nested-model test behave under a planted truth (including uniform p-values under the null over many
 * simulated datasets), residuals are what their definitions say, the QQ band holds a correct model and not a wrong one,
 * the k-index flags a basis that is too small, concurvity finds a duplicated covariate, and Cook's distance is the
 * regression one for an unpenalised model.
 */

import { describe, expect, it } from 'vitest'
import { gamma, log, poisson } from '../dists/families/index.ts'
import { foldIn, key, normal, poisson as poissonDraw } from '../random/index.ts'
import {
  compareGams,
  concurvity,
  cooksDistance,
  fitContext,
  fitGam,
  fitGamOuter,
  GAM_DEFAULTS,
  type GamHyper,
  gamInformation,
  gamProblem,
  gamResiduals,
  gamSummary,
  kIndex,
  linear,
  qqData,
  residualsVsFitted,
  s,
  smoothingCorrection,
} from './index.ts'

const noise = (seed: number, n: number): Float64Array => normal(key(seed), [n]).data as Float64Array
const fitOf = (x: number[][], y: number[], hyper: GamHyper) => {
  const problem = gamProblem(x, y, hyper)
  const { state } = hyper.iteration === 'outer' ? fitGamOuter(problem, hyper) : fitGam(problem, hyper)
  return { problem, state, ctx: fitContext(problem, state) }
}

describe('the double penalty', () => {
  const n = 150
  const z = noise(3, 3 * n)
  const x = Array.from({ length: n }, (_, i) => [
    (i + 0.5) / n,
    Math.abs(Math.sin(i * 12.9898)),
    Math.abs(Math.cos(i * 4.1414)),
  ])
  const y = x.map((r, i) => Math.sin(2 * Math.PI * (r[0] ?? 0)) + 0.3 * (z[i] ?? 0))

  it('shrinks a term with no effect out of the model and leaves a real one alone', () => {
    const plain = fitOf(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 8 }), s(1, { k: 8 })] })
    const shrunk = fitOf(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 8, select: true }), s(1, { k: 8, select: true })] })
    expect(shrunk.problem.design.nPenalties).toBe(4)
    expect(shrunk.state.edfTerms[1]).toBeLessThan(0.8)
    expect(shrunk.state.edfTerms[1]).toBeLessThan((plain.state.edfTerms[1] ?? 0) - 0.3)
    expect(Math.abs((shrunk.state.edfTerms[0] ?? 0) - (plain.state.edfTerms[0] ?? 0))).toBeLessThan(0.5)
  })

  it('works with the outer iteration too', () => {
    const hyper = {
      ...GAM_DEFAULTS,
      iteration: 'outer' as const,
      terms: [s(0, { k: 8, select: true }), s(1, { k: 8, select: true })],
    }
    const fit = fitOf(x, y, hyper)
    expect(fit.state.edfTerms[1]).toBeLessThan(0.8)
  })
})

describe('information criteria and the nested-model test', () => {
  const n = 120
  const z = noise(8, 2 * n)
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n, Math.abs(Math.sin(i * 7.77))])
  const y = x.map((r, i) => 1 + Math.sin(2 * Math.PI * (r[0] ?? 0)) + 0.3 * (z[i] ?? 0))

  it('has a Gaussian log-likelihood equal to the closed form and a corrected AIC above the conditional one', () => {
    const hyper = { ...GAM_DEFAULTS, iteration: 'outer' as const, terms: [s(0, { k: 8 })] }
    const { problem, state, ctx } = fitOf(x, y, hyper)
    const info = gamInformation(problem, state, hyper)
    const rss = y.reduce((a, v, i) => a + (v - (ctx.mu[i] ?? 0)) ** 2, 0)
    expect(info.logLik).toBeCloseTo(-0.5 * n * Math.log(2 * Math.PI * ctx.scale) - rss / (2 * ctx.scale), 8)
    expect(info.edf2).toBeGreaterThanOrEqual(info.edf)
    expect(info.edf2).toBeLessThan(info.edf + 2)
    expect(info.aic).toBeGreaterThanOrEqual(info.aicConditional)
    expect(info.bic).toBeGreaterThan(info.aic)
    expect(smoothingCorrection(problem, state, hyper).vc.length).toBe(problem.design.p ** 2)
  })

  it('prefers the model with the real effect to the model without it', () => {
    const real = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] }
    const none = { ...GAM_DEFAULTS, terms: [linear(1)] }
    const a = fitOf(x, y, real)
    const b = fitOf(x, y, none)
    expect(gamInformation(a.problem, a.state, real).aic).toBeLessThan(gamInformation(b.problem, b.state, none).aic)
  })

  it('has p-values near uniform for an irrelevant extra term, over 80 simulated datasets, and small for a real one', () => {
    const reps = 80
    const nn = 100
    let rejectsNull = 0
    let rejectsReal = 0
    for (let r = 0; r < reps; r++) {
      const e = noise(500 + r, nn)
      const xs = Array.from({ length: nn }, (_, i) => [
        (i + 0.5) / nn,
        Math.abs(Math.sin((i + r) * 9.13)),
        Math.abs(Math.cos((i + 3 * r) * 2.7)),
      ])
      const small = fitOf(
        xs,
        xs.map((row, i) => 1 + Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.4 * (e[i] ?? 0)),
        {
          ...GAM_DEFAULTS,
          terms: [s(0, { k: 7 })],
        },
      )
      const yNull = xs.map((row, i) => 1 + Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.4 * (e[i] ?? 0))
      const largeNull = fitOf(xs, yNull, { ...GAM_DEFAULTS, terms: [s(0, { k: 7 }), s(1, { k: 5 })] })
      if (compareGams(small.ctx, largeNull.ctx).pValue < 0.05) rejectsNull++
      const yReal = xs.map(
        (row, i) =>
          1 + Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.8 * Math.sin(2 * Math.PI * (row[2] ?? 0)) + 0.4 * (e[i] ?? 0),
      )
      const smallReal = fitOf(xs, yReal, { ...GAM_DEFAULTS, terms: [s(0, { k: 7 })] })
      const largeReal = fitOf(xs, yReal, { ...GAM_DEFAULTS, terms: [s(0, { k: 7 }), s(2, { k: 6 })] })
      if (compareGams(smallReal.ctx, largeReal.ctx).pValue < 0.01) rejectsReal++
    }
    // The nominal 5% is a little conservative for penalised models; never far above it. With 80 runs, 0.15 is 3.5 sigma.
    expect(rejectsNull / reps).toBeLessThan(0.15)
    expect(rejectsReal / reps).toBeGreaterThan(0.95)
  })

  it('refuses models given in the wrong order', () => {
    const a = fitOf(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] })
    expect(() => compareGams(a.ctx, a.ctx)).toThrow(/more effective/)
    expect(gamSummary(a.ctx).smooth.length).toBe(1)
  })
})

describe('residuals and the QQ band', () => {
  const n = 150
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
  const mean = (t: number): number => Math.exp(1 + Math.sin(2 * Math.PI * t))
  const counts = x.map((r, i) => poissonDraw(foldIn(key(5), i), mean(r[0] ?? 0), []).data[0] ?? 0)

  it('defines the residual kinds by their formulas', () => {
    const { ctx } = fitOf(x, counts, { ...GAM_DEFAULTS, family: poisson(), terms: [s(0, { k: 8 })] })
    const pearson = gamResiduals(ctx, 'pearson')
    const working = gamResiduals(ctx, 'working')
    const response = gamResiduals(ctx, 'response')
    const deviance = gamResiduals(ctx)
    for (let i = 0; i < n; i += 17) {
      const mu = ctx.mu[i] ?? 1
      expect(response[i]).toBeCloseTo((counts[i] ?? 0) - mu, 10)
      expect(pearson[i]).toBeCloseTo(((counts[i] ?? 0) - mu) / Math.sqrt(mu), 10)
      expect(working[i]).toBeCloseTo(((counts[i] ?? 0) - mu) / mu, 10)
      const y = counts[i] ?? 0
      const d = 2 * ((y === 0 ? 0 : y * Math.log(y / mu)) - (y - mu))
      expect((deviance[i] ?? 0) ** 2).toBeCloseTo(d, 8)
    }
    expect(residualsVsFitted(ctx).fitted.length).toBe(n)
  })

  it('holds the residuals of the correct model in the band and not those of a wrong family', () => {
    const inside = (q: ReturnType<typeof qqData>): number => {
      let k = 0
      q.sample.forEach((v, i) => {
        if (v >= (q.lower[i] ?? 0) && v <= (q.upper[i] ?? 0)) k++
      })
      return k / q.sample.length
    }
    const right = fitOf(x, counts, { ...GAM_DEFAULTS, family: poisson(), terms: [s(0, { k: 8 })] })
    expect(inside(qqData(right.ctx, key(1), { reps: 100 }))).toBeGreaterThan(0.9)
    // Strongly overdispersed positive data (a gamma with shape 0.5) fitted as Gaussian: the residuals are skewed.
    const skewed = x.map((r, i) => mean(r[0] ?? 0) * Math.max(1e-3, (noise(2, n)[i] ?? 0) ** 2))
    const wrong = fitOf(x, skewed, { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] })
    expect(inside(qqData(wrong.ctx, key(1), { reps: 100 }))).toBeLessThan(
      inside(qqData(right.ctx, key(1), { reps: 100 })),
    )
    const gam2 = fitOf(
      x,
      skewed.map((v) => v + 0.01),
      { ...GAM_DEFAULTS, family: gamma(log()), terms: [s(0, { k: 8 })] },
    )
    expect(inside(qqData(gam2.ctx, key(1), { reps: 100 }))).toBeGreaterThan(
      inside(qqData(wrong.ctx, key(1), { reps: 100 })),
    )
  })
})

describe('the basis check, concurvity and influence', () => {
  const n = 160
  const z = noise(14, n)
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
  const y = x.map((r, i) => Math.sin(8 * Math.PI * (r[0] ?? 0)) + 0.15 * (z[i] ?? 0))

  it('gives a small k-index and p-value to a basis that is too small and about 1 to an adequate one', () => {
    const small = kIndex(fitOf(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 5 })] }).ctx, key(0))[0]
    const large = kIndex(fitOf(x, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 20 })] }).ctx, key(0))[0]
    expect(small?.kIndex).toBeLessThan(0.7)
    expect(small?.pValue).toBeLessThan(0.05)
    expect(large?.kIndex).toBeGreaterThan(0.85)
    expect(large?.pValue).toBeGreaterThan(0.05)
  })

  it('finds the concurvity of a covariate that is a copy of another and not of independent ones', () => {
    const copy = x.map((r, i) => [r[0] ?? 0, (r[0] ?? 0) + 1e-4 * (z[i] ?? 0)])
    const dup = concurvity(fitOf(copy, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 5 }), s(1, { k: 5 })] }).ctx)
    expect(dup[0]?.worst).toBeGreaterThan(0.95)
    const indep = x.map((_, i) => [(i + 0.5) / n, Math.abs(Math.sin(i * 12.9898))])
    const free = concurvity(fitOf(indep, y, { ...GAM_DEFAULTS, terms: [s(0, { k: 5 }), s(1, { k: 5 })] }).ctx)
    expect(free[0]?.worst).toBeLessThan(0.5)
  })

  it("is Cook's distance of linear regression for an unpenalised model, and flags a planted outlier", () => {
    const xs = x.slice(0, 40)
    const ys = xs.map((r, i) => 1 + 2 * (r[0] ?? 0) + 0.2 * (z[i] ?? 0))
    ys[7] = 4
    const hyper = { ...GAM_DEFAULTS, terms: [linear(0)] }
    const { state, ctx } = fitOf(xs, ys, hyper)
    const d = cooksDistance(ctx, state)
    const m = ys.length
    const xb = xs.reduce((a, r) => a + (r[0] ?? 0), 0) / m
    const sxx = xs.reduce((a, r) => a + ((r[0] ?? 0) - xb) ** 2, 0)
    const slope = xs.reduce((a, r, i) => a + ((r[0] ?? 0) - xb) * (ys[i] ?? 0), 0) / sxx
    const icpt = ys.reduce((a, v) => a + v, 0) / m - slope * xb
    const res = ys.map((v, i) => v - icpt - slope * (xs[i]?.[0] ?? 0))
    const s2 = res.reduce((a, v) => a + v * v, 0) / (m - 2)
    xs.forEach((r, i) => {
      const h = 1 / m + ((r[0] ?? 0) - xb) ** 2 / sxx
      expect(d[i]).toBeCloseTo(((res[i] ?? 0) ** 2 * h) / (2 * s2 * (1 - h) ** 2), 8)
    })
    expect(d.indexOf(Math.max(...d))).toBe(7)
  })
})
