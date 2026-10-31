/**
 * GAMLSS: the three algorithms (generic extended Newton, Cole-Green, Rigby-Stasinopoulos) agree at fixed smoothing
 * parameters, find about the same smoothing parameters, recover planted location, scale and shape curves for the
 * Gaussian, gamma, negative binomial and Student-t families, and the quantile residuals of the true model are standard
 * normal with the worm inside its band.
 */

import { describe, expect, it } from 'vitest'
import { families, gamma, normal as normalDist, poisson, studentT } from '../dists/index.ts'
import { key, normal, split } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import {
  extendedModel,
  fitGamlss,
  GAMLSS_DEFAULTS,
  type GamlssMode,
  predictExtended,
  quantileResiduals,
  s,
  wormPlot,
} from './index.ts'

const flat = (v: Parameters<typeof toFlat>[0]): number[] => Array.from(toFlat(v))
const grid = (n: number): number[][] => Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
const rmse = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let sq = 0
  for (let i = 0; i < a.length; i++) sq += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return Math.sqrt(sq / a.length)
}
const MODES: GamlssMode[] = ['newton', 'cg', 'rs']

describe('GAMLSS algorithms', () => {
  const n = 150
  const x = grid(n)
  const z = flat(normal(key(3), [n]))
  const mean = (t: number): number => Math.sin(2 * Math.PI * t)
  const logsd = (t: number): number => -1.2 + 1.5 * t
  const y = x.map((r, i) => mean(r[0] ?? 0) + Math.exp(logsd(r[0] ?? 0)) * (z[i] ?? 0))
  const model = extendedModel(x, y, families.gamlssExtended('normal'), [[s(0, { k: 8 })], [s(0, { k: 5 })]])

  it('reach the same penalised maximum at fixed smoothing parameters', () => {
    const sp = [0.5, 2]
    const fits = MODES.map((mode) =>
      fitGamlss(model, { ...GAMLSS_DEFAULTS, mode, method: 'fixed', sp, maxIter: 400, tol: 1e-13 }),
    )
    for (const fit of fits.slice(1)) {
      expect(fit.state.logLik).toBeCloseTo(fits[0]?.state.logLik ?? 0, 6)
      fit.state.beta.forEach((b, j) => {
        expect(b).toBeCloseTo(fits[0]?.state.beta[j] ?? 0, 4)
      })
    }
  }, 120000)

  it('choose similar smoothing parameters and fits by REML, with a global deviance of minus twice the log-likelihood', () => {
    const fits = MODES.map((mode) => fitGamlss(model, { ...GAMLSS_DEFAULTS, mode }))
    const reference = predictExtended(model, fits[0] as never, x)
    for (const [i, fit] of fits.entries()) {
      expect(fit.converged, MODES[i]).toBe(true)
      expect(fit.globalDeviance).toBeCloseTo(-2 * fit.state.logLik, 10)
      const pred = predictExtended(model, fit, x)
      expect(rmse(pred.eta[0] as Float64Array, reference.eta[0] as Float64Array), `${MODES[i]} mean`).toBeLessThan(0.1)
      expect(rmse(pred.eta[1] as Float64Array, reference.eta[1] as Float64Array), `${MODES[i]} logsd`).toBeLessThan(0.2)
    }
    const labels = Object.keys(fits[0]?.edfByTerm ?? {})
    expect(labels).toEqual(['mu: s(x0)', 'sigma: s(x0)'])
  }, 180000)

  it('give quantile residuals of the true model that are standard normal, with the worm inside the band', () => {
    const fit = fitGamlss(model)
    const r = quantileResiduals(model, fit, y, key(5))
    const m = r.reduce((a, b) => a + b, 0) / n
    const v = r.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1)
    expect(Math.abs(m)).toBeLessThan(0.2)
    expect(Math.abs(Math.sqrt(v) - 1)).toBeLessThan(0.2)
    const w = wormPlot(r)
    let inside = 0
    for (let i = 0; i < n; i++) if (Math.abs(w.deviation[i] ?? 0) <= (w.band[i] ?? 0) * 1.5) inside++
    expect(inside / n).toBeGreaterThan(0.9)
  }, 120000)
})

describe('GAMLSS families on planted data', () => {
  it('gamma: the mean and the coefficient of variation both vary', () => {
    const n = 300
    const x = grid(n)
    const mu = (t: number): number => Math.exp(1 + 0.7 * Math.sin(2 * Math.PI * t))
    const sigma = (t: number): number => Math.exp(-0.9 + 0.8 * t)
    const y = flat(
      gamma(
        tensor(x.map((r) => 1 / sigma(r[0] ?? 0) ** 2)),
        tensor(x.map((r) => 1 / sigma(r[0] ?? 0) ** 2 / mu(r[0] ?? 0))),
      ).sample(key(7)),
    )
    const model = extendedModel(x, y, families.gamlssExtended('gamma'), [[s(0, { k: 8 })], [s(0, { k: 4 })]])
    const fit = fitGamlss(model, { ...GAMLSS_DEFAULTS, mode: 'cg' })
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(
      rmse(
        pred.eta[0] as Float64Array,
        x.map((r) => Math.log(mu(r[0] ?? 0))),
      ),
    ).toBeLessThan(0.2)
    expect(
      rmse(
        pred.eta[1] as Float64Array,
        x.map((r) => Math.log(sigma(r[0] ?? 0))),
      ),
    ).toBeLessThan(0.3)
  }, 180000)

  it('negative binomial: the mean and the dispersion both vary', () => {
    const n = 400
    const x = grid(n)
    const mu = (t: number): number => Math.exp(1.3 + 0.8 * Math.sin(2 * Math.PI * t))
    const sigma = (t: number): number => Math.exp(-1 + 1.0 * t) // variance mu + sigma mu^2
    const [k1, k2] = split(key(9), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
    const rate = flat(
      gamma(
        tensor(x.map((r) => 1 / sigma(r[0] ?? 0))),
        tensor(x.map((r) => 1 / sigma(r[0] ?? 0) / mu(r[0] ?? 0))),
      ).sample(k1),
    )
    const y = flat(poisson(tensor(rate)).sample(k2))
    const model = extendedModel(x, y, families.gamlssExtended('negativeBinomial'), [[s(0, { k: 8 })], [s(0, { k: 4 })]])
    const fit = fitGamlss(model, { ...GAMLSS_DEFAULTS, mode: 'rs' })
    expect(fit.converged).toBe(true)
    const pred = predictExtended(model, fit, x)
    expect(
      rmse(
        pred.eta[0] as Float64Array,
        x.map((r) => Math.log(mu(r[0] ?? 0))),
      ),
    ).toBeLessThan(0.25)
    // the dispersion is weakly identified: its average level, not its shape, is the claim
    const meanLog = (pred.eta[1] as Float64Array).reduce((a, b) => a + b, 0) / n
    const truth = x.reduce((a, r) => a + Math.log(sigma(r[0] ?? 0)), 0) / n
    expect(Math.abs(meanLog - truth)).toBeLessThan(0.5)
  }, 240000)

  it('Student-t: heavy tails give a small estimated degrees of freedom, a normal sample a large one', () => {
    const n = 300
    const x = grid(n)
    const mean = (t: number): number => 2 * Math.sin(2 * Math.PI * t)
    const heavy = flat(
      studentT(tensor(x.map(() => 3)), tensor(x.map((r) => mean(r[0] ?? 0))), tensor(x.map(() => 0.5))).sample(key(13)),
    )
    const light = flat(normalDist(tensor(x.map((r) => mean(r[0] ?? 0))), tensor(x.map(() => 0.5))).sample(key(14)))
    const dfOf = (y: number[]): { df: number; err: number } => {
      const model = extendedModel(x, y, families.gamlssExtended('studentT'), [[s(0, { k: 8 })], [], []])
      const fit = fitGamlss(model, { ...GAMLSS_DEFAULTS, mode: 'cg', maxIter: 150 })
      const pred = predictExtended(model, fit, x)
      return {
        df: Math.exp(pred.eta[2]?.[0] ?? 0),
        err: rmse(
          pred.eta[0] as Float64Array,
          x.map((r) => mean(r[0] ?? 0)),
        ),
      }
    }
    const h = dfOf(heavy)
    const l = dfOf(light)
    expect(h.df).toBeLessThan(8)
    expect(l.df).toBeGreaterThan(h.df * 2)
    expect(h.err).toBeLessThan(0.25)
  }, 300000)
})
