/**
 * Cox proportional hazards with smooth terms: with only linear terms the fit is the maximum partial likelihood of
 * `survival`'s `coxPh` (both tie rules), the criterion's gradient is the gradient of the criterion, and a planted smooth
 * log hazard is recovered, with the survival curves following it.
 */

import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { coxData, coxPh } from '../survival/index.ts'
import { toFlat } from '../tensor/index.ts'
import { coxBaseline, coxGamModel, coxSurvival, evaluateExtended, fitCoxGam, linear, s } from './index.ts'

const flat = (v: Parameters<typeof toFlat>[0]): number[] => Array.from(toFlat(v))
const FIXED = { method: 'fixed', sp: undefined, aux: undefined, maxIter: 3, solver: undefined } as const

/** Survival times with log hazard `f(x)`, exponential baseline, uniform censoring. */
function simulate(seed: number, n: number, f: (t: number) => number, ties = false) {
  const u = flat(uniform(key(seed), [3 * n]))
  const x = Array.from({ length: n }, (_, i) => [(u[i] ?? 0) * 1])
  const raw = x.map((r, i) => -Math.log(1 - (u[n + i] ?? 0.5)) / (0.4 * Math.exp(f(r[0] ?? 0))))
  const cens = x.map((_, i) => 0.5 + 6 * (u[2 * n + i] ?? 0.5))
  const time = raw.map((t, i) => {
    const v = Math.min(t, cens[i] ?? 0)
    return ties ? Math.round(v * 5) / 5 + 0.2 : v
  })
  const event = raw.map((t, i) => (t <= (cens[i] ?? 0) ? 1 : 0))
  return { x, time, event }
}

describe('Cox with smooth terms', () => {
  it('with linear terms only is the maximum partial likelihood of coxPh, with Breslow and Efron ties', () => {
    const { x, time, event } = simulate(5, 120, (t) => 0.8 * t, true)
    for (const ties of ['efron', 'breslow'] as const) {
      const model = coxGamModel(x, time, event, [linear(0)], { ties })
      const fit = fitCoxGam(model, FIXED)
      const reference = coxPh(coxData(time, event, x), { ties })
      expect(fit.state.beta[0] ?? Number.NaN, ties).toBeCloseTo(reference.beta[0] ?? 0, 6)
      expect(fit.state.logLik, ties).toBeCloseTo(reference.logLikelihood, 6)
    }
  })

  it('has the analytic criterion gradient in log lambda equal to differences of the criterion', () => {
    const { x, time, event } = simulate(6, 100, (t) => Math.sin(5 * t))
    const model = coxGamModel(x, time, event, [s(0, { k: 6 })])
    const ev = evaluateExtended(model, [2], [])
    const h = 1e-4
    const value = (l: number): number => evaluateExtended(model, [l], [], { gradient: false }).value
    expect(ev.gradient?.[0] ?? Number.NaN).toBeCloseTo((value(2 + h) - value(2 - h)) / (2 * h), 4)
  })

  it('recovers a planted smooth log hazard and the survival curves', () => {
    const f = (t: number): number => 1.1 * Math.sin(2 * Math.PI * t)
    const { x, time, event } = simulate(7, 400, f)
    const model = coxGamModel(x, time, event, [s(0, { k: 8 })])
    const fit = fitCoxGam(model)
    expect(fit.converged).toBe(true)
    const eta = fit.eta[0] as Float64Array
    // centre both on the sample (the baseline hazard absorbs the level)
    const truth = x.map((r) => f(r[0] ?? 0))
    const mean = (a: ArrayLike<number>): number => Array.from(a).reduce((p, c) => p + c, 0) / a.length
    const m1 = mean(eta)
    const m0 = mean(truth)
    let sq = 0
    eta.forEach((e, i) => {
      sq += (e - m1 - ((truth[i] ?? 0) - m0)) ** 2
    })
    expect(Math.sqrt(sq / eta.length)).toBeLessThan(0.3)
    // the survival curve of a low-hazard subject lies above that of a high-hazard subject
    const surv = coxSurvival(model, fit, [[0.75], [0.25]], [1, 3])
    expect(surv[0]?.[0] ?? 0).toBeGreaterThan(surv[1]?.[0] ?? 1)
    expect(surv[0]?.[1] ?? 0).toBeLessThan(surv[0]?.[0] ?? 0)
    expect(coxBaseline(model, fit)[0]?.cumulativeHazard.at(-1) ?? 0).toBeGreaterThan(0)
  }, 120000)
})
