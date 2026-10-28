import { describe, expect, it } from 'vitest'
import { expectResumable, simulateArma } from '../../testing/series.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const SES = T.smoothingModels['simple'] as T.EtsSpec

describe('exponential smoothing', () => {
  it('moves the level a fraction alpha towards each observation', () => {
    const f = T.etsFilter([12, 14, 13], SES, { alpha: 0.5, level0: 10 })
    expect(Array.from(f.level)).toEqual([11, 12.5, 12.75])
    expect(Array.from(f.fitted)).toEqual([10, 11, 12.5])
    expect(Array.from(f.residuals)).toEqual([2, 3, 0.5])
    expect(f.sigma2).toBeCloseTo((4 + 9 + 0.25) / 3, 12)
  })

  it('has Holt-Winters weights that keep an exactly seasonal series exactly forecast', () => {
    const season = [3, -1, -4, 2]
    const y = Array.from({ length: 24 }, (_, t) => 100 + 2 * t + (season[t % 4] ?? 0))
    const spec = T.holtWintersAdditive(4)
    const f = T.etsFilter(y, spec, { alpha: 0.3, beta: 0.1, gamma: 0.2, level0: 100 - 2, trend0: 2, seasonal0: season })
    f.residuals.forEach((e) => {
      expect(Math.abs(e)).toBeLessThan(1e-9)
    })
    const fc = T.etsForecast(
      {
        spec,
        params: { alpha: 0.3, level0: 0 },
        logLikelihood: 0,
        sigma2: 1,
        nparams: 0,
        aic: 0,
        bic: 0,
        nobs: 24,
        final: f.final,
        converged: true,
        iterations: 0,
      },
      4,
    )
    fc.mean.forEach((v, h) => {
      expect(v).toBeCloseTo(100 + 2 * (24 + h) + (season[(24 + h) % 4] ?? 0), 9)
    })
  })

  it('runs one observation per step and records the running likelihood', () => {
    const y = simulateArma(31, 30, [0.5], [], 20)
    const params = { alpha: 0.4, level0: 20 }
    const rec = recorder()
    const run = runAlgorithm(T.etsAlgorithm(y, SES, params), { maxSteps: 100, tracer: rec })
    expect(run.steps).toBe(30)
    const ll = metricSeries(rec.run(), 'logLikelihood').values
    expect(ll[ll.length - 1]).toBeCloseTo(T.etsFilter(y, SES, params).logLikelihood, 9)
    expectResumable(
      T.etsAlgorithm(y, T.holtWintersAdditive(4), {
        alpha: 0.2,
        beta: 0.1,
        gamma: 0.1,
        level0: 20,
        trend0: 0,
        seasonal0: [0, 0, 0, 0],
      }),
      7,
      9,
    )
  })

  it('fits the smoothing weight of a local-level series', () => {
    // A random walk plus noise: the optimal alpha is the steady-state Kalman gain.
    const noise = simulateArma(32, 400, [], [], 0, 1)
    const walk = simulateArma(33, 400, [], [], 0, 0.7).reduce<number[]>((a, v, t) => {
      a.push((a[t - 1] ?? 50) + v)
      return a
    }, [])
    const y = walk.map((v, t) => v + (noise[t] ?? 0))
    const fit = T.etsFit(y, SES)
    // With state variance q = 0.49 and noise variance r = 1 the steady-state prior variance is P = (q + sqrt(q^2 + 4 q r)) / 2
    // and the gain alpha = P / (P + r).
    const q = 0.49
    const p = (q + Math.sqrt(q * q + 4 * q)) / 2
    expect(Math.abs(fit.params.alpha - p / (p + 1))).toBeLessThan(0.12)
    expect(fit.nobs).toBe(400)
  })

  it('refuses a multiplicative model on a non-positive series and a seasonal model without a period', () => {
    expect(() => T.etsFit([1, 2, -1, 3, 4, 5, 2, 3], { error: 'mul', trend: 'none', seasonal: 'none' })).toThrow()
    expect(() =>
      T.etsFilter([1, 2, 3], { error: 'add', trend: 'none', seasonal: 'add' }, { alpha: 0.1, level0: 1 }),
    ).toThrow()
  })

  it('has analytic forecast variances that grow, and are exact for simple smoothing', () => {
    const fit = {
      spec: SES,
      params: { alpha: 0.3, level0: 0 },
      logLikelihood: 0,
      sigma2: 2,
      nparams: 1,
      aic: 0,
      bic: 0,
      nobs: 10,
      final: { level: 5, trend: 0, seasonal: [] },
      converged: true,
      iterations: 0,
    } as const
    const f = T.etsForecast(fit, 4)
    f.mean.forEach((v) => {
      expect(v).toBe(5)
    })
    f.se?.forEach((s, h) => {
      expect(s * s).toBeCloseTo(2 * (1 + 0.09 * h), 12)
    })
    expect(T.etsForecast({ ...fit, spec: { error: 'mul', trend: 'none', seasonal: 'none' } }, 2).se).toBeUndefined()
  })
})
