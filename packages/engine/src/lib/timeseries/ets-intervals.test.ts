import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import * as T from './index.ts'

const fitOf = (spec: T.EtsSpec, params: T.EtsParameters, sigma2: number, final: T.EtsFit['final']) =>
  ({ spec, params, sigma2, final }) as Pick<T.EtsFit, 'spec' | 'params' | 'sigma2' | 'final'>

const models: readonly [string, Pick<T.EtsFit, 'spec' | 'params' | 'sigma2' | 'final'>][] = [
  [
    'MNN',
    fitOf({ error: 'mul', trend: 'none', seasonal: 'none' }, { alpha: 0.3, level0: 50 }, 0.01, {
      level: 50,
      trend: 0,
      seasonal: [],
    }),
  ],
  [
    'MAN',
    fitOf({ error: 'mul', trend: 'add', seasonal: 'none' }, { alpha: 0.4, beta: 0.1, level0: 50, trend0: 1 }, 0.008, {
      level: 50,
      trend: 1,
      seasonal: [],
    }),
  ],
  [
    'MAdN',
    fitOf(
      { error: 'mul', trend: 'add', damped: true, seasonal: 'none' },
      { alpha: 0.4, beta: 0.1, phi: 0.9, level0: 50, trend0: 1 },
      0.008,
      { level: 50, trend: 1, seasonal: [] },
    ),
  ],
  [
    'MNA',
    fitOf(
      { error: 'mul', trend: 'none', seasonal: 'add', period: 4 },
      { alpha: 0.3, gamma: 0.2, level0: 50, seasonal0: [2, -1, -3, 2] },
      0.01,
      { level: 50, trend: 0, seasonal: [2, -1, -3, 2] },
    ),
  ],
  [
    'MAA',
    fitOf(
      { error: 'mul', trend: 'add', seasonal: 'add', period: 4 },
      { alpha: 0.3, beta: 0.05, gamma: 0.2, level0: 50, trend0: 0.5, seasonal0: [2, -1, -3, 2] },
      0.01,
      { level: 50, trend: 0.5, seasonal: [2, -1, -3, 2] },
    ),
  ],
  [
    'ANN',
    fitOf({ error: 'add', trend: 'none', seasonal: 'none' }, { alpha: 0.3, level0: 50 }, 4, {
      level: 50,
      trend: 0,
      seasonal: [],
    }),
  ],
  [
    'AAA',
    fitOf(
      { error: 'add', trend: 'add', seasonal: 'add', period: 4 },
      { alpha: 0.3, beta: 0.05, gamma: 0.2, level0: 50, trend0: 0.5, seasonal0: [2, -1, -3, 2] },
      4,
      { level: 50, trend: 0.5, seasonal: [2, -1, -3, 2] },
    ),
  ],
]

describe('ETS prediction variance', () => {
  it('classes the models by the form of their variance', () => {
    expect(T.etsClass({ error: 'add', trend: 'add', seasonal: 'add', period: 4 })).toBe(1)
    expect(T.etsClass({ error: 'mul', trend: 'add', seasonal: 'none' })).toBe(2)
    expect(T.etsClass({ error: 'mul', trend: 'none', seasonal: 'mul', period: 4 })).toBe(3)
    expect(T.etsClass({ error: 'add', trend: 'mul', seasonal: 'none' })).toBe(4)
  })
  it('has the one-step variance sigma2 mu^2 for multiplicative error and sigma2 for additive', () => {
    const mnn = models[0]?.[1] as (typeof models)[number][1]
    expect(T.etsPredictionVariance(mnn, 1)?.[0]).toBeCloseTo(0.01 * 50 * 50, 10)
    const ann = models[5]?.[1] as (typeof models)[number][1]
    expect(T.etsPredictionVariance(ann, 1)?.[0]).toBeCloseTo(4, 12)
  })
  it.each(models)('%s: the closed form agrees with the variance of simulated paths', (_, fit) => {
    const steps = 9
    const closed = T.etsPredictionVariance(fit, steps) as number[]
    const sim = T.etsPredictionIntervals(fit, steps, { method: 'simulation', key: key(7), paths: 40000 })
    for (let h = 0; h < steps; h++) expect(Math.abs((sim.variance[h] ?? 0) / (closed[h] ?? 1) - 1)).toBeLessThan(0.06)
    const mean = T.etsForecast(fit as T.EtsFit, steps).mean
    for (let h = 0; h < steps; h++) expect(Math.abs((sim.mean[h] ?? 0) - (mean[h] ?? 0))).toBeLessThan(0.25)
  })
})

describe('ETS prediction intervals', () => {
  const mnn = models[0]?.[1] as (typeof models)[number][1]
  it('widen with the horizon and use the closed form where there is one', () => {
    const r = T.etsPredictionIntervals(mnn, 6, { levels: [0.8, 0.95] })
    expect(r.method).toBe('analytic')
    const w = (i: number, h: number) => (r.intervals[i]?.upper[h] ?? 0) - (r.intervals[i]?.lower[h] ?? 0)
    for (let h = 1; h < 6; h++) expect(w(1, h)).toBeGreaterThan(w(1, h - 1))
    expect(w(1, 0)).toBeGreaterThan(w(0, 0))
  })
  it('falls back to simulation for a multiplicative season and refuses a closed form it does not have', () => {
    const mnm = fitOf(
      { error: 'mul', trend: 'none', seasonal: 'mul', period: 4 },
      { alpha: 0.3, gamma: 0.2, level0: 50, seasonal0: [1.1, 0.9, 0.8, 1.2] },
      0.01,
      { level: 50, trend: 0, seasonal: [1.1, 0.9, 0.8, 1.2] },
    )
    expect(() => T.etsPredictionIntervals(mnm, 4, { method: 'analytic' })).toThrow()
    expect(() => T.etsPredictionIntervals(mnm, 4)).toThrow(/key/)
    const r = T.etsPredictionIntervals(mnm, 4, { key: key(1), paths: 4000, levels: [0.9] })
    expect(r.method).toBe('simulation')
    // One step ahead the mean is the point forecast level * season, and the 90% band is about +-1.645 sigma wide.
    expect(r.mean[0]).toBeCloseTo(55, 0)
    const lo = r.intervals[0]?.lower[0] ?? 0
    const hi = r.intervals[0]?.upper[0] ?? 0
    expect((hi - lo) / 2 / (0.1 * 55)).toBeCloseTo(1.645, 1)
  })
})
