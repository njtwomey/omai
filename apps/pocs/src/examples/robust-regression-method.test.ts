import { type DictSpace, defaults, type UnionSpace } from 'omai-engine/space'
import { session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import {
  robustExample,
  robustRegressionMethod,
  robustRegressionProblem,
  robustRegressionSteps,
} from './robust-regression-method.ts'

const options = (robustRegressionMethod.config.spaces['inference'] as UnionSpace).options
const configOf = (kind: string, extra: Record<string, unknown> = {}, over: Record<string, unknown> = {}) => ({
  ...(defaults(robustRegressionMethod.config) as Record<string, unknown>),
  ...over,
  inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Record<string, unknown>), ...extra },
})
const run = (config: Record<string, unknown>) =>
  session(robustRegressionMethod, {
    config: config as never,
    steps: robustRegressionSteps(config as never),
    mode: 'eager',
  })
const NUTS = { numChains: 2, numWarmup: 100, numDraws: 200 }

describe('robustRegressionMethod', () => {
  it('marks the outliers, keeps the prefix property, and puts leverage points at the ends', () => {
    const base = defaults(robustRegressionMethod.config) as Record<string, unknown>
    const long = robustRegressionProblem({ ...base, rows: 60, outliers: 0.3 } as never)
    const short = robustRegressionProblem({ ...base, rows: 20, outliers: 0.3 } as never)
    expect(long.y.slice(0, 20)).toEqual(short.y)
    expect(long.outlier.slice(0, 20)).toEqual(short.outlier)
    expect(long.outlier.filter(Boolean).length).toBeGreaterThan(5)
    const lev = robustRegressionProblem({ ...base, rows: 60, outliers: 0.3, placement: 'leverage' } as never)
    const ends = lev.x.filter((_, i) => lev.outlier[i]).every((x) => Math.abs(x) > 0.79)
    expect(ends).toBe(true)
  })

  it('a Gaussian fit is dragged by the outliers and a Student-t fit is not, the degrees of freedom are learned small', () => {
    const over = { rows: 40, outliers: 0.2, outlierSize: 10, placement: 'leverage' }
    const g = run(configOf('nuts', NUTS, { ...over, likelihood: 'gaussian' }))
    const t = run(configOf('nuts', NUTS, { ...over, likelihood: 'studentT' }))
    const eg = (g.probe(g.t, 'robust/fit') as { errors: { method: number; inliers: number } }).errors
    const et = (t.probe(t.t, 'robust/fit') as { errors: { method: number; inliers: number } }).errors
    expect(eg.method).toBeGreaterThan(2 * et.method)
    const df = t.probe(t.t, 'robust/df') as { draws: number[] }
    const mean = df.draws.reduce((a, b) => a + b, 0) / df.draws.length
    expect(mean).toBeLessThan(5)
    expect(g.probe(g.t, 'robust/df')).toBeNull()
  })

  it('every likelihood runs, and the mixture and Laplace fits stay near the truth', () => {
    for (const likelihood of ['laplace', 'mixture']) {
      const s = run(configOf('nuts', NUTS, { likelihood, rows: 40, outliers: 0.2, outlierSize: 10 }))
      const f = s.probe(s.t, 'robust/fit') as { errors: { method: number; gaussian: number } }
      expect(f.errors.method).toBeLessThan(f.errors.gaussian)
    }
  })

  it('with a Gaussian likelihood and the noise known the exact posterior is the reference', () => {
    const s = run(configOf('nuts', NUTS, { likelihood: 'gaussian', noiseModel: 'known', rows: 25 }))
    const c = s.probe(s.t, 'robust/check') as { against: string; maxMeanError: number }
    expect(c.against).toBe('exact')
    expect(c.maxMeanError).toBeLessThan(0.5)
  })

  it('a heavy-tailed run is checked against a reference run of the same model', () => {
    const over = { rows: 25 }
    const ref = run(configOf('nuts', { numChains: 2, numWarmup: 150, numDraws: 300 }, over))
    const draws = ref.probe(ref.t, 'robust/draws') as { names: string[] }
    expect(draws.names.at(-1)).toBe('nu')
    const a = run(configOf('laplace', {}, over))
    expect(a.probe(a.t, 'robust/check')).toBeNull()
    const b = run(configOf('nuts', NUTS, over))
    const c = b.probe(b.t, 'robust/check', { reference: draws } as never) as { against: string }
    expect(c.against).toBe('reference')
  })

  it('offers the samplers and variational methods and says why the others do not apply', () => {
    const offer = robustExample.offer(defaults(robustRegressionMethod.config) as never)
    for (const kind of ['nuts', 'svi', 'laplace']) expect(offer.applicable).toContain(kind)
    expect(Object.keys(offer.unavailable).length).toBeGreaterThan(0)
  })

  it('EP is offered for the Student-t, Laplace and mixture likelihoods with a known scale, and agrees with a long NUTS run', () => {
    const base = defaults(robustRegressionMethod.config) as Record<string, unknown>
    for (const likelihood of ['studentT', 'laplace', 'mixture']) {
      const over = { likelihood, dfModel: 'fixed', df: 3, noiseModel: 'known', rows: 30, outliers: 0.2 }
      expect(robustExample.offer({ ...base, ...over } as never).applicable).toContain('ep')
      const ref = run(configOf('nuts', { numChains: 2, numWarmup: 150, numDraws: 300 }, over))
      const draws = ref.probe(ref.t, 'robust/draws') as { names: string[] }
      const ep = run(configOf('ep', {}, over))
      const c = ep.probe(ep.t, 'robust/check', { reference: draws } as never) as { maxMeanError: number }
      expect(c.maxMeanError).toBeLessThan(0.5)
    }
    // An unknown scale and learned degrees of freedom have no message rule: EP is refused with the reason.
    const learned = robustExample.offer({ ...base, likelihood: 'studentT', noiseModel: 'learned' } as never)
    expect(learned.applicable).not.toContain('ep')
    expect(learned.unavailable['ep']).toBeDefined()
  })
})
