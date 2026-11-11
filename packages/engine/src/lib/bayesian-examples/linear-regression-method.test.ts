import { describe, expect, it } from 'vitest'
import { type DictSpace, defaults, type UnionSpace } from '../space/index.ts'
import { session } from '../trace/index.ts'
import {
  linearExample,
  linearRegressionMethod,
  linearRegressionProblem,
  linearRegressionSteps,
} from './linear-regression-method.ts'

const options = (linearRegressionMethod.config.spaces['inference'] as UnionSpace).options
const configOf = (kind: string, extra: Record<string, unknown> = {}, over: Record<string, unknown> = {}) => ({
  ...(defaults(linearRegressionMethod.config) as Record<string, unknown>),
  ...over,
  inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Record<string, unknown>), ...extra },
})
const run = (config: Record<string, unknown>) =>
  session(linearRegressionMethod, {
    config: config as never,
    steps: linearRegressionSteps(config as never),
    mode: 'eager',
  })

describe('linearRegressionMethod', () => {
  it('generates the same rows with a prefix property, and the truth is padded to the fitted degree', () => {
    const base = {
      truth: 'quadratic',
      inputs: 'offset',
      noise: 0.3,
      dataSeed: 4,
      degree: 4,
      priorScale: 3,
      noiseModel: 'known',
      sigmaScale: 1,
    }
    const long = linearRegressionProblem({ ...base, rows: 30 })
    const short = linearRegressionProblem({ ...base, rows: 10 })
    expect(long.x.slice(0, 10)).toEqual(short.x)
    expect(long.y.slice(0, 10)).toEqual(short.y)
    expect(long.truth).toEqual([1, 2, -1.5, 0, 0])
    expect(long.names).toEqual(['intercept', 'slope', 'x^2', 'x^3', 'x^4'])
    expect(Math.min(...long.x)).toBeGreaterThanOrEqual(1)
  })

  it('offers the exact linear-Gaussian method and the samplers, with a reason for each that does not apply', () => {
    const offer = linearExample.offer(defaults(linearRegressionMethod.config) as never)
    for (const kind of ['nuts', 'laplace', 'svi']) expect(offer.applicable).toContain(kind)
    expect(offer.applicable.some((k) => k === 'gaussian' || k === 'conjugate')).toBe(true)
    expect(Object.keys(offer.unavailable).length).toBeGreaterThan(0)
  })

  it('with known noise a sampler is checked against the exact normal posterior', () => {
    const s = run(configOf('nuts', { numChains: 2, numWarmup: 100, numDraws: 300 }, { rows: 25 }))
    const c = s.probe(s.t, 'linreg/check') as { against: string; maxMeanError: number; medianSdRatio: number }
    expect(c.against).toBe('exact')
    expect(c.maxMeanError).toBeLessThan(0.5)
    expect(c.medianSdRatio).toBeGreaterThan(0.7)
    expect(c.medianSdRatio).toBeLessThan(1.3)
  })

  it('with learned noise the check needs a reference, and the reference is another run of the same problem', () => {
    const over = { rows: 25, noiseModel: 'learned' }
    const a = run(configOf('laplace', {}, over))
    expect(a.probe(a.t, 'linreg/check')).toBeNull()
    const ref = run(configOf('nuts', { numChains: 2, numWarmup: 100, numDraws: 300 }, over))
    const draws = ref.probe(ref.t, 'linreg/draws') as { names: string[] }
    expect(draws.names.at(-1)).toBe('sigma')
    const c = a.probe(a.t, 'linreg/check', { reference: draws } as never) as { against: string; maxMeanError: number }
    expect(c.against).toBe('reference')
    expect(c.maxMeanError).toBeLessThan(2)
  })

  it('shows the fit on the rows seen, with a band that narrows as rows accumulate, and the correlated coefficients', () => {
    const over = { rows: 60, inputs: 'offset' }
    const s = run(configOf('nuts', { numChains: 2, numWarmup: 100, numDraws: 200 }, over))
    const width = (seen: number): number => {
      const f = s.probe(s.t, 'linreg/fit', { seen }) as { lo: number[]; hi: number[]; seen: number }
      expect(f.seen).toBe(seen)
      return (f.hi[10] ?? 0) - (f.lo[10] ?? 0)
    }
    expect(width(60)).toBeLessThan(width(5))
    const pairs = s.probe(s.t, 'linreg/pairs') as {
      kind: string
      methods: { frames: { correlation: number[][] } }[]
    }
    expect(pairs.kind).toBe('posteriorPairs')
    const r = pairs.methods[0]?.frames.correlation.at(-1)?.[0] ?? 0
    expect(Math.abs(r)).toBeGreaterThan(0.5)
  })

  it('the sparse truth with a spike and slab prior finds the non-zero coefficients, and the noise can be a gamma precision that message passing accepts', () => {
    const over = { truth: 'sparse', rows: 40, features: 8, active: 2, prior: 'spikeSlab' }
    const s = run(configOf('nuts', { numChains: 2, numWarmup: 100, numDraws: 200 }, over))
    const c = s.probe(s.t, 'linreg/coefficients') as {
      names: string[]
      score: number[]
      selection: { found: number; falseAlarms: number; trulyActive: number }
    }
    expect(c.names.length).toBe(8)
    expect(c.selection).toMatchObject({ found: 2, falseAlarms: 0, trulyActive: 2 })
    expect(c.score[0]).toBeGreaterThan(0.9)
    expect(c.score[5]).toBeLessThan(0.3)
    expect(s.probe(s.t, 'linreg/fit')).toBeNull()
    const precision = linearExample.offer({
      ...(defaults(linearRegressionMethod.config) as Record<string, unknown>),
      noiseModel: 'precision',
    } as never)
    for (const kind of ['ep', 'vmp', 'nuts']) expect(precision.applicable).toContain(kind)
    const halfNormal = linearExample.offer({
      ...(defaults(linearRegressionMethod.config) as Record<string, unknown>),
      noiseModel: 'learned',
    } as never)
    expect(halfNormal.applicable).not.toContain('ep')
  })

  it('EP agrees with a long NUTS run when the noise is a gamma precision', () => {
    const over = { rows: 30, inputs: 'offset', noiseModel: 'precision' }
    const ref = run(configOf('nuts', { numChains: 2, numWarmup: 150, numDraws: 300 }, over))
    const draws = ref.probe(ref.t, 'linreg/draws') as { names: string[] }
    expect(draws.names.at(-1)).toBe('tau')
    const ep = run(configOf('ep', {}, over))
    const c = ep.probe(ep.t, 'linreg/check', { reference: draws } as never) as { maxMeanError: number }
    expect(c.maxMeanError).toBeLessThan(0.5)
  })

  it('EP is offered for the spike and slab and ARD priors and agrees with a long NUTS run', () => {
    for (const prior of ['spikeSlab', 'ard']) {
      const over = { truth: 'sparse', rows: 40, features: 6, active: 2, prior }
      expect(
        linearExample.offer({
          ...(defaults(linearRegressionMethod.config) as Record<string, unknown>),
          ...over,
        } as never).applicable,
      ).toContain('ep')
      const ref = run(configOf('nuts', { numChains: 2, numWarmup: 150, numDraws: 300 }, over))
      const draws = ref.probe(ref.t, 'linreg/draws') as { names: string[] }
      const ep = run(configOf('ep', {}, over))
      const c = ep.probe(ep.t, 'linreg/check', { reference: draws } as never) as { maxMeanError: number }
      expect(c.maxMeanError).toBeLessThan(0.5)
    }
  })
})
