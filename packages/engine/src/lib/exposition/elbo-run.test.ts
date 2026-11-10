import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { normal } from '../dists/index.ts'
import { observe, sample } from '../probprog/index.ts'
import { pathfinderMethod, sviMethod } from '../probprog/inference/index.ts'
import { distributionOracle } from '../probprog/validation/oracle.ts'
import { Log, session } from '../trace/index.ts'
import { elboRun, trailingMean } from './elbo-run.ts'
import { elboRunFromTape } from './elbo-run-tape.ts'

// m ~ N(0, 1), y ~ N(m, 1) with y = 2: the posterior is N(1, 1/2).
const model = (): void => {
  const m = sample('m', normal(0, 1))
  observe('y', normal(m, 1), 2)
}
const truth = { m: distributionOracle('m', normal(1, Math.SQRT1_2)) }

const method = sviMethod(model)
const s = session(method, {
  steps: 300,
  seed: 3,
  mode: 'eager',
  config: { learningRate: 0.05 },
  logs: { 'svi/elbo': Log(), 'svi/smoothed': Log(), 'svi/params/*': Log(), 'svi/snapshot': Log() },
})
const tape = s.run()

describe('trailingMean', () => {
  it('is the mean of the last window values, fewer at the start', () => {
    expect(trailingMean([1, 3, 5, 7], 2)).toEqual([1, 2, 4, 6])
    expect(trailingMean([1, 3, 5, 7], 1)).toEqual([1, 3, 5, 7])
    expect(trailingMean([2, 4], 10)).toEqual([2, 3])
  })
})

describe('elboRunFromTape of a stochastic variational run', () => {
  const r = elboRunFromTape({ run: tape }, { truth })

  it('reads the bound, its smoothing, the guide parameters and the beliefs of the run', () => {
    expect(r.kind).toBe('elboRun')
    expect([r.method, r.status]).toEqual(['svi', 'completed'])
    expect(r.elbo.steps.length).toBeGreaterThan(100)
    // The ELBO rises from a poor start to the neighbourhood of the log evidence, log N(2; 0, 2) = -2.27.
    expect(r.elbo.values.at(-1) as number).toBeGreaterThan(r.elbo.values[0] as number)
    expect(Math.abs((r.smoothed?.values.at(-1) as number) + 2.27)).toBeLessThan(0.5)
    expect(r.parameters.map((p) => [p.name, p.role, p.site])).toEqual([
      ['m_auto_loc', 'loc', 'm'],
      ['m_auto_scale', 'scale', 'm'],
    ])
    // The location moves to the posterior mean 1.
    expect(Math.abs((r.parameters[0]?.values.at(-1) as number) - 1)).toBeLessThan(0.25)
    const belief = r.beliefs.find((b) => b.name === 'm')
    expect(belief?.steps.length).toBeGreaterThan(1)
    expect(belief?.sd.length).toBe(belief?.values.length)
    expect(Math.abs((belief?.values.at(-1) as number) - 1)).toBeLessThan(0.25)
  })

  it('lays the final belief over the truth, as a normal against the oracle on a grid', () => {
    expect(r.final?.oracle).toBe('truth')
    const q = r.final?.quantities[0]
    expect(q?.truth?.x).toHaveLength(160)
    expect(q?.methods[0]?.normal?.mean).toBeCloseTo(r.beliefs[0]?.values.at(-1) ?? Number.NaN, 12)
    expect(q?.truth?.mean).toBeCloseTo(1, 12)
    expect(elboRunFromTape({ run: tape }).final).toBeUndefined()
  })

  it('serves a run in progress: the events so far give a running record, and more events extend it', () => {
    const half = tape.events.slice(0, Math.floor(tape.events.length / 2))
    const early = elboRunFromTape({ run: { events: half } })
    expect(early.status).toBe('running')
    expect(early.elbo.steps.length).toBeLessThan(r.elbo.steps.length)
    expect(early.elbo.steps.at(-1) as number).toBeLessThan(r.elbo.steps.at(-1) as number)
    expect(elboRunFromTape({ run: { events: [] } }).elbo.steps).toEqual([])
  })

  it('reads anytime snapshots alone: beliefs with no bound', () => {
    const snap = { summary: [{ site: 'm', name: 'm', mean: 0.8, sd: 0.7, lower: 0, upper: 1.5 }] }
    const r2 = elboRunFromTape({
      snapshots: [
        { step: 10, snapshot: snap },
        { step: 20, snapshot: { summary: [{ site: 'm', name: 'm', mean: 0.95, sd: 0.72, lower: 0, upper: 1.5 }] } },
      ],
    })
    expect(r2.elbo.steps).toEqual([])
    expect(r2.beliefs[0]).toMatchObject({
      name: 'm',
      steps: [10, 20],
      values: [0.8, 0.95],
      sd: [0.7, 0.72],
      lower: [0, 0],
      upper: [1.5, 1.5],
    })
  })

  it('thins every series to the points asked for, keeping the last, and smooths with a window of its own', () => {
    const thin = elboRunFromTape({ run: tape }, { maxPoints: 20, smoothing: 10 })
    expect(thin.elbo.steps).toHaveLength(20)
    expect(thin.elbo.steps.at(-1)).toBe(r.elbo.steps.at(-1))
    expect(thin.smoothing).toBe(10)
    expect(elboRunFromTape({ run: tape }, { smoothing: 0 }).smoothed).toBeUndefined()
  })
})

describe('elboRunFromTape with the diagnostics of the estimator', () => {
  it('takes the loss for the bound when only it is logged, the gradient norm and variance, and the other scalars', () => {
    const metric = (step: number, name: string, value: number) => ({ type: 'metric', step, name, value })
    const events = [1, 2, 3, 4].flatMap((t) => [
      metric(t, 'svi/loss', 10 - t),
      metric(t, 'svi/gradNorm', 4 / t),
      ...(t % 2 === 0 ? [metric(t, 'svi/gradVariance', 8 / t)] : []),
      metric(t, 'svi/diag/baseline.c', 0.1 * t),
    ])
    const r = elboRunFromTape({ run: { events } })
    expect(r.elbo.values).toEqual([-9, -8, -7, -6])
    expect(r.gradNorm?.values).toEqual([4, 2, 4 / 3, 1])
    expect(r.gradVariance).toEqual({ steps: [2, 4], values: [4, 2] })
    expect(r.diagnostics.map((d) => d.name)).toEqual(['baseline.c'])
  })

  it('reads a vector parameter as one trajectory per element', () => {
    const tensor = (step: number, name: string, values: number[]) => ({
      type: 'tensor',
      step,
      name,
      shape: [values.length],
      values,
    })
    const events = [1, 2].flatMap((t) => [
      tensor(t, 'svi/params/w_auto_loc', [t, 2 * t, 3 * t]),
      tensor(t, 'svi/params/auto_scale', [0.5 / t]),
    ])
    const r = elboRunFromTape({ run: { events } })
    expect(r.parameters.map((p) => [p.name, p.role, p.site])).toEqual([
      ['w_auto_loc[0]', 'loc', 'w'],
      ['w_auto_loc[1]', 'loc', 'w'],
      ['w_auto_loc[2]', 'loc', 'w'],
      ['auto_scale', 'scale', undefined],
    ])
    expect(r.parameters[2]?.values).toEqual([3, 6])
    expect(elboRunFromTape({ run: { events } }, { maxParameters: 2 }).parameters).toHaveLength(2)
  })
})

describe('elboRunFromTape of Pathfinder', () => {
  it('reads the best ELBO, the paths, the iterations, the Pareto k-hat and the effective sample size', () => {
    const m = pathfinderMethod(model)
    const ps = session(m, {
      steps: 60,
      mode: 'eager',
      config: { maxIterations: 10 },
      logs: {
        'pathfinder/elbo': Log(),
        'pathfinder/iterate': Log(),
        'pathfinder/path/*': Log(),
        'pathfinder/paretoK': Log(),
        'pathfinder/ess': Log(),
      },
    })
    const r = elboRunFromTape({ run: ps.run() })
    expect(r.method).toBe('pathfinder')
    expect(r.elbo.steps.length).toBeGreaterThan(1)
    expect(r.paths.length).toBeGreaterThan(0)
    expect(r.paths[0]?.name).toBe('path 0')
    expect(r.diagnostics[0]?.name).toBe('iterate')
    expect(Number.isFinite(r.paretoK)).toBe(true)
    expect(r.ess).toBeGreaterThan(1)
  })
})

describe('elboRun', () => {
  const base = { status: 'running', diagnostics: [], paths: [], parameters: [], beliefs: [] } as const

  it('checks that steps ascend and match the values', () => {
    expect(() => elboRun({ ...base, elbo: { steps: [1, 2], values: [1] } })).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
    expect(() => elboRun({ ...base, elbo: { steps: [2, 1], values: [1, 2] } })).toThrow(OmaiError)
    expect(() =>
      elboRun({ ...base, elbo: { steps: [], values: [] }, beliefs: [{ name: 'a', steps: [1], values: [1], sd: [] }] }),
    ).toThrow(expect.objectContaining({ code: 'shape' }))
  })
})
