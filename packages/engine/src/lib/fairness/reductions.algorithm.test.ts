import { describe, expect, it } from 'vitest'
import { defineEstimator, matrixView } from '../learn/index.ts'
import { tensor } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  exponentiatedGradientAlgorithm,
  exponentiatedGradientReduction,
  gridReduction,
  gridReductionAlgorithm,
  reductionConstraints,
} from './index.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const],
  detail: 'full' as const,
}
const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

// A threshold classifier on the first feature, at the weighted mean of the rows it is given.
const stump = defineEstimator(
  {
    name: 'stump',
    kind: 'classifier',
    fit: (data) => {
      const m = matrixView(data.x, 'stump')
      let s = 0
      for (let i = 0; i < m.n; i++) s += m.data[i * m.d] ?? 0
      return { t: s / m.n }
    },
    methods: (p) => ({
      classes: [0, 1],
      predict: (x) => {
        const m = matrixView(x, 'stump')
        return tensor(Array.from({ length: m.n }, (_, i) => ((m.data[i * m.d] ?? 0) > p.t ? 1 : 0)))
      },
    }),
  },
  {},
)
const x = [[0], [1], [2], [3], [0.5], [1.5], [2.5], [3.5], [0.2], [1.2], [2.2], [3.2]]
const data = {
  x,
  labels: [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
  groups: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
}

describe('exponentiatedGradientAlgorithm', () => {
  const options = { iterations: 8 }

  it('plays one round of the game per step, and the function is that run', () => {
    const rec = recorder()
    const alg = exponentiatedGradientAlgorithm(stump, data, options)
    const run = runAlgorithm(alg, { maxSteps: 8, tracer: rec, policy })
    const r = exponentiatedGradientReduction(stump, data, options)
    expect(run.state.steps).toHaveLength(r.steps.length)
    for (const [i, step] of r.steps.entries()) {
      expect(run.state.steps[i]?.gap).toBeCloseTo(step.gap, 12)
      expect(run.state.steps[i]?.averageError).toBeCloseTo(step.averageError, 12)
    }
    expect(metricsOf(rec, 'eg/gap')).toHaveLength(run.steps)
    expect(metricsOf(rec, 'eg/mixture').at(-1)).toBe(run.state.hs.length)
    expect(metricsOf(rec, 'eg/classifier')[0]).toBe(0)
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
  })

  it('resumes from a JSON checkpoint to the same game', () => {
    const alg = exponentiatedGradientAlgorithm(stump, data, options)
    const whole = runAlgorithm(alg, { maxSteps: 8 })
    const half = runAlgorithm(alg, { maxSteps: 3 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(exponentiatedGradientAlgorithm(stump, data, options), {
      maxSteps: 5,
      start: { state: stored, step: 3, serialised: true },
    })
    expect(rest.steps).toBe(whole.steps)
    expect(rest.state.hs.length).toBe(whole.state.hs.length)
    for (const [i, g] of whole.state.gaps.entries()) expect(rest.state.gaps[i]).toBeCloseTo(g, 12)
    for (const [c, v] of whole.state.theta.entries()) expect(rest.state.theta[c]).toBeCloseTo(v, 12)
  })

  it('refuses a non-positive epsilon', () => {
    expect(() => exponentiatedGradientAlgorithm(stump, data, { epsilon: 0 })).toThrow(/epsilon/)
  })
})

describe('gridReductionAlgorithm', () => {
  it('fits one grid point per step and picks the point of least trade-off loss', () => {
    const rec = recorder()
    const run = runAlgorithm(gridReductionAlgorithm(stump, data, { gridSize: 5 }), {
      maxSteps: 10,
      tracer: rec,
      policy,
    })
    expect(run.steps).toBe(5)
    expect(metricsOf(rec, 'grid/point')).toEqual([0, 1, 2, 3, 4])
    const points = gridReduction(stump, data, { gridSize: 5 })
    expect(points).toHaveLength(5)
    expect(points.filter((p) => p.best)).toHaveLength(1)
    const losses = run.state.points.map((p) => 0.5 * p.error + 0.5 * p.violation)
    expect(points.findIndex((p) => p.best)).toBe(losses.indexOf(Math.min(...losses)))
    expect(metricsOf(rec, 'grid/best').at(-1)).toBe(losses.indexOf(Math.min(...losses)))
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = gridReductionAlgorithm(stump, data, { gridSize: 5 })
    const whole = runAlgorithm(alg, { maxSteps: 10 }).state
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 8, start: { state: stored, step: 2, serialised: true } }).state
    expect(rest).toEqual(whole)
  })
})

describe('reductionConstraints', () => {
  it('names the constraints and says which need labels', () => {
    expect(reductionConstraints.names).toEqual([
      'demographicParity',
      'equalisedOdds',
      'truePositiveRateParity',
      'falsePositiveRateParity',
    ])
    expect(reductionConstraints.make({ kind: 'equalisedOdds' })).toBe('equalisedOdds')
    expect(reductionConstraints.where((p) => !p.conditionsOnLabel)).toEqual(['demographicParity'])
    const r = gridReduction(stump, data, {
      gridSize: 3,
      constraint: reductionConstraints.make({ kind: 'equalisedOdds' }),
    })
    expect(r).toHaveLength(3)
  })
})
