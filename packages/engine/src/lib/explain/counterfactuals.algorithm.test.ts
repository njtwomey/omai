import { describe, expect, it } from 'vitest'
import { fromStorage, matmul, sigmoid, type Value } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import { lambdaSchedules } from './counterfactual-options.ts'
import { counterfactualContract, wachterAlgorithm, wachterCounterfactual } from './counterfactuals.ts'

const w = fromStorage(Float64Array.from([1.5, -1, 0.2]), [3, 1], 'float64')
const proba = (v: Value): Value => sigmoid(matmul(v, w))
const settings = { target: 0.7, tolerance: 0.01, scales: [1, 1, 1] }

describe('wachterAlgorithm', () => {
  it('is wachterCounterfactual: the numbers the function gave before it was an algorithm', () => {
    const a = wachterCounterfactual(proba, [-1, 1, 0], settings)
    expect(a.x[0]).toBeCloseTo(1.2074461364578641, 12)
    expect(a.x.slice(1)).toEqual([1, 0])
    expect(a.prediction).toBeCloseTo(0.6923585987544327, 12)
    expect(a.distance).toBeCloseTo(2.207446136457864, 12)
    const b = wachterCounterfactual(proba, [-1, 1, 0], { target: 0.9, tolerance: 0.001, lambda: 0.05 })
    expect(b.x[0]).toBeCloseTo(2.127328004104453, 12)
    expect(b.prediction).toBeCloseTo(0.8994376685057257, 12)
    expect(b.valid).toBe(true)
  })

  it('resumes from a JSON checkpoint to the same end', () => {
    const alg = wachterAlgorithm(proba, [-1, 1, 0], settings)
    const whole = runAlgorithm(alg, { maxSteps: 4000 })
    const half = runAlgorithm(alg, { maxSteps: 7 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 4000, start: { state: stored, step: half.steps, serialised: true } })
    const json = (v: unknown): unknown => JSON.parse(JSON.stringify(stateToJson(v)))
    expect(json(rest.state)).toEqual(json(whole.state))
  })

  it('logs its paths with the right shapes, and the target is tested only at the end of a round', () => {
    const rec = recorder()
    const run = runAlgorithm(wachterAlgorithm(proba, [-1, 1, 0], settings), {
      maxSteps: 4000,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    for (const p of ['cf/prediction', 'cf/loss', 'cf/distance', 'cf/validity', 'cf/lambda', 'cf/round'])
      expect(metric(p)).toHaveLength(run.steps)
    const points = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'cf/x' ? [e] : []))
    expect(points[0]?.shape).toEqual([3])
    expect(Object.keys(counterfactualContract)).toContain('cf/validity')
    expect(metric('cf/validity').at(-1)).toBe(1)
    // lambda only grows, and only between rounds.
    const lambda = metric('cf/lambda')
    expect(lambda.every((l, i) => i === 0 || l >= (lambda[i - 1] as number))).toBe(true)
  })

  it('keeps immutable features fixed at every step', () => {
    const alg = wachterAlgorithm(proba, [-1, 1, 0], { ...settings, immutable: [0] })
    const s = runAlgorithm(alg, { maxSteps: 4000 }).state
    expect(s.point[0]).toBe(-1)
    expect(Math.abs(s.value - 0.7)).toBeLessThanOrEqual(0.01)
  })

  it('runs with each lambda schedule of the table', () => {
    for (const kind of lambdaSchedules.names) {
      const alg = wachterAlgorithm(proba, [-1, 1, 0], {
        ...settings,
        maxRounds: 6,
        maxIterations: 50,
        lambdaSchedule: { kind },
      })
      const s = runAlgorithm(alg, { maxSteps: 400 }).state
      expect(Number.isFinite(s.loss)).toBe(true)
      expect(s.round).toBeGreaterThan(0)
    }
    expect(lambdaSchedules.make({ kind: 'additive', increment: 2 })(1, 0)).toBe(3)
    expect(lambdaSchedules.make({ kind: 'constant' })(1, 3)).toBe(1)
  })
})
