import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import { gainSchedule, matchingParameters, mracAlgorithm } from './adaptive.ts'
import { lqr } from './optimal.ts'

const plant = { a: 1, k: 2 }
const model = { am: 3, km: 3 }
const rich = (t: number): number => Math.sin(t) + Math.sin(3.7 * t)
const run = (law: 'mit' | 'lyapunov', gamma: number, reference: (t: number) => number, steps = 6000) => {
  const rec = recorder()
  const { state } = runAlgorithm(mracAlgorithm({ plant, model, law, gamma, reference, dt: 0.005, steps }), {
    maxSteps: steps + 1,
    tracer: rec,
  })
  return { state, run: rec.run() }
}

describe('model-reference adaptive control', () => {
  it('has the matching parameters that make the loop the model: theta1 = km / k, theta2 = (am - a) / k', () => {
    expect(matchingParameters(plant, model)).toEqual({ theta1: 1.5, theta2: 1 })
    // With them, the closed loop y\' = -(a + k theta2) y + k theta1 r is the model exactly.
    const t = matchingParameters(plant, model)
    expect(plant.a + plant.k * t.theta2).toBeCloseTo(model.am, 14)
    expect(plant.k * t.theta1).toBeCloseTo(model.km, 14)
  })

  it('drives the error to zero and the parameters to the matching ones when the reference is rich (Lyapunov law)', () => {
    const { state, run: r } = run('lyapunov', 5, rich, 12000)
    const truth = matchingParameters(plant, model)
    expect(state.y[2]).toBeCloseTo(truth.theta1, 1)
    expect(state.y[3]).toBeCloseTo(truth.theta2, 1)
    const error = metricSeries(r, 'adaptation/error').values
    const early = Math.max(...error.slice(0, 1000).map(Math.abs))
    const late = Math.max(...error.slice(-1000).map(Math.abs))
    expect(late).toBeLessThan(0.05 * early)
    const parameterError = metricSeries(r, 'adaptation/parameterError').values
    expect(parameterError.at(-1)).toBeLessThan(0.1)
    expect(parameterError.at(-1)).toBeLessThan((parameterError[0] as number) / 10)
  })

  it('decreases the Lyapunov function at every step, whatever the adaptation gain', () => {
    for (const gamma of [0.5, 5, 50]) {
      const truth = matchingParameters(plant, model)
      const { run: r } = run('lyapunov', gamma, rich, 4000)
      const e = metricSeries(r, 'adaptation/error').values
      const pe = metricSeries(r, 'adaptation/parameterError').values
      // V = e^2 / 2 + k |theta - theta*|^2 / (2 gamma), V' = -am e^2 <= 0.
      const V = e.map((v, i) => 0.5 * v * v + (plant.k * (pe[i] as number) ** 2) / (2 * gamma))
      expect(truth.theta1).toBe(1.5)
      for (let i = 1; i < V.length; i++)
        expect(V[i] as number).toBeLessThanOrEqual((V[i - 1] as number) * (1 + 1e-6) + 1e-12)
    }
  })

  it('has the error go to zero with a constant reference without the parameters reaching the matching ones (no persistent excitation)', () => {
    const { state, run: r } = run('lyapunov', 5, () => 1, 12000)
    const error = metricSeries(r, 'adaptation/error').values
    expect(Math.abs(error.at(-1) as number)).toBeLessThan(1e-3)
    // One frequency cannot pin down two parameters: only a combination is learned, y matches ym at that frequency.
    const truth = matchingParameters(plant, model)
    const off = Math.hypot((state.y[2] as number) - truth.theta1, (state.y[3] as number) - truth.theta2)
    expect(off).toBeGreaterThan(0.05)
  })

  it('oscillates more as the adaptation gain grows: the MIT parameter overshoots and rings, then settles', () => {
    // A plant whose pole is the model's, so the MIT rule's sensitivity approximation is exact; a constant reference.
    const same = { plant: { a: 2, k: 2 }, model: { am: 2, km: 2 }, reference: () => 1, dt: 0.005, steps: 6000 }
    const behaviour = (law: 'mit' | 'lyapunov', gamma: number) => {
      const rec = recorder()
      runAlgorithm(mracAlgorithm({ ...same, law, gamma }), { maxSteps: 6001, tracer: rec })
      const theta = metricSeries(rec.run(), 'adaptation/parameter').values
      const error = metricSeries(rec.run(), 'adaptation/error').values
      let crossings = 0
      for (let i = 1; i < theta.length; i++)
        if (((theta[i] as number) - 1) * ((theta[i - 1] as number) - 1) < 0) crossings++
      return { overshoot: Math.max(...theta), crossings, final: Math.abs(error.at(-1) as number) }
    }
    const gains = [0.2, 1, 5, 20, 80]
    const mit = gains.map((g) => behaviour('mit', g))
    // No overshoot at a low gain, then more and more, with more and more oscillations.
    expect(mit[0]?.overshoot).toBeLessThan(1.01)
    expect(mit[0]?.crossings).toBe(0)
    for (let i = 1; i < gains.length; i++) {
      expect(mit[i]?.overshoot).toBeGreaterThan(mit[i - 1]?.overshoot as number)
      expect(mit[i]?.crossings).toBeGreaterThan(mit[i - 1]?.crossings as number)
    }
    expect(mit[3]?.overshoot).toBeGreaterThan(1.5)
    // Every one of them still ends with the error gone; and the Lyapunov rule behaves alike (its guarantee is stability, not damping).
    for (const m of mit.slice(1)) expect(m.final).toBeLessThan(1e-6)
    expect(behaviour('lyapunov', 20).final).toBeLessThan(1e-6)
  }, 20000)

  it('logs every path and resumes from a serialised state', () => {
    const rec = recorder()
    const alg = mracAlgorithm({ plant, model, gamma: 1, reference: rich, dt: 0.01, steps: 60 })
    const straight = runAlgorithm(alg, { maxSteps: 100, tracer: rec }).state
    for (const name of [
      'loop/output',
      'loop/reference',
      'adaptation/parameter',
      'adaptation/error',
      'adaptation/gain',
    ]) {
      expect(metricSeries(rec.run(), name).values.length).toBe(60)
    }
    const half = runAlgorithm(alg, { maxSteps: 25 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.y)).toEqual(Array.from(straight.y))
    expect(() => mracAlgorithm({ plant, model, gamma: 0, reference: rich, dt: 0.01, steps: 1 })).toThrow()
  })
})

describe('gain scheduling', () => {
  it('interpolates linearly between operating points and holds beyond them', () => {
    const k = gainSchedule([
      { at: 2, gain: [3, 20] },
      { at: 0, gain: [1, 10] },
      { at: 4, gain: [3, 0] },
    ])
    expect(k(0)).toEqual([1, 10])
    expect(k(1)).toEqual([2, 15])
    expect(k(3)).toEqual([3, 10])
    expect(k(-9)).toEqual([1, 10])
    expect(k(99)).toEqual([3, 0])
    expect(() => gainSchedule([])).toThrow()
    expect(() =>
      gainSchedule([
        { at: 0, gain: [1] },
        { at: 0, gain: [2] },
      ]),
    ).toThrow()
    expect(() =>
      gainSchedule([
        { at: 0, gain: [1] },
        { at: 1, gain: [2, 3] },
      ]),
    ).toThrow()
  })

  it('schedules LQR gains over a plant that changes: the interpolated gain stabilises the plant between the design points', () => {
    // x' = a(v) x + u with a(v) = v (unstable for v > 0): LQR gains at v = 0.5 and 3, interpolated at 1.5.
    const design = (v: number): number[] => toFlat(lqr([[v]], [[1]], [[1]], [[1]]).K)
    const k = gainSchedule([
      { at: 0.5, gain: design(0.5) },
      { at: 3, gain: design(3) },
    ])
    for (const v of [0.5, 1, 1.5, 2.2, 3]) expect(v - (k(v)[0] as number)).toBeLessThan(0)
    expect(k(0.5)).toEqual(design(0.5))
  })
})
