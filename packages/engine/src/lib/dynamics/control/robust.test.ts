import { describe, expect, it } from 'vitest'
import { brentq } from '../../numerics/roots/index.ts'
import { hinfNorm, isStable, ss } from '../../systems/index.ts'
import { item, toFlat } from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import { lqr } from './optimal.ts'
import {
  closedLoopFromDisturbance,
  type GeneralisedPlant,
  hinfBisectionAlgorithm,
  hinfStateFeedback,
  hinfSynthesis,
} from './robust.ts'

// x' = a x + b1 w1 + b2 u, z = (c x, u), y = x + w2.
const scalar = (a: number, b1: number, b2: number, c: number): GeneralisedPlant => ({
  a: [[a]],
  b1: [[b1, 0]],
  b2: [[b2]],
  c1: [[c], [0]],
  c2: [[1]],
  d12: [[0], [1]],
  d21: [[0, 1]],
})

// The stabilising Riccati solutions of the scalar problem in closed form.
const xOf = (a: number, b1: number, b2: number, c: number, g: number): number => {
  const kappa = b2 * b2 - (b1 * b1) / (g * g)
  return (a + Math.sqrt(a * a + kappa * c * c)) / kappa
}
const yOf = (a: number, b1: number, c: number, g: number): number => {
  const mu = 1 - (c * c) / (g * g)
  return (a + Math.sqrt(a * a + mu * b1 * b1)) / mu
}

describe('H-infinity state feedback', () => {
  const plant = scalar(1, 1, 1, 1)

  it('solves the Riccati equation (closed form in the scalar case) and gives u = -b2 X x', () => {
    for (const g of [1.5, 2, 5]) {
      const r = hinfStateFeedback(plant, g)
      expect(item(r.X)).toBeCloseTo(xOf(1, 1, 1, 1, g), 9)
      expect(item(r.K)).toBeCloseTo(-xOf(1, 1, 1, 1, g), 9)
    }
  })

  it('tends to the LQR gain as gamma grows', () => {
    const lq = item(lqr([[1]], [[1]], [[1]], [[1]]).K)
    expect(item(hinfStateFeedback(plant, 1e4).K)).toBeCloseTo(-lq, 3)
  })

  it('guarantees the gain from w to z is below gamma', () => {
    for (const g of [1.5, 2, 5]) {
      const K = item(hinfStateFeedback(plant, g).K)
      const closed = ss([[1 + K]], [[1]], [[1], [K]], [[0], [0]])
      expect(isStable(closed)).toBe(true)
      expect(item(hinfNorm(closed).norm)).toBeLessThan(g)
    }
  })
})

describe('H-infinity output feedback', () => {
  const plant = scalar(1, 1, 1, 1)

  it('is achievable exactly when the three conditions hold, and the Riccati solutions are the closed-form ones', () => {
    const good = hinfSynthesis(plant, 3)
    expect(good.feasible).toBe(true)
    expect(item(good.X as never)).toBeCloseTo(xOf(1, 1, 1, 1, 3), 9)
    expect(item(good.Y as never)).toBeCloseTo(yOf(1, 1, 1, 3), 9)
    expect(good.conditions.coupling).toBeCloseTo((xOf(1, 1, 1, 1, 3) * yOf(1, 1, 1, 3)) / 9, 9)
    expect(good.conditions.coupling).toBeLessThan(1)
    const bad = hinfSynthesis(plant, 2)
    expect(bad.feasible).toBe(false)
    expect(bad.conditions.coupling).toBeGreaterThan(1)
    expect(bad.controller).toBeUndefined()
    // Below max(b1 / b2, c) the Riccati equation has no stabilising solution at all.
    expect(hinfSynthesis(plant, 0.5).feasible).toBe(false)
  })

  it('gives a controller that stabilises the loop with a closed-loop gain below gamma', () => {
    for (const g of [2.8, 3, 6, 40]) {
      const k = hinfSynthesis(plant, g).controller
      expect(k).toBeDefined()
      if (k === undefined) continue
      const loop = closedLoopFromDisturbance(plant, k)
      expect(isStable(loop)).toBe(true)
      expect(item(hinfNorm(loop).norm)).toBeLessThan(g)
    }
  })

  it('works for a plant with two states and a vector measurement', () => {
    // A lightly damped oscillator, force disturbance, position measured, control acting on the velocity.
    const p: GeneralisedPlant = {
      a: [
        [0, 1],
        [-2, -0.2],
      ],
      b1: [
        [0, 0],
        [1, 0],
      ],
      b2: [[0], [1]],
      c1: [
        [1, 0],
        [0, 0],
      ],
      c2: [[1, 0]],
      d12: [[0], [1]],
      d21: [[0, 1]],
    }
    const k = hinfSynthesis(p, 20).controller
    expect(k).toBeDefined()
    if (k === undefined) return
    const loop = closedLoopFromDisturbance(p, k)
    expect(isStable(loop)).toBe(true)
    expect(item(hinfNorm(loop).norm)).toBeLessThan(20)
    expect(toFlat(k.a).length).toBe(4)
  })

  it('refuses plants that break the normalisation', () => {
    expect(() => hinfSynthesis({ ...plant, d12: [[0], [2]] }, 3)).toThrow(/D12/)
    expect(() => hinfSynthesis({ ...plant, d21: [[0, 3]] }, 3)).toThrow(/D21/)
    expect(() => hinfSynthesis({ ...plant, c1: [[1], [1]] }, 3)).toThrow(/C1/)
  })
})

describe('the search for the smallest gamma', () => {
  it('lands on the level where X Y = gamma^2, found independently by root-finding on the closed forms', () => {
    const plant = scalar(1, 1, 1, 1)
    const optimum = brentq((g) => xOf(1, 1, 1, 1, g) * yOf(1, 1, 1, g) - g * g, 1.2, 5, { xtol: 1e-13 }).root
    const rec = recorder()
    const { state } = runAlgorithm(hinfBisectionAlgorithm(plant), { maxSteps: 100, tracer: rec })
    expect(state.feasible).toBeGreaterThanOrEqual(optimum)
    expect(state.feasible).toBeCloseTo(optimum, 4)
    expect(state.infeasible).toBeLessThanOrEqual(optimum)
    // The controller's norm cannot be below the optimal level and is below the level at which it was built.
    expect(state.achieved).toBeLessThan(state.feasible)
    expect(state.achieved).toBeGreaterThan(0.5 * optimum)
    const bracket = metricSeries(rec.run(), 'hinf/bracket').values
    for (let i = 1; i < bracket.length; i++) expect(bracket[i] as number).toBeLessThan(bracket[i - 1] as number)
    const feasible = metricSeries(rec.run(), 'hinf/feasible').values
    expect(feasible.some((v) => v === 0)).toBe(true)
    expect(feasible.some((v) => v === 1)).toBe(true)
  })

  it('resumes from a serialised state and refuses an upper level that is not achievable', () => {
    const plant = scalar(1, 1, 1, 1)
    const alg = hinfBisectionAlgorithm(plant)
    const straight = runAlgorithm(alg, { maxSteps: 100 }).state
    const half = runAlgorithm(alg, { maxSteps: 7 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.iteration) === undefined) s = alg.step(s, ctx)
    expect(s.feasible).toBe(straight.feasible)
    expect(() => hinfBisectionAlgorithm(plant, { upper: 1.5 }).init(null as never)).toThrow()
  })
})
