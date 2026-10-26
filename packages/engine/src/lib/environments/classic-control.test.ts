import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { evaluatePolicy, make } from '../rl/index.ts'
import {
  acrobot,
  acrobotTerminated,
  angleNormalize,
  cartPole,
  mountainCar,
  mountainCarContinuous,
  pendulum,
  pendulumDynamics,
} from './classic-control.ts'

describe('classic control', () => {
  it('starts episodes from gymnasium reset distributions', () => {
    for (let i = 0; i < 20; i++) {
      const c = cartPole.create().reset(key(i)).state
      expect(c.every((x) => Math.abs(x) <= 0.05)).toBe(true)
      const m = mountainCar.create().reset(key(i)).state
      expect(m[0] >= -0.6 && m[0] <= -0.4 && m[1] === 0).toBe(true)
      const p = pendulum.create().reset(key(i)).state
      expect(Math.abs(p[0]) <= Math.PI && Math.abs(p[1]) <= 1).toBe(true)
      const a = acrobot.create().reset(key(i)).state
      expect(a.every((x) => Math.abs(x) <= 0.1 && Math.fround(x) === x)).toBe(true)
    }
  })

  it('offers the Sutton-Barto reward for CartPole and a goal velocity for MountainCar', () => {
    const env = cartPole.create({ suttonBartoReward: true })
    expect(env.step([0, 0, 0, 0], 1, key(0)).reward).toBe(0)
    expect(env.step([0, 0, 0.2, 2], 1, key(0)).reward).toBe(-1)
    const fast = mountainCar.create({ goalVelocity: 0.05 })
    expect(fast.step([0.55, 0.01], 1, key(0)).terminated).toBe(false)
    expect(mountainCar.create().step([0.55, 0.01], 1, key(0)).terminated).toBe(true)
    expect(mountainCarContinuous.create().step([0.5, 0.05], [0], key(0)).reward).toBe(100)
  })

  it('wraps pendulum angles and charges the quadratic cost', () => {
    expect(angleNormalize(Math.PI)).toBe(-Math.PI)
    expect(angleNormalize(-3 * Math.PI)).toBeCloseTo(-Math.PI, 12)
    expect(pendulumDynamics([Math.PI, 0], 0).cost).toBeCloseTo(Math.PI ** 2, 12)
    expect(pendulumDynamics([0, 100], 0).state[1]).toBe(8)
    expect(pendulum.create({ g: 0 }).step([0.5, 0], [0], key(0)).state[0]).toBe(0.5)
  })

  it('ends Acrobot when the tip clears the bar, and an energy-pumping policy does so quickly', () => {
    expect(acrobotTerminated([0, 0, 0, 0])).toBe(false)
    const pump = (o: readonly number[]): number => ((o[4] ?? 0) + (o[5] ?? 0) > 0 ? 2 : 0)
    expect(evaluatePolicy(make(acrobot), pump, { episodes: 2, seed: 0 }).meanReturn).toBeGreaterThan(-200)
    // Rocking in the direction of motion gets the mountain car up the hill.
    const rock = (o: readonly number[]): number => ((o[1] ?? 0) >= 0 ? 2 : 0)
    expect(evaluatePolicy(make(mountainCar), rock, { episodes: 2, seed: 0 }).meanReturn).toBeGreaterThan(-200)
  })
})
