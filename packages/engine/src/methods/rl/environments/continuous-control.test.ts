import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { evaluatePolicy, make } from '../../../lib/rl/index.ts'
import {
  cartDoublePendulum,
  cartDoublePendulumDynamics,
  cartDoublePendulumEnergy,
  cartDoublePendulumParameters,
  doubleIntegrator,
  doubleIntegratorDynamics,
  doubleIntegratorParameters,
  makeCartDoublePendulum,
  makePlanarQuadrotor,
  makePointMass,
  makeReacher2d,
  planarQuadrotor,
  planarQuadrotorDynamics,
  planarQuadrotorParameters,
  pointMass,
  pointMassDynamics,
  pointMassParameters,
  reacher2d,
  reacher2dDynamics,
  reacher2dEnergy,
  reacher2dParameters,
} from './continuous-control.ts'

const iterate = (
  f: (s: readonly number[]) => readonly number[],
  s: readonly number[],
  n: number,
): readonly number[] => {
  let x = s
  for (let i = 0; i < n; i++) x = f(x)
  return x
}

describe('continuous-control environments', () => {
  it('reset within their documented ranges, render scenes and carry time limits', () => {
    for (let i = 0; i < 10; i++) {
      const pm = makePointMass({ goalRange: 0.2 }).reset(key(i)).state
      expect(Math.abs(pm[0] ?? 1) <= 0.29 && Math.abs(pm[4] ?? 1) <= 0.2 && pm[2] === 0).toBe(true)
      const r = makeReacher2d().reset(key(i))
      expect(Math.hypot(r.state[4] ?? 1, r.state[5] ?? 1)).toBeLessThanOrEqual(0.2)
      expect(Math.abs(r.state[0] ?? 1)).toBeLessThanOrEqual(0.1)
      const q = makePlanarQuadrotor().reset(key(i)).state
      expect(q.slice(0, 6).every((x) => Math.abs(x) <= 0.1)).toBe(true)
      const c = makeCartDoublePendulum().reset(key(i)).state
      expect(c.every((x) => Math.abs(x) <= 0.1)).toBe(true)
    }
    const expected = {
      'PointMass-v0': 1000,
      'Reacher2D-v0': 50,
      'DoubleIntegrator-v0': 200,
      'PlanarQuadrotor-v0': 300,
      'CartDoublePendulum-v0': 1000,
    }
    for (const def of [pointMass, reacher2d, doubleIntegrator, planarQuadrotor, cartDoublePendulum]) {
      const env = def.create()
      expect(env.spec.maxEpisodeSteps).toBe(expected[def.id as keyof typeof expected])
      const s = env.reset(key(0)).state as never
      expect(env.render(s).kind).toBe('scene')
      expect(env.actionSpace.kind).toBe('box')
    }
    expect(make(reacher2d).spec.maxEpisodeSteps).toBe(50)
    expect(pointMass.create({ maxEpisodeSteps: 20 }).spec.maxEpisodeSteps).toBe(20)
  })

  it('keep equilibria fixed', () => {
    expect(pointMassDynamics([0.1, -0.1, 0, 0, 0, 0], [0, 0], pointMassParameters())).toEqual([0.1, -0.1, 0, 0, 0, 0])
    expect(doubleIntegratorDynamics([0, 0], [0], doubleIntegratorParameters())).toEqual([0, 0])
    const rp = reacher2dParameters()
    expect(reacher2dDynamics([0.3, -1, 0, 0, 0, 0], [0, 0], rp).slice(0, 4)).toEqual([0.3, -1, 0, 0])
    const qp = planarQuadrotorParameters()
    const hover = (qp.mass * qp.g) / 2
    const s = iterate((x) => planarQuadrotorDynamics(x, [hover, hover], qp), [0.5, 1, 0, 0, 0, 0, 0, 0], 100)
    for (const [i, v] of [0.5, 1, 0, 0, 0, 0].entries()) expect(Math.abs((s[i] ?? 0) - v)).toBeLessThan(1e-12)
    const cp = cartDoublePendulumParameters()
    expect(iterate((x) => cartDoublePendulumDynamics(x, [0], cp), [0.3, 0, 0, 0, 0, 0], 50)).toEqual([
      0.3, 0, 0, 0, 0, 0,
    ])
    // Hanging straight down is an equilibrium too.
    const down = iterate((x) => cartDoublePendulumDynamics(x, [0], cp), [0, Math.PI, Math.PI, 0, 0, 0], 50)
    for (const [i, v] of [0, Math.PI, Math.PI, 0, 0, 0].entries())
      expect(Math.abs((down[i] ?? 0) - v)).toBeLessThan(1e-12)
  })

  it('conserve energy and momentum without actuation or friction', () => {
    // RK4 conserves the energy up to its O(h^4) error: halving the substep cuts the drift more than tenfold.
    const drift = (substeps: number, s0: readonly number[], steps: number): number => {
      const cp = cartDoublePendulumParameters({ swingUp: true, substeps })
      const end = iterate((x) => cartDoublePendulumDynamics(x, [0], cp), s0, steps)
      return Math.abs(cartDoublePendulumEnergy(end, cp) - cartDoublePendulumEnergy(s0, cp))
    }
    const violent = [0, 2, -1, 0.5, 1, -2]
    expect(drift(4, violent, 500)).toBeLessThan(drift(2, violent, 500) / 10)
    expect(drift(2, [0, Math.PI + 0.3, Math.PI - 0.2, 0, 0, 0], 500)).toBeLessThan(1e-6)
    // Horizontal momentum is conserved too (no horizontal external force), up to the integration error.
    const cp = cartDoublePendulumParameters({ swingUp: true })
    const momentum = (s: readonly number[]): number =>
      1.2 * (s[3] ?? 0) + 0.09 * Math.cos(s[1] ?? 0) * (s[4] ?? 0) + 0.03 * Math.cos(s[2] ?? 0) * (s[5] ?? 0)
    const swing = [0, Math.PI + 0.3, Math.PI - 0.2, 0.2, 0, 0]
    const after = iterate((x) => cartDoublePendulumDynamics(x, [0], cp), swing, 500)
    expect(momentum(after)).toBeCloseTo(momentum(swing), 8)
    const rp = reacher2dParameters({ damping: 0 })
    const r0 = [0.4, 1, 3, -2, 0, 0]
    const end = iterate((x) => reacher2dDynamics(x, [0, 0], rp), r0, 200)
    expect(Math.abs(reacher2dEnergy(end, rp) - reacher2dEnergy(r0, rp)) / reacher2dEnergy(r0, rp)).toBeLessThan(1e-8)
    // An undamped point mass coasts at constant velocity; a damped one slows down.
    const free = iterate(
      (x) => pointMassDynamics(x, [0, 0], pointMassParameters({ damping: 0 })),
      [0, 0, 0.1, 0.05, 0, 0],
      10,
    )
    expect(free[2]).toBe(0.1)
    expect(free[0]).toBeCloseTo(0.1 * 0.2, 14)
    const damped = pointMassDynamics([0, 0, 0.1, 0, 0, 0], [0, 0], pointMassParameters())
    expect(damped[2]).toBeCloseTo(0.1 * Math.exp(-0.02 / 0.3), 15)
    // A quadrotor in zero gravity with no thrust drifts and spins at constant rates.
    const qp = planarQuadrotorParameters({ g: 0 })
    const q = iterate((x) => planarQuadrotorDynamics(x, [0, 0], qp), [0, 0, 0, 1, -1, 2, 0, 0], 50)
    expect(q.slice(3, 6)).toEqual([1, -1, 2])
    expect(q[2]).toBeCloseTo(2, 12)
  })

  it('are symmetric under reflection', () => {
    const cp = cartDoublePendulumParameters({ swingUp: true })
    const mirror = (s: readonly number[]): number[] => s.map((v) => -v)
    let a = [0.1, 0.4, -0.3, 0.2, -1, 0.5]
    let b = mirror(a)
    for (let t = 0; t < 100; t++) {
      a = cartDoublePendulumDynamics(a, [3], cp) as number[]
      b = cartDoublePendulumDynamics(b, [-3], cp) as number[]
    }
    expect(b).toEqual(mirror(a))
    // Quadrotor: x -> -x, theta -> -theta and the rotors swap.
    const qp = planarQuadrotorParameters()
    const flip = (s: readonly number[]): number[] => s.map((v, i) => ([1, 4, 7].includes(i) ? v : -v))
    let q = [0.2, 0.1, 0.3, -0.5, 0.2, 1, 0.5, 0.5]
    let r = flip(q)
    for (let t = 0; t < 50; t++) {
      q = planarQuadrotorDynamics(q, [3, 6], qp) as number[]
      r = planarQuadrotorDynamics(r, [6, 3], qp) as number[]
    }
    for (const [i, v] of flip(q).entries()) expect(r[i]).toBeCloseTo(v, 12)
    // Reacher: reflecting the joint angles, velocities and torques.
    const rp = reacher2dParameters()
    let u = [0.4, -0.7, 1, 2, 0, 0]
    let w = mirror(u)
    for (let t = 0; t < 50; t++) {
      u = reacher2dDynamics(u, [0.5, -0.2], rp) as number[]
      w = reacher2dDynamics(w, [-0.5, 0.2], rp) as number[]
    }
    expect(w.map((v) => v + 0)).toEqual(mirror(u).map((v) => v + 0))
    // Point mass: swapping the axes.
    const pp = pointMassParameters()
    const x = pointMassDynamics([0.1, -0.2, 0.05, 0.02, 0, 0], [0.3, -1], pp)
    const y = pointMassDynamics([-0.2, 0.1, 0.02, 0.05, 0, 0], [-1, 0.3], pp)
    expect([y[1], y[0], y[3], y[2]]).toEqual(x.slice(0, 4))
  })

  it('clip actions, stop at walls and end episodes where documented', () => {
    const pp = pointMassParameters()
    const wall = iterate((s) => pointMassDynamics(s, [1, 0], pp), [0.29, 0, 0.1, 0, 0, 0], 20)
    expect(wall[0]).toBe(0.3)
    expect(wall[2]).toBe(0)
    expect(pointMassDynamics([0, 0, 0, 0, 0, 0], [5, 0], pp)).toEqual(pointMassDynamics([0, 0, 0, 0, 0, 0], [1, 0], pp))
    const qp = planarQuadrotorParameters()
    const s = [0, 0, 0, 0, 0, 0, 0, 0]
    expect(planarQuadrotorDynamics(s, [-1, 100], qp)).toEqual(planarQuadrotorDynamics(s, [0, qp.maxThrust], qp))
    const di = doubleIntegrator.create()
    expect(di.step([0, 0], [100], key(0)).state).toEqual(di.step([0, 0], [10], key(0)).state)
    const balance = cartDoublePendulum.create()
    expect(balance.step([0, 0.9, 0.9, 0, 0, 0], [0], key(0)).terminated).toBe(true)
    expect(balance.step([2.5, 0, 0, 0, 0, 0], [0], key(0)).terminated).toBe(true)
    const swing = cartDoublePendulum.create({ swingUp: true })
    expect(swing.step([0, Math.PI, Math.PI, 0, 0, 0], [0], key(0)).terminated).toBe(false)
    // Doing nothing, the double pendulum falls within a second or two.
    const lazy = evaluatePolicy(make(cartDoublePendulum), () => [0], { episodes: 2, seed: 0 })
    expect(Math.max(...lazy.lengths)).toBeLessThan(150)
  })

  it('pay the documented rewards', () => {
    const reach = reacher2d.create()
    const at = reach.step([0, Math.PI / 2, 0, 0, 0.1, 0.11], [0, 0], key(0))
    expect(Math.abs(at.reward)).toBeLessThan(1e-15)
    expect(reach.step([0, 0, 0, 0, 0, 0], [1, 1], key(0)).reward).toBeCloseTo(-0.21 - 0.2, 12)
    expect(reach.observe?.([0, 0, 0, 0, 0.1, 0.1])).toHaveLength(10)
    const pm = pointMass.create({ margin: 0.1 })
    // At the margin the Gaussian tolerance is 0.1; full actions keep 4/5 of it.
    expect(pm.step([0.115, 0, 0, 0, 0, 0], [0, 0], key(0)).reward).toBeCloseTo(0.1, 12)
    expect(pm.step([0, 0, 0, 0, 0, 0], [1, 1], key(0)).reward).toBeCloseTo(0.8, 12)
    const di = doubleIntegrator.create({ positionWeight: 2, controlWeight: 0.5 })
    expect(di.step([1, 1], [2], key(0)).reward).toBe(-(2 + 1 + 2))
    const cd = cartDoublePendulum.create()
    expect(cd.step([0, 0, 0, 0, 0, 0], [0], key(0)).reward).toBe(10)
  })
})
