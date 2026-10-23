import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, runEpisode, trainAgent } from '../../../lib/rl/index.ts'
import { add, cos, mul, sliceAxis, square, sub, sum, toArray, type Value } from '../../../lib/tensor/index.ts'
import { recorder } from '../../../lib/trace/index.ts'
import {
  angleNormalize,
  cartDoublePendulum,
  cartDoublePendulumModel,
  cartPole,
  cartPoleModel,
  doubleIntegrator,
  doubleIntegratorMatrices,
  doubleIntegratorModel,
  pendulum,
  pendulumDynamics,
  pendulumModel,
} from '../environments/index.ts'
import { cemAgent, ilqr, ilqrAlgorithm, lqr, lqrAgent, mpcAgent, swingUpAgent } from './control.ts'

const angleOf = (o: readonly number[]): number => Math.atan2(o[1] ?? 0, o[0] ?? 1)
const toState = (o: readonly number[]): number[] => [angleOf(o), o[2] ?? 0]
const hanging = { inner: [Math.PI, 0], t: 0 } as never

describe('LQR', () => {
  it('balances CartPole for the full 500 steps from the linearisation at upright', () => {
    const agent = lqrAgent<number>(cartPoleModel())
    const result = evaluatePolicy(make(cartPole), greedyPolicyOf(agent, agent.init(key(0))), { episodes: 3, seed: 0 })
    expect(result.lengths).toEqual([500, 500, 500])
  })

  it('is optimal on the double integrator: the return is minus x0^T P x0', () => {
    const agent = lqrAgent(doubleIntegratorModel())
    const ep = runEpisode(make(doubleIntegrator), greedyPolicyOf(agent, agent.init(key(0))), key(3))
    const x0 = ep.states[0] as unknown as { inner: [number, number] }
    const { A, B } = doubleIntegratorMatrices({ mass: 1, dt: 0.1 })
    const P = toArray(
      lqr(
        A,
        B,
        [
          [1, 0],
          [0, 1],
        ],
        [[1]],
      ).P,
    ) as number[][]
    const [p, v] = x0.inner
    const cost = (P[0]?.[0] ?? 0) * p * p + 2 * (P[0]?.[1] ?? 0) * p * v + (P[1]?.[1] ?? 0) * v * v
    expect(ep.return).toBeCloseTo(-cost, 6)
  })

  it('balances the cart double pendulum for the full 1000 steps', () => {
    const agent = lqrAgent(cartDoublePendulumModel())
    const result = evaluatePolicy(make(cartDoublePendulum), greedyPolicyOf(agent, agent.init(key(0))), {
      episodes: 2,
      seed: 1,
    })
    expect(result.lengths).toEqual([1000, 1000])
  })
})

describe('swing-up', () => {
  it('pumps the pendulum up from hanging down and holds it upright with LQR', () => {
    const agent = swingUpAgent()
    const ep = runEpisode(make(pendulum), greedyPolicyOf(agent, agent.init(key(0))), key(0), { start: hanging })
    const last = ep.observations.slice(-50).map((o) => Math.abs(angleOf(o)))
    expect(Math.max(...last)).toBeLessThan(0.01)
    // Every torque respected the limit.
    expect(ep.actions.every((a) => Math.abs(a[0] ?? 0) <= 2)).toBe(true)
  })
})

describe('iLQR', () => {
  // A large terminal cost on not being upright and at rest.
  const upright = (x: Value): Value =>
    add(sum(mul(sub(1, cos(sliceAxis(x, -1, 0, 1))), 100), -1), sum(mul(square(sliceAxis(x, -1, 1, 2)), 10), -1))
  // A warm start that rocks the pendulum at its natural frequency (from the hanging equilibrium itself every local
  // method sits in the no-swing optimum).
  const rock = Array.from({ length: 60 }, (_, t) => [0.5 * Math.sin(Math.sqrt(15) * t * 0.05)])

  it('decreases the cost monotonically and swings the pendulum up within the torque limit', () => {
    const rec = recorder()
    const r = ilqr(pendulumModel(), [Math.PI, 0], { initial: rock, terminalCost: upright, tracer: rec })
    expect(r.converged).toBe(true)
    for (let i = 1; i < r.history.length; i++) expect(r.history[i]).toBeLessThan(r.history[i - 1] ?? Infinity)
    expect(r.cost).toBeLessThan(r.history[0] ?? 0)
    expect(r.controls.every((u) => Math.abs(u[0] ?? 0) <= 2)).toBe(true)
    // Replayed open loop in the environment's dynamics, the controls end near upright.
    let s: readonly [number, number] = [Math.PI, 0]
    for (const u of r.controls) s = pendulumDynamics(s, u[0] ?? 0).state
    expect(Math.abs(angleNormalize(s[0]))).toBeLessThan(0.25)
    const metrics = rec
      .run()
      .events.filter((e) => e.type === 'metric')
      .map((e) => (e as { name: string }).name)
    expect(new Set(metrics)).toEqual(new Set(['cost', 'mu', 'alpha', 'expected']))
  })

  it('recovers the LQR solution on a linear-quadratic problem', () => {
    const model = doubleIntegratorModel()
    const r = ilqr(model, [1, 0.5], { horizon: 150, squash: false, mu: 0 })
    const { A, B } = doubleIntegratorMatrices({ mass: 1, dt: 0.1 })
    const K = toArray(
      lqr(
        A,
        B,
        [
          [1, 0],
          [0, 1],
        ],
        [[1]],
      ).K,
    ) as number[][]
    // A linear-quadratic problem is solved by the first Newton step; the first control is the infinite-horizon one.
    expect(r.history.length).toBeLessThanOrEqual(3)
    expect(r.controls[0]?.[0]).toBeCloseTo(-((K[0]?.[0] ?? 0) * 1 + (K[0]?.[1] ?? 0) * 0.5), 6)
    const alg = ilqrAlgorithm(model, [1, 0], { horizon: 5 })
    expect(alg.init(key(0)).controls).toHaveLength(5)
  })
})

describe('planning agents on Pendulum', () => {
  const short = () => make(pendulum, {}, { maxEpisodeSteps: 100 })
  const zero = evaluatePolicy(short(), () => [0], { episodes: 1, seed: 3 }).meanReturn

  it('MPC with iLQR beats a zero-torque policy and keeps warm-started plans in its state', () => {
    const agent = mpcAgent(pendulumModel(), { horizon: 20, iterations: 2, toState })
    const result = trainAgent(short(), agent, { episodes: 1, seed: 3 })
    expect(result.returns[0]).toBeGreaterThan(zero + 300)
    expect(result.agent.replans).toBe(99)
  })

  for (const planner of ['cem', 'mppi'] as const) {
    it(`MPC with batched ${planner} beats a zero-torque policy`, () => {
      const agent = mpcAgent(pendulumModel(), { planner, horizon: 15, population: 32, toState, iterations: 2 })
      const result = trainAgent(short(), agent, { episodes: 1, seed: 3 })
      expect(result.returns[0]).toBeGreaterThan(zero + 300)
    })
  }

  it('CEM in the true environment beats a zero-torque policy', () => {
    const agent = cemAgent(pendulum.create(), { toState: (o) => toState(o) as never, horizon: 15, population: 32 })
    const result = trainAgent(short(), agent, { episodes: 1, seed: 3 })
    expect(result.returns[0]).toBeGreaterThan(zero + 300)
    expect(result.agent.replans).toBe(99)
  })
})
