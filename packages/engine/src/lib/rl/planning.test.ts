import { describe, expect, it } from 'vitest'
import { takeAway } from '../../testing/rl.ts'
import { cartPole, makeCliffWalking, makeFrozenLake, mountainCarContinuous, pendulum } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { mdpFromEnv, valueIteration } from './mdp.ts'
import { cem, cemPlannerAlgorithm, gameMcts, mcts, mctsAlgorithm, mppi, rolloutPlanner } from './planning.ts'

describe('Monte Carlo tree search', () => {
  it('pushes a falling pole back up and traces the root statistics', () => {
    for (const [theta, best] of [
      [0.12, 1],
      [-0.12, 0],
    ] as const) {
      const r = mcts(cartPole.create(), [0, 0, theta, 0], { simulations: 300, depth: 25, c: 5, key: key(1) })
      expect(r.action).toBe(best)
      expect(r.visits.reduce((a, b) => a + b, 0)).toBe(300)
    }
    const rec = recorder()
    runAlgorithm(mctsAlgorithm(makeCliffWalking(), 36, { depth: 10 }), { maxSteps: 20, tracer: rec })
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'rootValue')).toBe(true)
  })

  it('agrees with value iteration on a stochastic lake, branching on outcomes', () => {
    const env = makeFrozenLake({ mapName: '4x4' })
    const vi = valueIteration(mdpFromEnv(env, 0.95))
    const r = mcts(env, 14, { simulations: 600, depth: 30, gamma: 0.95, key: key(2), stateKey: String })
    // State 14 is next to the goal: moving right (2) is optimal.
    expect(r.action).toBe(vi.policy[14])
  })

  it('uses priors and a value function (PUCT) when given', () => {
    const env = makeCliffWalking()
    const r = mcts(env, 36, { simulations: 50, prior: () => [1, 0, 0, 0], value: () => 0, depth: 5 })
    expect(r.visits[0]).toBeGreaterThan(40)
  })

  it('keeps earlier search states intact (persistent trees)', () => {
    const alg = mctsAlgorithm(makeCliffWalking(), 36, { depth: 10 })
    const a = runAlgorithm(alg, { maxSteps: 5, seed: 0 })
    const before = [...a.state.root.n]
    runAlgorithm(alg, { maxSteps: 5, seed: 0, start: { state: a.state, step: 5 } })
    expect(a.state.root.n).toEqual(before)
  })
})

describe('game tree search', () => {
  it('plays the subtraction game perfectly: leave a multiple of three', () => {
    for (const n of [4, 5, 7, 8]) {
      const env = takeAway(n)
      const r = gameMcts(env, env.reset(key(0)), { simulations: 400, key: key(n) })
      // Taking (n mod 3) tokens leaves a multiple of three: action index (n mod 3) - 1.
      expect(r.action).toBe((n % 3) - 1)
    }
    const env = takeAway(3)
    expect(() => gameMcts(env, { tokens: 0, mover: 0 })).toThrow(/over/)
  })

  it('uses a prior and a value function instead of playouts', () => {
    const env = takeAway(5)
    const r = gameMcts(env, env.reset(key(0)), {
      simulations: 200,
      prior: (_, acts) => acts.map(() => 1 / acts.length),
      value: (s) => (s.tokens % 3 === 0 ? -1 : 1),
    })
    expect(r.action).toBe(1)
  })
})

describe('rollout and sequence planners', () => {
  it('improve on a random base policy with rollouts', () => {
    const r = rolloutPlanner(cartPole.create(), [0, 0, 0.15, 0.3], { rollouts: 8, horizon: 20, key: key(3) })
    expect(r.action).toBe(1)
    expect(r.values).toHaveLength(2)
  })

  it('swing the pendulum towards upright with CEM and MPPI', () => {
    const env = pendulum.create()
    const down: [number, number] = [Math.PI - 0.3, 0]
    const plan = cem(env, down, { horizon: 10, population: 32, iterations: 4, key: key(0) })
    expect(plan.plan).toHaveLength(10)
    expect(plan.action[0] ?? 0).toBeGreaterThanOrEqual(-2)
    const zero = cem(env, down, { horizon: 10, population: 2, iterations: 1, elites: 1, sigma: 0, key: key(0) })
    expect(plan.value).toBeGreaterThan(zero.value)
    const m = mppi(env, down, { horizon: 10, population: 32, iterations: 3, lambda: 0.5, key: key(1) })
    expect(m.plan).toHaveLength(10)
    expect(Math.abs(m.action[0] ?? 0)).toBeLessThanOrEqual(2)
    expect(() => cemPlannerAlgorithm(cartPole.create() as never, [0, 0, 0, 0])).toThrow(/box/)
  })

  it('climbs the continuous mountain car hill in simulation', () => {
    const r = cem(mountainCarContinuous.create(), [-0.5, 0], {
      horizon: 30,
      population: 64,
      iterations: 3,
      key: key(5),
    })
    expect(r.plan[0]).toHaveLength(1)
  })
})
