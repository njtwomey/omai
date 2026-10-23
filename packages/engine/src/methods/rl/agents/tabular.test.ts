import { describe, expect, it } from 'vitest'
import { maxOf } from '../../../lib/base/index.ts'
import { key } from '../../../lib/random/index.ts'
import {
  behaviourPolicy,
  evaluatePolicy,
  greedyPolicyOf,
  make,
  mdpFromEnv,
  policyEvaluation,
  timeLimit,
  trainAgent,
  valueIteration,
} from '../../../lib/rl/index.ts'
import { box, discrete } from '../../../lib/space/index.ts'
import { recorder, tracePolicy } from '../../../lib/trace/index.ts'
import { DYNA_MAZE, randomMdpEnv, randomWalkEnv } from '../../../testing/rl-tabular.ts'
import { blackjack, cliffWalking, makeGridWorld, windyGridWorld } from '../environments/index.ts'
import { randomAgent } from './random.ts'
import {
  doubleQLearning,
  dynaQ,
  expectedSarsa,
  greedyTable,
  monteCarloControl,
  nStepSarsa,
  prioritizedSweeping,
  qLearning,
  qTable,
  sarsa,
  sarsaLambda,
  td0,
  tdLambda,
  valueTable,
} from './tabular.ts'

const maxError = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number => {
  let m = 0
  for (const [i, r] of a.entries()) for (const [j, x] of r.entries()) m = Math.max(m, Math.abs(x - (b[i]?.[j] ?? 0)))
  return m
}
const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('Q-learning and SARSA on CliffWalking (Sutton and Barto, Example 6.6)', () => {
  const env = make(cliffWalking)
  const options = { alpha: 0.5, gamma: 1, epsilon: 0.1 } as const
  const runs = [qLearning(env, options), sarsa(env, options)].map((agent) => {
    const r = trainAgent(env, agent as never, { episodes: 500, seed: 0, checkpoints: false })
    const greedy = evaluatePolicy(env, greedyPolicyOf(agent as never, r.agent), { episodes: 1 })
    return { greedy: greedy.meanReturn, online: mean(r.returns.slice(-100)) }
  })
  const [q, s] = runs as [(typeof runs)[0], (typeof runs)[0]]

  it('Q-learning learns the optimal path along the cliff', () => expect(q.greedy).toBe(-13))
  it('SARSA learns a safer path away from the edge', () => {
    expect(s.greedy).toBeLessThan(-13)
    expect(s.greedy).toBeGreaterThanOrEqual(-17)
  })
  it('SARSA earns more online, since exploration near the edge costs Q-learning falls', () =>
    expect(s.online).toBeGreaterThan(q.online + 10))
})

describe('agreement with exact solutions', () => {
  const gamma = 0.8
  const exact = valueIteration(mdpFromEnv(randomMdpEnv(1), gamma)).Q
  const env = timeLimit(randomMdpEnv(1), 50)
  const common = { alpha: (n: number) => 1 / n ** 0.7, gamma, epsilon: 1 } as const

  for (const [name, agent] of [
    ['Q-learning', qLearning(env, common)],
    ['expected SARSA with a greedy target', expectedSarsa(env, { ...common, targetEpsilon: 0 })],
    ['double Q-learning', doubleQLearning(env, common)],
  ] as const) {
    it(`${name} converges to Q* from uniformly random behaviour (bootstrapping through time limits)`, () => {
      const r = trainAgent(env, agent as never, { maxSteps: 20_000, seed: 0, checkpoints: false })
      expect(maxError(qTable(r.agent as never), exact)).toBeLessThan(0.1)
    })
  }

  it('Dyna-Q and prioritised sweeping reach Q* of a deterministic maze from exploratory experience', () => {
    const maze = makeGridWorld({ map: DYNA_MAZE, slip: 0, stepReward: 0, goalReward: 1 })
    const star = valueIteration(mdpFromEnv(maze, 0.95)).Q
    for (const agent of [
      dynaQ(maze, { alpha: 0.5, gamma: 0.95, planningSteps: 20, epsilon: 1 }),
      prioritizedSweeping(maze, { alpha: 1, gamma: 0.95, planningSteps: 20, theta: 1e-9, epsilon: 1 }),
    ]) {
      const r = trainAgent(maze, agent as never, { maxSteps: 4000, seed: 1, checkpoints: false })
      // Every reachable pair has been tried; pairs of the goal and the walls are never acted in.
      const visits = (r.agent as { visits: Float64Array }).visits
      const learned = qTable(r.agent as never).map((row, s) =>
        row.map((x, a) => ((visits[s * 4 + a] ?? 0) > 0 ? x : (star[s]?.[a] ?? 0))),
      )
      expect(maxError(learned, star)).toBeLessThan(1e-3)
    }
  })

  it('prioritised sweeping leaves its values solving the Bellman equations of its learned model', () => {
    const maze = makeGridWorld({ map: DYNA_MAZE, slip: 0, stepReward: 0, goalReward: 1 })
    const agent = prioritizedSweeping(maze, { alpha: 1, gamma: 0.95, planningSteps: 50, theta: 1e-9 })
    const s = trainAgent(maze, agent, { episodes: 30, seed: 1, checkpoints: false }).agent
    expect(agent.stats?.(s)['queueSize']).toBe(0)
    for (const i of s.model.observed) {
      const next = s.model.next[i] ?? -1
      const future = next < 0 ? 0 : 0.95 * maxOf(s.Q.subarray(next * 4, next * 4 + 4))
      expect(Math.abs((s.model.reward[i] ?? 0) + future - (s.Q[i] ?? 0))).toBeLessThan(1e-9)
    }
  })

  it('TD(0) and TD(lambda) converge to the values of the random walk (Example 6.2)', () => {
    const walk = randomWalkEnv()
    const truth = policyEvaluation(mdpFromEnv(walk, 1), new Array<number>(7).fill(0))
    expect(truth.slice(1, 6).map((v) => Number(v.toFixed(12)))).toEqual(
      [1, 2, 3, 4, 5].map((i) => Number((i / 6).toFixed(12))),
    )
    for (const agent of [
      td0(walk, { alpha: (n) => 1 / n ** 0.6, gamma: 1 }),
      tdLambda(walk, { alpha: (n) => 1 / n ** 0.6, gamma: 1, lambda: 0.5 }),
      tdLambda(walk, { alpha: (n) => 1 / n ** 0.6, gamma: 1, lambda: 0.5, trace: 'replacing' }),
      tdLambda(walk, { alpha: 0.01, gamma: 1, lambda: 0.5, trace: 'dutch' }),
    ]) {
      const r = trainAgent(walk, agent, { episodes: 2000, seed: 2, checkpoints: false })
      expect(maxError([valueTable(r.agent)], [truth])).toBeLessThan(0.06)
    }
  })

  it('TD(lambda) evaluates a stochastic policy on a random MDP', () => {
    const mdpEnv = randomMdpEnv(3, 4, 2, 0.2)
    const policy = Array.from({ length: 4 }, () => [0.3, 0.7])
    const truth = policyEvaluation(mdpFromEnv(mdpEnv, 0.9), policy)
    const agent = tdLambda(mdpEnv, { policy, alpha: (n) => 1 / n ** 0.6, gamma: 0.9, lambda: 0.5 })
    const r = trainAgent(mdpEnv, agent, { maxSteps: 30_000, seed: 3, checkpoints: false })
    expect(maxError([valueTable(r.agent)], [truth])).toBeLessThan(0.1)
  })
})

describe('planning agents learn faster (Sutton and Barto, Example 8.1)', () => {
  it('Dyna-Q and prioritised sweeping need far fewer steps than Q-learning on the Dyna maze', () => {
    const maze = makeGridWorld({ map: DYNA_MAZE, slip: 0, stepReward: 0, goalReward: 1 })
    const steps = [
      qLearning(maze, { alpha: 0.1, gamma: 0.95 }),
      dynaQ(maze, { alpha: 0.1, gamma: 0.95, planningSteps: 50 }),
      prioritizedSweeping(maze, { alpha: 0.5, gamma: 0.95, planningSteps: 5 }),
      dynaQ(maze, { alpha: 0.1, gamma: 0.95, planningSteps: 10, kappa: 1e-3 }),
    ].map((agent) => {
      const r = trainAgent(maze, agent as never, { episodes: 20, seed: 0, checkpoints: false })
      // Steps from the second episode on (the first is the same random walk for every agent).
      return r.lengths.slice(1).reduce((a, b) => a + b, 0)
    })
    const [q, dyna, ps, plus] = steps as [number, number, number, number]
    expect(dyna).toBeLessThan(q / 3)
    expect(ps).toBeLessThan(q / 2)
    expect(plus).toBeLessThan(q / 2)
  })
})

describe('more control agents', () => {
  for (const [name, make1] of [
    ['n-step SARSA', (e: never) => nStepSarsa(e, { n: 4, alpha: 0.2, gamma: 1 })],
    ['n-step expected SARSA', (e: never) => nStepSarsa(e, { n: 3, alpha: 0.3, gamma: 1, expected: true })],
    ['SARSA(lambda)', (e: never) => sarsaLambda(e, { lambda: 0.8, alpha: 0.2, gamma: 1 })],
    [
      'SARSA(lambda), replacing',
      (e: never) => sarsaLambda(e, { lambda: 0.8, alpha: 0.2, gamma: 1, trace: 'replacing' }),
    ],
    ['expected SARSA', (e: never) => expectedSarsa(e, { alpha: 0.5, gamma: 1 })],
  ] as const) {
    it(`${name} solves the windy grid world (Example 6.5)`, () => {
      const env = make(windyGridWorld)
      const agent = make1(env as never)
      const r = trainAgent(env, agent as never, { episodes: 200, seed: 0, checkpoints: false })
      const greedy = evaluatePolicy(env, greedyPolicyOf(agent as never, r.agent), { episodes: 1 })
      expect(greedy.meanReturn).toBeGreaterThanOrEqual(-20)
    })
  }

  it('Monte Carlo control with exploring starts approaches the Blackjack policy of Figure 5.2', () => {
    const env = blackjack.create()
    const agent = monteCarloControl(env, { exploringStarts: true, gamma: 1 })
    const r = trainAgent(env, agent, { episodes: 10_000, seed: 0, checkpoints: false })
    const policy = greedyPolicyOf(agent, r.agent)
    const greedy = evaluatePolicy(env, policy, { episodes: 2000, seed: 1 })
    const random = evaluatePolicy(env, behaviourPolicy(randomAgent(env), null), { episodes: 2000, seed: 1 })
    expect(greedy.meanReturn).toBeGreaterThan(-0.12)
    expect(greedy.meanReturn).toBeGreaterThan(random.meanReturn + 0.2)
    // Coarse shape of the optimal policy without a usable ace: stick on 20 and 21, hit on 11 or less.
    let agree = 0
    let total = 0
    for (let dealer = 1; dealer <= 10; dealer++) {
      for (const sum of [4, 6, 8, 10, 11, 20, 21]) {
        total += 1
        if (agent.greedy(r.agent, [sum, dealer, 0]) === (sum >= 20 ? 0 : 1)) agree += 1
      }
    }
    expect(agree / total).toBeGreaterThan(0.8)
  })

  it('every-visit epsilon-soft Monte Carlo control improves on random play', () => {
    const env = blackjack.create()
    const agent = monteCarloControl(env, { everyVisit: true, epsilon: 0.1, gamma: 1 })
    const r = trainAgent(env, agent, { episodes: 4000, seed: 4, checkpoints: false })
    const greedy = evaluatePolicy(env, greedyPolicyOf(agent, r.agent), { episodes: 2000, seed: 1 })
    expect(greedy.meanReturn).toBeGreaterThan(-0.2)
  })
})

describe('the agent contract', () => {
  it('runs under trainAgent with traced statistics and an epsilon schedule over episodes', () => {
    const env = make(cliffWalking)
    const agent = qLearning(env, { epsilon: (e) => 1 / (1 + e), epsilonUnit: 'episodes', alpha: 0.5, gamma: 1 })
    const rec = recorder()
    const r = trainAgent(env, agent, {
      episodes: 5,
      seed: 0,
      tracer: rec,
      policy: tracePolicy({ metrics: ['epsilon', 'tdError'] }),
    })
    const names = new Set(
      rec
        .run()
        .events.filter((e) => e.type === 'metric')
        .map((e) => (e as { name: string }).name),
    )
    expect(names).toEqual(new Set(['epsilon', 'tdError']))
    expect(agent.stats?.(r.agent)['epsilon']).toBeCloseTo(1 / 6, 12)
  })

  it('learn is pure and greedy tables are exposed', () => {
    const env = cliffWalking.create()
    for (const agent of [
      qLearning(env),
      sarsa(env),
      expectedSarsa(env),
      nStepSarsa(env),
      monteCarloControl(env),
      sarsaLambda(env),
      doubleQLearning(env),
      dynaQ(env),
      prioritizedSweeping(env),
    ]) {
      const s0 = agent.init(key(0))
      const before = Array.from(s0.Q)
      let s = s0
      for (const [i, t] of [
        { observation: 36, action: 0, reward: -1, nextObservation: 24, terminated: false, truncated: false },
        { observation: 24, action: 1, reward: -1, nextObservation: 25, terminated: false, truncated: false },
        { observation: 25, action: 2, reward: -100, nextObservation: 36, terminated: true, truncated: false },
      ].entries())
        s = (agent.learn as (g: unknown, t: unknown, k: unknown) => typeof s)(s, t, key(i + 1))
      expect(Array.from(s0.Q)).toEqual(before)
      expect(s.steps).toBe(3)
      expect(s.episodes).toBe(1)
      expect(greedyTable(s)).toHaveLength(48)
      expect(
        qTable(s)
          .flat()
          .some((x) => x !== 0),
      ).toBe(true)
    }
  })

  it('codes tuple observations, accepts custom codings and rejects what it cannot code', () => {
    expect(qLearning(blackjack.create()).config).toMatchObject({ nStates: 32 * 11 * 2, nActions: 2 })
    const spaces = { observationSpace: box(0, 1, [2]), actionSpace: discrete(3) }
    expect(() => qLearning(spaces)).toThrow(/nStates/)
    const coded = qLearning<readonly number[]>(spaces, {
      nStates: 4,
      encode: (o) => 2 * Math.round(o[0] ?? 0) + Math.round(o[1] ?? 0),
    })
    const s = coded.learn(
      coded.init(key(0)),
      { observation: [1, 1], action: 2, reward: 1, nextObservation: [0, 0], terminated: true, truncated: false },
      key(1),
    )
    expect(qTable(s)[3]).toEqual([0, 0, 0.1])
    expect(coded.greedy(s, [1, 1])).toBe(2)
    expect(() => coded.act(s, [5, 5], key(2))).toThrow(/outside/)
    expect(() => qLearning({ observationSpace: discrete(2), actionSpace: box(0, 1, [1]) })).toThrow(/discrete action/)
    expect(() => nStepSarsa(cliffWalking.create(), { n: 0 })).toThrow(/positive integer/)
  })

  it('optimistic initial values explore every action of a state before repeating one', () => {
    const env = cliffWalking.create()
    const agent = qLearning(env, { initialValue: 10, epsilon: 0, alpha: 0.5 })
    let s = agent.init(key(0))
    const seen = new Set<number>()
    for (let i = 0; i < 4; i++) {
      const a = agent.act(s, 0, key(10 + i))
      seen.add(a)
      s = agent.learn(
        s,
        { observation: 0, action: a, reward: -1, nextObservation: 0, terminated: true, truncated: false },
        key(i),
      )
    }
    expect(seen.size).toBe(4)
  })

  it('a prediction agent follows its policy: functions, probability tables and the uniform default', () => {
    const env = randomWalkEnv()
    expect(td0(env, { policy: () => 0 }).act(td0(env).init(key(0)), 3, key(1))).toBe(0)
    const table = Array.from({ length: 7 }, () => [1])
    expect(tdLambda(env, { policy: table }).greedy(td0(env).init(key(0)), 3)).toBe(0)
    expect(td0(env).config).toMatchObject({ policy: 'uniform' })
  })
})
