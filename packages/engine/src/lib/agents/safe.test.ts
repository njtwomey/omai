import { describe, expect, it, vi } from 'vitest'
import { hazardCells, hazardGrid } from '../../testing/rl-continuous.ts'
import { argmaxOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import {
  type Agent,
  behaviourPolicy,
  type Env,
  greedyPolicyOf,
  oneHotObservation,
  runEpisode,
  timeLimit,
  trainAgent,
} from '../rl/index.ts'
import { ppo } from './policy-gradient.ts'
import {
  costWrapper,
  lagrangianPpo,
  modelShield,
  predicateShield,
  shield,
  shieldAction,
  shieldedEnv,
  transitionCost,
} from './safe.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

type Vec = readonly number[]
const hazards = hazardCells()
const options = { hidden: [16], learningRate: 3e-3, nEpochs: 4, nSteps: 128, batchSize: 32 } as const

// The hazard grid as a constrained MDP: cost 1 for every step into a hazard, one-hot observations, 20-step episodes.
function constrained(counter?: { visits: number }, inner: Env<number, number, number> = hazardGrid()) {
  const costed = costWrapper(inner, (_s, _a, step) => {
    const c = hazards.has(step.state) ? 1 : 0
    if (counter !== undefined) counter.visits += c
    return c
  })
  return oneHotObservation(timeLimit(costed, 20))
}

function averageCost<G>(env: Env<unknown, Vec, number>, agent: Agent<G, Vec, number>, state: G): number {
  let total = 0
  for (let i = 0; i < 50; i++)
    for (const info of runEpisode(env, behaviourPolicy(agent, state), key(100 + i)).infos)
      total += info['cost'] as number
  return total / 50
}

describe('constrained MDPs', () => {
  it('adds costs to the step information and reads them back from transitions', () => {
    const env = constrained()
    const r = env.reset(key(0))
    const step = env.step(r.state, 1, key(1))
    expect(step.info['cost']).toBe(1)
    expect(step.info['prob']).toBe(1)
    expect(
      transitionCost({
        observation: r.observation,
        action: 1,
        reward: step.reward,
        nextObservation: step.observation,
        terminated: false,
        truncated: false,
        info: step.info,
      }),
    ).toBe(1)
  })

  it('Lagrangian PPO keeps the average episode cost at the limit while plain PPO does not', () => {
    const env = constrained()
    const plain = ppo<number>(env, options)
    const plainState = trainAgent(env, plain, { maxSteps: 2500, seed: 0, checkpoints: false }).agent
    expect(averageCost(env, plain, plainState)).toBeGreaterThan(1.5)

    const agent = lagrangianPpo<number>(env, { ...options, costLimit: 0.5, lagrangianLearningRate: 0.2 })
    const s = trainAgent(env, agent, { maxSteps: 2500, seed: 2, checkpoints: false }).agent
    expect(averageCost(env, agent, s)).toBeLessThan(0.6)
    expect(s.lagrange).toBeGreaterThan(0)
    expect(s.stats['costValueLoss']).toBeDefined()
    // The greedy policy goes around the hazards to the goal.
    const greedy = runEpisode(env, greedyPolicyOf(agent, s), key(1))
    expect(greedy.terminated).toBe(true)
    expect(greedy.infos.every((i) => i['cost'] === 0)).toBe(true)
  })

  it('PID control of the multiplier also satisfies the constraint', () => {
    const env = constrained()
    const agent = lagrangianPpo<number>(env, { ...options, costLimit: 0.5, pid: { kp: 1, ki: 0.2 } })
    const s = trainAgent(env, agent, { maxSteps: 2500, seed: 1, checkpoints: false }).agent
    expect(averageCost(env, agent, s)).toBeLessThan(0.6)
    expect(agent.config).toEqual({ costLimit: 0.5, nSteps: 128, gamma: 0.99, pid: true })
  })
})

describe('shielding', () => {
  const grid = hazardGrid()
  const model = grid.model
  if (model === undefined) throw new Error('no model')

  it('computes the risk of actions by a model lookahead', () => {
    const one = modelShield(model, (s) => hazards.has(s))
    expect([0, 1, 2, 3].map((a) => one.risk(0, a))).toEqual([0, 1, 0, 0])
    // From cell 5 (below the first hazard) moving up enters it; with two steps of lookahead moving right does not
    // force a hazard, since the next move can go right again.
    expect(one.risk(5, 0)).toBe(1)
    const two = modelShield(model, (s) => hazards.has(s), { horizon: 2 })
    expect(two.risk(5, 1)).toBe(0)
    expect(shieldAction(one, 0, 1)).toEqual({ action: 0, replaced: true })
    expect(
      shieldAction(
        predicateShield((_c: number, a: number) => a > 5, [1, 2]),
        0,
        1,
      ).replaced,
    ).toBe(true)
  })

  it('a preemptively shielded agent never enters a hazard, even while learning', () => {
    const counter = { visits: 0 }
    const env = constrained(counter)
    const safe = shield(
      ppo<number>(env, options),
      modelShield(model, (s) => hazards.has(s)),
      (o: Vec) => argmaxOf(o),
    )
    const result = trainAgent(env, safe, { maxSteps: 2000, seed: 0, checkpoints: false })
    expect(counter.visits).toBe(0)
    expect(safe.name).toBe('shielded:ppo')
    const greedy = runEpisode(env, greedyPolicyOf(safe, result.agent), key(1))
    expect(greedy.return).toBeCloseTo(0.6, 12)
  })

  it('a post-posed shield in the environment keeps an unshielded learner out of hazards', () => {
    const counter = { visits: 0 }
    const shielded = shieldedEnv(
      grid,
      modelShield(model, (s) => hazards.has(s)),
      { penalty: 0.1 },
    )
    const env = constrained(counter, shielded)
    const agent = ppo<number>(env, options)
    trainAgent(env, agent, { maxSteps: 1500, seed: 0, checkpoints: false })
    expect(counter.visits).toBe(0)
    const step = shielded.step(0, 1, key(0))
    expect(step.info).toMatchObject({ shielded: 1, proposedAction: 1 })
    expect(step.reward).toBeCloseTo(-0.2, 12)
    // The first of the safest actions (up, into the wall) replaces the move into the hazard.
    expect(step.state).toBe(0)
  })
})
