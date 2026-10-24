import { describe, expect, it } from 'vitest'
import { type Key, key, uniform } from '../../../lib/random/index.ts'
import {
  aecSingleAgentView,
  evaluatePolicy,
  greedyPolicyOf,
  jointEnv,
  parallelTimeLimit,
  policyAgent,
  trainAgent,
} from '../../../lib/rl/index.ts'
import { kuhnPoker, kuhnPokerState, matrixGame, nim } from '../../../methods/rl/index.ts'
import { expectedReturns } from '../../../testing/games.ts'
import { type TabularQState, tabularQ } from './_multi-agent.ts'
import {
  fictitiousSelfPlay,
  fictitiousSelfPlayAlgorithm,
  independentLearners,
  minimaxQ,
  minimaxQStrategy,
  selfPlay,
} from './multi-agent.ts'
import { dqn } from './value-based.ts'

// Learning tests stay small; the generous timeout covers a loaded machine.
const SLOW = { timeout: 30_000 }

describe('independent learners', () => {
  it('find mutual cooperation in a repeated stag hunt (optimistic tabular Q)', SLOW, () => {
    const base = matrixGame.create({ game: 'stagHunt' })
    const env = jointEnv(matrixGame.create({ game: 'stagHunt', rounds: 10 }))
    const agent = independentLearners(base, {
      initialValue: 10,
      epsilonStart: 0.1,
      epsilonEnd: 0.01,
      explorationSteps: 500,
      gamma: 0.9,
    })
    const r = trainAgent(env, agent, { maxSteps: 2000, key: key(0), checkpoints: false })
    // Both hunt the stag every round: 4 + 4 per round.
    expect(evaluatePolicy(env, greedyPolicyOf(agent, r.agent), { episodes: 1, key: key(1) }).meanReturn).toBe(80)
  })

  it("take any single-agent learner per agent and learn from each agent's own reward", () => {
    const base = matrixGame.create({ game: 'prisonersDilemma' })
    const agent = independentLearners(base, { learner: (a) => policyAgent(() => (a === 'player_0' ? 1 : 0), a) })
    const s = agent.init(key(0))
    expect(agent.act(s, { player_0: 0, player_1: 0 }, key(1))).toEqual({ player_0: 1, player_1: 0 })
    // Defecting against a cooperator: the tabular Q-learners update their own values from their own rewards.
    const q = independentLearners(base, { alpha: 0.5 })
    const t = {
      observation: { player_0: 0, player_1: 0 },
      action: { player_0: 1, player_1: 0 },
      reward: 5,
      nextObservation: { player_0: 3, player_1: 3 },
      terminated: true,
      truncated: false,
      info: { rewards: { player_0: 5, player_1: 0 } },
    }
    const learned = q.learn(q.init(key(0)), t, key(2)) as Record<string, { q: Record<string, number[]> }>
    expect(learned['player_0']?.q['0']).toEqual([0, 2.5])
    expect(learned['player_1']?.q['0']).toEqual([0, 0])
  })
})

describe('minimax-Q', () => {
  it('approaches the mixed equilibria of matching pennies and rock-paper-scissors', SLOW, () => {
    for (const [game, n] of [
      ['matchingPennies', 2],
      ['rockPaperScissors', 3],
    ] as const) {
      const base = matrixGame.create({ game })
      const agent = minimaxQ(base, { gamma: 0.5 })
      const r = trainAgent(jointEnv(parallelTimeLimit(base, 20)), agent, {
        maxSteps: 1500,
        key: key(2),
        checkpoints: false,
      })
      for (const agentName of ['player_0', 'player_1']) {
        const pi = minimaxQStrategy(r.agent, agentName, 1)
        expect(pi.length).toBe(n)
        for (const p of pi) expect(Math.abs(p - 1 / n), game).toBeLessThan(0.05)
      }
    }
    expect(() => minimaxQ({ possibleAgents: ['a'], observationSpaces: {}, actionSpaces: {} })).toThrow(/two agents/)
  })
})

// A uniformly random legal Nim move read off the one-hot heaps of a [1, 2, 3] observation.
const randomNim = (o: readonly number[], k: Key): number => {
  const heaps = [o.slice(0, 2).indexOf(1), o.slice(2, 5).indexOf(1), o.slice(5, 9).indexOf(1)]
  const legal = heaps.flatMap((h, i) => Array.from({ length: h }, (_, t) => i * 3 + t))
  return legal[Math.floor((uniform(k, []).data[0] ?? 0) * legal.length)] ?? 0
}

describe('self-play', () => {
  it('trains against its own exploratory snapshots at Nim and learns to beat a random player', SLOW, () => {
    const env = nim.create({ heaps: [1, 2, 3] })
    const learner = tabularQ(9, { alpha: 0.5, gamma: 1, explorationSteps: 2000, epsilonEnd: 0.1 })
    const score = (g: TabularQState): number => {
      let total = 0
      for (const seat of ['player_0', 'player_1'] as const) {
        const other = seat === 'player_0' ? 'player_1' : 'player_0'
        const view = aecSingleAgentView(env, seat, { [other]: randomNim })
        total += evaluatePolicy(view, greedyPolicyOf(learner, g), { episodes: 40, key: key(5) }).meanReturn / 2
      }
      return total
    }
    const r = selfPlay(env, learner, {
      generations: 8,
      stepsPerGeneration: 400,
      alternateSeats: true,
      opponent: 'behaviour',
      key: key(1),
    })
    expect(r.pool.length).toBe(9)
    // The untrained learner forfeits with an illegal move; self-play teaches it to win nearly every game.
    expect(score(r.pool[0] as TabularQState)).toBe(-1)
    expect(score(r.learner)).toBeGreaterThan(0.8)
  })

  it('runs neural learners too', SLOW, () => {
    const env = nim.create({ heaps: [1, 2] })
    const spaces = {
      observationSpace: env.observationSpaces['player_0'] ?? { kind: 'bool' as const },
      actionSpace: env.actionSpaces['player_0'] ?? { kind: 'bool' as const },
    }
    const learner = dqn(spaces, { hidden: [8], learningStarts: 20, batchSize: 8 })
    const r = selfPlay(env, learner, { generations: 2, stepsPerGeneration: 40, poolSize: 2, key: key(3) })
    expect(r.pool.length).toBe(2)
    expect(r.learner.steps).toBe(80)
  })

  it('works on simultaneous games through the single-agent view', () => {
    const r = selfPlay(
      matrixGame.create({ rounds: 5 }),
      policyAgent(() => 1),
      { generations: 2, stepsPerGeneration: 5 },
    )
    // Mutual defection pays 1 per round.
    expect(r.state.meanReturn).toBe(5)
    expect(r.state.generation).toBe(2)
  })
})

// The exact expected value for `seat` of a policy against an always-bet (and call) opponent at Kuhn poker.
function valueAgainstBettor(seat: 0 | 1, policy: (o: readonly number[]) => readonly number[]): number {
  const env = kuhnPoker.create()
  let v = 0
  for (const [a, b] of [
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [2, 0],
    [2, 1],
  ] as const) {
    const values = expectedReturns(env, kuhnPokerState([a, b]), (st, agent, legal) =>
      agent === `player_${seat}` ? policy(env.observe(st, agent)) : legal.map((x) => (x === 1 ? 1 : 0)),
    )
    v += values[seat] / 6
  }
  return v
}

describe('fictitious self-play', () => {
  it('learns average Kuhn poker policies that exploit an always-bet opponent better than uniform play', SLOW, () => {
    const r = fictitiousSelfPlay(kuhnPoker.create(), {
      hidden: [32],
      averageHidden: [32],
      episodes: 1200,
      eta: 0.2,
      learningStarts: 50,
      explorationSteps: 1000,
      targetUpdateInterval: 100,
      gamma: 1,
      trainFrequency: 2,
      batchSize: 32,
      averageBatchSize: 32,
      key: key(2),
    })
    const uniformPlay = (): number[] => [0.5, 0.5]
    for (const seat of [0, 1] as const) {
      const learned = valueAgainstBettor(seat, (o) => r.averagePolicy(`player_${seat}`, o))
      expect(learned).toBeGreaterThan(valueAgainstBettor(seat, uniformPlay) + 0.2)
    }
    const p = r.averagePolicy('player_0', [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    expect(p[0] ?? 0).toBeCloseTo(1 - (p[1] ?? 0), 12)
    expect(r.state.episodes).toBe(1200)
    expect([0, 1]).toContain(r.policy('player_0')([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], key(0)))
    expect(() => r.averagePolicy('nobody', [])).toThrow(/no agent/)
    expect(fictitiousSelfPlayAlgorithm(kuhnPoker.create()).name).toBe('fictitiousSelfPlay')
  })
})
