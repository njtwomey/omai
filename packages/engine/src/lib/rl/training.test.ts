import { describe, expect, it } from 'vitest'
import { cartPole } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { episodes, recorder } from '../trace/index.ts'
import { behaviourPolicy, greedyPolicyOf, policyAgent } from './agent.ts'
import { make } from './gym.ts'
import { replayEpisode, trainAgent, trainingAlgorithm } from './training.ts'

const balance = (o: readonly number[]): number => ((o[2] ?? 0) + 0.5 * (o[3] ?? 0) > 0 ? 1 : 0)

describe('training loops', () => {
  it('train an agent for episodes, evaluate greedily and record agent statistics', () => {
    const rec = recorder()
    let calls = 0
    const counting = {
      ...policyAgent(balance, 'counting'),
      init: () => 0,
      learn: (s: number) => {
        calls += 1
        return s + 1
      },
      act: (_: number, o: readonly number[]) => balance(o),
      greedy: (_: number, o: readonly number[]) => balance(o),
      metrics: { updates: { description: 'learning updates' } },
      stats: (s: number) => ({ updates: s }),
    }
    const result = trainAgent(make(cartPole, {}, { maxEpisodeSteps: 30 }), counting, {
      episodes: 4,
      evalEvery: 2,
      evalEpisodes: 2,
      seed: 0,
      tracer: rec,
    })
    expect(result.returns).toEqual([30, 30, 30, 30])
    expect(result.agent).toBe(120)
    expect(calls).toBe(120)
    expect(result.evaluations.map((e) => e.episode)).toEqual([2, 4])
    expect(result.evaluations[0]?.meanReturn).toBe(30)
    expect(episodes(rec.run(), result.runId)).toHaveLength(4)
    expect(() => trainAgent(make(cartPole), counting)).toThrow(/episodes or maxSteps/)
  })

  it('keep doubling-spaced checkpoints and replay any episode exactly', () => {
    const env = make(cartPole)
    const agent = policyAgent((_: readonly number[], k) => (k.words[0] % 2 === 0 ? 1 : 0), 'coin')
    const result = trainAgent(env, agent, { episodes: 6, seed: 4 })
    const steps = result.state.checkpoints.map((c) => c.step)
    expect(steps.slice(0, 5)).toEqual([0, 1, 2, 4, 8])
    for (let e = 0; e < 6; e++) {
      const ep = replayEpisode(env, agent, result, e)
      expect(ep.return).toBe(result.returns[e])
      expect(ep.frames.length).toBe(ep.length + 1)
    }
    expect(() => replayEpisode(env, agent, result, 6)).toThrow(/did not finish/)
    const noCheckpoints = trainAgent(env, agent, { episodes: 2, seed: 4, checkpoints: false })
    expect(noCheckpoints.state.checkpoints).toHaveLength(0)
    expect(replayEpisode(env, agent, noCheckpoints, 1).return).toBe(noCheckpoints.returns[1])
  })

  it('expose behaviour and greedy policies of an agent state', () => {
    const agent = policyAgent((o: number) => o * 2)
    expect(behaviourPolicy(agent, null)(2, key(0))).toBe(4)
    expect(greedyPolicyOf(agent, null)(3, key(0))).toBe(6)
    expect(trainingAlgorithm(make(cartPole), agent as never).schema?.episodes).toBe(true)
  })
})
