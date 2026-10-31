import { describe, expect, it } from 'vitest'
import { reachGoal } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, trainAgent } from '../rl/index.ts'
import { actionTileFeatures, tileCoder } from './approximation.ts'
import { type GoalStep, goalConditionedQ, relabelEpisode } from './goal-conditioned.ts'

// A straight walk to the right: the point reaches 0.1, 0.2, ... 0.5 while pursuing a goal it never gets to.
const walk: readonly GoalStep[] = Array.from({ length: 5 }, (_, t) => ({
  observation: [0.1 * t, 0, 1, 1],
  action: 1,
  reward: -1,
  nextObservation: [0.1 * (t + 1), 0, 1, 1],
  terminated: false,
}))

describe('hindsight relabelling', () => {
  it('relabels every step with the last achieved goal and rewards the steps that reached it', () => {
    const out = relabelEpisode(walk, { strategy: 'final' }, key(0))
    expect(out).toHaveLength(5)
    for (const s of out) expect(s.observation.slice(2)).toEqual([0.5, 0])
    expect(out.map((s) => s.reward)).toEqual([-1, -1, -1, -1, 0])
    expect(out.map((s) => s.terminated)).toEqual([false, false, false, false, true])
    expect(out[2]?.nextObservation).toEqual([0.30000000000000004, 0, 0.5, 0])
  })

  it('draws k goals per step from the achieved goals at or after the step (future) or from any step (episode)', () => {
    const future = relabelEpisode(walk, { strategy: 'future', k: 3 }, key(1))
    expect(future).toHaveLength(15)
    for (const [i, s] of future.entries()) {
      const t = Math.floor(i / 3)
      const goal = s.observation[2] ?? Number.NaN
      expect(goal).toBeGreaterThanOrEqual(0.1 * (t + 1) - 1e-9)
      expect(goal).toBeLessThanOrEqual(0.5 + 1e-9)
      // The reward is 0 exactly when the step's own outcome is the goal.
      expect(s.reward === 0, `step ${t} goal ${goal}`).toBe(Math.abs(goal - 0.1 * (t + 1)) < 0.075)
    }
    const episode = relabelEpisode(walk, { strategy: 'episode', k: 4 }, key(2))
    const goals = new Set(episode.map((s) => Math.round((s.observation[2] ?? 0) * 10)))
    expect([...goals].every((g) => g >= 1 && g <= 5)).toBe(true)
    expect(goals.size).toBeGreaterThan(2) // drawn from the whole episode, not only the end
    expect(relabelEpisode(walk, { strategy: 'future', k: 3 }, key(1))).toEqual(future)
    expect(() => relabelEpisode([], { strategy: 'final' }, key(0))).toThrow()
  })
})

describe('goal-conditioned Q-learning', () => {
  const coder = tileCoder({ low: [-1, -1], high: [1, 1], tilings: 1, tiles: 20, offsets: 'none' })
  const relative = (o: readonly number[]): number[] => [(o[2] ?? 0) - (o[0] ?? 0), (o[3] ?? 0) - (o[1] ?? 0)]
  const features = actionTileFeatures(coder, 4, relative)

  it('learns from the relabelled transitions of an episode that never reached its goal', () => {
    const env = make(reachGoal, { horizon: 5 })
    const t = (obs: readonly number[], next: readonly number[], end: boolean) => ({
      observation: obs,
      action: 1,
      reward: -1,
      nextObservation: next,
      terminated: false,
      truncated: end,
    })
    const play = (strategy: 'none' | 'future') => {
      const agent = goalConditionedQ(env, { features, alpha: 0.5, her: { strategy, k: 2 } })
      let s = agent.init(key(0))
      for (let i = 0; i < 4; i++) s = agent.learn(s, t([0.1 * i, 0, 1, 1], [0.1 * (i + 1), 0, 1, 1], i === 3), key(i))
      return s
    }
    const plain = play('none')
    const hindsight = play('future')
    expect(plain.relabelled).toBe(0)
    expect(hindsight.relabelled).toBe(8)
    expect(plain.successes).toBe(0)
    // The relabelled transitions touch more of the value function: the plain agent learnt only from the real goal.
    const touched = (w: Float64Array): number => w.filter((x) => x !== 0).length
    expect(touched(hindsight.w)).toBeGreaterThan(touched(plain.w))
  })

  it('raises the success rate on a task whose goals are rarely reached by chance', () => {
    const env = make(reachGoal, { horizon: 16, minDistance: 0.9 })
    const success = (strategy: 'none' | 'future', seed: number): number => {
      const agent = goalConditionedQ(env, {
        features,
        alpha: 0.5,
        gamma: 0.98,
        epsilon: 0.3,
        replayPasses: 3,
        her: { strategy, k: 4 },
      })
      const trained = trainAgent(env, agent, { episodes: 1500, checkpoints: false, seed })
      const ev = evaluatePolicy(env, greedyPolicyOf(agent, trained.agent), { episodes: 300, key: key(seed + 100) })
      return ev.returns.filter((x) => x > -16).length / ev.returns.length
    }
    const seeds = [0, 1, 2, 3]
    const plain = seeds.map((s) => success('none', s))
    const hindsight = seeds.map((s) => success('future', s))
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(hindsight)).toBeGreaterThan(mean(plain) + 0.05)
    expect(hindsight.filter((h, i) => h > (plain[i] ?? 0)).length).toBeGreaterThanOrEqual(3)
  }, 60_000)
})
