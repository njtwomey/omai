import { describe, expect, it } from 'vitest'
import { cartPole, frozenLake } from '../../methods/rl/index.ts'
import { key } from '../random/index.ts'
import { episodes, frames, recorder, resume, runAlgorithm, tracePolicy } from '../trace/index.ts'
import { make } from './gym.ts'
import { compareAgents, evaluatePolicy, rolloutAlgorithm, runEpisode } from './rollout.ts'

const balance = (o: readonly number[]): number => ((o[2] ?? 0) + 0.5 * (o[3] ?? 0) > 0 ? 1 : 0)

describe('rollouts', () => {
  it('evaluate policies: a balancing controller solves CartPole, pushing one way does not', () => {
    const env = make(cartPole)
    expect(evaluatePolicy(env, balance, { episodes: 5, seed: 0 }).meanReturn).toBe(500)
    const push = evaluatePolicy(env, () => 1, { episodes: 5, seed: 0 })
    expect(push.meanReturn).toBeLessThan(20)
    expect(push.stdReturn).toBeGreaterThanOrEqual(0)
  })

  it('record episodes and rendered frames by policy', () => {
    const rec = recorder()
    runAlgorithm(
      rolloutAlgorithm(make(cartPole), () => 0, { episodes: 2 }),
      {
        maxSteps: 1000,
        seed: 3,
        tracer: rec,
        policy: tracePolicy({ frames: true }),
      },
    )
    const eps = episodes(rec.run())
    expect(eps).toHaveLength(2)
    expect(eps.every((e) => e.terminated)).toBe(true)
    expect(frames(rec.run()).length).toBeGreaterThan(eps.reduce((n, e) => n + e.length, 0))
  })

  it('run single episodes as pure functions of a key', () => {
    const env = make(frozenLake)
    const random = (_: number, k: Parameters<typeof runEpisode>[2]): number => Math.floor(((k.words[0] % 4) + 4) % 4)
    const a = runEpisode(env, random, key(5), { frames: true })
    const b = runEpisode(env, random, key(5))
    expect(a.rewards).toEqual(b.rewards)
    expect(a.observations.length).toBe(a.length + 1)
    expect(a.frames?.length).toBe(a.length + 1)
    expect(a.terminated || a.truncated).toBe(true)
    expect(() =>
      runEpisode(make(cartPole, {}, { maxEpisodeSteps: null }), balance, key(0), { maxSteps: 5, strict: true }),
    ).toThrow(/no end/)
  })

  it('compare policies with common random numbers', () => {
    const env = make(cartPole)
    const r = compareAgents(env, { balance, left: () => 0, right: () => 1 }, { episodes: 4, seed: 1 })
    expect(r.best).toBe('balance')
    expect(r.paired).toHaveLength(3)
    const first = r.paired[0]
    expect(first?.a).toBe('balance')
    expect(first?.winRate).toBe(1)
    // The same policy against itself differs by exactly zero under common random numbers.
    const same = compareAgents(env, { x: balance, y: balance }, { episodes: 3 })
    expect(same.paired[0]?.meanDifference).toBe(0)
    expect(same.paired[0]?.standardError).toBe(0)
    expect(() => compareAgents(env, {})).toThrow(/no policies/)
  })

  it('resume a rollout from any state and get the same trajectory', () => {
    const rec = recorder()
    const alg = rolloutAlgorithm(make(cartPole), () => 1, { episodes: 3 })
    const full = runAlgorithm(alg, {
      maxSteps: 1000,
      seed: 2,
      tracer: rec,
      policy: tracePolicy({ checkpoints: false }),
    })
    const half = runAlgorithm(alg, { maxSteps: 7, seed: 2 })
    const rest = runAlgorithm(alg, { maxSteps: 1000, seed: 2, start: { state: half.state, step: 7 } })
    expect(rest.state.returns).toEqual(full.state.returns)
    expect(typeof resume).toBe('function')
  })
})
