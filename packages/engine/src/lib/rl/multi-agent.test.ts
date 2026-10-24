import { describe, expect, it } from 'vitest'
import { AGENTS, coordination, takeAway } from '../../testing/rl.ts'
import { key } from '../random/index.ts'
import {
  aecSingleAgentView,
  aecToParallel,
  jointEnv,
  parallelTimeLimit,
  parallelToAec,
  playAec,
  playParallel,
  singleAgentView,
  stateObservations,
} from './multi-agent.ts'

describe('joint and single-agent views', () => {
  it('turn a parallel environment into one env over joint observations and actions', () => {
    const env = jointEnv(parallelTimeLimit(coordination, 2))
    const r0 = env.reset(key(0))
    expect(r0.observation).toEqual({ player_0: 0, player_1: 0 })
    const r1 = env.step(r0.state, { player_0: 1, player_1: 1 }, key(1))
    expect(r1.reward).toBe(2)
    expect(r1.info['rewards']).toEqual({ player_0: 1, player_1: 1 })
    expect(r1.truncated).toBe(false)
    const r2 = env.step(r1.state, { player_0: 0, player_1: 1 }, key(2))
    expect(r2.reward).toBe(0)
    expect([r2.terminated, r2.truncated]).toEqual([false, true])
    expect(env.actionSpace.kind).toBe('dict')
    const mean = jointEnv(coordination, { reward: 'mean' })
    expect(mean.step(mean.reset(key(0)).state, { player_0: 1, player_1: 1 }, key(1)).reward).toBe(1)
    expect(jointEnv(coordination, { reward: { agent: 'player_1' } }).spec.id).toBe('Coordination-v0/joint')
    expect(() => jointEnv(coordination, { reward: { agent: 'nobody' } })).toThrow(/no agent/)
    expect(() => env.step(r0.state, { player_0: 1 }, key(1))).toThrow(/no action/)
  })

  it('give one player of a turn-based game a single-agent view with the opponents moving in between', () => {
    // takeAway(4): we take one, the opponent takes one, we take one, the opponent takes the last.
    const env = aecSingleAgentView(takeAway(4), 'player_0', { player_1: () => 0 })
    const r0 = env.reset(key(0))
    expect(r0.observation).toBe(4)
    expect(r0.info['legalActions']).toEqual([0, 1])
    const r1 = env.step(r0.state, 0, key(1))
    expect([r1.observation, r1.reward, r1.terminated]).toEqual([2, 0, false])
    const r2 = env.step(r1.state, 0, key(2))
    expect([r2.reward, r2.terminated]).toEqual([-1, true])
    // As the second player the opponent opens at reset.
    const second = aecSingleAgentView(takeAway(3), 'player_1', { player_0: () => 0 })
    const s0 = second.reset(key(0))
    expect(s0.observation).toBe(2)
    expect(second.step(s0.state, 1, key(1)).reward).toBe(1)
    // When the opponent's opening move ends the game, the next step is the final one and pays the agent's reward.
    const lost = aecSingleAgentView(takeAway(1), 'player_1', { player_0: () => 0 })
    const l0 = lost.reset(key(0))
    expect(l0.info).toEqual({})
    expect(lost.step(l0.state, 0, key(1))).toMatchObject({ reward: -1, terminated: true, truncated: false })
    expect(() => aecSingleAgentView(takeAway(3), 'nobody', {})).toThrow(/no agent/)
    expect(() => aecSingleAgentView(takeAway(3), 'player_1', {}).reset(key(0))).toThrow(/no policy/)
  })

  it('expose full states as observations', () => {
    const env = stateObservations(takeAway(3))
    expect(env.observe(env.reset(key(0)), 'player_0')).toEqual({ tokens: 3, mover: 0 })
  })
})

describe('turn-based environments', () => {
  it('play episodes with a policy per agent', () => {
    const ep = playAec(takeAway(4), { player_0: () => 0, player_1: () => 0 }, key(0))
    // Four single takes: player_1 takes the last token.
    expect(ep.agents).toEqual(['player_0', 'player_1', 'player_0', 'player_1'])
    expect(ep.returns).toEqual({ player_0: -1, player_1: 1 })
    expect(() => playAec(takeAway(2), { player_0: () => 0 }, key(0))).toThrow(/no policy/)
  })

  it('convert to a parallel environment that cycles through the movers', () => {
    const par = aecToParallel(takeAway(3))
    const r = par.reset(key(0))
    expect(par.agents(r.state)).toEqual(AGENTS)
    const s = par.step(r.state, { player_0: 0, player_1: 0 }, key(1))
    expect(s.state.inner.tokens).toBe(1)
    const end = par.step(s.state, { player_0: 0, player_1: 0 }, key(2))
    expect(end.rewards).toEqual({ player_0: 1, player_1: -1 })
    expect(end.terminations['player_0']).toBe(true)
  })
})

describe('simultaneous environments', () => {
  it('play with a time limit and convert to turns', () => {
    const env = parallelTimeLimit(coordination, 3)
    const ep = playParallel(env, { player_0: () => 1, player_1: () => 1 }, key(0))
    expect(ep.returns).toEqual({ player_0: 3, player_1: 3 })
    expect(ep.length).toBe(3)
    const aec = parallelToAec(coordination)
    let s = aec.reset(key(0))
    expect(aec.currentAgent(s)).toBe('player_0')
    const first = aec.step(s, 1, key(1))
    expect(first.rewards).toEqual({})
    s = first.state
    expect(aec.currentAgent(s)).toBe('player_1')
    const second = aec.step(s, 1, key(2))
    expect(second.rewards).toEqual({ player_0: 1, player_1: 1 })
  })

  it('give one agent a single-agent view with the others on fixed policies', () => {
    const env = singleAgentView(coordination, 'player_1', { player_0: () => 0 })
    const s = env.reset(key(0)).state
    expect(env.step(s, 0, key(1)).reward).toBe(1)
    expect(env.step(s, 1, key(1)).reward).toBe(0)
    expect(env.spec.id).toBe('Coordination-v0/player_1')
    expect(() => singleAgentView(coordination, 'nobody', {})).toThrow(/no agent/)
    expect(() => singleAgentView(coordination, 'player_1', {}).step(s, 0, key(0))).toThrow(/no policy/)
  })
})
