import { describe, expect, it } from 'vitest'
import { AGENTS, coordination, takeAway } from '../../testing/rl.ts'
import { key } from '../random/index.ts'
import {
  aecToParallel,
  parallelTimeLimit,
  parallelToAec,
  playAec,
  playParallel,
  singleAgentView,
} from './multi-agent.ts'

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
