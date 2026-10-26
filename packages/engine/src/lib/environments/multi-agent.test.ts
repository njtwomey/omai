import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { isEnvDefinition, parallelTimeLimit, parallelToAec, playParallel, singleAgentView } from '../rl/index.ts'
import { contains } from '../space/index.ts'
import {
  cooperativeNavigation,
  MATRIX_GAMES,
  MPE,
  matrixGame,
  navigationCoverage,
  particleStep,
  pursuitEvasion,
} from './multi-agent.ts'

describe('matrix games', () => {
  it('pay the bimatrix and observe the previous joint action', () => {
    const env = matrixGame.create({ game: 'prisonersDilemma' })
    expect(isEnvDefinition(matrixGame)).toBe(false)
    const r0 = env.reset(key(0))
    expect(r0.observations).toEqual({ player_0: 0, player_1: 0 })
    const r = env.step(r0.state, { player_0: 1, player_1: 0 }, key(1))
    expect(r.rewards).toEqual({ player_0: 5, player_1: 0 })
    expect(r.observations['player_0']).toBe(1 + 1 * 2 + 0)
    expect(env.spec.maxEpisodeSteps).toBeUndefined()
    expect(env.agents(r.state)).toEqual(['player_0', 'player_1'])
    expect(() => env.step(r0.state, { player_0: 2, player_1: 0 }, key(1))).toThrow(/range/)
    expect(() => matrixGame.create({ game: 'chess' })).toThrow(/no game/)
  })

  it('work with the single-agent view, time limits and turn-based views', () => {
    const view = singleAgentView(matrixGame.create({ game: 'prisonersDilemma' }), 'player_0', { player_1: () => 0 })
    expect(view.step(view.reset(key(0)).state, 1, key(1)).reward).toBe(5)
    const limited = parallelTimeLimit(matrixGame.create({ game: 'prisonersDilemma' }), 3)
    expect(playParallel(limited, { player_0: () => 0, player_1: () => 0 }, key(0)).returns).toEqual({
      player_0: 9,
      player_1: 9,
    })
    const aec = parallelToAec(matrixGame.create({ game: 'prisonersDilemma' }))
    expect(aec.currentAgent(aec.reset(key(0)))).toBe('player_0')
  })

  it('offer the classic games, custom payoffs and a round limit', () => {
    expect(MATRIX_GAMES['rockPaperScissors']?.[1]?.[0]).toEqual([1, -1])
    const pennies = matrixGame.create({ game: 'matchingPennies' })
    expect(pennies.step(pennies.reset(key(0)).state, { player_0: 1, player_1: 1 }, key(0)).rewards).toEqual({
      player_0: 1,
      player_1: -1,
    })
    const custom = matrixGame.create({
      payoffs: [
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      ],
      rounds: 2,
    })
    expect(custom.actionSpaces['player_1']).toEqual({ kind: 'discrete', n: 3 })
    const ep = playParallel(custom, { player_0: () => 0, player_1: () => 2 }, key(0))
    expect(ep.returns).toEqual({ player_0: 10, player_1: 12 })
    expect(ep.length).toBe(2)
    expect(custom.render(ep.states[1] ?? { t: 0 }).kind).toBe('scene')
  })
})

describe('pursuit-evasion', () => {
  it('moves everyone at once, rewards tags and catches, and ends when every evader is caught', () => {
    const env = pursuitEvasion.create({ size: 5, pursuers: 2, evaderControl: 'agent' })
    const s = { pursuers: [[0, 0] as const, [4, 4] as const], evaders: [[0, 2] as const], free: [true], t: 0 }
    // pursuer_0 moves right to (0, 1), next to the evader, which stays.
    const r = env.step(s, { pursuer_0: 4, pursuer_1: 0, evader_0: 0 }, key(0))
    expect(r.state.free).toEqual([false])
    expect(r.rewards['pursuer_0']).toBeCloseTo(-0.1 + 0.01 + 5, 12)
    expect(r.rewards['pursuer_1']).toBeCloseTo(-0.1, 12)
    expect(r.rewards['evader_0']).toBeCloseTo(0.1 - 5, 12)
    expect(r.terminations).toEqual({ pursuer_0: true, pursuer_1: true, evader_0: true })
    expect(env.agents(r.state)).toEqual([])
    // With nCatch 2 one pursuer is not enough.
    const two = pursuitEvasion.create({ size: 5, nCatch: 2, evaderControl: 'agent' })
    expect(two.step(s, { pursuer_0: 4, pursuer_1: 0, evader_0: 0 }, key(0)).state.free).toEqual([true])
  })

  it('observes relative positions or a local window, inside the spaces, and truncates', () => {
    for (const observation of ['vector', 'local'] as const) {
      const env = pursuitEvasion.create({ observation, evaderControl: 'flee', maxCycles: 4 })
      const r = env.reset(key(2))
      for (const a of env.agents(r.state)) {
        const space = env.observationSpaces[a]
        expect(space !== undefined && contains(space, r.observations[a] as never)).toBe(true)
      }
      const ep = playParallel(env, { pursuer_0: () => 0, pursuer_1: () => 0 }, key(3))
      expect(ep.length).toBeLessThanOrEqual(4)
      expect(env.render(r.state).kind).toBe('scene')
    }
    const env = pursuitEvasion.create({ size: 5 })
    const s = { pursuers: [[0, 0] as const, [4, 4] as const], evaders: [[2, 2] as const], free: [true], t: 0 }
    const obs = env.step(s, { pursuer_0: 0, pursuer_1: 0 }, key(0)).observations['pursuer_0'] ?? []
    expect(obs.slice(0, 4)).toEqual([0, 0, 4 / 5, 4 / 5])
    expect(() => pursuitEvasion.create({ obsRange: 4 })).toThrow(/odd/)
    expect(() => pursuitEvasion.create({ size: 3, pursuers: 8, evaders: 2 })).toThrow(/too many/)
  })

  it('lets a fleeing evader run from the pursuers', () => {
    const env = pursuitEvasion.create({ size: 5, pursuers: 1, evaderControl: 'flee', captureRadius: 0 })
    const s = { pursuers: [[2, 1] as const], evaders: [[2, 2] as const], free: [true], t: 0 }
    expect(env.step(s, { pursuer_0: 0 }, key(0)).state.evaders[0]).toEqual([1, 2])
  })
})

describe('cooperative navigation', () => {
  it('integrates damped particle dynamics with soft contact forces', () => {
    const free = particleStep([[0, 0]], [[1, 0]], [[0, 1]])
    expect(free.velocities[0]?.[0]).toBeCloseTo(0.75, 15)
    expect(free.velocities[0]?.[1]).toBeCloseTo(MPE.sensitivity * MPE.dt, 15)
    const overlap = particleStep(
      [
        [0, 0],
        [0.1, 0],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [0, 0],
      ],
    )
    expect(overlap.velocities[0]?.[0]).toBeLessThan(0)
    expect(overlap.velocities[1]?.[0]).toBeCloseTo(-(overlap.velocities[0]?.[0] ?? 0), 12)
  })

  it('shares minus the summed nearest distances and penalises collisions', () => {
    const env = cooperativeNavigation.create({ agents: 2 })
    const s = {
      positions: [[0, 0] as const, [0.25, 0] as const],
      velocities: [[0, 0] as const, [0, 0] as const],
      landmarks: [[1, 0] as const, [0, 1] as const],
      t: 0,
    }
    // Pushing towards each other balances the contact force (penetration 0.05), so they stay overlapped.
    const r = env.step(s, { agent_0: 2, agent_1: 1 }, key(0))
    const team = navigationCoverage(r.state)
    expect(r.infos['agent_0']).toEqual({ teamReward: team, collisions: 1 })
    expect(r.rewards['agent_0']).toBeCloseTo(0.5 * team - 0.5, 12)
    const obs = r.observations['agent_0'] ?? []
    expect(obs.length).toBe(4 + 4 + 2 + 2)
  })

  it('takes discrete or continuous actions and truncates after the cycles', () => {
    const env = cooperativeNavigation.create({ continuous: true, maxCycles: 3 })
    expect(env.actionSpaces['agent_0']).toMatchObject({ kind: 'box', shape: [2] })
    const ep = playParallel(env, { agent_0: () => [1, 0], agent_1: () => [0, 1], agent_2: () => [-1, 0] }, key(0))
    expect(ep.length).toBe(3)
    const r = env.reset(key(4))
    for (const l of r.state.landmarks) expect(Math.max(Math.abs(l[0]), Math.abs(l[1]))).toBeLessThanOrEqual(0.8)
    expect(env.render(r.state).kind).toBe('scene')
  })
})
