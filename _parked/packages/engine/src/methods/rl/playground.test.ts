import { describe, expect, it, vi } from 'vitest'
import { evaluatePolicy } from '../../lib/rl/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'
import { frozenLake } from './environments/index.ts'
import { playground, playgroundPairing, playgroundPairings, playgroundSession, replayRecorded } from './playground.ts'

// Each default trains in well under a second alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 30_000 })

describe('the playground', () => {
  for (const pairing of playgroundPairings)
    it(`${pairing.id}: the defaults improve the return`, () => {
      const run = playground({ pairing: pairing.id, seed: 0 })
      expect(run.after).toBeGreaterThan(run.before)
      const [record] = run.exposition
      expect(record?.kind).toBe('episodes')
      if (record?.kind !== 'episodes') return
      expect(record.returns).toHaveLength(run.episodes)
      expect(record.recorded?.length).toBeGreaterThanOrEqual(2)
      for (const r of record.recorded ?? []) expect(r.states).toHaveLength(r.rewards.length + 1)
      // The last finished episode is always recorded, and its rewards sum to its return.
      const last = record.recorded?.at(-1)
      expect(last?.episode).toBe(run.episodes - 1)
      expect(last?.rewards.reduce((a, b) => a + b, 0)).toBeCloseTo(run.returns.at(-1) ?? Number.NaN, 9)
      if (pairing.family !== 'classic-control' || pairing.id.startsWith('mountain-car')) {
        expect(record.snapshots?.[0]?.episode).toBe(0)
        expect(record.snapshots?.at(-1)?.episode).toBe(run.episodes)
        // The snapshot's layout: a grid world's rows and columns, a bandit's one row of arms.
        const [rows, cols] = record.snapshotShape ?? [0, 0]
        expect(rows * cols).toBe(record.snapshots?.at(-1)?.values?.length)
      }
    })

  it('learns the textbook amounts on the cliff, the lake and CartPole', () => {
    expect(playground({ pairing: 'cliff-walking/sarsa', seed: 0 }).after).toBeGreaterThan(-40)
    expect(playground({ pairing: 'frozen-lake/q-learning', seed: 0 }).after).toBeGreaterThan(0.8)
    expect(playground({ pairing: 'cart-pole/reinforce', seed: 0 }).after).toBeGreaterThan(150)
    expect(playground({ pairing: 'mountain-car/sarsa-lambda', seed: 0 }).after).toBeGreaterThan(-180)
  })

  it('steps as an algorithm, reports progress, and records frames on request', () => {
    const seen: number[] = []
    const session = playgroundSession({
      environment: 'cliff-walking',
      agent: 'q-learning',
      episodes: 6,
      record: [1],
      frames: true,
      onEpisode: (p) => seen.push(p.episode),
    })
    const half = runAlgorithm(session.algorithm, { maxSteps: 50, seed: 3 })
    expect(session.record(half.state).returns.length).toBeLessThan(6)
    const full = runAlgorithm(session.algorithm, { maxSteps: 1_000_000, seed: 3 })
    expect(full.status).toBe('completed')
    expect(seen).toEqual([1, 2, 3, 4, 5, 6])
    const episodes = session.episodesOf(full.state)
    expect(episodes.map((e) => e.episode)).toEqual([1, 5])
    for (const e of episodes) expect(e.frames).toHaveLength(e.states.length)
    const same = playground({ pairing: 'cliff-walking/q-learning', episodes: 6, seed: 3, frames: true, record: [1] })
    expect(same.returns).toEqual(full.state.train.returns)
    expect(same.frames?.map((f) => f.length)).toEqual(episodes.map((e) => e.states.length))
  })

  it('looks pairings up by id or by names, and rejects unknown ones', () => {
    expect(playgroundPairing('bandit', 'ucb1').id).toBe('bandit/ucb1')
    expect(() => playgroundPairing('cliff-walking/dqn')).toThrow(/no pairing/)
    expect(() => playground({})).toThrow(/give a pairing/)
    expect(new Set(playgroundPairings.map((p) => p.id)).size).toBe(playgroundPairings.length)
  })
})

describe('replaying a recorded episode', () => {
  it('reproduces the recorded rewards and draws the recorded states', () => {
    const run = playground({ pairing: 'frozen-lake/value-iteration', episodes: 20, seed: 1 })
    const [record] = run.exposition
    const last = record?.kind === 'episodes' ? record.recorded?.at(-1) : undefined
    if (last === undefined) throw new Error('no recorded episode')
    const { env, policy } = replayRecorded('frozen-lake/value-iteration', last)
    const ev = evaluatePolicy(env, policy, { episodes: 1, seed: 0 })
    expect(ev.lengths).toEqual([last.actions.length])
    expect(ev.meanReturn).toBe(run.returns.at(-1))
    expect(env.render(0)).toEqual(frozenLake.create().render(0))
  })
})
