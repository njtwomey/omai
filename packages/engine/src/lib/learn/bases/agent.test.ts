import { describe, expect, it } from 'vitest'
import { greedyTable, qLearning } from '../../agents/index.ts'
import { key } from '../../random/index.ts'
import { dict, real } from '../../space/index.ts'
import { Log, messages, metricSeries } from '../../trace/index.ts'
import { defineAgentMethod } from './agent.ts'

interface Hyper {
  readonly alpha: number
  readonly slip: number
  readonly unit: 'step' | 'episode'
}

const ACTIONS = ['up', 'right', 'down', 'left']

/** Q-learning on Russell and Norvig's 4 x 3 grid world (3 rows, 4 columns). */
const q = defineAgentMethod<Hyper, ReturnType<ReturnType<typeof qLearning>['init']>, number, number>({
  name: 'q-learning/grid-world',
  defaults: { alpha: 0.3, slip: 0.1, unit: 'step' },
  hyperSpace: (h) => dict({ alpha: real(0.01, 1, { default: h.alpha, log: true }) }),
  environment: (h) => ({ id: 'gridWorld', options: { slip: h.slip } }),
  agent: (env, h) => qLearning(env, { alpha: h.alpha, gamma: 0.95, epsilon: 0.2 }),
  schedule: (h) => ({ episodes: 250, maxEpisodeSteps: 50, evalEvery: 50, evalEpisodes: 10, unit: h.unit }),
  grid: () => ({ shape: [3, 4], actionNames: ACTIONS }),
})

describe('the agent base', () => {
  it('trains Q-learning on a grid world and logs episodes and evaluations', () => {
    const trained = q.train({ key: key(1) })
    expect(trained.status).toBe('completed')
    expect(trained.returns).toHaveLength(250)
    const returns = metricSeries(trained.run, 'episode/return').values
    expect(returns).toHaveLength(250)
    expect(metricSeries(trained.run, 'episode/length').values).toHaveLength(250)
    // The returns improve: the last fifty beat the first fifty.
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(returns.slice(-50))).toBeGreaterThan(mean(returns.slice(0, 50)))
    const evals = metricSeries(trained.run, 'eval/return')
    expect(evals.values).toHaveLength(5)
    expect(evals.values.at(-1)).toBeGreaterThan(0)
    // The agent's own statistics ride along as agent/ paths only when asked for.
    expect(metricSeries(trained.run, 'agent/epsilon').values).toEqual([])
    const withStats = q.train({ key: key(1), logs: { 'agent/tdError': Log({ every: 100 }) } })
    expect(metricSeries(withStats.run, 'agent/tdError').values.length).toBeGreaterThan(0)
    expect(trained.state.agent.Q.length).toBe(12 * 4)
  })

  it('seeks, forks and reads the agent at any step, with the same trajectory', () => {
    const s = q.session({ key: key(1) })
    s.advance({ steps: 400 })
    const early = s.agentAt(100)
    expect(early.steps).toBe(100)
    expect(s.agentAt(400).steps).toBe(400)
    const fork = s.fork(150)
    fork.advance({ steps: 250 })
    expect(Array.from(fork.state.agent.Q)).toEqual(Array.from(s.state.agent.Q))
    expect(s.policyAt(400)(0, key(0))).toBe(greedyTable(s.agentAt(400))[0])
    expect(s.stepsOf('rl/episode/end').length).toBeGreaterThan(5)
  })

  it('probes an evaluation episode as an episodes record and the value and action maps as valueGrids', () => {
    const s = q.session({ key: key(1), mode: 'eager', steps: 5000, logs: 'off' })
    const episode = s.probe(s.t, 'eval/episode', { seed: 3 }) as {
      readonly kind: string
      readonly recorded: readonly { readonly states: readonly unknown[]; readonly rewards: readonly number[] }[]
      readonly snapshotShape: readonly number[]
      readonly snapshots: readonly { readonly policy: readonly number[] }[]
    }
    expect(episode.kind).toBe('episodes')
    expect(episode.recorded[0]?.states).toHaveLength((episode.recorded[0]?.rewards.length ?? 0) + 1)
    expect(episode.snapshotShape).toEqual([3, 4])
    expect(episode.snapshots[0]?.policy).toHaveLength(12)
    const values = s.probe(s.t, 'agent/value-map') as {
      readonly kind: string
      readonly grids: readonly { readonly values: readonly (readonly number[])[] }[]
    }
    expect(values.kind).toBe('valueGrids')
    expect(values.grids[0]?.values).toHaveLength(3)
    expect(values.grids[0]?.values[0]).toHaveLength(4)
    const actions = s.probe(s.t, 'agent/action-map') as { readonly grids: readonly { readonly name: string }[] }
    expect(actions.grids.map((g) => g.name)).toEqual([...ACTIONS, 'greedy action'])
  })

  it('runs an episode to a step and rejects a schedule without a budget', () => {
    const byEpisode = q.with({ unit: 'episode' })
    const trained = byEpisode.train({ key: key(1) })
    expect(trained.run.events.filter((e) => e.type === 'metric' && e.name === 'episode/return')).toHaveLength(250)
    expect(messages(trained.run, 'rl/episode/end')).toEqual([])
    const bad = defineAgentMethod({
      name: 'no-budget',
      defaults: {},
      environment: () => 'gridWorld',
      agent: (env) => qLearning(env),
      schedule: () => ({}),
    })
    expect(() => bad.session()).toThrow(/schedule/)
  })
})
