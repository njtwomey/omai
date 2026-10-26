import { describe, expect, it } from 'vitest'
import { randint } from '../../lib/random/index.ts'
import { make, oneHotObservation, policyAgent } from '../../lib/rl/index.ts'
import { dict, real } from '../../lib/space/index.ts'
import { recorder } from '../../lib/trace/index.ts'
import { tinyQLearning } from '../../testing/rl-studies.ts'
import { dqn } from './agents/index.ts'
import { cliffWalking, frozenLake, levelSplit, proceduralMaze } from './environments/index.ts'
import {
  type AgentFactory,
  compareLearningCurves,
  explorationStudy,
  generalisationStudy,
  sensitivityStudy,
  visitCountingAgent,
} from './studies.ts'

const lake = () => make(frozenLake, { isSlippery: false })
const q =
  (options = {}): AgentFactory =>
  () =>
    tinyQLearning(4, { alpha: 0.5, gamma: 0.95, epsilon: 0.2, ...options }) as never
const randomWalk: AgentFactory = () => policyAgent((_, k) => randint(k, [], 0, 4).data[0] ?? 0, 'random') as never
const stuck: AgentFactory = () => policyAgent(() => 0, 'stuck') as never

describe('learning-curve comparisons', { timeout: 30_000 }, () => {
  it('train agents x seeds x tasks as child runs of one study and aggregate curves, scores and improvements', () => {
    const rec = recorder()
    const study = compareLearningCurves({
      agents: { q: q(), stuck },
      envs: [lake(), make(cliffWalking, {}, { maxEpisodeSteps: 100 })],
      normalization: [
        [0, 1],
        [-100, -13],
      ],
      seeds: 3,
      episodes: 40,
      reps: 200,
      tracer: rec,
    })
    expect(study.runs).toHaveLength(2 * 2 * 3)
    const runs = rec.events.filter((e) => e.type === 'run') as { runId: string; parent?: string; name: string }[]
    const root = runs.filter((r) => r.parent === undefined)
    expect(root.map((r) => r.name)).toEqual(['study:compareLearningCurves'])
    expect(runs.filter((r) => r.parent === root[0]?.runId)).toHaveLength(12)
    expect(study.scores['q']).toHaveLength(3)
    expect(study.scores['q']?.[0]).toHaveLength(2)
    expect(study.best).toBe('q')
    const p = study.improvement['q > stuck']
    expect(p?.value).toBeGreaterThan(0.5)
    expect(p?.value).toBeCloseTo(1 - (study.improvement['stuck > q']?.value ?? 0), 12)
    const iqm = study.aggregates['q']?.iqm
    expect(iqm?.interval[0]).toBeLessThanOrEqual(iqm?.value ?? 0)
    const curve = study.curves['q']
    expect(curve?.x).toHaveLength(40)
    expect(curve?.value[39]).toBeGreaterThan(curve?.value[0] ?? 0)
    expect(study.profiles['q']?.x).toHaveLength(11)
  })

  it('gives every agent the same keys for the same seed, so studies are reproducible', () => {
    const run = () =>
      compareLearningCurves({ agents: { a: q(), b: q() }, envs: [lake()], seeds: 2, episodes: 10, reps: 20 })
    const one = run()
    expect(one.scores['a']).toEqual(one.scores['b'])
    expect(run().runs).toEqual(one.runs)
  })

  it('plugs in a DQN under a step budget, with curves over environment steps', () => {
    const env = make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
    const study = compareLearningCurves({
      agents: { dqn: (e) => dqn(e as never, { hidden: [8], learningStarts: 16, batchSize: 8 }) as never, stuck },
      envs: [env],
      seeds: 2,
      maxSteps: 150,
      x: 'steps',
      points: 5,
      reps: 20,
    })
    expect(study.runs).toHaveLength(4)
    expect(study.curves['stuck']?.x).toHaveLength(1)
    expect(() => compareLearningCurves({ agents: { stuck }, envs: [env] })).toThrow(/episodes or maxSteps/)
  })
})

describe('hyperparameter sensitivity', { timeout: 30_000 }, () => {
  it('sweeps a grid of the space and summarises each setting with an interval', () => {
    const r = sensitivityStudy({
      agent: (c) => tinyQLearning(4, { ...(c as { alpha: number; epsilon: number }) }) as never,
      space: dict({ alpha: real(0.1, 0.9), epsilon: real(0.05, 0.5) }),
      points: 2,
      envs: [lake()],
      seeds: 3,
      episodes: 30,
      reps: 100,
    })
    expect(r.settings.map((s) => s.config)).toEqual([
      { alpha: 0.1, epsilon: 0.05 },
      { alpha: 0.1, epsilon: 0.5 },
      { alpha: 0.9, epsilon: 0.05 },
      { alpha: 0.9, epsilon: 0.5 },
    ])
    expect(r.runs).toHaveLength(12)
    for (const s of r.settings) {
      expect(s.scores).toHaveLength(3)
      expect(s.estimate.interval[0]).toBeLessThanOrEqual(s.estimate.value)
      expect(s.estimate.interval[1]).toBeGreaterThanOrEqual(s.estimate.value)
      expect(r.best.estimate.value).toBeGreaterThanOrEqual(s.estimate.value)
    }
  })
})

describe('exploration studies', { timeout: 30_000 }, () => {
  it('measure state coverage and time to the goal from training transitions', () => {
    const env = make(frozenLake, { mapName: '8x8', isSlippery: false })
    const r = explorationStudy({ agents: { q: q({ epsilon: 0.3 }), stuck }, env, maxSteps: 1500, seeds: 3, reps: 100 })
    const explorer = r.agents['q']
    const lazy = r.agents['stuck']
    if (explorer === undefined || lazy === undefined) throw new Error('missing summaries')
    expect(lazy.finalCoverage.value).toBe(1 / 64)
    expect(lazy.reached).toBe(0)
    expect(lazy.timeToGoal.value).toBe(1500)
    expect(explorer.finalCoverage.value).toBeGreaterThan(0.3)
    for (let i = 1; i < explorer.coverage.value.length; i++) {
      expect(explorer.coverage.value[i]).toBeGreaterThanOrEqual(explorer.coverage.value[i - 1] ?? 0)
    }
    expect(explorer.coverage.x[explorer.coverage.x.length - 1]).toBe(1500)
    expect(explorer.visits.reduce((a, b) => a + b, 0)).toBeGreaterThan(1500)
    expect(r.runs).toHaveLength(6)
    if (explorer.reached === 1) expect(explorer.timeToGoal.value).toBeLessThan(1500)
  })

  it('counts visits through a wrapped agent that acts and learns as before', () => {
    const inner = tinyQLearning(4)
    const counted = visitCountingAgent(inner, 16)
    expect(counted.name).toBe(inner.name)
    expect(() =>
      explorationStudy({
        agents: { stuck },
        env: make(cliffWalking, {}, { wrappers: [(e) => oneHotObservation(e as never) as never] }),
        maxSteps: 5,
      }),
    ).toThrow(/nStates/)
  })
})

describe('generalisation studies', { timeout: 30_000 }, () => {
  it('train on some procedural levels, evaluate on held-out ones and report the gap', () => {
    const split = levelSplit(2, 4)
    const r = generalisationStudy({
      agents: { q: q({ epsilon: 0.3, gamma: 0.9 }), randomWalk },
      definition: proceduralMaze,
      envOptions: { rows: 2, cols: 3, observation: 'index' },
      ...split,
      seeds: 2,
      episodes: 300,
      evalEpisodes: 10,
      reps: 100,
    })
    const tab = r.agents['q']
    if (tab === undefined) throw new Error('missing summary')
    // A table over level-specific states memorises its training levels and cannot transfer.
    expect(tab.train.value).toBeGreaterThanOrEqual(0.9)
    expect(tab.test.value).toBeLessThan(tab.train.value)
    expect(tab.gap.value).toBeCloseTo(tab.train.value - tab.test.value, 12)
    expect(r.runs).toHaveLength(4)
    expect(() =>
      generalisationStudy({
        agents: { randomWalk },
        definition: proceduralMaze,
        train: { startLevel: 0, numLevels: 5 },
        test: { startLevel: 3, numLevels: 5 },
        episodes: 1,
      }),
    ).toThrow(/disjoint/)
  })
})
