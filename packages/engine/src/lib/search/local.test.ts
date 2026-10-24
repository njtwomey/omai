import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { attackingPairs, queensLocal } from '../../testing/search.ts'
import { key } from '../random/index.ts'
import { dict, int } from '../space/index.ts'
import { metricSeries, policies, recorder, runTree, validateAgainstSchema } from '../trace/index.ts'
import {
  bitFlipNeighbours,
  exponentialSchedule,
  geometricSchedule,
  hillClimbing,
  hillClimbingAlgorithm,
  lateAcceptance,
  localBeamSearch,
  logarithmicSchedule,
  permutationNeighbours,
  randomRestartHillClimbing,
  simulatedAnnealing,
  simulatedAnnealingAlgorithm,
  spaceProblem,
  tabuSearch,
} from './index.ts'

const start = [0, 1, 2, 3, 4, 5, 6, 7]

describe('local search on eight queens', () => {
  it('climbs to a local optimum with no better neighbour, never descending', () => {
    const p = queensLocal(8, start)
    const r = hillClimbing(p)
    expect(['goal', 'local-optimum']).toContain(r.status)
    if (r.status === 'local-optimum') {
      for (const n of p.neighbours(r.state)) expect(p.value(n)).toBeLessThanOrEqual(r.value)
    }
    for (let i = 1; i < r.history.length; i++) expect(r.history[i]).toBeGreaterThanOrEqual(r.history[i - 1] ?? 0)
    for (const variant of ['first-choice', 'stochastic'] as const) {
      const v = hillClimbing(p, { variant, seed: 3 })
      expect(v.value).toBeGreaterThanOrEqual(p.value(start))
    }
  })

  it('solves with sideways moves, restarts, annealing, tabu, beams and late acceptance', () => {
    const p = queensLocal(8)
    const restart = randomRestartHillClimbing(p, { restarts: 200, sideways: 100, seed: 1 })
    expect(restart.status).toBe('goal')
    expect(attackingPairs(restart.state)).toBe(0)
    const tabu = tabuSearch(queensLocal(8, start), { tenure: 10, maxIterations: 500 })
    expect(tabu.status).toBe('goal')
    const late = lateAcceptance(queensLocal(8, start), { length: 20, maxIterations: 20_000, seed: 5 })
    expect(late.status).toBe('goal')
    const anneal = simulatedAnnealing(queensLocal(8, start), {
      schedule: geometricSchedule(2, 0.999, 1e-3),
      maxIterations: 20_000,
      seed: 2,
    })
    expect(anneal.value).toBeGreaterThanOrEqual(-1)
    const beam = localBeamSearch(p, { k: 10, seed: 3 })
    expect(beam.value).toBeGreaterThanOrEqual(-2)
    const stochastic = localBeamSearch(p, { k: 10, stochastic: true, temperature: 0.5, seed: 3, maxIterations: 50 })
    expect(Number.isFinite(stochastic.value)).toBe(true)
  })

  it('traces moves and restarts as child runs', () => {
    const rec = recorder()
    randomRestartHillClimbing(queensLocal(6), { restarts: 3, tracer: rec, policy: policies.teaching, seed: 0 })
    const tree = runTree(rec.run())
    expect(tree.length).toBe(1)
    expect(tree[0]?.children.length).toBeGreaterThan(0)
    const algorithm = simulatedAnnealingAlgorithm(queensLocal(6, [0, 1, 2, 3, 4, 5]))
    const rec2 = recorder()
    simulatedAnnealing(queensLocal(6, [0, 1, 2, 3, 4, 5]), { tracer: rec2, policy: policies.teaching, seed: 1 })
    expect(validateAgainstSchema(rec2.events, algorithm.schema ?? (undefined as never))).toEqual([])
    expect(metricSeries(rec2.run(), 'temperature').values[1]).toBeCloseTo(20, 12)
    expect(hillClimbingAlgorithm(queensLocal(4, [0, 0, 0, 0])).schema).toBeDefined()
  })

  it('rejects bad options and problems without a start', () => {
    expect(() => hillClimbing({ neighbours: () => [], value: () => 0 })).toThrow()
    expect(() => hillClimbing(queensLocal(4, [0, 1, 2, 3]), { sideways: -1 })).toThrow()
    expect(() => randomRestartHillClimbing({ initial: 0, neighbours: () => [], value: () => 0 })).toThrow()
    expect(() => localBeamSearch(queensLocal(4), { k: 0 })).toThrow()
    expect(() => lateAcceptance(queensLocal(4, [0, 1, 2, 3]), { length: 0 })).toThrow()
    expect(() => geometricSchedule(1, 1)).toThrow()
  })
})

describe('schedules, neighbourhoods and spaces', () => {
  it('cools as specified', () => {
    const T = exponentialSchedule({ k: 10, lambda: 0.1, limit: 5 })
    expect(T(1)).toBeCloseTo(10 * Math.exp(-0.1), 14)
    expect(T(5)).toBe(0)
    expect(geometricSchedule(1, 0.5, 0.2)(3)).toBe(0)
    expect(logarithmicSchedule(1, 3)(3)).toBe(0)
  })

  it('generates valid permutation and bit neighbourhoods', () => {
    fc.assert(
      fc.property(fc.integer({ min: 2, max: 7 }), (n) => {
        const perm = Array.from({ length: n }, (_, i) => i)
        const sorted = (p: readonly number[]): number[] => [...p].sort((a, b) => a - b)
        for (const move of ['swap', 'two-opt', 'insert'] as const) {
          const ns = permutationNeighbours(perm, move)
          for (const q of ns) expect(sorted(q)).toEqual(perm)
          expect(new Set(ns.map((q) => q.join())).size).toBe(ns.length)
          expect(ns.some((q) => q.join() === perm.join())).toBe(false)
        }
        expect(permutationNeighbours(perm, 'swap').length).toBe((n * (n - 1)) / 2)
        expect(bitFlipNeighbours(perm.map(() => 0)).length).toBe(n)
      }),
    )
  })

  it('searches a configuration space', () => {
    const space = dict({ a: int(0, 20), b: int(0, 20) })
    const f = (v: unknown): number => {
      const { a, b } = v as { a: number; b: number }
      return -((a - 13) ** 2) - (b - 4) ** 2
    }
    const p = spaceProblem(space, f)
    const r = randomRestartHillClimbing({ ...p, isGoal: (v) => f(v) === 0 }, { restarts: 5, seed: 4 })
    expect(r.state).toEqual({ a: 13, b: 4 })
    expect(p.random?.(key(0))).toBeDefined()
  })
})
