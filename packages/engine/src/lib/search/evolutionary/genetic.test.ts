import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { metricSeries, policies, recorder, validateAgainstSchema } from '../../trace/index.ts'
import {
  arithmeticPrimitives,
  bitStrings,
  evaluateTree,
  evolutionStrategy,
  formatTree,
  geneticAlgorithm,
  geneticAlgorithmAlgorithm,
  geneticProgramming,
  permutations,
  pointMutation,
  rampedHalfAndHalf,
  randomTree,
  realVectors,
  regressionFitness,
  subtreeCrossover,
  subtreeMutation,
  treeDepth,
  treeSize,
  trees,
} from '../index.ts'

const onemax = (g: readonly number[]): number => g.reduce((a, b) => a + b, 0)

/** A small travelling-salesman instance: cities on a circle, so the optimal tour visits them in angular order. */
const cities = [0, 3, 6, 1, 4, 7, 2, 5].map((k) => [Math.cos((2 * Math.PI * k) / 8), Math.sin((2 * Math.PI * k) / 8)])
const tourLength = (p: readonly number[]): number =>
  p.reduce((s, c, i) => {
    const a = cities[c] ?? [0, 0]
    const b = cities[p[(i + 1) % p.length] ?? 0] ?? [0, 0]
    return s + Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0))
  }, 0)
const optimum = 8 * 2 * Math.sin(Math.PI / 8)

describe('genetic algorithms', () => {
  it('solve OneMax with every selection and crossover', () => {
    for (const selection of ['tournament', 'roulette', 'rank', 'universal'] as const) {
      for (const crossover of ['one-point', 'two-point', 'uniform'] as const) {
        const r = geneticAlgorithm(bitStrings(16, { crossover }), onemax, {
          selection,
          target: 16,
          generations: 300,
          seed: 1,
        })
        expect(r.fitness).toBeGreaterThanOrEqual(15)
        for (let i = 1; i < r.history.length; i++) expect(r.history[i]).toBeGreaterThanOrEqual(r.history[i - 1] ?? 0)
      }
    }
  })

  it('find the optimal tour of a small travelling-salesman problem', () => {
    for (const crossover of ['order', 'pmx', 'cycle'] as const) {
      const r = geneticAlgorithm(permutations(8, { crossover }), (p) => -tourLength(p), {
        populationSize: 80,
        generations: 150,
        target: -optimum + 1e-9,
        seed: 2,
      })
      expect(-r.fitness).toBeCloseTo(optimum, 9)
    }
  })

  it('optimise real vectors with every crossover and mutation', () => {
    const sphere = (x: readonly number[]): number => -x.reduce((a, v) => a + (v - 0.3) ** 2, 0)
    for (const crossover of ['blend', 'sbx', 'arithmetic'] as const) {
      for (const mutation of ['gaussian', 'polynomial'] as const) {
        const r = geneticAlgorithm(realVectors([-1, -1, -1], [1, 1, 1], { crossover, mutation }), sphere, {
          generations: 150,
          seed: 3,
        })
        expect(r.fitness).toBeGreaterThan(-0.01)
        for (const x of r.best) expect(Math.abs(x)).toBeLessThanOrEqual(1)
      }
    }
  })

  it('traces generations and rejects bad options', () => {
    const algorithm = geneticAlgorithmAlgorithm(bitStrings(8), onemax)
    const rec = recorder()
    geneticAlgorithm(bitStrings(8), onemax, { generations: 5, tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
    expect(metricSeries(rec.run(), 'best').values.length).toBe(6)
    expect(() => geneticAlgorithm(bitStrings(4), onemax, { populationSize: 1 })).toThrow()
    expect(() => geneticAlgorithm(bitStrings(4), onemax, { populationSize: 4, elitism: 4 })).toThrow()
    expect(() => realVectors([0], [0, 1])).toThrow()
    expect(() => realVectors([1], [0])).toThrow()
  })
})

describe('evolution strategies', () => {
  const sphere = (x: readonly number[]): number => -x.reduce((a, v) => a + v * v, 0)
  it('converge on the sphere in comma and plus forms with each recombination', () => {
    for (const plus of [false, true]) {
      for (const recombination of ['intermediate', 'discrete', 'none'] as const) {
        const r = evolutionStrategy(sphere, [2, -1, 3, 0.5], { plus, recombination, generations: 300, seed: 4 })
        expect(r.fitness).toBeGreaterThan(-1e-3)
        expect(r.best.sigma).toBeGreaterThan(0)
      }
    }
    expect(evolutionStrategy(sphere, [1], { target: -0.5, seed: 0 }).status).toBe('target')
    expect(() => evolutionStrategy(sphere, [1], { mu: 10, lambda: 5 })).toThrow()
  })
})

describe('genetic programming', () => {
  const set = { functions: arithmeticPrimitives(), variables: ['x'], constants: { low: -1, high: 1 } }

  it('builds, measures, prints and varies trees within depth limits', () => {
    const full = randomTree(key(1), set, { depth: 4, method: 'full' })
    expect(treeDepth(full)).toBe(4)
    expect(treeSize(full)).toBe(31)
    const ramp = rampedHalfAndHalf(key(2), set, 20, { minDepth: 1, maxDepth: 3 })
    for (const t of ramp) expect(treeDepth(t)).toBeLessThanOrEqual(3)
    expect(formatTree({ op: 'mul', children: [{ variable: 'x' }, { constant: 0.5 }] })).toBe('(mul x 0.5)')
    for (let s = 0; s < 30; s++) {
      const [a, b] = subtreeCrossover(key(s), full, ramp[s % 20] ?? full, { maxDepth: 5 })
      expect(treeDepth(a)).toBeLessThanOrEqual(5)
      expect(treeDepth(b)).toBeLessThanOrEqual(5)
      expect(treeDepth(subtreeMutation(key(s), full, set, { maxDepth: 5 }))).toBeLessThanOrEqual(5)
      const p = pointMutation(key(s), full, set, 0.5)
      expect(treeSize(p)).toBe(treeSize(full))
      expect(Number.isFinite(evaluateTree(p, set.functions, { x: 0.7 }))).toBe(true)
    }
    expect(trees(set).random(key(0))).toBeDefined()
    expect(() => evaluateTree({ variable: 'y' }, set.functions, { x: 1 })).toThrow()
    expect(() => evaluateTree({ op: 'pow', children: [] }, set.functions, {})).toThrow()
    expect(() => randomTree(key(0), { functions: set.functions, variables: [] })).toThrow()
  })

  it('rediscovers a polynomial by symbolic regression', () => {
    const xs = [-3, -2, -1, 0, 1, 2, 3].map((x) => ({ x }))
    const fitness = regressionFitness(
      set.functions,
      xs,
      xs.map(({ x }) => x * x * x - x),
    )
    const r = geneticProgramming({ functions: set.functions, variables: ['x'] }, fitness, {
      populationSize: 300,
      generations: 60,
      target: -1e-9,
      seed: 7,
    })
    expect(r.status).toBe('target')
    expect(evaluateTree(r.best, set.functions, { x: 4 })).toBeCloseTo(60, 9)
  })
})
