import { describe, expect, it } from 'vitest'
import { policies, recorder, tensorSeries, validateAgainstSchema } from '../trace/index.ts'
import {
  antColonyAlgorithm,
  antColonyOptimization,
  discreteParticleSwarm,
  discreteParticleSwarmAlgorithm,
  permutationNeighbours,
} from './index.ts'

/** Ten cities on a circle in scrambled order: the optimal tour has length 20 sin(pi / 10). */
const order = [0, 7, 3, 9, 5, 1, 8, 2, 6, 4]
const points = order.map((k) => [Math.cos((2 * Math.PI * k) / 10), Math.sin((2 * Math.PI * k) / 10)] as const)
const distances = points.map((a) => points.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])))
const optimum = 20 * Math.sin(Math.PI / 10)

/** The optimal tour length by exhaustive search over tours from city 0 (for small asymmetric instances). */
function bruteForce(d: readonly (readonly number[])[]): number {
  const n = d.length
  let best = Number.POSITIVE_INFINITY
  const rest = Array.from({ length: n - 1 }, (_, i) => i + 1)
  const walk = (tour: number[], left: number[]): void => {
    if (left.length === 0) {
      let s = 0
      for (let i = 0; i < n; i++) s += d[tour[i] ?? 0]?.[tour[(i + 1) % n] ?? 0] ?? 0
      best = Math.min(best, s)
      return
    }
    left.forEach((c, i) => {
      walk([...tour, c], [...left.slice(0, i), ...left.slice(i + 1)])
    })
  }
  walk([0], rest)
  return best
}

describe('ant colony optimisation', () => {
  it('finds the optimal tour with each update rule', () => {
    for (const variant of ['ant-system', 'elitist', 'max-min'] as const) {
      const r = antColonyOptimization(distances, { variant, iterations: 60, seed: 1 })
      expect(r.length).toBeCloseTo(optimum, 9)
      expect([...r.tour].sort((a, b) => a - b)).toEqual([...Array(10).keys()])
      for (let i = 1; i < r.history.length; i++) expect(r.history[i]).toBeLessThanOrEqual(r.history[i - 1] ?? 0)
    }
  })

  it('handles asymmetric distances', () => {
    const d = [
      [0, 2, 9, 10, 7],
      [1, 0, 6, 4, 3],
      [15, 7, 0, 8, 3],
      [6, 3, 12, 0, 11],
      [9, 7, 5, 6, 0],
    ]
    expect(antColonyOptimization(d, { iterations: 40, seed: 2 }).length).toBe(bruteForce(d))
  })

  it('traces pheromone and rejects bad matrices', () => {
    const algorithm = antColonyAlgorithm(distances)
    const rec = recorder()
    antColonyOptimization(distances, { iterations: 3, tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
    expect(tensorSeries(rec.run(), 'pheromone').length).toBe(4)
    expect(() => antColonyOptimization([[0, 1]])).toThrow()
    expect(() =>
      antColonyOptimization([
        [0, 0],
        [0, 0],
      ]),
    ).toThrow()
    expect(() => antColonyOptimization(distances, { rho: 0 })).toThrow()
    expect(permutationNeighbours([0, 1, 2], 'insert').length).toBe(4)
  })
})

describe('binary particle swarm', () => {
  it('solves OneMax and a knapsack, reproducibly', () => {
    const onemax = (b: readonly number[]): number => b.reduce((s, x) => s + x, 0)
    const r = discreteParticleSwarm(onemax, 24, { iterations: 200, target: 24, seed: 3 })
    expect(r.fitness).toBe(24)
    expect(r.status).toBe('target')
    expect(discreteParticleSwarm(onemax, 12, { iterations: 10, seed: 5 }).history).toEqual(
      discreteParticleSwarm(onemax, 12, { iterations: 10, seed: 5 }).history,
    )
    const values = [10, 13, 7, 8, 9, 4, 11, 6]
    const weights = [5, 6, 3, 4, 5, 2, 6, 3]
    const fit = (b: readonly number[]): number => {
      const w = b.reduce((s, x, i) => s + x * (weights[i] ?? 0), 0)
      return w <= 15 ? b.reduce((s, x, i) => s + x * (values[i] ?? 0), 0) : 0
    }
    // Exhaustive optimum over the 256 subsets.
    let best = 0
    for (let m = 0; m < 256; m++) best = Math.max(best, fit(values.map((_, i) => (m >> i) & 1)))
    expect(discreteParticleSwarm(fit, 8, { iterations: 100, seed: 4 }).fitness).toBe(best)
    expect(() => discreteParticleSwarmAlgorithm(onemax, 0)).toThrow()
  })
})
