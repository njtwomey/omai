import { describe, expect, it } from 'vitest'
import { type Key, normal, uniform } from '../../random/index.ts'
import { gaussianMutation, mapElites, mapElitesAlgorithm, noveltySearch } from '../index.ts'

const square = {
  random: (k: Key) => Array.from(uniform(k, [2]).data),
  mutate: (k: Key, x: number[]) => gaussianMutation(k, x, { sigma: 0.1, lower: [0, 0], upper: [1, 1] }),
  crossover: (_k: Key, a: number[], b: number[]) => a.map((v, i) => (v + (b[i] ?? 0)) / 2),
  fitness: (x: number[]) => (x[0] ?? 0) + (x[1] ?? 0),
  descriptor: (x: number[]) => x,
}

describe('MAP-Elites', () => {
  it('fills the grid and keeps the fittest per cell', () => {
    const r = mapElites(square, {
      bins: [4, 4],
      bounds: [
        [0, 1],
        [0, 1],
      ],
      iterations: 60,
      seed: 1,
    })
    expect(r.coverage).toBe(1)
    for (const [cell, e] of r.archive) {
      const i = Math.min(3, Math.floor((e.descriptor[0] ?? 0) * 4))
      const j = Math.min(3, Math.floor((e.descriptor[1] ?? 0) * 4))
      expect(cell).toBe(i * 4 + j)
    }
    expect(r.best?.fitness).toBeGreaterThan(1.8)
    expect(r.qdScore).toBeGreaterThan(0)
    expect(() => mapElites(square, { bins: [4], bounds: [] })).toThrow()
    expect(() => mapElitesAlgorithm(square, { bins: [0], bounds: [[0, 1]] })).toThrow()
  })

  it('never loses an elite as iterations increase', () => {
    const short = mapElites(square, {
      bins: [3, 3],
      bounds: [
        [0, 1],
        [0, 1],
      ],
      iterations: 5,
      seed: 2,
    })
    const long = mapElites(square, {
      bins: [3, 3],
      bounds: [
        [0, 1],
        [0, 1],
      ],
      iterations: 30,
      seed: 2,
    })
    for (const [cell, e] of short.archive)
      expect(long.archive.get(cell)?.fitness ?? -1).toBeGreaterThanOrEqual(e.fitness)
  })
})

describe('novelty search', () => {
  it('explores a deceptive maze-free walk and tracks fitness', () => {
    // Fitness pulls towards 0; the goal lies at |x| > 15, which novelty finds and fitness alone would not seek.
    const walker = {
      random: () => 0,
      mutate: (k: Key, x: number) => x + (normal(k, []).data[0] ?? 0),
      behaviour: (x: number) => [x],
      fitness: (x: number) => -Math.abs(x),
    }
    const r = noveltySearch(walker, {
      generations: 150,
      k: 10,
      threshold: 0.5,
      isGoal: (x) => Math.abs(x) > 15,
      seed: 3,
    })
    expect(r.found).not.toBeNull()
    expect(r.archive.length).toBeGreaterThan(0)
    expect(r.bestFitness).toBeLessThanOrEqual(0)
    expect(() => noveltySearch(walker, { populationSize: 1 })).toThrow()
  })
})
