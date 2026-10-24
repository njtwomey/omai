import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { cmaesOnPlane, evolutionStrategyOnPlane } from './evolution-strategies.ts'
import { evolveBitStrings, evolveOnPlane, knapsack, oneMax } from './genetic-algorithms.ts'

describe('evolutionary studies', () => {
  it('solves OneMax and packs a knapsack, generation by generation', () => {
    const r = evolveBitStrings(oneMax(20), { key: key(1), generations: 80 })
    expect(r.fitness).toBe(20)
    const [board] = recordsOf(r.exposition, 'boardSteps')
    expect(board?.layout).toEqual({ type: 'grid', rows: 30, columns: 20 })
    expect(r.history.at(-1)).toBeGreaterThanOrEqual(r.history[0] ?? 0)
    const k = knapsack(key(2), 16)
    const packed = evolveBitStrings(k, { key: key(3) })
    expect(packed.fitness).toBeGreaterThan(0)
    expect(recordsOf(packed.exposition, 'populationSteps')[0]?.generations.length).toBeGreaterThan(1)
  })

  it('brings populations near the minimum on the plane', () => {
    const ga = evolveOnPlane('himmelblau', { key: key(4) })
    const es = evolutionStrategyOnPlane('rastrigin', { key: key(5) })
    const cma = cmaesOnPlane('rosenbrock', { key: key(6), generations: 80 })
    expect(ga.fitness).toBeLessThan(0.5)
    expect(es.value).toBeLessThan(2.5)
    expect(cma.value).toBeLessThan(1e-3)
    const [steps] = recordsOf(cma.exposition, 'populationSteps')
    expect(steps?.generations[1]?.points?.length).toBe(6)
    expect(steps?.generations[1]?.covariance).toHaveLength(3)
    expect(steps?.surface?.values.length).toBe(61)
  })
})

describe('symbolicRegression', () => {
  it('fits the cubic and names its best program', async () => {
    const { symbolicRegression } = await import('./genetic-programming.ts')
    const r = symbolicRegression({ target: 'cubic', key: key(7) })
    expect(r.mse).toBeLessThan(0.05)
    expect(r.expression).toContain('x')
    const [steps] = recordsOf(r.exposition, 'populationSteps')
    expect(steps?.data?.grid.length).toBe(101)
    expect(steps?.generations.at(-1)?.best.curve?.length).toBe(101)
  })
})
