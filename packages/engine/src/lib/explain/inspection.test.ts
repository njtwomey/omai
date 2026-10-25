import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import {
  accumulatedLocalEffects2D,
  centredIce,
  conditionalPermutationImportance,
  dropColumnImportance,
  functionalAnova,
  hStatistic,
  iceDerivative,
  partialDependence,
  permutationImportance,
} from './inspection.ts'

const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), (i % 5) / 5])
const y = x.map(([a = 0, b = 0]) => 3 * a + b)
const model = leastSquares().fit({ x, y })

describe('inspection', () => {
  it('ranks the features a linear model uses and permutes groups together', () => {
    const r = permutationImportance(model, { x, y }, { nRepeats: 4, key: key(1) })
    expect(r.mean[0] ?? 0).toBeGreaterThan(r.mean[1] ?? 0)
    expect(Math.abs(r.mean[2] ?? 1)).toBeLessThan(1e-9)
    const g = permutationImportance(model, { x, y }, { nRepeats: 2, features: [[0, 1]], key: key(1) })
    expect(g.importances.length).toBe(1)
    expect(permutationImportance(model, { x, y }, { nRepeats: 3, key: key(1) })).toEqual(
      permutationImportance(model, { x, y }, { nRepeats: 3, key: key(1) }),
    )
  })

  it('partial dependence of a linear model is linear with its coefficient as slope', () => {
    const r = partialDependence(model, x, [0], { gridResolution: 5 })
    const g = r.values[0] ?? []
    const avg = r.average[0] ?? []
    const slope = ((avg[4] ?? 0) - (avg[0] ?? 0)) / ((g[4] ?? 0) - (g[0] ?? 0))
    expect(slope).toBeCloseTo(3, 9)
    expect(r.individual[0]?.length).toBe(30)
    expect(() => partialDependence(model, x, [7])).toThrow(/out of range/)
  })

  it('finds no interactions in an additive model', () => {
    expect(hStatistic(model, x, 0, 1)).toBeLessThan(1e-12)
    const anova = functionalAnova(model, x.slice(0, 12), { maxOrder: 1 })
    expect(anova.components.length).toBe(3)
    expect(anova.residualShare).toBeGreaterThan(-1e-9)
  })

  it('centres ICE curves at a grid point and differentiates them exactly for quadratics', () => {
    const c = centredIce([[3, 4, 9]], 1)
    expect(c).toEqual([[-1, 0, 5]])
    const grid = [0, 0.5, 2, 3]
    const curve = grid.map((g) => g * g)
    const d = iceDerivative(grid, [curve])[0] ?? []
    // Central differences are exact for a quadratic inside the grid.
    expect(d[1]).toBeCloseTo(1, 12)
    expect(d[2]).toBeCloseTo(4, 12)
    expect(() => iceDerivative([0], [[1]])).toThrow(/two grid points/)
  })

  it('second-order ALE finds the interaction of a product and nothing in a sum', () => {
    const product = {
      estimator: 'p',
      kind: 'regressor' as const,
      params: {},
      predict: (v: unknown) => {
        const t = v as { shape: number[]; data: Float64Array }
        return tensor(
          Array.from({ length: t.shape[0] ?? 0 }, (_, i) => (t.data[3 * i] ?? 0) * (t.data[3 * i + 1] ?? 0)),
        )
      },
    }
    const r = accumulatedLocalEffects2D(product, x, [0, 1], { bins: 4 })
    const flat = r.effects.flat()
    expect(Math.max(...flat) - Math.min(...flat)).toBeGreaterThan(0.1)
    expect(r.counts.flat().reduce((a, b) => a + b, 0)).toBe(30)
    const additive = accumulatedLocalEffects2D(model, x, [0, 1], { bins: 4 })
    for (const v of additive.effects.flat()) expect(Math.abs(v)).toBeLessThan(1e-9)
    expect(() => accumulatedLocalEffects2D(model, x, [0, 0])).toThrow(/differ/)
  })

  it('drop-column importance refits without each feature', () => {
    const r = dropColumnImportance(leastSquares(), { x, y })
    expect(r.importances[0] ?? 0).toBeGreaterThan(r.importances[1] ?? 0)
    expect(Math.abs(r.importances[2] ?? 1)).toBeLessThan(1e-9)
    expect(r.baseline).toBeCloseTo(1, 12)
  })

  it('conditional permutation shares no importance with a copy, unlike the marginal one', () => {
    const xd = x.map(([a = 0, b = 0]) => [a, a + 1e-3 * b, b])
    const yd = xd.map(([a = 0, , c = 0]) => 2 * a + c)
    const fitted = leastSquares().fit({ x: xd, y: yd })
    const marginal = permutationImportance(fitted, { x: xd, y: yd }, { nRepeats: 3, key: key(2) })
    const conditional = conditionalPermutationImportance(
      fitted,
      { x: xd, y: yd },
      { nRepeats: 3, bins: 8, key: key(2) },
    )
    expect(conditional.conditionOn[0]).toContain(1)
    expect(conditional.mean[0] ?? 1).toBeLessThan(0.5 * (marginal.mean[0] ?? 0) + 1e-9)
    const none = conditionalPermutationImportance(fitted, { x: xd, y: yd }, { conditionOn: [[], [], []], key: key(2) })
    expect(none.mean[2] ?? 0).toBeGreaterThan(0)
  })
})
