import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { functionalAnova, hStatistic, partialDependence, permutationImportance } from './inspection.ts'

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
})
