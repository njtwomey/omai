import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { laplaceMethod } from '../inference/index.ts'
import {
  heldOutScores,
  polynomialFeatureMap,
  probitClassifier,
  probitData,
  probitPredictive,
  probitPredictiveGrid,
} from './probit-classification.ts'

const points = Array.from({ length: 30 }, (_, i) => [Math.cos(i), Math.sin(2 * i) * 2])

describe('polynomialFeatureMap', () => {
  it('has the intercept and every monomial up to the degree', () => {
    for (const [degree, size] of [
      [1, 3],
      [2, 6],
      [4, 15],
      [6, 28],
    ] as const)
      expect(polynomialFeatureMap(points, degree).size).toBe(size)
    expect(polynomialFeatureMap(points, 2).names).toEqual(['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2'])
  })

  it('standardises the columns on the training points and divides the design by the weight scale', () => {
    const map = polynomialFeatureMap(points, 3)
    const rows = map.transform(points)
    expect(rows.every((r) => r[0] === 1 / map.weightScale)).toBe(true)
    for (let j = 1; j < map.size; j++) {
      const col = rows.map((r) => (r[j] ?? 0) * map.weightScale)
      const mean = col.reduce((a, b) => a + b, 0) / col.length
      const variance = col.reduce((a, b) => a + (b - mean) ** 2, 0) / col.length
      expect(mean).toBeCloseTo(0, 10)
      expect(variance).toBeCloseTo(1, 10)
    }
    expect(map.weightScale).toBeCloseTo(2 * Math.sqrt(9), 12)
  })

  it('rejects a degree below one and no points', () => {
    expect(() => polynomialFeatureMap(points, 0)).toThrow(/degree/)
    expect(() => polynomialFeatureMap([], 2)).toThrow(/no points/)
  })
})

describe('probit classifier', () => {
  const x = [
    [-2, -1],
    [-1.5, 0.5],
    [-1, -0.5],
    [1, 0.4],
    [1.5, -0.3],
    [2, 1],
  ]
  const y = [0, 0, 0, 1, 1, 1]
  const map = polynomialFeatureMap(x, 1)

  it('binds the design and the labels to the model', () => {
    const bound = probitClassifier(1).bind(probitData(map.transform(x), y))
    expect(bound.sizes).toEqual({ row: 6, feature: 3 })
  })

  it('is solved by a method of the library: the slope along the separating input is positive', () => {
    const bound = probitClassifier(map.weightScale).bind(probitData(map.transform(x), y))
    const method = laplaceMethod(bound)
    const s = session(method, { seed: 1, steps: 1000, mode: 'eager' })
    const post = method.posterior?.(s.state, s.config)
    const w = toFlat(post?.mean('w') as never)
    expect(w).toHaveLength(3)
    expect(w[1]).toBeGreaterThan(0)
    const p = probitPredictive(
      [Array.from(w)],
      map.transform([
        [-3, 0],
        [3, 0],
      ]),
    )
    expect(p[0]).toBeLessThan(0.5)
    expect(p[1]).toBeGreaterThan(0.5)
  })
})

describe('probitPredictive', () => {
  it('averages the normal CDF of the linear predictor over the draws', () => {
    expect(probitPredictive([[0, 0]], [[1, 5]])).toEqual([0.5])
    const p = probitPredictive(
      [
        [1, 0],
        [-1, 0],
      ],
      [[1, 3]],
    )
    expect(p[0]).toBeCloseTo(0.5, 12)
    expect(probitPredictive([[2, 0]], [[1, 0]])[0]).toBeCloseTo(0.9772498680518208, 12)
  })

  it('rejects mismatched widths and no draws', () => {
    expect(() => probitPredictive([], [[1]])).toThrow(/no draws/)
    expect(() => probitPredictive([[1, 2]], [[1]])).toThrow(/columns/)
  })
})

describe('probitPredictiveGrid', () => {
  it('lays the predictive over the rectangle, rows by ordinate', () => {
    const map = polynomialFeatureMap(points, 1)
    // weights that make the predictive grow with the first input only: eta = x0 standardised
    const w = [0, map.weightScale, 0]
    const g = probitPredictiveGrid([w], map, { x: [-1, 1], y: [-2, 2] }, 3)
    expect(g.xs).toEqual([-1, 0, 1])
    expect(g.ys).toEqual([-2, 0, 2])
    expect(g.probabilities).toHaveLength(9)
    // along a row the probability rises with x; down a column it does not move
    expect(g.probabilities[0]).toBeLessThan(g.probabilities[2] ?? 0)
    expect(g.probabilities[0]).toBeCloseTo(g.probabilities[6] ?? 0, 12)
  })
})

describe('heldOutScores', () => {
  it('reads accuracy and the mean log probability of the labels', () => {
    const s = heldOutScores([0.9, 0.2, 0.6, 0.4], [1, 0, 0, 1])
    expect(s.accuracy).toBe(0.5)
    expect(s.logPredictive).toBeCloseTo((Math.log(0.9) + Math.log(0.8) + Math.log(0.4) + Math.log(0.4)) / 4, 12)
  })
})

describe('feature scaling', () => {
  it('"none" leaves the raw monomials (columns of very different size); the default standardises them', () => {
    const points = Array.from({ length: 30 }, (_, i) => [i / 3, 20 + i * 2])
    const spread = (m: ReturnType<typeof polynomialFeatureMap>): number => {
      const rows = m.transform(points).map((r) => r.slice(1))
      const sds = rows[0]?.map((_, j) => {
        const v = rows.map((r) => r[j] ?? 0)
        const mean = v.reduce((a, b) => a + b, 0) / v.length
        return Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length)
      })
      return Math.max(...(sds ?? [0])) / Math.min(...(sds ?? [1]).filter((x) => x > 0))
    }
    const standard = polynomialFeatureMap(points, 4)
    const raw = polynomialFeatureMap(points, 4, { scaling: 'none' })
    expect(standard.size).toBe(raw.size)
    expect(spread(standard)).toBeLessThan(1.01)
    expect(spread(raw)).toBeGreaterThan(1e3)
  })
})
