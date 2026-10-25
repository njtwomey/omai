import { describe, expect, it } from 'vitest'
import { logisticRegression } from '../../methods/linear/index.ts'
import { generate } from '../data/index.ts'
import { biasedCredit } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { exponentiatedGradientReduction, gridReduction } from './in-processing.ts'

const { data } = generate(biasedCredit, { key: key(3), knobs: { n: 500, labelBias: 0.4 } })
const x = toFlat(data.x)
const rows = {
  x: data.x,
  labels: Array.from(toFlat(data.y)),
  groups: Array.from({ length: 500 }, (_, i) => x[4 * i + 3] ?? 0),
}

describe('reductions', () => {
  it('exponentiated gradient: weights form a distribution, the violation falls below the baseline', () => {
    const r = exponentiatedGradientReduction(logisticRegression(), rows, { iterations: 8, epsilon: 0.02, key: key(1) })
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
    expect(r.decisions.every((p) => p >= 0 && p <= 1)).toBe(true)
    expect(r.steps[7]?.averageViolation).toBeLessThan(r.baseline.violation)
    expect(() => exponentiatedGradientReduction(logisticRegression(), rows, { epsilon: -1 })).toThrow(/epsilon/)
  })

  it('a grid of multipliers spans violations, zero multiplier being the unconstrained fit', () => {
    const points = gridReduction(logisticRegression(), rows, { gridSize: 5, key: key(2) })
    expect(points.map((p) => p.lambda)).toEqual([-0.5, -0.25, 0, 0.25, 0.5])
    expect(Math.max(...points.map((p) => p.violation))).toBeGreaterThan(Math.min(...points.map((p) => p.violation)))
  })
})
