import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { linprog } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, matmul, sigmoid, type Tensor, type Value } from '../tensor/index.ts'
import {
  diverseCounterfactuals,
  faceCounterfactual,
  linearRecourse,
  medianAbsoluteDeviations,
  wachterCounterfactual,
} from './counterfactuals.ts'

const w = fromStorage(Float64Array.from([1.5, -1, 0.2]), [3, 1], 'float64')
const proba = (v: Value): Value => sigmoid(matmul(v, w))

describe('counterfactuals', () => {
  it('Wachter moves the most effective mutable feature and reaches the target', () => {
    const cf = wachterCounterfactual(proba, [-1, 1, 0], { target: 0.7, tolerance: 0.01, scales: [1, 1, 1] })
    expect(cf.valid).toBe(true)
    // The L1 optimum changes only the feature with the largest |w| / scale.
    expect(Math.abs((cf.x[1] ?? 0) - 1)).toBeLessThan(1e-6)
    expect(Math.abs((cf.x[2] ?? 0) - 0)).toBeLessThan(1e-6)
    const fixed = wachterCounterfactual(proba, [-1, 1, 0], { target: 0.7, tolerance: 0.01, immutable: [0] })
    expect(fixed.x[0]).toBe(-1)
    expect(fixed.valid).toBe(true)
    expect(
      medianAbsoluteDeviations([
        [1, 5],
        [2, 5],
        [9, 5],
      ]),
    ).toEqual([1, 1])
  })

  it('DiCE returns valid and spread-out counterfactuals, reproducibly', () => {
    const r = diverseCounterfactuals(proba, [-1, 1, 0], { count: 3, key: key(2), maxIterations: 400 })
    expect(r.valid.every(Boolean)).toBe(true)
    expect(r.diversity).toBeGreaterThan(0.05)
    expect(diverseCounterfactuals(proba, [-1, 1, 0], { count: 3, key: key(2), maxIterations: 400 }).points).toEqual(
      r.points,
    )
  })

  it('FACE paths stay on the data and end at an accepted, dense point', () => {
    const data = Array.from({ length: 30 }, (_, i) => [Math.cos(i) + i / 10, Math.sin(i)])
    const predict = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => ((d[2 * i] ?? 0) > 2 ? 1 : 0)),
        [n],
        'float64',
      )
    }
    const r = faceCounterfactual(predict, [0, 0], data, { accept: (p) => p === 1, kNeighbors: 4 })
    expect(r).toBeDefined()
    expect((r?.counterfactual[0] ?? 0) > 2).toBe(true)
    expect(r?.path.every((i) => i >= 0 && i < 30)).toBe(true)
    expect(faceCounterfactual(predict, [0, 0], data, { accept: (p) => p === 7 })).toBeUndefined()
  })

  it('linear recourse matches the linear program it solves', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -5, max: 5 }), { minLength: 3, maxLength: 3 }),
        fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 3, maxLength: 3 }),
        fc.array(fc.integer({ min: 1, max: 3 }), { minLength: 3, maxLength: 3 }),
        (ws, cs, hs) => {
          const r = linearRecourse(ws, -6, [0, 0, 0], { costs: cs, bounds: hs.map((h) => [-h, h] as const) })
          // Split each change into positive and negative parts: minimise sum c (p + q) s.t. w . (p - q) >= 6.
          const lp = linprog({
            c: [...cs, ...cs],
            aUb: [[...ws.map((v) => -v), ...ws]],
            bUb: [-6],
            bounds: [...hs, ...hs].map((h) => [0, h] as const),
          })
          expect(r.feasible).toBe(lp.status === 'optimal')
          if (lp.status === 'optimal') expect(r.cost).toBeCloseTo(lp.fun, 8)
        },
      ),
      { numRuns: 60 },
    )
  })
})
