import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { solve } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { add, fromStorage, mul, square, sub, sum, toFlat, type Value } from '../tensor/index.ts'
import { shapleyValues } from './shapley.ts'
import { dataShapley, influenceFunctions, knnShapley, knnUtility, tracIn } from './valuation.ts'

describe('data valuation', () => {
  it('KNN-Shapley equals the exact Shapley values of the k-NN utility', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: -20, max: 20 }), fc.integer({ min: 0, max: 1 })), {
          minLength: 2,
          maxLength: 7,
        }),
        fc.array(fc.tuple(fc.integer({ min: -20, max: 20 }), fc.integer({ min: 0, max: 1 })), {
          minLength: 1,
          maxLength: 3,
        }),
        fc.integer({ min: 1, max: 3 }),
        (train, test, k) => {
          // Distinct distances avoid ties, which the recursion breaks by index.
          const tx = train.map(([v], i) => [v + i * 0.001])
          const ty = train.map(([, c]) => c)
          const sx = test.map(([v]) => [v + 0.0003])
          const sy = test.map(([, c]) => c)
          const fast = knnShapley(tx, ty, sx, sy, k)
          const exact = shapleyValues(knnUtility(tx, ty, sx, sy, k), tx.length)
          fast.forEach((v, i) => {
            expect(v).toBeCloseTo(exact[i] ?? 0, 10)
          })
        },
      ),
      { numRuns: 60 },
    )
  })

  it('influence functions predict the leave-one-out change of ridge parameters', () => {
    const xs = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(3 * i)])
    const ys = xs.map(([a = 0, b = 0], i) => 2 * a - b + 0.3 * Math.sin(7 * i))
    const lambda = 0.1
    const loss = (p: Value, x: Value, y: number): Value =>
      add(square(sub(sum(mul(p, x)), y)), mul(sum(square(p)), lambda / 40))
    const fitWithout = (skip: number): number[] => {
      const a = new Float64Array(4)
      const b = new Float64Array(2)
      xs.forEach((r, i) => {
        if (i === skip) return
        for (let u = 0; u < 2; u++) {
          b[u] = (b[u] ?? 0) + (r[u] ?? 0) * (ys[i] ?? 0)
          for (let v = 0; v < 2; v++) a[u * 2 + v] = (a[u * 2 + v] ?? 0) + (r[u] ?? 0) * (r[v] ?? 0)
        }
      })
      const m = skip < 0 ? 40 : 39
      for (let u = 0; u < 2; u++) a[u * 2 + u] = (a[u * 2 + u] ?? 0) + (lambda * m) / 40
      return toFlat(solve(fromStorage(a, [2, 2], 'float64'), fromStorage(b, [2], 'float64')))
    }
    const theta = fitWithout(-1)
    const r = influenceFunctions(loss, theta, { x: xs, y: ys }, { x: [[1, 0]], y: [2] })
    for (const i of [0, 5, 17]) {
      const exact = fitWithout(i).map((v, a) => v - (theta[a] ?? 0))
      const approx = r.parameterChange[i] ?? []
      exact.forEach((v, a) => {
        expect(Math.abs(v - (approx[a] ?? 0))).toBeLessThan(0.15 * Math.abs(v) + 1e-4)
      })
    }
  })

  it('TracIn sums learning-rate-weighted gradient products over checkpoints', () => {
    const loss = (p: Value, x: Value, y: number): Value => square(sub(sum(mul(p, x)), y))
    const one = tracIn(loss, [{ params: [0.5], learningRate: 0.1 }], { x: [[2]], y: [0] }, { x: [[1]], y: [1] })
    // gradients: train 2 (0.5 * 2 - 0) 2 = 4; test 2 (0.5 - 1) 1 = -1; 0.1 * 4 * -1.
    expect(one[0]?.[0]).toBeCloseTo(-0.4, 12)
  })

  it('truncated Monte Carlo data Shapley approaches the exact values', () => {
    const train = { x: [[0], [1], [2], [3]], y: [0, 1, 2, 10] }
    const test = { x: [[1.5]], y: [1.5] }
    const r = dataShapley(leastSquares(), train, test, {
      scoring: 'neg_mean_absolute_error',
      nPermutations: 300,
      truncationTolerance: 0,
      emptyUtility: -5,
      key: key(3),
    })
    const exact = shapleyValues((s) => {
      const idx = s.flatMap((on, i) => (on ? [i] : []))
      if (idx.length === 0) return -5
      const f = leastSquares().fit({ x: idx.map((i) => train.x[i] ?? [0]), y: idx.map((i) => train.y[i] ?? 0) })
      return -Math.abs((toFlat(f.predict(test.x))[0] ?? 0) - 1.5)
    }, 4)
    exact.forEach((v, i) => {
      expect(Math.abs((r.values[i] ?? 0) - v)).toBeLessThan(5 * (r.standardErrors[i] ?? 0) + 1e-9)
    })
  })
})
