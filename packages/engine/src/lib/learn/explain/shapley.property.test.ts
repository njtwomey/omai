import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import {
  interventionalValue,
  kernelShap,
  samplingShapley,
  shapleyInteractionValues,
  shapleyValues,
  type TreeModel,
  treeConditionalValue,
  treePredict,
  treeShap,
} from './shapley.ts'

/** A random full binary tree of depth 3 over 3 features, built breadth-first. */
const trees = fc
  .tuple(
    fc.array(fc.integer({ min: 0, max: 2 }), { minLength: 7, maxLength: 7 }),
    fc.array(fc.integer({ min: -10, max: 10 }), { minLength: 7, maxLength: 7 }),
    fc.array(fc.integer({ min: -50, max: 50 }), { minLength: 8, maxLength: 8 }),
    fc.array(fc.integer({ min: 1, max: 9 }), { minLength: 8, maxLength: 8 }),
  )
  .map(([features, thresholds, leaves, counts]): TreeModel => {
    const childrenLeft: number[] = []
    const childrenRight: number[] = []
    const feature: number[] = []
    const threshold: number[] = []
    for (let i = 0; i < 15; i++) {
      const internal = i < 7
      childrenLeft.push(internal ? 2 * i + 1 : -1)
      childrenRight.push(internal ? 2 * i + 2 : -1)
      feature.push(internal ? (features[i] ?? 0) : -2)
      threshold.push(internal ? (thresholds[i] ?? 0) / 10 : -2)
    }
    const cover = new Array<number>(15).fill(0)
    const value = new Array<number>(15).fill(0)
    for (let l = 0; l < 8; l++) {
      cover[7 + l] = counts[l] ?? 1
      value[7 + l] = (leaves[l] ?? 0) / 10
    }
    for (let i = 6; i >= 0; i--) {
      cover[i] = (cover[2 * i + 1] ?? 0) + (cover[2 * i + 2] ?? 0)
      value[i] =
        ((cover[2 * i + 1] ?? 0) * (value[2 * i + 1] ?? 0) + (cover[2 * i + 2] ?? 0) * (value[2 * i + 2] ?? 0)) /
        (cover[i] ?? 1)
    }
    return { childrenLeft, childrenRight, feature, threshold, value, cover }
  })

const points = fc.array(
  fc.integer({ min: -15, max: 15 }).map((v) => v / 10),
  { minLength: 3, maxLength: 3 },
)

describe('Shapley values', () => {
  it('Tree SHAP equals exact Shapley values of the path-dependent game and is efficient', () => {
    fc.assert(
      fc.property(trees, points, (tree, x) => {
        const r = treeShap(tree, [x])
        const exact = shapleyValues(treeConditionalValue(tree, x), 3)
        exact.forEach((v, j) => {
          expect(r.values[0]?.[j] ?? 0).toBeCloseTo(v, 10)
        })
        const fx = treePredict(tree, [x]).data[0] ?? 0
        expect((r.values[0] ?? []).reduce((a, b) => a + b, 0) + r.baseValue).toBeCloseTo(fx, 10)
        const inter = shapleyInteractionValues(treeConditionalValue(tree, x), 3)
        inter.forEach((row, i) => {
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(exact[i] ?? 0, 10)
          row.forEach((v, j) => {
            expect(v).toBeCloseTo(inter[j]?.[i] ?? 0, 12)
          })
        })
      }),
      { numRuns: 60 },
    )
  })

  it('satisfy symmetry and the dummy axiom', () => {
    const v = (s: readonly boolean[]) => (s[0] ? 1 : 0) + (s[1] ? 1 : 0) + (s[0] && s[1] ? 2 : 0)
    const phi = shapleyValues(v, 3)
    expect(phi[0]).toBeCloseTo(phi[1] ?? 0, 12)
    expect(phi[2]).toBe(0)
    expect(() => shapleyValues(v, 25)).toThrow()
  })

  it('sampling and sampled Kernel SHAP approach the exact values', () => {
    const predict = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => (d[3 * i] ?? 0) * (d[3 * i + 1] ?? 0) + (d[3 * i + 2] ?? 0)),
        [n],
        'float64',
      )
    }
    const bg = [
      [0, 0, 0],
      [1, -1, 2],
      [-1, 2, 0],
    ]
    const x = [1.5, 1, -1]
    const exact = shapleyValues(interventionalValue(predict, x, bg), 3)
    const s = samplingShapley(predict, x, bg, { nPermutations: 3000, key: key(2) })
    exact.forEach((v, j) => {
      expect(Math.abs((s.values[j] ?? 0) - v)).toBeLessThan(6 * (s.standardErrors[j] ?? 0) + 1e-9)
    })
    const k = kernelShap(predict, [x], bg, { nSamples: 4, key: key(3) })
    expect((k.values[0] ?? []).reduce((a, b) => a + b, 0)).toBeCloseTo(
      exact.reduce((a, b) => a + b, 0),
      10,
    )
  })
})
