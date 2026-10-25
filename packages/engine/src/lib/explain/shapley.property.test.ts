import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import {
  conditionalValue,
  correlationHierarchy,
  type FeatureHierarchy,
  interventionalValue,
  kernelShap,
  linearShap,
  owenValues,
  partitionShap,
  sage,
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

  it('Owen values are efficient, reduce to Shapley values without structure, and are reproducible', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: -9, max: 9 }), { minLength: 16, maxLength: 16 }), (table) => {
        const v = (s: readonly boolean[]): number => table[s.reduce((b, x, j) => (x ? b | (1 << j) : b), 0)] ?? 0
        const total = (table[15] ?? 0) - (table[0] ?? 0)
        const shap = shapleyValues(v, 4)
        const flat = owenValues(v, [0, 1, 2, 3])
        flat.forEach((x, j) => {
          expect(x).toBeCloseTo(shap[j] ?? 0, 10)
        })
        const singletons = owenValues(v, [[0], [1], [2], [3]])
        singletons.forEach((x, j) => {
          expect(x).toBeCloseTo(shap[j] ?? 0, 10)
        })
        const nested: FeatureHierarchy = [[0, [1, 2]], 3]
        const owen = owenValues(v, nested)
        expect(owen.reduce((a, b) => a + b, 0)).toBeCloseTo(total, 10)
      }),
      { numRuns: 40 },
    )
    expect(() => owenValues(() => 0, [0, [0, 1]])).toThrow(/exactly once/)
  })

  it('partition SHAP of an additive model is the Shapley value whatever the hierarchy', () => {
    const f = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => 2 * (d[3 * i] ?? 0) - (d[3 * i + 2] ?? 0)),
        [n],
        'float64',
      )
    }
    const bg = [
      [0, 1, 2],
      [2, 0, 1],
      [1, 3, 0],
    ]
    const r = partitionShap(f, [[3, 3, 3]], bg)
    expect(r.values[0]?.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([4, 0, -2])
    expect(leaves(correlationHierarchy(bg)).sort()).toEqual([0, 1, 2])
  })

  it('correlated linear SHAP is efficient and matches the interventional one for uncorrelated data', () => {
    const square = [
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ]
    const ind = linearShap([2, 3], 1, [[2, 0.5]], square)
    const cor = linearShap([2, 3], 1, [[2, 0.5]], square, { perturbation: 'correlation' })
    cor.values[0]?.forEach((v, j) => {
      expect(v).toBeCloseTo(ind.values[0]?.[j] ?? 0, 10)
    })
    const line = [
      [0, 0.1],
      [1, 0.9],
      [2, 2.1],
      [3, 2.9],
    ]
    const c = linearShap([1, 0], 0, [[3, 3]], line, { perturbation: 'correlation' })
    // The second feature carries information about the first: it shares the credit, though its weight is 0.
    expect(c.values[0]?.[1] ?? 0).toBeGreaterThan(0.3)
    expect((c.values[0] ?? []).reduce((a, b) => a + b, 0)).toBeCloseTo(3 - 1.5, 10)
  })

  it('the Gaussian conditional value function holds the full and empty coalitions', () => {
    const f = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => (d[2 * i] ?? 0) + (d[2 * i + 1] ?? 0)),
        [n],
        'float64',
      )
    }
    const bg = [
      [0, 0],
      [1, 1.2],
      [2, 1.8],
      [3, 3.1],
    ]
    const v = conditionalValue(f, [3, 3], bg, { nSamples: 2000, key: key(5) })
    expect(v([true, true])).toBeCloseTo(6, 12)
    expect(v([false, false])).toBeCloseTo(3.025, 1)
    // Knowing x1 = 3 moves the expectation of x2 well above its mean.
    expect(v([true, false])).toBeGreaterThan(5)
  })

  it('SAGE by sampling approaches the exact values, which add up to the loss removed', () => {
    const f = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => 2 * (d[3 * i] ?? 0) + (d[3 * i + 1] ?? 0)),
        [n],
        'float64',
      )
    }
    const x = Array.from({ length: 12 }, (_, i) => [Math.sin(i), Math.cos(3 * i), i % 3])
    const y = x.map(([a = 0, b = 0]) => 2 * a + b)
    const exact = sage(f, { x, y }, x, { exact: true })
    expect(exact.values.reduce((a, b) => a + b, 0)).toBeCloseTo(exact.lossEmpty - exact.lossFull, 10)
    expect(Math.abs(exact.values[2] ?? 1)).toBeLessThan(1e-12)
    const sampled = sage(f, { x, y }, x, { nPermutations: 600, key: key(7) })
    sampled.values.forEach((v, j) => {
      expect(Math.abs(v - (exact.values[j] ?? 0))).toBeLessThan(5 * (sampled.standardErrors[j] ?? 0) + 1e-9)
    })
  })
})

function leaves(h: FeatureHierarchy): number[] {
  return typeof h === 'number' ? [h] : h.flatMap(leaves)
}
