import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from './decision-tree.ts'
import { costComplexityPath, pruneTree } from './prune.ts'
import { applyTree, isLeaf, leafCount, type Tree } from './tree.ts'

/** Distinct continuous rows (no tied splits) with labels from a seed. */
const dataset = fc
  .record({
    n: fc.integer({ min: 8, max: 40 }),
    d: fc.integer({ min: 1, max: 3 }),
    k: fc.integer({ min: 2, max: 3 }),
    seed: fc.integer({ min: 1, max: 10_000 }),
  })
  .map(({ n, d, k, seed }) => {
    let s = seed
    const next = (): number => {
      s = (s * 48271) % 2147483647
      return s / 2147483647
    }
    const x = Array.from({ length: n }, () => Array.from({ length: d }, () => next()))
    const y = x.map((row) => Math.floor(((row[0] ?? 0) + 0.3 * next()) * k) % k)
    return { x, y, t: x.map((row) => row.reduce((a, b) => a + b, 0) + 0.1 * next()) }
  })

const consistent = (t: Tree): void => {
  for (let i = 0; i < t.feature.length; i++) {
    if (isLeaf(t, i)) continue
    const l = t.left[i] ?? 0
    const r = t.right[i] ?? 0
    expect(t.nSamples[i]).toBe((t.nSamples[l] ?? 0) + (t.nSamples[r] ?? 0))
    expect(t.weightedNSamples[i] ?? 0).toBeCloseTo((t.weightedNSamples[l] ?? 0) + (t.weightedNSamples[r] ?? 0), 9)
    expect(t.value[i]?.reduce((a, b) => a + b, 0) ?? 0).toBeCloseTo(
      (t.value[l]?.reduce((a, b) => a + b, 0) ?? 0) + (t.value[r]?.reduce((a, b) => a + b, 0) ?? 0),
      9,
    )
  }
}

describe('tree growth properties', () => {
  it('children partition their parent, and an unlimited tree fits distinct training rows', () => {
    fc.assert(
      fc.property(dataset, ({ x, y }) => {
        const f = decisionTreeClassifier().fit({ x, y })
        consistent(f.params)
        expect(f.params.nSamples[0]).toBe(x.length)
        expect(f.score?.({ x, y })).toBe(1)
        // Leaves hold the training rows: the rows reaching each leaf are its samples.
        const counts = new Map<number, number>()
        for (const leaf of applyTree(f.params, x)) counts.set(leaf, (counts.get(leaf) ?? 0) + 1)
        for (const [leaf, c] of counts) expect(f.params.nSamples[leaf]).toBe(c)
      }),
      { numRuns: 25 },
    )
  })

  it('depth-first and breadth-first growth give the same partition', () => {
    fc.assert(
      fc.property(dataset, ({ x, t }) => {
        const a = decisionTreeRegressor({ maxDepth: 4 }).fit({ x, y: t })
        const b = decisionTreeRegressor({ maxDepth: 4, growth: 'breadth' }).fit({ x, y: t })
        expect(leafCount(a.params)).toBe(leafCount(b.params))
        const pa = toFlat(a.predict(x))
        const pb = toFlat(b.predict(x))
        pa.forEach((v, i) => {
          expect(v).toBeCloseTo(pb[i] ?? 0, 12)
        })
      }),
      { numRuns: 20 },
    )
  })

  it('the predictions do not depend on the order of the rows', () => {
    fc.assert(
      fc.property(dataset, ({ x, y }) => {
        const order = x.map((_, i) => (i * 7 + 3) % x.length)
        if (new Set(order).size !== x.length) return
        const a = decisionTreeClassifier({ maxDepth: 3 }).fit({ x, y })
        const b = decisionTreeClassifier({ maxDepth: 3 }).fit({
          x: order.map((i) => x[i] ?? []),
          y: order.map((i) => y[i] ?? 0),
        })
        expect(toFlat(a.predict(x))).toEqual(toFlat(b.predict(x)))
      }),
      { numRuns: 20 },
    )
  })

  it('the pruning path has rising alphas, falling leaf counts and a tree for every alpha', () => {
    fc.assert(
      fc.property(dataset, ({ x, y }) => {
        const tree = decisionTreeClassifier().fit({ x, y }).params
        const path = costComplexityPath(tree)
        for (let i = 1; i < path.alphas.length; i++) {
          expect(path.alphas[i] ?? 0).toBeGreaterThanOrEqual(path.alphas[i - 1] ?? 0)
          expect(path.leaves[i] ?? 0).toBeLessThan(path.leaves[i - 1] ?? 0)
        }
        for (const [i, a] of path.alphas.entries()) {
          if (i > 0 && path.alphas[i] === path.alphas[i + 1]) continue
          expect(leafCount(pruneTree(tree, a))).toBe(path.leaves[i])
        }
      }),
      { numRuns: 20 },
    )
  })

  it('a leaf limit is respected exactly', () => {
    fc.assert(
      fc.property(dataset, fc.integer({ min: 2, max: 8 }), ({ x, y }, limit) => {
        const f = decisionTreeClassifier({ maxLeafNodes: limit }).fit({ x, y })
        expect(leafCount(f.params)).toBeLessThanOrEqual(limit)
        consistent(f.params)
      }),
      { numRuns: 20 },
    )
  })
})
