import { describe, expect, it } from 'vitest'
import { matrixView } from '../learn/validation.ts'
import { key } from '../random/index.ts'
import {
  ALL_PRESENT,
  type Criterion,
  impurityOf,
  isClassificationCriterion,
  nodeValue,
  partitionRows,
  type SplitProblem,
  searchSplits,
} from './criteria.ts'

function problem(x: number[][], y: number[], criterion: Criterion, extra: Partial<SplitProblem> = {}): SplitProblem {
  const k = isClassificationCriterion(criterion) ? Math.max(...y) + 1 : 0
  return {
    x: matrixView(x, 'test', { allowNaN: true }),
    y: Float64Array.from(y),
    w: new Float64Array(y.length).fill(1),
    k,
    criterion,
    minSamplesLeaf: 1,
    minWeightLeaf: 0,
    totalWeight: y.length,
    ...extra,
  }
}

const rows = (n: number): Int32Array => Int32Array.from({ length: n }, (_, i) => i)

describe('impurities and node values', () => {
  it('Gini and entropy of class proportions, and the variance, absolute deviation and Poisson deviance', () => {
    const c = problem([[0], [1], [2], [3]], [0, 0, 0, 1], 'gini')
    expect(impurityOf(c, rows(4))).toBeCloseTo(1 - 0.75 ** 2 - 0.25 ** 2, 14)
    expect(impurityOf({ ...c, criterion: 'entropy' }, rows(4))).toBeCloseTo(0.8112781244591328, 14)
    expect(impurityOf({ ...c, criterion: 'log_loss' }, rows(4))).toBeCloseTo(0.8112781244591328, 14)
    const r = problem([[0], [1], [2], [3]], [1, 2, 3, 10], 'squared_error')
    expect(impurityOf(r, rows(4))).toBeCloseTo(12.5, 12)
    expect(nodeValue(r, rows(4))).toEqual([4])
    const a = { ...r, criterion: 'absolute_error' as const }
    // The median of 1, 2, 3, 10 is 2.5; the mean absolute deviation from it is 2.5.
    expect(nodeValue(a, rows(4))).toEqual([2.5])
    expect(impurityOf(a, rows(4))).toBeCloseTo(2.5, 12)
    const p = { ...r, criterion: 'poisson' as const }
    const mean = 4
    expect(impurityOf(p, rows(4))).toBeCloseTo([1, 2, 3, 10].reduce((s, v) => s + v * Math.log(v / mean), 0) / 4, 12)
  })

  it('weights count: a heavy row dominates the class counts', () => {
    const c = problem([[0], [1]], [0, 1], 'gini', { w: Float64Array.of(1, 3) })
    expect(nodeValue(c, rows(2))).toEqual([1, 3])
    expect(impurityOf(c, rows(2))).toBeCloseTo(1 - 0.25 ** 2 - 0.75 ** 2, 14)
  })
})

describe('searching splits', () => {
  it('finds the midpoint threshold and records every candidate, or none', () => {
    const p = problem([[0], [1], [2], [3]], [0, 0, 1, 1], 'gini')
    const s = searchSplits(p, rows(4), [0], { splitter: 'best' })
    expect(s.best?.threshold).toBe(1.5)
    expect(s.best?.decrease).toBeCloseTo(0.5, 14)
    expect(s.candidates).toHaveLength(3)
    expect(searchSplits(p, rows(4), [0], { splitter: 'best', record: 1 }).candidates).toHaveLength(1)
    expect(searchSplits(p, rows(4), [0], { splitter: 'best', record: 0 }).candidates).toHaveLength(0)
  })

  it('respects the minimum leaf size', () => {
    const p = problem([[0], [1], [2], [3]], [0, 1, 1, 1], 'gini', { minSamplesLeaf: 2 })
    expect(searchSplits(p, rows(4), [0], { splitter: 'best' }).best?.threshold).toBe(1.5)
  })

  it('a random split draws one threshold per feature, keyed', () => {
    const p = problem([[0], [1], [2], [3]], [0, 0, 1, 1], 'gini')
    const a = searchSplits(p, rows(4), [0], { splitter: 'random', key: key(1) })
    const b = searchSplits(p, rows(4), [0], { splitter: 'random', key: key(1) })
    expect(a.best?.threshold).toBe(b.best?.threshold)
    expect(a.candidates).toHaveLength(1)
  })

  it('splits a categorical feature by the best subset of codes', () => {
    // Codes 0 and 3 are class 1; 1 and 2 class 0: the best subset separates {0, 3} from {1, 2}.
    const x = [[0], [1], [2], [3], [0], [1], [2], [3]]
    const y = [1, 0, 0, 1, 1, 0, 0, 1]
    const p = problem(x, y, 'gini', { categorical: [true] })
    const best = searchSplits(p, rows(8), [0], { splitter: 'best' }).best
    expect(best?.decrease).toBeCloseTo(0.5, 12)
    expect(best?.categories).toBeDefined()
    const side = best?.categories ?? []
    expect(side.length).toBe(2)
    expect(side.includes(0) === side.includes(3)).toBe(true)
  })

  it('sends missing values to the better side and can split missing from present', () => {
    // Small values are class 0, large class 1; the missing rows are class 1.
    const x = [[0], [1], [2], [8], [9], [Number.NaN], [Number.NaN]]
    const y = [0, 0, 0, 1, 1, 1, 1]
    const p = problem(x, y, 'gini')
    const best = searchSplits(p, rows(7), [0], { splitter: 'best' }).best
    expect(best?.threshold).toBe(5)
    expect(best?.missingLeft).toBe(false)
    expect(best?.rightRows.length).toBe(4)
    const all = problem([[0], [1], [Number.NaN], [Number.NaN]], [0, 0, 1, 1], 'gini')
    const s = searchSplits(all, rows(4), [0], { splitter: 'best' }).best
    expect(s?.threshold).toBe(ALL_PRESENT)
    expect(s?.decrease).toBeCloseTo(0.5, 12)
  })

  it('partitions rows consistently with the split it describes', () => {
    const p = problem([[0], [5], [Number.NaN]], [0, 1, 1], 'gini')
    const { leftRows, rightRows } = partitionRows(p, rows(3), { feature: 0, threshold: 2 }, true)
    expect(Array.from(leftRows)).toEqual([0, 2])
    expect(Array.from(rightRows)).toEqual([1])
  })
})
