import { describe, expect, it } from 'vitest'
import type { TestResult } from '../../base/index.ts'
import { etaSquared, histogram, kendallTau, rankdata, spearman } from '../index.ts'
import * as T from './index.ts'

const c = [3, 3, 3, 3]
const withNaN = [1, 2, Number.NaN, 4, 7]
const other = [3, 5, 6, 7, 2]

// Degenerate input to the tests: a constant or identical sample makes the statistic undefined (NaN, and then the p-value
// is NaN too, never a number); a NaN in data that a routine sorts or ranks is an error, not a tie; a count below zero
// is an error.
describe('hypothesis tests on degenerate input', () => {
  const undefinedTests: readonly (readonly [string, () => TestResult])[] = [
    ['fligner of constant groups', () => T.fligner([c, c])],
    ['fligner of two different constants', () => T.fligner([c, [5, 5, 5, 5]])],
    ['fOneway of identical groups', () => T.fOneway([c, c])],
    ['kruskal of identical groups', () => T.kruskal([c, c])],
    ['levene of constants', () => T.levene([c, c])],
    ['jarqueBera of a constant', () => T.jarqueBera(c)],
    ['lilliefors of a constant', () => T.lilliefors(c)],
  ]
  it.each(undefinedTests)('%s: statistic and p are NaN', (_name, call) => {
    const r = call()
    expect(r.statistic).toBeNaN()
    expect(r.pValue).toBeNaN()
  })

  it('fOneway of two different constants is F = Infinity, p = 0, eta squared exactly 1', () => {
    const r = T.fOneway([c, [5, 5, 5, 5]])
    expect(r.statistic).toBe(Number.POSITIVE_INFINITY)
    expect(r.pValue).toBe(0)
    expect(r.effectSize?.interval).toEqual([1, 1])
    expect(etaSquared([c, [5, 5, 5, 5]]).value).toBe(1)
  })

  const rejected: readonly (readonly [string, () => unknown])[] = [
    ['mannWhitneyU', () => T.mannWhitneyU(withNaN, other)],
    ['wilcoxon', () => T.wilcoxon(withNaN, other)],
    ['kruskal', () => T.kruskal([withNaN, other])],
    ['fligner', () => T.fligner([withNaN, other])],
    ['ks2samp', () => T.ks2samp(withNaN, other)],
    ['signTest', () => T.signTest(withNaN)],
    ['runsTest', () => T.runsTest(withNaN)],
    ['spearman', () => spearman(withNaN, other)],
    ['kendallTau', () => kendallTau(withNaN, other)],
    ['rankdata', () => rankdata(withNaN)],
    ['histogram', () => histogram(withNaN, { bins: 2 })],
    ['chisquare with a negative count', () => T.chisquare([5, -5])],
    ['fOneway of singletons', () => T.fOneway([[1], [2], [3]])],
  ]
  it.each(rejected)('%s throws value', (_name, call) => {
    expect(call).toThrow(expect.objectContaining({ code: 'value' }))
  })
})
