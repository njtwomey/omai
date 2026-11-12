import { describe, expect, it } from 'vitest'
import { fit } from './compute.ts'
import { CATEGORIES, counts, drawRatings, ITEMS, RATERS, shuffled } from './data.ts'
import { readData, readResult } from './reading.ts'

describe('the how-people-rate-things computation', () => {
  it('has ratings from 1 to 5 whose truth is a rise in the cut points', () => {
    const r = drawRatings()
    const n = r.score.length
    expect(r.item).toHaveLength(n)
    expect(r.rater).toHaveLength(n)
    expect(r.score.every((s) => Number.isInteger(s) && s >= 0 && s < CATEGORIES)).toBe(true)
    expect(r.quality).toHaveLength(ITEMS)
    expect(r.generosity).toHaveLength(RATERS)
    expect(r.cuts).toHaveLength(CATEGORIES - 1)
    expect(r.cuts.every((c, k) => k === 0 || c > (r.cuts[k - 1] as number))).toBe(true)
    // no rater scores an item twice
    expect(new Set(r.item.map((item, i) => `${item}|${r.rater[i]}`)).size).toBe(n)
    expect(counts(r.score).reduce((s, c) => s + c, 0)).toBe(n)
  })

  it('shuffles the ratings without changing them', () => {
    const r = drawRatings()
    const s = shuffled(r, 3)
    expect(s.score).toHaveLength(r.score.length)
    const pairs = (x: typeof r): string[] => x.item.map((item, i) => `${item}|${x.rater[i]}|${x.score[i]}`).sort()
    expect(pairs(s)).toEqual(pairs(r))
    expect(s.item).not.toEqual(r.item)
    expect(s.quality).toEqual(r.quality)
  })

  it('returns plain data of the right shapes, narrowing the cut points as ratings arrive', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), { stops: [10, 30] })
    const n = r.ratings.score.length
    expect(r.history.map((h) => h.after)).toEqual([10, 30, n])
    expect(r.final.cuts.mean).toHaveLength(CATEGORIES - 1)
    expect(r.final.quality.mean).toHaveLength(ITEMS)
    expect(r.final.generosity.mean).toHaveLength(RATERS)
    expect(r.final.settled).toBe(true)
    // the cut points come out in increasing order and their spread shrinks with the ratings
    expect(r.final.cuts.mean.every((c, k) => k === 0 || c > (r.final.cuts.mean[k - 1] as number))).toBe(true)
    const spread = r.history.map((h) => h.cuts.sd.reduce((s, v) => s + v, 0) / h.cuts.sd.length)
    expect(spread[0]).toBeGreaterThan(spread[1] as number)
    expect(spread[1]).toBeGreaterThan(spread[2] as number)
    // the expected counts add up to the number of ratings
    expect(r.final.expected).toHaveLength(CATEGORIES)
    expect(r.final.expected.reduce((s, v) => s + v, 0)).toBeCloseTo(n, 6)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 120000)

  it('reads the data and the result for the prose', () => {
    const r = fit(2, undefined, { stops: [10] })
    const data = readData(r.ratings, RATERS, ITEMS)
    expect(data.counts.reduce((s, c) => s + c, 0)).toBe(data.total)
    expect(data.raterHigh).toBeGreaterThan(data.raterLow)
    const reading = readResult(r)
    expect(reading.cuts).toHaveLength(CATEGORIES - 1)
    // the fit follows the true qualities, and gets the counts of each score close
    expect(reading.qualityCorrelation).toBeGreaterThan(0.7)
    expect(reading.generosityCorrelation).toBeGreaterThan(0.3)
    expect(reading.largestGap).toBeLessThan(3)
    expect(reading.cutsInside).toBeGreaterThanOrEqual(0)
    expect(reading.spreadAfter(10)).toBeGreaterThan(reading.spreadAfter(r.ratings.score.length) as number)
  }, 120000)
})
