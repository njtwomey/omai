import { describe, expect, it } from 'vitest'
import {
  calibrationOf,
  chosenKey,
  compareTimes,
  formatMicros,
  machineFactor,
  median,
  summarise,
  table,
} from './ep-perf-stats.ts'

// The pure parts of the EP performance harness: the statistics, the calibration normalisation and the baseline comparison.
describe('ep-perf statistics', () => {
  it('takes the median of an odd and an even count, whatever the order', () => {
    expect(median([3, 1, 2])).toBe(2)
    expect(median([4, 1, 3, 2])).toBe(2.5)
    expect(median([7])).toBe(7)
    expect(() => median([])).toThrow()
  })

  it('summarises the range and the relative spread', () => {
    const s = summarise([10, 12, 11, 9, 13])
    expect(s).toEqual({ median: 11, min: 9, max: 13, n: 5, spread: 4 / 11 })
    expect(summarise([5]).spread).toBe(0)
  })

  it('averages the two ends of the calibration and reports the drift', () => {
    const c = calibrationOf([100, 110, 90], [150, 160, 140])
    expect(c.ms).toBe(125)
    expect(c.drift).toBe(1.5)
  })
})

describe('ep-perf comparison', () => {
  const base = new Map([
    ['a', 1],
    ['b', 2],
    ['c', 4],
  ])

  it('flags only what is slower than the ratio after normalising by the machine', () => {
    const now = new Map([
      ['a', 1.4],
      ['b', 4.5],
      ['c', 4],
      ['new', 9],
    ])
    const out = compareTimes(base, now, 1, 1.5)
    expect(out.map((c) => c.key)).toEqual(['b', 'a', 'c'])
    expect(out.filter((c) => c.slower).map((c) => c.key)).toEqual(['b'])
  })

  it('does not blame a slower machine: a run that is 2x slower on a 2x slower machine has ratio 1', () => {
    const factor = machineFactor(200, 100)
    const now = new Map([
      ['a', 2],
      ['b', 4],
      ['c', 8],
    ])
    for (const c of compareTimes(base, now, factor)) {
      expect(c.ratio).toBeCloseTo(1, 12)
      expect(c.slower).toBe(false)
    }
  })

  it('takes the machine factor per row when it is a function of the key', () => {
    // 'a' was timed on a machine twice as slow as when the baseline was taken, 'b' on an equal one.
    const out = compareTimes(
      base,
      new Map([
        ['a', 2],
        ['b', 2],
      ]),
      (k) => (k === 'a' ? 2 : 1),
      1.5,
    )
    expect(out.find((c) => c.key === 'a')?.ratio).toBe(1)
    expect(out.find((c) => c.key === 'b')?.ratio).toBe(1)
    expect(out.filter((c) => c.slower)).toEqual([])
    expect(compareTimes(base, new Map([['b', 5]]), (k) => (k === 'a' ? 2 : 1), 1.5)[0]?.slower).toBe(true)
  })

  it('still flags a regression on a slower machine', () => {
    const out = compareTimes(base, new Map([['a', 4]]), 2, 1.5)
    expect(out[0]?.ratio).toBe(2)
    expect(out[0]?.slower).toBe(true)
  })
})

describe('ep-perf selection and formatting', () => {
  it('chooses keys by prefix, of the whole key or of what follows the first slash', () => {
    expect(chosenKey(undefined, 'item/atom/gaussian')).toBe(true)
    expect(chosenKey(['item/atom'], 'item/atom/gaussian')).toBe(true)
    expect(chosenKey(['atom/gauss'], 'item/atom/gaussian')).toBe(true)
    expect(chosenKey(['op/'], 'item/atom/gaussian')).toBe(false)
  })

  it('formats durations and tables', () => {
    expect(formatMicros(3.1)).toBe('3.10 us')
    expect(formatMicros(250)).toBe('250.0 us')
    expect(formatMicros(41_000)).toBe('41.0 ms')
    expect(formatMicros(2_310_000)).toBe('2.31 s')
    expect(table(['a', 'b'], [['1', '2']])).toBe('| a | b |\n| --- | --- |\n| 1 | 2 |')
  })
})
