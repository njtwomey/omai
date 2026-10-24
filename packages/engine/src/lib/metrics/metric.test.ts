import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

describe('metric infrastructure', () => {
  it('streams batches to the same result as one call', () => {
    const s = M.streaming(M.f1Score)
    s.update([0, 1, 2, 2], [0, 2, 2, 2])
    s.update([1, 0], [1, 1], [2, 1])
    expect(s.count).toBe(6)
    const whole = M.f1Score([0, 1, 2, 2, 1, 0], [0, 2, 2, 2, 1, 1], {
      average: 'macro',
      sampleWeight: [1, 1, 1, 1, 2, 1],
    })
    expect(s.compute({ average: 'macro' }).estimate.value).toBe(whole.estimate.value)
    s.reset()
    expect(() => s.compute()).toThrow()
    const strings = M.streaming(M.accuracyScore)
    strings.update(['a', 'b'], ['a', 'a'])
    strings.update(['c'], ['c'])
    expect(strings.compute().estimate.value).toBeCloseTo(2 / 3, 15)
  })

  it('selects rows of numeric and string inputs', () => {
    expect(
      M.selectRows(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        [2, 0],
      ),
    ).toEqual([
      [5, 6],
      [1, 2],
    ])
    expect(M.selectRows(['a', 'b', 'c'], [1, 1])).toEqual(['b', 'b'])
    expect(
      M.rowCount([
        [1, 2],
        [3, 4],
      ]),
    ).toBe(2)
  })

  it('computes sklearn weighted percentiles', () => {
    expect(M.weightedPercentile([3, 1, 2], undefined, 50)).toBe(2)
    expect(M.weightedPercentile([1, 2, 3, 4], [1, 1, 1, 1], 50, false)).toBe(2)
    expect(M.weightedPercentile([1, 2, 3], [0, 0, 0], 50)).toBeNaN()
  })

  it('rejects binary averaging of multi-class targets and mismatched weights', () => {
    expect(() => M.precisionScore([0, 1, 2], [0, 1, 2])).toThrow()
    expect(() => M.precisionScore([0, 1], [0, 1], { posLabel: 7 })).toThrow()
    expect(() => M.accuracyScore([0, 1], [0, 1], { sampleWeight: [1] })).toThrow()
  })

  it('builds metrics from specs and per-output scores', () => {
    const spec = { name: 'demo', direction: 'lower', range: [0, Infinity], inputs: 'values', category: 'demo' } as const
    const absError = M.perOutputMetric(spec, (t, p, w) =>
      M.weightedMean(
        Array.from(t, (ti, i) => Math.abs(ti - (p[i] ?? 0))),
        w,
      ),
    )
    expect(absError.spec).toBe(spec)
    expect(absError([1, 2, 3], [1, 3, 5]).estimate.value).toBe(1)
    expect(absError([1, 2, 3], [1, 3, 5], { sampleWeight: [0, 1, 1] }).estimate.value).toBe(1.5)
    const two = absError(
      [
        [1, 0],
        [2, 0],
      ],
      [
        [1, 1],
        [4, 3],
      ],
    )
    expect(two.estimate.value).toBe(1.5)
    expect(two.breakdown).toEqual({ 'output 0': 1, 'output 1': 2 })
    const custom = M.defineMetric(spec, (x: number) => M.metricResult('demo', x * 2, 1, { half: x }))
    expect(custom(3).estimate.value).toBe(6)
    expect(custom(3).breakdown).toEqual({ half: 3 })
    expect(custom.spec.name).toBe('demo')
  })

  it('reads weights, flat values and columns, and recognises string inputs', () => {
    expect(Array.from(M.sampleWeights('demo', 3))).toEqual([1, 1, 1])
    expect(Array.from(M.sampleWeights('demo', 2, [2, 0.5]))).toEqual([2, 0.5])
    expect(() => M.sampleWeights('demo', 2, [1])).toThrow()
    expect(
      Array.from(
        M.flat([
          [1, 2],
          [3, 4],
        ]),
      ),
    ).toEqual([1, 2, 3, 4])
    const c = M.columns(
      'demo',
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
    )
    expect([c.n, c.k]).toEqual([3, 2])
    expect(() => M.columns('demo', [1, 2], [1, 2, 3])).toThrow()
    expect(M.weightedMean([1, 3], [3, 1])).toBe(1.5)
    expect(M.isStringInput(['a', 'b'])).toBe(true)
    expect(M.isStringInput([1, 2])).toBe(false)
    expect(M.isStringInput([])).toBe(false)
  })
})
