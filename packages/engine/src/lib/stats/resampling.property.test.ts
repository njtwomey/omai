import { describe, expect, it } from 'vitest'
import { coverage, expectUniformPValues, nullPValues } from '../../testing/stats.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as S from './index.ts'

describe('resampling', () => {
  it('permutation tests are valid under the null', () => {
    const ps = nullPValues(200, 3, (k) => {
      const x = toFlat(normal(foldIn(k, 0), [8]))
      const y = toFlat(normal(foldIn(k, 1), [9]))
      return S.permutationTest([x, y], S.meanDifference, { key: foldIn(k, 2), nResamples: 199 }).pValue
    })
    expectUniformPValues(ps, { maxDistance: 0.15 })
  })

  it('bootstrap intervals of the mean cover at about the nominal rate', () => {
    for (const method of ['percentile', 'basic', 'bca'] as const) {
      const intervals = Array.from({ length: 120 }, (_, i) => {
        const x = toFlat(normal(foldIn(key(21), i), [25]))
        return (
          S.bootstrap([x], (v) => S.mean(v), { key: foldIn(key(22), i), nResamples: 300, method }).estimate.interval ??
          ([0, 0] as const)
        )
      })
      expect(coverage(intervals, 0)).toBeGreaterThan(0.85)
    }
    const intervals = Array.from({ length: 60 }, (_, i) => {
      const x = toFlat(normal(foldIn(key(23), i), [15]))
      return (
        S.bootstrap([x], (v) => S.mean(v), {
          key: foldIn(key(24), i),
          nResamples: 200,
          method: 'studentized',
          standardError: (v) => S.sem(v),
        }).estimate.interval ?? ([0, 0] as const)
      )
    })
    expect(coverage(intervals, 0)).toBeGreaterThan(0.85)
  })

  it('the jackknife standard error of the mean is the standard error of the mean', () => {
    const x = [2, 7, 1, 8, 2, 8, 1, 8]
    expect(S.jackknife(x, (v) => S.mean(v)).estimate.standardError).toBeCloseTo(S.sem(x), 12)
    expect(S.jackknife(x, (v) => S.mean(v)).bias).toBeCloseTo(0, 12)
  })

  it('records its progress as a traced run and is reproducible from its key', () => {
    const rec = recorder()
    const a = S.permutationTest(
      [
        [1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 10],
      ],
      S.meanDifference,
      {
        seed: 5,
        nResamples: 500,
        tracer: rec,
      },
    )
    const b = S.permutationTest(
      [
        [1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 10],
      ],
      S.meanDifference,
      {
        seed: 5,
        nResamples: 500,
      },
    )
    expect(a.exact).toBe(false)
    expect(a.pValue).toBe(b.pValue)
    expect(rec.run().events.length).toBeGreaterThan(0)
    const mc = S.monteCarloTest(
      [[3, 4, 5]],
      (k, n) => normal(k, [n]).data,
      (v) => S.mean(v),
      { seed: 1, nResamples: 99 },
    )
    expect(mc.pValue).toBeLessThan(0.05)
  })
})
