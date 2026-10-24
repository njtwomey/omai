import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const noise = (k: ReturnType<typeof key>, n: number): number[] => toFlat(normal(k, [n]))
const walk = (k: ReturnType<typeof key>, n: number): number[] => {
  let s = 0
  return noise(k, n).map((v) => {
    s += v
    return s
  })
}

describe('time-series tests', () => {
  it('portmanteau and runs tests are valid for white noise', () => {
    expectUniformPValues(nullPValues(300, 1, (k) => T.ljungBox(noise(k, 100), { lags: 5 }).pValue))
    expectUniformPValues(
      nullPValues(300, 2, (k) => T.runsTest(noise(k, 60)).pValue),
      { discrete: true },
    )
    expectUniformPValues(
      nullPValues(300, 3, (k) => T.durbinWatson(noise(k, 100)).pValue),
      { maxDistance: 0.15 },
    )
  })

  it('ADF keeps its size on random walks and rejects for stationary series; KPSS the reverse', () => {
    const ps = nullPValues(150, 4, (k) => T.adfuller(walk(k, 120), { autolag: null, maxlag: 1 }).pValue)
    expectUniformPValues(ps, { maxDistance: 0.15 })
    expect(T.adfuller(noise(key(5), 150)).pValue).toBeLessThan(0.01)
    expect(T.kpss(walk(key(6), 200)).pValue).toBe(0.01)
    expect(T.kpss(noise(key(7), 200)).pValue).toBeGreaterThan(0.05)
  })

  it('critical values tighten towards the asymptotic ones', () => {
    const inf = T.mackinnoncrit(Number.POSITIVE_INFINITY, 'ct')
    const small = T.mackinnoncrit(50, 'ct')
    expect(small[0]).toBeLessThan(inf[0])
    expect(T.mackinnonp(inf[1], 'ct')).toBeCloseTo(0.05, 2)
    expect(() => T.adfuller([1, 1, 1, 1, 1])).toThrow()
  })
})
