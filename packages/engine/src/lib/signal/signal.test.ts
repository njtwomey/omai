import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { complex, tensor, toArray } from '../tensor/index.ts'
import * as S from './index.ts'

describe('signal values', () => {
  it('creates signals and their times', () => {
    const s = S.createSignal(
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      { fs: 2, start: 1, channels: ['a', 'b'], units: 'V' },
    )
    expect(S.duration(s)).toBe(1.5)
    expect(toArray(S.sampleTimes(s))).toEqual([1, 1.5, 2])
    expect(() => S.createSignal([1], { fs: 0 })).toThrow()
    expect(() => S.createSignal([[1, 2]], { channels: ['a'] })).toThrow()
  })

  it('takes decibels of complex values by magnitude', () => {
    expect(toArray(S.magnitudeDb(complex([3, 0], [4, 10])))).toEqual(toArray(S.magnitudeDb([5, 10])))
  })

  it('differentiates decibels and unwrapping', () => {
    expectConformant((x) => S.magnitudeDb(x), [tensor([0.5, 2, 3])])
    expectConformant((x) => S.magnitudeDb(x, { power: true, topDb: 3 }), [tensor([0.5, 2, 3])])
    expectConformant((x) => S.unwrap(x), [tensor([0.1, 3, 6.2, 1])])
  })
})

describe('window metrics', () => {
  it('gives the classical figures of merit', () => {
    const hann = S.windowMetrics(S.hann(128, { sym: false }))
    expect(hann.enbw).toBeCloseTo(1.5, 12)
    expect(hann.coherentGain).toBeCloseTo(0.5, 12)
    expect(hann.sidelobeLevel).toBeCloseTo(-31.47, 1)
    expect(hann.scallopLoss).toBeCloseTo(1.42, 2)
    expect(hann.halfPowerWidth).toBeCloseTo(1.44, 1)
    const hamming = S.windowMetrics(S.hamming(128, { sym: false }))
    expect(hamming.sidelobeLevel).toBeCloseTo(-42.7, 0)
    expect(hamming.enbw).toBeCloseTo(1.36, 2)
    const rect = S.windowMetrics(S.boxcar(64))
    expect(rect.mainLobeWidth).toBe(2)
    expect(rect.sidelobeLevel).toBeCloseTo(-13.26, 1)
    expect(rect.scallopLoss).toBeCloseTo(3.92, 1)
    const bh = S.windowMetrics(S.blackmanHarris(128, { sym: false }))
    expect(bh.sidelobeLevel).toBeLessThan(-92)
  })

  it('rejects unknown windows and missing parameters', () => {
    expect(() => S.getWindow('nope', 8)).toThrow()
    expect(() => S.getWindow('kaiser', 8)).toThrow()
    expect(() => S.getWindow(['hann', 1], 8)).toThrow()
    expect(() => S.dpss(8, 5)).toThrow()
  })
})
