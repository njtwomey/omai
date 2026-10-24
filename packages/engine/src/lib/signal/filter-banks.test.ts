import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { concat, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('filter banks against scipy and PyWavelets', 'signal/filter_banks', {
  qmf: ([h]) => S.qmf(h),
  packetNodes: ([x], o) => {
    const p = S.waveletPacket(x, o['wavelet'] as string, {
      mode: o['mode'] as S.WaveletMode,
      level: o['level'] as number,
    })
    return concat(S.packetPaths(o['level'] as number, o['order'] as 'freq').map((path) => p.nodes.get(path) as never))
  },
  packetReconstruct: ([x], o) =>
    S.packetReconstruct(
      S.waveletPacket(x, o['wavelet'] as string, { mode: o['mode'] as S.WaveletMode, level: o['level'] as number }),
    ),
})

describe('perfect reconstruction', () => {
  it('holds for every wavelet bank, with power complementarity for the orthogonal ones', () => {
    const failures: string[] = []
    // The discrete Meyer filters are a truncated approximation, so only nearly perfect.
    for (const name of S.waveletList().filter((n) => n !== 'dmey')) {
      const bank = S.waveletBank(name)
      const check = S.checkReconstruction(bank)
      if (!check.perfect) failures.push(name)
      if (S.wavelet(name).orthogonal) expect(S.powerComplementarity(bank)).toBeLessThan(1e-9)
    }
    expect(failures).toEqual([])
    expect(S.checkReconstruction({ h0: [1, 1], h1: [1, -1], g0: [1, 1], g1: [1, 1] }).perfect).toBe(false)
  })

  it('reconstructs through DFT filter banks and wavelet packets', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 32, maxLength: 32 }),
        fc.constantFrom<[number, number]>([8, 2], [8, 4], [16, 4]),
        (x, [channels, decimation]) => {
          const y = toFlat(S.dftSynthesis(S.dftAnalysis(x, { channels, decimation }), 32, { channels, decimation }))
          for (const [i, v] of x.entries()) expect(y[i]).toBeCloseTo(v, 9)
          const p = S.waveletPacket(x, 'db3', { level: 2 })
          const r = toFlat(S.packetReconstruct(p))
          for (const [i, v] of x.entries()) expect(r[i]).toBeCloseTo(v, 9)
        },
      ),
      { numRuns: RUNS },
    )
  })
})
