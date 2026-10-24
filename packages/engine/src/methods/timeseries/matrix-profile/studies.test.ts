import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import * as T from './index.ts'

describe('matrix-profile studies', () => {
  it('recovers planted motifs, discords and regime boundaries', () => {
    for (const seed of [2, 3, 4]) {
      const s = T.discoveryStudy(key(seed), { n: 300, m: 15 })
      expect(s.motifRecall).toBeGreaterThanOrEqual(2 / 3)
      expect(s.discordFound).toBe(true)
      expect(s.boundaryError).toBeLessThan(2 * 15)
    }
  })

  it('anytime errors never increase and end exact', () => {
    const t = T.plantedSeries(key(5), { n: 160, m: 10 }).t
    for (const method of ['scrump', 'stamp'] as const) {
      const curve = T.anytimeStudy(t, 10, method, { batches: 8, key: key(6) })
      for (let i = 1; i < curve.length; i++)
        expect(curve[i]?.meanError ?? 0).toBeLessThanOrEqual((curve[i - 1]?.meanError ?? 0) + 1e-12)
      expect(curve.at(-1)?.exactFraction).toBe(1)
    }
  })

  it('the one-dimensional row of mSTUMP is the best single-dimension profile', () => {
    const a = T.plantedSeries(key(7), { n: 120, m: 8 }).t
    const b = Float64Array.from(a, (v, i) => Math.cos(i / 4) + 0.1 * v)
    const mp = T.mstump([Array.from(a), Array.from(b)], 8)
    const pa = T.stump(a, 8).P
    const pb = T.stump(b, 8).P
    for (let i = 0; i < pa.length; i++) expect(mp.P[i]).toBeCloseTo(Math.min(pa[i] ?? 0, pb[i] ?? 0), 6)
  })

  it('chains are mutual left-right links and partition their windows', () => {
    const t = T.plantedSeries(key(8), { n: 150, m: 8 }).t
    const mp = T.stump(t, 8)
    const { chains } = T.allc(mp.IL, mp.IR)
    const seen = new Set<number>()
    for (const c of chains)
      for (let i = 0; i < c.length; i++) {
        const w = c[i] ?? 0
        if (i > 0) expect(mp.IL[w]).toBe(c[i - 1])
        if (i + 1 < c.length) expect(mp.IR[w]).toBe(c[i + 1])
        seen.add(w)
      }
    expect(seen.size).toBe(mp.IL.length)
  })
})
