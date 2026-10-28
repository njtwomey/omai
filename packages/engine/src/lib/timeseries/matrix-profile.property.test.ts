import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { exclusionZone, zNormalize } from '../signal/index.ts'
import { policies, recorder } from '../trace/index.ts'
import * as T from './index.ts'

const series = fc
  .array(fc.double({ min: -5, max: 5, noNaN: true, noDefaultInfinity: true }), { minLength: 30, maxLength: 60 })
  .map((xs) => Float64Array.from(xs, (v, i) => v + Math.sin(i / 2)))

/** Brute-force nearest-neighbour distances: z-normalise every window and compare all pairs outside the zone. */
function brute(t: Float64Array, m: number, normalize: boolean): Float64Array {
  const l = t.length - m + 1
  const w = Array.from({ length: l }, (_, i) => (normalize ? zNormalize(t.slice(i, i + m)) : t.slice(i, i + m)))
  const zone = exclusionZone(m)
  const out = new Float64Array(l).fill(Number.POSITIVE_INFINITY)
  for (let i = 0; i < l; i++)
    for (let j = 0; j < l; j++) {
      if (Math.abs(i - j) <= zone) continue
      let s = 0
      for (let q = 0; q < m; q++) s += ((w[i]?.[q] ?? 0) - (w[j]?.[q] ?? 0)) ** 2
      out[i] = Math.min(out[i] ?? 0, Math.sqrt(s))
    }
  return out
}

describe('matrix-profile algorithms', () => {
  it('STUMP, STOMP, STAMP and full SCRIMP++ equal the brute-force profile', () => {
    fc.assert(
      fc.property(series, fc.integer({ min: 4, max: 8 }), (t, m) => {
        const ref = brute(t, m, true)
        for (const mp of [T.stump(t, m), T.stomp(t, m), T.stamp(t, m), T.scrump(t, m, { percentage: 1 })])
          for (let i = 0; i < ref.length; i++) expect(mp.P[i]).toBeCloseTo(ref[i] ?? 0, 5)
        const aamp = T.stump(t, m, { normalize: false })
        const raw = brute(t, m, false)
        for (let i = 0; i < raw.length; i++) expect(aamp.P[i]).toBeCloseTo(raw[i] ?? 0, 6)
      }),
      { numRuns: 15 },
    )
  })

  it('anytime algorithms give upper bounds of the exact profile', () => {
    fc.assert(
      fc.property(series, fc.integer({ min: 0, max: 1000 }), (t, seed) => {
        const m = 5
        const exact = T.stump(t, m).P
        const bounds = [
          T.prescrump(t, m, { key: key(seed) }).P,
          T.scrump(t, m, { percentage: 0.2, key: key(seed) }).P,
          T.stamp(t, m, { percentage: 0.3, key: key(seed) }).P,
        ]
        for (const b of bounds)
          for (let i = 0; i < exact.length; i++) expect(b[i] ?? 0).toBeGreaterThanOrEqual((exact[i] ?? 0) - 1e-9)
      }),
      { numRuns: 15 },
    )
  })

  it('top-k rows are ascending and the first column is the k = 1 profile', () => {
    fc.assert(
      fc.property(series, (t) => {
        const m = 5
        const k3 = T.stump(t, m, { k: 3 })
        const k1 = T.stump(t, m)
        for (let i = 0; i < k1.P.length; i++) {
          expect(k3.P[i * 3]).toBeCloseTo(k1.P[i] ?? 0, 9)
          expect(k3.P[i * 3] ?? 0).toBeLessThanOrEqual(k3.P[i * 3 + 1] ?? 0)
          expect(k3.P[i * 3 + 1] ?? 0).toBeLessThanOrEqual(k3.P[i * 3 + 2] ?? 0)
        }
      }),
      { numRuns: 15 },
    )
  })

  it('the incremental profile without egress equals STUMP on the extended series', () => {
    fc.assert(
      fc.property(series, (t) => {
        const m = 5
        const n0 = 20
        let s = T.stumpiInit(t.slice(0, n0), m, { egress: false })
        for (let i = n0; i < t.length; i++) s = T.stumpiUpdate(s, t[i] ?? 0)
        const ref = T.stump(t, m)
        for (let i = 0; i < ref.P.length; i++) expect(s.profile.P[i]).toBeCloseTo(ref.P[i] ?? 0, 6)
      }),
      { numRuns: 10 },
    )
  })

  it('traces progress of an anytime run', () => {
    const t = Float64Array.from({ length: 50 }, (_, i) => Math.sin(i / 3) + 0.1 * Math.cos(i * 1.7))
    const tracer = recorder()
    T.scrump(t, 6, { percentage: 0.5, tracer, policy: policies.minimal })
    const fractions = tracer.events.flatMap((e) => (e.type === 'metric' && e.name === 'fraction' ? [e.value] : []))
    expect(fractions.length).toBeGreaterThan(2)
    expect(fractions.at(-1)).toBeCloseTo(0.5, 1)
  })
})
