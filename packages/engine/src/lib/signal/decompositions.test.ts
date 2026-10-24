import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as S from './index.ts'

goldenSuite('decompositions against vmdpy and numpy', 'signal/decompositions', {
  vmd: ([x], o) => S.vmd(x, o as unknown as S.VmdOptions).modes,
  vmdOmega: ([x], o) => S.vmd(x, o as unknown as S.VmdOptions).omega,
  ssa: ([x], o) => S.ssa(x, o['window'] as number).components,
})

const sumRows = (m: { shape: readonly number[] }, data: readonly number[]): number[] => {
  const [k = 0, n = 0] = m.shape
  return Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let r = 0; r < k; r++) s += data[r * n + i] ?? 0
    return s
  })
}

describe('decompositions', () => {
  it('are complete: modes and residue sum to the signal', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 40, maxLength: 60 }), (x) => {
        const d = S.emd(x, { maxImfs: 4 })
        const total = sumRows(d.imfs, toFlat(d.imfs))
        const r = toFlat(d.residue)
        for (const [i, v] of x.entries()) expect((total[i] ?? 0) + (r[i] ?? 0)).toBeCloseTo(v, 10)
        const s = S.ssa(x, 8)
        const c = sumRows(s.components, toFlat(s.components))
        for (const [i, v] of x.entries()) expect(c[i]).toBeCloseTo(v, 9)
      }),
      { numRuns: RUNS },
    )
  })

  it('separates tones and keeps ensemble decompositions complete', () => {
    const n = 400
    const fast = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * i) / 8))
    const slow = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * i) / 80))
    const x = fast.map((v, i) => v + (slow[i] ?? 0))
    const d = S.emd(x)
    const first = toFlat(d.imfs).slice(0, n)
    let err = 0
    let energy = 0
    for (let i = 50; i < n - 50; i++) {
      err += ((first[i] ?? 0) - (fast[i] ?? 0)) ** 2
      energy += (fast[i] ?? 0) ** 2
    }
    expect(err / energy).toBeLessThan(0.05)
    for (const e of [
      S.eemd(x, { key: key(3), trials: 6, maxImfs: 3 }),
      S.ceemdan(x, { key: key(4), trials: 6, maxImfs: 3 }),
    ]) {
      const total = sumRows(e.imfs, toFlat(e.imfs))
      const r = toFlat(e.residue)
      for (const [i, v] of x.entries()) expect((total[i] ?? 0) + (r[i] ?? 0)).toBeCloseTo(v, 9)
    }
  })

  it('traces the VMD updates', () => {
    const x = Array.from({ length: 100 }, (_, i) => Math.cos(0.3 * i) + Math.cos(1.5 * i))
    const rec = recorder()
    const r = S.vmd(x, { k: 2, tracer: rec })
    expect(r.iterations).toBeGreaterThan(1)
    expect(rec.run().events.length).toBeGreaterThan(0)
  })
})
