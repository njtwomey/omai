import { describe, expect, it } from 'vitest'
import { meanOf } from '../base/index.ts'
import { foldIn, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  maternClusterProcess,
  ripleysK,
  ripleysL,
  spatialPoisson,
  thomasK,
  thomasProcess,
} from './spatial-processes.ts'

describe('spatial point processes', () => {
  it('a homogeneous Poisson pattern has K(r) close to pi r^2', () => {
    const pts = spatialPoisson(key(9), 400)
    const r = [0.05, 0.1]
    const k = ripleysK(pts, r, { correction: 'translation' })
    for (let i = 0; i < 2; i++) expect(Math.abs((k[i] ?? 0) - Math.PI * (r[i] ?? 0) ** 2)).toBeLessThan(0.01)
    const l = ripleysL(pts, r, { correction: 'ripley' })
    expect(Math.abs((l[1] ?? 0) - 0.1)).toBeLessThan(0.02)
    const thin = spatialPoisson(key(10), (x) => (x < 0.5 ? 100 : 0), undefined, 100)
    expect(
      toFlat(thin)
        .filter((_, i) => i % 2 === 0)
        .every((x) => x < 0.5),
    ).toBe(true)
    expect(() => spatialPoisson(key(0), (x) => x)).toThrow()
  })

  it('Thomas patterns cluster: their K matches the closed form', () => {
    const params = { parentRate: 25, meanChildren: 8, sigma: 0.03 }
    const runs = Array.from(
      { length: 8 },
      (_, i) => ripleysK(thomasProcess(foldIn(key(11), i), params), [0.05], { correction: 'translation' })[0] ?? 0,
    )
    const want = toFlat(thomasK([0.05], params))[0] ?? 0
    expect(Math.abs(meanOf(runs) - want)).toBeLessThan(0.25 * want)
    expect(want).toBeGreaterThan(Math.PI * 0.05 ** 2)
    const m = maternClusterProcess(key(12), { parentRate: 20, meanChildren: 5, radius: 0.05 })
    expect(m.shape[1]).toBe(2)
    expect(() => thomasProcess(key(0), { ...params, sigma: 0 })).toThrow()
  })
})
