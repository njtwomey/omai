import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { randomProjection } from './random-projection.ts'

const x = Array.from({ length: 30 }, (_, i) => Array.from({ length: 200 }, (_, j) => Math.sin(i * 0.37 + j * 0.11)))

describe('random projection', () => {
  it('Gaussian and sparse projections roughly preserve pairwise distances at the JL dimension', () => {
    for (const kind of ['gaussian', 'sparse'] as const) {
      const f = randomProjection({ nComponents: 'auto', eps: 0.5, kind }).fit({ x }, { key: key(3) })
      expect(f.params.components.length).toBeGreaterThan(10)
      expect(f.params.distortion.min).toBeGreaterThan(0.5)
      expect(f.params.distortion.max).toBeLessThan(1.5)
      expect(f.transform(x).shape).toEqual([30, f.params.components.length])
    }
  })

  it('is keyed and has a pseudo-inverse direction', () => {
    const a = randomProjection({ nComponents: 4 }).fit({ x }, { key: key(1) })
    const b = randomProjection({ nComponents: 4 }).fit({ x }, { key: key(1) })
    expect(a.params.components).toEqual(b.params.components)
    expect(toFlat(a.inverseTransform(a.transform(x))).length).toBe(30 * 200)
  })
})
