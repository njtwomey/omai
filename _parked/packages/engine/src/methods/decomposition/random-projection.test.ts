import { describe, expect, it } from 'vitest'
import { key, normal } from '../../lib/random/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { RandomProjection, randomProjection } from './random-projection.ts'

const x = toArray(normal(key(0), [40, 200])) as number[][]

describe('random projections', () => {
  it('keep pairwise distances roughly, Gaussian and sparse', () => {
    const g = randomProjection({ nComponents: 100 }).fit({ x }, { key: key(1) })
    expect(g.params.distortion.mean).toBeGreaterThan(0.8)
    expect(g.params.distortion.mean).toBeLessThan(1.2)
    const s = new RandomProjection({ nComponents: 100, kind: 'sparse' }).fit(x, undefined, { key: key(2) }).model
    expect(s.transform(x).shape).toEqual([40, 100])
    expect(s.params.distortion.max).toBeLessThan(1.6)
  })
})
