import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { swissRoll } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { IncrementalPCA, PCA, pca } from './pca.ts'

const { data } = generate(swissRoll, { key: key(0), knobs: { n: 80 } })

describe('PCA', () => {
  it('variance ratios are sorted, sum to at most one, and reconstruction error falls with rank', () => {
    const f = pca().fit(data)
    const r = f.params.explainedVarianceRatio
    for (let i = 1; i < r.length; i++) expect(r[i] ?? 0).toBeLessThanOrEqual(r[i - 1] ?? 0)
    expect(r.reduce((a, v) => a + v, 0)).toBeLessThanOrEqual(1 + 1e-12)
    expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10)
    const [rec] = recordsOf(f.exposition, 'components')
    expect(rec?.scores?.length).toBe(80)
    const err = rec?.reconstructionError ?? []
    expect(err.at(-1)).toBeCloseTo(0, 10)
    const back = toArray(f.inverseTransform(f.transform(data.x))) as number[][]
    const x = toArray(data.x as never) as number[][]
    expect(back[3]?.[1]).toBeCloseTo(x[3]?.[1] ?? 0, 10)
  })

  it('whitening gives unit-variance scores; randomised and incremental PCA find the same axes', () => {
    const w = new PCA({ nComponents: 2, whiten: true }).fitTransform(data.x)
    const z = toArray(w) as number[][]
    const v = z.reduce((a, r) => a + (r[0] ?? 0) ** 2, 0) / (z.length - 1)
    expect(v).toBeCloseTo(1, 10)
    const full = pca({ nComponents: 2 }).fit(data).params.components
    const rand = pca({ nComponents: 2, svdSolver: 'randomized' }).fit(data, { key: key(1) }).params.components
    for (let c = 0; c < 2; c++) for (let j = 0; j < 3; j++) expect(rand[c]?.[j] ?? 0).toBeCloseTo(full[c]?.[j] ?? 0, 6)
    const inc = new IncrementalPCA({ nComponents: 3, batchSize: 20 }).fit(data.x).model.params
    for (let j = 0; j < 3; j++)
      expect(Math.abs(inc.components[0]?.[j] ?? 0)).toBeCloseTo(Math.abs(full[0]?.[j] ?? 0), 6)
  })
})
