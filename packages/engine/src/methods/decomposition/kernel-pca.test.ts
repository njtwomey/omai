import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { circles } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { KernelPCA, kernelPca } from './kernel-pca.ts'

const { data } = generate(circles, { key: key(0), knobs: { n: 60, noise: 0.03, factor: 0.3 } })

describe('kernel PCA', () => {
  it('an RBF kernel separates concentric circles along one of its leading components', () => {
    const f = kernelPca({ nComponents: 4, kernel: 'rbf', gamma: 10 }).fit(data)
    const z = toArray(f.transform(data.x)) as number[][]
    const y = toFlat(data.y as never)
    const mean = (c: number, j: number) => {
      const v = z.filter((_, i) => y[i] === c).map((r) => r[j] ?? 0)
      return v.reduce((a, b) => a + b, 0) / v.length
    }
    expect(Math.max(...[0, 1, 2, 3].map((j) => Math.abs(mean(0, j) - mean(1, j))))).toBeGreaterThan(0.1)
    expect(recordsOf(f.exposition, 'components')[0]?.scores?.length).toBe(60)
  })

  it('pre-images map projections back near the data', () => {
    const m = new KernelPCA({ nComponents: 4, gamma: 2, fitInverseTransform: true, alpha: 0.01 }).fit(data.x)
    const back = toArray(m.inverseTransform(m.transform(data.x))) as number[][]
    const x = toArray(data.x as never) as number[][]
    const err =
      back.reduce((a, r, i) => a + Math.hypot((r[0] ?? 0) - (x[i]?.[0] ?? 0), (r[1] ?? 0) - (x[i]?.[1] ?? 0)), 0) / 60
    expect(err).toBeLessThan(0.3)
  })
})
