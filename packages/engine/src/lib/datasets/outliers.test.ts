import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import { blobsWithOutliers, linearOutliers, moonsWithOutliers } from './outliers.ts'

const flat = (t: unknown): number[] => Array.from((t as Tensor).data as ArrayLike<number>)

describe('contaminated datasets', () => {
  it('blobs and moons with outliers label exactly the contamination fraction, outliers in the square', () => {
    for (const g of [blobsWithOutliers, moonsWithOutliers]) {
      const { data, truth } = generate(g, { key: key(1), knobs: { n: 200, contamination: 0.15 } })
      const y = flat(data.y)
      expect(y.filter((v) => v === 1).length).toBe(30)
      expect(flat(truth.latents?.['outlier'])).toEqual(y)
      expect(data.classes).toEqual(['inlier', 'outlier'])
      const x = toArray(data.x) as number[][]
      for (const [i, row] of x.entries())
        if (y[i] === 1) for (const v of row) expect(Math.abs(v)).toBeLessThanOrEqual(6)
      // The rows are shuffled: outliers are not all at the end.
      expect(y.slice(0, 100).some((v) => v === 1)).toBe(true)
    }
  })

  it('blob inliers sit near their centres on the circle of radius 2.5', () => {
    const { data } = generate(blobsWithOutliers, {
      key: key(2),
      knobs: { n: 400, contamination: 0, centers: 4, clusterStd: 0.2, shuffle: false },
    })
    const x = toArray(data.x) as number[][]
    const radii = x.map(([a = 0, b = 0]) => Math.hypot(a, b))
    const mean = radii.reduce((s, r) => s + r, 0) / radii.length
    expect(mean).toBeGreaterThan(2.3)
    expect(mean).toBeLessThan(2.7)
    expect(flat(data.y).every((v) => v === 0)).toBe(true)
  })

  it('moons without noise or outliers are sklearn’s moons scaled by 4 about (0.5, 0.25)', () => {
    const { data } = generate(moonsWithOutliers, { knobs: { n: 4, contamination: 0, noise: 0, shuffle: false } })
    // make_moons(4, shuffle=False): (1, 0), (-1, 0), (0, 0.5), (2, 0.5).
    const want = [
      [2, -1],
      [-6, -1],
      [-2, 1],
      [6, 1],
    ]
    const got = toArray(data.x) as number[][]
    for (const [i, r] of want.entries()) for (const [j, v] of r.entries()) expect(got[i]?.[j]).toBeCloseTo(v, 12)
  })

  it('a line with vertical outliers: inliers follow the line, outliers sit offset above it', () => {
    const knobs = { n: 400, contamination: 0.2, noise: 0.5, offset: 20, slope: 2, intercept: -1 }
    const { data, truth } = generate(linearOutliers, { key: key(3), knobs })
    const xs = flat(data.x)
    const ys = flat(data.y)
    const out = flat(truth.latents?.['outlier'])
    expect(out.filter((v) => v === 1).length).toBe(80)
    const resid = ys.map((v, i) => v - (2 * (xs[i] ?? 0) - 1))
    const inlier = resid.filter((_, i) => out[i] === 0)
    const outlier = resid.filter((_, i) => out[i] === 1)
    expect(Math.max(...inlier.map(Math.abs))).toBeLessThan(2.5)
    expect(Math.min(...outlier)).toBeGreaterThan(12)
    expect(toFlat(truth.regressionFunction?.(tensor([[0], [2]])) ?? tensor([]))).toEqual([-1, 3])
  })

  it('leverage outliers sit at large x below the line; both mixes the two kinds', () => {
    const lev = generate(linearOutliers, { key: key(4), knobs: { n: 100, contamination: 0.1, outliers: 'leverage' } })
    const xs = flat(lev.data.x)
    const out = flat(lev.truth.latents?.['outlier'])
    for (const [i, o] of out.entries()) if (o === 1) expect(xs[i]).toBeGreaterThanOrEqual(9)
    const both = generate(linearOutliers, { key: key(4), knobs: { n: 100, contamination: 0.2, outliers: 'both' } })
    const bx = flat(both.data.x)
    const by = flat(both.data.y)
    const bo = flat(both.truth.latents?.['outlier'])
    const above = bo.filter((o, i) => o === 1 && (by[i] ?? 0) > 2 + 1.5 * (bx[i] ?? 0)).length
    expect(above).toBe(10)
  })
})
