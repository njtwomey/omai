import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { xy } from '../data/index.ts'
import { meanSquaredError } from '../metrics/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { gridSearchCv, type TabularData, type TunableEstimator } from './index.ts'

const fixture = loadFixture('tune/wrappers')

/** scikit-learn's Ridge (closed form, optionally with an unpenalised intercept), for the test only. */
function ridge(params: SpaceValue): TunableEstimator {
  const p = params as { alpha: number; fit_intercept?: boolean }
  const intercept = p.fit_intercept ?? true
  return {
    name: 'ridge',
    fit: (data: TabularData) => {
      const [n = 0, d = 0] = data.x.shape
      const x = toFlat(data.x)
      const y = toFlat(data.y)
      const mx = Array.from({ length: d }, (_, j) =>
        intercept ? x.filter((_, i) => i % d === j).reduce((a, b) => a + b, 0) / n : 0,
      )
      const my = intercept ? y.reduce((a, b) => a + b, 0) / n : 0
      const a = Array.from({ length: d }, (_, i) =>
        Array.from({ length: d }, (_, j) => {
          let s = i === j ? p.alpha : 0
          for (let r = 0; r < n; r++) s += ((x[r * d + i] ?? 0) - (mx[i] ?? 0)) * ((x[r * d + j] ?? 0) - (mx[j] ?? 0))
          return s
        }),
      )
      const b = Array.from({ length: d }, (_, i) => {
        let s = 0
        for (let r = 0; r < n; r++) s += ((x[r * d + i] ?? 0) - (mx[i] ?? 0)) * ((y[r] ?? 0) - my)
        return s
      })
      const w = toFlat(solve(a, b))
      const b0 = my - w.reduce((s, wi, i) => s + wi * (mx[i] ?? 0), 0)
      return {
        predict: (xs: TensorLike) => {
          const t = asTensor(xs)
          const [m = 0] = t.shape
          const f = toFlat(t)
          return Array.from({ length: m }, (_, r) => b0 + w.reduce((s, wi, i) => s + wi * (f[r * d + i] ?? 0), 0))
        },
      }
    },
  }
}

describe('search wrappers against scikit-learn', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const grids = c.inputs[2] as Record<string, (number | boolean)[]>[]
      const fitted = gridSearchCv(ridge, grids, { metric: meanSquaredError, cv: Number(c.params['cv']) }).fit(xy(x, y))
      const out = c.output as { params: SpaceValue[]; mean: number[]; std: number[]; rank: number[]; bestIndex: number }
      expect(fitted.cvResults.params).toEqual(out.params)
      for (const [i, w] of out.mean.entries()) expect(fitted.cvResults.meanTestScore[i]).toBeCloseTo(w, 9)
      for (const [i, w] of out.std.entries()) expect(fitted.cvResults.stdTestScore[i]).toBeCloseTo(w, 9)
      expect(fitted.cvResults.rankTestScore).toEqual(out.rank)
      expect(fitted.bestIndex).toBe(out.bestIndex)
    })
  }
})
