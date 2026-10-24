import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { xy, xyClasses } from '../data/index.ts'
import { accuracyScore, meanSquaredError } from '../metrics/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { real, type SpaceValue } from '../space/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  gridSearchCv,
  halvingGridSearchCv,
  halvingRandomSearchCv,
  type TabularData,
  type TunableEstimator,
} from './index.ts'

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

const schedules = loadFixture('tune/schedules')

/** Predicts a constant (the candidate's c); the halving schedule does not depend on the scores. */
function constant(params: SpaceValue): TunableEstimator {
  const c = Number((params as { c?: number }).c ?? 0)
  return {
    name: 'constant',
    fit: () => ({ predict: (xs: TensorLike) => new Array<number>(asTensor(xs).shape[0] ?? 0).fill(c) }),
  }
}

describe('halving searches run scikit-learn schedules', () => {
  for (const c of schedules.cases) {
    it(c.name, () => {
      const out = c.output as { nCandidates: number[]; nResources: number[] }
      const p = c.params
      const factor = Number(p['factor'])
      const cv = Number(p['cv'])
      const requested = p['minResources']
      const minResources = typeof requested === 'number' ? requested : (requested as 'exhaust' | 'smallest')
      if (c.fn === 'halvingRandom') {
        const n = Number(p['nSamples'])
        const data = xy(
          Array.from({ length: n }, (_, i) => [i]),
          new Array<number>(n).fill(0),
        )
        const fitted = halvingRandomSearchCv(
          constant,
          { c: real(-1, 1) },
          { metric: meanSquaredError, cv, factor, minResources },
        ).fit(data)
        expect([fitted.nCandidates, fitted.nResources]).toEqual([out.nCandidates, out.nResources])
        return
      }
      const nCandidates = Number(p['nCandidates'])
      const resource = p['resource'] === undefined ? undefined : 'epochs'
      const n = resource === undefined ? Number(p['nSamples']) : 40
      const classes = Number(p['classes'])
      const rows = Array.from({ length: n }, (_, i) => [i])
      const data =
        classes > 0
          ? xyClasses(
              rows,
              rows.map((_, i) => i % classes),
            )
          : xy(rows, new Array<number>(n).fill(0))
      const fitted = halvingGridSearchCv(
        constant,
        { c: Array.from({ length: nCandidates }, (_, i) => i) },
        {
          metric: classes > 0 ? accuracyScore : meanSquaredError,
          cv,
          factor,
          minResources,
          aggressiveElimination: p['aggressiveElimination'] === true,
          ...(resource === undefined ? {} : { resource, maxResources: Number(p['maxResources']) }),
        },
      ).fit(data)
      expect([fitted.nCandidates, fitted.nResources]).toEqual([out.nCandidates, out.nResources])
    })
  }
})
