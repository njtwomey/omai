import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { ridge } from '../../testing/learn.ts'
import { xy, xyClasses } from '../data/index.ts'
import { defineEstimator } from '../learn/index.ts'
import { accuracyScore, meanSquaredError } from '../metrics/index.ts'
import { real, type SpaceValue } from '../space/index.ts'
import { asTensor, full, type Tensor, type TensorLike } from '../tensor/index.ts'
import { gridSearchCv, halvingGridSearchCv, halvingRandomSearchCv } from './index.ts'

const fixture = loadFixture('tune/wrappers')

/** scikit-learn's parameter names for the ridge of the test helpers. */
function ridgeFor(params: SpaceValue) {
  const p = params as { alpha: number; fit_intercept?: boolean }
  return ridge({ alpha: p.alpha, fitIntercept: p.fit_intercept ?? true })
}

describe('search wrappers against scikit-learn', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const grids = c.inputs[2] as Record<string, (number | boolean)[]>[]
      const fitted = gridSearchCv(ridgeFor, grids, { metric: meanSquaredError, cv: Number(c.params['cv']) }).fit(
        xy(x, y),
      )
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

/** Predicts a constant (the candidate's c), as a regressor or a classifier; the schedule ignores the scores. */
const constant = (kind: 'regressor' | 'classifier') =>
  defineEstimator(
    {
      name: 'constant',
      kind,
      fit: () => ({}),
      methods: (_p, h: { readonly c: number }) => ({
        predict: (x: TensorLike) => full([asTensor(x).shape[0] ?? 0], h.c),
      }),
    },
    { c: 0 },
  )

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
          constant('regressor'),
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
        constant(classes > 0 ? 'classifier' : 'regressor'),
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
