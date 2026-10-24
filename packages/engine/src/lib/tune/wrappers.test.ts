import { describe, expect, it } from 'vitest'
import { xy, xyClasses } from '../data/index.ts'
import { defineEstimator, isEstimator } from '../learn/index.ts'
import { crossValScore } from '../learn/model-selection/index.ts'
import { accuracyScore, meanSquaredError } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { real } from '../space/index.ts'
import { asTensor, full, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { recorder, runTree } from '../trace/index.ts'
import {
  estimatorObjective,
  gridSearchCv,
  halvingGridSearchCv,
  halvingRandomSearchCv,
  randomizedSearchCv,
  tune,
} from './index.ts'

/** Predicts c (plus 1 / epochs when an epoch budget is set). */
const constant = defineEstimator(
  {
    name: 'constant',
    kind: 'regressor',
    fit: () => ({}),
    methods: (_p, h: { readonly c: number; readonly epochs?: number }) => ({
      predict: (x: TensorLike) => full([asTensor(x).shape[0] ?? 0], h.c + (h.epochs === undefined ? 0 : 1 / h.epochs)),
    }),
  },
  { c: 0 },
)

/** Classifies by a threshold t on the first feature. */
const stump = defineEstimator(
  {
    name: 'stump',
    kind: 'classifier',
    fit: () => ({}),
    methods: (_p, h: { readonly t: number }) => ({
      predict: (rows: TensorLike) => tensor(toFlat(asTensor(rows)).map((v) => (v > h.t ? 1 : 0))),
    }),
  },
  { t: 0.5 },
)

const data = xy(
  Array.from({ length: 60 }, (_, i) => [i]),
  new Array<number>(60).fill(2),
)

describe('search wrappers', () => {
  it('grid search ranks every candidate and refits the best', () => {
    const fitted = gridSearchCv(constant, { c: [0, 1, 2, 3] }, { metric: meanSquaredError, cv: 3 }).fit(data)
    expect(fitted.bestParams).toEqual({ c: 2 })
    expect(fitted.cvResults.rankTestScore).toEqual([4, 2, 1, 2])
    expect(toFlat(fitted.predict?.([[0], [1]]) ?? asTensor([]))[0]).toBe(2)
    const unfitted = gridSearchCv(constant, { c: [1] }, { metric: meanSquaredError, refit: false }).fit(data)
    expect(unfitted.predict).toBeUndefined()
    expect(unfitted.best).toBeUndefined()
    expect(() => gridSearchCv(constant, [], { metric: meanSquaredError })).toThrow(/empty/)
  })
  it('grid search over a space grids it, and the estimator score is the default', () => {
    const fitted = gridSearchCv(constant, { kind: 'dict', spaces: { c: real(0, 4) } }, { cv: 2, points: 5 }).fit(
      xy(
        Array.from({ length: 6 }, (_, i) => [i]),
        [1, 2, 3, 1, 2, 3],
      ),
    )
    expect(fitted.cvResults.params.length).toBe(5)
    expect(fitted.bestParams).toEqual({ c: 2 })
    expect(fitted.studies[0]?.directions).toEqual(['maximize'])
  })
  it('random search is reproducible from its key', () => {
    const search = randomizedSearchCv(constant, { c: real(0, 4) }, { metric: meanSquaredError, cv: 2, nIter: 5 })
    expect(search.fit(data, { key: key(7) }).cvResults.params).toEqual(
      search.fit(data, { key: key(7) }).cvResults.params,
    )
  })
  it('halving with a hyperparameter as the resource', () => {
    const fitted = halvingGridSearchCv(
      constant,
      { c: [0, 1, 2, 3, 4, 5, 6, 7, 8] },
      { metric: meanSquaredError, cv: 2, resource: 'epochs', maxResources: 9 },
    ).fit(data)
    expect(fitted.nResources).toEqual([1, 3, 9])
    expect(fitted.cvResults.params.slice(-1)).toEqual([{ c: 2, epochs: 9 }])
    expect(fitted.bestParams).toEqual({ c: 2, epochs: 9 })
    expect(() =>
      halvingGridSearchCv(constant, { c: [1] }, { metric: meanSquaredError, resource: 'epochs' }).fit(data),
    ).toThrow()
  })
  it('halving random search needs one of its two exhausts', () => {
    expect(() =>
      halvingRandomSearchCv(constant, { c: real(0, 1) }, { metric: meanSquaredError, minResources: 'exhaust' }),
    ).toThrow()
  })
  it('searches are learn estimators: they nest, revive and cross-validate', () => {
    const inner = gridSearchCv(constant, { c: [1, 2] }, { metric: meanSquaredError, cv: 2 })
    expect(isEstimator(inner)).toBe(true)
    const outer = estimatorObjective(() => inner, data, { metric: meanSquaredError, cv: 3 })
    expect(tune(outer, real(0, 1), { nTrials: 1 }).best?.values[0]).toBe(0)
    const fitted = inner.fit(data)
    const revived = inner.revive(fitted.params)
    expect(revived.bestParams).toEqual({ c: 2 })
    expect(toFlat(revived.predict?.([[0]]) ?? asTensor([]))).toEqual([2])
    expect(crossValScore(inner, data, { cv: 2, scoring: 'neg_mean_squared_error' })).toEqual([-0, -0])
  })
  it('classification searches use stratified folds and traces nest', () => {
    const x = Array.from({ length: 24 }, (_, i) => [i / 24])
    const y = x.map(([v]) => ((v ?? 0) > 0.5 ? 1 : 0))
    const rec = recorder()
    const fitted = gridSearchCv(stump, { t: [0.25, 0.5, 0.75] }, { metric: accuracyScore, cv: 4 }).fit(
      xyClasses(x, y),
      {
        tracer: rec,
      },
    )
    expect(fitted.bestParams).toEqual({ t: 0.5 })
    expect(fitted.kind).toBe('classifier')
    expect(runTree(rec.run())[0]?.children[0]?.children.length).toBe(3)
  })
})
