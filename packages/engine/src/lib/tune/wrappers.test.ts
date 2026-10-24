import { describe, expect, it } from 'vitest'
import { xy, xyClasses } from '../data/index.ts'
import { accuracyScore, meanSquaredError } from '../metrics/index.ts'
import { real } from '../space/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { recorder, runTree } from '../trace/index.ts'
import {
  estimatorObjective,
  gridSearchCv,
  halvingGridSearchCv,
  halvingRandomSearchCv,
  randomizedSearchCv,
  type TunableEstimator,
  tune,
} from './index.ts'

function constant(p: unknown): TunableEstimator {
  const { c, epochs } = p as { c: number; epochs?: number }
  return {
    name: 'constant',
    fit: () => ({
      predict: (x: TensorLike) =>
        new Array<number>(asTensor(x).shape[0] ?? 0).fill(c + (epochs === undefined ? 0 : 1 / epochs)),
    }),
  }
}

const data = xy(
  Array.from({ length: 60 }, (_, i) => [i]),
  new Array<number>(60).fill(2),
)

describe('search wrappers', () => {
  it('grid search ranks every candidate and refits the best', () => {
    const fitted = gridSearchCv(constant, { c: [0, 1, 2, 3] }, { metric: meanSquaredError, cv: 3 }).fit(data)
    expect(fitted.bestParams).toEqual({ c: 2 })
    expect(fitted.cvResults.rankTestScore).toEqual([4, 2, 1, 2])
    expect(toFlat(asTensor(fitted.predict([[0], [1]])))[0]).toBe(2)
    expect(() =>
      gridSearchCv(constant, { c: [1] }, { metric: meanSquaredError, refit: false })
        .fit(data)
        .predict([[0]]),
    ).toThrow()
  })
  it('grid search over a space grids it', () => {
    const fitted = gridSearchCv(
      constant,
      { kind: 'dict', spaces: { c: real(0, 4) } },
      { metric: meanSquaredError, cv: 2, points: 5 },
    ).fit(data)
    expect(fitted.cvResults.params.length).toBe(5)
    expect(fitted.bestParams).toEqual({ c: 2 })
  })
  it('random search is reproducible from its key', () => {
    const search = randomizedSearchCv(
      constant,
      { c: real(0, 4) },
      { metric: meanSquaredError, cv: 2, nIter: 5, seed: 7 },
    )
    expect(search.fit(data).cvResults.params).toEqual(search.fit(data).cvResults.params)
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
      halvingRandomSearchCv(constant, { c: real(0, 1) }, { metric: meanSquaredError, minResources: 'exhaust' }).fit(
        data,
      ),
    ).toThrow()
  })
  it('searches nest: a search is an estimator', () => {
    const inner = gridSearchCv(constant, { c: [1, 2] }, { metric: meanSquaredError, cv: 2 })
    const outer = estimatorObjective(() => inner, data, { metric: meanSquaredError, cv: 3 })
    expect(tune(outer, real(0, 1), { nTrials: 1 }).best?.values[0]).toBe(0)
  })
  it('classification searches use stratified folds and traces nest', () => {
    const x = Array.from({ length: 24 }, (_, i) => [i / 24])
    const y = x.map(([v]) => ((v ?? 0) > 0.5 ? 1 : 0))
    const stump = (p: unknown): TunableEstimator => ({
      name: 'stump',
      fit: () => ({
        predict: (rows: TensorLike) => toFlat(asTensor(rows)).map((v) => (v > (p as { t: number }).t ? 1 : 0)),
      }),
    })
    const rec = recorder()
    const fitted = gridSearchCv(stump, { t: [0.25, 0.5, 0.75] }, { metric: accuracyScore, cv: 4, tracer: rec }).fit(
      xyClasses(x, y),
    )
    expect(fitted.bestParams).toEqual({ t: 0.5 })
    expect(runTree(rec.run())[0]?.children[0]?.children.length).toBe(3)
  })
})
