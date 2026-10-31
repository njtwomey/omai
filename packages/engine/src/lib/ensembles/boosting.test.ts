import { describe, expect, it } from 'vitest'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { iris } from '../datasets/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, messages, metricSeries, tensorSeries } from '../trace/index.ts'
import { adaBoostClassifier, adaBoostRegressor, bestStump, sortedRows } from './adaboost.ts'
import {
  explainableBoosting,
  explainableBoostingClassifier,
  explainableBoostingRegressor,
  glassBoxPredict,
} from './cyclic.ts'
import { gradientBoostingClassifier, gradientBoostingRegressor } from './gradient-boosting.ts'
import { binFeatures, histGradientBoostingClassifier, histGradientBoostingRegressor } from './histogram-boosting.ts'

const flowers = iris()
const moons = classificationSet('moons', { n: 160, noise: 0.2 }, 3)
const sine = regressionSet('sineCurve', { n: 100 }, 2)

describe('the weighted stump search', () => {
  it('finds the threshold and polarity of least weighted error', () => {
    const values = [
      [0.1, 0.4, 0.6, 0.9],
      [0.5, 0.1, 0.9, 0.3],
    ]
    const s = bestStump(values, values.map(sortedRows), (i) => i >= 2, [0.25, 0.25, 0.25, 0.25])
    expect([s.feature, s.threshold, s.polarity, s.error]).toEqual([0, 0.5, -1, 0])
    // Up-weighting the row the stump gets wrong changes the choice.
    const w = bestStump([[0.1, 0.4, 0.6, 0.9]], [[0, 1, 2, 3]], (i) => i === 0 || i === 3, [0.1, 0.1, 0.1, 0.7])
    expect(w.error).toBeLessThan(0.3)
  })
})

describe('AdaBoost', () => {
  it('concentrates the row weights on the misclassified and shrinks the training error', () => {
    const est = adaBoostClassifier({ nEstimators: 15 })
    const s = est.session(moons, {
      key: key(0),
      logs: {
        'boost/error': Log(),
        'boost/alpha': Log(),
        'boost/trainingError': Log(),
        'ensemble/target': Log({ detail: 'full' }),
      },
    })
    s.advance()
    const err = metricSeries(s.run(), 'boost/trainingError').values
    expect(err.at(-1)).toBeLessThan(err[0] ?? 0)
    const w = s.probe(5, 'boost/weights') as number[]
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(Math.max(...w)).toBeGreaterThan(1 / w.length)
    expect(tensorSeries(s.run(), 'ensemble/target')).toHaveLength(s.t)
    const margins = s.probe(s.t, 'model/margins') as number[]
    expect(margins).toHaveLength(160)
    expect(Math.max(...margins)).toBeLessThanOrEqual(1 + 1e-12)
  })

  it('SAMME.R works with probabilities, and records its stages', () => {
    const f = adaBoostClassifier({ algorithm: 'SAMME.R', nEstimators: 10 }).fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.9)
    expect(f.params.weights.every((v) => v === 1)).toBe(true)
    const stages = recordsOf(f.exposition, 'ensembleStages')[0]
    expect(stages?.targets?.length).toBe(f.params.members.length)
    expect(() =>
      adaBoostClassifier({ algorithm: 'SAMME.R', estimator: gradientBoostingClassifier({ nEstimators: 1 }) }).fit(
        flowers,
      ),
    ).not.toThrow()
  })

  it('stops with an event when a member is perfect, and AdaBoost.R2 fits a curve', () => {
    const x = [[0], [1], [2], [3]]
    const f = adaBoostClassifier({ nEstimators: 20 }).fit({ x, y: [0, 0, 1, 1] }, { logs: { 'boost/stopped': Log() } })
    expect(f.params.members).toHaveLength(1)
    expect(messages(f.run, 'boost/stopped')).toHaveLength(1)
    const r = adaBoostRegressor({ nEstimators: 15 }).fit(sine, { key: key(2) })
    expect(r.score?.(sine)).toBeGreaterThan(0.8)
  })
})

describe('gradient boosting', () => {
  it('one stage per step: the training loss falls, the residuals shrink, staged predictions refine', () => {
    const s = gradientBoostingRegressor({ nEstimators: 20, maxDepth: 2 }).session(sine, {
      key: key(0),
      logs: {
        'boost/trainLoss': Log(),
        'boost/residuals': Log({ detail: 'full' }),
        'boost/leafValues': Log(),
        'eval/loss': Log(),
      },
    })
    s.advance({ steps: 6 })
    const loss = metricSeries(s.run(), 'boost/trainLoss').values
    expect(loss).toHaveLength(6)
    expect(loss.at(-1)).toBeLessThan(loss[0] ?? 0)
    const res = tensorSeries(s.run(), 'boost/residuals')
    const norm = (v: readonly number[] | undefined): number => Math.sqrt((v ?? []).reduce((a, b) => a + b * b, 0))
    expect(norm(res.at(-1)?.values)).toBeLessThan(norm(res[0]?.values))
    const staged = s.probe(3, 'model/stagedPrediction') as number[]
    expect(staged).toHaveLength(100)
    expect(s.fittedAt(3).stagedDecisionFunction(sine.x)).toHaveLength(3)
  })

  it('Newton leaves, subsampling and early stopping', () => {
    const newton = gradientBoostingClassifier({ nEstimators: 30, maxDepth: 2, leaves: 'newton', l2: 1 }).fit(moons)
    expect(newton.score?.(moons)).toBeGreaterThan(0.9)
    expect(() => gradientBoostingRegressor({ leaves: 'newton', loss: 'quantile' }).fit(sine)).toThrow(/smooth/)
    const sub = gradientBoostingRegressor({ nEstimators: 10, subsample: 0.5 }).fit(sine, {
      key: key(1),
      logs: { 'boost/validLoss': Log() },
    })
    expect(metricSeries(sub.run, 'boost/validLoss').values).toHaveLength(10)
    const stopped = gradientBoostingRegressor({
      nEstimators: 300,
      learningRate: 0.5,
      maxDepth: 4,
      validationFraction: 0.3,
      nIterNoChange: 3,
    }).fit(sine, { key: key(1), logs: { 'boost/stopped': Log() } })
    expect(stopped.params.stages.length).toBeLessThan(300)
    expect(messages(stopped.run, 'boost/stopped')).toHaveLength(1)
  })

  it('the absolute, Huber, quantile and Poisson losses and the exponential loss all fit', () => {
    for (const loss of ['absolute_error', 'huber', 'quantile'] as const)
      expect(gradientBoostingRegressor({ loss, nEstimators: 30, alpha: 0.7 }).fit(sine).score?.(sine)).toBeGreaterThan(
        0.5,
      )
    const counts = { x: sine.x, y: toFlat(sine.y as never).map((v) => Math.round(Math.exp(v))) }
    expect(gradientBoostingRegressor({ loss: 'poisson', nEstimators: 30 }).fit(counts).score?.(counts)).toBeGreaterThan(
      0.3,
    )
    expect(
      gradientBoostingClassifier({ loss: 'exponential', nEstimators: 20 }).fit(moons).score?.(moons),
    ).toBeGreaterThan(0.9)
  })

  it('multiclass boosting grows one tree per class per stage', () => {
    const f = gradientBoostingClassifier({ nEstimators: 5, maxDepth: 2 }).fit(flowers)
    expect(f.params.stages[0]).toHaveLength(3)
    expect(f.score?.(flowers)).toBeGreaterThan(0.95)
  })
})

describe('histogram boosting', () => {
  it('bins as scikit-learn: a bin per value when few, percentiles otherwise, missing last', () => {
    const b = binFeatures({ n: 5, d: 1, data: Float64Array.of(1, 2, 2, 4, Number.NaN) }, 255)
    expect(b.edges[0]).toEqual([1.5, 3])
    expect(Array.from(b.codes)).toEqual([0, 1, 1, 2, 3])
    const many = binFeatures({ n: 100, d: 1, data: Float64Array.from({ length: 100 }, (_, i) => i) }, 4)
    expect(many.edges[0]?.length).toBe(3)
  })

  it('fits leaf-wise with missing values, and a session logs the loss per tree', () => {
    const x = sine.x
    const withNaN = toFlat(x as never).map((v, i) => (i % 9 === 0 ? Number.NaN : v))
    const data = { x: withNaN.map((v) => [v]), y: sine.y }
    const f = histGradientBoostingRegressor({ nEstimators: 30, minSamplesLeaf: 5 }).fit(data)
    expect(f.score?.(data)).toBeGreaterThan(0.6)
    expect(toFlat(f.predict([[Number.NaN]]))).toHaveLength(1)
    const s = histGradientBoostingClassifier({ nEstimators: 10, minSamplesLeaf: 5 }).session(moons, {
      key: key(0),
      logs: { 'boost/trainLoss': Log(), 'boost/bins': Log() },
    })
    s.advance({ steps: 4 })
    expect(metricSeries(s.run(), 'boost/trainLoss').values).toHaveLength(4)
    expect(metricSeries(s.run(), 'boost/bins').values[0]).toBeGreaterThan(10)
    expect(toFlat(s.fittedAt(2).predict(moons.x))).toHaveLength(160)
  })
})

describe('explainable boosting', () => {
  const x = Array.from({ length: 60 }, (_, i) => [i % 4, Math.floor(i / 4) % 3])
  const y = x.map(([a = 0, b = 0]) => 2 * a + 4 * b)

  it('keeps the old whole-model function, now the end of a stepped booster', () => {
    const m = explainableBoosting({ x, y }, { bins: 4, rounds: 300, learningRate: 0.2 })
    const first = m.terms[0]?.values as unknown as number[]
    expect(Math.round((first[3] ?? 0) - (first[0] ?? 0))).toBe(6)
    expect(glassBoxPredict(m, [[3, 2]])[0]).toBeCloseTo(14, 0)
  })

  it('an estimator whose probe shows the shape functions as they were learnt, cycle by cycle', () => {
    const est = explainableBoostingRegressor({ bins: 4, rounds: 100, learningRate: 0.3 })
    const s = est.session({ x, y }, { key: key(0), logs: { 'boost/trainLoss': Log() } })
    s.advance()
    const early = s.probe(2, 'model/shapeFunctions') as { terms: { values: number[] }[] }
    const late = s.probe(s.t, 'model/shapeFunctions') as { terms: { values: number[] }[] }
    const range = (v: number[]): number => Math.max(...v) - Math.min(...v)
    expect(range(late.terms[0]?.values ?? [])).toBeGreaterThan(range(early.terms[0]?.values ?? []))
    const loss = metricSeries(s.run(), 'boost/trainLoss').values
    expect(loss.at(-1)).toBeLessThan(loss[0] ?? 0)
    expect(est.fit({ x, y }).score?.({ x, y })).toBeGreaterThan(0.99)
    const c = explainableBoostingClassifier({ bins: 4, rounds: 80, learningRate: 0.3 }).fit({
      x,
      y: y.map((v) => (v > 8 ? 1 : 0)),
    })
    expect(c.score?.({ x, y: y.map((v) => (v > 8 ? 1 : 0)) })).toBeGreaterThan(0.9)
    expect(c.contributions(x)[0]).toHaveLength(2)
  })
})
