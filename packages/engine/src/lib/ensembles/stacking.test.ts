import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares, logisticRegression } from '../../testing/learn.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { iris } from '../datasets/index.ts'
import { crossValPredict } from '../learn/model-selection/cross-validation.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, metricSeries, tensorSeries } from '../trace/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/decision-tree.ts'
import { stackingClassifier, stackingRegressor, votingClassifier, votingRegressor } from './stacking.ts'

const flowers = iris()
const moons = classificationSet('moons', { n: 160, noise: 0.25 }, 3)
const sine = regressionSet('sineCurve', { n: 80 }, 2)

const bases = [
  ['stump', decisionTreeClassifier({ maxDepth: 1, record: 0 })],
  ['tree', decisionTreeClassifier({ maxDepth: 4, record: 0 })],
  ['bayes', gaussianNB()],
] as const

describe('voting', () => {
  it('soft voting averages the bases’ probabilities, hard voting takes the weighted majority', () => {
    const soft = votingClassifier({ estimators: bases, voting: 'soft' }).fit(moons)
    const p = toFlat(soft.predictProba?.(moons.x) as never)
    const each = bases.map(([, e]) => toFlat(e.fit(moons).predictProba?.(moons.x) as never))
    p.forEach((v, i) => {
      expect(v).toBeCloseTo(each.reduce((s, e) => s + (e[i] ?? 0), 0) / 3, 12)
    })
    const hard = votingClassifier({ estimators: bases, voting: 'hard' }).fit(moons)
    expect(hard.predictProba).toBeUndefined()
    expect(hard.score?.(moons)).toBeGreaterThan(0.8)
    // A heavy weight on one base is that base.
    const weighted = votingClassifier({ estimators: bases, weights: [1, 10, 1] }).fit(moons)
    const tree = bases[1][1].fit(moons)
    expect(toFlat(weighted.predict(moons.x))).toEqual(toFlat(tree.predict(moons.x)))
  })

  it('a regression vote is the weighted mean', () => {
    const f = votingRegressor({
      estimators: [
        ['a', decisionTreeRegressor({ maxDepth: 2, record: 0 })],
        ['b', leastSquares()],
      ],
      weights: [1, 3],
    }).fit(sine)
    const a = decisionTreeRegressor({ maxDepth: 2 }).fit(sine).predict(sine.x)
    const b = leastSquares().fit(sine).predict(sine.x)
    toFlat(f.predict(sine.x)).forEach((v, i) => {
      expect(v).toBeCloseTo((toFlat(a)[i] ?? 0) / 4 + (3 * (toFlat(b)[i] ?? 0)) / 4, 10)
    })
  })

  it('logs the bases’ scores, their disagreement and the vote’s accuracy as bases are added', () => {
    const s = votingClassifier({ estimators: bases }).session(moons, {
      key: key(0),
      logs: {
        'ensemble/baseScores': Log(),
        'ensemble/diversity': Log(),
        'eval/accuracy': Log(),
        'ensemble/weights': Log(),
      },
    })
    s.advance()
    expect(s.t).toBe(3)
    const scores = tensorSeries(s.run(), 'ensemble/baseScores')
    expect(scores.map((e) => e.shape[0])).toEqual([1, 2, 3])
    const diversity = tensorSeries(s.run(), 'ensemble/diversity').at(-1)
    expect(diversity?.shape).toEqual([3, 3])
    expect(metricSeries(s.run(), 'eval/accuracy').values).toHaveLength(3)
    expect((s.probe(2, 'model/diversity') as number[][])[0]?.[0]).toBe(0)
    expect(toFlat(s.fittedAt(1).predict(moons.x))).toHaveLength(160)
  })
})

describe('stacking', () => {
  it('fits the final estimator on out-of-fold predictions, never the bases’ in-sample ones', () => {
    const final = logisticRegression()
    const stack = stackingClassifier({
      estimators: [
        ['tree', decisionTreeClassifier({ record: 0 })],
        ['bayes', gaussianNB()],
      ],
      final,
      cv: 4,
    })
    const f = stack.fit(moons, { key: key(2) })
    expect(f.params.final).not.toBeNull()
    expect(f.score?.(moons)).toBeGreaterThan(0.75)
    // The unpruned tree fits its own rows perfectly, but out of fold it does not: the meta-learner knows.
    const oof = toFlat(
      crossValPredict(decisionTreeClassifier({ record: 0 }), moons, { cv: 4, method: 'predictProba', key: key(2) }),
    )
    const own = toFlat(decisionTreeClassifier({ record: 0 }).fit(moons).predictProba?.(moons.x) as never)
    const agree = (a: readonly number[]): number =>
      a.filter((v, i) => (i % 2 === 1 ? Math.round(v) === moons.y.data[(i - 1) / 2] : false)).length
    expect(agree(own)).toBeGreaterThan(agree(oof))
  })

  it('works for regression, with passthrough, and its last step is the final estimator', () => {
    const est = stackingRegressor({
      estimators: [
        ['tree', decisionTreeRegressor({ maxDepth: 3, record: 0 })],
        ['line', leastSquares()],
      ],
      final: leastSquares(),
      passthrough: true,
      cv: 4,
    })
    const s = est.session(sine, { key: key(1), logs: { 'eval/r2': Log() } })
    s.advance()
    expect(s.t).toBe(3)
    expect(metricSeries(s.run(), 'eval/r2').values).toHaveLength(3)
    expect(s.fittedAt(s.t).score?.(sine)).toBeGreaterThan(0.8)
    // Before the last step there is no stack yet: the model is the vote of the bases so far.
    expect(s.fittedAt(2).params.final).toBeNull()
  })

  it('rejects bad hyperparameters', () => {
    expect(() => votingClassifier({ estimators: [] }).fit(flowers)).toThrow(/at least one/)
    expect(() => votingClassifier({ estimators: bases, weights: [1] }).fit(moons)).toThrow(/weights/)
  })
})
