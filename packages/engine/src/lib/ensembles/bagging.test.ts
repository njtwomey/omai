import { describe, expect, it } from 'vitest'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, metricSeries, tensorSeries } from '../trace/index.ts'
import { decisionTreeClassifier } from '../trees/decision-tree.ts'
import { baggingClassifier, baggingRegressor } from './bagging.ts'

const moons = classificationSet('moons', { n: 150, noise: 0.25 }, 1)
const sine = regressionSet('sineCurve', { n: 80 }, 2)

describe('bagging', () => {
  it('averages bootstrapped trees: it beats one tree out of sample and reports an out-of-bag score', () => {
    const test = classificationSet('moons', { n: 300, noise: 0.25 }, 9)
    const one = decisionTreeClassifier().fit(moons)
    const bag = baggingClassifier({ nEstimators: 25 }).fit(moons, { key: key(1) })
    expect(bag.score?.(test)).toBeGreaterThanOrEqual((one.score?.(test) ?? 1) - 0.02)
    expect(bag.params.members).toHaveLength(25)
    expect(bag.params.oobScore).toBeGreaterThan(0.75)
    expect(toFlat(bag.predictProba?.(test.x) as never)).toHaveLength(300 * 2)
    expect(bag.predictDistribution(test.x).batchShape).toEqual([300])
  })

  it('records the stages of the fit: weights, training and out-of-bag scores, each member’s rows', () => {
    const f = baggingClassifier({ nEstimators: 6 }).fit(moons, { key: key(2) })
    const stages = recordsOf(f.exposition, 'ensembleStages')[0]
    expect(stages?.weights).toEqual([1, 1, 1, 1, 1, 1])
    expect(stages?.trainScores).toHaveLength(6)
    expect(stages?.validationScores?.length).toBe(6)
    expect(stages?.bootstraps?.[0]?.length).toBeGreaterThan(70)
    expect(stages?.bootstraps?.[0]?.length).toBeLessThan(150)
  })

  it('is deterministic for a key, and different keys give different ensembles', () => {
    const a = baggingClassifier({ nEstimators: 4 }).fit(moons, { key: key(5) })
    const b = baggingClassifier({ nEstimators: 4 }).fit(moons, { key: key(5) })
    const c = baggingClassifier({ nEstimators: 4 }).fit(moons, { key: key(6) })
    expect(a.params).toEqual(b.params)
    expect(a.params).not.toEqual(c.params)
  })

  it('regression bagging smooths the curve, with a predictive spread', () => {
    const f = baggingRegressor({ nEstimators: 20 }).fit(sine, { key: key(3) })
    expect(f.score?.(sine)).toBeGreaterThan(0.8)
    expect(f.predictDistribution(sine.x).batchShape).toEqual([80])
  })

  it('pasting and random subspaces are modes', () => {
    const pasted = baggingClassifier({ nEstimators: 5, bootstrap: false, maxSamples: 0.6 }).fit(moons, { key: key(1) })
    expect(pasted.params.members).toHaveLength(5)
    expect(pasted.params.oobScore).not.toBeNull()
    const sub = baggingClassifier({ nEstimators: 5, maxFeatures: 1, bootstrap: true }).fit(moons, { key: key(1) })
    expect(sub.params.members[0]?.features).toEqual([0, 1])
    const half = baggingClassifier({ nEstimators: 6, maxFeatures: 0.5 }).fit(moons, { key: key(1) })
    expect(half.params.members.every((m) => m.features.length === 1)).toBe(true)
    expect(half.predict(moons.x).shape).toEqual([150])
  })

  it('a session adds one member per step, logging the out-of-bag error, and probes a member’s surface', () => {
    const s = baggingClassifier({ nEstimators: 8 }).session(moons, {
      key: key(1),
      logs: {
        'ensemble/member': Log(),
        'ensemble/oobError': Log(),
        'ensemble/memberError': Log(),
        'ensemble/bootstrap': Log({ detail: 'full' }),
      },
    })
    s.advance({ steps: 4 })
    expect(metricSeries(s.run(), 'ensemble/member').values).toEqual([1, 2, 3, 4])
    expect(metricSeries(s.run(), 'ensemble/oobError').values.length).toBe(4)
    expect(metricSeries(s.run(), 'ensemble/memberError').values.length).toBe(4)
    const bags = tensorSeries(s.run(), 'ensemble/bootstrap')
    expect(bags[0]?.shape).toEqual([150])
    expect(bags[0]?.values?.reduce((a, b) => a + b, 0)).toBe(150)
    const surface = s.probe(2, 'model/memberSurface', { member: 1, resolution: 6 }) as { readonly kind: string }
    expect(surface.kind).toBe('decisionSurface')
    expect(toFlat(s.fittedAt(2).predict(moons.x))).toHaveLength(150)
    expect(s.fittedAt(2).params.members).toHaveLength(2)
    s.advance()
    expect(s.fittedAt(s.t).params.members).toHaveLength(8)
  })
})
