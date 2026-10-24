import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { SVC, svc } from './classifier.ts'
import { linearSvc } from './linear.ts'
import { oneClassSvm, svdd } from './one-class.ts'
import { svr } from './regressor.ts'

const moons = classificationSet('moons', { n: 80, noise: 0.15 }, 1)

describe('support vector machines', () => {
  it('SVC records its margins and the SMO path', () => {
    const f = new SVC({ C: 5, probability: true }).fit(moons.x, moons.y, { key: key(1) })
    expect(f.score(moons.x, moons.y)).toBeGreaterThan(0.9)
    const margin = recordsOf(f.exposition, 'margin')[0]
    expect(margin?.support.length).toBeGreaterThan(3)
    expect(margin?.dualObjective?.length).toBe(margin?.workingSets?.length)
    expect((margin?.dualObjective?.at(-1) ?? 0) <= (margin?.dualObjective?.[0] ?? 0)).toBe(true)
    expect(recordsOf(f.exposition, 'decisionSurface')).toHaveLength(1)
    // The surface through training: steps with the support vectors then, the rows, and the margins f = ±1.
    const surface = recordsOf(f.exposition, 'decisionSurface')[0]
    expect(surface?.steps?.length).toBeGreaterThan(2)
    expect(surface?.steps?.length).toBeLessThanOrEqual(41)
    expect(surface?.steps?.[0]?.iteration).toBe(0)
    expect(surface?.steps?.at(-1)?.highlight).toHaveLength(margin?.support.length ?? -1)
    expect(surface?.decision).toHaveLength(surface?.y.length ?? -1)
    expect(surface?.levels).toEqual([-1, 1])
    expect(surface?.rows?.points).toHaveLength(80)
    expect(margin?.rows?.labels).toHaveLength(80)
    expect(margin?.surface).toBe(surface)
    expect(margin?.steps).toHaveLength(margin?.dualObjective?.length ?? -1)
    expect(f.predictProba(moons.x).shape).toEqual([80, 2])
    const flowers = iris()
    const m = svc({ probability: true }).fit(flowers, { key: key(2) })
    expect(m.score?.(flowers)).toBeGreaterThan(0.9)
    expect(recordsOf(m.exposition, 'margin')).toHaveLength(3)
    const pr = m.predictProba?.(flowers.x)
    expect(pr?.shape).toEqual([150, 3])
    expect(Math.abs((pr?.data[0] ?? 0) + (pr?.data[1] ?? 0) + (pr?.data[2] ?? 0) - 1)).toBeLessThan(1e-6)
    expect(
      linearSvc()
        .fit(flowers, { key: key(3) })
        .score?.(flowers),
    ).toBeGreaterThan(0.85)
  })

  it('SVR, one-class SVM and SVDD', () => {
    const sine = regressionSet('sineCurve', { n: 60 }, 2)
    const r = svr({ C: 10 }).fit(sine)
    expect(r.score?.(sine)).toBeGreaterThan(0.8)
    expect(recordsOf(r.exposition, 'margin')[0]?.margins).toHaveLength(60)
    expect(r.predictDistribution(sine.x).batchShape).toEqual([60])
    const o = oneClassSvm({ nu: 0.2 }).fit(moons)
    const outliers = Array.from(o.predict(moons.x).data).filter((v) => v < 0).length
    expect(outliers).toBeGreaterThan(5)
    expect(outliers).toBeLessThan(30)
    expect(recordsOf(o.exposition, 'anomalyScores')[0]?.scores).toHaveLength(80)
    const s = svdd({ nu: 0.1 }).fit(moons)
    expect(Array.from(s.predict(moons.x).data).filter((v) => v > 0).length).toBeGreaterThan(60)
  })
})
