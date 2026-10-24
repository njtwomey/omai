import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet } from '../../testing/supervised.ts'
import {
  LogisticRegression,
  logisticRegression,
  perceptron,
  probitRegression,
  ridgeClassifier,
} from './classification.ts'

const blobs = classificationSet('blobs', { n: 200 }, 1)
const flowers = iris()

describe('linear classifiers', () => {
  it('logistic regression: binary, multinomial and one-versus-rest', () => {
    const f = logisticRegression().fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.9)
    expect(f.predictProba?.(flowers.x).shape).toEqual([150, 3])
    expect(recordsOf(f.exposition, 'coefficientTable')[0]?.names[0]).toBe('setosa:sepal length (cm)')
    expect(logisticRegression({ multiClass: 'ovr' }).fit(flowers).score?.(flowers)).toBeGreaterThan(0.85)
    expect(logisticRegression({ penalty: 'l1', C: 0.5 }).fit(flowers).score?.(flowers)).toBeGreaterThan(0.85)
    const two = new LogisticRegression().fit(blobs.x, blobs.y)
    const surface = recordsOf(two.exposition, 'decisionSurface')[0]
    expect(surface?.probabilities?.length).toBe(3)
    expect(two.model.predictDistribution?.(blobs.x).batchShape).toEqual([200])
  })

  it('probit, ridge classifier and perceptron', () => {
    const binary = { x: flowers.x, y: Array.from(flowers.y.data, (c) => (c === 0 ? 1 : 0)) }
    const p = probitRegression().fit(binary)
    expect(p.score?.(binary)).toBeGreaterThan(0.95)
    expect(recordsOf(p.exposition, 'coefficientTable')[0]?.standardErrors).toHaveLength(5)
    expect(ridgeClassifier().fit(flowers).score?.(flowers)).toBeGreaterThan(0.75)
    const per = perceptron().fit(blobs, { key: key(2) })
    expect(per.score?.(blobs)).toBeGreaterThan(0.6)
    expect(recordsOf(per.exposition, 'decisionSurface')).toHaveLength(1)
  })

  it('record their decision surface through training (IRLS iterations, perceptron updates)', () => {
    const moons = classificationSet('moons', { n: 80, noise: 0.2 }, 3)
    const s = recordsOf(logisticRegression().fit(moons).exposition, 'decisionSurface')[0]
    expect(s?.stepName).toBe('IRLS iteration')
    expect(s?.steps?.length).toBeGreaterThan(2)
    expect(s?.steps?.at(-1)?.probabilities).toEqual(s?.probabilities)
    expect(s?.highlightName).toBe('misclassified')
    expect(s?.objective?.values.every(Number.isFinite)).toBe(true)
    const p = recordsOf(perceptron().fit(moons, { key: key(1) }).exposition, 'decisionSurface')[0]
    expect(p?.stepName).toBe('update')
    expect(p?.steps?.[1]?.highlight).toHaveLength(1)
    expect(p?.steps?.length).toBeLessThanOrEqual(40)
    expect(recordsOf(logisticRegression().fit(moons, { explain: false }).exposition, 'decisionSurface')[0]?.steps).toBe(
      undefined,
    )
  })
})
