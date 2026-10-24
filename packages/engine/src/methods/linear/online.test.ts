import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { type Fitted, type LearnData, requireVerb } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  onlinePerceptron,
  PassiveAggressiveClassifier,
  passiveAggressiveRegressor,
  SGDClassifier,
  sgdClassifier,
  sgdRegressor,
} from './online.ts'

const score = (f: Fitted, d: LearnData): number => requireVerb(f, 'score', 'test')(d)
const { data } = generate(blobs, { key: key(5), knobs: { n: 120, centers: 3, clusterStd: 0.8 } })

describe('online linear models', () => {
  for (const [name, make] of [
    ['sgd hinge', () => sgdClassifier()],
    ['sgd log loss', () => sgdClassifier({ loss: 'log_loss' })],
    ['online perceptron', () => onlinePerceptron({ maxIter: 20 })],
  ] as const) {
    it(`${name} separates blobs and records its learning curve`, () => {
      const f = make().fit(data, { key: key(0) })
      expect(score(f, data)).toBeGreaterThan(0.9)
      const [curve] = recordsOf(f.exposition, 'learningCurve')
      expect(curve?.sizes[0]).toBe(12)
      expect(curve?.train.length).toBe(curve?.sizes.length)
      expect(curve?.validation.at(-1)).toBeLessThan(0.15)
      expect(recordsOf(f.exposition, 'decisionSurface')[0]?.labels?.length).toBe(30)
      expect(recordsOf(f.exposition, 'coefficientTable')[0]?.names.length).toBe(9)
    })
  }

  it('log loss gives probabilities; the class form and passive-aggressive fit', () => {
    const f = sgdClassifier({ loss: 'log_loss' }).fit(data, { key: key(1) })
    for (const r of toFlat(requireVerb(f, 'predictProba', 'test')(data.x))) expect(r).toBeGreaterThanOrEqual(0)
    const pa = new PassiveAggressiveClassifier({ maxIter: 20 }).fit(data.x, data.y, { key: key(2) }).model
    expect(score(pa, data)).toBeGreaterThan(0.85)
    expect(new SGDClassifier().fit(data.x, data.y, { key: key(3) }).model.params.coef.length).toBe(3)
  })

  it('SGD and passive-aggressive regressors fit a line', () => {
    const line = {
      x: data.x,
      y: toFlat(data.x)
        .filter((_, i) => i % 2 === 0)
        .map((v) => 2 * v + 1),
    }
    const r = sgdRegressor({ maxIter: 200 }).fit(line, { key: key(4) })
    expect(r.params.coef[0]).toBeCloseTo(2, 1)
    const [curve] = recordsOf(r.exposition, 'learningCurve')
    expect(curve?.validation.at(-1)).toBeLessThan((curve?.validation[0] ?? 0) + 1e-9)
    const pa = passiveAggressiveRegressor({ maxIter: 50 }).fit(line, { key: key(5) })
    expect(score(pa, line)).toBeGreaterThan(0.95)
  })
})
