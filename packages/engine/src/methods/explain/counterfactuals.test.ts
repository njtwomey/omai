import { describe, expect, it } from 'vitest'
import { breastCancer, digits } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { matrixView, takeColumns, takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { randomForestClassifier } from '../ensemble/index.ts'
import { logisticRegression } from '../linear/index.ts'
import {
  type CounterfactualMethod,
  explainCounterfactuals,
  prototypesAndCriticisms,
  trustScoreStudy,
} from './counterfactuals.ts'

const full = takeData(
  breastCancer(),
  Array.from({ length: 160 }, (_, i) => i),
)
const five = takeColumns(matrixView(full.x, 'test'), [0, 1, 2, 3, 4])
const data = { ...full, x: fromStorage(five.data, [five.n, 5], 'float64'), featureNames: full.featureNames.slice(0, 5) }
const logistic = logisticRegression({ C: 1 }).fit(data)

describe('counterfactual methods', () => {
  it('finds counterfactuals by every method on a logistic model', () => {
    const methods: CounterfactualMethod[] = ['growing spheres', 'face', 'wachter', 'dice', 'prototype', 'cem']
    for (const method of methods) {
      const r = explainCounterfactuals(logistic, data, 3, { method, count: 2, key: key(1) })
      const [rec] = recordsOf(r.exposition, 'counterfactuals')
      expect(rec?.original.length).toBe(5)
      expect(rec?.counterfactuals.length ?? 0).toBeGreaterThan(0)
    }
  })

  it('model-agnostic methods work on a forest, with constraints and refits', () => {
    const forest = randomForestClassifier({ nEstimators: 5, maxDepth: 3 }).fit(data, { key: key(2) })
    const r = explainCounterfactuals(forest, data, 3, {
      constraints: { immutable: [0] },
      estimator: randomForestClassifier({ nEstimators: 3, maxDepth: 3 }),
      refits: 2,
      key: key(3),
    })
    const c = recordsOf(r.exposition, 'counterfactuals')[0]?.counterfactuals[0]
    expect(c?.changed.includes(0)).toBe(false)
    expect(c?.robustness).toBeDefined()
    expect(() => explainCounterfactuals(forest, data, 3, { method: 'wachter' })).toThrow(/linear/)
  })

  it('prototypes and criticisms of tabular rows and of images, and trust scores against confidence', () => {
    const p = prototypesAndCriticisms(data, { nPrototypes: 3, nCriticisms: 2, maxRows: 80 })
    expect(recordsOf(p.exposition, 'prototypes')[0]?.points?.length).toBe(80)
    const images = takeData(
      digits(),
      Array.from({ length: 60 }, (_, i) => i),
    )
    const q = prototypesAndCriticisms(images, { nPrototypes: 4, nCriticisms: 2, imageShape: [8, 8] })
    expect(recordsOf(q.exposition, 'prototypes')[0]?.prototypeData[0]?.length).toBe(64)
    const t = trustScoreStudy(logistic, data, data, { alpha: 0.1, kFilter: 5 })
    expect(recordsOf(t.exposition, 'curves')[0]?.series.length).toBe(2)
  })
})
