import { describe, expect, it } from 'vitest'
import { asTensor, type TensorLike, tensor, toFlat } from '../../tensor/index.ts'
import { defineEstimator } from '../estimator.ts'
import { gaussianNaiveBayes } from '../naive-bayes.ts'
import { fixedThresholdClassifier, thresholdMetrics, tunedThresholdClassifierCV } from './threshold.ts'

const column = (x: TensorLike): number[] => {
  const t = asTensor(x)
  const d = t.shape[1] ?? 1
  const v = toFlat(t)
  return Array.from({ length: t.shape[0] ?? 0 }, (_, i) => v[i * d] ?? 0)
}

/** A classifier whose probability of class 1 is the (first) feature, or whose decision score is that minus a half. */
const identity = (withProba = true) =>
  defineEstimator(
    {
      name: 'identity',
      kind: 'classifier',
      fit: () => ({}),
      methods: () => ({
        classes: [0, 1],
        predict: (x: TensorLike) => tensor(column(x).map((v) => +(v >= 0.5))),
        ...(withProba
          ? { predictProba: (x: TensorLike) => tensor(column(x).map((v) => [1 - v, v])) }
          : { decisionFunction: (x: TensorLike) => tensor(column(x).map((v) => v - 0.5)) }),
      }),
    },
    {},
  )

const x = [[0.05], [0.1], [0.2], [0.3], [0.45], [0.5], [0.55], [0.6], [0.7], [0.8], [0.9], [0.95]]
const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

describe('fixedThresholdClassifier', () => {
  it('cuts the probabilities at the threshold, inclusively', () => {
    const f = fixedThresholdClassifier({ estimator: identity(), threshold: 0.3 }).fit({ x, y })
    expect(toFlat(f.predict([[0.2], [0.3], [0.8]]))).toEqual([0, 1, 1])
    expect(f.classes).toEqual([0, 1])
  })

  it("'auto' is 0.5 for probabilities and 0 for decision scores", () => {
    const p = fixedThresholdClassifier({ estimator: identity() }).fit({ x, y })
    expect(p.params.threshold).toBe(0.5)
    const d = fixedThresholdClassifier({ estimator: identity(false) }).fit({ x, y })
    expect(d.params.threshold).toBe(0)
    expect(toFlat(d.predict([[0.4], [0.6]]))).toEqual([0, 1])
  })

  it('a positive label of the first class flips the cut', () => {
    const f = fixedThresholdClassifier({ estimator: identity(), threshold: 0.7, posLabel: 0 }).fit({ x, y })
    // P(class 0) = 1 - x: at least 0.7 means x <= 0.3.
    expect(toFlat(f.predict([[0.2], [0.5]]))).toEqual([0, 1])
  })

  it('refuses more than two classes and revives from its parameters', () => {
    expect(() =>
      fixedThresholdClassifier({ estimator: identity() }).fit({ x: [[0], [1], [2]], y: [0, 1, 2] }),
    ).toThrow()
    const est = fixedThresholdClassifier({ estimator: identity(), threshold: 0.3 })
    expect(toFlat(est.revive(est.fit({ x, y }).params).predict([[0.35]]))).toEqual([1])
  })

  it('with() reaches the wrapped estimator by estimator.<name> keys and replaces the threshold', () => {
    const base = fixedThresholdClassifier({ estimator: gaussianNaiveBayes(), threshold: 0.5 })
    const changed = base.with({ threshold: 0.2, 'estimator.varSmoothing': 1e-3 })
    expect(changed.hyper.threshold).toBe(0.2)
    expect(changed.hyper.estimator.hyper).toMatchObject({ varSmoothing: 1e-3 })
    expect(base.hyper.threshold).toBe(0.5)
  })
})

describe('tunedThresholdClassifierCV', () => {
  it('finds a threshold that separates the classes perfectly when one exists', () => {
    const f = tunedThresholdClassifierCV({ estimator: identity(), cv: 2, scoring: 'accuracy', thresholds: 200 }).fit({
      x,
      y,
    })
    expect(f.params.bestScore).toBeCloseTo(1, 12)
    expect(f.params.threshold).toBeGreaterThan(0.45)
    expect(f.params.threshold).toBeLessThanOrEqual(0.5 + 1e-9)
    expect(toFlat(f.predict([[0.4], [0.52]]))).toEqual([0, 1])
    expect(f.params.curve?.thresholds.length).toBe(200)
  })

  it('maximises the chosen metric: recall is best at a threshold at or below every positive score', () => {
    const f = tunedThresholdClassifierCV({ estimator: identity(), cv: 2, scoring: 'recall', thresholds: 50 }).fit({
      x,
      y,
    })
    expect(f.params.bestScore).toBeCloseTo(1, 12)
  })

  it('takes a custom metric and explicit thresholds', () => {
    const calls: number[] = []
    const f = tunedThresholdClassifierCV({
      estimator: identity(),
      cv: 2,
      thresholds: [0.1, 0.5, 0.9],
      scoring: (t, p) => {
        calls.push(t.length)
        return thresholdMetrics['accuracy']?.(t, p, 1) ?? 0
      },
    }).fit({ x, y })
    expect(f.params.curve?.thresholds).toEqual([0.1, 0.5, 0.9])
    expect(f.params.threshold).toBe(0.5)
    expect(calls.length).toBe(6)
  })

  it('cuts decision scores when there are no probabilities, and rejects unknown scoring', () => {
    const f = tunedThresholdClassifierCV({ estimator: identity(false), cv: 2, scoring: 'accuracy' }).fit({ x, y })
    expect(f.params.responseMethod).toBe('decisionFunction')
    expect(toFlat(f.predict([[0.1], [0.9]]))).toEqual([0, 1])
    expect(() => tunedThresholdClassifierCV({ estimator: identity(), scoring: 'nope' }).fit({ x, y })).toThrow()
  })

  it('revives with the same cut', () => {
    const est = tunedThresholdClassifierCV({ estimator: identity(), cv: 2, scoring: 'f1' })
    const f = est.fit({ x, y })
    expect(toFlat(est.revive(f.params).predict(x))).toEqual(toFlat(f.predict(x)))
  })
})
