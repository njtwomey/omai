import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { moons } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { LabelSpreading, labelPropagation, labelSpreading } from './graph-based.ts'
import { hideLabels, UNLABELLED } from './labels.ts'

const { data } = generate(moons, { key: key(1), knobs: { n: 120, noise: 0.05 } })
const truth = Array.from(toFlat(data.y))
const y = hideLabels(data.y, { perClass: 3, key: key(2) })
const accuracy = (pred: readonly number[]): number => pred.filter((p, i) => p === truth[i]).length / truth.length

describe('graph-based semi-supervised learning', () => {
  it('hides all but three labels per class', () => {
    expect(y.filter((v) => v !== UNLABELLED).length).toBe(6)
  })

  for (const [name, make] of [
    ['propagation', () => labelPropagation({ kernel: 'knn', nNeighbors: 7 })],
    ['spreading', () => labelSpreading({ kernel: 'knn', nNeighbors: 7, maxIter: 200 })],
  ] as const) {
    it(`label ${name} labels the moons from six labels and records its steps`, () => {
      const f = make().fit({ ...data, y }, { key: key(0) })
      expect(accuracy(f.params.transduction)).toBeGreaterThan(0.95)
      expect(accuracy(Array.from(toFlat(f.predict(data.x))))).toBeGreaterThan(0.95)
      const [steps] = recordsOf(f.exposition, 'clusteringSteps')
      expect(steps?.assignments[0]?.filter((a) => a >= 0).length).toBe(6)
      expect(steps?.assignments.at(-1)?.every((a) => a >= 0)).toBe(true)
      expect(steps?.assignments.length).toBeLessThanOrEqual(60)
      expect(recordsOf(f.exposition, 'decisionSurface')[0]?.probabilities?.length).toBe(2)
    })
  }

  it('rbf spreading converges and the class form fits', () => {
    const m = new LabelSpreading({ gamma: 20 }).fit(data.x, y, { key: key(0) }).model
    expect(m.params.labelDistributions.length).toBe(120)
    for (const r of m.params.labelDistributions) expect(r.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10)
    expect(m.params.nIter).toBeLessThanOrEqual(30)
  })

  it('small-alpha spreading keeps going until the labels reach every connected row', () => {
    // At alpha 0.1 the total change drops below tol within a few steps, when only rows a few hops from a label hold
    // mass; scikit-learn stops there. The iteration runs on until no row gains its first label mass.
    const f = labelSpreading({ kernel: 'knn', nNeighbors: 7, alpha: 0.1, maxIter: 200 }).fit(
      { ...data, y },
      { key: key(0) },
    )
    const [steps] = recordsOf(f.exposition, 'clusteringSteps')
    expect(steps?.assignments.at(-1)?.every((a) => a >= 0)).toBe(true)
    expect(accuracy(f.params.transduction)).toBeGreaterThan(0.9)
    expect(f.params.nIter).toBeGreaterThan(5)
  })
})
