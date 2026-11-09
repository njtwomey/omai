import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { pipeline } from '../compose.ts'
import { naiveBayes } from '../naive-bayes.ts'
import { chainResamplers, smoteEnn, smoteTomek } from './combined.ts'
import { kMeansSmote, smoten, smotenc, svmSmote, valueDifferenceMetric } from './oversampling.ts'
import { randomUnderSampler, smote } from './resampling.ts'
import {
  allKnn,
  clusterCentroids,
  condensedNearestNeighbour,
  editedNearestNeighbours,
  instanceHardnessThreshold,
  neighbourhoodCleaningRule,
  oneSidedSelection,
  repeatedEditedNearestNeighbours,
} from './undersampling.ts'

// A line of majority samples with a few minority samples among them.
const line = {
  x: [[0], [0.1], [0.2], [0.3], [0.45], [0.5], [0.6], [1], [1.1], [1.2]],
  y: [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
}

describe('edited nearest neighbours and its relatives', () => {
  it('edit the majority class where the neighbours disagree, by the rule asked for', () => {
    const all = editedNearestNeighbours({ nNeighbors: 2 }).fitResample(line)
    // Class 0 is edited: 0.3 and 0.5 sit beside minority samples, 1.1 and 1.2 between them and the far one.
    expect(Array.from(all.indices)).toEqual([0, 1, 2, 4, 6, 7])
    const mode = editedNearestNeighbours({ nNeighbors: 2, kindSel: 'mode' }).fitResample(line)
    // A tie goes to the smaller label, which is the sample's own class here: only 0.5, between two minority samples, goes.
    expect(Array.from(mode.indices)).toEqual([0, 1, 2, 3, 8, 9, 4, 6, 7])
    expect(toFlat(all.y)).toEqual(Array.from(all.indices, (i) => line.y[i]))
  })

  it('repeat and grow the neighbourhood: nothing is kept that a single pass removes', () => {
    const one = new Set(editedNearestNeighbours({ nNeighbors: 2 }).fitResample(line).indices)
    for (const sampler of [repeatedEditedNearestNeighbours({ nNeighbors: 2 }), allKnn({ nNeighbors: 2 })]) {
      const r = sampler.fitResample(line)
      expect(Array.from(r.indices).every((i) => one.has(i))).toBe(true)
    }
  })

  it('the neighbourhood cleaning rule also removes neighbours of a minority sample its own neighbours misclassify', () => {
    const r = neighbourhoodCleaningRule({ nNeighbors: 2 }).fitResample(line)
    const kept = Array.from(r.indices)
    expect(kept).toContain(4)
    expect(kept).toContain(6)
    expect(kept.length).toBeLessThan(10)
  })

  it('accept a record of the classes to edit', () => {
    const r = editedNearestNeighbours({ nNeighbors: 2, samplingStrategy: { 1: 0 } }).fitResample(line)
    // Only class 1 is edited: its sample at 0.45 has majority neighbours and goes.
    expect(Array.from(r.indices)).not.toContain(4)
    expect(Array.from(r.indices)).toContain(3)
  })
})

describe('condensing and hardness', () => {
  const data = {
    x: [[0], [0.1], [0.2], [0.3], [0.4], [0.5], [2], [2.1]],
    y: [0, 0, 0, 0, 0, 0, 1, 1],
  }

  it('condensed nearest neighbour keeps a store that classifies the class, starting from the seeds', () => {
    const r = condensedNearestNeighbour({ seeds: { 0: [0] } }).fitResample(data)
    expect(Array.from(r.indices)).toEqual([0, 6, 7])
    expect(() => condensedNearestNeighbour({ seeds: { 0: [99] } }).fitResample(data)).toThrow(/outside the class/)
    expect(condensedNearestNeighbour().fitResample(data, { key: key(3) }).y.shape[0] ?? 0).toBeLessThan(8)
  })

  it('one-sided selection adds what a one-nearest-neighbour store gets wrong, then drops Tomek links', () => {
    const r = oneSidedSelection({ seeds: { 0: [0] } }).fitResample(data)
    expect(Array.from(r.indices).slice(-2)).toEqual([6, 7])
    expect(Array.from(r.indices)).toContain(0)
  })

  it('instance hardness keeps the samples a model finds easiest, as many as the target', () => {
    const x = Array.from({ length: 30 }, (_, i) => [i < 24 ? i / 8 : 1.4 + (i - 24) / 10])
    const y = x.map((_, i) => (i < 24 ? 0 : 1))
    const r = instanceHardnessThreshold({ estimator: naiveBayes() }).fitResample({ x, y }, { key: key(0) })
    expect(toFlat(r.y).filter((v) => v === 0).length).toBeGreaterThanOrEqual(6)
    expect(toFlat(r.y).filter((v) => v === 1).length).toBe(6)
  })

  it('cluster centroids honour counts for named classes', () => {
    const r = clusterCentroids({ samplingStrategy: { 0: 3 } }).fitResample(data, { key: key(1) })
    expect(toFlat(r.y).filter((v) => v === 0).length).toBe(3)
    expect(toFlat(r.y).filter((v) => v === 1).length).toBe(2)
  })
})

describe('SMOTE variants', () => {
  it('SMOTEN takes the most common category among the neighbours', () => {
    // Class 1 is (0, 0) twice and (1, 0) once: every neighbourhood's mode in feature 1 is 0.
    const x = [
      [2, 1],
      [2, 1],
      [2, 2],
      [2, 2],
      [2, 1],
      [2, 2],
      [0, 0],
      [0, 0],
      [1, 0],
    ]
    const r = smoten({ kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1, 1, 1] }, { key: key(0) })
    const made = toFlat(r.x).slice(18)
    expect(made).toHaveLength(3 * 2)
    expect(made.filter((_, i) => i % 2 === 1).every((v) => v === 0)).toBe(true)
    expect(made.filter((_, i) => i % 2 === 0).every((v) => v === 0 || v === 1)).toBe(true)
  })

  it('SMOTEN and SMOTE-NC refuse a class too small for the neighbours', () => {
    const x = [
      [0, 1],
      [0, 1],
      [0, 1],
      [1, 0],
    ]
    expect(() => smoten({ kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 1] })).toThrow(/more than 2 samples/)
    expect(() => smotenc({ categorical: [0], kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 1] })).toThrow(
      /more than 2/,
    )
  })

  it('the value difference metric rejects a category it has not seen', () => {
    const vdm = valueDifferenceMetric([[0], [1]], [0, 1])
    expect(() => vdm([[5]])).toThrow(/not in the training data/)
  })

  it('SVM-SMOTE needs a support vector machine, and k-means SMOTE a cluster holding enough of the class', () => {
    expect(() => svmSmote({ svm: naiveBayes() }).fitResample(line, { key: key(0) })).toThrow(/supportIndex/)
    expect(() => kMeansSmote({ clusterBalanceThreshold: 2 }).fitResample(line, { key: key(0) })).toThrow(/no cluster/)
  })

  it('SMOTE + ENN and SMOTE + Tomek follow the over-sampler they are given', () => {
    const x = [[0], [0.1], [0.2], [0.3], [0.4], [0.5], [3], [3.1], [3.2], [3.3]]
    const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    for (const sampler of [
      smoteEnn({ smote: smote({ kNeighbors: 2 }) }),
      smoteTomek({ smote: smote({ kNeighbors: 2 }) }),
    ]) {
      const r = sampler.fitResample({ x, y }, { key: key(2) })
      // The clouds are far apart, so nothing is removed: all 12 samples remain, 2 of them synthetic.
      expect(r.y.shape[0]).toBe(12)
      expect(Array.from(r.indices).filter((i) => i === -1)).toHaveLength(2)
    }
    expect(() => chainResamplers('none', [])).toThrow()
  })
})

describe('in a pipeline', () => {
  it('an edited-neighbours resampler cleans the training rows and is skipped when predicting', () => {
    const x = Array.from({ length: 40 }, (_, i) => [i < 30 ? i / 30 : 1.2 + (i - 30) / 20])
    const y = x.map((_, i) => (i < 30 ? 0 : 1))
    const model = pipeline([
      ['clean', smoteEnn({ smote: smote({ kNeighbors: 3 }) })],
      ['clf', naiveBayes()],
    ]).fit({ x, y }, { key: key(0) })
    expect(toFlat(model.predict([[0.1], [1.4]]))).toEqual([0, 1])
    expect(toFlat(randomUnderSampler().fitResample({ x, y }, { key: key(0) }).y).length).toBe(20)
  })
})
