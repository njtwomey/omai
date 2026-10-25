import { describe, expect, it } from 'vitest'
import { breastCancer, diabetes, digits } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { matrixView, takeColumns, takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { randomForestClassifier } from '../ensemble/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { lenet } from '../neural/vision/index.ts'
import { tcavStudy } from './concepts.ts'
import { anchorExplanation, explainableBoostingMachine, logisticScorecard, ruleFitExplanation } from './glassbox.ts'
import { mislabelledRowStudy } from './valuation.ts'

const full = takeData(
  breastCancer(),
  Array.from({ length: 150 }, (_, i) => i),
)
const four = takeColumns(matrixView(full.x, 'test'), [0, 1, 2, 3])
const cancer = {
  ...full,
  x: fromStorage(four.data, [four.n, 4], 'float64'),
  featureNames: full.featureNames.slice(0, 4),
}

describe('glass-box, rule, concept and valuation methods', () => {
  it('fits an EBM with a pair and reads its shape functions and importances', () => {
    const r = explainableBoostingMachine(cancer, { rounds: 20, interactions: 1, bins: 8 })
    const [dep] = recordsOf(r.exposition, 'dependence')
    expect(dep?.method).toBe('shape function')
    expect(dep?.curves.length).toBe(4)
    expect(dep?.surfaces?.length).toBe(1)
    expect(recordsOf(r.exposition, 'importances')[0]?.features.length).toBe(5)
    expect(r.predictions.every((p) => p > 0 && p < 1)).toBe(true)
  })

  it('a scorecard, RuleFit and an anchor as rule records', () => {
    const logistic = logisticRegression({ C: 1 }).fit(cancer)
    const card = logisticScorecard(logistic, cancer, { bins: 3 })
    expect(recordsOf(card.exposition, 'rules')[0]?.rules.length).toBe(12)
    const reg = takeData(
      diabetes(),
      Array.from({ length: 120 }, (_, i) => i),
    )
    const rf = ruleFitExplanation(reg, { nTrees: 4, key: key(1) })
    expect(recordsOf(rf.exposition, 'rules')[0]?.method).toBe('RuleFit')
    const forest = randomForestClassifier({ nEstimators: 5, maxDepth: 3 }).fit(cancer, { key: key(2) })
    const a = anchorExplanation(forest, cancer, 4, { key: key(3) })
    expect(recordsOf(a.exposition, 'rules')[0]?.rules[0]?.precision).toBeDefined()
  })

  it('TCAV on a LeNet and a mislabelled-row study', () => {
    const images = takeData(
      digits(),
      Array.from({ length: 120 }, (_, i) => i),
    )
    const net = lenet({ epochs: 1, channels: [3, 4], hidden: 8, snapshots: 2 }).fit(images, {
      key: key(4),
      explain: false,
    })
    const zeros = Array.from({ length: 120 }, (_, i) => i).filter((i) => i % 10 === 0)
    const r = tcavStudy(net, images, [{ name: 'zeros', rows: zeros }], { nRandom: 3, randomSize: 10, key: key(5) })
    expect(recordsOf(r.exposition, 'concepts')[0]?.concepts[0]?.scores.length).toBe(3)
    const m = mislabelledRowStudy(logisticRegression({ C: 1 }), cancer, {
      trainRows: 40,
      testRows: 30,
      nSamples: 8,
      nPermutations: 2,
      key: key(6),
    })
    expect(recordsOf(m.exposition, 'curves')[0]?.series.length).toBe(4)
    expect(m.flipped.length).toBe(4)
  })
})
