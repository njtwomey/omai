/**
 * Hub entry for Rosenblatt's perceptron as a neural network, one update per misclassified row until an epoch makes no
 * mistake, on two-dimensional classification data.
 */

import type { Exposition } from 'whitebox-engine/exposition'
import { rosenblattPerceptron } from 'whitebox-engine/methods/neural/foundations'
import { asTensor, toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator, learnDataOf } from '../../../entry/run.ts'

const method = rosenblattPerceptron({ maxEpochs: 50 })
const { params } = fitEstimator(method)

/** Rosenblatt's perceptron: one update per misclassified row, until an epoch makes no mistake. */
export const perceptron: HubEntry = defineEntry({
  id: 'neural/foundations/perceptron',
  title: 'Rosenblatt’s perceptron',
  summary: 'Each misclassified row nudges the weights towards it; on separable data it stops within (R/γ)² updates.',
  collection: 'neural/foundations',
  tags: ['classification', 'neural networks', 'linear', 'online learning'],
  datasets: ['classification2d'],
  dataset: 'anisotropicBlobs',
  params,
  run: (input) => {
    const learn = learnDataOf(input.data)
    // Two classes: the first class against the rest when the dataset has more.
    const y = toFlat(asTensor(learn.y ?? []))
    const two = new Set(y).size <= 2
    const data = two ? learn : { x: learn.x, y: y.map((v) => (v === 0 ? 0 : 1)), classes: ['0', 'the rest'] }
    const fitted = method.with(input.params).fit(data, { key: input.key })
    return { ...fitted, exposition: fitted.exposition as readonly Exposition[] }
  },
  views: [
    { record: 'decisionSurface', title: 'The separating line' },
    { record: 'learningCurve', title: 'Mistakes per epoch and updates so far', area: 'side' },
    { record: 'coefficientTable', title: 'Weights and Novikoff’s bound', area: 'side' },
  ],
  methods: [{ file: 'methods/neural/foundations/perceptron.ts', name: 'rosenblattPerceptron' }],
  references: [
    'Rosenblatt (1958), The perceptron: a probabilistic model for information storage and organization in the brain',
    'Novikoff (1962), On convergence proofs on perceptrons',
  ],
})
