import { mlpClassifier as mlpClassifierMethod } from 'whitebox-engine/methods/neural/foundations'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { mlpArchitecture, trainingRun } from '../_train.ts'

/** A multi-layer perceptron classifier, its decision surface drawn as it trains. */
export const mlpClassifier: HubEntry = defineEntry({
  id: 'neural/foundations/mlp-classifier',
  title: 'Multi-layer perceptron classifier',
  summary: 'Hidden layers of tanh units trained by Adam: watch the decision boundary bend to the data epoch by epoch.',
  collection: 'neural/foundations',
  tags: ['classification', 'neural networks', 'backpropagation', 'adam'],
  datasets: ['classification2d', 'classificationTabular'],
  ...trainingRun(
    mlpClassifierMethod({
      hiddenLayerSizes: [16, 16],
      activation: 'tanh',
      learningRateInit: 0.03,
      maxIter: 300,
      nIterNoChange: 30,
    }),
  ),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'embeddingSteps', title: 'The last hidden layer’s view of the data, epoch by epoch', area: 'side' },
    { record: 'decisionSurface', index: 1, title: 'Each first-layer unit over the input plane', area: 'side' },
    { record: 'learningCurve', title: 'Loss per epoch', area: 'below' },
    { record: 'learningCurve', index: 1, title: 'Accuracy per epoch', area: 'below' },
    { record: 'confusionMatrix', title: 'True against predicted class, epoch by epoch', area: 'below' },
    mlpArchitecture(),
  ],
  methods: [{ file: 'methods/neural/foundations/mlp.ts', name: 'mlpClassifier' }],
  references: ['Rumelhart, Hinton and Williams (1986), Learning representations by back-propagating errors'],
})
