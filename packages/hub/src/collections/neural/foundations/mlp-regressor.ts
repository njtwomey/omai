import { mlpRegressor as mlpRegressorMethod } from 'whitebox-engine/methods/neural/foundations'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { mlpArchitecture, trainingRun } from '../_train.ts'

/** A multi-layer perceptron regressor fitting a curve, drawn as it trains. */
export const mlpRegressor: HubEntry = defineEntry({
  id: 'neural/foundations/mlp-regressor',
  title: 'Multi-layer perceptron regressor',
  summary: 'Two hidden layers of tanh units fit a one-dimensional curve by Adam on minibatches, epoch by epoch.',
  collection: 'neural/foundations',
  tags: ['regression', 'neural networks', 'universal approximation'],
  datasets: ['regression1d', 'regressionTabular'],
  ...trainingRun(
    mlpRegressorMethod({
      hiddenLayerSizes: [32, 32],
      activation: 'tanh',
      learningRateInit: 0.02,
      batchSize: 25,
      maxIter: 150,
      tol: 1e-5,
      nIterNoChange: 40,
    }),
  ),
  views: [
    { record: 'decisionSurface', title: 'The fitted curve' },
    { record: 'learningCurve', title: 'Loss per epoch', area: 'side' },
    { record: 'learningCurve', index: 1, title: 'R² per epoch', area: 'side' },
    { record: 'embeddingSteps', title: 'The last hidden layer’s view of the data', area: 'below' },
    mlpArchitecture(),
  ],
  methods: [{ file: 'methods/neural/foundations/mlp.ts', name: 'mlpRegressor' }],
})
