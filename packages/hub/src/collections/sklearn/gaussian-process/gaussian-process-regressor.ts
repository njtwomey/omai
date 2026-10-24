import { gaussianProcessRegressor as gaussianProcessRegressorMethod } from 'whitebox-engine/methods/gp'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gaussian process regression: a distribution over functions, conditioned on the data, with kernel hyperparameters fitted by marginal likelihood. */
export const gaussianProcessRegressor: HubEntry = defineEntry({
  id: 'sklearn/gaussian-process/gaussian-process-regressor',
  title: 'Gaussian process regression',
  summary:
    'A distribution over functions conditioned on the data: the band widens between and beyond the points, and the kernel is fitted by marginal likelihood.',
  collection: 'sklearn/gaussian-process',
  tags: ['regression', 'bayesian', 'kernels', 'nonparametric'],
  datasets: ['regression1d'],
  knobs: { n: 25 },
  ...fitEstimator(gaussianProcessRegressorMethod({ alpha: 0.01, nRestarts: 1 }), { surface: true }),
  views: [
    { record: 'posteriorGrid', title: 'Posterior' },
    { record: 'hyperparameterSurface', title: 'Marginal likelihood', area: 'side' },
    { record: 'coefficientTable', title: 'Hyperparameters', area: 'side' },
  ],
  methods: [{ file: 'methods/gp/regression.ts', name: 'gaussianProcessRegressor' }],
})
