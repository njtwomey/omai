import { bayesianGaussianMixture as bayesianGaussianMixtureMethod } from 'whitebox-engine/methods/mixture'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Bayesian Gaussian mixture: a mixture with a prior on its weights that switches spare components off. */
export const bayesianGaussianMixture: HubEntry = defineEntry({
  id: 'sklearn/mixture/bayesian-gaussian-mixture',
  title: 'Bayesian Gaussian mixture',
  summary: 'A mixture with a prior on its weights that switches spare components off.',
  collection: 'sklearn/mixture',
  tags: ['clustering', 'density estimation', 'bayesian', 'variational'],
  datasets: ['clustering2d', 'density2d'],
  ...fitEstimator(bayesianGaussianMixtureMethod(), { unsupervised: true }),
  views: [{ record: 'mixtureSteps', title: 'EM steps' }],
  methods: [{ file: 'methods/mixture/bayesian.ts', name: 'bayesianGaussianMixture' }],
})
