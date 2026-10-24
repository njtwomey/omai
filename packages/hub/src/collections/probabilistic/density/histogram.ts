import { histogramDensity } from 'whitebox-engine/methods/mixture'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Histogram density: the oldest density estimator, with its bin width chosen by a rule. */
export const histogram: HubEntry = defineEntry({
  id: 'probabilistic/density/histogram',
  title: 'Histogram density',
  summary: 'The oldest density estimator, with its bin width chosen by a rule.',
  collection: 'probabilistic/density',
  tags: ['density estimation', 'nonparametric'],
  datasets: ['density2d', 'clustering2d'],
  ...fitEstimator(histogramDensity(), { unsupervised: true }),
  views: [{ record: 'decisionSurface', title: 'Density' }],
  methods: [{ file: 'methods/mixture/density.ts', name: 'histogramDensity' }],
})
