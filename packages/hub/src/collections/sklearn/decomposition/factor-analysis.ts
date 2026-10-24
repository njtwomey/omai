import { factorAnalysis as factorAnalysisMethod } from 'whitebox-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Factor analysis: a few latent factors plus independent noise per feature. */
export const factorAnalysis: HubEntry = defineEntry({
  id: 'sklearn/decomposition/factor-analysis',
  title: 'Factor analysis',
  summary: 'A few latent factors plus independent noise per feature.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'probabilistic', 'latent variables'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(factorAnalysisMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/probabilistic.ts', name: 'factorAnalysis' }],
})
