/**
 * Hub entry for multidimensional scaling by SMACOF, placing points so their distances match the data's, on
 * three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { mds as mdsMethod } from 'mdml-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** Multidimensional scaling: place points so their distances match the data’s (SMACOF stress). */
export const mds: HubEntry = defineEntry({
  id: 'sklearn/manifold/mds',
  title: 'Multidimensional scaling',
  summary:
    'Place points so their distances match the data’s by SMACOF: the Swiss roll stays rolled, since straight-line distances cut across it.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding'],
  datasets: ['manifold3d', 'classificationTabular'],
  knobs: { n: 150 },
  ...fitEstimator(mdsMethod({ maxIter: 100 }), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/mds.ts', name: 'mds' }],
})
