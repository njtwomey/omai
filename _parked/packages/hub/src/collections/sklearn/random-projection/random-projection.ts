/**
 * Hub entry for random projection onto random directions, which roughly keeps distances (the Johnson-Lindenstrauss
 * lemma), on three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { randomProjection as randomProjectionMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Random projection: project onto random directions: distances survive (Johnson–Lindenstrauss). */
export const randomProjection: HubEntry = defineEntry({
  id: 'sklearn/random-projection/random-projection',
  title: 'Random projection',
  summary: 'Project onto random directions: distances survive (Johnson–Lindenstrauss).',
  collection: 'sklearn/random-projection',
  tags: ['decomposition', 'randomised'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(randomProjectionMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/random-projection.ts', name: 'randomProjection' }],
})
