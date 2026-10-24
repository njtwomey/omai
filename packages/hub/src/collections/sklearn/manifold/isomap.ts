import { isomap as isomapMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** Isomap: classical scaling of geodesic distances along a neighbour graph. */
export const isomap: HubEntry = defineEntry({
  id: 'sklearn/manifold/isomap',
  title: 'Isomap',
  summary:
    'Classical scaling of geodesic distances along a neighbour graph: the Swiss roll unrolls into a strip, its colours in order.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'graphs'],
  datasets: ['manifold3d'],
  knobs: { n: 300 },
  ...fitEstimator(isomapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/isomap.ts', name: 'isomap' }],
})
