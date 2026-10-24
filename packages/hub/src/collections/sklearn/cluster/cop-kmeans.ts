import { copKMeans } from 'whitebox-engine/methods/cluster'
import { split, uniform } from 'whitebox-engine/random'
import { dict, int } from 'whitebox-engine/space'
import { asTensor, toFlat } from 'whitebox-engine/tensor'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { learnDataOf } from '../../../entry/run.ts'

/**
 * COP k-means: k-means that respects must-link and cannot-link constraints. The demo draws random pairs of rows and
 * links them by their true labels (must-link within a class, cannot-link across), as a user labelling a few pairs.
 */
export const copKmeans: HubEntry = defineEntry({
  id: 'sklearn/cluster/cop-kmeans',
  title: 'COP k-means',
  summary:
    'k-means that obeys must-link and cannot-link pairs: a few labelled pairs pull the stretched blobs apart where plain k-means cuts across them.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'centroids', 'constraints', 'semi-supervised'],
  datasets: ['clustering2d'],
  dataset: 'anisotropicBlobs',
  params: dict({
    nClusters: int(1, 20, { default: 3, label: 'clusters' }),
    constraints: int(0, 60, { default: 40, label: 'labelled pairs' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = (asTensor(learn.x).shape[0] ?? 0) as number
    const y = learn.y === undefined ? [] : toFlat(asTensor(learn.y))
    const count = y.length === 0 ? 0 : (params['constraints'] as number)
    const [pairKey, fitKey] = split(key) as [typeof key, typeof key]
    const u = count === 0 ? [] : toFlat(uniform(pairKey, [2 * count]))
    const mustLink: [number, number][] = []
    const cannotLink: [number, number][] = []
    for (let c = 0; c < count; c++) {
      const a = Math.floor((u[2 * c] ?? 0) * n)
      const b = Math.floor((u[2 * c + 1] ?? 0) * n)
      if (a === b) continue
      if (y[a] === y[b]) mustLink.push([a, b])
      else cannotLink.push([a, b])
    }
    const model = copKMeans({ nClusters: params['nClusters'] as number, mustLink, cannotLink })
    return model.fit({ x: learn.x }, { key: fitKey })
  },
  views: [{ record: 'clusteringSteps', title: 'Constrained Lloyd iterations' }],
  methods: [{ file: 'methods/cluster/constrained.ts', name: 'copKMeans' }],
})
