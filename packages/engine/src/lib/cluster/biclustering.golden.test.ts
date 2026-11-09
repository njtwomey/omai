import { goldenSuite } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { canonicalLabels } from './_support.ts'
import { bistochasticNormalize, logNormalize, scaleNormalize } from './bicluster-normalise.ts'
import { consensusScore } from './bicluster-score.ts'
import { spectralBiclustering, spectralCoclustering } from './biclustering.ts'

const rows = (t: Tensor | undefined): number[][] => toArray(t as Tensor) as number[][]
const sets = (i: readonly (Tensor | undefined)[], at: number) => ({ rows: rows(i[at]), columns: rows(i[at + 1]) })

const coclustering = (i: readonly (Tensor | undefined)[], k: number) =>
  spectralCoclustering.with({ nClusters: k }).fit({ x: i[0] as Tensor }, { key: key(0) })

const biclustering = (i: readonly (Tensor | undefined)[], o: Readonly<Record<string, unknown>>) =>
  spectralBiclustering
    .with({ nClusters: [o['kr'] as number, o['kc'] as number], method: 'log', nComponents: 6, nBest: 3 })
    .fit({ x: i[0] as Tensor }, { key: key(0) })

const labels = (l: number[]): Tensor => tensor(canonicalLabels(l))

goldenSuite('spectral biclustering against scikit-learn', 'cluster/biclustering', {
  scale: (i) => tensor(scaleNormalize(rows(i[0])).matrix),
  bistochastic: (i) => tensor(bistochasticNormalize(rows(i[0]))),
  log: (i) => tensor(logNormalize(rows(i[0]))),
  consensus: (i) => consensusScore(sets(i, 0), sets(i, 2)),
  coclusteringRows: (i, o) => labels(coclustering(i, o['k'] as number).params.rowLabels),
  coclusteringColumns: (i, o) => labels(coclustering(i, o['k'] as number).params.columnLabels),
  coclusteringConsensus: (i, o) => consensusScore(coclustering(i, o['k'] as number).params, sets(i, 1)),
  biclusteringRows: (i, o) => labels(biclustering(i, o).params.rowLabels),
  biclusteringColumns: (i, o) => labels(biclustering(i, o).params.columnLabels),
})
