import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { canonicalLabels } from './_support.ts'
import { dbscan, hdbscan, optics, opticsOrdering } from './density.ts'

const finite = (v: readonly number[]): number[] => v.map((x) => (Number.isFinite(x) ? x : -1))

goldenSuite('density clustering against scikit-learn', 'methods/cluster/density', {
  reachability: ([x], o) => tensor(finite(opticsOrdering(x, { minSamples: o['minSamples'] as number }).reachability)),
  core: ([x], o) => tensor(finite(opticsOrdering(x, { minSamples: o['minSamples'] as number }).coreDistances)),
  ordering: ([x], o) => tensor(opticsOrdering(x, { minSamples: o['minSamples'] as number }).ordering),
  opticsLabels: ([x], o) =>
    tensor(optics.with({ minSamples: o['minSamples'] as number, eps: 0.5 }).fit({ x }).params.labels),
  dbscan: ([x], o) =>
    tensor(dbscan.with({ eps: o['eps'] as number, minSamples: o['minSamples'] as number }).fit({ x }).params.labels),
  hdbscan: ([x], o) => tensor(canonicalLabels(hdbscan(x, { minClusterSize: o['minClusterSize'] as number }).labels)),
})
