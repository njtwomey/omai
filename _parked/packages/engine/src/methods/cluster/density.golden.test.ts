import { tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { dbscan, hdbscan, optics, opticsOrdering } from './density.ts'

const canonical = (labels: readonly number[]): number[] => {
  const seen = new Map<number, number>()
  return labels.map((l) => {
    if (l < 0) return -1
    if (!seen.has(l)) seen.set(l, seen.size)
    return seen.get(l) ?? 0
  })
}
const finite = (v: readonly number[]): number[] => v.map((x) => (Number.isFinite(x) ? x : -1))

goldenSuite('density clustering against scikit-learn', 'methods/cluster/density', {
  reachability: ([x], o) => tensor(finite(opticsOrdering(x, o['minSamples'] as number).reachability)),
  core: ([x], o) => tensor(finite(opticsOrdering(x, o['minSamples'] as number).coreDistances)),
  ordering: ([x], o) => tensor(opticsOrdering(x, o['minSamples'] as number).ordering),
  opticsLabels: ([x], o) =>
    tensor(optics({ minSamples: o['minSamples'] as number, eps: 0.5 }).fit({ x }).params.labels),
  dbscan: ([x], o) =>
    tensor(dbscan({ eps: o['eps'] as number, minSamples: o['minSamples'] as number }).fit({ x }).params.labels),
  hdbscan: ([x], o) =>
    tensor(canonical(hdbscan({ minClusterSize: o['minClusterSize'] as number }).fit({ x }).params.labels)),
})
