import { tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { affinityPropagation } from './exemplar.ts'

const fit = (
  x: Parameters<ReturnType<typeof affinityPropagation>['fit']>[0]['x'],
  o: Readonly<Record<string, unknown>>,
) =>
  affinityPropagation(o['preference'] === undefined ? {} : { preference: o['preference'] as number }).fit({ x }).params

goldenSuite('affinity propagation against scikit-learn', 'methods/cluster/exemplar', {
  centres: ([x], o) => tensor(fit(x, o).exemplarIndices),
  labels: ([x], o) => tensor(fit(x, o).labels),
})
