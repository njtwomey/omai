import { goldenSuite } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import { affinityPropagationFit } from './exemplar.ts'

const fit = (x: Parameters<typeof affinityPropagationFit>[0], o: Readonly<Record<string, unknown>>) =>
  affinityPropagationFit(x, {
    noiseKey: key(0),
    ...(o['preference'] === undefined ? {} : { preference: o['preference'] as number }),
  })

goldenSuite('affinity propagation against scikit-learn', 'methods/cluster/exemplar', {
  centres: ([x], o) => tensor(fit(x, o).exemplars),
  labels: ([x], o) => tensor(fit(x, o).labels),
})
