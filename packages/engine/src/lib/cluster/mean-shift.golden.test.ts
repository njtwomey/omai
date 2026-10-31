import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { estimateBandwidth, meanShift } from './density.ts'

const fit = (x: Parameters<typeof estimateBandwidth>[0], o: Readonly<Record<string, unknown>>) =>
  meanShift.with({ bandwidth: o['bandwidth'] as number, maxIter: 300 }).fit({ x }).params

goldenSuite('mean shift against scikit-learn', 'cluster/mean_shift', {
  centres: ([x], o) => tensor(fit(x, o).centres),
  labels: ([x], o) => tensor(fit(x, o).labels),
  bandwidth: ([x]) => estimateBandwidth(x, 0.3),
})
