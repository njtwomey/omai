import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { copheneticDistances, type Linkage, linkageMatrix } from './hierarchical.ts'

goldenSuite('agglomerative linkage matrices against scipy', 'methods/cluster/hierarchical', {
  linkage: ([x], o) => tensor(linkageMatrix(x, o['method'] as Linkage)),
  cophenet: ([x]) => tensor(copheneticDistances(linkageMatrix(x, 'average'), x.shape[0] ?? 0)),
})
