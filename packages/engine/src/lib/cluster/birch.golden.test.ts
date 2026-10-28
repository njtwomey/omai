import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { canonicalLabels } from './_support.ts'
import { birch, birchAlgorithm, birchLeaves } from './hierarchical.ts'

const tree = (x: Parameters<typeof birchAlgorithm>[0], o: Readonly<Record<string, unknown>>) =>
  runAlgorithm(
    birchAlgorithm(x, { threshold: o['threshold'] as number, branchingFactor: o['branchingFactor'] as number }),
    { maxSteps: 1000 },
  ).state

goldenSuite('BIRCH against scikit-learn', 'cluster/birch', {
  subclusters: ([x], o) => tensor(birchLeaves(tree(x, o)).map((s) => s.ls.map((v) => v / s.n))),
  labels: ([x], o) =>
    tensor(
      canonicalLabels(
        birch
          .with({ threshold: o['threshold'] as number, branchingFactor: o['branchingFactor'] as number, nClusters: 3 })
          .fit({ x }).params.labels,
      ),
    ),
})
