import { metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as M from './index.ts'

goldenSuite('external clustering metrics against sklearn', 'metrics/clustering', {
  contingencyMatrix: ([t, p]) => tensor(M.contingencyMatrix(t, p).matrix as number[][]),
  pairConfusionMatrix: ([t, p]) => tensor(M.pairConfusionMatrix(t, p)),
  randScore: metricValue(M.randScore),
  adjustedRandScore: metricValue(M.adjustedRandScore),
  fowlkesMallowsScore: metricValue(M.fowlkesMallowsScore),
  mutualInfoScore: metricValue(M.mutualInfoScore),
  normalizedMutualInfoScore: metricValue(M.normalizedMutualInfoScore),
  adjustedMutualInfoScore: metricValue(M.adjustedMutualInfoScore),
  homogeneityScore: metricValue(M.homogeneityScore),
  completenessScore: metricValue(M.completenessScore),
  vMeasureScore: metricValue(M.vMeasureScore),
  purityScore: metricValue(M.purityScore),
  bCubedF1: metricValue(M.bCubedF1),
  bCubedPrecision: ([t, p]) => M.bCubed(t, p).precision,
  bCubedRecall: ([t, p]) => M.bCubed(t, p).recall,
})

goldenSuite('internal cluster-validity indices against sklearn', 'metrics/cluster_validity', {
  silhouetteScore: ([x, l], p) => M.silhouetteScore(x, l, p).estimate.value,
  silhouetteSamples: ([x, l], p) => tensor(M.silhouetteSamples(x, l, p)),
  calinskiHarabaszScore: ([x, l]) => M.calinskiHarabaszScore(x, l).estimate.value,
  daviesBouldinScore: ([x, l]) => M.daviesBouldinScore(x, l).estimate.value,
  dunnIndex: ([x, l], p) => M.dunnIndex(x, l, p).estimate.value,
})
