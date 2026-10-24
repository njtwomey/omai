import { metricArgs, metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as M from './index.ts'

goldenSuite('classification metrics against sklearn', 'metrics/classification', {
  precisionScore: metricValue(M.precisionScore),
  recallScore: metricValue(M.recallScore),
  f1Score: metricValue(M.f1Score),
  fbetaScore: metricValue(M.fbetaScore),
  jaccardScore: metricValue(M.jaccardScore),
  accuracyScore: metricValue(M.accuracyScore),
  balancedAccuracyScore: metricValue(M.balancedAccuracyScore),
  specificityScore: metricValue(M.specificityScore),
  negativePredictiveValue: metricValue(M.negativePredictiveValue),
  confusionMatrix: (inputs, params) => {
    const [t, p, o] = metricArgs(inputs, params)
    return tensor(M.confusionMatrix(t, p, o).matrix as number[][])
  },
  multilabelConfusionMatrix: (inputs, params) => {
    const [t, p, o] = metricArgs(inputs, params)
    return tensor(M.multilabelConfusionMatrix(t, p, o))
  },
})
