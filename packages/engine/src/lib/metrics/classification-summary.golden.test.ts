import { metricArgs, metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

goldenSuite('agreement, ranking and report metrics against sklearn', 'metrics/classification_summary', {
  matthewsCorrcoef: metricValue(M.matthewsCorrcoef),
  cohenKappaScore: metricValue(M.cohenKappaScore),
  quadraticWeightedKappa: metricValue(M.quadraticWeightedKappa),
  positiveLikelihoodRatio: metricValue(M.positiveLikelihoodRatio),
  negativeLikelihoodRatio: metricValue(M.negativeLikelihoodRatio),
  diagnosticOddsRatio: metricValue(M.diagnosticOddsRatio),
  hammingLoss: metricValue(M.hammingLoss),
  zeroOneLoss: metricValue(M.zeroOneLoss),
  topKAccuracyScore: metricValue(M.topKAccuracyScore),
  labelRankingAveragePrecisionScore: metricValue(M.labelRankingAveragePrecisionScore as never),
  coverageError: metricValue(M.coverageError as never),
  labelRankingLoss: metricValue(M.labelRankingLoss as never),
  classificationReport: (inputs, params) => {
    const [t, p, o] = metricArgs(inputs, params)
    return M.classificationReport(t, p, { zeroDivision: 0, ...o }).text
  },
})
