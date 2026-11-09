import { learnGolden, num, str } from '../../testing/learn.ts'
import { metricArgs, metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
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

learnGolden('class_likelihood_ratios against sklearn, defined and ill-defined', 'metrics/class_likelihood_ratios', {
  ratios: ([yTrue, yPred, w], p) => {
    const replace = str(p, 'replace')
    const { positive, negative } = M.classLikelihoodRatios(yTrue as Tensor, yPred as Tensor, {
      ...(p['weighted'] === true ? { sampleWeight: w as Tensor } : {}),
      replaceUndefinedBy:
        replace === 'nan'
          ? Number.NaN
          : replace === 'one'
            ? 1
            : { positive: num(p, 'pairPositive'), negative: num(p, 'pairNegative') },
    })
    return { positive: positive.estimate.value, negative: negative.estimate.value }
  },
})
