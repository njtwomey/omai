import { metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

const run = (m: unknown): ReturnType<typeof metricValue> => metricValue(m as never)

goldenSuite('ranking metrics against sklearn, torchmetrics and trec_eval definitions', 'metrics/ranking', {
  dcgScore: run(M.dcgScore),
  ndcgScore: run(M.ndcgScore),
  precisionAtK: run(M.precisionAtK),
  recallAtK: run(M.recallAtK),
  hitRate: run(M.hitRate),
  rPrecision: run(M.rPrecision),
  meanAveragePrecision: run(M.meanAveragePrecision),
  meanReciprocalRank: run(M.meanReciprocalRank),
  expectedReciprocalRank: run(M.expectedReciprocalRank),
  rankBiasedPrecision: run(M.rankBiasedPrecision),
})
