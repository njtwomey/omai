import { metricArgs, metricValue } from '../../testing/metrics.ts'
import { type CaseFn, goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as M from './index.ts'

/** A curve case: one array field of a curve computed from (yTrue, yScore[, weights]) and the options. */
const field =
  <C>(curve: (t: never, s: Tensor, o: Record<string, unknown>) => C, pick: (c: C) => readonly number[]): CaseFn =>
  (inputs, params) => {
    const [t, , o] = metricArgs(inputs, params)
    return tensor([...pick(curve(t as never, inputs[1], o))])
  }

goldenSuite('ROC and DET curves against sklearn', 'metrics/curves', {
  thresholdTps: field(M.confusionMatrixAtThresholds, (c) => c.tps),
  thresholdFps: field(M.confusionMatrixAtThresholds, (c) => c.fps),
  thresholdValues: field(M.confusionMatrixAtThresholds, (c) => c.thresholds),
  rocFpr: field(M.rocCurve, (c) => c.fpr),
  rocTpr: field(M.rocCurve, (c) => c.tpr),
  rocThresholds: field(M.rocCurve, (c) => c.thresholds),
  detFpr: field(M.detCurve, (c) => c.fpr),
  detFnr: field(M.detCurve, (c) => c.fnr),
  detThresholds: field(M.detCurve, (c) => c.thresholds),
  rocAucScore: metricValue(M.rocAucScore),
  auc: ([x, y]) => M.auc(toFlat(x), toFlat(y)),
})

goldenSuite('precision-recall and PR-gain curves against sklearn and the authors', 'metrics/precision_recall', {
  prPrecision: field(M.precisionRecallCurve, (c) => c.precision),
  prRecall: field(M.precisionRecallCurve, (c) => c.recall),
  prThresholds: field(M.precisionRecallCurve, (c) => c.thresholds),
  averagePrecisionScore: metricValue(M.averagePrecisionScore),
  prgRecallGain: field(M.prGainCurve, (c) => c.recallGain),
  prgPrecisionGain: field(M.prGainCurve, (c) => c.precisionGain),
  prGainAuc: metricValue(M.prGainAuc),
})

goldenSuite('operating points against torchmetrics and sklearn', 'metrics/operating_points', {
  recallAtFixedPrecision: metricValue(M.recallAtFixedPrecision as never),
  precisionAtFixedRecall: metricValue(M.precisionAtFixedRecall as never),
  sensitivityAtSpecificity: metricValue(M.sensitivityAtSpecificity as never),
  specificityAtSensitivity: metricValue(M.specificityAtSensitivity as never),
  fprAtTpr: metricValue(M.fprAtTpr as never),
  equalErrorRate: metricValue(M.equalErrorRate),
  youdenIndex: metricValue(M.youdenIndex),
  minimumExpectedCost: metricValue(M.minimumExpectedCost),
  costCurve: field(M.costCurve, (c) => c.normalizedExpectedCost),
})
