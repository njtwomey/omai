import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as M from './index.ts'

goldenSuite(
  'forecast metrics against torchmetrics, properscoring and the competition definitions',
  'metrics/forecasting',
  {
    symmetricMeanAbsolutePercentageError: ([y, p]) => M.symmetricMeanAbsolutePercentageError(y, p).estimate.value,
    weightedMeanAbsolutePercentageError: ([y, p]) => M.weightedMeanAbsolutePercentageError(y, p).estimate.value,
    meanAbsoluteScaledError: ([y, p, train], o) =>
      M.meanAbsoluteScaledError(y, p, { ...o, yTrain: train as Tensor }).estimate.value,
    rootMeanSquaredScaledError: ([y, p, train], o) =>
      M.rootMeanSquaredScaledError(y, p, { ...o, yTrain: train as Tensor }).estimate.value,
    intervalScore: ([lo, hi, y], o) => M.intervalScore(lo, hi, y as Tensor, o).estimate.value,
    meanScaledIntervalScore: ([lo, hi, y, train], o) =>
      M.meanScaledIntervalScore(lo, hi, y as Tensor, { ...o, yTrain: train as Tensor }).estimate.value,
    intervalCoverage: ([lo, hi, y]) => M.intervalCoverage(lo, hi, y as Tensor).estimate.value,
    quantileLoss: ([y, q], o) => M.quantileLoss(y, q, o as unknown as M.QuantileLossOptions).estimate.value,
    crpsEnsemble: ([y, e], o) => M.crpsEnsemble(y, e, o).estimate.value,
    crpsNormal: ([y, m, s]) => M.crpsNormal(y, m, s as Tensor).estimate.value,
    theilsU2: ([y, p]) => M.theilsU2(y, p).estimate.value,
  },
)
