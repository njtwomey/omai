import { metricArgs, metricValue } from '../../testing/metrics.ts'
import { type CaseFn, goldenSuite } from '../../testing/suite.ts'
import { exponential, gamma, laplace, logistic, normal, uniform } from '../dists/index.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as M from './index.ts'

const curve =
  (pick: (c: M.CalibrationCurve) => readonly number[]): CaseFn =>
  (inputs, params) => {
    const [t, , o] = metricArgs(inputs, params)
    return tensor([...pick(M.calibrationCurve(t, inputs[1], o))])
  }
const part =
  (key: string): CaseFn =>
  (inputs, params) => {
    const [t, p, o] = metricArgs(inputs, params)
    return M.brierDecomposition(t, p, o).breakdown?.[key] ?? Number.NaN
  }

goldenSuite('probabilistic classification metrics against sklearn and torchmetrics', 'metrics/probabilistic', {
  logLoss: metricValue(M.logLoss),
  d2LogLossScore: metricValue(M.d2LogLossScore),
  brierScoreLoss: metricValue(M.brierScoreLoss),
  calibrationProbTrue: curve((c) => c.probTrue),
  calibrationProbPred: curve((c) => c.probPred),
  brierReliability: part('reliability'),
  brierResolution: part('resolution'),
  brierUncertainty: part('uncertainty'),
  expectedCalibrationError: metricValue(M.expectedCalibrationError),
  maximumCalibrationError: metricValue(M.maximumCalibrationError),
  classwiseCalibrationError: metricValue(M.classwiseCalibrationError),
})

goldenSuite('proper scoring rules against properscoring and scipy', 'metrics/scoring_rules', {
  logScoreNormal: ([m, s, y]) => M.logScore(normal(m, s), y as Tensor).estimate.value,
  crpsNormal: ([m, s, y]) => M.crps(normal(m, s), y as Tensor).estimate.value,
  crpsLaplace: ([m, s, y]) => M.crps(laplace(m, s), y as Tensor).estimate.value,
  crpsLogistic: ([m, s, y]) => M.crps(logistic(m, s), y as Tensor).estimate.value,
  crpsExponential: ([r, y]) => M.crps(exponential(r), y).estimate.value,
  crpsUniform: ([a, b, y]) => M.crps(uniform(a, b), y as Tensor).estimate.value,
  crpsGamma: ([k, y]) => M.crps(gamma(k, 1), y).estimate.value,
  sphericalScoreNormal: ([m, s, y]) => M.sphericalScore(normal(m, s), y as Tensor).estimate.value,
  sphericalScoreGamma: ([k, y]) => M.sphericalScore(gamma(k, 1), y).estimate.value,
  energyScore: ([x, y]) => M.energyScore(x, y).estimate.value,
})
