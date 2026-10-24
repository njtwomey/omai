import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as M from './index.ts'

const interval =
  (method: 'wilson' | 'analytic', end: 0 | 1) =>
  (inputs: readonly [Tensor, Tensor, ...(Tensor | undefined)[]], p: Readonly<Record<string, unknown>>) =>
    M.withInterval(M.accuracyScore, { method, level: (p['level'] as number | undefined) ?? 0.95 })(inputs[0], inputs[1])
      .estimate.interval?.[end] ?? Number.NaN

goldenSuite('intervals and model comparisons against statsmodels and fast DeLong', 'metrics/foundations', {
  wilsonLow: interval('wilson', 0),
  wilsonHigh: interval('wilson', 1),
  waldLow: interval('analytic', 0),
  waldHigh: interval('analytic', 1),
  accuracy: ([t, p]) => M.accuracyScore(t, p).estimate.value,
  delongVariance: ([t, s]) => M.delongVariance(t, s).variance,
  delongStatistic: ([t, a, b]) => M.delongTest(t, a, b as Tensor).statistic,
  delongPValue: ([t, a, b]) => M.delongTest(t, a, b as Tensor).pValue,
  mcnemarStatistic: ([t, a, b], p) => M.mcnemarTest(t, a, b as Tensor, p).statistic,
  mcnemarPValue: ([t, a, b], p) => M.mcnemarTest(t, a, b as Tensor, p).pValue,
  expectedCost: ([t, p, c]) => M.expectedCost(t, p, c as Tensor).estimate.value,
  bayesDecision: ([p, c]) => tensor(M.bayesDecision(p, c)),
})
