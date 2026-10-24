import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('quantiles against numpy and scipy.stats', 'stats/quantiles', {
  quantile: ([x, q], p) => S.quantile(x, q, { method: p['method'] as S.QuantileMethod }),
  median: ([x]) => S.median(x),
  iqr: ([x], p) =>
    S.iqr(x, {
      ...(p['scale'] === undefined ? {} : { scale: p['scale'] as 'normal' }),
      ...(p['lo'] === undefined ? {} : { rng: [Number(p['lo']), Number(p['hi'])] as const }),
    }),
  percentileOfScore: ([x, scores], p) =>
    tensor(toFlat(scores).map((s) => S.percentileOfScore(x, s, { kind: p['kind'] as S.PercentileKind }))),
  ecdfQuantiles: ([x]) => S.ecdf(x).quantiles,
  ecdfProbabilities: ([x]) => S.ecdf(x).probabilities,
  ecdfEvaluate: ([x, t]) => S.ecdf(x).evaluate(t),
  ecdfLow: ([x], p) => S.ecdf(x).confidenceInterval(0.9, p['method'] as 'linear').low,
  ecdfHigh: ([x], p) => S.ecdf(x).confidenceInterval(0.9, p['method'] as 'linear').high,
  medianAbsoluteDeviation: ([x], p) =>
    S.medianAbsoluteDeviation(x, p['scale'] === undefined ? {} : { scale: p['scale'] as 'normal' }),
})
