import { goldenSuite, type Params } from '../../testing/suite.ts'
import * as S from './index.ts'

const range = (p: Params): { range?: readonly [number, number] } =>
  p['lo'] === undefined ? {} : { range: [Number(p['lo']), Number(p['hi'])] }

goldenSuite('histograms against numpy and scipy.stats', 'stats/histograms', {
  edges: ([x], p) => S.histogramBinEdges(x, { bins: p['bins'] as S.BinRule, ...range(p) }),
  histogram: ([x, w], p) =>
    S.histogram(x, {
      bins: Number(p['bins']),
      density: p['density'] === true,
      ...range(p),
      ...(w === undefined ? {} : { weights: w }),
    }).counts,
  histogramEdges: ([x, edges]) => S.histogram(x, { bins: edges }).counts,
  histogram2d: ([x, y, w], p) =>
    S.histogram2d(x, y, {
      bins: [Number(p['bx']), Number(p['by'])],
      density: p['density'] === true,
      ...(w === undefined ? {} : { weights: w }),
    }).counts,
  binnedStatistic: ([x, y], p) =>
    S.binnedStatistic(x, y, { statistic: p['statistic'] as S.BinStatistic, bins: Number(p['bins']) }).statistic,
  binNumber: ([x, y], p) => S.binnedStatistic(x, y, { bins: Number(p['bins']), ...range(p) }).binNumber,
})
