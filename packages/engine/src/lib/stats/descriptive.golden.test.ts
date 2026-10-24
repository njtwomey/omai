import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('descriptive statistics against numpy, scipy.stats and statsmodels', 'stats/descriptive', {
  mean: ([x, w]) => S.mean(x, { ...(w === undefined ? {} : { weights: w }) }),
  variance: ([x, w], p) => S.variance(x, { ddof: Number(p['ddof'] ?? 0), ...(w === undefined ? {} : { weights: w }) }),
  std: ([x, w], p) => S.std(x, { ddof: Number(p['ddof'] ?? 0), ...(w === undefined ? {} : { weights: w }) }),
  skewness: ([x], p) => S.skewness(x, { bias: p['bias'] !== false }),
  kurtosis: ([x], p) => S.kurtosis(x, { bias: p['bias'] !== false, fisher: p['fisher'] !== false }),
  moment: ([x], p) => S.moment(x, Number(p['order']), p['center'] === undefined ? {} : { center: Number(p['center']) }),
  describe: ([x]) => {
    const d = S.describe(x)
    return tensor([d.n, d.min, d.max, d.mean, d.variance, d.skewness, d.kurtosis])
  },
  zscore: ([x], p) => S.zscore(x, { ddof: Number(p['ddof'] ?? 0) }),
  sem: ([x]) => S.sem(x),
  mode: ([x]) => {
    const m = S.mode(x)
    return tensor([m.mode, m.count])
  },
  geometricMean: ([x, w]) => S.geometricMean(x, w === undefined ? {} : { weights: w }),
  harmonicMean: ([x, w]) => S.harmonicMean(x, w === undefined ? {} : { weights: w }),
})
