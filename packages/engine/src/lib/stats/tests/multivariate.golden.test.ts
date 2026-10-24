import { goldenSuite } from '../../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const tp = (r: { statistic: number; pValue: number }): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('multivariate tests against pingouin and statsmodels', 'stats/tests/multivariate', {
  hotellingOne: ([x]) => tp(T.hotellingT2(x)),
  hotellingMu: ([x, mu]) => tp(T.hotellingT2(x, undefined, { mu: toFlat(mu) })),
  hotellingPaired: ([x, y]) => tp(T.hotellingT2(x, y, { paired: true })),
  hotellingTwo: ([x, y]) => tp(T.hotellingT2(x, y)),
  manova: ([a, b, c], p) => {
    const r = T.manova([a, b, c as Tensor])[p['stat'] as 'wilks']
    return tensor([r.value, r.F, r.df[0], r.df[1], r.pValue])
  },
  boxM: ([a, b, c]) => {
    const r = T.boxM([a, b, c as Tensor])
    return tensor([r.statistic, r.pValue, Number(r.df)])
  },
  mardia: ([x]) => {
    const r = T.mardia(x)
    return tensor([r.skewness.statistic, r.skewness.pValue, r.kurtosis.statistic, r.kurtosis.pValue])
  },
})
