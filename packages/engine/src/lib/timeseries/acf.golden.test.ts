import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')

goldenSuite('sample correlation functions against statsmodels', 'timeseries/acf', {
  autocovariance: ([x], o) =>
    vec(
      T.autocovariance(toFlat(x), o['nlags'] as number, {
        adjusted: (o['adjusted'] as boolean | undefined) ?? false,
        demean: (o['demean'] as boolean | undefined) ?? true,
      }),
    ),
  acf: ([x], o) => vec(T.acf(toFlat(x), o['nlags'] as number).acf),
  acfBand: ([x], o) =>
    vec((T.acf(toFlat(x), o['nlags'] as number, { alpha: o['alpha'] as number }).band ?? []).slice(1)),
  pacf: ([x], o) => vec(T.pacf(toFlat(x), o['nlags'] as number, o['method'] as T.PacfMethod)),
  ccf: ([x, y], o) => vec(T.ccf(toFlat(x), toFlat(y as never), o['nlags'] as number, o['adjusted'] as boolean)),
})
