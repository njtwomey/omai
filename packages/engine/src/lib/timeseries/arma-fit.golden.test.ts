import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')

goldenSuite('ARMA estimators against statsmodels', 'timeseries/arma_fit', {
  hannanRissanen: ([x], o) => {
    const unbiased = o['unbiased'] as boolean | undefined
    const e = T.hannanRissanen(toFlat(x), o['order'] as [number, number], unbiased === undefined ? {} : { unbiased })
    return vec([...e.ar, ...e.ma, e.sigma2])
  },
  fitMa: ([x], o) => {
    const e = T.fitMa(toFlat(x), o['order'] as number)
    return vec([...e.ma, e.sigma2])
  },
  fitArOls: ([x], o) => {
    const e = T.fitAr(toFlat(x), o['order'] as number, { method: 'ols' })
    const intercept = e.mean * (1 - e.ar.reduce((a, b) => a + b, 0))
    return vec([intercept, ...e.ar, e.sigma2])
  },
  selectArOrder: ([x], o) => vec(T.selectArOrder(toFlat(x), o['maxOrder'] as number, o['criterion'] as 'aic').scores),
})
