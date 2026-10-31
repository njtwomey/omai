import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')
const nums = (t: Parameters<typeof toFlat>[0]) => toFlat(t)

goldenSuite('ARMA theory against statsmodels', 'timeseries/arma', {
  armaAutocovariance: ([ar, ma], o) =>
    vec(T.armaAutocovariance(nums(ar), nums(ma), o['sigma2'] as number, (o['nlags'] as number) + 0).subarray(0, 13)),
  armaAcf: ([ar, ma], o) => vec(T.armaAcf(nums(ar), nums(ma), o['nlags'] as number)),
  armaPacf: ([ar, ma], o) => vec(T.armaPacf(nums(ar), nums(ma), o['nlags'] as number)),
  armaPsiWeights: ([ar, ma], o) => vec(T.armaPsiWeights(nums(ar), nums(ma), o['n'] as number)),
  armaPiWeights: ([ar, ma], o) => vec(T.armaPiWeights(nums(ar), nums(ma), o['n'] as number)),
  innovations: ([gamma]) => {
    const r = T.innovations(nums(gamma as never))
    // The first three predictor coefficients of every row from the second, zero-padded.
    const rows = r.theta.slice(1).flatMap((row) => [0, 1, 2].map((j) => row[j] ?? 0))
    return vec([...r.v, ...rows])
  },
})
