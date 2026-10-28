import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')
const rowsOf = (t: Tensor | undefined): number[][] | undefined => {
  if (t === undefined) return undefined
  const [n, k] = t.shape as [number, number]
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * k, (i + 1) * k))
}

const specOf = (o: Readonly<Record<string, unknown>>, exog?: Tensor): T.ArimaSpec => {
  const seasonal = o['seasonal'] as number[]
  const rows = rowsOf(exog)
  return {
    order: o['order'] as [number, number, number],
    seasonal: { order: [seasonal[0] ?? 0, seasonal[1] ?? 0, seasonal[2] ?? 0], period: seasonal[3] || 1 },
    trend: o['trend'] as T.Trend,
    ...(rows === undefined ? {} : { exog: rows }),
  }
}

/** statsmodels' parameter order: regression, ar, ma, seasonal ar, seasonal ma, sigma2. */
const paramsOf = (f: T.ArimaFit): number[] => [
  ...f.regression,
  ...f.ar,
  ...f.ma,
  ...f.seasonalAr,
  ...f.seasonalMa,
  f.sigma2,
]

goldenSuite('ARIMA against statsmodels', 'timeseries/arima', {
  arimaFit: ([y, exog], o) => vec(paramsOf(T.arimaFit(toFlat(y), specOf(o, exog)))),
  arimaLikelihood: ([y, exog], o) => {
    const f = T.arimaFit(toFlat(y), specOf(o, exog))
    return vec([f.logLikelihood, f.aic, f.bic])
  },
  arimaForecast: ([y, params, allExog], o) => {
    const spec = specOf(o, allExog)
    const rs = spec.seasonal?.order ?? [0, 0, 0]
    const p = toFlat(params as Tensor)
    const [P, Q] = [rs[0], rs[2]]
    const [ord0, , ord2] = spec.order
    const kReg = p.length - 1 - ord0 - ord2 - P - Q
    let at = 0
    const take = (k: number) => {
      const part = p.slice(at, at + k)
      at += k
      return part
    }
    const fit: T.ArimaFit = {
      order: spec.order,
      seasonalOrder: [rs[0], rs[1], rs[2]],
      period: spec.seasonal?.period ?? 1,
      trend: spec.trend ?? 'n',
      regressionNames: [],
      regression: take(kReg),
      ar: take(ord0),
      ma: take(ord2),
      seasonalAr: take(P),
      seasonalMa: take(Q),
      sigma2: p[p.length - 1] ?? 1,
      logLikelihood: 0,
      aic: 0,
      bic: 0,
      nobs: 0,
      stdErrors: [],
      method: 'mle',
      converged: true,
      iterations: 0,
    }
    const history = toFlat(y)
    const steps = o['steps'] as number
    const exog = rowsOf(allExog)
    const f = T.arimaForecast(fit, history, steps, exog === undefined ? {} : { exog })
    const ci = f.intervals[0]
    return fromStorage(
      Float64Array.from([...f.mean, ...f.se, ...(ci?.lower ?? []), ...(ci?.upper ?? [])]),
      [4, steps],
      'float64',
    )
  },
})
