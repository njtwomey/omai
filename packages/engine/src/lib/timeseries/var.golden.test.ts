import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const rowsOf = (t: Tensor): number[][] => {
  const [n, k] = t.shape as [number, number]
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * k, (i + 1) * k))
}
const tensorOf = (rows: readonly (readonly number[])[]) =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')
const vec = (a: ArrayLike<number>) => fromStorage(Float64Array.from(a), [a.length], 'float64')
const cube = (m: readonly (readonly (readonly number[])[])[]) =>
  fromStorage(Float64Array.from(m.flat(2)), [m.length, m[0]?.length ?? 0, m[0]?.[0]?.length ?? 0], 'float64')

const fitOf = (y: Tensor, o: Readonly<Record<string, unknown>>) =>
  T.varFit(rowsOf(y), { lags: o['lags'] as number, trend: (o['trend'] as T.VarTrend | undefined) ?? 'c' })

goldenSuite('vector autoregression against statsmodels', 'timeseries/var', {
  // statsmodels' params: deterministic terms, then lag 1 of all variables, lag 2, ..., one column per equation.
  varParams: ([y], o) => {
    const f = fitOf(y, o)
    const k = f.nVars
    const rows = [
      ...f.deterministic.map((r) => [...r]),
      ...f.coefficients.flatMap((a) => Array.from({ length: k }, (_, v) => a.map((eq) => eq[v] ?? 0))),
    ]
    return tensorOf(rows)
  },
  varSigma: ([y], o) => tensorOf(fitOf(y, o).sigma),
  varStdErrors: ([y], o) => tensorOf(fitOf(y, o).stdErrors),
  varCriteria: ([y], o) => {
    const f = fitOf(y, o)
    return vec([f.logLikelihood, f.aic, f.bic, f.hqic, f.fpe])
  },
  varForecast: ([y], o) => {
    const f = fitOf(y, o)
    const r = T.varForecast(f, rowsOf(y), o['steps'] as number, o['alpha'] as number)
    return cube([r.mean, r.lower, r.upper])
  },
  varStability: ([y], o) => vec(T.varStability(fitOf(y, o)).moduli),
  varSelectOrder: ([y], o) => {
    const s = T.varSelectOrder(rowsOf(y), o['maxLags'] as number)
    return tensorOf([
      s.criteria.map((c) => c.aic),
      s.criteria.map((c) => c.bic),
      s.criteria.map((c) => c.hqic),
      s.criteria.map((c) => c.fpe),
    ])
  },
  varIrf: ([y], o) =>
    cube(T.varImpulseResponse(fitOf(y, o), o['steps'] as number, { orthogonal: o['orth'] as boolean })),
  varIrfCum: ([y], o) => cube(T.varImpulseResponse(fitOf(y, o), o['steps'] as number, { cumulative: true })),
  varFevd: ([y], o) => cube(T.varForecastErrorVarianceDecomposition(fitOf(y, o), o['periods'] as number)),
  granger: ([y], o) => {
    const t = T.grangerCausality(fitOf(y, o), o['caused'] as number[], o['causing'] as number[], o['kind'] as 'f')
    return vec([t.statistic, t.pValue])
  },
})
