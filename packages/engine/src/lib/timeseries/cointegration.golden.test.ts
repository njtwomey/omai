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
const absRows = (m: readonly (readonly number[])[]) => tensorOf(m.map((r) => r.map(Math.abs)))

const johansenOf = (y: Tensor, o: Readonly<Record<string, unknown>>) =>
  T.johansen(rowsOf(y), o['det'] as T.JohansenDeterministicOrder, o['kd'] as number)
const vecmOf = (y: Tensor, o: Readonly<Record<string, unknown>>) =>
  T.vecmFit(rowsOf(y), {
    kArDiff: o['kd'] as number,
    rank: o['rank'] as number,
    deterministic: o['det'] as T.VecmDeterministic,
  })

goldenSuite('cointegration against statsmodels', 'timeseries/cointegration', {
  engleGranger: ([y], o) => {
    const rows = rowsOf(y)
    const cols = o['cols'] as number[]
    const r = T.engleGranger(
      rows.map((row) => row[0] ?? 0),
      rows.map((row) => cols.map((c) => row[c] ?? 0)),
      {
        trend: o['trend'] as 'c',
        maxlag: o['maxlag'] as number,
        autolag: o['autolag'] === null ? null : (String(o['autolag']).toUpperCase() as 'AIC'),
      },
    )
    return vec([r.statistic, r.pValue, ...r.critical])
  },
  johansenEig: ([y], o) => vec(johansenOf(y, o).eigenvalues),
  johansenStats: ([y], o) => {
    const r = johansenOf(y, o)
    return tensorOf([[...r.traceStatistics], [...r.maxEigenvalueStatistics]])
  },
  johansenCrit: ([y], o) => {
    const r = johansenOf(y, o)
    return tensorOf([...r.traceCritical, ...r.maxEigenvalueCritical])
  },
  johansenEvec: ([y], o) => absRows(johansenOf(y, o).eigenvectors),
  vecmPart: ([y], o) => {
    const f = vecmOf(y, o)
    switch (o['part']) {
      case 'alpha':
        return tensorOf(f.alpha)
      case 'beta':
        return tensorOf(f.beta)
      case 'gamma':
        return tensorOf(f.gamma)
      case 'sigma':
        return tensorOf(f.sigma)
      case 'llf':
        return tensorOf([[f.logLikelihood]])
      case 'det':
        return tensorOf(f.detCoef)
      default:
        return tensorOf(f.detCoefCoint)
    }
  },
  vecmForecast: ([y], o) => {
    const f = vecmOf(y, o)
    const r = T.vecmForecast(f, o['steps'] as number, o['alpha'] as number)
    return fromStorage(
      Float64Array.from([...r.mean.flat(), ...r.lower.flat(), ...r.upper.flat()]),
      [3, r.mean.length, f.nVars],
      'float64',
    )
  },
})
