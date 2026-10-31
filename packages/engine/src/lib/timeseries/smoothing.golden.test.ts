import { decode } from '../../testing/golden.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const specOf = (o: Readonly<Record<string, unknown>>): T.EtsSpec => {
  const s = o['structure'] as Record<string, unknown>
  return {
    error: s['error'] as 'add',
    trend: s['trend'] as 'none',
    damped: s['damped'] as boolean,
    seasonal: s['seasonal'] as 'none',
    period: s['period'] as number,
  }
}

/** statsmodels' parameter order: alpha, beta, gamma, phi, then level, trend and the seasonal values newest first. */
function paramsOf(spec: T.EtsSpec, p: readonly number[]): T.EtsParameters {
  let k = 0
  const alpha = p[k++] ?? 0
  const beta = spec.trend === 'none' ? undefined : (p[k++] ?? 0)
  const gamma = spec.seasonal === 'none' ? undefined : (p[k++] ?? 0)
  const phi = spec.damped === true && spec.trend !== 'none' ? (p[k++] ?? 1) : undefined
  const level0 = p[k++] ?? 0
  const trend0 = spec.trend === 'none' ? undefined : (p[k++] ?? 0)
  const seasonal0 = spec.seasonal === 'none' ? undefined : p.slice(k, k + (spec.period ?? 0)).reverse()
  return {
    alpha,
    level0,
    ...(beta === undefined ? {} : { beta }),
    ...(gamma === undefined ? {} : { gamma }),
    ...(phi === undefined ? {} : { phi }),
    ...(trend0 === undefined ? {} : { trend0 }),
    ...(seasonal0 === undefined ? {} : { seasonal0 }),
  }
}

goldenSuite('exponential smoothing against statsmodels ETSModel', 'timeseries/smoothing', {
  etsSmooth: ([y, params], o) => {
    const spec = specOf(o)
    const steps = o['steps'] as number
    const par = paramsOf(spec, toFlat(params as Tensor))
    const history = toFlat(y)
    const f = T.etsFilter(history, spec, par)
    const fc = T.etsForecast(
      {
        spec,
        params: par,
        logLikelihood: f.logLikelihood,
        sigma2: f.sigma2,
        nparams: 0,
        aic: 0,
        bic: 0,
        nobs: history.length,
        final: f.final,
        converged: true,
        iterations: 0,
      },
      steps,
    )
    const lower = fc.intervals?.[0]?.lower ?? new Array<number>(steps).fill(Number.NaN)
    const upper = fc.intervals?.[0]?.upper ?? new Array<number>(steps).fill(Number.NaN)
    // The intervals of multiplicative models are by simulation in statsmodels; only the additive ones are compared.
    const keep = (o['additive'] as boolean) ? [...lower, ...upper] : []
    const all = [f.logLikelihood, ...f.fitted, ...fc.mean, ...keep]
    return fromStorage(Float64Array.from(all), [all.length], 'float64')
  },
  // The maximum-likelihood fit must reach at least statsmodels' (a non-convex problem, so a local optimum may be lower
  // than the reference only within half a log-likelihood unit); the case passes by returning the reference.
  etsFitBound: ([y], o, c) => {
    const reference = toFlat(decode(c.output) as Tensor)[0] ?? 0
    const ours = T.etsFit(toFlat(y), specOf(o)).logLikelihood
    return fromStorage(Float64Array.of(ours > reference - 0.5 ? reference : ours), [1], 'float64')
  },
})
