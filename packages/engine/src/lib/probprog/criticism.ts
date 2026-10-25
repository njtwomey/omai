/**
 * Model criticism and comparison, ArviZ style: WAIC and Pareto-smoothed importance-sampling leave-one-out
 * cross-validation (PSIS-LOO) as estimates of out-of-sample predictive accuracy, posterior predictive checks of a test
 * statistic, and model comparison by elpd with stacking and pseudo-BMA weights.
 */

import { logSumExpOf, MdmlError } from '../base/index.ts'
import { lbfgs } from '../optim/index.ts'
import { paretoSmoothedWeights } from '../sampling/index.ts'
import {
  asTensor,
  concat,
  fromStorage,
  log,
  matmul,
  neg,
  primalOf,
  softmax,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/** An estimate of the expected log pointwise predictive density (elpd), with its parts. */
export interface ElpdEstimate {
  readonly method: 'waic' | 'loo'
  /** The estimated elpd (higher is better). */
  readonly elpd: number
  /** Its standard error, sqrt(N var(pointwise)) with the n - 1 denominator (as ArviZ). */
  readonly se: number
  /** The effective number of parameters (p_waic or p_loo). */
  readonly p: number
  /** The log pointwise predictive density on the training data. */
  readonly lppd: number
  readonly pointwise: Float64Array
  /** PSIS-LOO only: the Pareto k of each observation's importance weights. */
  readonly paretoK?: Float64Array
  /** Whether some diagnostic flags the estimate as unreliable (WAIC variances above 0.4, or k above the threshold). */
  readonly warning: boolean
  readonly observations: number
  readonly draws: number
}

function matrix(logLik: TensorLike): { readonly S: number; readonly N: number; readonly data: ArrayLike<number> } {
  const t = primalOf(asTensor(logLik))
  if (t.shape.length !== 2) throw new MdmlError('shape', 'criticism: log-likelihoods must be [draws, observations]')
  const S = t.shape[0] ?? 0
  const N = t.shape[1] ?? 0
  if (S < 2) throw new MdmlError('value', 'criticism: need at least two draws')
  return { S, N, data: toFlat(t) }
}

function column(
  m: { readonly S: number; readonly N: number; readonly data: ArrayLike<number> },
  i: number,
): Float64Array {
  return Float64Array.from({ length: m.S }, (_, s) => m.data[s * m.N + i] ?? Number.NaN)
}

/** sqrt(n var(pointwise)): ArviZ uses ddof 1 for elpd standard errors and ddof 0 for those of differences. */
function seOf(pointwise: Float64Array, ddof: number): number {
  const n = pointwise.length
  if (n <= ddof) return 0
  const mean = pointwise.reduce((a, b) => a + b, 0) / n
  const v = pointwise.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - ddof)
  return Math.sqrt(n * v)
}

/**
 * The widely applicable information criterion (Watanabe 2010), in ArviZ's elpd form: elpd_waic = lppd - p_waic with
 * lppd_i = log mean_s p(y_i | theta_s) and p_waic_i = var_s log p(y_i | theta_s), summed over observations.
 *
 * @param logLik - Pointwise log-likelihoods `[draws, observations]` (see `pointwiseLogLikelihood`).
 * @returns The estimate; `warning` when some p_waic_i exceeds 0.4.
 * @throws MdmlError with code `shape` for a non-matrix input, or `value` for fewer than two draws.
 * @concept model-comparison
 * @example
 * ```ts
 * import { waic } from 'mdml-engine/probprog'
 * // Two draws with identical log-likelihoods: no variance, so elpd = lppd.
 * waic([[-1, -2], [-1, -2]]).elpd // => -3
 * ```
 */
export function waic(logLik: TensorLike): ElpdEstimate {
  const m = matrix(logLik)
  const pointwise = new Float64Array(m.N)
  let lppd = 0
  let p = 0
  let warning = false
  for (let i = 0; i < m.N; i++) {
    const c = column(m, i)
    const l = logSumExpOf(c) - Math.log(m.S)
    const mean = c.reduce((a, b) => a + b, 0) / m.S
    const v = c.reduce((a, b) => a + (b - mean) ** 2, 0) / (m.S - 1)
    if (v > 0.4) warning = true
    lppd += l
    p += v
    pointwise[i] = l - v
  }
  return {
    method: 'waic',
    elpd: lppd - p,
    se: seOf(pointwise, 1),
    p,
    lppd,
    pointwise,
    warning,
    observations: m.N,
    draws: m.S,
  }
}

/**
 * Pareto-smoothed importance-sampling leave-one-out cross-validation (Vehtari, Gelman and Gabry 2017; ArviZ's `loo`):
 * for each observation, the importance ratios 1 / p(y_i | theta_s) are Pareto-smoothed (`sampling`'s PSIS) and
 * elpd_loo_i = log sum_s w_is p(y_i | theta_s). Each observation's Pareto k diagnoses its estimate.
 *
 * @param logLik - Pointwise log-likelihoods `[draws, observations]`.
 * @param options - The relative efficiency of the draws (default 1) and the k threshold (default min(1 - 1 / log10 S,
 *   0.7), as ArviZ).
 * @returns The estimate with per-observation Pareto k; `warning` when some k exceeds the threshold.
 * @throws MdmlError with code `shape` for a non-matrix input, or `value` for fewer than 25 draws.
 * @concept model-comparison
 * @example
 * ```ts
 * import { psisLoo } from 'mdml-engine/probprog'
 * import { key, normal } from 'mdml-engine/random'
 * import { reshape } from 'mdml-engine/tensor'
 * const ll = reshape(normal(key(0), [400], { loc: -1, scale: 0.1 }), [200, 2])
 * psisLoo(ll).warning // => false
 * ```
 */
export function psisLoo(
  logLik: TensorLike,
  options: { readonly rEff?: number; readonly kThreshold?: number } = {},
): ElpdEstimate {
  const m = matrix(logLik)
  const threshold = options.kThreshold ?? Math.min(1 - 1 / Math.log10(m.S), 0.7)
  const pointwise = new Float64Array(m.N)
  const paretoK = new Float64Array(m.N)
  let lppd = 0
  for (let i = 0; i < m.N; i++) {
    const c = column(m, i)
    const ps = paretoSmoothedWeights(
      Float64Array.from(c, (v) => -v),
      options.rEff === undefined ? {} : { rEff: options.rEff },
    )
    const lw = toFlat(ps.logWeights)
    pointwise[i] = logSumExpOf(Float64Array.from(c, (v, s) => v + (lw[s] ?? Number.NEGATIVE_INFINITY)))
    paretoK[i] = ps.paretoK
    lppd += logSumExpOf(c) - Math.log(m.S)
  }
  const elpd = pointwise.reduce((a, b) => a + b, 0)
  return {
    method: 'loo',
    elpd,
    se: seOf(pointwise, 1),
    p: lppd - elpd,
    lppd,
    pointwise,
    paretoK,
    warning: paretoK.some((k) => k > threshold),
    observations: m.N,
    draws: m.S,
  }
}

/** The result of {@link posteriorPredictiveCheck}. */
export interface PredictiveCheck {
  /** The statistic on the observed data. */
  readonly observed: number
  /** The statistic on each replicated data set. */
  readonly replicated: Float64Array
  /** P(T(y_rep) >= T(y)): values near 0 or 1 flag misfit. */
  readonly pValue: number
}

/**
 * A posterior predictive check (Gelman, Meng and Stern 1996): a test statistic on the observed data against its
 * distribution over replicated data sets, with the posterior predictive p-value.
 *
 * @param observed - The observed data.
 * @param replicated - Replicated data sets `[S, ...shape]` (from `posteriorPredictive`).
 * @param statistic - The test statistic of one data set.
 * @returns The observed statistic, the replicated ones and the p-value.
 * @concept predictive-checks
 * @example
 * ```ts
 * import { posteriorPredictiveCheck } from 'mdml-engine/probprog'
 * import { item, max, tensor } from 'mdml-engine/tensor'
 * posteriorPredictiveCheck(tensor([0, 1]), tensor([[0, 2], [0, 0.5]]), (x) => item(max(x))).pValue // => 0.5
 * ```
 */
export function posteriorPredictiveCheck(
  observed: TensorLike,
  replicated: TensorLike,
  statistic: (data: Tensor) => number,
): PredictiveCheck {
  const y = primalOf(asTensor(observed))
  const rep = primalOf(asTensor(replicated))
  const S = rep.shape[0] ?? 0
  const shape = rep.shape.slice(1)
  const size = shape.reduce((a, b) => a * b, 1)
  const flat = toFlat(rep)
  const t0 = statistic(y)
  const ts = Float64Array.from({ length: S }, (_, s) =>
    statistic(fromStorage(Float64Array.from(flat.slice(s * size, (s + 1) * size)), shape, 'float64')),
  )
  const pValue = ts.reduce((a, t) => a + (t >= t0 ? 1 : 0), 0) / Math.max(1, S)
  return { observed: t0, replicated: ts, pValue }
}

/**
 * Stacking weights of predictive distributions (Yao, Vehtari, Simpson and Gelman 2018; ArviZ's `compare` with
 * `method='stacking'`): the weights on the simplex maximising sum_i log sum_k w_k exp(elpd_ik) of the models'
 * pointwise LOO densities, found by L-BFGS on a softmax parameterisation.
 *
 * @param pointwise - Each model's pointwise elpd (all of the same length).
 * @returns The weights (summing to one).
 * @throws MdmlError with code `value` for no models or unequal lengths.
 * @concept model-comparison
 * @example
 * ```ts
 * import { stackingWeights } from 'mdml-engine/probprog'
 * // The second model predicts every point better: it takes all the weight.
 * stackingWeights([[-2, -2, -2], [-1, -1, -1]]).map((w) => Math.round(w * 1e4) / 1e4) // => [0, 1]
 * ```
 */
export function stackingWeights(pointwise: readonly ArrayLike<number>[]): number[] {
  const K = pointwise.length
  if (K === 0) throw new MdmlError('value', 'stackingWeights: no models')
  const N = pointwise[0]?.length ?? 0
  if (pointwise.some((p) => p.length !== N)) throw new MdmlError('value', 'stackingWeights: unequal lengths')
  if (K === 1) return [1]
  // exp(elpd_ik - max_k elpd_ik): the shift cancels in the argmax.
  const rows = new Float64Array(N * K)
  for (let i = 0; i < N; i++) {
    let m = Number.NEGATIVE_INFINITY
    for (let k = 0; k < K; k++) m = Math.max(m, pointwise[k]?.[i] ?? Number.NEGATIVE_INFINITY)
    for (let k = 0; k < K; k++) rows[i * K + k] = Math.exp((pointwise[k]?.[i] ?? Number.NEGATIVE_INFINITY) - m)
  }
  const E = fromStorage(rows, [N, K], 'float64')
  const objective = (z: Value): Value => {
    const w = softmax(concat([z, tensor([0])]), 0)
    return neg(sum(log(matmul(E, w))))
  }
  const r = lbfgs(objective, new Float64Array(K - 1), { maxIterations: 1000, gradTol: 1e-12 })
  const z = toFlat(primalOf(asTensor(r.x as Value)))
  const w = toFlat(primalOf(softmax(concat([tensor(z), tensor([0])]), 0)))
  return w
}

/**
 * Pseudo-BMA weights: exp(elpd_k) normalised (Yao et al. 2018), without the Bayesian bootstrap.
 *
 * @param elpds - Each model's elpd.
 * @returns The weights.
 * @concept model-comparison
 * @example
 * ```ts
 * import { pseudoBmaWeights } from 'mdml-engine/probprog'
 * pseudoBmaWeights([0, Math.log(3)]) // => [0.25, 0.75]
 * ```
 */
export function pseudoBmaWeights(elpds: readonly number[]): number[] {
  const z = logSumExpOf(elpds)
  return elpds.map((e) => Math.exp(e - z))
}

/** One row of {@link compareModels}. */
export interface ComparisonRow {
  readonly name: string
  readonly rank: number
  readonly elpd: number
  readonly p: number
  /** This model's elpd minus the best model's (0 for the best, negative otherwise, as ArviZ 1.x). */
  readonly elpdDiff: number
  readonly se: number
  /** The standard error of the difference to the best model, from the pointwise differences. */
  readonly dse: number
  readonly weight: number
  readonly warning: boolean
}

/**
 * Compare models by their elpd estimates (ArviZ's `compare`): ranked by elpd, with differences to the best model and
 * their standard errors from the pointwise differences, and stacking (default) or pseudo-BMA weights.
 *
 * @param estimates - Each model's estimate by name (all WAIC or all LOO, on the same observations).
 * @param options - The weighting method.
 * @returns One row per model, best first.
 * @throws MdmlError with code `value` when the estimates mix methods or observation counts.
 * @concept model-comparison
 * @example
 * ```ts
 * import { compareModels, waic } from 'mdml-engine/probprog'
 * const a = waic([[-1, -1], [-1, -1]])
 * const b = waic([[-2, -1], [-2, -1]])
 * compareModels({ a, b }).map((r) => [r.name, r.elpdDiff]) // => [['a', 0], ['b', -1]]
 * ```
 */
export function compareModels(
  estimates: Readonly<Record<string, ElpdEstimate>>,
  options: { readonly method?: 'stacking' | 'pseudoBma' } = {},
): ComparisonRow[] {
  const names = Object.keys(estimates)
  const list = names.map((n) => estimates[n] as ElpdEstimate)
  if (list.length === 0) throw new MdmlError('value', 'compareModels: no models')
  if (new Set(list.map((e) => e.method)).size > 1 || new Set(list.map((e) => e.observations)).size > 1)
    throw new MdmlError('value', 'compareModels: estimates must share the method and the observations')
  const weights =
    options.method === 'pseudoBma'
      ? pseudoBmaWeights(list.map((e) => e.elpd))
      : stackingWeights(list.map((e) => e.pointwise))
  const order = names.map((_, i) => i).sort((a, b) => (list[b]?.elpd ?? 0) - (list[a]?.elpd ?? 0))
  const best = list[order[0] ?? 0] as ElpdEstimate
  return order.map((i, rank) => {
    const e = list[i] as ElpdEstimate
    const diff = Float64Array.from(best.pointwise, (v, j) => v - (e.pointwise[j] ?? 0))
    return {
      name: names[i] ?? '',
      rank,
      elpd: e.elpd,
      p: e.p,
      elpdDiff: e.elpd - best.elpd,
      se: e.se,
      dse: rank === 0 ? 0 : seOf(diff, 0),
      weight: weights[i] ?? 0,
      warning: e.warning,
    }
  })
}
