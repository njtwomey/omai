/**
 * Linear-Gaussian state-space models of a univariate series (structural time-series models, after Durbin and Koopman):
 * a local level (a random walk observed in noise) or a local linear trend, with noise variances fitted by maximum
 * likelihood and the states recovered by the Kalman filter and smoother.
 */

// TODO: factor analysis, PPCA and Bayesian PCA as programs (decomposition/ has the EM estimators), seasonal and
// regression components (state-space.ts), and multivariate observations in the exposition.
import { MdmlError } from '../../lib/base/index.ts'
import { kalmanEm, kalmanFilter, type LinearGaussianModel, rtsSmoother } from '../../lib/dynamics/filtering/index.ts'
import {
  coefficientTable,
  type Explained,
  type Exposition,
  learningCurve,
  posteriorGrid,
} from '../../lib/exposition/index.ts'
import { asTensor, item, type TensorLike, toArray, toFlat } from '../../lib/tensor/index.ts'

/** The structural models {@link linearGaussianStateSpace} builds. */
export type StructuralModel = 'local-level' | 'local-linear-trend'

/** Options of {@link linearGaussianStateSpace}. */
export interface StateSpaceOptions {
  /** The model: a random-walk level, or a level with a random-walk slope (default local level). */
  readonly model?: StructuralModel
  /** Starting noise variances: the observation's and each state's (default from the data's variance). */
  readonly observationVariance?: number
  readonly stateVariance?: number
  /** Learn the noise variances by EM (default true; complete series only). */
  readonly learn?: boolean
  /** EM iterations (default 50). */
  readonly maxIter?: number
  readonly tol?: number
}

/** The result of {@link linearGaussianStateSpace}. */
export interface StateSpaceFit {
  /** The fitted model's matrices as numbers. */
  readonly model: { F: number[][]; H: number[][]; Q: number[][]; R: number[][]; x0: number[]; P0: number[][] }
  /** The smoothed and filtered level (first state) with variances, per step. */
  readonly smoothed: { readonly mean: readonly number[]; readonly variance: readonly number[] }
  readonly filtered: { readonly mean: readonly number[]; readonly variance: readonly number[] }
  readonly logLikelihood: number
  /** The log likelihood after each EM iteration (empty without learning). */
  readonly history: readonly number[]
}

/** The level of a state sequence: the first state's means and variances. */
function levelOf(means: TensorLike, covs: TensorLike): { mean: number[]; variance: number[] } {
  const m = toArray(asTensor(means)) as number[][]
  const c = toArray(asTensor(covs)) as number[][][]
  return { mean: m.map((r) => r[0] ?? 0), variance: c.map((p) => p[0]?.[0] ?? 0) }
}

/**
 * A linear-Gaussian latent model of a univariate series (a structural time-series model; Durbin and Koopman): a local
 * level (random walk observed in noise) or a local linear trend, its noise variances learned by EM through the
 * Kalman filter and RTS smoother of `dynamics`. Missing values (NaN) are skipped by the filter (and, as the core's
EM needs a complete series, the variances are then kept at their starting values). Exposition: the
 * smoothed level as a `posteriorGrid` over time with the observations, the filtered level likewise (a second record),
 * a `learningCurve` of the EM log likelihood per iteration (when learned) and the noise variances as a coefficient
 * table.
 *
 * @param series - The observations `[T]` (or `[T, 1]`).
 * @param options - The model, starting variances, whether to learn, iterations and tolerance.
 * @returns The model, the smoothed and filtered levels and the likelihoods, with the exposition records.
 * @throws MdmlError `value` for an empty series.
 */
export function linearGaussianStateSpace(
  series: TensorLike,
  options: StateSpaceOptions = {},
): Explained<StateSpaceFit> {
  const z = Array.from(toFlat(asTensor(series)))
  const T = z.length
  if (T === 0) throw new MdmlError('value', 'linearGaussianStateSpace: no observations')
  const seen = z.filter((v) => Number.isFinite(v))
  const mean = seen.reduce((a, v) => a + v, 0) / Math.max(seen.length, 1)
  const variance = seen.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(seen.length - 1, 1) || 1
  const r = options.observationVariance ?? variance / 2
  const q = options.stateVariance ?? variance / 10
  const trend = options.model === 'local-linear-trend'
  const start: LinearGaussianModel = trend
    ? {
        F: [
          [1, 1],
          [0, 1],
        ],
        H: [[1, 0]],
        Q: [
          [q, 0],
          [0, q / 10],
        ],
        R: [[r]],
        x0: [seen[0] ?? 0, 0],
        P0: [
          [variance * 10, 0],
          [0, variance],
        ],
      }
    : { F: [[1]], H: [[1]], Q: [[q]], R: [[r]], x0: [seen[0] ?? 0], P0: [[variance * 10]] }
  const observations = z.map((v) => [v])
  const learned =
    (options.learn ?? true) && seen.length === T
      ? kalmanEm(start, observations, {
          learn: { F: false, H: false, Q: true, R: true, x0: false, P0: false },
          tol: options.tol ?? 1e-6,
          maxIterations: options.maxIter ?? 50,
        })
      : undefined
  const model: LinearGaussianModel = learned?.model ?? start
  const filtered = kalmanFilter(model, observations)
  const smoothed = rtsSmoother(model, filtered)
  const num = (v: TensorLike): number[][] => {
    const t = asTensor(v)
    return (t.shape.length === 2 ? toArray(t) : [toArray(t)]) as number[][]
  }
  const fitModel = {
    F: num(model.F),
    H: num(model.H),
    Q: num(model.Q),
    R: num(model.R),
    x0: Array.from(toFlat(asTensor(model.x0))),
    P0: num(model.P0),
  }
  const s = levelOf(smoothed.means, smoothed.covariances)
  const f = levelOf(filtered.means, filtered.covariances)
  const times = z.map((_, t) => [t])
  const observed = {
    inputs: times.filter((_, t) => Number.isFinite(z[t])),
    targets: z.filter((v) => Number.isFinite(v)),
  }
  const history = learned?.history ?? []
  const names = ['observation variance', ...(trend ? ['level variance', 'slope variance'] : ['level variance'])]
  const estimates = [fitModel.R[0]?.[0] ?? 0, ...fitModel.Q.map((row, i) => row[i] ?? 0)]
  const records: Exposition[] = [
    posteriorGrid({ inputs: times, mean: s.mean, variance: s.variance, observed }),
    posteriorGrid({ inputs: times, mean: f.mean, variance: f.variance, observed }),
    ...(history.length > 0
      ? [
          learningCurve({
            sizes: history.map((_, i) => i + 1),
            train: history,
            validation: history,
            resource: 'EM iterations',
            labels: { train: 'log likelihood', validation: 'log likelihood' },
          }),
        ]
      : []),
    coefficientTable({ names, estimates }),
  ]
  return {
    model: fitModel,
    smoothed: s,
    filtered: f,
    logLikelihood: item(asTensor(filtered.logLikelihood)),
    history,
    exposition: records,
  }
}
