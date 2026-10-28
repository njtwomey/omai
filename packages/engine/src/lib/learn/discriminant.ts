/**
 * Discriminant analysis: linear (pooled covariance), quadratic (one covariance per class) and everything between, as
 * one estimator whose covariance is shrunk towards a scaled identity (a fixed intensity or Ledoit-Wolf's), mixed
 * between the pooled and per-class estimates (Friedman's regularised discriminant analysis) and floored by an
 * eigenvalue regularisation (scikit-learn's `reg_param`); Fisher's discriminant axes as a transform; and the kernel
 * Fisher discriminant, which finds the axes in a feature space through the kernel matrix with a ridge on the
 * within-class scatter. The estimators sit on the iterative base and absorb the training rows one per step, so
 * `fittedAt(t)` is the discriminant of the first `t` rows.
 */

import {
  type Kernel,
  type PairwiseKernelMetric,
  type PairwiseKernelOptions,
  pairwiseKernels,
} from '../kernels/index.ts'
import { dgemm, dgemv, eigh } from '../numerics/linalg/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { ledoitWolf } from '../stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import {
  classifierFromScores,
  prefixOf,
  type RowsState,
  rowsAlgorithm,
  type ScoreClassifierVerbs,
} from './_classify.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { LearnData } from './estimator.ts'
import { encodeLabels, type MatrixView, matrixView, rowsOf, targetVector } from './validation.ts'

/** The hyperparameters of {@link discriminantAnalysis}. */
export interface DiscriminantHyper {
  /** One pooled covariance (linear discriminant) or one per class (quadratic). */
  readonly covariance: 'pooled' | 'class'
  /**
   * Shrinkage of every class covariance towards `(trace / d) I`: an intensity in `[0, 1]`, or `'auto'` for
   * Ledoit-Wolf's, estimated per class on the standardised features as scikit-learn does.
   */
  readonly shrinkage: number | 'auto'
  /** Eigenvalue regularisation `(1 - r) Sigma + r I` of the covariances (scikit-learn's QDA `reg_param`; the shrinkage is its `eigen` solver's). */
  readonly regParam: number
  /** Friedman's mix of a class covariance with the pooled one, `mix * class + (1 - mix) * pooled` (class mode only). */
  readonly mix: number
  /**
   * Divide scatter by `n - k` (pooled) or `n_c - 1` (class) rather than by `n` or `n_c` (default
   * false, which is scikit-learn's: its LDA and QDA both take maximum-likelihood covariances).
   */
  readonly unbiased: boolean
  /** The class priors (default the class frequencies). */
  readonly priors?: readonly number[]
  /** The number of discriminant axes of `transform` (default `min(k - 1, d)`). */
  readonly nComponents?: number
  /** Eigenvalues of a covariance below this fraction of the largest count as zero (the pseudo-inverse). */
  readonly rankTolerance: number
}

/** The fitted parameters of {@link discriminantAnalysis}: plain JSON. */
export type DiscriminantParams = {
  readonly classes: number[]
  /** The class priors; a class with prior 0 has not been seen. */
  readonly priors: number[]
  /** The class means `[k][d]`. */
  readonly means: number[][]
  /** The overall mean (the prior-weighted mean of the class means). */
  readonly xbar: number[]
  /** The pooled within-class covariance `[d][d]`. */
  readonly covariance: number[][]
  /** The linear scores' weights `Sigma^+ mu_c`, `[k][d]` (pooled mode; empty otherwise). */
  readonly coef: number[][]
  /** The linear scores' offsets `-mu_c' Sigma^+ mu_c / 2 + log prior_c` (pooled mode; empty otherwise). */
  readonly intercept: number[]
  /** Per class, the whitening `W` (`[d][r]`, `W W' = Sigma_c^+`) and the log determinant of the retained part (class mode). */
  readonly whiteners: { readonly w: number[][]; readonly logdet: number }[]
  /** The discriminant axes `[d][q]` (largest-magnitude entry of each positive), normalised to unit within-class variance. */
  readonly scalings: number[][]
  /** The share of the between-class variance along each of the `k - 1` axes. */
  readonly explainedVarianceRatio: number[]
}

/** What a fitted discriminant offers. */
export interface DiscriminantVerbs extends ScoreClassifierVerbs {
  /** Project onto the discriminant axes after centring on the overall mean. */
  readonly transform: (x: TensorLike) => Tensor
}

interface Whitener {
  readonly w: Float64Array
  readonly r: number
  readonly logdet: number
}

/** `W` with `W W' = C^+` over the eigenvalues of the symmetric matrix `C` above `tol` times the largest. */
function whitener(c: ArrayLike<number>, d: number, tol: number): Whitener {
  const e = eigh(fromStorage(Float64Array.from(c), [d, d], 'float64'))
  const lam = toFlat(primalOf(e.eigenvalues))
  const v = toFlat(primalOf(e.eigenvectors))
  const top = lam.reduce((a, b) => Math.max(a, b), 0)
  const keep = Array.from({ length: d }, (_, i) => i).filter((i) => (lam[i] ?? 0) > tol * top && (lam[i] ?? 0) > 0)
  const w = new Float64Array(d * keep.length)
  keep.forEach((i, col) => {
    const s = 1 / Math.sqrt(lam[i] ?? 1)
    for (let a = 0; a < d; a++) w[a * keep.length + col] = (v[a * d + i] ?? 0) * s
  })
  return { w, r: keep.length, logdet: keep.reduce((acc, i) => acc + Math.log(lam[i] ?? 1), 0) }
}

const nested = (flat: ArrayLike<number>, rows: number, cols: number): number[][] =>
  Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => flat[i * cols + j] ?? 0))

const outer = (a: ArrayLike<number>, b: ArrayLike<number>, scale: number, into: Float64Array): void => {
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++)
      into[i * b.length + j] = (into[i * b.length + j] ?? 0) + scale * (a[i] ?? 0) * (b[j] ?? 0)
}

function shrink(c: Float64Array, d: number, s: number): Float64Array {
  let mu = 0
  for (let i = 0; i < d; i++) mu += (c[i * d + i] ?? 0) / d
  const out = c.map((v) => (1 - s) * v)
  for (let i = 0; i < d; i++) out[i * d + i] = (out[i * d + i] ?? 0) + s * mu
  return out
}

/** The covariance of a class's rows, divided and shrunk as the hyperparameters say. */
function classCovariance(rows: MatrixView, mean: ArrayLike<number>, h: DiscriminantHyper): Float64Array {
  const { n, d, data } = rows
  const divisor = h.unbiased && n > 1 ? n - 1 : n
  const scatter = new Float64Array(d * d)
  const centred = new Float64Array(d)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) centred[j] = (data[i * d + j] ?? 0) - (mean[j] ?? 0)
    outer(centred, centred, 1, scatter)
  }
  if (h.shrinkage === 'auto' && n > 1) {
    // Ledoit-Wolf on the standardised rows, rescaled back (scikit-learn's `_cov`).
    const sd = Float64Array.from({ length: d }, (_, j) => {
      const s = Math.sqrt((scatter[j * d + j] ?? 0) / n)
      return s < 10 * Number.EPSILON ? 1 : s
    })
    const z = new Float64Array(n * d)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) z[i * d + j] = ((data[i * d + j] ?? 0) - (mean[j] ?? 0)) / (sd[j] ?? 1)
    const lw = toFlat(primalOf(ledoitWolf(fromStorage(z, [n, d], 'float64')).covariance))
    return Float64Array.from(
      { length: d * d },
      (_, ij) => ((lw[ij] ?? 0) * (sd[Math.floor(ij / d)] ?? 1) * (sd[ij % d] ?? 1) * n) / divisor,
    )
  }
  const cov = scatter.map((v) => v / divisor)
  return typeof h.shrinkage === 'number' && h.shrinkage > 0 ? shrink(cov, d, h.shrinkage) : cov
}

const regularise = (c: Float64Array, d: number, r: number): Float64Array => {
  if (r === 0) return c
  const out = c.map((v) => (1 - r) * v)
  for (let i = 0; i < d; i++) out[i * d + i] = (out[i * d + i] ?? 0) + r
  return out
}

/** Fit a discriminant analysis to rows and their class codes (the function the estimators run). */
export function fitDiscriminant(
  m: MatrixView,
  codes: ArrayLike<number>,
  classes: readonly number[],
  h: DiscriminantHyper,
): DiscriminantParams {
  const { n, d, data } = m
  const k = classes.length
  const members = Array.from({ length: k }, (_, c) =>
    Array.from({ length: n }, (_, i) => i).filter((i) => codes[i] === c),
  )
  const counts = members.map((rows) => rows.length)
  const seen = counts.map((c) => c > 0)
  const present = seen.filter(Boolean).length
  const means = members.map((rows) =>
    Array.from(
      { length: d },
      (_, j) => rows.reduce((s, i) => s + (data[i * d + j] ?? 0), 0) / Math.max(1, rows.length),
    ),
  )
  const subset = (rows: readonly number[]): MatrixView => ({
    n: rows.length,
    d,
    data: Float64Array.from(rows.flatMap((i) => Array.from(data.subarray(i * d, (i + 1) * d)))),
  })
  const rawPriors = h.priors === undefined ? counts.map((c) => c / n) : [...h.priors]
  const priorSum = rawPriors.reduce((a, b) => a + b, 0)
  const priors = rawPriors.map((p, c) => (seen[c] === true ? p / priorSum : 0))
  const total = priors.reduce((a, b) => a + b, 0)
  const live = priors.map((p) => (total > 0 ? p / total : p))
  const covs = members.map((rows, c) =>
    rows.length === 0 ? new Float64Array(d * d) : classCovariance(subset(rows), means[c] ?? [], h),
  )
  // The pooled covariance: scatter over n - k when unbiased, else the class covariances weighted by the class frequencies.
  const pooledRaw = new Float64Array(d * d)
  const weights = members.map((rows) =>
    h.unbiased && n > present ? Math.max(0, rows.length - 1) / (n - present) : rows.length / n,
  )
  covs.forEach((cv, c) => {
    for (let ij = 0; ij < d * d; ij++) pooledRaw[ij] = (pooledRaw[ij] ?? 0) + (weights[c] ?? 0) * (cv[ij] ?? 0)
  })
  const traceOf = (c: Float64Array): number =>
    Array.from({ length: d }, (_, i) => c[i * d + i] ?? 0).reduce((a, b) => a + b, 0)
  const identity = (): Float64Array =>
    Float64Array.from({ length: d * d }, (_, ij) => (Math.floor(ij / d) === ij % d ? 1 : 0))
  const pooled = traceOf(pooledRaw) > 0 ? regularise(pooledRaw, d, h.regParam) : identity()
  const xbar = Array.from({ length: d }, (_, j) => means.reduce((s, mu, c) => s + (live[c] ?? 0) * (mu[j] ?? 0), 0))
  const pooledW = whitener(pooled, d, h.rankTolerance)
  let coef: number[][] = []
  let intercept: number[] = []
  let whiteners: DiscriminantParams['whiteners'] = []
  if (h.covariance === 'pooled') {
    coef = means.map((mu) => {
      const t = dgemv(pooledW.w, d, pooledW.r, mu, true)
      return Array.from(dgemv(pooledW.w, d, pooledW.r, t))
    })
    intercept = means.map(
      (mu, c) => -0.5 * mu.reduce((s, v, j) => s + v * (coef[c]?.[j] ?? 0), 0) + Math.log(priors[c] ?? 0),
    )
  } else {
    whiteners = covs.map((cv) => {
      const mixed = h.mix === 1 ? cv : cv.map((v, ij) => h.mix * v + (1 - h.mix) * (pooledRaw[ij] ?? 0))
      const base = traceOf(mixed) > 0 ? regularise(mixed, d, h.regParam) : identity()
      const wh = whitener(base, d, h.rankTolerance)
      return { w: nested(wh.w, d, wh.r), logdet: wh.logdet }
    })
  }
  // Fisher's axes: the generalised eigenproblem Sb v = lambda Sw v in the whitened space of the pooled covariance.
  const between = new Float64Array(d * d)
  const diff = new Float64Array(d)
  means.forEach((mu, c) => {
    for (let j = 0; j < d; j++) diff[j] = (mu[j] ?? 0) - (xbar[j] ?? 0)
    outer(diff, diff, live[c] ?? 0, between)
  })
  const r = pooledW.r
  const sbw = dgemm(dgemm(pooledW.w, between, r, d, d, { transposeA: true }), pooledW.w, r, d, r)
  let scalings: number[][] = []
  let explained: number[] = []
  if (r > 0 && present > 1) {
    const e = eigh(fromStorage(sbw, [r, r], 'float64'))
    const lam = toFlat(primalOf(e.eigenvalues))
    const u = toFlat(primalOf(e.eigenvectors))
    const order = Array.from({ length: r }, (_, i) => r - 1 - i)
    const axes = Math.min(present - 1, r)
    const lead = order.slice(0, axes)
    const sum = lead.reduce((s, i) => s + Math.max(0, lam[i] ?? 0), 0)
    explained = lead.map((i) => (sum > 0 ? Math.max(0, lam[i] ?? 0) / sum : 0))
    const q = Math.min(h.nComponents ?? axes, axes)
    const cols = lead.slice(0, q).map((i) => {
      const col = Float64Array.from({ length: r }, (_, a) => u[a * r + i] ?? 0)
      const v = dgemv(pooledW.w, d, r, col)
      let big = 0
      for (let j = 1; j < d; j++) if (Math.abs(v[j] ?? 0) > Math.abs(v[big] ?? 0)) big = j
      return (v[big] ?? 0) < 0 ? v.map((x) => -x) : v
    })
    scalings = Array.from({ length: d }, (_, j) => cols.map((col) => col[j] ?? 0))
  }
  return {
    classes: [...classes],
    priors,
    means,
    xbar,
    covariance: nested(pooled, d, d),
    coef,
    intercept,
    whiteners,
    scalings,
    explainedVarianceRatio: explained,
  }
}

function discriminantVerbs(p: DiscriminantParams): DiscriminantVerbs {
  const k = p.classes.length
  const d = p.xbar.length
  const logPriors = p.priors.map((v) => Math.log(v))
  const scores = (m: MatrixView): Float64Array => {
    const out = new Float64Array(m.n * k)
    if (p.coef.length > 0) {
      for (let i = 0; i < m.n; i++)
        for (let c = 0; c < k; c++) {
          let s = p.intercept[c] ?? 0
          const w = p.coef[c] ?? []
          for (let j = 0; j < d; j++) s += (w[j] ?? 0) * (m.data[i * d + j] ?? 0)
          out[i * k + c] = s
        }
      return out
    }
    for (let c = 0; c < k; c++) {
      const wh = p.whiteners[c]
      const w = Float64Array.from((wh?.w ?? []).flat())
      const r = wh?.w[0]?.length ?? 0
      for (let i = 0; i < m.n; i++) {
        const z = new Float64Array(d)
        for (let j = 0; j < d; j++) z[j] = (m.data[i * d + j] ?? 0) - (p.means[c]?.[j] ?? 0)
        const t = dgemv(w, d, r, z, true)
        out[i * k + c] =
          -0.5 * (t.reduce((s, v) => s + v * v, 0) + (wh?.logdet ?? 0)) + (logPriors[c] ?? Number.NEGATIVE_INFINITY)
      }
    }
    return out
  }
  const base = classifierFromScores(p.classes, d, scores, 'discriminantAnalysis')
  return {
    ...base,
    transform: (x) => {
      const m = matrixView(x, 'discriminantAnalysis', { nFeatures: d })
      const q = p.scalings[0]?.length ?? 0
      const centred = Float64Array.from(m.data, (v, i) => v - (p.xbar[i % d] ?? 0))
      return fromStorage(dgemm(centred, Float64Array.from(p.scalings.flat()), m.n, d, q), [m.n, q], 'float64')
    },
  }
}

const DEFAULTS: DiscriminantHyper = {
  covariance: 'pooled',
  shrinkage: 0,
  regParam: 0,
  mix: 1,
  unbiased: false,
  rankTolerance: 1e-10,
}

/** The classes and codes of a dataset, and the training rows within the first `n`. */
function training(
  data: LearnData,
  n: number,
  who: string,
): { readonly m: MatrixView; readonly codes: Int32Array; readonly classes: readonly number[] } {
  const { classes } = encodeLabels(data.y ?? [], who, rowsOf(data.x))
  const head = prefixOf(data, n)
  const index = new Map(classes.map((c, i) => [c, i]))
  return {
    m: matrixView(head.x, who),
    codes: Int32Array.from(targetVector(head.y ?? [], who), (v) => index.get(v) ?? 0),
    classes,
  }
}

/**
 * Discriminant analysis (`sklearn.discriminant_analysis`): classes are Gaussians, scored by `log prior + log
 * density`. With one pooled covariance the scores are linear in `x` (LDA); with one covariance per class they are
 * quadratic (QDA); `mix` interpolates (Friedman's RDA), `shrinkage` pulls each covariance towards `(trace / d) I`
 * (a fixed intensity or Ledoit-Wolf's) and `regParam` floors the eigenvalues. Singular covariances are inverted on
 * their retained eigenspace. The `transform` verb projects onto Fisher's discriminant axes, the generalised
 * eigenvectors of the between-class against the pooled within-class covariance, and `explainedVarianceRatio` is each
 * axis's share of the between-class variance. The training rows arrive one per step, so a session shows the
 * ellipses growing and `fittedAt(t)` is the discriminant of the first `t` rows.
 *
 * @param hyper - `covariance`, `shrinkage`, `regParam`, `mix`, `unbiased`, `priors`, `nComponents`, `rankTolerance`.
 * @returns The estimator.
 * @reference J. H. Friedman, Regularized discriminant analysis, JASA 84 (1989).
 * @example
 * ```ts
 * import { discriminantAnalysis } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [1, 0.2], [0.2, 1], [4, 4], [5, 4.2], [4.2, 5]]
 * const lda = discriminantAnalysis().fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * toFlat(lda.predict([[0.5, 0.5], [4.5, 4.5]])) // => [0, 1]
 * lda.explainedVarianceRatio // => [1]
 * ```
 */
export function discriminantAnalysis(
  hyper: Partial<DiscriminantHyper> = {},
): IterativeEstimator<
  DiscriminantHyper,
  RowsState,
  DiscriminantParams,
  DiscriminantVerbs & { readonly explainedVarianceRatio: number[] }
> {
  return iterative<
    DiscriminantHyper,
    RowsState,
    DiscriminantParams,
    DiscriminantVerbs & { readonly explainedVarianceRatio: number[] }
  >(
    {
      name: 'discriminant-analysis',
      kind: 'classifier',
      defaults: DEFAULTS,
      hyperSpace: (h) =>
        dict({
          regParam: real(0, 1, { default: h.regParam }),
          covariance: choice(['pooled', 'class'] as const, { default: h.covariance }),
          ...(typeof h.shrinkage === 'number' ? { shrinkage: real(0, 1, { default: h.shrinkage }) } : {}),
          ...(h.covariance === 'class' ? { mix: real(0, 1, { default: h.mix }) } : {}),
        }),
      algorithm: (data) => rowsAlgorithm('discriminant-analysis', rowsOf(data.x)),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data, h) => {
        const t = training(data, s.n, 'discriminantAnalysis')
        return fitDiscriminant(t.m, t.codes, t.classes, h)
      },
      verbs: (p) => ({ ...discriminantVerbs(p), explainedVarianceRatio: p.explainedVarianceRatio }),
    },
    { ...DEFAULTS, ...hyper },
  )
}

/**
 * Linear discriminant analysis: {@link discriminantAnalysis} with one pooled covariance (scikit-learn's
 * `LinearDiscriminantAnalysis`, every solver: the maximum-likelihood pooled covariance; `shrinkage` regularises).
 *
 * @param hyper - As {@link discriminantAnalysis}.
 * @returns The estimator.
 * @example
 * ```ts
 * import { linearDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const f = linearDiscriminant({ shrinkage: 0.1 }).fit({ x: [[0], [1], [5], [6]], y: [0, 0, 1, 1] })
 * toFlat(f.predict([[2], [4]])) // => [0, 1]
 * ```
 */
export function linearDiscriminant(
  hyper: Partial<Omit<DiscriminantHyper, 'covariance'>> = {},
): ReturnType<typeof discriminantAnalysis> {
  return discriminantAnalysis({ ...hyper, covariance: 'pooled' })
}

/**
 * Quadratic discriminant analysis: {@link discriminantAnalysis} with one covariance per class (scikit-learn's
 * `QuadraticDiscriminantAnalysis`; its `reg_param` is `regParam`).
 *
 * @param hyper - As {@link discriminantAnalysis}.
 * @returns The estimator.
 * @example
 * ```ts
 * import { quadraticDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * // Class 1 is spread widely around class 0, so the far points on both sides are class 1.
 * const x = [[-0.2], [0.1], [0], [0.2], [-4], [4], [-5], [5]]
 * const f = quadraticDiscriminant().fit({ x, y: [0, 0, 0, 0, 1, 1, 1, 1] })
 * toFlat(f.predict([[0.05], [-6], [6]])) // => [0, 1, 1]
 * ```
 */
export function quadraticDiscriminant(
  hyper: Partial<Omit<DiscriminantHyper, 'covariance'>> = {},
): ReturnType<typeof discriminantAnalysis> {
  return discriminantAnalysis({ ...hyper, covariance: 'class' })
}

/** The hyperparameters of {@link kernelFisherDiscriminant}. */
export interface KernelFisherHyper {
  /** A kernel of `mdml-engine/kernels`, or a scikit-learn metric name with its parameters. */
  readonly kernel: Kernel | PairwiseKernelMetric
  readonly kernelOptions: PairwiseKernelOptions
  /** The ridge `mu` added to the within-class scatter of the kernel's span. */
  readonly regularisation: number
  /** The number of axes (default `k - 1`). */
  readonly nComponents?: number
}

/** The fitted parameters of the kernel Fisher discriminant: plain JSON. */
export type KernelFisherParams = {
  readonly classes: number[]
  /** The training rows the kernel expansion runs over `[n][d]`. */
  readonly x: number[][]
  /** The expansion coefficients of the axes `[n][q]`. */
  readonly alpha: number[][]
  /** The mean of the projected training rows is subtracted: its value per axis. */
  readonly offset: number[]
  /** The Gaussian classifier in the projected space. */
  readonly projected: DiscriminantParams
}

/** What a fitted kernel Fisher discriminant offers. */
export interface KernelFisherVerbs extends ScoreClassifierVerbs {
  /** The projections `z(x) = sum_i alpha_i k(x_i, x)`, centred on the training mean. */
  readonly transform: (x: TensorLike) => Tensor
}

const FISHER_DEFAULTS: KernelFisherHyper = { kernel: 'rbf', kernelOptions: {}, regularisation: 1e-3 }

function kernelFisherFit(
  m: MatrixView,
  codes: ArrayLike<number>,
  classes: readonly number[],
  h: KernelFisherHyper,
): KernelFisherParams {
  const { n, d } = m
  const k = classes.length
  const rows = fromStorage(m.data, [n, d], 'float64')
  const gram = Float64Array.from(toFlat(primalOf(pairwiseKernels(rows, undefined, h.kernel, h.kernelOptions))))
  const members = Array.from({ length: k }, (_, c) =>
    Array.from({ length: n }, (_, i) => i).filter((i) => codes[i] === c),
  )
  const present = members.filter((r) => r.length > 0).length
  if (present < 2) {
    // One class seen so far: nothing to separate, every row projects to zero.
    const zero = { n, d: 1, data: new Float64Array(n) }
    return {
      classes: [...classes],
      x: nested(m.data, n, d),
      alpha: Array.from({ length: n }, () => [0]),
      offset: [0],
      projected: fitDiscriminant(zero, codes, classes, { ...DEFAULTS, covariance: 'pooled' }),
    }
  }
  // The class mean maps M_c = K 1_c / n_c, the overall M = K 1 / n.
  const meanMap = (idx: readonly number[]): Float64Array =>
    Float64Array.from(
      { length: n },
      (_, i) => idx.reduce((s, j) => s + (gram[i * n + j] ?? 0), 0) / Math.max(1, idx.length),
    )
  const overall = meanMap(Array.from({ length: n }, (_, i) => i))
  const between = new Float64Array(n * n)
  const within = new Float64Array(n * n)
  members.forEach((idx) => {
    if (idx.length === 0) return
    const mc = meanMap(idx)
    const diff = mc.map((v, i) => v - (overall[i] ?? 0))
    outer(diff, diff, idx.length, between)
    // K_c (I - 11'/n_c) K_c'
    const centred = new Float64Array(n * idx.length)
    for (let i = 0; i < n; i++) {
      const rowMean = idx.reduce((s, j) => s + (gram[i * n + j] ?? 0), 0) / idx.length
      idx.forEach((j, col) => {
        centred[i * idx.length + col] = (gram[i * n + j] ?? 0) - rowMean
      })
    }
    const part = dgemm(centred, centred, n, idx.length, n, { transposeB: true })
    for (let ij = 0; ij < n * n; ij++) within[ij] = (within[ij] ?? 0) + (part[ij] ?? 0)
  })
  for (let i = 0; i < n; i++) within[i * n + i] = (within[i * n + i] ?? 0) + h.regularisation
  const wh = whitener(within, n, 1e-12)
  const r = wh.r
  const sbw = dgemm(dgemm(wh.w, between, r, n, n, { transposeA: true }), wh.w, r, n, r)
  const e = eigh(fromStorage(sbw, [r, r], 'float64'))
  const u = toFlat(primalOf(e.eigenvectors))
  const q = Math.max(1, Math.min(h.nComponents ?? present - 1, present - 1, r))
  const alphaCols = Array.from({ length: q }, (_, a) => {
    const col = Float64Array.from({ length: r }, (_, i) => u[i * r + (r - 1 - a)] ?? 0)
    return dgemv(wh.w, n, r, col)
  })
  const alpha = Array.from({ length: n }, (_, i) => alphaCols.map((col) => col[i] ?? 0))
  const flatAlpha = Float64Array.from(alpha.flat())
  const z = dgemm(gram, flatAlpha, n, n, q)
  const offset = Array.from({ length: q }, (_, a) => {
    let s = 0
    for (let i = 0; i < n; i++) s += (z[i * q + a] ?? 0) / n
    return s
  })
  const centredZ = z.map((v, i) => v - (offset[i % q] ?? 0))
  const projected = fitDiscriminant({ n, d: q, data: centredZ }, codes, classes, {
    ...DEFAULTS,
    covariance: 'pooled',
    unbiased: false,
  })
  return { classes: [...classes], x: nested(m.data, n, d), alpha, offset, projected }
}

function kernelFisherVerbs(p: KernelFisherParams, h: KernelFisherHyper): KernelFisherVerbs {
  const d = p.x[0]?.length ?? 0
  const q = p.offset.length
  const train = fromStorage(Float64Array.from(p.x.flat()), [p.x.length, d], 'float64')
  const alpha = Float64Array.from(p.alpha.flat())
  const project = (x: TensorLike): MatrixView => {
    const m = matrixView(x, 'kernelFisherDiscriminant', { nFeatures: d })
    const cross = Float64Array.from(
      toFlat(primalOf(pairwiseKernels(fromStorage(m.data, [m.n, d], 'float64'), train, h.kernel, h.kernelOptions))),
    )
    const z = dgemm(cross, alpha, m.n, p.x.length, q)
    return { n: m.n, d: q, data: z.map((v, i) => v - (p.offset[i % q] ?? 0)) }
  }
  const classifier = classifierFromScores(
    p.classes,
    d,
    (m) => {
      const z = project(fromStorage(m.data, [m.n, d], 'float64'))
      return scoreProjected(p.projected, z)
    },
    'kernelFisherDiscriminant',
  )
  return {
    ...classifier,
    transform: (x) => {
      const z = project(x)
      return fromStorage(z.data, [z.n, q], 'float64')
    },
  }
}

function scoreProjected(p: DiscriminantParams, z: MatrixView): Float64Array {
  const k = p.classes.length
  const d = z.d
  const out = new Float64Array(z.n * k)
  for (let i = 0; i < z.n; i++)
    for (let c = 0; c < k; c++) {
      let s = p.intercept[c] ?? 0
      const w = p.coef[c] ?? []
      for (let j = 0; j < d; j++) s += (w[j] ?? 0) * (z.data[i * d + j] ?? 0)
      out[i * k + c] = s
    }
  return out
}

/**
 * The kernel Fisher discriminant (Mika et al.): Fisher's discriminant in the feature space of a kernel. The axes are
 * expansions `z(x) = sum_i alpha_i k(x_i, x)` whose coefficients solve `(N + mu I)^-1 B alpha = lambda alpha`, with `B`
 * the between-class and `N = sum_c K_c (I - 11'/n_c) K_c'` the within-class scatter of the kernel's span; `mu`
 * (the regularisation) keeps the within-class scatter invertible. Classes are then Gaussians with a pooled covariance
 * in the projected space. With a linear kernel and `mu -> 0` the axes are the linear discriminant's.
 *
 * @param hyper - `kernel`, `kernelOptions`, `regularisation`, `nComponents`.
 * @returns The estimator.
 * @reference S. Mika, G. Ratsch, J. Weston, B. Scholkopf and K.-R. Muller, Fisher discriminant analysis with kernels,
 * Neural Networks for Signal Processing IX (1999).
 * @example
 * ```ts
 * import { kernelFisherDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * // Two concentric classes, inseparable by a line.
 * const x = [[0.1, 0], [0, 0.2], [-0.1, 0], [0, -0.2], [2, 0], [0, 2.1], [-2, 0], [0, -2.1]]
 * const f = kernelFisherDiscriminant({ kernelOptions: { gamma: 0.5 } }).fit({ x, y: [0, 0, 0, 0, 1, 1, 1, 1] })
 * toFlat(f.predict([[0.05, 0.05], [1.9, 0.3]])) // => [0, 1]
 * ```
 */
export function kernelFisherDiscriminant(
  hyper: Partial<KernelFisherHyper> = {},
): IterativeEstimator<KernelFisherHyper, RowsState, KernelFisherParams, KernelFisherVerbs> {
  return iterative<KernelFisherHyper, RowsState, KernelFisherParams, KernelFisherVerbs>(
    {
      name: 'kernel-fisher-discriminant',
      kind: 'classifier',
      defaults: FISHER_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          regularisation: real(Math.min(1e-6, h.regularisation), Math.max(10, h.regularisation), {
            default: h.regularisation,
            log: true,
          }),
          ...(h.nComponents === undefined
            ? {}
            : { nComponents: int(1, Math.max(10, h.nComponents), { default: h.nComponents }) }),
        }),
      algorithm: (data) => rowsAlgorithm('kernel-fisher-discriminant', rowsOf(data.x)),
      maxSteps: (_h, data) => rowsOf(data.x),
      logs: { 'solver/rows': Log() },
      params: (s, data, h) => {
        const t = training(data, s.n, 'kernelFisherDiscriminant')
        return kernelFisherFit(t.m, t.codes, t.classes, h)
      },
      verbs: kernelFisherVerbs,
    },
    { ...FISHER_DEFAULTS, ...hyper },
  )
}
