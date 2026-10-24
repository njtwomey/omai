import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { coefficientTable, componentsRecord, type Exposition } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type Fitted,
  featureNamesOf,
  type LearnData,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { eigh, pinv, slogdet } from '../../lib/numerics/linalg/index.ts'
import { choice, dict, real, type Space } from '../../lib/space/index.ts'
import { empiricalCovariance, ledoitWolf, shrunkCovariance } from '../../lib/stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'

const flat = (v: Parameters<typeof toFlat>[0]): number[] => toFlat(primalOf(v))
const matrix = (data: ArrayLike<number>, r: number, c: number): Tensor =>
  fromStorage(Float64Array.from(data), [r, c], 'float64')

/** Class labels, codes, priors, means and rows. */
interface ClassStats {
  readonly classes: readonly number[]
  readonly codes: Int32Array
  readonly priors: number[]
  readonly means: number[][]
  readonly rows: number[][]
}

function classStats(m: MatrixView, y: TensorLike, priors: readonly number[] | undefined, who: string): ClassStats {
  const { classes, codes } = encodeLabels(y, who, m.n)
  const k = classes.length
  if (k < 2) throw new WhiteboxError('value', `${who}: needs at least two classes`)
  const rows = classes.map((_, c) => Array.from(codes.keys()).filter((i) => codes[i] === c))
  const means = rows.map((rs) =>
    Array.from({ length: m.d }, (_, j) => rs.reduce((a, i) => a + (m.data[i * m.d + j] ?? 0), 0) / rs.length),
  )
  return { classes, codes, priors: priors === undefined ? rows.map((rs) => rs.length / m.n) : [...priors], means, rows }
}

const rowsMatrix = (m: MatrixView, rows: readonly number[]): Float64Array => {
  const out = new Float64Array(rows.length * m.d)
  rows.forEach((r, i) => {
    out.set(m.data.subarray(r * m.d, (r + 1) * m.d), i * m.d)
  })
  return out
}

/** A class covariance as scikit-learn's `_cov`: empirical (biased), Ledoit-Wolf on standardised data, or shrunk. */
function classCovariance(m: MatrixView, rows: readonly number[], shrinkage: number | 'auto' | undefined): number[] {
  const x = matrix(rowsMatrix(m, rows), rows.length, m.d)
  if (shrinkage === undefined) return flat(empiricalCovariance(x))
  if (shrinkage === 'auto') {
    const data = rowsMatrix(m, rows)
    const n = rows.length
    const scale = Array.from({ length: m.d }, (_, j) => {
      let s = 0
      let s2 = 0
      for (let i = 0; i < n; i++) {
        s += data[i * m.d + j] ?? 0
        s2 += (data[i * m.d + j] ?? 0) ** 2
      }
      const v = s2 / n - (s / n) ** 2
      return v > 0 ? Math.sqrt(v) : 1
    })
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m.d; j++) data[i * m.d + j] = (data[i * m.d + j] ?? 0) / (scale[j] ?? 1)
    const s = flat(ledoitWolf(matrix(data, n, m.d)).covariance)
    return s.map((v, idx) => v * (scale[Math.floor(idx / m.d)] ?? 1) * (scale[idx % m.d] ?? 1))
  }
  return flat(shrunkCovariance(x, { shrinkage }).covariance)
}

/** Hyperparameters of {@link linearDiscriminantAnalysis}. */
export interface LdaHyper {
  /** 'svd' (scikit-learn's default: the pooled within-class covariance) or 'lsqr' (prior-weighted class covariances, with shrinkage). */
  readonly solver: 'svd' | 'lsqr'
  /** Covariance shrinkage for 'lsqr': a value in [0, 1] or Ledoit-Wolf ('auto'). */
  readonly shrinkage?: number | 'auto'
  readonly priors?: readonly number[]
  /** Discriminant directions kept by `transform` (default min(K - 1, d)). */
  readonly nComponents?: number
}

/** Parameters of a fitted linear discriminant. */
export type LdaParams = {
  readonly classes: number[]
  readonly coef: number[][]
  readonly intercept: number[]
  readonly means: number[][]
  readonly xbar: number[]
  /** Discriminant directions `[d][components]`. */
  readonly scalings: number[][]
  readonly explainedVarianceRatio: number[]
}

/** Verbs of a fitted linear discriminant. */
export interface LdaVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** Project onto the discriminant directions (Fisher's LDA as a transformer). */
  readonly transform: (x: TensorLike) => Tensor
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted linear discriminant. */
export type FittedLda = Fitted<LdaParams> & LdaVerbs

function softmaxRows(scores: Float64Array, m: number, k: number): Float64Array {
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    let max = Number.NEGATIVE_INFINITY
    for (let c = 0; c < k; c++) max = Math.max(max, scores[i * k + c] ?? 0)
    let z = 0
    for (let c = 0; c < k; c++) z += Math.exp((scores[i * k + c] ?? 0) - max)
    for (let c = 0; c < k; c++) out[i * k + c] = Math.exp((scores[i * k + c] ?? 0) - max) / z
  }
  return out
}

function ldaVerbs(p: LdaParams): LdaVerbs {
  const d = p.xbar.length
  const k = p.classes.length
  const scores = (x: TensorLike): { m: number; s: Float64Array } => {
    const v = matrixView(x, 'lda', { nFeatures: d })
    const s = new Float64Array(v.n * k)
    for (let i = 0; i < v.n; i++)
      for (let c = 0; c < k; c++) {
        let t = p.intercept[c] ?? 0
        for (let j = 0; j < d; j++) t += (v.data[i * d + j] ?? 0) * (p.coef[c]?.[j] ?? 0)
        s[i * k + c] = t
      }
    return { m: v.n, s }
  }
  const proba = (x: TensorLike): Tensor => {
    const { m, s } = scores(x)
    return fromStorage(softmaxRows(s, m, k), [m, k], 'float64')
  }
  return {
    predict: (x) => {
      const { m, s } = scores(x)
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(s.subarray(i * k, (i + 1) * k))] ?? 0),
        [m],
        'float64',
      )
    },
    predictProba: proba,
    predictDistribution: (x) => categorical({ probs: proba(x) }),
    decisionFunction: (x) => {
      const { m, s } = scores(x)
      if (k === 2)
        return fromStorage(
          Float64Array.from({ length: m }, (_, i) => (s[i * 2 + 1] ?? 0) - (s[i * 2] ?? 0)),
          [m],
          'float64',
        )
      return fromStorage(s, [m, k], 'float64')
    },
    transform: (x) => {
      const v = matrixView(x, 'lda', { nFeatures: d })
      const q = p.scalings[0]?.length ?? 0
      const out = new Float64Array(v.n * q)
      for (let i = 0; i < v.n; i++)
        for (let a = 0; a < q; a++) {
          let t = 0
          for (let j = 0; j < d; j++) t += ((v.data[i * d + j] ?? 0) - (p.xbar[j] ?? 0)) * (p.scalings[j]?.[a] ?? 0)
          out[i * q + a] = t
        }
      return fromStorage(out, [v.n, q], 'float64')
    },
    classes: p.classes,
    nFeaturesIn: d,
  }
}

/**
 * Linear discriminant analysis (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): Gaussian classes sharing
 * one covariance, so the discriminants are linear; with shrinkage (a value or Ledoit-Wolf) for the 'lsqr' solver; and
 * Fisher's discriminant directions (generalised eigenvectors of the between- and within-class scatter) for
 * `transform`. Exposition: the coefficient table (per class), the discriminant directions with their explained
 * variance ratios and the training scores, and for two features the decision surface.
 *
 * @param hyper - Solver (default svd), shrinkage, priors and components.
 * @returns The estimator.
 */
export function linearDiscriminantAnalysis(hyper: Partial<LdaHyper> = {}): ExplainedEstimator<LdaHyper, FittedLda> {
  const est: ExplainedEstimator<LdaHyper, FittedLda> = defineExplainedEstimator(
    {
      name: 'linear-discriminant-analysis',
      kind: 'classifier',
      hyperSpace: (h: LdaHyper): Space =>
        dict({
          solver: choice(['svd', 'lsqr'], {
            label: 'solver',
            description: 'How the discriminant directions are computed.',
            default: h.solver,
          }),
          shrinkage: real(0, 1, {
            label: 'shrinkage',
            description: 'How far the covariance is pulled towards a scaled identity (0 none, 1 all the way).',
            default: typeof h.shrinkage === 'number' ? h.shrinkage : 0,
            when: { key: 'solver', in: ['lsqr'] },
          }),
        }),
      fit: (data: LearnData, h: LdaHyper) => {
        const m = matrixView(data.x, 'linearDiscriminantAnalysis')
        const st = classStats(m, data.y ?? [], h.priors, 'linearDiscriminantAnalysis')
        const k = st.classes.length
        const d = m.d
        if (h.solver === 'svd' && h.shrinkage !== undefined)
          throw new WhiteboxError('value', 'linearDiscriminantAnalysis: shrinkage needs the lsqr solver')
        let sigma: number[]
        if (h.solver === 'svd') {
          // The pooled within-class covariance (the within-class scatter / n).
          sigma = new Array<number>(d * d).fill(0)
          for (let i = 0; i < m.n; i++) {
            const mu = st.means[st.codes[i] ?? 0] ?? []
            for (let a = 0; a < d; a++)
              for (let b = 0; b < d; b++)
                sigma[a * d + b] =
                  (sigma[a * d + b] ?? 0) +
                  ((m.data[i * d + a] ?? 0) - (mu[a] ?? 0)) * ((m.data[i * d + b] ?? 0) - (mu[b] ?? 0))
          }
          sigma = sigma.map((v) => v / m.n)
        } else {
          sigma = new Array<number>(d * d).fill(0)
          for (const [c, rows] of st.rows.entries()) {
            const cov = classCovariance(m, rows, h.shrinkage)
            for (let i = 0; i < d * d; i++) sigma[i] = (sigma[i] ?? 0) + (st.priors[c] ?? 0) * (cov[i] ?? 0)
          }
        }
        const inv = flat(pinv(matrix(sigma, d, d)))
        const xbar = Array.from({ length: d }, (_, j) =>
          st.means.reduce((a, mu, c) => a + (st.priors[c] ?? 0) * (mu[j] ?? 0), 0),
        )
        const centreOn = h.solver === 'svd' ? xbar : new Array<number>(d).fill(0)
        const coef = st.means.map((mu) =>
          Array.from({ length: d }, (_, a) => {
            let s = 0
            for (let b = 0; b < d; b++) s += (inv[a * d + b] ?? 0) * ((mu[b] ?? 0) - (centreOn[b] ?? 0))
            return s
          }),
        )
        const intercept = st.means.map((mu, c) => {
          const row = coef[c] ?? []
          let quad = 0
          for (let j = 0; j < d; j++) quad += ((mu[j] ?? 0) - (centreOn[j] ?? 0)) * (row[j] ?? 0)
          let shift = 0
          for (let j = 0; j < d; j++) shift += (centreOn[j] ?? 0) * (row[j] ?? 0)
          return -0.5 * quad + Math.log(st.priors[c] ?? 1) - shift
        })
        // Fisher's directions: eigenvectors of Sw^{-1/2} Sb Sw^{-1/2}, mapped back by Sw^{-1/2}.
        const sb = new Array<number>(d * d).fill(0)
        for (const [c, mu] of st.means.entries())
          for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++)
              sb[a * d + b] =
                (sb[a * d + b] ?? 0) +
                (st.priors[c] ?? 0) * ((mu[a] ?? 0) - (xbar[a] ?? 0)) * ((mu[b] ?? 0) - (xbar[b] ?? 0))
        const ew = eigh(matrix(sigma, d, d))
        const wv = flat(ew.eigenvalues)
        const wvec = flat(ew.eigenvectors)
        const half = new Array<number>(d * d).fill(0)
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) {
            let s = 0
            for (let q = 0; q < d; q++) {
              const lam = wv[q] ?? 0
              if (lam > 1e-12) s += ((wvec[a * d + q] ?? 0) * (wvec[b * d + q] ?? 0)) / Math.sqrt(lam)
            }
            half[a * d + b] = s
          }
        const mid = new Array<number>(d * d).fill(0)
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) {
            let s = 0
            for (let p = 0; p < d; p++)
              for (let q = 0; q < d; q++) s += (half[a * d + p] ?? 0) * (sb[p * d + q] ?? 0) * (half[q * d + b] ?? 0)
            mid[a * d + b] = s
          }
        const eb = eigh(matrix(mid, d, d))
        const vals = flat(eb.eigenvalues)
        const vecs = flat(eb.eigenvectors)
        const order = vals.map((_, i) => i).sort((a, b) => (vals[b] ?? 0) - (vals[a] ?? 0))
        const nComp = Math.min(h.nComponents ?? Math.min(k - 1, d), Math.min(k - 1, d))
        const kept = order.slice(0, nComp)
        const scalings = Array.from({ length: d }, (_, a) =>
          kept.map((q) => {
            let s = 0
            for (let b = 0; b < d; b++) s += (half[a * d + b] ?? 0) * (vecs[b * d + q] ?? 0)
            return s
          }),
        )
        const totalVal = order.slice(0, Math.min(k - 1, d)).reduce((a, q) => a + Math.max(vals[q] ?? 0, 0), 0)
        const params: LdaParams = {
          classes: [...st.classes],
          coef,
          intercept,
          means: st.means,
          xbar,
          scalings,
          explainedVarianceRatio: kept.map((q) => Math.max(vals[q] ?? 0, 0) / (totalVal || 1)),
        }
        const fitted = est.revive(params)
        const features = featureNamesOf(data, d)
        const classNames = classNamesOf(data, st.classes)
        const scores = fitted.transform(data.x)
        const exposition: Exposition[] = [
          coefficientTable({
            names: classNames.flatMap((c) => [...features.map((f) => `${c}:${f}`), `${c}:intercept`]),
            estimates: coef.flatMap((row, c) => [...row, intercept[c] ?? 0]),
          }),
          componentsRecord({
            method: 'lda',
            features,
            components: kept.map((_, a) => scalings.map((row) => row[a] ?? 0)),
            explainedVarianceRatio: params.explainedVarianceRatio,
            scores: Array.from({ length: m.n }, (_, i) => Array.from(scores.data.subarray(i * nComp, (i + 1) * nComp))),
          }),
        ]
        if (d === 2) exposition.push(decisionSurfaceOf(fitted, data.x, { featureNames: features, classNames }))
        return { params, exposition }
      },
      methods: (p: LdaParams) => ldaVerbs(p),
    },
    {
      solver: hyper.solver ?? 'svd',
      ...(hyper.shrinkage === undefined ? {} : { shrinkage: hyper.shrinkage }),
      ...(hyper.priors === undefined ? {} : { priors: hyper.priors }),
      ...(hyper.nComponents === undefined ? {} : { nComponents: hyper.nComponents }),
    },
  )
  return est
}

/** Hyperparameters of {@link quadraticDiscriminantAnalysis}. */
export interface QdaHyper {
  /** Shrink each class covariance towards the identity: (1 - r) Sigma_k + r I (default 0). */
  readonly regParam: number
  readonly priors?: readonly number[]
}

/** Parameters of a fitted quadratic discriminant. */
export type QdaParams = {
  readonly classes: number[]
  readonly priors: number[]
  readonly means: number[][]
  /** Inverse class covariances, row-major. */
  readonly precisions: number[][]
  readonly logDets: number[]
}

/** Verbs of a fitted quadratic discriminant. */
export interface QdaVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** Per-class log posteriors up to a constant, `[m, k]`. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted quadratic discriminant. */
export type FittedQda = Fitted<QdaParams> & QdaVerbs

/**
 * Quadratic discriminant analysis (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis): a Gaussian per class
 * with its own (maximum-likelihood, optionally regularised) covariance, so the boundaries are quadratic. Exposition: the class
 * means as a table and, for two features, the decision surface.
 *
 * @param hyper - Regularisation (default 0) and priors.
 * @returns The estimator.
 */
export function quadraticDiscriminantAnalysis(hyper: Partial<QdaHyper> = {}): ExplainedEstimator<QdaHyper, FittedQda> {
  const est: ExplainedEstimator<QdaHyper, FittedQda> = defineExplainedEstimator(
    {
      name: 'quadratic-discriminant-analysis',
      kind: 'classifier',
      hyperSpace: (h: QdaHyper): Space =>
        dict({ regParam: real(0, 1, { default: h.regParam, label: 'regularisation' }) }),
      fit: (data: LearnData, h: QdaHyper) => {
        const m = matrixView(data.x, 'quadraticDiscriminantAnalysis')
        const st = classStats(m, data.y ?? [], h.priors, 'quadraticDiscriminantAnalysis')
        const d = m.d
        const covs = st.rows.map((rows, c) => {
          if (rows.length < 2)
            throw new WhiteboxError('value', 'quadraticDiscriminantAnalysis: each class needs two rows')
          const mu = st.means[c] ?? []
          const cov = new Array<number>(d * d).fill(0)
          for (const i of rows)
            for (let a = 0; a < d; a++)
              for (let b = 0; b < d; b++)
                cov[a * d + b] =
                  (cov[a * d + b] ?? 0) +
                  ((m.data[i * d + a] ?? 0) - (mu[a] ?? 0)) * ((m.data[i * d + b] ?? 0) - (mu[b] ?? 0))
          return cov.map(
            (v, idx) => (1 - h.regParam) * (v / rows.length) + (Math.floor(idx / d) === idx % d ? h.regParam : 0),
          )
        })
        const params: QdaParams = {
          classes: [...st.classes],
          priors: st.priors,
          means: st.means,
          precisions: covs.map((cov) => flat(pinv(matrix(cov, d, d)))),
          logDets: covs.map((cov) => toFlat(primalOf(slogdet(matrix(cov, d, d)).logabsdet))[0] ?? 0),
        }
        const fitted = est.revive(params)
        const features = featureNamesOf(data, d)
        const classNames = classNamesOf(data, st.classes)
        const exposition: Exposition[] = [
          coefficientTable({
            names: classNames.flatMap((c) => features.map((f) => `${c}:${f}`)),
            estimates: st.means.flat(),
          }),
        ]
        if (d === 2) exposition.push(decisionSurfaceOf(fitted, data.x, { featureNames: features, classNames }))
        return { params, exposition }
      },
      methods: (p: QdaParams): QdaVerbs => {
        const d = p.means[0]?.length ?? 0
        const k = p.classes.length
        const scores = (x: TensorLike): { m: number; s: Float64Array } => {
          const v = matrixView(x, 'qda', { nFeatures: d })
          const s = new Float64Array(v.n * k)
          for (let i = 0; i < v.n; i++)
            for (let c = 0; c < k; c++) {
              const mu = p.means[c] ?? []
              const pr = p.precisions[c] ?? []
              let q = 0
              for (let a = 0; a < d; a++)
                for (let b = 0; b < d; b++)
                  q +=
                    ((v.data[i * d + a] ?? 0) - (mu[a] ?? 0)) *
                    (pr[a * d + b] ?? 0) *
                    ((v.data[i * d + b] ?? 0) - (mu[b] ?? 0))
              s[i * k + c] = -0.5 * ((p.logDets[c] ?? 0) + q) + Math.log(p.priors[c] ?? 1)
            }
          return { m: v.n, s }
        }
        const proba = (x: TensorLike): Tensor => {
          const { m, s } = scores(x)
          return fromStorage(softmaxRows(s, m, k), [m, k], 'float64')
        }
        return {
          predict: (x) => {
            const { m, s } = scores(x)
            return fromStorage(
              Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(s.subarray(i * k, (i + 1) * k))] ?? 0),
              [m],
              'float64',
            )
          },
          predictProba: proba,
          predictDistribution: (x) => categorical({ probs: proba(x) }),
          decisionFunction: (x) => {
            const { m, s } = scores(x)
            return fromStorage(s, [m, k], 'float64')
          },
          classes: p.classes,
          nFeaturesIn: d,
        }
      },
    },
    { regParam: hyper.regParam ?? 0, ...(hyper.priors === undefined ? {} : { priors: hyper.priors }) },
  )
  return est
}

/** {@link linearDiscriminantAnalysis} as a class. */
export class LinearDiscriminantAnalysis extends ExplainedEstimatorClass<LdaHyper, FittedLda> {
  /** @param hyper - As {@link linearDiscriminantAnalysis}. */
  constructor(hyper: Partial<LdaHyper> = {}) {
    super(linearDiscriminantAnalysis(hyper))
  }
}

/** {@link quadraticDiscriminantAnalysis} as a class. */
export class QuadraticDiscriminantAnalysis extends ExplainedEstimatorClass<QdaHyper, FittedQda> {
  /** @param hyper - As {@link quadraticDiscriminantAnalysis}. */
  constructor(hyper: Partial<QdaHyper> = {}) {
    super(quadraticDiscriminantAnalysis(hyper))
  }
}
