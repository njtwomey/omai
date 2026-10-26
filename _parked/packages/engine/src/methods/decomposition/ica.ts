/**
 * Independent component analysis by FastICA (Hyvärinen and Oja, 2000; as in scikit-learn): unmixing observed signals
 * into statistically independent sources by whitening the data and then running a fixed-point iteration that maximises
 * non-Gaussianity, traced step by step. The fit function, a class, and the symmetric decorrelation it relies on.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { componentsRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { foldIn, type Key, normal } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, toArray, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { centred, columnMeans, mm, pinvRows, project, reconstruct, rowsOf, thinSvd, tr } from './_linear.ts'
import type { ProjectionVerbs } from './pca.ts'

/** The contrast function G of FastICA: its derivative g and the mean of g' per component. */
export type IcaContrast = 'logcosh' | 'exp' | 'cube'

function contrast(fun: IcaContrast, wx: readonly number[]): { g: number[]; gPrimeMean: number } {
  let s = 0
  const g = wx.map((v) => {
    if (fun === 'logcosh') {
      const t = Math.tanh(v)
      s += 1 - t * t
      return t
    }
    if (fun === 'exp') {
      const e = Math.exp(-(v * v) / 2)
      s += (1 - v * v) * e
      return v * e
    }
    s += 3 * v * v
    return v * v * v
  })
  return { g, gPrimeMean: s / wx.length }
}

/**
 * Symmetric decorrelation (W W^T)^(-1/2) W, by the eigendecomposition of `numerics/linalg`.
 *
 * @param w - A square matrix `[k][k]`.
 * @returns The decorrelated matrix.
 */
export function symmetricDecorrelation(w: readonly (readonly number[])[]): number[][] {
  const { eigenvalues, eigenvectors } = linalg.eigh(mm(w, tr(w)))
  const s = toFlat(eigenvalues).map((v) => Math.max(v, Number.MIN_VALUE))
  const u = toArray(eigenvectors) as number[][]
  const scaled = u.map((row) => row.map((v, c) => v / Math.sqrt(s[c] ?? 1)))
  return mm(mm(scaled, tr(u)), w)
}

/** The state of parallel FastICA: the unmixing matrix of the whitened data and the last change. */
export interface FastIcaState {
  readonly w: number[][]
  readonly change: number
}

/**
 * Parallel FastICA (Hyvarinen and Oja 2000; sklearn's `_ica_par`) as a traced algorithm: each step applies the
 * fixed-point update W <- E[g(W x) x^T] - diag(E[g'(W x)]) W to all components at once, then decorrelates
 * symmetrically, until max | |diag(W_new W^T)| - 1 | < tol.
 *
 * @param z - Whitened data `[k][n]` (components by samples).
 * @param init - The initial unmixing matrix `[k][k]`.
 * @param options - The contrast (logcosh) and tolerance (1e-4).
 * @returns The algorithm.
 */
export function fastIcaAlgorithm(
  z: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: { readonly fun?: IcaContrast; readonly tol?: number } = {},
): Algorithm<FastIcaState> {
  const fun = options.fun ?? 'logcosh'
  const tol = options.tol ?? 1e-4
  const n = z[0]?.length ?? 0
  return {
    name: 'fast-ica',
    config: { fun, tol },
    init: () => ({ w: symmetricDecorrelation(init), change: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const wx = mm(s.w, z)
      const parts = wx.map((row) => contrast(fun, row))
      const gx = parts.map((p) => p.g)
      const update = mm(gx, tr(z)).map((row, c) =>
        row.map((v, j) => v / n - (parts[c]?.gPrimeMean ?? 0) * (s.w[c]?.[j] ?? 0)),
      )
      const w = symmetricDecorrelation(update)
      let change = 0
      for (const [c, row] of w.entries())
        change = Math.max(change, Math.abs(Math.abs(row.reduce((a, v, j) => a + v * (s.w[c]?.[j] ?? 0), 0)) - 1))
      return { w, change }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.change) ? { change: s.change } : {} }),
  }
}

/** Hyperparameters of FastICA. */
export type FastIcaHyper = {
  readonly nComponents?: number
  readonly fun?: IcaContrast
  readonly tol?: number
  readonly maxIter?: number
  /** The initial unmixing matrix `[k][k]` (default standard normal draws). */
  readonly wInit?: readonly (readonly number[])[]
}

/** The parameters of a fitted ICA: the mean, the unmixing components and the mixing matrix. */
export type IcaParams = {
  readonly mean: number[]
  /** components[k][f]: the unmixing rows (sources = (x - mean) components^T). */
  readonly components: number[][]
  /** mixing[f][k]: the pseudo-inverse of the components. */
  readonly mixing: number[][]
  readonly nIter: number
}

/** A fitted ICA. */
export type FittedIca = ExplainedFitted<IcaParams, Omit<ProjectionVerbs, 'score'>>

/**
 * FastICA (sklearn.decomposition.FastICA with `whiten='unit-variance'`, the parallel algorithm): whitening by the SVD
 * of the centred data, the traced fixed-point iteration, and sources scaled to unit variance. Its exposition is a
 * `components` record with the unmixing rows and the recovered sources as scores. TODO: deflation, Infomax.
 *
 * @param hyper - Components, contrast, tolerance, iteration cap and an optional initial unmixing matrix.
 * @returns The estimator.
 */
export function fastIca(hyper: FastIcaHyper = {}): Estimator<FastIcaHyper, FittedIca> {
  return defineExplainedEstimator<FastIcaHyper, IcaParams, Omit<ProjectionVerbs, 'score'>>(
    {
      name: 'fast-ica',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          fun: choice(['logcosh', 'exp', 'cube'], {
            label: 'contrast function',
            description: 'The non-Gaussianity measure: log cosh, exponential or cube.',
            default: h.fun ?? 'logcosh',
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'fastIca', { minSamples: 2 })
        const k = h.nComponents ?? Math.min(m.n, m.d)
        if (!(Number.isInteger(k) && k >= 1 && k <= Math.min(m.n, m.d)))
          throw new MdmlError('value', `fastIca: nComponents must be in 1..${Math.min(m.n, m.d)}`)
        const mean = columnMeans(m)
        const xc = centred(m, mean)
        const { s, vt } = thinSvd(xc, m.n, m.d)
        // sklearn's whitening: u = V with the first row positive, K = (u / d)^T, z = K x^T sqrt(n).
        const signs = vt.map((row) => Math.sign(row[0] ?? 1) || 1)
        const whitening = vt.slice(0, k).map((row, c) => row.map((v) => (v * (signs[c] ?? 1)) / (s[c] ?? 1)))
        const xRows = rowsOf(fromFlat(xc, m.n, m.d))
        const z = mm(whitening, tr(xRows)).map((row) => row.map((v) => v * Math.sqrt(m.n)))
        const key = fitRunOptions(options, 'fast-ica').key as Key
        const wInit = h.wInit?.map((r) => [...r]) ?? (toArray(normal(foldIn(key, 0), [k, k])) as number[][])
        const out = runAlgorithm(fastIcaAlgorithm(z, wInit, { fun: h.fun ?? 'logcosh', tol: h.tol ?? 1e-4 }), {
          ...fitRunOptions(options, 'fast-ica'),
          key: foldIn(key, 1),
          maxSteps: h.maxIter ?? 200,
        })
        let components = mm(out.state.w, whitening)
        const sources = mm(xRows, tr(components))
        const sd = Array.from({ length: k }, (_, c) => {
          const col = sources.map((r) => r[c] ?? 0)
          const mu = col.reduce((a, v) => a + v, 0) / col.length
          return Math.sqrt(col.reduce((a, v) => a + (v - mu) ** 2, 0) / col.length)
        })
        components = components.map((row, c) => row.map((v) => v / (sd[c] ?? 1)))
        const params: IcaParams = { mean, components, mixing: pinvRows(components), nIter: out.steps }
        return {
          params,
          exposition: [
            componentsRecord({
              method: 'fastIca',
              features: featureNamesOf(data, m.d),
              components,
              scores: rowsOf(project(data.x, mean, components)),
            }),
          ],
        }
      },
      methods: (p) => ({
        transform: (x) => project(x, p.mean, p.components),
        inverseTransform: (z) => reconstruct(z, p.mean, tr(p.mixing)),
        nFeaturesIn: p.mean.length,
        featureNamesOut: () => p.components.map((_, c) => `fastica${c}`),
      }),
    },
    hyper,
  )
}

function fromFlat(flat: Float64Array, n: number, d: number): Tensor {
  return fromStorage(flat, [n, d], 'float64')
}

/** FastICA as a class. */
export class FastICA extends EstimatorClass<FastIcaHyper, FittedIca> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link fastIca}.
   */
  constructor(hyper: FastIcaHyper = {}) {
    super(fastIca(hyper))
  }
}
