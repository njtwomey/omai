/**
 * Independent component analysis by FastICA (Hyvarinen and Oja 2000; as in scikit-learn): unmixing observed signals into
 * statistically independent sources by whitening the data and then running a fixed-point iteration that maximises
 * non-Gaussianity, either on all components at once with symmetric decorrelation (parallel, one iteration per step) or
 * one component at a time against those already found (deflation, one component per step).
 */

import { OmaiError } from '../base/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { normal } from '../random/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { centred, columnMeans, mm, pinvRows, project, reconstruct, thinSvd, tr } from './_linear.ts'
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
 * @param w - A square matrix `[k, k]`.
 * @returns The decorrelated matrix.
 * @example
 * ```ts
 * import { symmetricDecorrelation } from 'omai-engine/decomposition'
 * symmetricDecorrelation([[2, 0], [0, 3]]) // => [[1, 0], [0, 1]]
 * ```
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
  /** max | |diag(W_new W^T)| - 1 | of the last step (a large number before the first). */
  readonly change: number
}

const icaSchema = channels({
  metrics: { change: { description: 'how far the unmixing rows turned this step (large before the first)' } },
  tensors: { unmixing: { role: 'state', description: 'the unmixing matrix of the whitened data', shape: ['k', 'k'] } },
})

/**
 * Parallel FastICA (sklearn's `_ica_par`) as a traced {@link Algorithm}: each step applies the fixed-point update W <-
 * E[g(W x) x^T] - diag(E[g'(W x)]) W to all components at once, then decorrelates symmetrically, until max | |diag(W_new
 * W^T)| - 1 | < tol.
 *
 * @param z - Whitened data `[k, n]` (components by samples).
 * @param init - The initial unmixing matrix `[k, k]`.
 * @param options - The contrast (logcosh) and tolerance (1e-4).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { fastIcaAlgorithm } from 'omai-engine/decomposition'
 * fastIcaAlgorithm([[1, -1, 0.5, -0.5]], [[1]]).name // => 'fast-ica'
 * ```
 */
export function fastIcaAlgorithm(
  z: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: { readonly fun?: IcaContrast; readonly tol?: number } = {},
): Algorithm<FastIcaState> {
  const fun = options.fun ?? 'logcosh'
  const tol = options.tol ?? 1e-4
  const n = z[0]?.length ?? 0
  const k = init.length
  return {
    name: 'fast-ica',
    config: { fun, tol },
    schema: icaSchema,
    init: () => ({ w: symmetricDecorrelation(init), change: Number.MAX_VALUE }),
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
    observe: (s) => ({
      metrics: s.change === Number.MAX_VALUE ? {} : { change: s.change },
      tensors: { unmixing: { role: 'state', value: s.w.flat(), shape: [k, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FastIcaState,
  }
}

/**
 * Deflationary FastICA (sklearn's `_ica_def`) as a traced {@link Algorithm}, one component per step: the fixed-point
 * iteration on a single unmixing vector, orthogonalised against those found (Gram-Schmidt) after each update, run to
 * convergence.
 *
 * @param z - Whitened data `[k, n]`.
 * @param init - The initial unmixing matrix `[k, k]` (row j starts component j).
 * @param options - The contrast (logcosh), tolerance (1e-4) and iteration cap per component (200).
 * @returns The algorithm; k steps.
 * @example
 * ```ts
 * import { fastIcaDeflationAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const z = [[1, -1, 0.5, -0.5], [0.5, 0.5, -1, 1]]
 * runAlgorithm(fastIcaDeflationAlgorithm(z, [[1, 0.2], [0.1, 1]]), { maxSteps: 2 }).state.w.length // => 2
 * ```
 */
export function fastIcaDeflationAlgorithm(
  z: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: { readonly fun?: IcaContrast; readonly tol?: number; readonly maxIter?: number } = {},
): Algorithm<FastIcaState> {
  const fun = options.fun ?? 'logcosh'
  const tol = options.tol ?? 1e-4
  const maxIter = options.maxIter ?? 200
  const n = z[0]?.length ?? 0
  const k = init.length
  const zt = tr(z)
  return {
    name: 'fast-ica-deflation',
    config: { fun, tol, maxIter },
    schema: icaSchema,
    init: () => ({ w: [], change: Number.MAX_VALUE }),
    step: (s) => {
      const j = s.w.length
      const unit = (v: readonly number[]): number[] => {
        const norm = Math.hypot(...v) || 1
        return v.map((e) => e / norm)
      }
      let w = unit(init[j] ?? [])
      let change = Number.MAX_VALUE
      for (let it = 0; it < maxIter; it++) {
        const wx = zt.map((col) => col.reduce((a, v, q) => a + v * (w[q] ?? 0), 0))
        const { g, gPrimeMean } = contrast(fun, wx)
        let w1 = w.map((_, q) => {
          let a = 0
          for (let i = 0; i < n; i++) a += (z[q]?.[i] ?? 0) * (g[i] ?? 0)
          return a / n - gPrimeMean * (w[q] ?? 0)
        })
        for (const prev of s.w) {
          const dot = w1.reduce((a, v, q) => a + v * (prev[q] ?? 0), 0)
          w1 = w1.map((v, q) => v - dot * (prev[q] ?? 0))
        }
        w1 = unit(w1)
        change = Math.abs(Math.abs(w1.reduce((a, v, q) => a + v * (w[q] ?? 0), 0)) - 1)
        w = w1
        if (change < tol) break
      }
      return { w: [...s.w, w], change }
    },
    done: (s) => (s.w.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: s.change === Number.MAX_VALUE ? {} : { change: s.change },
      tensors: { unmixing: { role: 'state', value: s.w.flat(), shape: [s.w.length, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FastIcaState,
  }
}

/** Hyperparameters of {@link fastIca}. */
export interface FastIcaHyper {
  readonly nComponents: number
  readonly fun: IcaContrast
  /** All components at once (parallel) or one at a time (deflation). */
  readonly algorithm: 'parallel' | 'deflation'
  readonly tol: number
  readonly maxIter: number
  /** The initial unmixing matrix `[k, k]` (`null`: standard normal draws). */
  readonly wInit: number[][] | null
}

/** The fitted parameters of {@link fastIca}: the mean, the unmixing components and the mixing matrix. */
export type IcaParams = {
  readonly mean: number[]
  /** components[k][f]: the unmixing rows (sources = (x - mean) components^T), scaled to unit-variance sources. */
  readonly components: number[][]
  /** mixing[f][k]: the pseudo-inverse of the components. */
  readonly mixing: number[][]
}

interface Whitened {
  readonly mean: number[]
  readonly whitening: number[][]
  readonly z: number[][]
  readonly xRows: number[][]
}

const whitenedCache = new WeakMap<LearnData, Map<number, Whitened>>()

/** sklearn's whitening: u = V with the first row positive, K = (u / d)^T, z = K x^T sqrt(n); cached per dataset. */
function whiten(data: LearnData, k: number): Whitened {
  let byK = whitenedCache.get(data)
  const hit = byK?.get(k)
  if (hit !== undefined) return hit
  const m = matrixView(data.x, 'fastIca', { minSamples: 2 })
  if (!(Number.isInteger(k) && k >= 1 && k <= Math.min(m.n, m.d)))
    throw new OmaiError('value', `fastIca: nComponents must be in 1..${Math.min(m.n, m.d)}`)
  const mean = columnMeans(m)
  const xc = centred(m, mean)
  const { s, vt } = thinSvd(xc, m.n, m.d)
  const signs = vt.map((row) => Math.sign(row[0] ?? 1) || 1)
  const whitening = vt.slice(0, k).map((row, c) => row.map((v) => (v * (signs[c] ?? 1)) / (s[c] ?? 1)))
  const xRows = Array.from({ length: m.n }, (_, i) => Array.from(xc.subarray(i * m.d, (i + 1) * m.d)))
  const z = mm(whitening, tr(xRows)).map((row) => row.map((v) => v * Math.sqrt(m.n)))
  const out = { mean, whitening, z, xRows }
  if (byK === undefined) {
    byK = new Map()
    whitenedCache.set(data, byK)
  }
  byK.set(k, out)
  return out
}

/** The ICA verbs: the unmixing, and the mixing back. */
export type IcaVerbs = Omit<ProjectionVerbs, 'score'>

/**
 * FastICA (sklearn.decomposition.FastICA with `whiten='unit-variance'`) on the iterative base: whitening by the SVD of
 * the centred data, then a fixed-point iteration per step (`'parallel'`) or a component per step (`'deflation'`),
 * with the sources scaled to unit variance.
 *
 * @index estimator FastICA independent component analysis.
 * @example
 * ```ts
 * import { fastIca } from 'omai-engine/decomposition'
 * const t = Array.from({ length: 60 }, (_, i) => i / 6)
 * const x = t.map((v) => [Math.sin(2 * v) + 0.5 * ((v * 3) % 1), Math.sign(Math.sin(3 * v)) + 0.4 * Math.sin(2 * v)])
 * fastIca.with({ nComponents: 2 }).fit({ x }).params.components.length // => 2
 * ```
 */
export const fastIca: IterativeEstimator<FastIcaHyper, FastIcaState, IcaParams, IcaVerbs> = iterative<
  FastIcaHyper,
  FastIcaState,
  IcaParams,
  IcaVerbs
>({
  name: 'decomposition/fast-ica',
  kind: 'transformer',
  defaults: { nComponents: 2, fun: 'logcosh', algorithm: 'parallel', tol: 1e-4, maxIter: 200, wInit: null },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
      fun: choice(['logcosh', 'exp', 'cube'] as const, { default: h.fun, label: 'contrast function' }),
      algorithm: choice(['parallel', 'deflation'] as const, { default: h.algorithm }),
    }),
  algorithm: (data, h, key) => {
    const { z } = whiten(data, h.nComponents)
    const wInit = h.wInit ?? (toArray(normal(key, [h.nComponents, h.nComponents])) as number[][])
    return h.algorithm === 'deflation'
      ? fastIcaDeflationAlgorithm(z, wInit, { fun: h.fun, tol: h.tol, maxIter: h.maxIter })
      : fastIcaAlgorithm(z, wInit, { fun: h.fun, tol: h.tol })
  },
  maxSteps: (h) => (h.algorithm === 'deflation' ? h.nComponents : h.maxIter),
  params: (s, data, h) => {
    const { mean, whitening, xRows } = whiten(data, h.nComponents)
    let components = mm(s.w, whitening)
    if (components.length === 0) return { mean, components: [], mixing: [] }
    const sources = mm(xRows, tr(components))
    const sd = Array.from({ length: components.length }, (_, c) => {
      const col = sources.map((r) => r[c] ?? 0)
      const mu = col.reduce((a, v) => a + v, 0) / col.length
      return Math.sqrt(col.reduce((a, v) => a + (v - mu) ** 2, 0) / col.length)
    })
    components = components.map((row, c) => row.map((v) => v / (sd[c] || 1)))
    return { mean, components, mixing: pinvRows(components) }
  },
  verbs: (p) => ({
    transform: (x: TensorLike) => project(x, p.mean, p.components),
    inverseTransform: (z: TensorLike) => reconstruct(z, p.mean, tr(p.mixing)),
    nFeaturesIn: p.mean.length,
    featureNamesOut: () => p.components.map((_, c) => `fastica${c}`),
  }),
})
