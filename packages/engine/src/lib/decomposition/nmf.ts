/**
 * Non-negative matrix factorisation (scikit-learn's `NMF`): approximating a non-negative matrix by a product `W H` of
 * non-negative factors, which yields additive, parts-based components. Lee and Seung's multiplicative updates for the
 * Frobenius or Kullback-Leibler loss, and hierarchical alternating least squares (HALS, cyclic coordinate descent) for
 * the Frobenius loss, one sweep per step, with NNDSVD initialisation.
 */

import { OmaiError, toRows } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { type Key, normal } from '../random/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { mm, thinSvd, tr } from './_linear.ts'

/** The divergence NMF minimises: squared Frobenius (beta = 2) or generalised Kullback-Leibler (beta = 1). */
export type NmfLoss = 'frobenius' | 'kullback-leibler'

/** The update rule: Lee and Seung's multiplicative updates, or HALS coordinate descent (Frobenius only). */
export type NmfSolver = 'mu' | 'hals'

// sklearn's guard against zero denominators (float32 machine epsilon).
const EPSILON = 1.1920928955078125e-7

type Rows = readonly (readonly number[])[]

/**
 * The beta divergence between X and W H (sklearn's `_beta_divergence` with `square_root`: the Frobenius norm, or
 * sqrt(2 KL)).
 *
 * @param x - The data `[n, d]`.
 * @param w - Weights `[n, k]`.
 * @param h - Components `[k, d]`.
 * @param loss - The loss.
 * @returns The divergence.
 * @example
 * ```ts
 * import { nmfDivergence } from 'omai-engine/decomposition'
 * nmfDivergence([[1, 2]], [[1]], [[1, 1]], 'frobenius') // => 1
 * ```
 */
export function nmfDivergence(x: Rows, w: Rows, h: Rows, loss: NmfLoss): number {
  const wh = mm(w, h)
  let s = 0
  for (const [i, row] of x.entries())
    for (const [j, v] of row.entries()) {
      const y = wh[i]?.[j] ?? 0
      if (loss === 'frobenius') s += (v - y) ** 2 / 2
      else s += (v > 0 ? v * Math.log(v / Math.max(y, EPSILON)) : 0) - v + y
    }
  return Math.sqrt(2 * s)
}

/** The state of NMF: the factors, the error, and what the stopping rule needs. */
export interface NmfState {
  readonly w: number[][]
  readonly h: number[][]
  readonly error: number
  /** The error at the start, which the stopping rule's decrease is relative to. */
  readonly initialError: number
  /** The error at the last check (every tenth update). */
  readonly checked: number
  /** Whether the relative decrease since the last check fell below the tolerance. */
  readonly stalled: boolean
}

const divide = (num: number[][], den: number[][]): number[][] =>
  num.map((row, i) => row.map((v, j) => v / ((den[i]?.[j] ?? 0) === 0 ? EPSILON : (den[i]?.[j] ?? 0))))

/** One multiplicative update of W then H (sklearn's `_multiplicative_update_w` and `_h`). */
function multiplicative(x: Rows, s: NmfState, loss: NmfLoss, fixH: boolean): { w: number[][]; h: number[][] } {
  const safeRatio = (w: number[][], h: number[][]): number[][] =>
    mm(w, h).map((row, i) => row.map((v, j) => (x[i]?.[j] ?? 0) / Math.max(v, EPSILON)))
  let w: number[][]
  let h: number[][]
  if (loss === 'frobenius') {
    w = divide(mm(x, tr(s.h)), mm(s.w, mm(s.h, tr(s.h)))).map((row, i) => row.map((v, c) => v * (s.w[i]?.[c] ?? 0)))
    h = fixH
      ? s.h
      : divide(mm(tr(w), x), mm(mm(tr(w), w), s.h)).map((row, c) => row.map((v, j) => v * (s.h[c]?.[j] ?? 0)))
    return { w, h }
  }
  const hSum = s.h.map((row) => row.reduce((a, v) => a + v, 0))
  const numW = mm(safeRatio(s.w, s.h), tr(s.h))
  w = divide(
    numW,
    numW.map((row) => row.map((_, c) => hSum[c] ?? 0)),
  ).map((row, i) => row.map((v, c) => v * (s.w[i]?.[c] ?? 0)))
  if (fixH) return { w, h: s.h }
  const wSum = (w[0] ?? []).map((_, c) => w.reduce((a, r) => a + (r[c] ?? 0), 0) || 1)
  const numH = mm(tr(w), safeRatio(w, s.h))
  h = divide(
    numH,
    numH.map((row, c) => row.map(() => wSum[c] ?? 1)),
  ).map((row, c) =>
    row.map((v, j) => {
      const out = v * (s.h[c]?.[j] ?? 0)
      return out < Number.EPSILON ? 0 : out
    }),
  )
  return { w, h }
}

/** One HALS sweep over the columns of `w` given `h`: w[i][t] <- max(w[i][t] - grad / (h h^T)[t][t], 0), in place. */
function halsSweep(x: Rows, w: number[][], h: Rows): void {
  const hht = mm(h, tr(h))
  const xht = mm(x, tr(h))
  const k = hht.length
  for (let t = 0; t < k; t++) {
    const hess = hht[t]?.[t] ?? 0
    if (hess === 0) continue
    for (const [i, row] of w.entries()) {
      let grad = -(xht[i]?.[t] ?? 0)
      for (let r = 0; r < k; r++) grad += (hht[t]?.[r] ?? 0) * (row[r] ?? 0)
      row[t] = Math.max((row[t] ?? 0) - grad / hess, 0)
    }
  }
}

/** Options of {@link nmfAlgorithm}. */
export interface NmfOptions {
  readonly loss?: NmfLoss
  readonly solver?: NmfSolver
  /** Relative decrease of the error, checked every 10 updates (as sklearn), that stops the updates (default 1e-4; 0: never). */
  readonly tol?: number
  /** Keep H fixed and fit W only (the transform of new rows). */
  readonly fixH?: boolean
}

/**
 * NMF as a traced {@link Algorithm}, one update per step: multiplicative (Lee and Seung 2001; sklearn's `solver='mu'`,
 * Frobenius or KL) or HALS (cyclic coordinate descent; sklearn's `solver='cd'`, Frobenius), each updating W and then H.
 * It observes the divergence and the factors.
 *
 * @param x - Non-negative data `[n, d]`.
 * @param w0 - Initial weights `[n, k]`.
 * @param h0 - Initial components `[k, d]`.
 * @param options - Loss, solver, tolerance and whether H is fixed.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { nmfAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [2, 4], [3, 6]]
 * const out = runAlgorithm(nmfAlgorithm(x, [[1], [1], [1]], [[1, 1]], { solver: 'hals' }), { maxSteps: 3 })
 * out.state.error < 0.001 // => true
 * ```
 */
export function nmfAlgorithm(x: Rows, w0: Rows, h0: Rows, options: NmfOptions = {}): Algorithm<NmfState> {
  const loss = options.loss ?? 'frobenius'
  const solver = options.solver ?? 'mu'
  const tol = options.tol ?? 1e-4
  const fixH = options.fixH ?? false
  if (solver === 'hals' && loss !== 'frobenius')
    throw new OmaiError('value', 'nmf: HALS minimises the Frobenius loss only')
  const n = x.length
  const d = x[0]?.length ?? 0
  const k = h0.length
  return {
    name: `nmf-${solver}`,
    config: { loss, solver, tol, fixH },
    schema: channels({
      metrics: { error: { description: 'the beta divergence between the data and W H', direction: 'lower' } },
      tensors: {
        components: { role: 'state', description: 'the non-negative components H', shape: ['k', 'd'] },
        weights: { role: 'state', description: 'the non-negative weights W', shape: ['n', 'k'] },
      },
    }),
    init: () => {
      const w = w0.map((r) => [...r])
      const h = h0.map((r) => [...r])
      const error = nmfDivergence(x, w, h, loss)
      return { w, h, error, initialError: error, checked: error, stalled: false }
    },
    step: (s, ctx) => {
      let w: number[][]
      let h: number[][]
      if (solver === 'hals') {
        w = s.w.map((r) => [...r])
        halsSweep(x, w, s.h)
        if (fixH) h = s.h
        else {
          const ht = tr(s.h)
          halsSweep(tr(x), ht, tr(w))
          h = tr(ht)
        }
      } else ({ w, h } = multiplicative(x, s, loss, fixH))
      const error = nmfDivergence(x, w, h, loss)
      const check = tol > 0 && (ctx.t + 1) % 10 === 0
      return {
        w,
        h,
        error,
        initialError: s.initialError,
        checked: check ? error : s.checked,
        stalled: check && (s.checked - error) / (s.initialError || 1) < tol,
      }
    },
    done: (s) => (s.stalled ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { error: s.error },
      tensors: {
        components: { role: 'state', value: s.h.flat(), shape: [k, d] },
        weights: { role: 'state', value: s.w.flat(), shape: [n, k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NmfState,
  }
}

/**
 * NNDSVD initialisation (Boutsidis and Gallopoulos 2008), with zeros filled by the data mean (sklearn's 'nndsvda').
 *
 * @param x - Non-negative data `[n, d]`.
 * @param k - Components.
 * @returns Initial W `[n, k]` and H `[k, d]`.
 * @example
 * ```ts
 * import { nndsvda } from 'omai-engine/decomposition'
 * nndsvda([[1, 2], [2, 4]], 1).h[0]?.map((v) => Math.round(v * 1000) / 1000) // => [1, 2]
 * ```
 */
export function nndsvda(x: Rows, k: number): { w: number[][]; h: number[][] } {
  const n = x.length
  const d = x[0]?.length ?? 0
  const { u, s, vt } = thinSvd(Float64Array.from(x.flat()), n, d)
  const w = Array.from({ length: n }, () => new Array<number>(k).fill(0))
  const h = Array.from({ length: k }, () => new Array<number>(d).fill(0))
  for (let j = 0; j < k; j++) {
    const ux = u.map((r) => r[j] ?? 0)
    const vy = vt[j] ?? []
    let uu: number[]
    let vv: number[]
    let sigma: number
    if (j === 0) {
      uu = ux.map(Math.abs)
      vv = vy.map(Math.abs)
      sigma = 1
    } else {
      const xp = ux.map((v) => Math.max(v, 0))
      const yp = vy.map((v) => Math.max(v, 0))
      const xn = ux.map((v) => Math.max(-v, 0))
      const yn = vy.map((v) => Math.max(-v, 0))
      const mp = dnrm2(xp) * dnrm2(yp)
      const mn = dnrm2(xn) * dnrm2(yn)
      if (mp > mn) {
        uu = xp.map((v) => v / (dnrm2(xp) || 1))
        vv = yp.map((v) => v / (dnrm2(yp) || 1))
        sigma = mp
      } else {
        uu = xn.map((v) => v / (dnrm2(xn) || 1))
        vv = yn.map((v) => v / (dnrm2(yn) || 1))
        sigma = mn
      }
    }
    const lbd = Math.sqrt((s[j] ?? 0) * sigma)
    for (let i = 0; i < n; i++) (w[i] as number[])[j] = lbd * (uu[i] ?? 0)
    for (let q = 0; q < d; q++) (h[j] as number[])[q] = lbd * (vv[q] ?? 0)
  }
  const avg = x.flat().reduce((a, v) => a + v, 0) / (n * d)
  const fill = (m: number[][]): number[][] => m.map((r) => r.map((v) => (v < Number.EPSILON ? avg : v)))
  return { w: fill(w), h: fill(h) }
}

/** Hyperparameters of {@link nmf}. */
export interface NmfHyper {
  readonly nComponents: number
  readonly betaLoss: NmfLoss
  readonly solver: NmfSolver
  /** How W and H start: NNDSVD with the zeros filled by the mean, random draws, or the given `wInit` and `hInit`. */
  readonly init: 'nndsvda' | 'random' | 'custom'
  readonly wInit: number[][] | null
  readonly hInit: number[][] | null
  readonly maxIter: number
  readonly tol: number
}

/** The fitted parameters of {@link nmf}. */
export type NmfParams = {
  readonly components: number[][]
  /** The weights of the training rows. */
  readonly weights: number[][]
  readonly reconstructionErr: number
  readonly betaLoss: NmfLoss
  readonly solver: NmfSolver
}

/** The verbs of a fitted NMF: non-negative weights for new rows (H fixed) and reconstructions. */
export interface NmfVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (w: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * The starting factors of an NMF fit: NNDSVD with the zeros filled by the mean (`nndsvda`), random draws scaled by
 * `sqrt(mean(x) / k)` (`random`), or the given `wInit` and `hInit` (`custom`).
 *
 * @param x - Non-negative data `[n, d]`.
 * @param hyper - The number of components, the rule and the given factors.
 * @param key - Randomness for `random`.
 * @returns Initial W `[n, k]` and H `[k, d]`.
 * @throws OmaiError with code `value` for negative data, or `custom` without both factors.
 * @example
 * ```ts
 * import { initialNmfFactors } from 'omai-engine/decomposition'
 * import { key } from 'omai-engine/random'
 * initialNmfFactors([[1, 2], [2, 4]], { nComponents: 1, init: 'random', wInit: null, hInit: null }, key(0)).h.length // => 1
 * ```
 */
export function initialNmfFactors(
  x: Rows,
  hyper: Pick<NmfHyper, 'nComponents' | 'init' | 'wInit' | 'hInit'>,
  key: Key,
): { w: number[][]; h: number[][] } {
  if (x.some((r) => r.some((v) => v < 0))) throw new OmaiError('value', 'nmf: the data must be non-negative')
  const n = x.length
  const d = x[0]?.length ?? 0
  const k = hyper.nComponents
  if (hyper.init === 'custom') {
    if (hyper.wInit === null || hyper.hInit === null)
      throw new OmaiError('value', "nmf: init 'custom' needs wInit and hInit")
    return { w: hyper.wInit, h: hyper.hInit }
  }
  if (hyper.init === 'random') {
    const avg = Math.sqrt(x.flat().reduce((a, v) => a + v, 0) / (n * d) / k)
    const z = normal(key, [k * d + n * k]).data
    return {
      h: Array.from({ length: k }, (_, c) => Array.from({ length: d }, (_, j) => avg * Math.abs(z[c * d + j] ?? 0))),
      w: Array.from({ length: n }, (_, i) =>
        Array.from({ length: k }, (_, c) => avg * Math.abs(z[k * d + i * k + c] ?? 0)),
      ),
    }
  }
  return nndsvda(x, k)
}

const asRows = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return toRows(m.data, m.n, m.d)
}
const toTensor = (rows: number[][]): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

/**
 * NMF X ~ W H (sklearn.decomposition.NMF) on the iterative base, Frobenius or KL by multiplicative updates, or
 * Frobenius by HALS, one update of W and H per step.
 *
 * @index estimator Non-negative matrix factorisation by multiplicative or HALS updates.
 * @example
 * ```ts
 * import { nmf } from 'omai-engine/decomposition'
 * const x = [[1, 2, 0], [2, 4, 0], [0, 0, 3], [0, 0, 6]]
 * nmf.with({ nComponents: 2, maxIter: 100 }).fit({ x }).params.components.length // => 2
 * ```
 */
export const nmf: IterativeEstimator<NmfHyper, NmfState, NmfParams, NmfVerbs> = iterative<
  NmfHyper,
  NmfState,
  NmfParams,
  NmfVerbs
>({
  name: 'decomposition/nmf',
  kind: 'transformer',
  defaults: {
    nComponents: 2,
    betaLoss: 'frobenius',
    solver: 'mu',
    init: 'nndsvda',
    wInit: null,
    hInit: null,
    maxIter: 200,
    tol: 1e-4,
  },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
      betaLoss: choice(['frobenius', 'kullback-leibler'] as const, { default: h.betaLoss, label: 'beta divergence' }),
      solver: choice(['mu', 'hals'] as const, { default: h.solver }),
      maxIter: int(1, 1000, { default: h.maxIter, label: 'updates' }),
    }),
  algorithm: (data, h, key) => {
    const x = asRows(data.x, 'nmf')
    const { w: w0, h: h0 } = initialNmfFactors(x, h, key)
    return nmfAlgorithm(x, w0, h0, { loss: h.betaLoss, solver: h.solver, tol: h.tol })
  },
  maxSteps: (h) => h.maxIter,
  params: (s, _data, h) => ({
    components: s.h,
    weights: s.w,
    reconstructionErr: s.error,
    betaLoss: h.betaLoss,
    solver: h.solver,
  }),
  verbs: (p) => {
    const d = p.components[0]?.length ?? 0
    const k = p.components.length
    return {
      transform: (x) => {
        const rows = asRows(x, 'nmf transform', d)
        const avg = Math.sqrt(rows.flat().reduce((a, v) => a + v, 0) / Math.max(1, rows.length * d) / k)
        const w0 = rows.map(() => new Array<number>(k).fill(avg || 1e-3))
        const out = runAlgorithm(
          nmfAlgorithm(rows, w0, p.components, { loss: p.betaLoss, solver: p.solver, fixH: true }),
          { maxSteps: 200 },
        )
        return toTensor(out.state.w)
      },
      inverseTransform: (w) => toTensor(mm(asRows(w, 'nmf inverseTransform', k), p.components)),
      nFeaturesIn: d,
    }
  },
})
