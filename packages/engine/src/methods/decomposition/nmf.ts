import { WhiteboxError } from '../../lib/base/index.ts'
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
import { dnrm2 } from '../../lib/numerics/linalg/index.ts'
import { type Key, normal } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { mm, thinSvd, tr } from './_linear.ts'

/** The divergence NMF minimises: squared Frobenius (beta = 2) or generalised Kullback-Leibler (beta = 1). */
export type NmfLoss = 'frobenius' | 'kullback-leibler'

// sklearn's guard against zero denominators (float32 machine epsilon).
const EPSILON = 1.1920928955078125e-7

/**
 * The beta divergence between X and W H (sklearn's `_beta_divergence` with `square_root`: the Frobenius norm, or
 * sqrt(2 KL)).
 *
 * @param x - The data `[n][d]`.
 * @param w - Weights `[n][k]`.
 * @param h - Components `[k][d]`.
 * @param loss - The loss.
 * @returns The divergence.
 */
export function nmfDivergence(
  x: readonly (readonly number[])[],
  w: readonly (readonly number[])[],
  h: readonly (readonly number[])[],
  loss: NmfLoss,
): number {
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

/** The state of multiplicative-update NMF. */
export interface NmfState {
  readonly w: number[][]
  readonly h: number[][]
  readonly error: number
}

const divide = (num: number[][], den: number[][]): number[][] =>
  num.map((row, i) => row.map((v, j) => v / ((den[i]?.[j] ?? 0) === 0 ? EPSILON : (den[i]?.[j] ?? 0))))

/**
 * NMF by multiplicative updates (Lee and Seung 2001; sklearn's `solver='mu'`) as a traced algorithm: each step
 * multiplies W, then H, by the ratio of the negative and positive parts of the gradient, recording the divergence.
 *
 * @param x - Non-negative data `[n][d]`.
 * @param w0 - Initial weights `[n][k]`.
 * @param h0 - Initial components `[k][d]`.
 * @param loss - The loss.
 * @returns The algorithm.
 */
export function nmfMultiplicativeAlgorithm(
  x: readonly (readonly number[])[],
  w0: readonly (readonly number[])[],
  h0: readonly (readonly number[])[],
  loss: NmfLoss = 'frobenius',
): Algorithm<NmfState> {
  const safeRatio = (w: number[][], h: number[][]): number[][] =>
    mm(w, h).map((row, i) => row.map((v, j) => (x[i]?.[j] ?? 0) / Math.max(v, EPSILON)))
  return {
    name: 'nmf-multiplicative',
    config: { loss },
    init: () => {
      const w = w0.map((r) => [...r])
      const h = h0.map((r) => [...r])
      return { w, h, error: nmfDivergence(x, w, h, loss) }
    },
    step: (s) => {
      let w: number[][]
      let h: number[][]
      if (loss === 'frobenius') {
        w = divide(mm(x, tr(s.h)), mm(s.w, mm(s.h, tr(s.h)))).map((row, i) => row.map((v, c) => v * (s.w[i]?.[c] ?? 0)))
        h = divide(mm(tr(w), x), mm(mm(tr(w), w), s.h)).map((row, c) => row.map((v, j) => v * (s.h[c]?.[j] ?? 0)))
      } else {
        const hSum = s.h.map((row) => row.reduce((a, v) => a + v, 0))
        const numW = mm(safeRatio(s.w, s.h), tr(s.h))
        w = divide(
          numW,
          numW.map((row) => row.map((_, c) => hSum[c] ?? 0)),
        ).map((row, i) => row.map((v, c) => v * (s.w[i]?.[c] ?? 0)))
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
      }
      return { w, h, error: nmfDivergence(x, w, h, loss) }
    },
    observe: (s) => ({ metrics: { error: s.error } }),
  }
}

/**
 * NNDSVD initialisation (Boutsidis and Gallopoulos 2008), with zeros filled by the data mean (sklearn's 'nndsvda').
 *
 * @param x - Non-negative data `[n][d]`.
 * @param k - Components.
 * @returns Initial W `[n][k]` and H `[k][d]`.
 */
export function nndsvda(x: readonly (readonly number[])[], k: number): { w: number[][]; h: number[][] } {
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

/** Hyperparameters of NMF. */
export type NmfHyper = {
  readonly nComponents?: number
  readonly betaLoss?: NmfLoss
  /** 'nndsvda' (default), 'random', or given W and H. */
  readonly init?:
    | 'nndsvda'
    | 'random'
    | { readonly w: readonly (readonly number[])[]; readonly h: readonly (readonly number[])[] }
  readonly maxIter?: number
  /** Relative decrease of the error (checked every 10 steps, as sklearn) that stops the updates (default 1e-4). */
  readonly tol?: number
}

/** The parameters of a fitted NMF. */
export type NmfParams = {
  readonly components: number[][]
  readonly reconstructionErr: number
  readonly nIter: number
  readonly betaLoss: NmfLoss
}

/** The verbs of a fitted NMF: non-negative weights for new rows (H fixed) and reconstructions. */
export interface NmfVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (w: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** A fitted NMF. */
export type FittedNmf = ExplainedFitted<NmfParams, NmfVerbs>

const asRows = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}
const toTensor = (rows: number[][]): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

/**
 * Non-negative matrix factorisation X ~ W H (sklearn.decomposition.NMF with `solver='mu'`), Frobenius or KL, traced
 * update by update. Its exposition is a `components` record (H as components, W as scores). TODO: HALS and
 * coordinate descent, online NMF.
 *
 * @param hyper - Components, loss, initialisation, iteration cap and tolerance.
 * @returns The estimator.
 */
export function nmf(hyper: NmfHyper = {}): Estimator<NmfHyper, FittedNmf> {
  const factorise = (
    x: number[][],
    w0: number[][],
    h0: number[][],
    loss: NmfLoss,
    maxIter: number,
    tol: number,
    fixH: boolean,
    run: ReturnType<typeof fitRunOptions> | undefined,
  ) => {
    const algorithm = nmfMultiplicativeAlgorithm(x, w0, h0, loss)
    const base = fixH
      ? {
          ...algorithm,
          step: (s: NmfState, c: Parameters<typeof algorithm.step>[1]) => ({ ...algorithm.step(s, c), h: s.h }),
        }
      : algorithm
    const first = nmfDivergence(x, w0, h0, loss)
    let previous = first
    const stopping: Algorithm<NmfState> = {
      ...base,
      done: (s, t) => {
        if (tol <= 0 || t === 0 || t % 10 !== 0) return undefined
        const stop = (previous - s.error) / first < tol
        previous = s.error
        return stop ? 'converged' : undefined
      },
    }
    return runAlgorithm(stopping, { ...(run ?? {}), maxSteps: maxIter })
  }
  return defineExplainedEstimator<NmfHyper, NmfParams, NmfVerbs>(
    {
      name: 'nmf',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          betaLoss: choice(['frobenius', 'kullback-leibler'], { default: h.betaLoss ?? 'frobenius' }),
          maxIter: int(1, 1000, { default: h.maxIter ?? 200, label: 'updates' }),
        }),
      fit: (data, h, options) => {
        const x = asRows(data.x, 'nmf')
        if (x.some((r) => r.some((v) => v < 0))) throw new WhiteboxError('value', 'nmf: the data must be non-negative')
        const n = x.length
        const d = x[0]?.length ?? 0
        const k = h.nComponents ?? Math.min(n, d)
        const run = fitRunOptions(options, 'nmf')
        const init = h.init ?? 'nndsvda'
        let w0: number[][]
        let h0: number[][]
        if (typeof init !== 'string') {
          w0 = init.w.map((r) => [...r])
          h0 = init.h.map((r) => [...r])
        } else if (init === 'random') {
          const avg = Math.sqrt(x.flat().reduce((a, v) => a + v, 0) / (n * d) / k)
          const z = toFlat(normal(run.key as Key, [k * d + n * k]))
          h0 = Array.from({ length: k }, (_, c) =>
            Array.from({ length: d }, (_, j) => avg * Math.abs(z[c * d + j] ?? 0)),
          )
          w0 = Array.from({ length: n }, (_, i) =>
            Array.from({ length: k }, (_, c) => avg * Math.abs(z[k * d + i * k + c] ?? 0)),
          )
        } else ({ w: w0, h: h0 } = nndsvda(x, k))
        const loss = h.betaLoss ?? 'frobenius'
        const out = factorise(x, w0, h0, loss, h.maxIter ?? 200, h.tol ?? 1e-4, false, run)
        const s = out.state
        return {
          params: { components: s.h, reconstructionErr: s.error, nIter: out.steps, betaLoss: loss },
          exposition: [
            componentsRecord({ method: 'nmf', features: featureNamesOf(data, d), components: s.h, scores: s.w }),
          ],
        }
      },
      methods: (p) => {
        const d = p.components[0]?.length ?? 0
        const k = p.components.length
        return {
          transform: (x) => {
            const rows = asRows(x, 'nmf transform', d)
            const avg = Math.sqrt(rows.flat().reduce((a, v) => a + v, 0) / Math.max(1, rows.length * d) / k)
            const w0 = rows.map(() => new Array<number>(k).fill(avg || 1e-3))
            return toTensor(factorise(rows, w0, p.components, p.betaLoss, 200, 1e-4, true, undefined).state.w)
          },
          inverseTransform: (w) => toTensor(mm(asRows(w, 'nmf inverseTransform', k), p.components)),
          nFeaturesIn: d,
        }
      },
    },
    hyper,
  )
}

/** NMF as a class. */
export class NMF extends EstimatorClass<NmfHyper, FittedNmf> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link nmf}.
   */
  constructor(hyper: NmfHyper = {}) {
    super(nmf(hyper))
  }
}
