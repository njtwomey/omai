/**
 * Cross-decomposition of two blocks of variables (scikit-learn's `cross_decomposition`): PLS regression (latent
 * directions of the inputs that best predict the outputs), PLS canonical, canonical correlation analysis (directions of
 * maximal correlation between the blocks), all by NIPALS with one component per step, and PLS-SVD (the singular vectors
 * of the cross-covariance).
 */

import { MdmlError } from '../base/index.ts'
import {
  defineEstimator,
  type Estimator,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  requireTargets,
} from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { dict, int } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { mm, pinvRows, thinSvd, tr } from './_linear.ts'

/** Which PLS: regression (PLS2), canonical (PLS mode A), CCA (mode B), or the SVD of the cross-covariance. */
export type PlsVariant = 'regression' | 'canonical' | 'cca' | 'svd'

const EPS = Number.EPSILON
const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)
const column = (m: readonly (readonly number[])[], j: number): number[] => m.map((r) => r[j] ?? 0)
const matVec = (m: readonly (readonly number[])[], v: readonly number[]): number[] => m.map((r) => dot(r, v))
const tMatVec = (m: readonly (readonly number[])[], v: readonly number[]): number[] =>
  (m[0] ?? []).map((_, j) => m.reduce((s, r, i) => s + (r[j] ?? 0) * (v[i] ?? 0), 0))

/** sklearn's `_center_scale_xy`: centre, and divide by the ddof-1 standard deviation (zeros become one). */
function standardise(rows: number[][], scale: boolean): { z: number[][]; mean: number[]; std: number[] } {
  const n = rows.length
  const d = rows[0]?.length ?? 0
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((a, r) => a + (r[j] ?? 0), 0) / n)
  const std = Array.from({ length: d }, (_, j) => {
    if (!scale) return 1
    const s = Math.sqrt(rows.reduce((a, r) => a + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / (n - 1))
    return s === 0 ? 1 : s
  })
  return { z: rows.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (std[j] ?? 1))), mean, std }
}

/** The first pair of singular vectors of X^T Y by the NIPALS power method (sklearn's, modes A and B). */
function powerMethod(
  x: number[][],
  y: number[][],
  mode: 'A' | 'B',
  normYWeights: boolean,
  maxIter: number,
  tol: number,
): { xw: number[]; yw: number[] } {
  const q = y[0]?.length ?? 0
  let yScore = Array.from({ length: q }, (_, j) => column(y, j)).find((c) => c.some((v) => Math.abs(v) > EPS))
  if (yScore === undefined) throw new MdmlError('value', 'pls: the y residual is constant')
  const xPinv = mode === 'B' ? pinvRows(x) : undefined
  const yPinv = mode === 'B' ? pinvRows(y) : undefined
  let xw: number[] = []
  let yw: number[] = []
  let old: number[] | undefined
  for (let it = 0; it < maxIter; it++) {
    xw =
      xPinv !== undefined
        ? matVec(xPinv, yScore)
        : tMatVec(x, yScore).map((v) => v / dot(yScore as number[], yScore as number[]))
    const xn = Math.sqrt(dot(xw, xw)) + EPS
    xw = xw.map((v) => v / xn)
    const xScore = matVec(x, xw)
    yw = yPinv !== undefined ? matVec(yPinv, xScore) : tMatVec(y, xScore).map((v) => v / dot(xScore, xScore))
    if (normYWeights) {
      const yn = Math.sqrt(dot(yw, yw)) + EPS
      yw = yw.map((v) => v / yn)
    }
    const yss = dot(yw, yw) + EPS
    yScore = matVec(y, yw).map((v) => v / yss)
    const diff =
      old === undefined ? Number.POSITIVE_INFINITY : xw.reduce((s, v, i) => s + (v - (old?.[i] ?? 0)) ** 2, 0)
    if (diff < tol || q === 1) break
    old = xw
  }
  return { xw, yw }
}

/** The two blocks, standardised, with what is needed to standardise new rows. */
interface Blocks {
  readonly x: ReturnType<typeof standardise>
  readonly y: ReturnType<typeof standardise>
}

const blocksCache = new WeakMap<LearnData, Map<boolean, Blocks>>()

function blocksOf(data: LearnData, scale: boolean): Blocks {
  let byScale = blocksCache.get(data)
  const hit = byScale?.get(scale)
  if (hit !== undefined) return hit
  const xm = matrixView(data.x, 'pls', { minSamples: 2 })
  const yt = asTensor(requireTargets(data, 'pls'))
  const q = yt.shape.length === 1 ? 1 : (yt.shape[1] ?? 1)
  const yFlat = toFlat(yt)
  const yRows = Array.from({ length: yt.shape[0] ?? 0 }, (_, i) => yFlat.slice(i * q, (i + 1) * q))
  const xRows = Array.from({ length: xm.n }, (_, i) => Array.from(xm.data.subarray(i * xm.d, (i + 1) * xm.d)))
  const out = { x: standardise(xRows, scale), y: standardise(yRows, scale) }
  if (byScale === undefined) {
    byScale = new Map()
    blocksCache.set(data, byScale)
  }
  byScale.set(scale, out)
  return out
}

/** The state of NIPALS: the residual blocks and the weights, loadings and correlation of each component found. */
export interface PlsState {
  readonly xResidual: number[][]
  readonly yResidual: number[][]
  /** One row per component found. */
  readonly xWeights: number[][]
  readonly yWeights: number[][]
  readonly xLoadings: number[][]
  readonly yLoadings: number[][]
  /** The correlation between the x and y scores of each component. */
  readonly correlations: number[]
}

/** Options of {@link plsAlgorithm}. */
export interface PlsOptions {
  readonly maxIter?: number
  readonly tol?: number
}

/**
 * NIPALS for PLS regression, PLS canonical and CCA (scikit-learn's `_PLS`) as a traced {@link Algorithm}, one component
 * per step: the first pair of singular vectors of the residual cross-covariance by the power method, the scores, the
 * loadings, and the deflation of both blocks. It observes the correlation of the scores of the component just found and
 * the weights so far.
 *
 * @param x - Standardised inputs `[n, p]`.
 * @param y - Standardised outputs `[n, q]`.
 * @param variant - `'regression'`, `'canonical'` or `'cca'`.
 * @param k - Components to find.
 * @param options - Power-method tolerance (1e-6) and iteration cap (500).
 * @returns The algorithm; k steps.
 * @example
 * ```ts
 * import { plsAlgorithm } from 'mdml-engine/decomposition'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-1, 0.2], [0, -0.3], [1, 0.1]]
 * const y = [[-1], [0], [1]]
 * runAlgorithm(plsAlgorithm(x, y, 'regression', 1), { maxSteps: 1 }).state.correlations[0] // => 0.999916
 * ```
 */
export function plsAlgorithm(
  x: number[][],
  y: number[][],
  variant: 'regression' | 'canonical' | 'cca',
  k: number,
  options: PlsOptions = {},
): Algorithm<PlsState> {
  const q = y[0]?.length ?? 1
  const p = x[0]?.length ?? 0
  const mode = variant === 'cca' ? 'B' : 'A'
  const normY = variant !== 'regression'
  const limit = variant === 'regression' ? p : Math.min(x.length, p, q)
  if (!(Number.isInteger(k) && k >= 1 && k <= limit))
    throw new MdmlError('value', `pls: nComponents must be in 1..${limit}`)
  return {
    name: `pls-${variant}`,
    config: { variant, k },
    schema: channels({
      metrics: {
        correlation: { description: 'the correlation between the x and y scores of the component just found' },
      },
      tensors: { xWeights: { role: 'state', description: 'the x weights found so far', shape: ['k', 'p'] } },
    }),
    init: () => ({
      xResidual: x.map((r) => [...r]),
      yResidual: y.map((r) => [...r]),
      xWeights: [],
      yWeights: [],
      xLoadings: [],
      yLoadings: [],
      correlations: [],
    }),
    step: (s) => {
      const yk = s.yResidual.map((r) => [...r])
      for (let c = 0; c < q; c++) if (yk.every((r) => Math.abs(r[c] ?? 0) < 10 * EPS)) for (const r of yk) r[c] = 0
      let { xw, yw } = powerMethod(s.xResidual, yk, mode, normY, options.maxIter ?? 500, options.tol ?? 1e-6)
      const big = xw.reduce((b, v, i) => (Math.abs(v) > Math.abs(xw[b] ?? 0) ? i : b), 0)
      const sign = Math.sign(xw[big] ?? 1) || 1
      xw = xw.map((v) => v * sign)
      yw = yw.map((v) => v * sign)
      const xs = matVec(s.xResidual, xw)
      const yss = normY ? 1 : dot(yw, yw)
      const ys = matVec(yk, yw).map((v) => v / yss)
      const xl = tMatVec(s.xResidual, xs).map((v) => v / dot(xs, xs))
      const xResidual = s.xResidual.map((r, i) => r.map((v, f) => v - (xs[i] ?? 0) * (xl[f] ?? 0)))
      const yl =
        variant === 'regression'
          ? tMatVec(yk, xs).map((v) => v / dot(xs, xs))
          : tMatVec(yk, ys).map((v) => v / dot(ys, ys))
      const by = variant === 'regression' ? xs : ys
      const yResidual = yk.map((r, i) => r.map((v, t) => v - (by[i] ?? 0) * (yl[t] ?? 0)))
      const correlation = dot(xs, ys) / (Math.sqrt(dot(xs, xs) * dot(ys, ys)) || 1)
      return {
        xResidual,
        yResidual,
        xWeights: [...s.xWeights, xw],
        yWeights: [...s.yWeights, yw],
        xLoadings: [...s.xLoadings, xl],
        yLoadings: [...s.yLoadings, yl],
        correlations: [...s.correlations, correlation],
      }
    },
    done: (s) => (s.xWeights.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: s.correlations.length === 0 ? {} : { correlation: s.correlations.at(-1) ?? 0 },
      tensors: { xWeights: { role: 'state', value: s.xWeights.flat(), shape: [s.xWeights.length, p] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PlsState,
  }
}

/** The fitted parameters of a cross-decomposition. */
export type PlsParams = {
  readonly variant: PlsVariant
  readonly xMean: number[]
  readonly yMean: number[]
  readonly xStd: number[]
  readonly yStd: number[]
  /** weights[f][k], loadings[f][k] and rotations[f][k], as sklearn's `x_weights_` and friends. */
  readonly xWeights: number[][]
  readonly yWeights: number[][]
  readonly xLoadings: number[][]
  readonly yLoadings: number[][]
  readonly xRotations: number[][]
  readonly yRotations: number[][]
  /** The correlation of the x and y scores of each component. */
  readonly correlations: number[]
  /** coef[t][f]: predictions are (x - xMean) coef^T + yMean (regression variants). */
  readonly coef: number[][]
}

/** The verbs of a cross-decomposition: x scores, and for PLS regression predictions and R^2. */
export interface PlsVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly predict?: (x: TensorLike) => Tensor
  readonly score?: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** Hyperparameters of the cross-decompositions. */
export interface PlsHyper {
  readonly nComponents: number
  /** Standardise X and Y. */
  readonly scale: boolean
  readonly maxIter: number
  readonly tol: number
}

function paramsOf(variant: PlsVariant, s: PlsState, b: Blocks): PlsParams {
  const xW = tr(s.xWeights)
  const yW = tr(s.yWeights)
  const base = { variant, xMean: b.x.mean, yMean: b.y.mean, xStd: b.x.std, yStd: b.y.std, correlations: s.correlations }
  if (s.xWeights.length === 0)
    return {
      ...base,
      xWeights: [],
      yWeights: [],
      xLoadings: [],
      yLoadings: [],
      xRotations: [],
      yRotations: [],
      coef: [],
    }
  const xL = tr(s.xLoadings)
  const yL = tr(s.yLoadings)
  const xRotations = mm(xW, pinvRows(mm(tr(xL), xW)))
  const yRotations = mm(yW, pinvRows(mm(tr(yL), yW)))
  const coef =
    variant === 'regression'
      ? tr(mm(xRotations, tr(yL)).map((row) => row.map((v, t) => v * (b.y.std[t] ?? 1)))).map((row) =>
          row.map((v, f) => v / (b.x.std[f] ?? 1)),
        )
      : []
  return { ...base, xWeights: xW, yWeights: yW, xLoadings: xL, yLoadings: yL, xRotations, yRotations, coef }
}

function plsVerbs(p: PlsParams): PlsVerbs {
  const d = p.xMean.length
  const rows = (x: TensorLike): number[][] => {
    const m = matrixView(x, 'pls', { nFeatures: d })
    return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d)))
  }
  const toTensor = (r: number[][], k: number): Tensor =>
    fromStorage(Float64Array.from(r.flat()), [r.length, k], 'float64')
  const predict = (x: TensorLike): number[][] =>
    rows(x).map((r) =>
      p.yMean.map(
        (mu, t) =>
          (p.coef[t] === undefined
            ? 0
            : dot(
                p.coef[t],
                r.map((v, f) => v - (p.xMean[f] ?? 0)),
              )) + mu,
      ),
    )
  const regression = p.variant === 'regression'
  return {
    transform: (x) =>
      toTensor(
        p.xRotations.length === 0
          ? rows(x).map(() => [])
          : mm(
              rows(x).map((r) => r.map((v, f) => (v - (p.xMean[f] ?? 0)) / (p.xStd[f] ?? 1))),
              p.xRotations,
            ),
        p.xRotations[0]?.length ?? 0,
      ),
    ...(regression
      ? {
          predict: (x: TensorLike) => {
            const y = predict(x)
            return p.yMean.length === 1
              ? fromStorage(Float64Array.from(y.map((r) => r[0] ?? 0)), [y.length], 'float64')
              : toTensor(y, p.yMean.length)
          },
          score: (data: LearnData) => {
            const y = predict(data.x)
            const truth = toFlat(asTensor(requireTargets(data, 'pls score')))
            return r2Score(truth, y.flat()).estimate.value
          },
        }
      : {}),
    nFeaturesIn: d,
  }
}

const plsEstimator = (
  variant: 'regression' | 'canonical' | 'cca',
  name: string,
): IterativeEstimator<PlsHyper, PlsState, PlsParams, PlsVerbs> =>
  iterative<PlsHyper, PlsState, PlsParams, PlsVerbs>({
    name,
    kind: variant === 'regression' ? 'regressor' : 'transformer',
    defaults: { nComponents: 2, scale: true, maxIter: 500, tol: 1e-6 },
    hyperSpace: (h) => dict({ nComponents: int(1, 20, { default: h.nComponents, label: 'components' }) }),
    algorithm: (data, h) => {
      const b = blocksOf(data, h.scale)
      return plsAlgorithm(b.x.z, b.y.z, variant, h.nComponents, { maxIter: h.maxIter, tol: h.tol })
    },
    maxSteps: (h) => h.nComponents,
    params: (s, data, h) => paramsOf(variant, s, blocksOf(data, h.scale)),
    verbs: plsVerbs,
  })

/**
 * PLS regression (sklearn.cross_decomposition.PLSRegression, PLS2 by NIPALS with regression deflation) on the iterative
 * base: one latent direction per step.
 *
 * @index estimator Partial least squares regression; CCA beside it.
 * @example
 * ```ts
 * import { plsRegression } from 'mdml-engine/decomposition'
 * const x = [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1]]
 * const y = [1, 2, 3, 4, 5]
 * plsRegression.with({ nComponents: 1 }).fit({ x, y }).params.xWeights.length // => 2
 * ```
 */
export const plsRegression: IterativeEstimator<PlsHyper, PlsState, PlsParams, PlsVerbs> = plsEstimator(
  'regression',
  'decomposition/pls-regression',
)

/**
 * PLS canonical (sklearn.cross_decomposition.PLSCanonical, mode A with canonical deflation), one component per step.
 *
 * @example
 * ```ts
 * import { plsCanonical } from 'mdml-engine/decomposition'
 * const x = [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1]]
 * const y = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0]]
 * plsCanonical.with({ nComponents: 1 }).fit({ x, y }).params.correlations.length // => 1
 * ```
 */
export const plsCanonical: IterativeEstimator<PlsHyper, PlsState, PlsParams, PlsVerbs> = plsEstimator(
  'canonical',
  'decomposition/pls-canonical',
)

/**
 * Canonical correlation analysis (sklearn.cross_decomposition.CCA, NIPALS mode B), one canonical pair per step; the
 * correlation of each pair's scores is in the parameters and on the tape.
 *
 * @example
 * ```ts
 * import { cca } from 'mdml-engine/decomposition'
 * const x = [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1]]
 * const y = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0]]
 * cca.with({ nComponents: 1 }).fit({ x, y }).params.correlations.length // => 1
 * ```
 */
export const cca: IterativeEstimator<PlsHyper, PlsState, PlsParams, PlsVerbs> = plsEstimator('cca', 'decomposition/cca')

/** Hyperparameters of {@link plsSvd}. */
export interface PlsSvdHyper {
  readonly nComponents: number
  readonly scale: boolean
}

/**
 * PLS-SVD (sklearn.cross_decomposition.PLSSVD): the leading singular vectors of the standardised cross-covariance.
 *
 * @param hyper - Components and scaling.
 * @returns The estimator.
 * @example
 * ```ts
 * import { plsSvd } from 'mdml-engine/decomposition'
 * const x = [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1]]
 * const y = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0]]
 * plsSvd({ nComponents: 1 }).fit({ x, y }).params.xWeights.length // => 2
 * ```
 */
export function plsSvd(hyper: Partial<PlsSvdHyper> = {}): Estimator<PlsSvdHyper, Fitted<PlsParams> & PlsVerbs> {
  return defineEstimator<PlsSvdHyper, PlsParams, PlsVerbs>(
    {
      name: 'decomposition/pls-svd',
      kind: 'transformer',
      hyperSpace: (h) => dict({ nComponents: int(1, 20, { default: h.nComponents, label: 'components' }) }),
      fit: (data, h) => {
        const b = blocksOf(data, h.scale)
        const p = b.x.z[0]?.length ?? 0
        const q = b.y.z[0]?.length ?? 1
        const limit = Math.min(b.x.z.length, p, q)
        if (!(Number.isInteger(h.nComponents) && h.nComponents >= 1 && h.nComponents <= limit))
          throw new MdmlError('value', `plsSvd: nComponents must be in 1..${limit}`)
        const c = mm(tr(b.x.z), b.y.z)
        const { u, vt } = thinSvd(Float64Array.from(c.flat()), p, q)
        const xWeights: number[][] = []
        const yWeights: number[][] = []
        for (let j = 0; j < h.nComponents; j++) {
          let xw = column(u, j)
          let yw = vt[j] ?? []
          const big = xw.reduce((bi, v, i) => (Math.abs(v) > Math.abs(xw[bi] ?? 0) ? i : bi), 0)
          const sign = Math.sign(xw[big] ?? 1) || 1
          xw = xw.map((v) => v * sign)
          yw = yw.map((v) => v * sign)
          xWeights.push(xw)
          yWeights.push(yw)
        }
        return {
          variant: 'svd',
          xMean: b.x.mean,
          yMean: b.y.mean,
          xStd: b.x.std,
          yStd: b.y.std,
          xWeights: tr(xWeights),
          yWeights: tr(yWeights),
          xLoadings: [],
          yLoadings: [],
          xRotations: tr(xWeights),
          yRotations: tr(yWeights),
          correlations: [],
          coef: [],
        }
      },
      methods: plsVerbs,
    },
    { nComponents: 2, scale: true, ...hyper },
  )
}
