/**
 * Kernel versions of the two-block methods, through the Gram matrix: regularised kernel canonical correlation analysis
 * (`kernelCca`, the singular value decomposition of the product of the two ridge smoothers `K (K + kappa I)^(-1)`), and
 * kernel partial least squares regression (`kernelPlsRegression`, NIPALS on the deflated Gram matrix, one component per
 * step of an `Algorithm`, with the linear PLS of `decomposition` as its linear-kernel case).
 */

import { MdmlError } from '../base/index.ts'
import {
  centerKernel,
  type Kernel,
  type KernelChoice,
  kernelMatrix,
  kernelSpace,
  type ResolvedKernel,
  rbfKernel,
  resolveKernel,
} from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { ddot, dgemm, dgemv, dnrm2, inv, solve, svd } from '../numerics/linalg/index.ts'
import { dict, int } from '../space/index.ts'
import {
  add,
  asTensor,
  eye,
  fromStorage,
  matmul,
  mul,
  primalOf,
  sliceAxis,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { KERNEL_DEFAULTS, type KernelOptions, queryView } from './_classifier.ts'

/** Options of {@link kernelCca}. */
export interface KernelCcaOptions {
  /** The kernel on the first block (default RBF). */
  readonly kernelX?: Kernel
  /** The kernel on the second block (default RBF). */
  readonly kernelY?: Kernel
  /** The ridge `kappa > 0` that keeps the canonical correlations from being one (default 0.1). */
  readonly regularisation?: number
  /** The number of canonical pairs (default 1). */
  readonly nComponents?: number
}

/** A fitted kernel canonical correlation analysis. */
export interface KernelCca {
  /** The regularised canonical correlations, largest first. */
  readonly correlations: number[]
  /** The dual weights of the first block `[n, r]`. */
  readonly alpha: Tensor
  /** The dual weights of the second block `[n, r]`. */
  readonly beta: Tensor
  /** The canonical variates of rows of the first block, `[m, r]`. */
  transformX(x: TensorLike): Tensor
  /** The canonical variates of rows of the second block, `[m, r]`. */
  transformY(y: TensorLike): Tensor
}

/**
 * Regularised kernel canonical correlation analysis (Lai and Fyfe; Hardoon, Szedmak and Shawe-Taylor): with centred Gram
 * matrices `K_x`, `K_y` and the ridge smoothers `H = K (K + kappa I)^(-1)`, the regularised canonical correlations are the
 * singular values of `H_x H_y`, and the dual weights are `alpha = (K_x + kappa I)^(-1) u`, `beta = (K_y + kappa I)^(-1) v`
 * for its singular vectors `u`, `v`. The canonical variates of the training rows are `K_x alpha` and `K_y beta`.
 *
 * @param x - First block `[n, p]`.
 * @param y - Second block `[n, q]`.
 * @param options - The two kernels, the ridge and the number of components.
 * @returns The fitted analysis.
 * @throws MdmlError `shape` for blocks with different numbers of rows; `value` for a non-positive ridge.
 * @concept kernel-multivariate
 * @reference D. R. Hardoon, S. Szedmak and J. Shawe-Taylor, Canonical correlation analysis: an overview with application
 *   to learning methods (Neural Computation 16, 2004).
 * @example
 * ```ts
 * import { kernelCca } from 'mdml-engine/kernel-methods'
 * import { linearKernel } from 'mdml-engine/kernels'
 * const x = [[0], [1], [2], [3], [4]]
 * kernelCca(x, x, { kernelX: linearKernel(), kernelY: linearKernel(), regularisation: 1e-6 }).correlations[0]?.toFixed(3) // => '1.000'
 * ```
 */
export function kernelCca(x: TensorLike, y: TensorLike, options: KernelCcaOptions = {}): KernelCca {
  const kernelX = options.kernelX ?? rbfKernel()
  const kernelY = options.kernelY ?? rbfKernel()
  const kappa = options.regularisation ?? 0.1
  if (!(kappa > 0)) throw new MdmlError('value', `kernelCca: the regularisation must be positive, got ${kappa}`)
  const xs = primalOf(asTensor(x)) as Tensor
  const ys = primalOf(asTensor(y)) as Tensor
  const n = xs.shape[0] ?? 0
  if (ys.shape[0] !== n) throw new MdmlError('shape', `kernelCca: the blocks have ${n} and ${ys.shape[0]} rows`)
  const rawX = kernelX.k(xs)
  const rawY = kernelY.k(ys)
  const kx = centerKernel(rawX)
  const ky = centerKernel(rawY)
  const rx = inv(add(kx, mul(eye(n), kappa)))
  const ry = inv(add(ky, mul(eye(n), kappa)))
  const hx = sub(eye(n), mul(rx, kappa))
  const hy = sub(eye(n), mul(ry, kappa))
  const { u, s, vh } = svd(matmul(hx, hy))
  const r = Math.min(options.nComponents ?? 1, n)
  const alpha = matmul(rx, sliceAxis(u, 1, 0, r))
  const beta = matmul(ry, sliceAxis(swapaxes(vh, 0, 1), 1, 0, r))
  const project = (kernel: Kernel, train: Tensor, raw: Value, weights: Value, query: TensorLike): Tensor => {
    const q = primalOf(asTensor(query)) as Tensor
    return primalOf(matmul(centerKernel(kernel.k(q, train), raw), weights)) as Tensor
  }
  return {
    correlations: Array.from(toFlat(primalOf(s))).slice(0, r),
    alpha: primalOf(alpha) as Tensor,
    beta: primalOf(beta) as Tensor,
    transformX: (q) => project(kernelX, xs, rawX, alpha, q),
    transformY: (q) => project(kernelY, ys, rawY, beta, q),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernel partial least squares
// ---------------------------------------------------------------------------------------------------------------------

/** The hyperparameters of {@link kernelPlsRegression}. */
export interface KernelPlsHyper extends KernelOptions {
  /** The number of latent components (default 2). */
  readonly nComponents: number
  /** The most NIPALS iterations per component (default 500). */
  readonly maxIter: number
  /** Convergence of the score vector within a component (default 1e-10). */
  readonly tol: number
}

/** The state of kernel PLS: the deflated Gram matrix and targets, and the scores and weights found. */
export type KernelPlsState = {
  /** Components found. */
  readonly found: number
  readonly n: number
  readonly q: number
  /** The deflated centred Gram matrix, row-major `n * n`. */
  readonly gram: number[]
  /** The deflated centred targets, row-major `n * q`. */
  readonly targets: number[]
  /** The unit score vectors `t`, one per component, each `n` long. */
  readonly scores: number[][]
  /** The weight vectors `u`, one per component. */
  readonly weights: number[][]
  /** The NIPALS iterations the last component took. */
  readonly iterations: number
}

/** The fitted parameters of {@link kernelPlsRegression}: plain JSON. */
export type KernelPlsParams = {
  readonly kernel: ResolvedKernel
  readonly xFit: number[][]
  /** The dual regression coefficients `[n][q]`, so that `y = K_c(x) B + mean`. */
  readonly coef: number[][]
  readonly yMean: number[]
  /** Column means of the training Gram matrix and its grand mean, to centre cross-kernels. */
  readonly gramMeans: number[]
  readonly gramMean: number
  readonly multiOutput: boolean
  readonly nFeaturesIn: number
}

/** What a fitted kernel PLS offers. */
export interface KernelPlsVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

const PLS_DEFAULTS: KernelPlsHyper = { ...KERNEL_DEFAULTS, nComponents: 2, maxIter: 500, tol: 1e-10 }
const WHO = 'kernelPlsRegression'

function plsSetup(
  data: LearnData,
  h: KernelPlsHyper,
): {
  n: number
  d: number
  q: number
  multi: boolean
  k: Float64Array
  gram: Float64Array
  y: Float64Array
  yMean: Float64Array
  gramMeans: Float64Array
  gramMean: number
  resolved: ResolvedKernel
} {
  const m = matrixView(data.x, WHO)
  const yt = primalOf(asTensor(data.y ?? [])) as Tensor
  const q = yt.shape[1] ?? 1
  if ((yt.shape[0] ?? 0) !== m.n) throw new MdmlError('shape', `${WHO}: x has ${m.n} rows but y has ${yt.shape[0]}`)
  const resolved = resolveKernel(h as KernelChoice, data.x)
  const k = kernelMatrix(h, resolved, data.x)
  const n = m.n
  const gramMeans = new Float64Array(n)
  let gramMean = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) gramMeans[j] = (gramMeans[j] ?? 0) + (k[i * n + j] ?? 0) / n
  for (const v of gramMeans) gramMean += v / n
  const gram = Float64Array.from(
    k,
    (v, idx) => v - (gramMeans[Math.floor(idx / n)] ?? 0) - (gramMeans[idx % n] ?? 0) + gramMean,
  )
  const yRaw = Float64Array.from(toFlat(yt))
  const yMean = new Float64Array(q)
  for (let i = 0; i < n; i++) for (let c = 0; c < q; c++) yMean[c] = (yMean[c] ?? 0) + (yRaw[i * q + c] ?? 0) / n
  const y = Float64Array.from(yRaw, (v, idx) => v - (yMean[idx % q] ?? 0))
  return { n, d: m.d, q, multi: yt.shape.length === 2, k, gram, y, yMean, gramMeans, gramMean, resolved }
}

/**
 * Kernel partial least squares (Rosipal and Trejo) as an algorithm, one latent component per step: NIPALS on the centred
 * Gram matrix `K` and targets `Y`: iterate `t = K u / |K u|`, `c = Y' t`, `u = Y c / |Y c|` to convergence, then deflate
 * `K <- (I - t t') K (I - t t')`, `Y <- Y - t t' Y`. With a linear kernel the predictions are those of linear PLS regression
 * with the same number of components.
 *
 * @param data - Features and targets (`[n]` or `[n, q]`).
 * @param hyper - The kernel options and the number of components.
 * @returns The algorithm, logging the squared norm of the remaining targets and the iterations per component.
 * @throws MdmlError `value` for more components than rows.
 * @concept kernel-multivariate
 * @reference R. Rosipal and L. J. Trejo, Kernel partial least squares regression in reproducing kernel Hilbert space
 *   (JMLR 2, 2001).
 * @example
 * ```ts
 * import { kernelPlsAlgorithm, kernelPlsDefaults } from 'mdml-engine/kernel-methods'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = kernelPlsAlgorithm({ x: [[0], [1], [2], [3]], y: [0, 1, 2, 3.5] }, { ...kernelPlsDefaults, kernel: 'linear', gamma: 1, nComponents: 1 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.found // => 1
 * ```
 */
export function kernelPlsAlgorithm(data: LearnData, hyper: KernelPlsHyper): Algorithm<KernelPlsState> {
  if (!Number.isInteger(hyper.nComponents) || hyper.nComponents < 1)
    throw new MdmlError('value', `${WHO}: nComponents must be a positive integer`)
  const s = plsSetup(data, hyper)
  if (hyper.nComponents > s.n) throw new MdmlError('value', `${WHO}: more components than rows`)
  const { n, q } = s
  return {
    name: 'kernel-pls',
    config: { nComponents: hyper.nComponents, kernel: s.resolved.kernel },
    schema: channels({
      metrics: {
        residual: {
          description: 'the squared norm of the targets left after the components so far',
          direction: 'lower',
        },
        iterations: { description: 'NIPALS iterations the last component took' },
      },
    }),
    init: () => ({
      found: 0,
      n,
      q,
      gram: Array.from(s.gram),
      targets: Array.from(s.y),
      scores: [],
      weights: [],
      iterations: 0,
    }),
    step: (st) => {
      const g = Float64Array.from(st.gram)
      const y = Float64Array.from(st.targets)
      let u: Float64Array = Float64Array.from({ length: n }, (_, i) => y[i * q] ?? 0)
      let t: Float64Array = new Float64Array(n)
      let iterations = 0
      for (; iterations < hyper.maxIter; iterations++) {
        const previous = t
        t = dgemv(g, n, n, u)
        const tn = dnrm2(t) || 1
        t = t.map((v) => v / tn)
        const c = dgemv(y, n, q, t, true)
        u = dgemv(y, n, q, c)
        const un = dnrm2(u) || 1
        u = u.map((v) => v / un)
        if (q === 1 || dnrm2(t.map((v, i) => v - (previous[i] ?? 0))) < hyper.tol) {
          iterations++
          break
        }
      }
      // Deflate: K <- (I - t t') K (I - t t'), Y <- Y - t (t' Y).
      const kt = dgemv(g, n, n, t)
      const tkt = ddot(t, kt)
      const next = Float64Array.from(g, (v, idx) => {
        const i = Math.floor(idx / n)
        const j = idx % n
        return v - (t[i] ?? 0) * (kt[j] ?? 0) - (kt[i] ?? 0) * (t[j] ?? 0) + (t[i] ?? 0) * (t[j] ?? 0) * tkt
      })
      const ty = dgemv(y, n, q, t, true)
      const yNext = Float64Array.from(y, (v, idx) => v - (t[Math.floor(idx / q)] ?? 0) * (ty[idx % q] ?? 0))
      return {
        ...st,
        found: st.found + 1,
        gram: Array.from(next),
        targets: Array.from(yNext),
        scores: [...st.scores, Array.from(t)],
        weights: [...st.weights, Array.from(u)],
        iterations,
      }
    },
    done: (st) => (st.found >= hyper.nComponents ? 'converged' : undefined),
    observe: (st) => ({ metrics: { residual: st.targets.reduce((a, v) => a + v * v, 0), iterations: st.iterations } }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as KernelPlsState,
  }
}

/** The default hyperparameters of {@link kernelPlsRegression}. */
export const kernelPlsDefaults: KernelPlsHyper = PLS_DEFAULTS

/**
 * Kernel partial least squares regression on the iterative base: the algorithm of {@link kernelPlsAlgorithm}, one latent
 * component per step, so `fittedAt(t)` is the model with `t` components. The prediction is
 * `K_c(x) U (T' K U)^(-1) T' Y` plus the target mean, with `K_c` the cross-kernel centred on the training Gram matrix.
 *
 * @param hyper - The kernel options and `nComponents`.
 * @returns The estimator.
 * @concept kernel-multivariate
 * @example
 * ```ts
 * import { kernelPlsRegression } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3], [4]]
 * const fit = kernelPlsRegression({ kernel: 'linear', gamma: 1, nComponents: 1 }).fit({ x, y: [1, 3, 5, 7, 9] })
 * toFlat(fit.predict([[5]]))[0]?.toFixed(2) // => '11.00'
 * ```
 */
export function kernelPlsRegression(
  hyper: Partial<KernelPlsHyper> = {},
): IterativeEstimator<KernelPlsHyper, KernelPlsState, KernelPlsParams, KernelPlsVerbs> {
  return iterative<KernelPlsHyper, KernelPlsState, KernelPlsParams, KernelPlsVerbs>(
    {
      name: 'kernel-methods/kernel-pls-regression',
      kind: 'regressor',
      defaults: PLS_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, Math.max(20, h.nComponents), { default: h.nComponents, label: 'components' }),
          ...kernelSpace(h),
        }),
      algorithm: (data, h) => kernelPlsAlgorithm(data, h),
      maxSteps: (h) => h.nComponents,
      params: (st, data, h) => {
        const s = plsSetup(data, h)
        const r = st.found
        const m = matrixView(data.x, WHO)
        // B = U (T' K U)^(-1) T' Y with the original centred K and centred Y.
        const t = Float64Array.from({ length: s.n * r }, (_, idx) => st.scores[idx % r]?.[Math.floor(idx / r)] ?? 0)
        const u = Float64Array.from({ length: s.n * r }, (_, idx) => st.weights[idx % r]?.[Math.floor(idx / r)] ?? 0)
        let coef: Float64Array = new Float64Array(s.n * s.q)
        if (r > 0) {
          const ku = dgemm(s.gram, u, s.n, s.n, r)
          const tku = dgemm(t, ku, r, s.n, r, { transposeA: true })
          const ty = dgemm(t, s.y, r, s.n, s.q, { transposeA: true })
          const core = Float64Array.from(
            toFlat(primalOf(solve(fromStorage(tku, [r, r], 'float64'), fromStorage(ty, [r, s.q], 'float64')))),
          )
          coef = dgemm(u, core, s.n, r, s.q)
        }
        return {
          kernel: s.resolved,
          xFit: Array.from({ length: s.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
          coef: Array.from({ length: s.n }, (_, i) => Array.from(coef.subarray(i * s.q, (i + 1) * s.q))),
          yMean: Array.from(s.yMean),
          gramMeans: Array.from(s.gramMeans),
          gramMean: s.gramMean,
          multiOutput: s.multi,
          nFeaturesIn: s.d,
        }
      },
      verbs: (p, h) => {
        const n = p.xFit.length
        const q = p.yMean.length
        const predict = (x: TensorLike): Tensor => {
          const qv = queryView(x, p.nFeaturesIn, WHO)
          const cross = kernelMatrix(h, p.kernel, x, p.xFit)
          const out = new Float64Array(qv.n * q)
          for (let i = 0; i < qv.n; i++) {
            let rowMean = 0
            for (let j = 0; j < n; j++) rowMean += (cross[i * n + j] ?? 0) / n
            for (let c = 0; c < q; c++) {
              let v = p.yMean[c] ?? 0
              for (let j = 0; j < n; j++)
                v += ((cross[i * n + j] ?? 0) - (p.gramMeans[j] ?? 0) - rowMean + p.gramMean) * (p.coef[j]?.[c] ?? 0)
              out[i * q + c] = v
            }
          }
          return fromStorage(out, p.multiOutput ? [qv.n, q] : [qv.n], 'float64')
        }
        return {
          predict,
          score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
          nFeaturesIn: p.nFeaturesIn,
        }
      },
      logs: { 'solver/residual': Log(), 'solver/iterations': Log() },
    },
    { ...PLS_DEFAULTS, ...hyper },
  )
}
