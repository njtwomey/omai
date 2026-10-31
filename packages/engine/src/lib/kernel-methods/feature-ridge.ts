/**
 * Kernel ridge regression over an explicit feature map: ridge regression in the Nystrom or random Fourier feature space,
 * solved in the primal for `D` features instead of in the dual for `n` observations. `ridgeSolve` is the closed form
 * `(Z'Z + alpha I)^(-1) Z'y`, `featureRidgeProblem` is the same ridge as a smooth problem for the solver slot,
 * `featureRidge` fits a given feature map, and `approximateKernelRidge` is the estimator that draws the map (landmarks
 * or frequencies) itself. The approximation to exact kernel ridge is explicit: `ridgeGap` gives the difference of the
 * fitted values together with the bound `||K - K~||_2 ||y|| / alpha` it satisfies, which falls as the features grow.
 */

import { MdmlError } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import {
  type FeatureMap,
  type Kernel,
  nystroem,
  orthogonalRandomFeatures,
  type RandomFourierFeatures,
  randomFourierFeatures,
  rbfKernel,
} from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { dgemm, solve, svdvals } from '../numerics/linalg/index.ts'
import { type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { infNorm, problemOf, type SlotState, type Smooth, type SolverChoice, slotted } from './_slot.ts'

/** The ridge solution in feature space: weights `[D][k]` and the intercepts `[k]`. */
export interface RidgeSolution {
  readonly weights: Float64Array
  readonly intercept: Float64Array
}

/** Targets as `n` rows of `k` outputs. */
function targetRows(y: TensorLike, who: string): { rows: Float64Array; n: number; k: number; multi: boolean } {
  const t = primalOf(asTensor(y))
  if (t.shape.length !== 1 && t.shape.length !== 2) {
    throw new MdmlError('shape', `${who}: expected y to be 1-D or 2-D, got [${t.shape.join(', ')}]`)
  }
  return { rows: Float64Array.from(toFlat(t)), n: t.shape[0] ?? 0, k: t.shape[1] ?? 1, multi: t.shape.length === 2 }
}

function featureRows(z: TensorLike, who: string): { rows: Float64Array; n: number; d: number } {
  const t = primalOf(asTensor(z))
  if (t.shape.length !== 2)
    throw new MdmlError('shape', `${who}: expected features [n, D], got [${t.shape.join(', ')}]`)
  return { rows: Float64Array.from(toFlat(t)), n: t.shape[0] ?? 0, d: t.shape[1] ?? 0 }
}

function checkAlpha(alpha: number, who: string): number {
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new MdmlError('value', `${who}: alpha must be a positive finite number, got ${alpha}`)
  }
  return alpha
}

const columnMeans = (a: Float64Array, n: number, d: number): Float64Array => {
  const m = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] = (m[j] ?? 0) + (a[i * d + j] ?? 0) / n
  return m
}

/**
 * Ridge regression in a feature space in closed form: the weights `(Z'Z + alpha I)^(-1) Z'y` of the penalised least
 * squares `|y - Z w - b|^2 + alpha |w|^2`, the intercept left unpenalised when fitted (the data are then centred).
 *
 * @param z - Features `[n, D]`.
 * @param y - Targets `[n]` or `[n, k]`.
 * @param alpha - Ridge penalty `alpha > 0` (default 1).
 * @param fitIntercept - Whether to fit an intercept (default false, as kernel ridge has none).
 * @returns The weights (`D * k`, row-major) and the intercepts (`k`).
 * @throws MdmlError `shape` if `z` and `y` disagree; `value` for a non-positive `alpha`.
 * @concept kernel-ridge
 * @example
 * ```ts
 * import { ridgeSolve } from 'mdml-engine/kernel-methods'
 * Array.from(ridgeSolve([[1], [2], [3]], [2, 4, 6], 1e-9).weights).map((v) => Math.round(v * 1e6) / 1e6) // => [2]
 * ```
 */
export function ridgeSolve(z: TensorLike, y: TensorLike, alpha = 1, fitIntercept = false): RidgeSolution {
  checkAlpha(alpha, 'ridgeSolve')
  const f = featureRows(z, 'ridgeSolve')
  const t = targetRows(y, 'ridgeSolve')
  if (f.n !== t.n) throw new MdmlError('shape', `ridgeSolve: z has ${f.n} rows but y has ${t.n}`)
  const { n, d } = f
  const k = t.k
  const zMean = fitIntercept ? columnMeans(f.rows, n, d) : new Float64Array(d)
  const yMean = fitIntercept ? columnMeans(t.rows, n, k) : new Float64Array(k)
  const zc = f.rows.map((v, i) => v - (zMean[i % d] ?? 0))
  const yc = t.rows.map((v, i) => v - (yMean[i % k] ?? 0))
  const a = dgemm(zc, zc, d, n, d, { transposeA: true })
  for (let i = 0; i < d; i++) a[i * d + i] = (a[i * d + i] ?? 0) + alpha
  const b = dgemm(zc, yc, d, n, k, { transposeA: true })
  const w = Float64Array.from(
    toFlat(primalOf(solve(fromStorage(a, [d, d], 'float64'), fromStorage(b, [d, k], 'float64')))),
  )
  const intercept = Float64Array.from({ length: k }, (_, j) => {
    let s = yMean[j] ?? 0
    for (let l = 0; l < d; l++) s -= (zMean[l] ?? 0) * (w[l * k + j] ?? 0)
    return s
  })
  return { weights: w, intercept: fitIntercept ? intercept : new Float64Array(k) }
}

/** The ridge problem in feature space as an analytic objective: the unknowns are the weights then the intercepts. */
function ridgeSmooth(
  zRows: Float64Array,
  yRows: Float64Array,
  n: number,
  d: number,
  k: number,
  alpha: number,
  fitIntercept: boolean,
): Smooth {
  const p = d * k + (fitIntercept ? k : 0)
  const residual = (x: ArrayLike<number>): Float64Array => {
    const r = new Float64Array(n * k)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < k; j++) {
        let s = fitIntercept ? (x[d * k + j] ?? 0) : 0
        for (let l = 0; l < d; l++) s += (zRows[i * d + l] ?? 0) * (x[l * k + j] ?? 0)
        r[i * k + j] = s - (yRows[i * k + j] ?? 0)
      }
    return r
  }
  return {
    value: (x) => {
      const r = residual(x)
      let s = 0
      for (const v of r) s += 0.5 * v * v
      for (let i = 0; i < d * k; i++) s += 0.5 * alpha * (x[i] ?? 0) ** 2
      return s
    },
    gradient: (x) => {
      const r = residual(x)
      const g = new Float64Array(p)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < k; j++) {
          const rij = r[i * k + j] ?? 0
          for (let l = 0; l < d; l++) g[l * k + j] = (g[l * k + j] ?? 0) + (zRows[i * d + l] ?? 0) * rij
          if (fitIntercept) g[d * k + j] = (g[d * k + j] ?? 0) + rij
        }
      for (let i = 0; i < d * k; i++) g[i] = (g[i] ?? 0) + alpha * (x[i] ?? 0)
      return g
    },
    hessian: () => {
      // Block structure: for each output j the same (Z'Z + alpha I) on its weights, plus the intercept border.
      const h = new Float64Array(p * p)
      for (let j = 0; j < k; j++) {
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) {
            let s = a === b ? alpha : 0
            for (let i = 0; i < n; i++) s += (zRows[i * d + a] ?? 0) * (zRows[i * d + b] ?? 0)
            h[(a * k + j) * p + (b * k + j)] = s
          }
        if (fitIntercept) {
          h[(d * k + j) * p + (d * k + j)] = n
          for (let a = 0; a < d; a++) {
            let s = 0
            for (let i = 0; i < n; i++) s += zRows[i * d + a] ?? 0
            h[(a * k + j) * p + (d * k + j)] = s
            h[(d * k + j) * p + (a * k + j)] = s
          }
        }
      }
      return h
    },
  }
}

/**
 * The ridge regression of a feature space as a smooth problem for the solver slot: the objective
 * `1/2 |y - Z w - b|^2 + alpha/2 |w|^2` with its gradient and Hessian, starting at zero. Its minimiser is
 * {@link ridgeSolve}'s; any solver of `smoothSolvers` reaches it.
 *
 * @param z - Features `[n, D]`.
 * @param y - Targets `[n]` or `[n, k]`.
 * @param alpha - Ridge penalty `alpha > 0` (default 1).
 * @param fitIntercept - Whether the unknowns include unpenalised intercepts (default false).
 * @returns The problem; its unknowns are the `D * k` weights (row-major) then the `k` intercepts.
 * @concept kernel-ridge
 * @example
 * ```ts
 * import { featureRidgeProblem } from 'mdml-engine/kernel-methods'
 * featureRidgeProblem([[1, 0], [0, 1], [1, 1]], [1, 2, 3], 0.5).x0.length // => 2
 * ```
 */
export function featureRidgeProblem(z: TensorLike, y: TensorLike, alpha = 1, fitIntercept = false): SmoothProblem {
  checkAlpha(alpha, 'featureRidgeProblem')
  const f = featureRows(z, 'featureRidgeProblem')
  const t = targetRows(y, 'featureRidgeProblem')
  if (f.n !== t.n) throw new MdmlError('shape', `featureRidgeProblem: z has ${f.n} rows but y has ${t.n}`)
  const p = f.d * t.k + (fitIntercept ? t.k : 0)
  return problemOf(ridgeSmooth(f.rows, t.rows, f.n, f.d, t.k, alpha, fitIntercept), new Array<number>(p).fill(0))
}

/** A ridge regression fitted in the space of a feature map. */
export interface FeatureRidge {
  readonly map: FeatureMap
  /** Weights `[D]` (or `[D, k]` for several outputs). */
  readonly weights: Tensor
  /** Intercepts `[k]` (zeros unless fitted). */
  readonly intercept: Tensor
  /** Predictions `[m]` (or `[m, k]`) at new rows. */
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * Ridge regression in the feature space of a map (Nystrom, random Fourier, or any other): kernel ridge regression with
 * the kernel `z(x) . z(y)` solved in the primal.
 *
 * @param map - The feature map.
 * @param x - Training rows `[n, d]`.
 * @param y - Targets `[n]` or `[n, k]`.
 * @param alpha - Ridge penalty (default 1).
 * @param fitIntercept - Whether to fit an unpenalised intercept (default false).
 * @returns The fitted regression.
 * @concept kernel-ridge
 * @example
 * ```ts
 * import { featureRidge } from 'mdml-engine/kernel-methods'
 * import { linearKernel, nystroem } from 'mdml-engine/kernels'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3]]
 * const fit = featureRidge(nystroem(linearKernel(), x, { indices: [1, 2] }), x, [0, 2, 4, 6], 1e-9)
 * toFlat(fit.predict([[4]])).map((v) => Math.round(v * 1e4) / 1e4) // => [8]
 * ```
 */
export function featureRidge(
  map: FeatureMap,
  x: TensorLike,
  y: TensorLike,
  alpha = 1,
  fitIntercept = false,
): FeatureRidge {
  const z = map.transform(x)
  const t = targetRows(y, 'featureRidge')
  const sol = ridgeSolve(z, y, alpha, fitIntercept)
  const d = (primalOf(z).shape[1] ?? 0) as number
  const weights = fromStorage(sol.weights, t.multi ? [d, t.k] : [d], 'float64')
  const intercept = fromStorage(sol.intercept, [t.k], 'float64')
  return {
    map,
    weights,
    intercept,
    predict: (xq) => predictFeatures(map, xq, sol.weights, sol.intercept, t.multi),
  }
}

function predictFeatures(map: FeatureMap, x: TensorLike, w: Float64Array, b: Float64Array, multi: boolean): Tensor {
  const z = primalOf(map.transform(x))
  const m = z.shape[0] ?? 0
  const d = z.shape[1] ?? 0
  const k = b.length
  const out = dgemm(Float64Array.from(toFlat(z)), w, m, d, k)
  for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) out[i * k + j] = (out[i * k + j] ?? 0) + (b[j] ?? 0)
  return fromStorage(out, multi ? [m, k] : [m], 'float64')
}

/** How far a feature-space ridge is from exact kernel ridge on the training rows. */
export interface RidgeGap {
  /** `|h - h~|_2` over the training rows: the fitted values `K (K + alpha I)^(-1) y` of the exact and approximate fits. */
  readonly fittedGap: number
  /** `|K - K~|_2`, the spectral norm of the Gram error. */
  readonly spectral: number
  /** The bound `|K - K~|_2 |y| / alpha` that `fittedGap` satisfies when `K~` and `K` are positive semi-definite. */
  readonly bound: number
}

/**
 * The explicit approximation error of ridge regression in a feature space against exact kernel ridge, on the training
 * rows. With `K~ = Z Z'` and `h = K (K + alpha I)^(-1) y`, `h~ = K~ (K~ + alpha I)^(-1) y` one has the identity
 * `h - h~ = alpha (K~ + alpha I)^(-1) (K - K~) (K + alpha I)^(-1) y`, and for positive semi-definite `K`, `K~` each
 * inverse has norm at most `1/alpha`, so `|h - h~| <= |K - K~|_2 |y| / alpha`. The bound is what shrinks as the features
 * grow (monotonically for nested Nystrom landmarks, whose `K - K~` is positive semi-definite and decreasing).
 *
 * @param exactGram - The exact Gram matrix `[n, n]`.
 * @param map - The feature map.
 * @param x - The training rows `[n, d]`.
 * @param y - Targets `[n]`.
 * @param alpha - The ridge penalty.
 * @returns The fitted-value gap, the spectral Gram error and the bound.
 * @concept kernel-ridge
 * @reference C. Cortes, M. Mohri and A. Talwalkar, On the impact of kernel approximation on learning accuracy (AISTATS 2010).
 * @example
 * ```ts
 * import { ridgeGap } from 'mdml-engine/kernel-methods'
 * import { gram, nystroem, rbfKernel } from 'mdml-engine/kernels'
 * const x = [[0], [1], [2], [3], [4]]
 * const k = rbfKernel()
 * const g = ridgeGap(gram(k, x), nystroem(k, x, { indices: [0, 2, 4] }), x, [0, 1, 0, -1, 0], 0.1)
 * g.fittedGap <= g.bound // => true
 * ```
 */
export function ridgeGap(
  exactGram: TensorLike,
  map: FeatureMap,
  x: TensorLike,
  y: TensorLike,
  alpha: number,
): RidgeGap {
  checkAlpha(alpha, 'ridgeGap')
  const t = targetRows(y, 'ridgeGap')
  const n = t.n
  const k = Float64Array.from(toFlat(primalOf(asTensor(exactGram))))
  const z = Float64Array.from(toFlat(primalOf(map.transform(x))))
  const d = z.length / n
  const approx = dgemm(z, z, n, d, n, { transposeB: true })
  const hat = (g: Float64Array): Float64Array => {
    const a = Float64Array.from(g)
    for (let i = 0; i < n; i++) a[i * n + i] = (a[i * n + i] ?? 0) + alpha
    const c = Float64Array.from(
      toFlat(primalOf(solve(fromStorage(a, [n, n], 'float64'), fromStorage(t.rows, [n, t.k], 'float64')))),
    )
    return dgemm(g, c, n, n, t.k)
  }
  const h = hat(k)
  const ht = hat(approx)
  let gap = 0
  for (let i = 0; i < h.length; i++) gap += ((h[i] ?? 0) - (ht[i] ?? 0)) ** 2
  const diff = k.map((v, i) => v - (approx[i] ?? 0))
  const spectral = toFlat(primalOf(svdvals(fromStorage(diff, [n, n], 'float64'))))[0] ?? 0
  let y2 = 0
  for (const v of t.rows) y2 += v * v
  return { fittedGap: Math.sqrt(gap), spectral, bound: (spectral * Math.sqrt(y2)) / alpha }
}

// ---------------------------------------------------------------------------------------------------------------------
// The estimator
// ---------------------------------------------------------------------------------------------------------------------

/** Which explicit feature map {@link approximateKernelRidge} draws. */
export type FeatureKind = 'nystroem' | 'fourier' | 'orthogonal-fourier'

/** The hyperparameters of {@link approximateKernelRidge}. */
export interface ApproximateRidgeHyper {
  /** The map: Nystrom landmarks drawn from the data, random Fourier features, or orthogonal random features. */
  readonly features: FeatureKind
  /** The number of features `D` (landmarks for Nystrom; default 50). */
  readonly nComponents: number
  /** The kernel being approximated (the Fourier maps need a stationary one; default `rbfKernel()`). */
  readonly kernel: Kernel
  /** The ridge penalty (default 1). */
  readonly alpha: number
  readonly fitIntercept: boolean
  /** `'closed-form'` (the normal equations, default) or a value of `smoothSolvers.space`. */
  readonly solver: SolverChoice
  readonly tol: number
  readonly maxIter: number
}

/** The recipe of a drawn map, plain JSON. */
export type FeatureMapRecipe =
  | { readonly kind: 'nystroem'; readonly landmarks: number[][] }
  | {
      readonly kind: 'fourier'
      readonly weights: number[][]
      readonly offset: number[]
      readonly outputScale: number
    }

/** The state of {@link approximateKernelRidge}: the solver's, the map drawn, and the weights reached. */
export type ApproximateRidgeState = SlotState & {
  readonly map: FeatureMapRecipe
  /** The weights `D * k` then the intercepts. */
  readonly theta: number[]
  readonly objective: number
  readonly gradient: number
}

/** The fitted parameters of {@link approximateKernelRidge}: plain JSON. */
export type ApproximateRidgeParams = {
  readonly map: FeatureMapRecipe
  /** The weights `[D][k]`. */
  readonly weights: number[][]
  readonly intercept: number[]
  readonly multiOutput: boolean
  readonly nFeaturesIn: number
}

/** What a fitted feature-space ridge offers. */
export interface ApproximateRidgeVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  /** The features `z(x)` `[m, D]`. */
  readonly transform: (x: TensorLike) => Tensor
  readonly featureMap: FeatureMap
  readonly nFeaturesIn: number
}

const DEFAULTS: ApproximateRidgeHyper = {
  features: 'nystroem',
  nComponents: 50,
  kernel: rbfKernel(),
  alpha: 1,
  fitIntercept: false,
  solver: 'closed-form',
  tol: 1e-9,
  maxIter: 200,
}

function mapOf(recipe: FeatureMapRecipe, h: ApproximateRidgeHyper): FeatureMap {
  if (recipe.kind === 'nystroem') {
    return nystroem(h.kernel, recipe.landmarks, { indices: recipe.landmarks.map((_, i) => i) })
  }
  return randomFourierFeatures({
    weights: recipe.weights,
    offset: recipe.offset,
    outputScale: recipe.outputScale,
  })
}

/** The scalar length scale of an RBF kernel, the only one the orthogonal features sample. */
function rbfLengthScale(kernel: Kernel): number {
  const l = kernel.params['lengthScale']
  const values = l === undefined ? [] : toFlat(primalOf(asTensor(l as TensorLike)))
  if (kernel.name !== 'rbf' || values.length !== 1) {
    throw new MdmlError('value', 'approximateKernelRidge: orthogonal features need an isotropic rbf kernel')
  }
  return values[0] ?? 1
}

function drawMap(
  data: LearnData,
  h: ApproximateRidgeHyper,
  who: string,
  keyed: ReturnType<typeof resolveKey>,
): FeatureMapRecipe {
  const m = matrixView(data.x, who)
  const rows = fromStorage(m.data, [m.n, m.d], 'float64')
  if (h.features === 'nystroem') {
    const ny = nystroem(h.kernel, rows, { nComponents: Math.min(h.nComponents, m.n), key: keyed })
    return { kind: 'nystroem', landmarks: toArray(primalOf(ny.components)) as number[][] }
  }
  const rff: RandomFourierFeatures =
    h.features === 'fourier'
      ? randomFourierFeatures({ inputDim: m.d, nComponents: h.nComponents, kernel: h.kernel, key: keyed })
      : orthogonalRandomFeatures({
          inputDim: m.d,
          nComponents: h.nComponents,
          lengthScale: rbfLengthScale(h.kernel),
          key: keyed,
        })
  return {
    kind: 'fourier',
    weights: toArray(primalOf(rff.weights)) as number[][],
    offset: Array.from(toFlat(primalOf(rff.offset))),
    outputScale: rff.outputScale,
  }
}

/**
 * Kernel ridge regression in an approximate feature space: a Nystrom map (landmarks drawn from the data), random Fourier
 * features or orthogonal random features of a kernel, then ridge regression on the `D` features, which costs
 * `O(n D^2)` instead of the dual's `O(n^3)`. The ridge is a smooth objective: the fit takes `solver` from the slot
 * (`'closed-form'` solves the normal equations; any of `smoothSolvers` reaches the same weights). The tape logs the
 * objective and gradient of the solve; {@link ridgeGap} measures the approximation against exact kernel ridge.
 *
 * @param hyper - `features`, `nComponents`, `kernel`, `alpha`, `fitIntercept`, `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @concept kernel-ridge
 * @reference A. Rahimi and B. Recht, Random features for large-scale kernel machines (NeurIPS 2007); C. K. I. Williams
 *   and M. Seeger, Using the Nystrom method to speed up kernel machines (NeurIPS 2001).
 * @example
 * ```ts
 * import { approximateKernelRidge } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const fit = approximateKernelRidge({ nComponents: 6, alpha: 1e-3 }).fit({ x, y: [0, 1, 0, -1, 0, 1] })
 * toFlat(fit.predict([[2]])).map((v) => Math.round(v * 10) / 10) // => [0]
 * ```
 */
export function approximateKernelRidge(
  hyper: Partial<ApproximateRidgeHyper> = {},
): IterativeEstimator<ApproximateRidgeHyper, ApproximateRidgeState, ApproximateRidgeParams, ApproximateRidgeVerbs> {
  const who = 'approximateKernelRidge'
  const verbsOf = (p: ApproximateRidgeParams, h: ApproximateRidgeHyper): ApproximateRidgeVerbs => {
    const map = mapOf(p.map, h)
    const w = Float64Array.from(p.weights.flat())
    const b = Float64Array.from(p.intercept)
    const predict = (x: TensorLike): Tensor => predictFeatures(map, x, w, b, p.multiOutput)
    return {
      predict,
      score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
      transform: (x) => primalOf(map.transform(x)),
      featureMap: map,
      nFeaturesIn: p.nFeaturesIn,
    }
  }
  return iterative<ApproximateRidgeHyper, ApproximateRidgeState, ApproximateRidgeParams, ApproximateRidgeVerbs>(
    {
      name: 'kernel-methods/approximate-kernel-ridge',
      kind: 'regressor',
      defaults: DEFAULTS,
      hyperSpace: (h) =>
        dict({
          features: choice(['nystroem', 'fourier', 'orthogonal-fourier'], {
            default: h.features,
            label: 'feature map',
          }),
          nComponents: int(1, Math.max(500, h.nComponents), { default: h.nComponents, label: 'features' }),
          alpha: real(Math.min(1e-6, h.alpha), Math.max(1e3, h.alpha), { log: true, default: h.alpha, label: 'alpha' }),
          ...(h.solver === 'closed-form' ? {} : { solver: smoothSolvers.space }),
        }),
      algorithm: (data, h, key) => {
        const m = matrixView(data.x, who)
        const t = targetRows(data.y ?? [], who)
        if (t.n !== m.n) throw new MdmlError('shape', `${who}: x has ${m.n} rows but y has ${t.n}`)
        checkAlpha(h.alpha, who)
        const recipe = drawMap(data, h, who, resolveKey({ key }, 'approximateKernelRidge'))
        const z = Float64Array.from(
          toFlat(primalOf(mapOf(recipe, h).transform(fromStorage(m.data, [m.n, m.d], 'float64')))),
        )
        const d = z.length / m.n
        const smooth = ridgeSmooth(z, t.rows, m.n, d, t.k, h.alpha, h.fitIntercept)
        const p = d * t.k + (h.fitIntercept ? t.k : 0)
        return slotted<ApproximateRidgeState>({
          name: 'approximate-kernel-ridge',
          config: { features: h.features, nComponents: h.nComponents, alpha: h.alpha, fitIntercept: h.fitIntercept },
          schema: {
            metrics: {
              objective: { description: 'the ridge objective in feature space', direction: 'lower' },
              gradient: { description: 'the largest gradient entry', direction: 'lower' },
            },
            tensors: {},
          },
          problem: problemOf(smooth, new Array<number>(p).fill(0)),
          smooth,
          solver: h.solver,
          closed: () => {
            const s = ridgeSolve(
              fromStorage(z, [m.n, d], 'float64'),
              fromStorage(t.rows, [m.n, t.k], 'float64'),
              h.alpha,
              h.fitIntercept,
            )
            return h.fitIntercept ? [...s.weights, ...s.intercept] : Array.from(s.weights)
          },
          key,
          tol: h.tol,
          view: (r) => ({ map: recipe, theta: Array.from(r.x), objective: r.value, gradient: infNorm(r.grad) }),
          observe: (s) => ({ metrics: { objective: s.objective, gradient: s.gradient } }),
        })
      },
      maxSteps: (h) => (h.solver === 'closed-form' ? 1 : h.maxIter),
      params: (s, data, h) => {
        const m = matrixView(data.x, who)
        const t = targetRows(data.y ?? [], who)
        const d = s.theta.length / t.k - (h.fitIntercept ? 1 : 0)
        const weights = Array.from({ length: d }, (_, l) => s.theta.slice(l * t.k, (l + 1) * t.k))
        return {
          map: s.map,
          weights,
          intercept: h.fitIntercept ? s.theta.slice(d * t.k) : new Array<number>(t.k).fill(0),
          multiOutput: t.multi,
          nFeaturesIn: m.d,
        }
      },
      verbs: verbsOf,
      logs: { 'solver/objective': Log(), 'solver/gradient': Log() },
    },
    { ...DEFAULTS, ...hyper },
  )
}
