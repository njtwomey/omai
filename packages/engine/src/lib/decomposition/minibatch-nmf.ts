/**
 * Mini-batch non-negative matrix factorisation (Lefevre and Bach's online NMF, scikit-learn's `MiniBatchNMF`): each
 * step takes the next batch of rows, updates their weights `W` by a multiplicative step, and updates the components
 * `H` from running averages of the numerator and denominator of the multiplicative rule, discounted by a forgetting
 * factor, so the data never has to be held at once. Any beta divergence (Frobenius, Kullback-Leibler, Itakura-Saito or a
 * number), l1 and l2 penalties, fresh restarts of W, and scikit-learn's two stopping rules (a small change of `H`, and
 * no improvement of the smoothed batch cost).
 */

import { OmaiError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import type { Key } from '../random/index.ts'
import { dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { mm, tr } from './_linear.ts'
import { initialNmfFactors, type NmfHyper } from './nmf.ts'

type Rows = readonly (readonly number[])[]

// scikit-learn's guard against zero denominators (float32 machine epsilon).
const EPSILON = 1.1920928955078125e-7

/** The beta divergence: a name (`frobenius` is beta 2, `kullback-leibler` 1, `itakura-saito` 0) or a number. */
export type BetaLoss = 'frobenius' | 'kullback-leibler' | 'itakura-saito' | number

const betaOf = (loss: BetaLoss): number =>
  loss === 'frobenius' ? 2 : loss === 'kullback-leibler' ? 1 : loss === 'itakura-saito' ? 0 : loss

const gammaOf = (beta: number): number => (beta < 1 ? 1 / (2 - beta) : beta > 2 ? 1 / (beta - 1) : 1)

const sumOf = (m: Rows): number => m.reduce((a, r) => a + r.reduce((b, v) => b + v, 0), 0)

/** The squared Frobenius norm of the difference of two row arrays divided by the norm of the second: scikit-learn's relative change. */
const relativeChange = (a: Rows, b: Rows): number => {
  let diff = 0
  let norm = 0
  for (const [i, row] of a.entries())
    for (const [j, v] of row.entries()) {
      diff += (v - (b[i]?.[j] ?? 0)) ** 2
      norm += v * v
    }
  return Math.sqrt(diff) / Math.sqrt(norm)
}

/**
 * The beta divergence between X and W H (scikit-learn's `_beta_divergence`): half the squared Frobenius distance for
 * beta 2, the generalised Kullback-Leibler for 1, Itakura-Saito for 0, and the general formula otherwise. With
 * `squareRoot`, `sqrt(2 d)`, which is the Frobenius norm for beta 2.
 *
 * @param x - Non-negative data `[n, d]`.
 * @param w - Weights `[n, k]`.
 * @param h - Components `[k, d]`.
 * @param loss - The divergence.
 * @param squareRoot - Return `sqrt(2 d)` (default false).
 * @returns The divergence.
 * @example
 * ```ts
 * import { betaDivergence } from 'omai-engine/decomposition'
 * betaDivergence([[1, 2]], [[1]], [[1, 1]], 'frobenius') // => 0.5
 * betaDivergence([[1, 2]], [[1]], [[1, 1]], 'frobenius', true) // => 1
 * ```
 */
export function betaDivergence(x: Rows, w: Rows, h: Rows, loss: BetaLoss, squareRoot = false): number {
  const beta = betaOf(loss)
  const wh = mm(w, h)
  let res = 0
  if (beta === 2) {
    for (const [i, row] of x.entries()) for (const [j, v] of row.entries()) res += (v - (wh[i]?.[j] ?? 0)) ** 2 / 2
  } else {
    let sumX = 0
    let sumWhBeta = 0
    let count = 0
    let crossLog = 0
    let power = 0
    let powerX = 0
    for (const [i, row] of x.entries())
      for (const [j, v] of row.entries()) {
        const y = wh[i]?.[j] ?? 0
        sumWhBeta += y ** beta
        count += 1
        if (v <= EPSILON) continue
        const safe = Math.max(y, EPSILON)
        sumX += v
        crossLog += v * Math.log(v / safe)
        power += safe ** (beta - 1) * v
        powerX += v ** beta
        if (beta === 0) res += v / safe - Math.log(v / safe)
      }
    if (beta === 1) res = crossLog + sumOf(wh) - sumX
    else if (beta === 0) res -= count
    else res = (powerX - beta * power + sumWhBeta * (beta - 1)) / (beta * (beta - 1))
  }
  return squareRoot ? Math.sqrt(2 * Math.max(res, 0)) : res
}

/** The penalties of scikit-learn's `_compute_regularization` for data with `n` rows and `d` columns. */
function penalties(
  n: number,
  d: number,
  alphaW: number,
  alphaH: number,
  l1Ratio: number,
): {
  readonly l1W: number
  readonly l1H: number
  readonly l2W: number
  readonly l2H: number
} {
  return {
    l1W: d * alphaW * l1Ratio,
    l1H: n * alphaH * l1Ratio,
    l2W: d * alphaW * (1 - l1Ratio),
    l2H: n * alphaH * (1 - l1Ratio),
  }
}

/** The numerator and denominator `X'` of the multiplicative rule for W or H: `(X (WH)^(b-2)) H^T` over `(WH)^(b-1) H^T`. */
function ratioTerms(x: Rows, w: Rows, h: Rows, beta: number): { num: number[][]; den: number[][] | null } {
  if (beta === 2) return { num: [], den: null }
  const wh = mm(w, h)
  const safeX = wh.map((row, i) =>
    row.map((v, j) => {
      const y = beta - 2 < 0 ? Math.max(v, EPSILON) : v
      const xv = x[i]?.[j] ?? 0
      if (beta === 1) return xv / y
      if (beta === 0) return xv * y ** -2
      return xv * y ** (beta - 2)
    }),
  )
  const powered = wh.map((row) => row.map((v) => (beta - 1 < 0 ? Math.max(v, EPSILON) : v) ** (beta - 1)))
  return { num: safeX, den: powered }
}

/**
 * One multiplicative update of W for fixed H (scikit-learn's `_multiplicative_update_w`).
 *
 * @param x - The rows `[b, d]`.
 * @param w - Their weights `[b, k]`.
 * @param h - Components `[k, d]`.
 * @param beta - The beta of the divergence.
 * @param l1 - The l1 penalty.
 * @param l2 - The l2 penalty.
 * @returns The new weights.
 */
function updateW(x: Rows, w: Rows, h: Rows, beta: number, l1: number, l2: number): number[][] {
  const gamma = gammaOf(beta)
  let numerator: number[][]
  let denominator: number[][]
  if (beta === 2) {
    numerator = mm(x, tr(h))
    denominator = mm(w, mm(h, tr(h)))
  } else {
    const t = ratioTerms(x, w, h, beta)
    numerator = mm(t.num, tr(h))
    if (beta === 1) {
      const hSum = h.map((r) => r.reduce((a, v) => a + v, 0))
      denominator = w.map(() => [...hSum])
    } else denominator = mm(t.den as number[][], tr(h))
  }
  return w.map((row, i) =>
    row.map((v, c) => {
      let den = denominator[i]?.[c] ?? 0
      if (l1 > 0) den += l1
      if (l2 > 0) den += l2 * v
      if (den === 0) den = EPSILON
      const delta = (numerator[i]?.[c] ?? 0) / den
      return v * (gamma === 1 ? delta : delta ** gamma)
    }),
  )
}

/** The accumulators of the online update of H: `A` (numerator) and `B` (denominator), and the forgetting `rho`. */
interface Accumulators {
  readonly a: number[][]
  readonly b: number[][]
  readonly rho: number
}

/**
 * One online multiplicative update of H from a batch (scikit-learn's `_multiplicative_update_h` with `A`, `B`, `rho`).
 *
 * @param x - The rows `[b, d]`.
 * @param w - Their weights `[b, k]`.
 * @param h - Components `[k, d]`.
 * @param beta - The beta of the divergence.
 * @param l1 - The l1 penalty.
 * @param l2 - The l2 penalty.
 * @param acc - The running numerator, denominator and forgetting factor.
 * @returns The new components and accumulators.
 */
function updateH(
  x: Rows,
  w: Rows,
  h: Rows,
  beta: number,
  l1: number,
  l2: number,
  acc: Accumulators,
): { h: number[][]; a: number[][]; b: number[][] } {
  const gamma = gammaOf(beta)
  let numerator: number[][]
  let denominator: number[][]
  if (beta === 2) {
    numerator = mm(tr(w), x)
    denominator = mm(mm(tr(w), w), h)
  } else {
    const t = ratioTerms(x, w, h, beta)
    numerator = mm(tr(w), t.num)
    if (beta === 1) {
      const wSum = (w[0] ?? []).map((_, c) => {
        const s = w.reduce((a, r) => a + (r[c] ?? 0), 0)
        return s === 0 ? 1 : s
      })
      denominator = h.map((row, c) => row.map(() => wSum[c] ?? 1))
    } else denominator = mm(tr(w), t.den as number[][])
  }
  const a = acc.a.map((row) => [...row])
  const b = acc.b.map((row) => [...row])
  const next = h.map((row, c) =>
    row.map((v, j) => {
      let den = denominator[c]?.[j] ?? 0
      if (l1 > 0) den += l1
      if (l2 > 0) den += l2 * v
      if (den === 0) den = EPSILON
      const base = gamma === 1 ? v : v ** (1 / gamma)
      const num = (numerator[c]?.[j] ?? 0) * base
      const aa = (a[c]?.[j] ?? 0) * acc.rho + num
      const bb = (b[c]?.[j] ?? 0) * acc.rho + den
      ;(a[c] as number[])[j] = aa
      ;(b[c] as number[])[j] = bb
      const out = aa / bb
      return gamma === 1 ? out : out ** gamma
    }),
  )
  return { h: next, a, b }
}

const mean = (x: Rows): number => sumOf(x) / Math.max(1, x.length * (x[0]?.length ?? 0))

/**
 * The weights of rows for fixed components (scikit-learn's `_solve_W`, the heart of `transform` and of fresh
 * restarts): multiplicative updates of W from the constant `sqrt(mean(x) / k)` until its relative change is at most
 * `tol` or `maxIter` updates have run.
 *
 * @param x - The rows `[n, d]`.
 * @param h - Components `[k, d]`.
 * @param options - The divergence, the penalties, the tolerance and the update budget.
 * @returns The weights `[n, k]`.
 * @example
 * ```ts
 * import { solveWeights } from 'omai-engine/decomposition'
 * solveWeights([[2, 4]], [[1, 2]], { maxIter: 200, tol: 1e-9 }).map((r) => r[0].toFixed(4)) // => ['2.0000']
 * ```
 */
export function solveWeights(
  x: Rows,
  h: Rows,
  options: {
    readonly betaLoss?: BetaLoss
    readonly alphaW?: number
    readonly l1Ratio?: number
    readonly maxIter: number
    readonly tol: number
  },
): number[][] {
  const beta = betaOf(options.betaLoss ?? 'frobenius')
  const { l1W, l2W } = penalties(x.length, x[0]?.length ?? 0, options.alphaW ?? 0, 0, options.l1Ratio ?? 0)
  const k = h.length
  const avg = Math.sqrt(mean(x) / k)
  let w: number[][] = x.map(() => new Array<number>(k).fill(avg))
  let previous = w
  for (let it = 0; it < options.maxIter; it++) {
    w = updateW(x, w, h, beta, l1W, l2W)
    if (options.tol > 0 && relativeChange(w, previous) <= options.tol) break
    previous = w
  }
  return w
}

/** Options of {@link miniBatchNmfAlgorithm} (scikit-learn's names and defaults). */
export interface MiniBatchNmfOptions {
  /** The divergence (default `frobenius`). */
  readonly betaLoss?: BetaLoss
  /** Rows per batch (default 1024, at most the number of rows). */
  readonly batchSize?: number
  /** Stop when the relative change of H over a step is at most this (default 1e-4; 0: never). */
  readonly tol?: number
  /** Stop after this many steps without an improvement of the smoothed batch cost (default 10; `null`: never). */
  readonly maxNoImprovement?: number | null
  /** The budget in passes over the data (default 200). */
  readonly maxIter?: number
  /** The penalty on W (default 0), on H (default: that of W), and the share of it that is l1 (default 0). */
  readonly alphaW?: number
  readonly alphaH?: number | 'same'
  readonly l1Ratio?: number
  /** How fast the past of H is forgotten, per pass over the data (default 0.7). */
  readonly forgetFactor?: number
  /** Solve W from scratch for each batch instead of updating it (default false), with this many updates (default 30). */
  readonly freshRestarts?: boolean
  readonly freshRestartsMaxIter?: number
}

/**
 * The state of mini-batch NMF: the factors, the running numerator and denominator of the update of H, and what the
 * stopping rules track. Unset costs are `-1`.
 */
export type MiniBatchNmfState = {
  readonly w: number[][]
  readonly h: number[][]
  readonly numerator: number[][]
  readonly denominator: number[][]
  readonly step: number
  readonly batchCost: number
  readonly ewaCost: number
  readonly ewaCostMin: number
  readonly noImprovement: number
  /** The relative change of H over the last step. */
  readonly hChange: number
  readonly converged: boolean
}

const miniBatchNmfSchema = (k: number, d: number, n: number): ReturnType<typeof channels> =>
  channels({
    metrics: {
      batchCost: { description: 'the mean cost (divergence and penalties) of the last batch', direction: 'lower' },
      hChange: { description: 'the relative change of the components over the last step' },
      noImprovement: { description: 'steps since the smoothed batch cost last improved' },
    },
    tensors: {
      components: { role: 'state', description: 'the non-negative components H', shape: [k, d] },
      weights: { role: 'state', description: 'the non-negative weights W of all rows', shape: [n, k] },
    },
  })

/**
 * Mini-batch NMF as a traced algorithm, one batch per step, as scikit-learn's `MiniBatchNMF` does it: the batches in
 * order and cycling, W of the batch updated (or solved afresh) by the multiplicative rule, the batch cost recorded, then
 * H by `H = A / B` with `A` and `B` the running sums (discounted by `rho = forgetFactor^(batch / n)`) of the
 * numerator and denominator of the multiplicative rule. It ends when H changes by a relative `tol` or less, when the
 * exponentially smoothed batch cost has not improved for `maxNoImprovement` steps, or after `maxIter` passes.
 *
 * @param x - Non-negative data `[n, d]`.
 * @param w0 - Initial weights `[n, k]`.
 * @param h0 - Initial components `[k, d]`.
 * @param options - See {@link MiniBatchNmfOptions}.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for negative data, or zeros in the data with a beta of at most 0.
 * @index algorithm Online NMF, one mini-batch of rows per step, with forgetting.
 * @example
 * ```ts
 * import { miniBatchNmfAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [2, 4], [3, 6], [4, 8]]
 * const out = runAlgorithm(miniBatchNmfAlgorithm(x, [[1], [1], [1], [1]], [[1, 1]], { batchSize: 2, tol: 0 }), { maxSteps: 60 })
 * (out.state.h[0][1] / out.state.h[0][0]).toFixed(3) // => '2.000'
 * ```
 */
export function miniBatchNmfAlgorithm(
  x: Rows,
  w0: Rows,
  h0: Rows,
  options: MiniBatchNmfOptions = {},
): Algorithm<MiniBatchNmfState> {
  const beta = betaOf(options.betaLoss ?? 'frobenius')
  const n = x.length
  const d = x[0]?.length ?? 0
  const k = h0.length
  if (x.some((r) => r.some((v) => v < 0))) throw new OmaiError('value', 'miniBatchNmf: the data must be non-negative')
  if (beta <= 0 && x.some((r) => r.some((v) => v === 0)))
    throw new OmaiError('value', 'miniBatchNmf: with a beta of at most 0 the data must have no zeros')
  const batch = Math.min(Math.max(1, Math.floor(options.batchSize ?? 1024)), n)
  const perPass = Math.ceil(n / batch)
  const maxIter = options.maxIter ?? 200
  const tol = options.tol ?? 1e-4
  const maxNoImprovement = options.maxNoImprovement === undefined ? 10 : options.maxNoImprovement
  const alphaW = options.alphaW ?? 0
  const alphaH = options.alphaH === undefined || options.alphaH === 'same' ? alphaW : options.alphaH
  const l1Ratio = options.l1Ratio ?? 0
  const rho = (options.forgetFactor ?? 0.7) ** (batch / n)
  const fresh = options.freshRestarts ?? false
  const freshIter = options.freshRestartsMaxIter ?? 30
  const budget = maxIter * perPass
  return {
    name: 'mini-batch-nmf',
    schema: miniBatchNmfSchema(k, d, n),
    config: { beta, batchSize: batch, tol, maxIter, forgetFactor: options.forgetFactor ?? 0.7, freshRestarts: fresh },
    init: () => ({
      w: w0.map((r) => [...r]),
      h: h0.map((r) => [...r]),
      numerator: h0.map((r) => [...r]),
      denominator: h0.map((r) => r.map(() => 1)),
      step: 0,
      batchCost: -1,
      ewaCost: -1,
      ewaCostMin: -1,
      noImprovement: 0,
      hChange: Number.MAX_VALUE,
      converged: false,
    }),
    step: (s) => {
      const pass = s.step % perPass
      const from = pass * batch
      const to = Math.min(n, from + batch)
      const xb = x.slice(from, to)
      const pen = penalties(xb.length, d, alphaW, alphaH, l1Ratio)
      let wb: number[][] =
        fresh || s.w.length === 0
          ? solveWeights(xb, s.h, { betaLoss: beta, alphaW, l1Ratio, maxIter: freshIter, tol })
          : updateW(xb, s.w.slice(from, to), s.h, beta, pen.l1W, pen.l2W)
      if (beta < 1) wb = wb.map((r) => r.map((v) => (v < Number.EPSILON ? 0 : v)))
      const batchCost =
        (betaDivergence(xb, wb, s.h, beta) +
          pen.l1W * sumOf(wb) +
          pen.l1H * sumOf(s.h) +
          pen.l2W * wb.reduce((a, r) => a + r.reduce((b, v) => b + v * v, 0), 0) +
          pen.l2H * s.h.reduce((a, r) => a + r.reduce((b, v) => b + v * v, 0), 0)) /
        xb.length
      const updated = updateH(xb, wb, s.h, beta, pen.l1H, pen.l2H, { a: s.numerator, b: s.denominator, rho })
      let h = updated.h
      if (beta <= 1) h = h.map((r) => r.map((v) => (v < Number.EPSILON ? 0 : v)))
      const w = s.w.map((r, i) => (i >= from && i < to ? [...(wb[i - from] as number[])] : r))
      const step = s.step + 1
      const hChange = relativeChange(h, s.h)
      let ewaCost = s.ewaCost
      let ewaCostMin = s.ewaCostMin
      let noImprovement = s.noImprovement
      let converged = false
      if (step > 1) {
        const alpha = Math.min(xb.length / (n + 1), 1)
        ewaCost = ewaCost < 0 ? batchCost : ewaCost * (1 - alpha) + batchCost * alpha
        if (tol > 0 && hChange <= tol) converged = true
        else {
          if (ewaCostMin < 0 || ewaCost < ewaCostMin) {
            noImprovement = 0
            ewaCostMin = ewaCost
          } else noImprovement += 1
          if (maxNoImprovement !== null && noImprovement >= maxNoImprovement) converged = true
        }
      }
      return {
        w,
        h,
        numerator: updated.a,
        denominator: updated.b,
        step,
        batchCost,
        ewaCost,
        ewaCostMin,
        noImprovement,
        hChange,
        converged,
      }
    },
    done: (s) => (s.converged ? 'converged' : s.step >= budget ? 'completed' : undefined),
    observe: (s) => ({
      metrics:
        s.step === 0
          ? {}
          : {
              batchCost: s.batchCost,
              hChange: s.hChange === Number.MAX_VALUE ? -1 : s.hChange,
              noImprovement: s.noImprovement,
            },
      tensors: {
        components: { role: 'state', value: s.h.flat(), shape: [k, d] },
        weights: { role: 'state', value: s.w.flat(), shape: [n, k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MiniBatchNmfState,
  }
}

/** Hyperparameters of {@link miniBatchNmf}. */
export interface MiniBatchNmfHyper {
  readonly nComponents: number
  /** How W and H start: `nndsvda` (default), `random`, or the given `wInit` and `hInit`. */
  readonly init: NmfHyper['init']
  readonly wInit: number[][] | null
  readonly hInit: number[][] | null
  readonly betaLoss: BetaLoss
  readonly batchSize: number
  readonly tol: number
  readonly maxNoImprovement: number | null
  readonly maxIter: number
  readonly alphaW: number
  readonly alphaH: number | 'same'
  readonly l1Ratio: number
  readonly forgetFactor: number
  readonly freshRestarts: boolean
  readonly freshRestartsMaxIter: number
  /** The update budget of `transform` (default: `maxIter`). */
  readonly transformMaxIter: number | null
}

/** The fitted parameters of {@link miniBatchNmf}. */
export type MiniBatchNmfParams = {
  readonly components: number[][]
  /** The weights of the training rows. */
  readonly weights: number[][]
  /** The square root of twice the beta divergence of the training data (scikit-learn's `reconstruction_err_`). */
  readonly reconstructionErr: number
  /** The mini-batch steps taken (scikit-learn's `n_steps_`). */
  readonly nSteps: number
  /** The passes over the data begun (scikit-learn's `n_iter_`). */
  readonly nIter: number
  readonly betaLoss: BetaLoss
  readonly alphaW: number
  readonly l1Ratio: number
  readonly transformMaxIter: number
  readonly tol: number
}

/** The verbs of a fitted mini-batch NMF. */
export interface MiniBatchNmfVerbs {
  /** The non-negative weights of new rows for the fitted components. */
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (w: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

const rowsOf = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}

const toTensor = (rows: number[][]): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

const startOf = (data: { readonly x: TensorLike }, h: MiniBatchNmfHyper, key: Key): { w: number[][]; h: number[][] } =>
  initialNmfFactors(rowsOf(data.x, 'miniBatchNmf'), h, key)

/**
 * Mini-batch NMF (`sklearn.decomposition.MiniBatchNMF`) on the iterative base: one batch per step, so a session shows
 * the components settling as batches arrive and `fittedAt(t)` is the model after `t` batches. `transform` solves the
 * weights of new rows for the fitted components.
 *
 * @index estimator Online NMF over mini-batches with forgetting; any beta divergence.
 * @example
 * ```ts
 * import { miniBatchNmf } from 'omai-engine/decomposition'
 * const x = [[1, 2, 0], [2, 4, 0], [0, 0, 3], [0, 0, 6], [1, 2, 0], [0, 0, 3]]
 * const f = miniBatchNmf.with({ nComponents: 2, batchSize: 3, maxIter: 100 }).fit({ x })
 * f.params.components.length // => 2
 * ```
 */
export const miniBatchNmf: IterativeEstimator<
  MiniBatchNmfHyper,
  MiniBatchNmfState,
  MiniBatchNmfParams,
  MiniBatchNmfVerbs
> = iterative<MiniBatchNmfHyper, MiniBatchNmfState, MiniBatchNmfParams, MiniBatchNmfVerbs>({
  name: 'decomposition/mini-batch-nmf',
  kind: 'transformer',
  defaults: {
    nComponents: 2,
    init: 'nndsvda',
    wInit: null,
    hInit: null,
    betaLoss: 'frobenius',
    batchSize: 1024,
    tol: 1e-4,
    maxNoImprovement: 10,
    maxIter: 200,
    alphaW: 0,
    alphaH: 'same',
    l1Ratio: 0,
    forgetFactor: 0.7,
    freshRestarts: false,
    freshRestartsMaxIter: 30,
    transformMaxIter: null,
  },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
      batchSize: int(1, 4096, { default: h.batchSize, label: 'batch size', log: true }),
    }),
  algorithm: (data, h, key) => {
    const x = rowsOf(data.x, 'miniBatchNmf')
    const start = startOf(data, h, key)
    return miniBatchNmfAlgorithm(x, start.w, start.h, {
      betaLoss: h.betaLoss,
      batchSize: h.batchSize,
      tol: h.tol,
      maxNoImprovement: h.maxNoImprovement,
      maxIter: h.maxIter,
      alphaW: h.alphaW,
      alphaH: h.alphaH,
      l1Ratio: h.l1Ratio,
      forgetFactor: h.forgetFactor,
      freshRestarts: h.freshRestarts,
      freshRestartsMaxIter: h.freshRestartsMaxIter,
    })
  },
  maxSteps: (h, data) => {
    const n = matrixView(data.x, 'miniBatchNmf').n
    return h.maxIter * Math.ceil(n / Math.min(Math.max(1, h.batchSize), n))
  },
  params: (s, data, h) => {
    const x = rowsOf(data.x, 'miniBatchNmf')
    const n = x.length
    const batch = Math.min(Math.max(1, Math.floor(h.batchSize)), n)
    const perPass = Math.ceil(n / batch)
    const transformMaxIter = h.transformMaxIter ?? h.maxIter
    const w = h.freshRestarts
      ? solveWeights(x, s.h, {
          betaLoss: h.betaLoss,
          alphaW: h.alphaW,
          l1Ratio: h.l1Ratio,
          maxIter: transformMaxIter,
          tol: h.tol,
        })
      : s.w
    return {
      components: s.h,
      weights: w,
      reconstructionErr: betaDivergence(x, w, s.h, h.betaLoss, true),
      nSteps: s.step,
      nIter: Math.ceil(s.step / perPass),
      betaLoss: h.betaLoss,
      alphaW: h.alphaW,
      l1Ratio: h.l1Ratio,
      transformMaxIter,
      tol: h.tol,
    }
  },
  verbs: (p) => {
    const d = p.components[0]?.length ?? 0
    const k = p.components.length
    return {
      transform: (x) =>
        toTensor(
          solveWeights(rowsOf(x, 'miniBatchNmf transform', d), p.components, {
            betaLoss: p.betaLoss,
            alphaW: p.alphaW,
            l1Ratio: p.l1Ratio,
            maxIter: p.transformMaxIter,
            tol: p.tol,
          }),
        ),
      inverseTransform: (w) => toTensor(mm(rowsOf(w, 'miniBatchNmf inverseTransform', k), p.components)),
      nFeaturesIn: d,
      featureNamesOut: () => p.components.map((_, c) => `minibatchnmf${c}`),
    }
  },
})
