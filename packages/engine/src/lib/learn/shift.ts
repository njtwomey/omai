/**
 * Label-shift estimation and correction: the target's class priors without target labels by Lipton's Black-Box Shift
 * Estimation (`blackBoxShiftEstimate`) or the Saerens-Latinne-Decaestecker EM (`emPriorAdjustmentAlgorithm`,
 * `emPriorAdjustment`), and the posteriors adjusted to them (`adjustPriors`). Covariate shift is in `density-ratio.ts`.
 */

import { argmaxOf, MdmlError } from '../base/index.ts'
import { dgemm, dgemv, pinv, solve } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { runOptionsFor } from './_shared.ts'
import type { FitOptions } from './estimator.ts'
import { encodeLabels, integerOption, matrixView, targetVector } from './validation.ts'

/**
 * Recalibrate source posterior probabilities `p_S(y | x)` to target class priors `pi_T` under label shift
 * (`p_T(x | y) = p_S(x | y)`): `p_T(y = c | x) proportional to p_S(y = c | x) * (pi_T(c) / pi_S(c))`.
 *
 * @param sourceProba - Source posterior probabilities `[n, k]`.
 * @param sourcePriors - Source class priors `[k]`.
 * @param targetPriors - Target class priors `[k]`.
 * @returns Recalibrated target posterior probabilities `[n, k]`.
 * @concept label-shift
 * @example
 * ```ts
 * import { adjustPriors } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(adjustPriors([[0.5, 0.5]], [0.5, 0.5], [0.2, 0.8])) // => [0.2, 0.8]
 * ```
 */
export function adjustPriors(
  sourceProba: TensorLike,
  sourcePriors: ArrayLike<number>,
  targetPriors: ArrayLike<number>,
): Tensor<'float64'> {
  const p = matrixView(sourceProba, 'adjustPriors')
  const piS = targetVector(sourcePriors as TensorLike, 'adjustPriors', p.d)
  const piT = targetVector(targetPriors as TensorLike, 'adjustPriors', p.d)
  const ratios = Float64Array.from({ length: p.d }, (_, c) => {
    const s = piS[c] ?? 0
    const t = piT[c] ?? 0
    if (s < 0 || t < 0) throw new MdmlError('value', 'adjustPriors: priors must be >= 0')
    return s > 1e-15 ? t / s : 0
  })
  const out = new Float64Array(p.n * p.d)
  for (let i = 0; i < p.n; i++) {
    let z = 0
    for (let c = 0; c < p.d; c++) {
      const u = (p.data[i * p.d + c] ?? 0) * (ratios[c] ?? 0)
      out[i * p.d + c] = u
      z += u
    }
    for (let c = 0; c < p.d; c++) {
      out[i * p.d + c] = z > 0 ? (out[i * p.d + c] ?? 0) / z : (piT[c] ?? 1 / p.d)
    }
  }
  return fromStorage(out, [p.n, p.d], 'float64')
}

/** Options of {@link blackBoxShiftEstimate}. */
export interface BbseOptions {
  /** Whether to form the hard label confusion matrix or soft probability confusion matrix (default `'auto'`). */
  readonly method?: 'auto' | 'hard' | 'soft'
  /** Explicit class list `[k]` when some classes may be absent in `sourceTrue`. */
  readonly classes?: readonly number[]
  /** Ridge regularisation added to `C` before inversion when near-singular (default 1e-10). */
  readonly regularisation?: number
}

/** Result of {@link blackBoxShiftEstimate}. */
export interface BbseResult {
  /** Ordered class labels `[k]`. */
  readonly classes: readonly number[]
  /** Empirical source class priors `pi_S` `[k]`. */
  readonly sourcePriors: readonly number[]
  /** Estimated target class priors `pi_T` `[k]`. */
  readonly targetPriors: readonly number[]
  /** Estimated class importance weights `w_c = pi_T(c) / pi_S(c)` `[k]`. */
  readonly weights: readonly number[]
  /** Source joint confusion matrix `C[yhat, y]` `[k, k]`, where `C * w = mu_T`. */
  readonly confusionMatrix: Tensor<'float64'>
}

/**
 * Lipton's Black-Box Shift Estimation (BBSE; Lipton, Wang and Smola 2018): estimate target class priors `pi_T` and
 * importance weights `w = pi_T / pi_S` without target labels by inverting the source confusion matrix `C` against the
 * target predicted distribution `mu_T = C * w` (both hard-label BBSE and soft-probability BBSE-soft).
 *
 * @param sourceTrue - Source ground-truth labels `[n_source]`.
 * @param sourcePred - Source predictions: class labels `[n_source]` or class probabilities `[n_source, k]`.
 * @param targetPred - Target predictions: class labels `[n_target]` or class probabilities `[n_target, k]`.
 * @param options - `method` (`'auto'`, `'hard'` or `'soft'`), `classes` and `regularisation`.
 * @returns Estimated `sourcePriors`, `targetPriors`, importance `weights`, `confusionMatrix` and `classes`.
 * @concept label-shift
 * @example
 * ```ts
 * import { blackBoxShiftEstimate } from 'mdml-engine/learn'
 * const yS = [0, 0, 0, 0, 1, 1, 1, 1]
 * const pS = [0, 0, 0, 1, 1, 1, 1, 0]
 * const pT = [1, 1, 1, 1, 1, 0, 0, 1]
 * const r = blackBoxShiftEstimate(yS, pS, pT)
 * r.targetPriors[1] > 0.6 // => true
 * ```
 */
export function blackBoxShiftEstimate(
  sourceTrue: TensorLike,
  sourcePred: TensorLike,
  targetPred: TensorLike,
  options: BbseOptions = {},
): BbseResult {
  const enc = encodeLabels(sourceTrue, 'blackBoxShiftEstimate')
  const classes = options.classes !== undefined ? [...options.classes] : [...enc.classes]
  const k = classes.length
  if (k < 2) throw new MdmlError('value', 'blackBoxShiftEstimate: need at least 2 classes')
  const classIndex = new Map(classes.map((c, i) => [c, i]))
  const yS = targetVector(sourceTrue, 'blackBoxShiftEstimate')
  const nS = yS.length
  const yCodes = Int32Array.from(yS, (v) => {
    const idx = classIndex.get(v)
    if (idx === undefined) throw new MdmlError('value', `blackBoxShiftEstimate: unknown source label ${v}`)
    return idx
  })

  const sTensor = asTensor(sourcePred)
  const tTensor = asTensor(targetPred)
  const isProba = sTensor.shape.length === 2 && (sTensor.shape[1] ?? 0) === k && tTensor.shape.length === 2
  const useSoft = options.method === 'soft' || (options.method !== 'hard' && isProba)

  const C = new Float64Array(k * k)
  const sourceCounts = new Float64Array(k)
  for (let i = 0; i < nS; i++) {
    const y = yCodes[i] ?? 0
    sourceCounts[y] = (sourceCounts[y] ?? 0) + 1
  }
  const sourcePriors = Array.from(sourceCounts, (c) => c / nS)

  if (useSoft) {
    const sMat = matrixView(sTensor, 'blackBoxShiftEstimate', { nFeatures: k })
    if (sMat.n !== nS)
      throw new MdmlError('shape', 'blackBoxShiftEstimate: sourcePred and sourceTrue row counts differ')
    const tMat = matrixView(tTensor, 'blackBoxShiftEstimate', { nFeatures: k })
    for (let i = 0; i < nS; i++) {
      const y = yCodes[i] ?? 0
      for (let yhat = 0; yhat < k; yhat++) {
        C[yhat * k + y] = (C[yhat * k + y] ?? 0) + (sMat.data[i * k + yhat] ?? 0) / nS
      }
    }
    const muT = new Float64Array(k)
    for (let j = 0; j < tMat.n; j++) {
      for (let yhat = 0; yhat < k; yhat++) muT[yhat] = (muT[yhat] ?? 0) + (tMat.data[j * k + yhat] ?? 0) / tMat.n
    }
    return solveBbse(C, muT, sourcePriors, classes, options.regularisation ?? 1e-10)
  }

  const readHardPreds = (pred: TensorLike, expectedN?: number): Int32Array => {
    const t = asTensor(pred)
    if (t.shape.length === 2 && (t.shape[1] ?? 0) === k) {
      const m = matrixView(t, 'blackBoxShiftEstimate', { nFeatures: k })
      if (expectedN !== undefined && m.n !== expectedN) {
        throw new MdmlError('shape', 'blackBoxShiftEstimate: sourcePred and sourceTrue row counts differ')
      }
      return Int32Array.from({ length: m.n }, (_, i) => argmaxOf(m.data.subarray(i * k, (i + 1) * k)))
    }
    const v = targetVector(pred, 'blackBoxShiftEstimate', expectedN)
    return Int32Array.from(v, (lbl) => {
      const idx = classIndex.get(lbl)
      if (idx === undefined) throw new MdmlError('value', `blackBoxShiftEstimate: unknown predicted label ${lbl}`)
      return idx
    })
  }

  const sHard = readHardPreds(sourcePred, nS)
  const tHard = readHardPreds(targetPred)
  for (let i = 0; i < nS; i++) {
    const y = yCodes[i] ?? 0
    const yhat = sHard[i] ?? 0
    C[yhat * k + y] = (C[yhat * k + y] ?? 0) + 1 / nS
  }
  const muT = new Float64Array(k)
  for (let j = 0; j < tHard.length; j++) {
    const yhat = tHard[j] ?? 0
    muT[yhat] = (muT[yhat] ?? 0) + 1 / tHard.length
  }
  return solveBbse(C, muT, sourcePriors, classes, options.regularisation ?? 1e-10)
}

function solveBbse(
  C: Float64Array,
  muT: Float64Array,
  sourcePriors: readonly number[],
  classes: readonly number[],
  reg: number,
): BbseResult {
  const k = classes.length
  // Solve (C^T C + reg I) w = C^T mu_T via pinv when reg === 0 or solve on normal equations.
  const cTensor = fromStorage(C, [k, k], 'float64')
  let rawW: number[]
  if (reg <= 0) {
    const inv = pinv(cTensor)
    rawW = Array.from(dgemv(toFlat(inv), k, k, muT))
  } else {
    const ctc = dgemm(C, C, k, k, k, { transposeA: true })
    for (let c = 0; c < k; c++) ctc[c * k + c] = (ctc[c * k + c] ?? 0) + reg
    const rhs = dgemv(C, k, k, muT, true)
    rawW = toFlat(solve(fromStorage(ctc, [k, k], 'float64'), fromStorage(rhs, [k], 'float64')))
  }
  const clippedW = rawW.map((w) => Math.max(0, w))
  const unnormPriors = clippedW.map((w, c) => w * (sourcePriors[c] ?? 0))
  const sumPriors = unnormPriors.reduce((a, b) => a + b, 0)
  const targetPriors = sumPriors > 0 ? unnormPriors.map((p) => p / sumPriors) : [...sourcePriors]
  const weights = targetPriors.map((pt, c) => {
    const ps = sourcePriors[c] ?? 0
    return ps > 1e-15 ? pt / ps : 0
  })
  return {
    classes,
    sourcePriors,
    targetPriors,
    weights,
    confusionMatrix: cTensor,
  }
}

/** Options of {@link emPriorAdjustmentAlgorithm} and {@link emPriorAdjustment}. */
export interface EmPriorAdjustmentOptions extends FitOptions {
  /** Maximum EM iterations (default 100). */
  readonly maxIterations?: number
  /** Stopping tolerance on `max_c |pi^{(t+1)}_c - pi^{(t)}_c|` (default 1e-8). */
  readonly tol?: number
}

/** Serialisable state of {@link emPriorAdjustmentAlgorithm}. */
export interface EmPriorAdjustmentState {
  readonly iteration: number
  readonly priors: Float64Array
  readonly posteriors: Float64Array
  readonly change: number
  readonly logLikelihood: number
}

/** Result of {@link emPriorAdjustment}. */
export interface EmPriorAdjustmentResult {
  /** Source class priors `pi_S` `[k]`. */
  readonly sourcePriors: readonly number[]
  /** Maximum-likelihood target class priors `pi_T` `[k]`. */
  readonly targetPriors: readonly number[]
  /** Class importance weights `w_c = pi_T(c) / pi_S(c)` `[k]`. */
  readonly weights: readonly number[]
  /** Recalibrated target posterior probabilities `p_T(y | x)` `[n_target, k]`. */
  readonly posteriors: Tensor<'float64'>
  /** Number of EM iterations taken. */
  readonly iterations: number
  /** Whether the EM loop converged within `tol`. */
  readonly converged: boolean
}

function resolveSourcePriors(sourceProbaOrPriors: TensorLike, k: number): Float64Array {
  const t = asTensor(sourceProbaOrPriors)
  if (t.shape.length === 1 || (t.shape.length === 2 && t.shape[1] === 1 && t.shape[0] === k)) {
    const v = targetVector(t, 'emPriorAdjustment', k)
    const sum = v.reduce((a, b) => a + b, 0)
    if (!(sum > 0)) throw new MdmlError('value', 'emPriorAdjustment: sourcePriors must have a positive sum')
    return v.map((p) => Math.max(p / sum, 1e-12))
  }
  const m = matrixView(t, 'emPriorAdjustment', { nFeatures: k })
  const out = new Float64Array(k)
  for (let i = 0; i < m.n; i++) {
    for (let c = 0; c < k; c++) out[c] = (out[c] ?? 0) + (m.data[i * k + c] ?? 0) / m.n
  }
  const sum = out.reduce((a, b) => a + b, 0)
  return out.map((p) => Math.max(p / Math.max(sum, 1e-12), 1e-12))
}

/**
 * Saerens-Latinne-Decaestecker EM algorithm (Saerens, Latinne and Decaestecker 2002; Alexandari, Kundaje and Shrikumar
 * 2020 MLLS) as a stepped {@link Algorithm}: iteratively alternate between recalibrating target posteriors
 * `q_j(c) proportional to p_S(c | x_j) * (pi_T^{(t)}(c) / pi_S(c))` (E-step) and updating target priors
 * `pi_T^{(t+1)}(c) = mean_j q_j(c)` (M-step).
 *
 * @param sourceProbaOrPriors - Source class priors `[k]` or source predicted probabilities `[n_source, k]`.
 * @param targetProba - Target predicted probabilities `[n_target, k]` under the source classifier.
 * @param options - `maxIterations` (default 100), `tol` (default 1e-8) and trace options.
 * @returns Stepped {@link Algorithm} over {@link EmPriorAdjustmentState}.
 * @concept label-shift
 * @example
 * ```ts
 * import { emPriorAdjustmentAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const pT = [[0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.2, 0.8]]
 * const alg = emPriorAdjustmentAlgorithm([0.5, 0.5], pT)
 * runAlgorithm(alg, { maxSteps: 50 }).state.priors[0] > 0.65 // => true
 * ```
 */
export function emPriorAdjustmentAlgorithm(
  sourceProbaOrPriors: TensorLike,
  targetProba: TensorLike,
  options: EmPriorAdjustmentOptions = {},
): Algorithm<EmPriorAdjustmentState> {
  const target = matrixView(targetProba, 'emPriorAdjustment')
  const nT = target.n
  const k = target.d
  if (k < 2) throw new MdmlError('value', 'emPriorAdjustment: targetProba must have at least 2 classes')
  const sourcePriors = resolveSourcePriors(sourceProbaOrPriors, k)
  const tol = options.tol ?? 1e-8
  if (!(tol > 0)) throw new MdmlError('value', 'emPriorAdjustment: tol must be > 0')

  const eStep = (priors: Float64Array): { readonly posteriors: Float64Array; readonly ll: number } => {
    const posteriors = new Float64Array(nT * k)
    let ll = 0
    for (let j = 0; j < nT; j++) {
      let z = 0
      for (let c = 0; c < k; c++) {
        const u = (target.data[j * k + c] ?? 0) * ((priors[c] ?? 0) / (sourcePriors[c] ?? 1))
        posteriors[j * k + c] = u
        z += u
      }
      ll += Math.log(Math.max(z, 1e-300)) / nT
      for (let c = 0; c < k; c++) {
        posteriors[j * k + c] = z > 0 ? (posteriors[j * k + c] ?? 0) / z : (priors[c] ?? 1 / k)
      }
    }
    return { posteriors, ll }
  }

  return {
    name: 'em-prior-adjustment',
    config: { nTarget: nT, k, tol, maxIterations: options.maxIterations ?? 100 },
    schema: channels({
      metrics: {
        'em/iteration': { description: 'EM iteration count' },
        'em/change': { description: 'max change in estimated target priors', direction: 'lower' },
        'em/logLikelihood': { description: 'mean target log-likelihood', direction: 'higher' },
      },
      tensors: {
        'estimate/priors': { role: 'state', description: 'estimated target class priors', shape: ['k'] },
      },
    }),
    init: () => {
      const initPriors = Float64Array.from(sourcePriors)
      const { posteriors, ll } = eStep(initPriors)
      return {
        iteration: 0,
        priors: initPriors,
        posteriors,
        change: Number.POSITIVE_INFINITY,
        logLikelihood: ll,
      }
    },
    step: (s) => {
      const nextPriors = new Float64Array(k)
      for (let j = 0; j < nT; j++) {
        for (let c = 0; c < k; c++) nextPriors[c] = (nextPriors[c] ?? 0) + (s.posteriors[j * k + c] ?? 0) / nT
      }
      let change = 0
      for (let c = 0; c < k; c++) change = Math.max(change, Math.abs((nextPriors[c] ?? 0) - (s.priors[c] ?? 0)))
      const { posteriors, ll } = eStep(nextPriors)
      return {
        iteration: s.iteration + 1,
        priors: nextPriors,
        posteriors,
        change,
        logLikelihood: ll,
      }
    },
    done: (s) => (s.iteration > 0 && s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        'em/iteration': s.iteration,
        'em/change': Number.isFinite(s.change) ? s.change : 1,
        'em/logLikelihood': s.logLikelihood,
      },
      tensors: {
        'estimate/priors': { role: 'state', value: s.priors, shape: [k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EmPriorAdjustmentState,
  }
}

/**
 * Saerens-Latinne-Decaestecker EM for label-shift prior estimation and posterior recalibration (Saerens et al. 2002).
 *
 * @param sourceProbaOrPriors - Source class priors `[k]` or source predicted probabilities `[n_source, k]`.
 * @param targetProba - Target predicted probabilities `[n_target, k]` under the source classifier.
 * @param options - `maxIterations` (default 100), `tol` (default 1e-8) and trace options.
 * @returns Estimated `sourcePriors`, `targetPriors`, importance `weights`, recalibrated `posteriors` and convergence.
 * @concept label-shift
 * @example
 * ```ts
 * import { emPriorAdjustment } from 'mdml-engine/learn'
 * const pT = [[0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.15, 0.85]]
 * const r = emPriorAdjustment([0.5, 0.5], pT)
 * r.converged && r.targetPriors[0] > 0.65 // => true
 * ```
 */
export function emPriorAdjustment(
  sourceProbaOrPriors: TensorLike,
  targetProba: TensorLike,
  options: EmPriorAdjustmentOptions = {},
): EmPriorAdjustmentResult {
  const target = matrixView(targetProba, 'emPriorAdjustment')
  const k = target.d
  const sourcePriors = resolveSourcePriors(sourceProbaOrPriors, k)
  const maxSteps = integerOption(options.maxIterations ?? 100, 'maxIterations', 'emPriorAdjustment', 1)
  const alg = emPriorAdjustmentAlgorithm(sourcePriors, targetProba, options)
  const res = runAlgorithm(alg, { ...runOptionsFor(options, 'emPriorAdjustment'), maxSteps })
  const targetPriors = Array.from(res.state.priors)
  const weights = targetPriors.map((pt, c) => pt / (sourcePriors[c] ?? 1))
  return {
    sourcePriors: Array.from(sourcePriors),
    targetPriors,
    weights,
    posteriors: fromStorage(res.state.posteriors, [target.n, k], 'float64'),
    iterations: res.steps,
    converged: res.status === 'converged',
  }
}
