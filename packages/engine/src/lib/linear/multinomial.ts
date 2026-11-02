/**
 * The multinomial logit (softmax regression) as a generalised linear model for an unordered categorical response:
 * `P(y = c | x) = exp(x w_c + b_c) / sum_k exp(x w_k + b_k)`. Without a penalty the model is identified by fixing the
 * first class as the reference (statsmodels' `MNLogit`); with a ridge penalty all classes keep their own weights
 * (scikit-learn's multinomial `LogisticRegression`). `multinomialAlgorithm` runs Newton steps with the exact Hessian
 * (so it converges in a handful of iterations) and `multinomial` is the classifier on the iterative base, with
 * class probabilities, the logits and a Wald table of the free coefficients.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { categorical } from '../dists/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import {
  encodeLabels,
  type IterativeEstimator,
  iterative,
  type LearnData,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
  withDefaultScore,
} from '../learn/index.ts'
import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { analyticObjective, type SmoothProblem, smoothSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { dict, real, type SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, Log } from '../trace/index.ts'
import { infNorm, newtonAt, type SlotState, slotAlgorithm } from './_slot.ts'
import { waldTable } from './inference.ts'

/** A multinomial problem: features, class indices, weights and the number of classes. */
export interface MultinomialProblem {
  readonly x: MatrixView
  readonly codes: ArrayLike<number>
  readonly w: ArrayLike<number>
  readonly nClasses: number
}

/** Read a multinomial problem from learn data. */
export function multinomialProblemOf(
  data: LearnData,
  who: string,
): { readonly problem: MultinomialProblem; readonly classes: readonly number[] } {
  const x = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, x.n)
  if (classes.length < 2) throw new MdmlError('value', `${who}: needs at least two classes`)
  return { problem: { x, codes, w: sampleWeightsOf(data.weights, x.n, who), nClasses: classes.length }, classes }
}

/** Options of {@link multinomialAlgorithm}. */
export interface MultinomialOptions {
  /** A ridge penalty `alpha |W|^2 / 2` on the slopes, per unit weight (default 0, which fixes class 0 as the reference). */
  readonly alpha?: number
  readonly fitIntercept?: boolean
  /** Stop when the largest gradient entry falls below this (default 1e-10). */
  readonly tol?: number
  /** The solver, a value of `smoothSolvers.space` (default Newton at `tol`). */
  readonly solver?: SpaceValue
  /** The run's key (for the stochastic solver's batches). */
  readonly key?: Key
}

/** The state of {@link multinomialAlgorithm}. */
export type MultinomialState = SlotState & {
  /** The free coefficients, class by class: slopes then intercept (classes 1 to C - 1 without a penalty, all with one). */
  readonly theta: number[]
  /** The penalised mean negative log-likelihood. */
  readonly objective: number
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/** The classes whose coefficients are free: all of them with a penalty, all but the first without. */
const freeClasses = (nClasses: number, alpha: number): number[] =>
  Array.from({ length: nClasses }, (_, c) => c).filter((c) => alpha > 0 || c > 0)

/**
 * The softmax probabilities `[n, C]` for logits `[n, C]` (row-wise, shifted by the row maximum).
 *
 * @param logits - Row-major logits.
 * @param nClasses - C.
 * @returns The probabilities.
 * @example
 * ```ts
 * import { softmaxRows } from 'mdml-engine/linear'
 * Array.from(softmaxRows([0, 0], 2)) // => [0.5, 0.5]
 * ```
 */
export function softmaxRows(logits: ArrayLike<number>, nClasses: number): Float64Array {
  const n = logits.length / nClasses
  const out = new Float64Array(logits.length)
  for (let i = 0; i < n; i++) {
    let mx = Number.NEGATIVE_INFINITY
    for (let c = 0; c < nClasses; c++) mx = Math.max(mx, logits[i * nClasses + c] ?? 0)
    let total = 0
    for (let c = 0; c < nClasses; c++) {
      const e = Math.exp((logits[i * nClasses + c] ?? 0) - mx)
      out[i * nClasses + c] = e
      total += e
    }
    for (let c = 0; c < nClasses; c++) out[i * nClasses + c] = (out[i * nClasses + c] ?? 0) / total
  }
  return out
}

/**
 * The multinomial objective as a smooth problem for the solver slot: the penalised mean negative log-likelihood
 * `-sum_i w_i log p_{i, y_i} / W + alpha |W|^2 / 2` with its exact gradient and Hessian, so Newton's method converges in
 * a handful of iterations and every other smooth solver minimises the same function.
 *
 * @param problem - Features, class indices and weights.
 * @param options - The penalty and the intercept.
 * @returns The objective over the free coefficients (class by class: slopes then intercept) and the start (zeros).
 * @example
 * ```ts
 * import { multinomialSmoothProblem } from 'mdml-engine/linear'
 * import { matrixView } from 'mdml-engine/learn'
 * const x = matrixView([[0], [1], [2], [3], [0.5], [2.5]], 'demo')
 * multinomialSmoothProblem({ x, codes: [0, 0, 1, 2, 1, 2], w: [1, 1, 1, 1, 1, 1], nClasses: 3 }).x0.length // => 4
 * ```
 */
export function multinomialSmoothProblem(problem: MultinomialProblem, options: MultinomialOptions = {}): SmoothProblem {
  const alpha = options.alpha ?? 0
  const fitIntercept = options.fitIntercept ?? true
  const { x, codes, w, nClasses: C } = problem
  const { n, d } = x
  const r = d + (fitIntercept ? 1 : 0)
  const free = freeClasses(C, alpha)
  const m = free.length * r
  const total = Array.from({ length: n }, (_, i) => w[i] ?? 0).reduce((a, b) => a + b, 0)
  const row = (i: number, k: number): number => (k < d ? (x.data[i * d + k] ?? 0) : 1)
  const probabilities = (theta: ArrayLike<number>): Float64Array => {
    const logits = new Float64Array(n * C)
    free.forEach((c, f) => {
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let k = 0; k < r; k++) s += row(i, k) * (theta[f * r + k] ?? 0)
        logits[i * C + c] = s
      }
    })
    return softmaxRows(logits, C)
  }
  const evaluate = (theta: number[]): { objective: number; grad: Float64Array; probs: Float64Array } => {
    const probs = probabilities(theta)
    let nll = 0
    for (let i = 0; i < n; i++) nll -= (w[i] ?? 0) * Math.log(Math.max(probs[i * C + (codes[i] ?? 0)] ?? 0, 1e-300))
    const grad = new Float64Array(m)
    free.forEach((c, f) => {
      for (let i = 0; i < n; i++) {
        const resid = ((w[i] ?? 0) * ((probs[i * C + c] ?? 0) - ((codes[i] ?? 0) === c ? 1 : 0))) / total
        for (let k = 0; k < r; k++) grad[f * r + k] = (grad[f * r + k] ?? 0) + resid * row(i, k)
      }
      for (let k = 0; k < d; k++) grad[f * r + k] = (grad[f * r + k] ?? 0) + alpha * (theta[f * r + k] ?? 0)
    })
    let penalty = 0
    free.forEach((_, f) => {
      for (let k = 0; k < d; k++) penalty += (theta[f * r + k] ?? 0) ** 2
    })
    return { objective: nll / total + (alpha * penalty) / 2, grad, probs }
  }
  const hessian = (probs: Float64Array): Float64Array => {
    const H = new Float64Array(m * m)
    for (let i = 0; i < n; i++) {
      const wi = (w[i] ?? 0) / total
      free.forEach((c, f) => {
        free.forEach((c2, f2) => {
          const pc = probs[i * C + c] ?? 0
          const coef = wi * pc * ((c === c2 ? 1 : 0) - (probs[i * C + c2] ?? 0))
          for (let a = 0; a < r; a++)
            for (let b = 0; b < r; b++)
              H[(f * r + a) * m + f2 * r + b] = (H[(f * r + a) * m + f2 * r + b] ?? 0) + coef * row(i, a) * row(i, b)
        })
      })
    }
    for (let f = 0; f < free.length; f++)
      for (let k = 0; k < d; k++) H[(f * r + k) * m + f * r + k] = (H[(f * r + k) * m + f * r + k] ?? 0) + alpha
    return H
  }
  const at = (theta: ArrayLike<number>): { objective: number; grad: Float64Array; probs: Float64Array } =>
    evaluate(Array.from(theta))
  return {
    objective: analyticObjective({
      value: (t) => at(t).objective,
      gradient: (t) => at(t).grad,
      hessian: (t) => hessian(at(t).probs),
    }),
    x0: new Float64Array(m),
  }
}

/**
 * Multinomial logistic regression as a traced algorithm over the solver slot: any solver of `smoothSolvers` minimises
 * {@link multinomialSmoothProblem} (default Newton with the exact Hessian). It logs `objective`, `gradient`, `gradNorm`,
 * `step` and the tensor `coefficients`.
 *
 * @param problem - Features, class indices and weights.
 * @param options - The penalty, the intercept, the tolerance and the solver.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { multinomialAlgorithm } from 'mdml-engine/linear'
 * import { matrixView } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = matrixView([[0], [1], [2], [3], [0.5], [2.5], [1.5], [3.5]], 'demo')
 * const problem = { x, codes: [0, 0, 1, 2, 1, 2, 0, 2], w: [1, 1, 1, 1, 1, 1, 1, 1], nClasses: 3 }
 * runAlgorithm(multinomialAlgorithm(problem), { maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function multinomialAlgorithm(
  problem: MultinomialProblem,
  options: MultinomialOptions = {},
): Algorithm<MultinomialState> {
  const alpha = options.alpha ?? 0
  const tol = options.tol ?? 1e-10
  const m = freeClasses(problem.nClasses, alpha).length * (problem.x.d + ((options.fitIntercept ?? true) ? 1 : 0))
  const solver = options.solver ?? newtonAt(tol)
  return slotAlgorithm<MultinomialState>({
    name: 'multinomial',
    config: { alpha, fitIntercept: options.fitIntercept ?? true, tol, classes: problem.nClasses, solver } as Json,
    schema: {
      metrics: {
        objective: { description: 'the penalised mean negative log-likelihood', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { coefficients: { role: 'param', description: 'the free coefficients, class by class', shape: [m] } },
    },
    problem: multinomialSmoothProblem(problem, options),
    solver,
    ...(options.key === undefined ? {} : { key: options.key }),
    tol,
    view: (r, evaluations, stalled) => ({
      theta: Array.from(r.x),
      objective: r.value,
      gradient: infNorm(r.grad),
      evaluations,
      stalled,
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { coefficients: { role: 'param', value: s.theta, shape: [m] } },
    }),
  })
}

/** The hyperparameters of {@link multinomial}. */
export interface MultinomialHyper {
  /** The ridge penalty per unit weight (default 0). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
  /** How the likelihood is minimised: a value of `smoothSolvers.space` (default Newton, whose steps use the exact Hessian). */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link multinomial}. */
export type MultinomialParams = {
  /** The class labels, ascending. */
  readonly classes: number[]
  /** The weights `[C][d]` (a zero row for the reference class when unpenalised). */
  readonly coef: number[][]
  readonly intercept: number[]
  /** The reference class label, or null when every class has its own weights (penalised). */
  readonly reference: number | null
  readonly logLikelihood: number
  /** The covariance of the free coefficients (inverse information of the weighted sum), `[m][m]`. */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted multinomial model offers. */
export interface MultinomialVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  /** Class probabilities `[m, C]`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The logits `[m, C]`. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => ReturnType<typeof categorical>
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

const MULTINOMIAL_DEFAULTS: MultinomialHyper = {
  alpha: 0,
  fitIntercept: true,
  tol: 1e-10,
  maxIter: 100,
  solver: newtonAt(1e-10),
}

function paramsOf(
  problem: MultinomialProblem,
  classes: readonly number[],
  h: MultinomialHyper,
  s: MultinomialState,
): MultinomialParams {
  const { x, codes, w, nClasses: C } = problem
  const { n, d } = x
  const r = d + (h.fitIntercept ? 1 : 0)
  const free = freeClasses(C, h.alpha)
  const m = free.length * r
  const coef = Array.from({ length: C }, () => new Array<number>(d).fill(0))
  const intercept = new Array<number>(C).fill(0)
  free.forEach((c, f) => {
    for (let k = 0; k < d; k++) (coef[c] as number[])[k] = s.theta[f * r + k] ?? 0
    if (h.fitIntercept) intercept[c] = s.theta[f * r + d] ?? 0
  })
  // The information of the unscaled weighted sum: W times the Hessian of the mean objective.
  const total = Array.from({ length: n }, (_, i) => w[i] ?? 0).reduce((a, b) => a + b, 0)
  const logits = new Float64Array(n * C)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < C; c++) {
      let t = intercept[c] ?? 0
      for (let k = 0; k < d; k++) t += (x.data[i * d + k] ?? 0) * (coef[c]?.[k] ?? 0)
      logits[i * C + c] = t
    }
  const probs = softmaxRows(logits, C)
  const info = new Float64Array(m * m)
  const rowVal = (i: number, k: number): number => (k < d ? (x.data[i * d + k] ?? 0) : 1)
  let ll = 0
  for (let i = 0; i < n; i++) {
    ll += (w[i] ?? 0) * Math.log(Math.max(probs[i * C + (codes[i] ?? 0)] ?? 0, 1e-300))
    free.forEach((c, f) => {
      free.forEach((c2, f2) => {
        const v = (w[i] ?? 0) * (probs[i * C + c] ?? 0) * ((c === c2 ? 1 : 0) - (probs[i * C + c2] ?? 0))
        for (let a = 0; a < r; a++)
          for (let b = 0; b < r; b++)
            info[(f * r + a) * m + f2 * r + b] =
              (info[(f * r + a) * m + f2 * r + b] ?? 0) + v * rowVal(i, a) * rowVal(i, b)
      })
    })
  }
  for (let f = 0; f < free.length; f++)
    for (let k = 0; k < d; k++)
      info[(f * r + k) * m + f * r + k] = (info[(f * r + k) * m + f * r + k] ?? 0) + h.alpha * total
  const chol = dpotrf(info, m)
  const covariance = Array.from({ length: m }, () => new Array<number>(m).fill(Number.NaN))
  if (chol.info === 0)
    for (let c = 0; c < m; c++) {
      const e = new Float64Array(m)
      e[c] = 1
      const col = dpotrs(chol.factor, e, m)
      for (let a = 0; a < m; a++) (covariance[a] as number[])[c] = col[a] ?? 0
    }
  return {
    classes: [...classes],
    coef,
    intercept,
    reference: h.alpha > 0 ? null : (classes[0] ?? null),
    logLikelihood: ll,
    covariance,
    iterations: s.evaluations,
    converged: s.gradient <= h.tol,
  }
}

function tableOf(p: MultinomialParams, h: MultinomialHyper): CoefficientTable {
  const d = p.coef[0]?.length ?? 0
  const classIndices = p.classes.map((_, c) => c).filter((c) => p.reference === null || c > 0)
  const names: string[] = []
  const estimates: number[] = []
  for (const c of classIndices) {
    for (let k = 0; k < d; k++) {
      names.push(`${p.classes[c]}:x${k}`)
      estimates.push(p.coef[c]?.[k] ?? 0)
    }
    if (h.fitIntercept) {
      names.push(`${p.classes[c]}:intercept`)
      estimates.push(p.intercept[c] ?? 0)
    }
  }
  const m = estimates.length
  const cov = Float64Array.from(
    { length: m * m },
    (_, idx) => p.covariance[Math.floor(idx / m)]?.[idx % m] ?? Number.NaN,
  )
  return waldTable(names, estimates, cov)
}

/**
 * Multinomial logistic regression (softmax regression; statsmodels' `MNLogit`, scikit-learn's multinomial
 * `LogisticRegression`): class probabilities `softmax(x W + b)` for unordered classes. Unpenalised, the first class is
 * the reference with zero weights and the Wald table covers the others; with `alpha > 0` every class has its own
 * weights. A session adds one Newton step per iteration (`solver/objective`, `solver/gradient`,
 * `solver/coefficients`).
 *
 * @param hyper - `alpha`, `fitIntercept`, `tol`, `maxIter`.
 * @returns The estimator.
 * @index estimator Multinomial logistic (softmax) regression.
 * @example
 * ```ts
 * import { multinomial } from 'mdml-engine/linear'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3], [0.5], [2.5], [1.5], [3.5], [0.2], [2.2], [1.2], [3.2]]
 * const y = [0, 0, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1]
 * const f = multinomial().fit({ x, y })
 * toFlat(f.predictProba([[1]])).reduce((a, b) => a + b, 0) // => 1
 * ```
 */
export function multinomial(
  hyper: Partial<MultinomialHyper> = {},
): IterativeEstimator<MultinomialHyper, MultinomialState, MultinomialParams, MultinomialVerbs> {
  return iterative<MultinomialHyper, MultinomialState, MultinomialParams, MultinomialVerbs>(
    {
      name: 'multinomial',
      kind: 'classifier',
      defaults: MULTINOMIAL_DEFAULTS,
      hyperSpace: (h) =>
        dict({ alpha: real(0, Math.max(10, h.alpha), { default: h.alpha }), solver: smoothSolvers.space }),
      algorithm: (data, h, key) =>
        multinomialAlgorithm(multinomialProblemOf(data, 'multinomial').problem, { ...h, key }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const { problem, classes } = multinomialProblemOf(data, 'multinomial')
        return paramsOf(problem, classes, h, s)
      },
      verbs: (p) => {
        const C = p.classes.length
        const d = p.coef[0]?.length ?? 0
        const logits = (x: TensorLike): Float64Array => {
          const m = matrixView(x, 'multinomial', { nFeatures: d })
          const out = new Float64Array(m.n * C)
          for (let i = 0; i < m.n; i++)
            for (let c = 0; c < C; c++) {
              let t = p.intercept[c] ?? 0
              for (let k = 0; k < d; k++) t += (m.data[i * d + k] ?? 0) * (p.coef[c]?.[k] ?? 0)
              out[i * C + c] = t
            }
          return out
        }
        const proba = (x: TensorLike): Tensor => {
          const l = logits(x)
          return fromStorage(softmaxRows(l, C), [l.length / C, C], 'float64')
        }
        const base = {
          classes: p.classes,
          nFeaturesIn: d,
          predictProba: proba,
          decisionFunction: (x: TensorLike): Tensor => {
            const l = logits(x)
            return fromStorage(l, [l.length / C, C], 'float64')
          },
          predictDistribution: (x: TensorLike): ReturnType<typeof categorical> => categorical({ probs: proba(x) }),
          predict: (x: TensorLike): Tensor => {
            const l = logits(x)
            const n = l.length / C
            return fromStorage(
              Float64Array.from({ length: n }, (_, i) => {
                let best = 0
                for (let c = 1; c < C; c++) if ((l[i * C + c] ?? 0) > (l[i * C + best] ?? 0)) best = c
                return p.classes[best] ?? Number.NaN
              }),
              [n],
              'float64',
            )
          },
          featureImportances: Array.from({ length: d }, (_, k) =>
            p.coef.reduce((a, row) => a + Math.abs(row[k] ?? 0), 0),
          ),
        }
        return { ...base, score: withDefaultScore('classifier', base).score ?? ((): number => Number.NaN) }
      },
      records: (s, data, h): readonly Exposition[] => {
        const { problem, classes } = multinomialProblemOf(data, 'multinomial')
        return [tableOf(paramsOf(problem, classes, h, s), h)]
      },
      logs: { 'solver/objective': Log(), 'solver/gradient': Log(), 'solver/coefficients': Log({ detail: 'full' }) },
    },
    { ...MULTINOMIAL_DEFAULTS, ...hyper },
  )
}
