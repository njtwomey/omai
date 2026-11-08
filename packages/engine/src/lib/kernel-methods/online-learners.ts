/**
 * More online kernel learners beside KRLS and NORMA, each an `Algorithm` over the stream with a plain-JSON state (the
 * kernel expansion so far), a budget on the expansion and an estimator on the iterative base whose `fittedAt(t)` is the
 * predictor after `t` observations: kernel least mean squares (`klms`) and its quantised form (`qklms`), where a new
 * input within the quantisation radius of a centre updates that centre instead of adding one; the kernel perceptron
 * (`kernelPerceptron`) with budgets by random removal, oldest removal, a shrinking forgetting factor and the Projectron's
 * projection (`budget`); and the kernel passive-aggressive classifier (`kernelPassiveAggressive`, PA, PA-I and PA-II).
 * The perceptron's mistake bound and the Projectron's bounded dictionary are checked in the tests.
 */

import { OmaiError } from '../base/index.ts'
import { type Distribution, normal } from '../dists/index.ts'
import { type KernelChoice, kernelMatrix, kernelSpace, type ResolvedKernel, resolveKernel } from '../kernels/index.ts'
import {
  encodeLabels,
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  targetVector,
} from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { choice, dict, int, real, type Space } from '../space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  classifierVerbs,
  KERNEL_DEFAULTS,
  type KernelClassifierVerbs,
  type KernelOptions,
  queryView,
} from './_classifier.ts'

/** The state of an online kernel expansion `f = sum_j coef_j k(centres_j, .)`: plain JSON. */
export type ExpansionState = {
  /** Observations absorbed. */
  readonly step: number
  readonly centres: number[][]
  readonly coef: number[]
  /** The step at which each centre entered. */
  readonly born: number[]
  /** Mistakes so far (classifiers). */
  readonly mistakes: number
  /** The cumulative loss: squared error (KLMS), hinge loss (passive-aggressive), mistakes (perceptron). */
  readonly loss: number
  /** The signed prediction error or margin of the last observation, before it was absorbed. */
  readonly last: number
  /** The squared distance to the span of the dictionary of the last Projectron projection (0 elsewhere). */
  readonly residual: number
}

const EMPTY: ExpansionState = { step: 0, centres: [], coef: [], born: [], mistakes: 0, loss: 0, last: 0, residual: 0 }

/** The prediction of an expansion at one point. */
function predictAt(h: KernelOptions, r: ResolvedKernel, s: ExpansionState, x: readonly number[]): number {
  if (s.centres.length === 0) return 0
  const k = kernelMatrix(h, r, s.centres, [Array.from(x)])
  let v = 0
  for (let j = 0; j < s.coef.length; j++) v += (k[j] ?? 0) * (s.coef[j] ?? 0)
  return v
}

const squaredDistance = (a: readonly number[], b: readonly number[]): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return s
}

/** Drop the oldest centres until at most `budget` remain (0: no budget). */
function trimOldest(s: ExpansionState, budget: number): ExpansionState {
  const over = budget > 0 ? Math.max(0, s.centres.length - budget) : 0
  return over === 0 ? s : { ...s, centres: s.centres.slice(over), coef: s.coef.slice(over), born: s.born.slice(over) }
}

function checkBudget(budget: number, who: string): void {
  if (!Number.isInteger(budget) || budget < 0)
    throw new OmaiError('value', `${who}: budget must be a non-negative integer`)
}

// ---------------------------------------------------------------------------------------------------------------------
// KLMS and quantised KLMS
// ---------------------------------------------------------------------------------------------------------------------

/** The hyperparameters of {@link klms}. */
export interface KlmsHyper extends KernelOptions {
  /** The step size `eta` of `f <- f + eta e k(x, .)` (default 0.3). */
  readonly eta: number
  /** The quantisation radius in input space: an input within it of a centre updates that centre (0: plain KLMS; default 0). */
  readonly quantisation: number
  /** The most centres kept, the oldest dropped (0: unbounded; default 0). */
  readonly budget: number
  /** Passes over the data (default 1). */
  readonly epochs: number
}

/** The fitted parameters of the online expansion learners: plain JSON. */
export type ExpansionParams = {
  readonly kernel: ResolvedKernel
  readonly classes: number[]
  readonly centres: number[][]
  readonly coef: number[]
  readonly nFeaturesIn: number
  readonly nObservations: number
  readonly mistakes: number
  /** The mean online loss (squared error, hinge loss or mistake rate). */
  readonly meanLoss: number
}

/** What a fitted online regressor offers. */
export interface OnlineRegressorVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly score: (data: LearnData) => number
  readonly dictionarySize: number
  readonly nFeaturesIn: number
}

const KLMS_DEFAULTS: KlmsHyper = { ...KERNEL_DEFAULTS, eta: 0.3, quantisation: 0, budget: 0, epochs: 1 }

/**
 * The KLMS algorithm (Liu, Pokharel and Principe) and its quantised form (Chen et al.): one step of `f <- f + eta e
 * k(x, .)` with the prediction error `e = y - f(x)`; with a positive `quantisation` an input whose nearest centre is
 * within that radius adds `eta e` to that centre's coefficient rather than becoming a new centre, so the dictionary stays
 * bounded on bounded inputs.
 *
 * @param data - Features and targets, visited in order, `epochs` times.
 * @param hyper - The hyperparameters.
 * @returns The algorithm, logging the squared error, its running mean and the dictionary size.
 * @throws OmaiError `value` for a non-positive `eta`, a negative `quantisation` or a bad budget.
 * @concept online-kernel
 * @reference W. Liu, P. Pokharel and J. Principe, The kernel least-mean-square algorithm (IEEE Trans. Signal Processing
 *   56, 2008); B. Chen, S. Zhao, P. Zhu and J. Principe, Quantized kernel least mean square algorithm (IEEE TNNLS 23, 2012).
 * @example
 * ```ts
 * import { klmsAlgorithm, klmsDefaults } from 'omai-engine/kernel-methods'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = klmsAlgorithm({ x: [[0], [1], [2]], y: [0, 1, 2] }, { ...klmsDefaults, kernel: 'linear', gamma: 1 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.step // => 3
 * ```
 */
export function klmsAlgorithm(data: LearnData, hyper: KlmsHyper): Algorithm<ExpansionState> {
  const who = 'klms'
  if (!(hyper.eta > 0)) throw new OmaiError('value', `${who}: eta must be positive`)
  if (!(hyper.quantisation >= 0)) throw new OmaiError('value', `${who}: quantisation must not be negative`)
  checkBudget(hyper.budget, who)
  const m = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, m.n)
  const kernel = resolveKernel(hyper as KernelChoice, data.x)
  const row = (i: number): number[] => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
  const total = m.n * hyper.epochs
  return {
    name: 'klms',
    config: { eta: hyper.eta, quantisation: hyper.quantisation, budget: hyper.budget, kernel: kernel.kernel },
    schema: channels({
      metrics: {
        loss: { description: 'the squared prediction error before the update', direction: 'lower' },
        meanLoss: { description: 'the mean squared error so far', direction: 'lower' },
        centres: { description: 'the number of centres' },
      },
      tensors: { coef: { role: 'param', description: 'the coefficients of the expansion' } },
    }),
    init: () => EMPTY,
    step: (s) => {
      const i = s.step % m.n
      const point = row(i)
      const error = (y[i] ?? 0) - predictAt(hyper, kernel, s, point)
      let nearest = -1
      let best = Number.POSITIVE_INFINITY
      if (hyper.quantisation > 0) {
        s.centres.forEach((c, j) => {
          const d = Math.sqrt(squaredDistance(c, point))
          if (d < best) {
            best = d
            nearest = j
          }
        })
      }
      const merge = nearest >= 0 && best <= hyper.quantisation
      const coef = s.coef.map((c, j) => (merge && j === nearest ? c + hyper.eta * error : c))
      const next: ExpansionState = {
        ...s,
        step: s.step + 1,
        centres: merge ? s.centres : [...s.centres, point],
        coef: merge ? coef : [...coef, hyper.eta * error],
        born: merge ? s.born : [...s.born, s.step],
        loss: s.loss + error * error,
        last: error,
      }
      return trimOldest(next, hyper.budget)
    },
    done: (s) => (s.step >= total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        loss: s.last * s.last,
        meanLoss: s.step === 0 ? 0 : s.loss / s.step,
        centres: s.centres.length,
      },
      tensors: { coef: { role: 'param', value: s.coef } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ExpansionState,
  }
}

/** The default hyperparameters of {@link klms}. */
export const klmsDefaults: KlmsHyper = KLMS_DEFAULTS

export function regressorVerbs(p: ExpansionParams, h: KernelOptions, who: string): OnlineRegressorVerbs {
  const predict = (x: TensorLike): Tensor => {
    const q = queryView(x, p.nFeaturesIn, who)
    const m = p.centres.length
    if (m === 0) return fromStorage(new Float64Array(q.n), [q.n], 'float64')
    const k = kernelMatrix(h, p.kernel, x, p.centres)
    return fromStorage(
      Float64Array.from({ length: q.n }, (_, i) => {
        let s = 0
        for (let j = 0; j < m; j++) s += (k[i * m + j] ?? 0) * (p.coef[j] ?? 0)
        return s
      }),
      [q.n],
      'float64',
    )
  }
  return {
    predict,
    predictDistribution: (x) => {
      const mean = predict(x)
      return normal(
        mean,
        fromStorage(
          new Float64Array(mean.shape.reduce((a, b) => a * b, 1)).fill(Math.max(Math.sqrt(p.meanLoss), 1e-12)),
          [...mean.shape],
          'float64',
        ),
      )
    },
    score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
    dictionarySize: p.centres.length,
    nFeaturesIn: p.nFeaturesIn,
  }
}

function klmsEstimator(
  name: string,
  defaults: KlmsHyper,
  hyper: Partial<KlmsHyper>,
): IterativeEstimator<KlmsHyper, ExpansionState, ExpansionParams, OnlineRegressorVerbs> {
  return iterative<KlmsHyper, ExpansionState, ExpansionParams, OnlineRegressorVerbs>(
    {
      name,
      kind: 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          eta: real(Math.min(1e-3, h.eta), Math.max(1, h.eta), { log: true, default: h.eta, label: 'step size' }),
          quantisation: real(0, Math.max(5, h.quantisation), { default: h.quantisation, label: 'quantisation radius' }),
          budget: int(0, Math.max(500, h.budget), { default: h.budget, label: 'most centres (0: unbounded)' }),
          ...kernelSpace(h),
        }),
      algorithm: (data, h) => klmsAlgorithm(data, h),
      maxSteps: (h, data) => matrixView(data.x, name).n * h.epochs,
      params: (s, data, h) => ({
        kernel: resolveKernel(h as KernelChoice, data.x),
        classes: [],
        centres: s.centres,
        coef: s.coef,
        nFeaturesIn: matrixView(data.x, name).d,
        nObservations: s.step,
        mistakes: 0,
        meanLoss: s.step === 0 ? 0 : s.loss / s.step,
      }),
      verbs: (p, h) => regressorVerbs(p, h, name),
      logs: { 'solver/loss': Log(), 'solver/meanLoss': Log(), 'solver/centres': Log() },
    },
    { ...defaults, ...hyper },
  )
}

/**
 * Kernel least mean squares (Liu, Pokharel and Principe), the kernel version of the LMS filter: stochastic gradient
 * descent on the squared error in the RKHS with a constant step, one new centre per observation (optionally truncated to
 * the last `budget`). The tape shows the squared error falling and the dictionary growing.
 *
 * @param hyper - The kernel options, `eta`, `budget`, `epochs`.
 * @returns The estimator.
 * @concept online-kernel
 * @example
 * ```ts
 * import { klms } from 'omai-engine/kernel-methods'
 * import { toFlat } from 'omai-engine/tensor'
 * const fit = klms({ kernel: 'linear', gamma: 1, eta: 0.2, epochs: 100 }).fit({ x: [[1], [2], [3]], y: [2, 4, 6] })
 * toFlat(fit.predict([[4]]))[0]?.toFixed(1) // => '8.0'
 * ```
 */
export function klms(
  hyper: Partial<KlmsHyper> = {},
): IterativeEstimator<KlmsHyper, ExpansionState, ExpansionParams, OnlineRegressorVerbs> {
  return klmsEstimator('kernel-methods/klms', KLMS_DEFAULTS, hyper)
}

/**
 * Quantised kernel least mean squares (Chen et al.): {@link klms} with a quantisation radius (default 0.5), so an input
 * near an existing centre adds to that centre's coefficient and the dictionary grows only to cover the input space.
 *
 * @param hyper - The kernel options, `eta`, `quantisation`, `budget`, `epochs`.
 * @returns The estimator.
 * @concept online-kernel
 * @example
 * ```ts
 * import { qklms } from 'omai-engine/kernel-methods'
 * const x = Array.from({ length: 40 }, (_, i) => [(i % 4) * 0.1])
 * qklms({ gamma: 1, quantisation: 0.25 }).fit({ x, y: x.map((r) => r[0] ?? 0) }).dictionarySize <= 4 // => true
 * ```
 */
export function qklms(
  hyper: Partial<KlmsHyper> = {},
): IterativeEstimator<KlmsHyper, ExpansionState, ExpansionParams, OnlineRegressorVerbs> {
  return klmsEstimator('kernel-methods/qklms', { ...KLMS_DEFAULTS, quantisation: 0.5 }, hyper)
}

// ---------------------------------------------------------------------------------------------------------------------
// The kernel perceptron and passive-aggressive classifiers
// ---------------------------------------------------------------------------------------------------------------------

/** How a budget on the support vectors is kept. */
export type BudgetRule = 'none' | 'random-removal' | 'oldest' | 'forgetron' | 'projectron'

/** The hyperparameters of the online kernel classifiers. */
export interface OnlineClassifierHyper extends KernelOptions {
  /** The budget rule (default `none`). */
  readonly budget: BudgetRule
  /** The most support vectors for `random-removal`, `oldest` and `forgetron` (default 50). */
  readonly size: number
  /** The factor the coefficients are multiplied by when the oldest support vector is dropped, `forgetron` (default 0.9). */
  readonly forgetting: number
  /** The Projectron's threshold on the squared distance to the span: below it the example is projected, not added (default 0.1). */
  readonly eta: number
  /** Passes over the data (default 1). */
  readonly epochs: number
}

/** The hyperparameters of {@link kernelPassiveAggressive}. */
export interface PassiveAggressiveHyper extends OnlineClassifierHyper {
  /** `pa` (the closest update that attains margin one), `pa1` (step capped at `C`) or `pa2` (soft). Default `pa1`. */
  readonly variant: 'pa' | 'pa1' | 'pa2'
  /** The aggressiveness `C` of `pa1` and `pa2` (default 1). */
  readonly C: number
}

const CLASSIFIER_DEFAULTS: OnlineClassifierHyper = {
  ...KERNEL_DEFAULTS,
  budget: 'none',
  size: 50,
  forgetting: 0.9,
  eta: 0.1,
  epochs: 1,
}

/** The projection of `phi(x)` on the span of the centres: the coefficients `d` solving `K d = k(centres, x)` and the squared distance. */
function projection(
  h: KernelOptions,
  r: ResolvedKernel,
  centres: number[][],
  x: readonly number[],
): { readonly d: number[]; readonly residual: number } {
  const m = centres.length
  const kxx = kernelMatrix(h, r, [Array.from(x)])[0] ?? 0
  if (m === 0) return { d: [], residual: kxx }
  const gram = kernelMatrix(h, r, centres)
  for (let i = 0; i < m; i++) gram[i * m + i] = (gram[i * m + i] ?? 0) + 1e-10
  const k = kernelMatrix(h, r, centres, [Array.from(x)])
  const d = Array.from(toFlat(primalOf(solve(fromStorage(gram, [m, m], 'float64'), fromStorage(k, [m, 1], 'float64')))))
  let dot = 0
  for (let j = 0; j < m; j++) dot += (d[j] ?? 0) * (k[j] ?? 0)
  return { d, residual: Math.max(kxx - dot, 0) }
}

/**
 * One mistake-driven step of the budgeted classifiers: the update adds `weight * y` at `x` (or, for the Projectron when
 * the example is nearly in the span, spreads it over the existing centres), then the budget is enforced.
 */
function classifierUpdate(
  s: ExpansionState,
  point: number[],
  label: number,
  weight: number,
  hyper: OnlineClassifierHyper,
  kernel: ResolvedKernel,
  key: Key,
): ExpansionState {
  if (hyper.budget === 'projectron') {
    const { d, residual } = projection(hyper, kernel, s.centres, point)
    if (s.centres.length > 0 && residual <= hyper.eta) {
      return { ...s, coef: s.coef.map((c, j) => c + weight * label * (d[j] ?? 0)), residual }
    }
    return {
      ...s,
      centres: [...s.centres, point],
      coef: [...s.coef, weight * label],
      born: [...s.born, s.step],
      residual,
    }
  }
  let next: ExpansionState = {
    ...s,
    centres: [...s.centres, point],
    coef: [...s.coef, weight * label],
    born: [...s.born, s.step],
  }
  if (hyper.budget === 'none' || next.centres.length <= hyper.size) return next
  if (hyper.budget === 'random-removal') {
    const drop = Math.min(next.centres.length - 2, Math.floor(item01(key) * (next.centres.length - 1)))
    next = {
      ...next,
      centres: next.centres.filter((_, j) => j !== drop),
      coef: next.coef.filter((_, j) => j !== drop),
      born: next.born.filter((_, j) => j !== drop),
    }
    return next
  }
  if (hyper.budget === 'forgetron') {
    next = { ...next, coef: next.coef.map((c, j) => (j === next.coef.length - 1 ? c : c * hyper.forgetting)) }
  }
  return trimOldest(next, hyper.size)
}

/** A uniform number in `[0, 1)` from a key. */
const item01 = (key: Key): number => toFlat(uniform(key, [1]))[0] ?? 0

function validateClassifier(h: OnlineClassifierHyper, who: string): void {
  if (!Number.isInteger(h.size) || h.size < 1) throw new OmaiError('value', `${who}: size must be a positive integer`)
  if (!(h.forgetting > 0 && h.forgetting <= 1)) throw new OmaiError('value', `${who}: forgetting must be in (0, 1]`)
  if (!(h.eta >= 0)) throw new OmaiError('value', `${who}: eta must not be negative`)
}

function signedLabels(data: LearnData, n: number, who: string): { y: Float64Array; classes: number[] } {
  const enc = encodeLabels(data.y ?? [], who, n)
  if (enc.classes.length !== 2) throw new OmaiError('value', `${who}: expected two classes, got ${enc.classes.length}`)
  return { y: Float64Array.from(enc.codes, (c) => (c === 1 ? 1 : -1)), classes: [...enc.classes] }
}

type Rule = (margin: number, selfK: number) => number

function classifierAlgorithm(
  name: string,
  data: LearnData,
  hyper: OnlineClassifierHyper,
  rule: Rule,
  config: Record<string, number | string>,
): Algorithm<ExpansionState> {
  validateClassifier(hyper, name)
  const m = matrixView(data.x, name)
  const { y } = signedLabels(data, m.n, name)
  const kernel = resolveKernel(hyper as KernelChoice, data.x)
  const row = (i: number): number[] => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
  const total = m.n * hyper.epochs
  return {
    name,
    config: { ...config, budget: hyper.budget, size: hyper.size, kernel: kernel.kernel },
    schema: channels({
      metrics: {
        loss: { description: 'the hinge loss of the observation before the update', direction: 'lower' },
        mistakes: { description: 'the mistakes so far', direction: 'lower' },
        supportVectors: { description: 'the number of support vectors' },
        residual: {
          description: 'the squared distance to the span of the dictionary of the last Projectron projection',
        },
      },
      tensors: { coef: { role: 'param', description: 'the coefficients of the expansion' } },
    }),
    init: () => EMPTY,
    step: (s, ctx) => {
      const i = s.step % m.n
      const point = row(i)
      const label = y[i] ?? 1
      const margin = label * predictAt(hyper, kernel, s, point)
      const mistake = margin <= 0
      const weight = rule(margin, kernelMatrix(hyper, kernel, [point])[0] ?? 1)
      const base: ExpansionState = {
        ...s,
        step: s.step + 1,
        mistakes: s.mistakes + (mistake ? 1 : 0),
        loss: s.loss + Math.max(0, 1 - margin),
        last: margin,
        residual: 0,
      }
      return weight === 0 ? base : classifierUpdate(base, point, label, weight, hyper, kernel, foldIn(ctx.key, s.step))
    },
    done: (s) => (s.step >= total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        loss: s.step === 0 ? 0 : Math.max(0, 1 - s.last),
        mistakes: s.mistakes,
        supportVectors: s.centres.length,
        residual: s.residual,
      },
      tensors: { coef: { role: 'param', value: s.coef } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ExpansionState,
  }
}

/**
 * The kernel perceptron as an algorithm: on a mistake (`y f(x) <= 0`) the support vector `(x, y)` is added, `f <- f + y
 * k(x, .)`. With separable data of margin `gamma` in a feature space of radius `R` it makes at most `(R / gamma)^2`
 * mistakes (Novikoff). A budget keeps at most `size` support vectors by removing a random one, the oldest one, the oldest
 * after shrinking the rest (a constant-factor form of the Forgetron), or none at all when the Projectron's projection onto
 * the span of the dictionary is accurate enough, which bounds the dictionary on bounded data (Orabona, Keshet and Caputo).
 *
 * @param data - Features and two-class labels, visited in order, `epochs` times.
 * @param hyper - The kernel options and the budget rule.
 * @returns The algorithm, logging the loss, mistakes and support vectors.
 * @throws OmaiError `value` unless there are two classes and the knobs are valid.
 * @concept online-kernel
 * @reference F. Rosenblatt (1958); O. Dekel, S. Shalev-Shwartz and Y. Singer, The Forgetron (SIAM J. Comput. 37, 2008);
 *   F. Orabona, J. Keshet and B. Caputo, The Projectron (ICML 2008).
 * @example
 * ```ts
 * import { kernelPerceptronAlgorithm, kernelPerceptronDefaults } from 'omai-engine/kernel-methods'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const data = { x: [[1], [2], [-1], [-2]], y: [1, 1, 0, 0] }
 * const alg = kernelPerceptronAlgorithm(data, { ...kernelPerceptronDefaults, kernel: 'linear', gamma: 1, epochs: 3 })
 * runAlgorithm(alg, { maxSteps: 100 }).state.mistakes // => 1
 * ```
 */
export function kernelPerceptronAlgorithm(data: LearnData, hyper: OnlineClassifierHyper): Algorithm<ExpansionState> {
  return classifierAlgorithm('kernel-perceptron', data, hyper, (margin) => (margin <= 0 ? 1 : 0), {})
}

/** The default hyperparameters of {@link kernelPerceptron}. */
export const kernelPerceptronDefaults: OnlineClassifierHyper = CLASSIFIER_DEFAULTS

/**
 * The kernel passive-aggressive algorithm (Crammer et al.) as an algorithm: on an example with hinge loss
 * `l = max(0, 1 - y f(x))` the step is `tau = l / k(x, x)` (`pa`), `min(C, l / k(x, x))` (`pa1`) or `l / (k(x, x) + 1/(2C))`
 * (`pa2`), and `f <- f + tau y k(x, .)`: passive when the margin is one, aggressive enough to reach it otherwise. For
 * separable data with a comparator of unit margin the cumulative squared loss is at most `R^2 |u|^2`.
 *
 * @param data - Features and two-class labels.
 * @param hyper - The kernel options, the budget rule, the variant and `C`.
 * @returns The algorithm.
 * @concept online-kernel
 * @reference K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz and Y. Singer, Online passive-aggressive algorithms
 *   (JMLR 7, 2006).
 * @example
 * ```ts
 * import { kernelPassiveAggressiveAlgorithm, kernelPassiveAggressiveDefaults } from 'omai-engine/kernel-methods'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = kernelPassiveAggressiveAlgorithm({ x: [[1], [-1]], y: [1, 0] }, { ...kernelPassiveAggressiveDefaults, kernel: 'linear', gamma: 1 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.centres.length // => 1
 * ```
 */
export function kernelPassiveAggressiveAlgorithm(
  data: LearnData,
  hyper: PassiveAggressiveHyper,
): Algorithm<ExpansionState> {
  if (!(hyper.C > 0)) throw new OmaiError('value', 'kernelPassiveAggressive: C must be positive')
  const rule: Rule = (margin, selfK) => {
    const loss = Math.max(0, 1 - margin)
    if (loss === 0) return 0
    if (hyper.variant === 'pa') return loss / selfK
    return hyper.variant === 'pa1' ? Math.min(hyper.C, loss / selfK) : loss / (selfK + 1 / (2 * hyper.C))
  }
  return classifierAlgorithm('kernel-passive-aggressive', data, hyper, rule, { variant: hyper.variant, C: hyper.C })
}

/** The default hyperparameters of {@link kernelPassiveAggressive}. */
export const kernelPassiveAggressiveDefaults: PassiveAggressiveHyper = { ...CLASSIFIER_DEFAULTS, variant: 'pa1', C: 1 }

function onlineClassifier<H extends OnlineClassifierHyper>(
  name: string,
  defaults: H,
  hyper: Partial<H>,
  algorithm: (data: LearnData, h: H) => Algorithm<ExpansionState>,
  extra: (h: H) => Record<string, Space>,
): IterativeEstimator<H, ExpansionState, ExpansionParams, KernelClassifierVerbs> {
  return iterative<H, ExpansionState, ExpansionParams, KernelClassifierVerbs>(
    {
      name,
      kind: 'classifier',
      defaults,
      hyperSpace: (h) =>
        dict({
          budget: choice(['none', 'random-removal', 'oldest', 'forgetron', 'projectron'], {
            default: h.budget,
            label: 'budget rule',
          }),
          size: int(1, Math.max(500, h.size), { default: h.size, label: 'support vectors' }),
          eta: real(0, Math.max(1, h.eta), {
            default: h.eta,
            label: 'projection threshold',
            when: { key: 'budget', in: ['projectron'] },
          }),
          ...extra(h),
          ...kernelSpace(h),
        }),
      algorithm,
      maxSteps: (h, data) => matrixView(data.x, name).n * h.epochs,
      params: (s, data, h) => {
        const m = matrixView(data.x, name)
        return {
          kernel: resolveKernel(h as KernelChoice, data.x),
          classes: signedLabels(data, m.n, name).classes,
          centres: s.centres,
          coef: s.coef,
          nFeaturesIn: m.d,
          nObservations: s.step,
          mistakes: s.mistakes,
          meanLoss: s.step === 0 ? 0 : s.loss / s.step,
        }
      },
      verbs: (p, h) =>
        classifierVerbs(p.classes, p.nFeaturesIn, name, (x) => {
          const q = queryView(x, p.nFeaturesIn, name)
          const m = p.centres.length
          const k = m === 0 ? new Float64Array(0) : kernelMatrix(h, p.kernel, x, p.centres)
          const values = new Float64Array(q.n)
          for (let i = 0; i < q.n; i++)
            for (let j = 0; j < m; j++) values[i] = (values[i] ?? 0) + (k[i * m + j] ?? 0) * (p.coef[j] ?? 0)
          return { m: q.n, k: 1, values }
        }),
      logs: { 'solver/loss': Log(), 'solver/mistakes': Log(), 'solver/supportVectors': Log() },
    },
    { ...defaults, ...hyper },
  )
}

/**
 * The kernel perceptron on the iterative base (binary): mistake-driven online classification with an optional budget on
 * the support vectors (`budget`: random removal, oldest removal, a shrinking forgetting factor or the Projectron). The
 * tape's `mistakes` and `supportVectors` show the run; `fittedAt(t)` is the classifier after `t` examples.
 *
 * @param hyper - The kernel options, `budget`, `size`, `forgetting`, `eta`, `epochs`.
 * @returns The estimator.
 * @concept online-kernel
 * @example
 * ```ts
 * import { kernelPerceptron } from 'omai-engine/kernel-methods'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[1, 1], [2, 1], [-1, -1], [-2, -1], [1.5, 2], [-1.5, -2]]
 * const fit = kernelPerceptron({ kernel: 'linear', gamma: 1, epochs: 3 }).fit({ x, y: [1, 1, 0, 0, 1, 0] })
 * toFlat(fit.predict([[3, 3], [-3, -3]])) // => [1, 0]
 * ```
 */
export function kernelPerceptron(
  hyper: Partial<OnlineClassifierHyper> = {},
): IterativeEstimator<OnlineClassifierHyper, ExpansionState, ExpansionParams, KernelClassifierVerbs> {
  return onlineClassifier(
    'kernel-methods/kernel-perceptron',
    CLASSIFIER_DEFAULTS,
    hyper,
    kernelPerceptronAlgorithm,
    () => ({
      forgetting: real(0.5, 1, {
        default: 0.9,
        label: 'forgetting factor',
        when: { key: 'budget', in: ['forgetron'] },
      }),
    }),
  )
}

/**
 * The kernel passive-aggressive classifier on the iterative base (binary): PA, PA-I or PA-II, with the same budgets as
 * {@link kernelPerceptron}.
 *
 * @param hyper - The kernel options, `variant`, `C`, the budget rule, `epochs`.
 * @returns The estimator.
 * @concept online-kernel
 * @example
 * ```ts
 * import { kernelPassiveAggressive } from 'omai-engine/kernel-methods'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[1, 1], [2, 1], [-1, -1], [-2, -1]]
 * const fit = kernelPassiveAggressive({ kernel: 'linear', gamma: 1, epochs: 2 }).fit({ x, y: [1, 1, 0, 0] })
 * toFlat(fit.predict([[3, 2], [-3, -2]])) // => [1, 0]
 * ```
 */
export function kernelPassiveAggressive(
  hyper: Partial<PassiveAggressiveHyper> = {},
): IterativeEstimator<PassiveAggressiveHyper, ExpansionState, ExpansionParams, KernelClassifierVerbs> {
  return onlineClassifier(
    'kernel-methods/kernel-passive-aggressive',
    kernelPassiveAggressiveDefaults,
    hyper,
    kernelPassiveAggressiveAlgorithm,
    (h) => ({
      variant: choice(['pa', 'pa1', 'pa2'], { default: h.variant, label: 'variant' }),
      C: real(1e-3, 1e3, { log: true, default: h.C, label: 'aggressiveness' }),
    }),
  )
}
