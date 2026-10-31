/**
 * Wrappers for several targets at once: `multiOutputRegressor` and `multiOutputClassifier` fit one base estimator per
 * output (independently), `regressorChain` fits them in an order with each output's features extended by the earlier
 * outputs (the true ones in training, the base's own predictions at prediction time), so the chain captures
 * dependences between targets that independent fits cannot. Each is an estimator on the iterative base that fits one
 * output per step, logging `solver/problems`; `outputChain` and `independentOutputs` are the engines `multilabel.ts`
 * builds its classifiers on.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { confidence, matrix, type StagedState, stagedAlgorithm } from './_staged.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { Estimator, EstimatorKind, Fitted, LearnData } from './estimator.ts'
import { fromView, hstackViews, type MatrixView, matrixView } from './validation.ts'

/** How a chain passes an earlier output on as a feature: the prediction, or the base's score. */
export type ChainMethod = 'predict' | 'predict_proba' | 'decision_function'

/** The hyperparameters of the output wrappers. */
export interface OutputsHyper {
  readonly estimator: Estimator
  /** Whether outputs are chained. */
  readonly chain: boolean
  /** The chain's order: output indices, or `'random'` (a permutation from the run's key); default the given order. */
  readonly order?: 'random' | readonly number[]
  /** What an earlier output contributes as a feature to later ones (chains only). */
  readonly chainMethod: ChainMethod
  /** Whether the outputs are class labels (the score is then exact match) or numbers (the mean R squared). */
  readonly task: 'classification' | 'regression'
}

/** The plan of an output wrapper: the order in which outputs are fitted. */
export type OutputsPlan = { readonly order: number[]; readonly constant: (number | null)[] }

/** The fitted parameters: the order, and each output's base parameters (or the constant it always takes). */
export type OutputsParams = {
  readonly nOutputs: number
  readonly nFeatures: number
  readonly plan: OutputsPlan
  readonly models: Json[]
}

/** What a fitted output wrapper offers. */
export interface OutputsVerbs {
  readonly nFeaturesIn: number
  /** The predictions `[m, T]`, columns in the original output order. */
  readonly predict: (x: TensorLike) => Tensor
  /** The base's class probabilities of each output (classifiers with probabilities), one `[m, k]` tensor each. */
  readonly predictProbaEach?: (x: TensorLike) => Tensor[]
  /** For binary outputs, the probability of the positive class `[m, T]`. */
  readonly predictProba?: (x: TensorLike) => Tensor
  /** The decision functions `[m, T]` of binary outputs, when the base has one. */
  readonly decisionFunction?: (x: TensorLike) => Tensor
  /** Exact match across the outputs (classification) or the mean R squared (regression). */
  readonly score: (data: LearnData) => number
  readonly nProblems: number
}

const CONSTANT = 'constant'

type ConstantModel = { readonly [CONSTANT]: number }

const isConstant = (m: Json): m is ConstantModel =>
  typeof m === 'object' && m !== null && !Array.isArray(m) && typeof (m as { constant?: unknown }).constant === 'number'

const isConstantFit = (f: Fitted | ConstantModel): f is ConstantModel => !('estimator' in f)

function targetsOf(data: LearnData, who: string): MatrixView {
  if (data.y === undefined) throw new MdmlError('value', `${who}: needs targets y of shape [n, outputs]`)
  return matrixView(data.y, who)
}

function orderOf(order: OutputsHyper['order'], t: number, key: Key): number[] {
  if (order === 'random') return Array.from(toFlat(permutation(key, t)))
  if (order === undefined) return Array.from({ length: t }, (_, i) => i)
  if ([...order].sort((a, b) => a - b).some((v, i) => v !== i) || order.length !== t) {
    throw new MdmlError('value', `outputs: the order ${order.join(',')} is not a permutation of 0..${t - 1}`)
  }
  return [...order]
}

function stageData(data: LearnData, plan: OutputsPlan, index: number, chain: boolean): LearnData {
  const x = matrixView(data.x, 'outputs')
  const y = targetsOf(data, 'outputs')
  const column = (j: number): MatrixView => ({
    n: y.n,
    d: 1,
    data: Float64Array.from({ length: y.n }, (_, i) => y.data[i * y.d + j] ?? 0),
  })
  const own = plan.order[index] ?? 0
  const augmented = chain && index > 0 ? hstackViews([x, ...plan.order.slice(0, index).map(column)]) : x
  return { ...data, x: fromView(augmented), y: matrix(column(own).data, [y.n]) }
}

/** The feature an earlier output contributes to the chain. */
function featureOf(f: Fitted | ConstantModel, x: TensorLike, n: number, method: ChainMethod): Float64Array {
  if (isConstantFit(f)) return new Float64Array(n).fill(f.constant)
  if (method === 'predict' && f.predict !== undefined) return Float64Array.from(toFlat(f.predict(x)))
  const verb =
    method === 'predict_proba' ? f.predictProba : method === 'decision_function' ? f.decisionFunction : undefined
  if (verb === undefined) return confidence(f, x)
  const t = verb(x)
  const k = t.shape[1] ?? 1
  const flat = toFlat(t)
  return Float64Array.from({ length: flat.length / k }, (_, r) => flat[r * k + (k - 1)] ?? 0)
}

/**
 * The engine of the output wrappers: independent fits, or a chain, of one base estimator per output, on the iterative
 * base. Prefer {@link multiOutputRegressor}, {@link multiOutputClassifier}, {@link regressorChain} and `multilabel`'s
 * classifiers.
 *
 * @param name - The estimator's name.
 * @param hyper - The base, whether to chain, the order, the chain method and the task.
 * @param kind - The estimator kind reported.
 * @returns The estimator.
 * @example
 * ```ts
 * import { kNeighboursRegressor, outputsEstimator } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const est = outputsEstimator('twins', {
 *   estimator: kNeighboursRegressor({ nNeighbors: 1 }),
 *   chain: false,
 *   chainMethod: 'predict',
 *   task: 'regression',
 * })
 * toArray(est.fit({ x: [[0], [1]], y: [[0, 5], [1, 6]] }).predict([[0.9]])) // => [[1, 6]]
 * ```
 */
export function outputsEstimator(
  name: string,
  hyper: OutputsHyper,
  kind: EstimatorKind = 'other',
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return iterative<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs>(
    {
      name,
      kind,
      defaults: hyper,
      algorithm: (data, h) => {
        const y = targetsOf(data, name)
        const constant = Array.from({ length: y.d }, (_, j) => {
          if (h.task !== 'classification' && !h.chain) return null
          const first = y.data[j] ?? 0
          for (let i = 1; i < y.n; i++) if (y.data[i * y.d + j] !== first) return null
          return h.task === 'classification' ? first : null
        })
        return stagedAlgorithm<OutputsPlan>(h.estimator, {
          name,
          plan: (key) => ({ order: orderOf(h.order, y.d, key), constant }),
          total: (plan) => plan.order.length,
          stage: (plan, index) => stageData(data, plan, index, h.chain),
          fixed: (plan, index) => {
            const c = plan.constant[plan.order[index] ?? 0]
            return c === null || c === undefined ? undefined : { [CONSTANT]: c }
          },
        })
      },
      maxSteps: (_h, data) => Math.max(1, targetsOf(data, name).d),
      params: (s, data) => ({
        nOutputs: targetsOf(data, name).d,
        nFeatures: matrixView(data.x, name).d,
        plan: s.plan,
        models: s.models,
      }),
      verbs: (p, h) => outputsVerbs(p, h),
      logs: { 'solver/problems': Log() },
    },
    hyper,
  )
}

function outputsVerbs(p: OutputsParams, h: OutputsHyper): OutputsVerbs {
  const T = p.nOutputs
  const fits: (Fitted | ConstantModel)[] = p.models.map((m) => (isConstant(m) ? m : h.estimator.revive(m)))
  const method = h.chain ? h.chainMethod : 'predict'
  /**
   * One pass over the fits on rows: each sees the features and, in a chain, the earlier fits' contributions; `visit`
   * is given the fit and the rows it saw.
   */
  const pass = (x: TensorLike, visit: (f: Fitted | ConstantModel, rows: Tensor, i: number) => void): number => {
    const m = matrixView(x, 'outputs', { nFeatures: p.nFeatures })
    const features: Float64Array[] = []
    fits.forEach((f, i) => {
      const earlier = h.chain ? features.slice(0, i) : []
      const view = earlier.length === 0 ? m : hstackViews([m, ...earlier.map((c) => ({ n: m.n, d: 1, data: c }))])
      const rows = fromView(view)
      visit(f, rows, i)
      features.push(featureOf(f, rows, m.n, method))
    })
    return m.n
  }
  /** Put the columns of the fits (in chain order) back in the original output order. */
  const unordered = (cols: readonly Float64Array[], n: number): Tensor => {
    const out = new Float64Array(n * T)
    p.plan.order.forEach((original, i) => {
      for (let r = 0; r < n; r++) out[r * T + original] = cols[i]?.[r] ?? Number.NaN
    })
    return matrix(out, [n, T])
  }
  const predict = (x: TensorLike): Tensor => {
    const cols: Float64Array[] = []
    const n = pass(x, (f, rows) => {
      cols.push(
        isConstantFit(f)
          ? new Float64Array(matrixView(rows, 'outputs').n).fill(f.constant)
          : Float64Array.from(toFlat((f.predict as (x: TensorLike) => Tensor)(rows))),
      )
    })
    return unordered(cols, n)
  }
  const respond = (x: TensorLike, verb: 'predictProba' | 'decisionFunction') => {
    const cols: Float64Array[] = []
    const raw: Tensor[] = []
    const n = pass(x, (f, rows) => {
      const rowsN = matrixView(rows, 'outputs').n
      if (isConstantFit(f)) {
        cols.push(new Float64Array(rowsN).fill(verb === 'predictProba' ? f.constant : (f.constant > 0 ? 1 : -1) * 1e9))
        raw.push(matrix(new Float64Array(rowsN).fill(f.constant), [rowsN, 1]))
        return
      }
      const t = (f[verb] as (x: TensorLike) => Tensor)(rows)
      const k = t.shape[1] ?? 1
      const flat = toFlat(t)
      raw.push(t)
      cols.push(Float64Array.from({ length: rowsN }, (_, r) => flat[r * k + (k - 1)] ?? 0))
    })
    return { n, cols, raw }
  }
  const everyHas = (verb: 'predictProba' | 'decisionFunction'): boolean =>
    fits.every((f) => isConstantFit(f) || f[verb] !== undefined)
  const score = (data: LearnData): number => {
    const pred = toFlat(predict(data.x))
    const truth = targetsOf(data, 'outputs')
    if (h.task === 'classification') {
      let hits = 0
      for (let i = 0; i < truth.n; i++) {
        let all = true
        for (let j = 0; j < T; j++) if (pred[i * T + j] !== truth.data[i * T + j]) all = false
        if (all) hits++
      }
      return hits / truth.n
    }
    let total = 0
    for (let j = 0; j < T; j++) {
      let mean = 0
      for (let i = 0; i < truth.n; i++) mean += (truth.data[i * T + j] ?? 0) / truth.n
      let ssRes = 0
      let ssTot = 0
      for (let i = 0; i < truth.n; i++) {
        ssRes += ((truth.data[i * T + j] ?? 0) - (pred[i * T + j] ?? 0)) ** 2
        ssTot += ((truth.data[i * T + j] ?? 0) - mean) ** 2
      }
      total += ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - ssRes / ssTot
    }
    return total / T
  }
  return {
    nFeaturesIn: p.nFeatures,
    predict,
    ...(everyHas('predictProba')
      ? {
          predictProba: (x: TensorLike): Tensor => {
            const { n, cols } = respond(x, 'predictProba')
            return unordered(cols, n)
          },
          predictProbaEach: (x: TensorLike): Tensor[] => {
            const { raw } = respond(x, 'predictProba')
            const byOriginal = new Array<Tensor>(T)
            p.plan.order.forEach((original, i) => {
              const t = raw[i]
              if (t !== undefined) byOriginal[original] = t
            })
            return byOriginal
          },
        }
      : {}),
    ...(everyHas('decisionFunction')
      ? {
          decisionFunction: (x: TensorLike): Tensor => {
            const { n, cols } = respond(x, 'decisionFunction')
            return unordered(cols, n)
          },
        }
      : {}),
    score,
    nProblems: p.models.length,
  }
}

/**
 * Independent fits of a base estimator on each output (`sklearn.multioutput`), the engine of two public wrappers.
 *
 * @param name - The estimator's name.
 * @param estimator - The base estimator.
 * @param task - Whether the outputs are class labels or numbers.
 * @returns The estimator.
 * @example
 * ```ts
 * import { independentOutputs, kNeighboursRegressor } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const est = independentOutputs('twins', kNeighboursRegressor({ nNeighbors: 1 }), 'regression')
 * toArray(est.fit({ x: [[0], [1]], y: [[0, 5], [1, 6]] }).predict([[0.1]])) // => [[0, 5]]
 * ```
 */
export function independentOutputs(
  name: string,
  estimator: Estimator,
  task: OutputsHyper['task'],
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return outputsEstimator(name, { estimator, chain: false, chainMethod: 'predict', task })
}

/**
 * A chain of a base estimator over the outputs, the engine of two public wrappers.
 *
 * @param name - The estimator's name.
 * @param estimator - The base estimator.
 * @param task - Whether the outputs are class labels or numbers.
 * @param options - The chain's `order` and `chainMethod`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { kNeighboursRegressor, outputChain } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const est = outputChain('link', kNeighboursRegressor({ nNeighbors: 1 }), 'regression', { order: [1, 0] })
 * toArray(est.fit({ x: [[0], [1]], y: [[0, 5], [1, 6]] }).predict([[0.1]])) // => [[0, 5]]
 * ```
 */
export function outputChain(
  name: string,
  estimator: Estimator,
  task: OutputsHyper['task'],
  options: { readonly order?: OutputsHyper['order']; readonly chainMethod?: ChainMethod } = {},
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return outputsEstimator(name, {
    estimator,
    chain: true,
    chainMethod: options.chainMethod ?? 'predict',
    task,
    ...(options.order === undefined ? {} : { order: options.order }),
  })
}

/**
 * One regressor per target (`sklearn.multioutput.MultiOutputRegressor`): the targets `[n, T]` are fitted
 * independently, so the model shares nothing between them. A session fits one target per step.
 *
 * @param estimator - The base regressor.
 * @returns The estimator; `predict` gives `[m, T]`, `score` the mean R squared.
 * @example
 * ```ts
 * import { multiOutputRegressor } from 'mdml-engine/learn'
 * import { kNeighboursRegressor } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const f = multiOutputRegressor(kNeighboursRegressor({ nNeighbors: 1 })).fit({ x: [[0], [1], [2]], y: [[0, 10], [1, 11], [2, 12]] })
 * toArray(f.predict([[1.1]])) // => [[1, 11]]
 * ```
 */
export function multiOutputRegressor(
  estimator: Estimator,
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return independentOutputs('multi-output-regressor', estimator, 'regression')
}

/**
 * One classifier per output (`sklearn.multioutput.MultiOutputClassifier`): each output is a class label of its own,
 * fitted independently. `predict` gives `[m, T]`, `predictProbaEach` the probabilities of each output, `score` the
 * share of rows with every output right.
 *
 * @param estimator - The base classifier.
 * @returns The estimator.
 * @example
 * ```ts
 * import { multiOutputClassifier, kNeighboursClassifier } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const x = [[0], [1], [10], [11]]
 * const f = multiOutputClassifier(kNeighboursClassifier({ nNeighbors: 1 })).fit({ x, y: [[0, 5], [0, 5], [1, 7], [1, 7]] })
 * toArray(f.predict([[0.4], [10.6]])) // => [[0, 5], [1, 7]]
 * ```
 */
export function multiOutputClassifier(
  estimator: Estimator,
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return independentOutputs('multi-output-classifier', estimator, 'classification')
}

/**
 * A chain of regressors (`sklearn.multioutput.RegressorChain`): the targets are fitted in `order`, each regressor
 * seeing the features and the true earlier targets; at prediction time it sees the chain's own predictions. A
 * session fits one link per step.
 *
 * @param estimator - The base regressor.
 * @param options - The `order` (a permutation, or `'random'`; default the given order).
 * @returns The estimator.
 * @example
 * ```ts
 * import { regressorChain, kNeighboursRegressor } from 'mdml-engine/learn'
 * import { toArray } from 'mdml-engine/tensor'
 * const f = regressorChain(kNeighboursRegressor({ nNeighbors: 1 }), { order: [1, 0] }).fit({
 *   x: [[0], [1], [2]],
 *   y: [[0, 10], [1, 11], [2, 12]],
 * })
 * toArray(f.predict([[2.2]])) // => [[2, 12]]
 * ```
 */
export function regressorChain(
  estimator: Estimator,
  options: { readonly order?: OutputsHyper['order'] } = {},
): IterativeEstimator<OutputsHyper, StagedState<OutputsPlan>, OutputsParams, OutputsVerbs> {
  return outputChain('regressor-chain', estimator, 'regression', options)
}
