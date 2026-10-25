/**
 * The estimator protocol every learner in the library follows, scikit-learn's ideas in functional form: an `Estimator`
 * has hyperparameters and an optional tuning space, and `fit` returns a new `Fitted` value whose parameters are plain
 * JSON and whose verbs (`predict`, `transform`, `predictProba` and so on) are closures over them. `defineEstimator`,
 * cloning, capability guards, serialisation and revival, and the `EstimatorClass` wrapper for `new X().fit(x,
 * y).predict(x)` code.
 */

import { type Json, WhiteboxError } from '../base/index.ts'
import type { Context } from '../context/index.ts'
import type { Distribution } from '../dists/index.ts'
import { accuracyScore, r2Score } from '../metrics/index.ts'
import type { Key } from '../random/index.ts'
import type { Space } from '../space/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import type { RunHandle, TracePolicy, Tracer } from '../trace/index.ts'

/** Hyperparameters: an object of plain values, nested estimators or functions, by name. */
export type Hyper = object

/**
 * What an estimator fits: features `x` and, for supervised learners, targets `y` and sample weights. Every tabular
 * dataset (`RegressionData`, `ClassificationData`) is one.
 */
export interface LearnData {
  /** Features `[n, d]`. */
  readonly x: TensorLike
  /** Targets `[n]` (or `[n, k]`), class labels for classifiers. */
  readonly y?: TensorLike
  /** Sample weights `[n]`. */
  readonly weights?: TensorLike
}

/** The options every `fit` accepts: randomness (section 3.2) and tracing, or a parent run to nest under. */
export interface FitOptions {
  readonly key?: Key
  readonly ctx?: Context
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  /** Fit as a child of this run (cross-validation folds, pipeline steps). */
  readonly parent?: RunHandle
  /**
   * Whether to record how training went in the exposition (a decision surface at each step, say): default true. Fits
   * inside model selection (folds, trials) pass false, so a search does not pay for records nobody reads.
   */
  readonly explain?: boolean
}

/** What an estimator learns to do. */
export type EstimatorKind = 'regressor' | 'classifier' | 'transformer' | 'clusterer' | 'density' | 'outlier' | 'other'

/** The verbs of section 2 a fitted estimator may offer. Each maps new data to a tensor and never mutates. */
export interface Verbs {
  /** Point predictions `[m]` (labels for classifiers). */
  readonly predict?: (x: TensorLike) => Tensor
  /** Class probabilities `[m, k]`, columns in the order of `classes`. */
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Decision scores: `[m]` for binary problems (positive class), `[m, k]` otherwise. */
  readonly decisionFunction?: (x: TensorLike) => Tensor
  /** The full predictive distribution (batched over the `m` rows). */
  readonly predictDistribution?: (x: TensorLike) => Distribution
  /** Map data forwards. */
  readonly transform?: (x: TensorLike) => Tensor
  /** Map transformed data back. */
  readonly inverseTransform?: (x: TensorLike) => Tensor
  /** The default metric on data (accuracy for classifiers, R^2 for regressors), higher is better. */
  readonly score?: (data: LearnData) => number
}

/** Facts about a fitted estimator that the toolkit reads (selection, inspection, feature names). */
export interface FittedFacts {
  /** The number of features seen in `fit`. */
  readonly nFeaturesIn?: number
  /** The class labels, sorted, for classifiers. */
  readonly classes?: readonly number[]
  /** One importance per input feature (|coefficients|, impurity decrease): what RFE and `selectFromModel` rank. */
  readonly featureImportances?: readonly number[]
  /** The names of the output features of a transformer, given the input names. */
  readonly featureNamesOut?: (input?: readonly string[]) => string[]
}

/**
 * A fitted estimator: the estimator's name and kind, its learnt parameters as plain JSON (serialisable, see
 * {@link serialiseFitted}) and the verbs as functions closed over those parameters.
 */
export interface Fitted<P extends Json = Json> extends Verbs, FittedFacts {
  readonly estimator: string
  readonly kind: EstimatorKind
  readonly params: P
}

/**
 * The estimator contract (section 3.7): a name, the hyperparameters it was made with, an optional `hyperSpace` for
 * tuning, `fit` returning a new {@link Fitted} value (never mutating), `revive` rebuilding a fitted value from its
 * parameters, and `with` returning a copy with some hyperparameters changed.
 */
export interface Estimator<H extends Hyper = Hyper, F extends Fitted = Fitted> {
  readonly name: string
  readonly kind: EstimatorKind
  readonly hyper: H
  readonly hyperSpace?: Space
  fit(data: LearnData, options?: FitOptions): F
  revive(params: Json): F
  with(changes: Readonly<Record<string, unknown>>): Estimator<H, F>
  /**
   * Fit and return the transform of the training data, when that differs from `fit(data).transform(data.x)` (a
   * cross-fitted target encoding); pipelines call it for intermediate steps.
   */
  readonly fitTransform?: (
    data: LearnData,
    options?: FitOptions,
  ) => { readonly fitted: F; readonly transformed: Tensor }
}

/** How to build an estimator: {@link defineEstimator} turns a spec and hyperparameters into an {@link Estimator}. */
export interface EstimatorSpec<H extends Hyper, P extends Json, V extends Verbs & FittedFacts> {
  readonly name: string
  readonly kind: EstimatorKind
  /** The tunable hyperparameters, given the current ones. */
  readonly hyperSpace?: (hyper: H) => Space
  /** Learn the parameters (plain JSON). */
  fit(data: LearnData, hyper: H, options: FitOptions): P
  /** The verbs and facts of a fitted value, from its parameters. */
  methods(params: P, hyper: H): V
  /** Apply hyperparameter changes (default: a shallow merge); composites route `step.param` keys here. */
  update?(hyper: H, changes: Readonly<Record<string, unknown>>): H
  /** The transform of the training data, when it is not `transform(data.x)` (cross-fitting). */
  fitTransform?(fitted: Fitted<P> & V, data: LearnData): Tensor
}

function defaultScore(kind: EstimatorKind, verbs: Verbs): ((data: LearnData) => number) | undefined {
  const predict = verbs.predict
  if (predict === undefined) return undefined
  if (kind === 'classifier') {
    return (data) => accuracyScore(requireTargets(data, 'score'), predict(data.x), optionalWeights(data)).estimate.value
  }
  if (kind === 'regressor') {
    return (data) => r2Score(requireTargets(data, 'score'), predict(data.x), optionalWeights(data)).estimate.value
  }
  return undefined
}

function optionalWeights(data: LearnData): { readonly sampleWeight?: TensorLike } {
  return data.weights === undefined ? {} : { sampleWeight: data.weights }
}

/**
 * The targets of a dataset, or a clear error when a supervised step is given none.
 *
 * @param data - The data.
 * @param who - The caller (for the error).
 * @returns `data.y`.
 * @throws WhiteboxError with code `value` when `y` is missing.
 * @example
 * ```ts
 * import { requireTargets } from 'whitebox-engine/learn'
 * requireTargets({ x: [[1]], y: [2] }, 'demo') // => [2]
 * requireTargets({ x: [[1]] }, 'demo') // => throws value
 * ```
 */
export function requireTargets(data: LearnData, who: string): TensorLike {
  if (data.y === undefined) throw new WhiteboxError('value', `${who}: this step needs targets y`)
  return data.y
}

/**
 * Build an estimator from a spec and hyperparameters. The fitted value gets the default `score` of its kind when the
 * spec gives none, and `with` re-applies the spec to changed hyperparameters, so every estimator made this way can be
 * cloned, tuned, cross-validated and serialised.
 *
 * @param spec - Name, kind, hyperparameter space, `fit` and `methods`.
 * @param hyper - The hyperparameters.
 * @returns The estimator.
 * @concept estimator-protocol
 * @example
 * ```ts
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { full, mean, item, toFlat } from 'whitebox-engine/tensor'
 * const meanRegressor = defineEstimator(
 *   {
 *     name: 'mean',
 *     kind: 'regressor',
 *     fit: (data) => ({ mean: item(mean(data.y)) }),
 *     methods: (p) => ({ predict: (x) => full([x.length], p.mean) }),
 *   },
 *   {},
 * )
 * const fitted = meanRegressor.fit({ x: [[0], [1]], y: [1, 3] })
 * fitted.params // => { mean: 2 }
 * toFlat(fitted.predict([[5]])) // => [2]
 * meanRegressor.revive({ mean: 4 }).params // => { mean: 4 }
 * ```
 */
export function defineEstimator<H extends Hyper, P extends Json, V extends Verbs & FittedFacts>(
  spec: EstimatorSpec<H, P, V>,
  hyper: H,
): Estimator<H, Fitted<P> & V> {
  const build = (params: P): Fitted<P> & V => {
    const verbs = spec.methods(params, hyper)
    const score = verbs.score ?? defaultScore(spec.kind, verbs)
    return { ...verbs, ...(score === undefined ? {} : { score }), estimator: spec.name, kind: spec.kind, params }
  }
  const space = spec.hyperSpace?.(hyper)
  const fit = (data: LearnData, options: FitOptions = {}): Fitted<P> & V => build(spec.fit(data, hyper, options))
  const crossFit = spec.fitTransform
  return {
    name: spec.name,
    kind: spec.kind,
    hyper,
    ...(space === undefined ? {} : { hyperSpace: space }),
    ...(crossFit === undefined
      ? {}
      : {
          fitTransform: (data: LearnData, options?: FitOptions) => {
            const fitted = fit(data, options)
            return { fitted, transformed: crossFit(fitted, data) }
          },
        }),
    fit,
    revive: (params) => build(params as P),
    with: (changes) => defineEstimator(spec, spec.update?.(hyper, changes) ?? ({ ...hyper, ...changes } as H)),
  }
}

/**
 * A copy of an estimator with some hyperparameters changed (sklearn's `clone` plus `set_params`): the original is
 * untouched, and nothing fitted is carried over.
 *
 * @param estimator - The estimator.
 * @param changes - Hyperparameters to change (composites accept `step.param` keys).
 * @returns The new estimator.
 * @example
 * ```ts
 * import { cloneEstimator } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * cloneEstimator(standardScaler(), { withMean: false }).hyper.withMean // => false
 * ```
 */
export function cloneEstimator<H extends Hyper, F extends Fitted>(
  estimator: Estimator<H, F>,
  changes: Readonly<Record<string, unknown>> = {},
): Estimator<H, F> {
  return estimator.with(changes)
}

/**
 * Whether a value follows the estimator contract.
 *
 * @param x - Any value.
 * @returns `true` for an estimator.
 * @example
 * ```ts
 * import { isEstimator } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * isEstimator(standardScaler()) // => true
 * isEstimator({ fit: 1 }) // => false
 * ```
 */
export function isEstimator(x: unknown): x is Estimator {
  if (typeof x !== 'object' || x === null) return false
  const e = x as Partial<Estimator>
  return (
    typeof e.name === 'string' &&
    typeof e.fit === 'function' &&
    typeof e.revive === 'function' &&
    typeof e.with === 'function'
  )
}

/** A fitted value known to offer one verb. */
export type With<V extends keyof Verbs> = Fitted & Required<Pick<Verbs, V>>

/**
 * Whether a fitted value offers a verb (a capability guard).
 *
 * @param fitted - The fitted value.
 * @param verb - The verb.
 * @returns `true` when the verb is available; the type narrows.
 * @example
 * ```ts
 * import { can } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * const f = standardScaler().fit({ x: [[1], [3]] })
 * can(f, 'inverseTransform') // => true
 * can(f, 'predictProba') // => false
 * ```
 */
export function can<V extends keyof Verbs>(fitted: Fitted, verb: V): fitted is With<V> {
  return typeof fitted[verb] === 'function'
}

/**
 * A verb of a fitted value, or a clear error naming the estimator and the caller.
 *
 * @param fitted - The fitted value.
 * @param verb - The verb.
 * @param who - The caller (for the error).
 * @returns The verb's function.
 * @throws WhiteboxError with code `value` when the estimator lacks the verb.
 * @example
 * ```ts
 * import { requireVerb } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * const f = standardScaler().fit({ x: [[1], [3]] })
 * typeof requireVerb(f, 'transform', 'demo') // => 'function'
 * requireVerb(f, 'predict', 'demo') // => throws value
 * ```
 */
export function requireVerb<V extends keyof Verbs>(fitted: Fitted, verb: V, who: string): NonNullable<Verbs[V]> {
  const f = fitted[verb]
  if (typeof f !== 'function') {
    throw new WhiteboxError('value', `${who}: ${fitted.estimator} has no ${verb}`, { estimator: fitted.estimator })
  }
  return f as NonNullable<Verbs[V]>
}

/**
 * Whether a fitted value is a classifier (it has classes and predicts labels).
 *
 * @param fitted - The fitted value.
 * @returns `true` for classifiers.
 * @example
 * ```ts
 * import { isClassifier } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * isClassifier(standardScaler().fit({ x: [[1], [3]] })) // => false
 * ```
 */
export function isClassifier(fitted: Fitted): boolean {
  return fitted.kind === 'classifier' && fitted.classes !== undefined
}

/** A fitted value as JSON: the estimator's name and its parameters. */
export interface SerialisedFitted {
  readonly estimator: string
  readonly params: Json
}

/**
 * A fitted value as JSON (its verbs are rebuilt by {@link revive}).
 *
 * @param fitted - The fitted value.
 * @returns `{ estimator, params }`.
 * @example
 * ```ts
 * import { serialiseFitted } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * serialiseFitted(standardScaler().fit({ x: [[1], [3]] })).estimator // => 'standard-scaler'
 * ```
 */
export function serialiseFitted(fitted: Fitted): SerialisedFitted {
  return { estimator: fitted.estimator, params: fitted.params }
}

/**
 * Rebuild a fitted value from JSON, with the estimator that made it (after `JSON.parse`, a worker message or storage).
 *
 * @param estimator - The estimator (same name and hyperparameters as at fit time).
 * @param json - What {@link serialiseFitted} produced.
 * @returns The fitted value with its verbs.
 * @throws WhiteboxError with code `value` when the names differ.
 * @example
 * ```ts
 * import { revive, serialiseFitted } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const est = standardScaler()
 * const json = JSON.parse(JSON.stringify(serialiseFitted(est.fit({ x: [[1], [3]] }))))
 * toArray(revive(est, json).transform([[3]])) // => [[1]]
 * ```
 */
export function revive<H extends Hyper, F extends Fitted>(estimator: Estimator<H, F>, json: SerialisedFitted): F {
  if (json.estimator !== estimator.name) {
    throw new WhiteboxError('value', `revive: parameters of ${json.estimator} given to ${estimator.name}`)
  }
  return estimator.revive(json.params)
}

/**
 * The base of sklearn-style class wrappers: `new EstimatorClass(est).fit(x, y).predict(x)`. The wrapper holds the
 * latest fitted value (the only mutable state, replaced on every `fit`); the functional estimator underneath is
 * unchanged. Subclasses fix the estimator in their constructor (`class StandardScaler extends EstimatorClass`).
 *
 * @concept estimator-protocol
 * @example
 * ```ts
 * import { EstimatorClass } from 'whitebox-engine/learn'
 * import { standardScaler } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const scaler = new EstimatorClass(standardScaler())
 * toArray(scaler.fitTransform([[1], [3]])) // => [[-1], [1]]
 * scaler.getParams().withMean // => true
 * ```
 */
export class EstimatorClass<H extends Hyper = Hyper, F extends Fitted = Fitted> {
  /** The functional estimator. */
  estimator: Estimator<H, F>
  /** The latest fitted value, if any. */
  fitted: F | undefined

  /**
   * Wrap an estimator.
   *
   * @param estimator - The functional estimator.
   */
  constructor(estimator: Estimator<H, F>) {
    this.estimator = estimator
    this.fitted = undefined
  }

  /** The fitted value, or an error when `fit` has not been called. */
  get model(): F {
    if (this.fitted === undefined) throw new WhiteboxError('value', `${this.estimator.name}: call fit first`)
    return this.fitted
  }

  /**
   * Fit on features and optional targets.
   *
   * @param x - Features.
   * @param y - Targets.
   * @param options - Weights, randomness and tracing.
   * @returns This wrapper.
   */
  fit(x: TensorLike, y?: TensorLike, options: FitOptions & { readonly weights?: TensorLike } = {}): this {
    const { weights, ...rest } = options
    this.fitted = this.estimator.fit(
      { x, ...(y === undefined ? {} : { y }), ...(weights === undefined ? {} : { weights }) },
      rest,
    )
    return this
  }

  /**
   * Predictions of the fitted model.
   *
   * @param x - Features.
   * @returns Predictions.
   */
  predict(x: TensorLike): Tensor {
    return requireVerb(this.model, 'predict', 'predict')(x)
  }

  /**
   * Class probabilities of the fitted model.
   *
   * @param x - Features.
   * @returns Probabilities.
   */
  predictProba(x: TensorLike): Tensor {
    return requireVerb(this.model, 'predictProba', 'predictProba')(x)
  }

  /**
   * Decision scores of the fitted model.
   *
   * @param x - Features.
   * @returns Scores.
   */
  decisionFunction(x: TensorLike): Tensor {
    return requireVerb(this.model, 'decisionFunction', 'decisionFunction')(x)
  }

  /**
   * Transform data with the fitted model.
   *
   * @param x - Features.
   * @returns Transformed features.
   */
  transform(x: TensorLike): Tensor {
    return requireVerb(this.model, 'transform', 'transform')(x)
  }

  /**
   * Map transformed data back.
   *
   * @param x - Transformed features.
   * @returns Features.
   */
  inverseTransform(x: TensorLike): Tensor {
    return requireVerb(this.model, 'inverseTransform', 'inverseTransform')(x)
  }

  /**
   * Fit, then transform the same data.
   *
   * @param x - Features.
   * @param y - Targets.
   * @returns Transformed features.
   */
  fitTransform(x: TensorLike, y?: TensorLike): Tensor {
    return this.fit(x, y).transform(x)
  }

  /**
   * The default metric of the fitted model.
   *
   * @param x - Features.
   * @param y - Targets.
   * @returns The score (higher is better).
   */
  score(x: TensorLike, y: TensorLike): number {
    return requireVerb(this.model, 'score', 'score')({ x, y })
  }

  /** The hyperparameters. */
  getParams(): H {
    return this.estimator.hyper
  }

  /**
   * Change hyperparameters (dropping the fitted model).
   *
   * @param changes - Hyperparameters to change.
   * @returns This wrapper.
   */
  setParams(changes: Readonly<Record<string, unknown>>): this {
    this.estimator = this.estimator.with(changes)
    this.fitted = undefined
    return this
  }
}
