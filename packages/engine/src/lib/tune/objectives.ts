import { type Json, meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { type ClassificationData, type RegressionData, take } from '../data/index.ts'
import type { Metric } from '../metrics/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  type Agent,
  type Env,
  evaluatePolicy,
  greedyPolicyOf,
  type TrainState,
  trainingAlgorithm,
} from '../rl/index.ts'
import type { Space, SpaceValue } from '../space/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, type RunHandle, runAlgorithm } from '../trace/index.ts'
import { kFoldIndices, stratifiedKFoldIndices } from './_folds.ts'
import type { StudyDirection } from './study.ts'

/**
 * What an objective sees of its trial: the trial number, a key, the trial's run (to nest the objective's own
 * algorithms as child runs), the budget a multi-fidelity schedule gave it, and `report`, which records an intermediate
 * value and asks the study's pruner whether to stop.
 */
export interface TrialContext {
  readonly number: number
  readonly key: Key
  readonly run: RunHandle
  /** The resource assigned by a schedule (epochs, samples, steps), if any. */
  readonly budget?: number
  /**
   * Record an intermediate value at a step; returns `true` when the trial has been pruned, after which the objective
   * should return promptly (its result is ignored).
   */
  report(step: number, value: number): boolean
  /** Whether the trial has been pruned. */
  shouldPrune(): boolean
}

/** What an objective returns: one value, one value per direction, or values with extra attributes. */
export type ObjectiveResult =
  | number
  | readonly number[]
  | { readonly values: number | readonly number[]; readonly attrs?: Readonly<Record<string, Json>> }

/**
 * A selection problem: a function from a configuration to one or more values, each minimised or maximised, optionally
 * reporting intermediate values for pruning. Every adapter below builds one, so the same samplers, pruners and
 * schedules tune estimators, training runs, agents, programs and plain functions.
 *
 * @concept hyperparameter-optimisation
 */
export interface Objective {
  readonly name: string
  readonly directions: readonly StudyDirection[]
  /** The default search space (an estimator's `hyperSpace`). */
  readonly space?: Space
  /** Names of the values, for tables and plots. */
  readonly metricNames?: readonly string[]
  evaluate(params: SpaceValue, trial: TrialContext): ObjectiveResult
}

/** Options shared by the objective adapters. */
export interface ObjectiveOptions {
  readonly name?: string
  readonly direction?: StudyDirection
  readonly space?: Space
  readonly metricNames?: readonly string[]
}

function common(options: ObjectiveOptions, name: string, direction: StudyDirection): Omit<Objective, 'evaluate'> {
  return {
    name: options.name ?? name,
    directions: [options.direction ?? direction],
    ...(options.space === undefined ? {} : { space: options.space }),
    ...(options.metricNames === undefined ? {} : { metricNames: options.metricNames }),
  }
}

/**
 * Any function as an objective (Optuna's `def objective(trial)`): it receives the configuration and the trial
 * context, and may call `trial.report(step, value)` and stop when that returns `true`.
 *
 * @param f - The function.
 * @param options - Name, direction (default minimise) or directions, default space and value names.
 * @returns The objective.
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { functionObjective, tune } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const objective = functionObjective((x) => ((x as number) - 0.3) ** 2)
 * Math.abs((tune(objective, real(0, 1), { nTrials: 200, seed: 1 }).best?.params as number) - 0.3) < 0.01 // => true
 * ```
 */
export function functionObjective(
  f: (params: SpaceValue, trial: TrialContext) => ObjectiveResult,
  options: ObjectiveOptions & { readonly directions?: readonly StudyDirection[] } = {},
): Objective {
  const base = common(options, 'function', 'minimize')
  return { ...base, ...(options.directions === undefined ? {} : { directions: options.directions }), evaluate: f }
}

/**
 * Several objectives evaluated on the same configuration, one value each (a multi-objective problem: accuracy and
 * size, return and cost). Intermediate reports of the first objective drive pruning; the others' are not recorded.
 *
 * @param objectives - The objectives.
 * @returns The combined objective (directions and names concatenated).
 * @throws WhiteboxError with code `value` when no objective is given.
 * @concept multi-objective-optimisation
 * @example
 * ```ts
 * import { functionObjective, multiObjective } from 'whitebox-engine/tune'
 * const both = multiObjective([
 *   functionObjective((x) => (x as number) ** 2, { name: 'error' }),
 *   functionObjective((x) => x as number, { name: 'size', direction: 'maximize' }),
 * ])
 * both.directions // => ['minimize', 'maximize']
 * ```
 */
export function multiObjective(objectives: readonly Objective[]): Objective {
  if (objectives.length === 0) throw new WhiteboxError('value', 'multiObjective: need at least one objective')
  const first = objectives[0] as Objective
  return {
    name: objectives.map((o) => o.name).join('+'),
    directions: objectives.flatMap((o) => o.directions),
    ...(first.space === undefined ? {} : { space: first.space }),
    metricNames: objectives.flatMap(
      (o) => o.metricNames ?? (o.directions.length === 1 ? [o.name] : o.directions.map((_, j) => `${o.name}_${j}`)),
    ),
    evaluate: (params, trial) => {
      const values: number[] = []
      for (const [i, o] of objectives.entries()) {
        const context: TrialContext =
          i === 0 ? trial : { ...trial, key: foldIn(trial.key, i), report: () => trial.shouldPrune() }
        const r = o.evaluate(params, context)
        if (trial.shouldPrune()) return values
        if (typeof r === 'number') values.push(r)
        else if (Array.isArray(r)) values.push(...(r as readonly number[]))
        else {
          const v = (r as { readonly values: number | readonly number[] }).values
          if (typeof v === 'number') values.push(v)
          else values.push(...v)
        }
      }
      return values
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Estimators

/** A fitted model: predictions, and class probabilities for classifiers. */
export interface FittedModel {
  predict(x: TensorLike): TensorLike
  predictProba?(x: TensorLike): TensorLike
}

/**
 * The part of the estimator contract that tuning needs: a name, an optional `hyperSpace`, `fit`, and `withHyper` to
 * make the same estimator with other hyperparameters (scikit-learn's `clone(est).set_params(**params)`).
 * TODO(learn): `learn/estimator.ts` will define the full contract; it satisfies this one.
 */
export interface TunableEstimator<D = TabularData> {
  readonly name: string
  readonly hyperSpace?: Space
  fit(data: D, options?: { readonly key?: Key }): FittedModel
  withHyper?(params: SpaceValue): TunableEstimator<D>
}

/** Tabular data with features and targets (the schemas estimators train on). */
export type TabularData = RegressionData | ClassificationData

/** An estimator to re-create with each configuration, or a function making one from a configuration. */
export type EstimatorSource<D = TabularData> = TunableEstimator<D> | ((params: SpaceValue) => TunableEstimator<D>)

/** One cross-validation split: row indices to train on and to test on. */
export interface Fold {
  readonly train: readonly number[]
  readonly test: readonly number[]
}

/** A splitter: the folds of a dataset (a key for shuffling ones). */
export type Splitter<D = TabularData> = (data: D, key: Key) => readonly Fold[]

/**
 * Make the estimator for a configuration.
 *
 * @param source - An estimator with `withHyper`, or a factory.
 * @param params - The configuration.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` when an estimator lacks `withHyper`.
 * @example
 * ```ts
 * import { estimatorFor } from 'whitebox-engine/tune'
 * const make = (p: unknown) => ({ name: `k=${(p as { k: number }).k}`, fit: () => ({ predict: () => [0] }) })
 * estimatorFor(make, { k: 3 }).name // => 'k=3'
 * ```
 */
export function estimatorFor<D>(source: EstimatorSource<D>, params: SpaceValue): TunableEstimator<D> {
  if (typeof source === 'function') return source(params)
  if (source.withHyper === undefined)
    throw new WhiteboxError('value', `${source.name}: tuning needs withHyper (or pass a factory)`)
  return source.withHyper(params)
}

function rows(data: TabularData): number {
  return data.y.shape[0] ?? 0
}

/**
 * The default folds for `cv: k`, as scikit-learn picks them: stratified k-fold for classification, k-fold otherwise,
 * unshuffled.
 *
 * @param data - The data.
 * @param k - The number of folds.
 * @returns The folds.
 * @throws WhiteboxError with code `value` for fewer than 2 folds or more folds than rows.
 * @example
 * ```ts
 * import { defaultFolds } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * defaultFolds(xy([[1], [2], [3], [4], [5]], [1, 2, 3, 4, 5]), 2).map((f) => f.test) // => [[0, 1, 2], [3, 4]]
 * ```
 */
export function defaultFolds(data: TabularData, k: number): Fold[] {
  return data.kind === 'classification' ? stratifiedKFoldIndices(toFlat(data.y), k) : kFoldIndices(rows(data), k)
}

/** Options of {@link estimatorObjective}. */
export interface EstimatorObjectiveOptions<D = TabularData> extends ObjectiveOptions {
  /** The metric scored on each test fold; its spec gives the direction. */
  readonly metric: Metric
  /** A splitter, or a number of folds (default 5). */
  readonly cv?: number | Splitter<D>
  /** Score `predictProba` instead of `predict` (default: when the metric reads probabilities or scores). */
  readonly response?: 'predict' | 'predictProba'
  /** The key of a shuffling splitter (default the trial's, so every trial sees different folds; fix it to compare). */
  readonly splitKey?: Key
}

function responseOf(options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>): 'predict' | 'predictProba' {
  const inputs = options.metric.spec.inputs
  return options.response ?? (inputs === 'probabilities' || inputs === 'scores' ? 'predictProba' : 'predict')
}

/** Fit on one dataset and score the metric on another. */
function fitAndScore(
  source: EstimatorSource,
  params: SpaceValue,
  train: TabularData,
  test: TabularData,
  options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>,
  key: Key,
): number {
  const estimator = estimatorFor(source, params)
  const fitted = estimator.fit(train, { key })
  let predicted: TensorLike
  if (responseOf(options) === 'predictProba') {
    if (fitted.predictProba === undefined)
      throw new WhiteboxError('value', `${estimator.name}: the metric needs predictProba`)
    predicted = fitted.predictProba(test.x)
  } else predicted = fitted.predict(test.x)
  return options.metric(test.y, asTensor(predicted)).estimate.value
}

/** The fold scores of a configuration (scikit-learn's `cross_val_score`), reporting the running mean after each fold. */
function crossValidate(
  source: EstimatorSource,
  data: TabularData,
  params: SpaceValue,
  folds: readonly Fold[],
  options: Pick<EstimatorObjectiveOptions, 'metric' | 'response'>,
  trial: TrialContext,
): number[] {
  const scores: number[] = []
  for (const [i, fold] of folds.entries()) {
    scores.push(
      fitAndScore(source, params, take(data, fold.train), take(data, fold.test), options, foldIn(trial.key, i)),
    )
    if (trial.report(i, meanOf(scores))) break
  }
  return scores
}

/**
 * Cross-validated estimator performance as an objective: fit the estimator with each configuration on every training
 * fold, score the metric on the test fold, and return the mean (direction from the metric's spec). The running mean
 * is reported after each fold, so pruners can stop hopeless configurations early. Attributes record the fold scores
 * and their standard deviation.
 *
 * @param estimator - An estimator with `withHyper`, or a factory from configurations to estimators.
 * @param data - The data.
 * @param options - The metric, the folds (`cv`: a splitter or a count, default 5), the prediction to score, the
 *   splitter's key, and a default space (default the estimator's `hyperSpace`).
 * @returns The objective.
 * @concept cross-validation
 * @example
 * ```ts
 * import { estimatorObjective, tune, gridSampler } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * import { dict, choice } from 'whitebox-engine/space'
 * // A constant predictor: the configuration is the constant.
 * const make = (p: unknown) => ({ name: 'constant', fit: () => ({ predict: (x: unknown) => (x as { shape: number[] }).shape.slice(0, 1).map(() => (p as { c: number }).c).flatMap((c) => Array((x as { shape: number[] }).shape[0]).fill(c)) }) })
 * const data = xy([[0], [1], [2], [3]], [2, 2, 2, 2])
 * const objective = estimatorObjective(make, data, { metric: meanSquaredError, cv: 2 })
 * tune(objective, dict({ c: choice([0, 2, 4]) }), { sampler: gridSampler() }).best?.params // => { c: 2 }
 * ```
 */
export function estimatorObjective(
  estimator: EstimatorSource,
  data: TabularData,
  options: EstimatorObjectiveOptions,
): Objective {
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  const spec = options.metric.spec
  return {
    ...common(
      { ...options, ...(space === undefined ? {} : { space }) },
      `cv:${spec.name}`,
      spec.direction === 'higher' ? 'maximize' : 'minimize',
    ),
    metricNames: options.metricNames ?? [spec.name],
    evaluate: (params, trial) => {
      const cv = options.cv ?? 5
      const folds =
        typeof cv === 'number' ? defaultFolds(data, cv) : cv(data, options.splitKey ?? foldIn(trial.key, 1e6))
      const scores = crossValidate(estimator, data, params, folds, options, trial)
      return {
        values: meanOf(scores),
        attrs: { foldScores: scores, std: Math.sqrt(varianceOf(scores)) },
      }
    },
  }
}

/**
 * Validation-set performance as an objective: fit on the training data, score the metric on the validation data.
 *
 * @param estimator - An estimator with `withHyper`, or a factory.
 * @param train - Training data.
 * @param validation - Validation data.
 * @param options - The metric, the prediction to score and a default space.
 * @returns The objective.
 * @concept model-selection
 * @example
 * ```ts
 * import { holdoutObjective } from 'whitebox-engine/tune'
 * import { xy } from 'whitebox-engine/data'
 * import { meanSquaredError } from 'whitebox-engine/metrics'
 * const make = () => ({ name: 'zero', fit: () => ({ predict: () => [0, 0] }) })
 * holdoutObjective(make, xy([[0], [1]], [0, 1]), xy([[0], [1]], [1, 1]), { metric: meanSquaredError }).directions // => ['minimize']
 * ```
 */
export function holdoutObjective(
  estimator: EstimatorSource,
  train: TabularData,
  validation: TabularData,
  options: Omit<EstimatorObjectiveOptions, 'cv' | 'splitKey'>,
): Objective {
  const space = options.space ?? (typeof estimator === 'function' ? undefined : estimator.hyperSpace)
  const spec = options.metric.spec
  return {
    ...common(
      { ...options, ...(space === undefined ? {} : { space }) },
      `holdout:${spec.name}`,
      spec.direction === 'higher' ? 'maximize' : 'minimize',
    ),
    metricNames: options.metricNames ?? [spec.name],
    evaluate: (params, trial) => {
      const score = fitAndScore(estimator, params, train, validation, options, trial.key)
      trial.report(0, score)
      return score
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Training runs, agents, programs

/** Options of {@link trainingObjective}. */
export interface TrainingObjectiveOptions<S> extends ObjectiveOptions {
  /** The value of a training state (validation loss, accuracy). */
  readonly value: (state: S) => number
  /** Training steps (epochs) when no schedule gives a budget (default 100). */
  readonly maxSteps?: number
  /** Report the value every this many steps (default 1). */
  readonly reportEvery?: number
}

/**
 * A training run as an objective: the configuration makes a traced `Algorithm` (an optimiser loop, `nn` training,
 * EM), which runs as a child of the trial for `budget` steps (the fidelity a schedule assigns) or `maxSteps`, reporting
 * its value every `reportEvery` steps so pruners can stop it. The objective is the value of the final state.
 *
 * @param make - The training algorithm for a configuration (and its budget).
 * @param options - The value of a state, the direction (default minimise), steps, report interval and a space.
 * @returns The objective.
 * @concept early-stopping
 * @example
 * ```ts
 * import { trainingObjective, tune, medianPruner } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * // Gradient descent on (w - 3)^2 with the learning rate as the configuration.
 * const objective = trainingObjective(
 *   (lr) => ({ name: 'gd', config: {}, init: () => 0, step: (w: number) => w - (lr as number) * 2 * (w - 3) }),
 *   { value: (w: number) => (w - 3) ** 2, maxSteps: 20 },
 * )
 * const result = tune(objective, real(0.01, 0.9, { log: true }), { nTrials: 20, seed: 0, pruner: medianPruner() })
 * (result.best?.values[0] ?? 1) < 1e-6 // => true
 * ```
 */
export function trainingObjective<S>(
  make: (params: SpaceValue, budget: number | undefined) => Algorithm<S>,
  options: TrainingObjectiveOptions<S>,
): Objective {
  const every = options.reportEvery ?? 1
  if (!(Number.isInteger(every) && every >= 1))
    throw new WhiteboxError('value', 'trainingObjective: reportEvery must be a positive integer')
  return {
    ...common(options, 'training', 'minimize'),
    evaluate: (params, trial) => {
      const inner = make(params, trial.budget)
      const wrapped: Algorithm<S> = {
        ...inner,
        done: (state, t) => {
          if (t > 0 && t % every === 0 && trial.report(t, options.value(state))) return 'stopped'
          return inner.done?.(state, t)
        },
      }
      const result = runAlgorithm(wrapped, {
        key: trial.key,
        parent: trial.run,
        maxSteps: trial.budget ?? options.maxSteps ?? 100,
      })
      return options.value(result.state)
    },
  }
}

/** Options of {@link agentObjective}. */
export interface AgentObjectiveOptions extends ObjectiveOptions {
  /** Training episodes when no schedule gives a budget (default 100). */
  readonly episodes?: number
  /** Greedy evaluation episodes scored at the end (default 10). */
  readonly evalEpisodes?: number
  /** Report the mean return of the last `window` training episodes every this many episodes (default 10). */
  readonly reportEvery?: number
  /** Episodes averaged in each report (default 10). */
  readonly window?: number
  /** A step budget for training, guarding against episodes that never end (default 1e6). */
  readonly maxSteps?: number
}

/**
 * An agent in an environment as an objective: the configuration makes an agent, which trains for `budget` (from a
 * schedule) or `episodes` episodes as a child run of the trial, reporting its recent mean return for pruning; the
 * objective is the mean return of the greedy policy over `evalEpisodes` evaluation episodes (maximised).
 *
 * @param env - The environment (with its time limit).
 * @param make - The agent for a configuration.
 * @param options - Episodes, evaluation episodes, reporting and a space.
 * @returns The objective.
 * @concept agents
 * @example
 * ```ts
 * import { agentObjective } from 'whitebox-engine/tune'
 * import { make, policyAgent } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * agentObjective(make(cartPole), () => policyAgent(() => 0)).directions // => ['maximize']
 * ```
 */
export function agentObjective<G, S, O, A>(
  env: Env<S, O, A>,
  make: (params: SpaceValue) => Agent<G, O, A>,
  options: AgentObjectiveOptions = {},
): Objective {
  const every = options.reportEvery ?? 10
  const window = options.window ?? 10
  return {
    ...common(options, 'agent', 'maximize'),
    metricNames: options.metricNames ?? ['meanReturn'],
    evaluate: (params, trial) => {
      const agent = make(params)
      const episodes = trial.budget ?? options.episodes ?? 100
      const inner = trainingAlgorithm(env, agent, { episodes, checkpoints: false })
      let reported = 0
      const wrapped: Algorithm<TrainState<G, S, O>> = {
        ...inner,
        done: (state, t) => {
          if (state.episode > reported && state.episode % every === 0) {
            reported = state.episode
            if (trial.report(state.episode, meanOf(state.returns.slice(-window)))) return 'stopped'
          }
          return inner.done?.(state, t)
        },
      }
      const trained = runAlgorithm(wrapped, {
        key: foldIn(trial.key, 0),
        parent: trial.run,
        maxSteps: options.maxSteps ?? 1_000_000,
      })
      if (trial.shouldPrune()) return Number.NaN
      return evaluatePolicy(env, greedyPolicyOf(agent, trained.state.agent), {
        episodes: options.evalEpisodes ?? 10,
        key: foldIn(trial.key, 1),
        parent: trial.run,
      }).meanReturn
    },
  }
}

/**
 * What fitting a probabilistic program reports for model selection: the log evidence (exact, or estimated), the ELBO
 * of a variational fit, or the expected log predictive density by PSIS-LOO or WAIC.
 */
export interface ProgramFit {
  readonly logEvidence?: number
  readonly elbo?: number
  readonly elpdLoo?: number
  readonly elpdWaic?: number
}

/**
 * A probabilistic program as an objective: the configuration (priors, structure, hyperparameters) is fitted and scored
 * by a model-selection criterion, maximised: log evidence (type-II maximum likelihood), ELBO, or the expected log
 * predictive density by PSIS-LOO or WAIC. The fit function receives the trial context, so a long fit (SVI) can report
 * its ELBO for pruning.
 * TODO(probprog): accept a `Model` and data and call `infer` and `criticism` directly once `probprog` lands.
 *
 * @param fit - Fits the program for a configuration and returns its criteria.
 * @param options - The criterion (default `'logEvidence'`), name and space.
 * @returns The objective.
 * @concept bayesian-model-selection
 * @example
 * ```ts
 * import { programObjective, tune, gridSampler } from 'whitebox-engine/tune'
 * import { dict, choice } from 'whitebox-engine/space'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * // Evidence of y = 1 under y ~ N(0, 1 + s^2) for a prior scale s: best near s = 1.
 * const objective = programObjective((p) => ({ logEvidence: item(normal(0, Math.sqrt(1 + (p as { s: number }).s ** 2)).logProb(1)) }))
 * tune(objective, dict({ s: choice([0.1, 1, 10]) }), { sampler: gridSampler() }).best?.params // => { s: 0.1 }
 * ```
 */
export function programObjective(
  fit: (params: SpaceValue, trial: TrialContext) => ProgramFit,
  options: ObjectiveOptions & { readonly criterion?: keyof ProgramFit } = {},
): Objective {
  const criterion = options.criterion ?? 'logEvidence'
  return {
    ...common(options, `program:${criterion}`, 'maximize'),
    metricNames: options.metricNames ?? [criterion],
    evaluate: (params, trial) => {
      const value = fit(params, trial)[criterion]
      if (value === undefined) throw new WhiteboxError('value', `programObjective: the fit reports no ${criterion}`)
      return value
    },
  }
}
