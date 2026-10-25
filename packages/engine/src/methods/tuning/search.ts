/**
 * Hyperparameter tuning by cross-validation: `tuneEstimator` searches an estimator's hyperparameter space with grid,
 * random, TPE or Gaussian-process samplers and optional pruners (the median rule, successive halving), in the manner of
 * scikit-learn's `GridSearchCV` and `RandomizedSearchCV` and Optuna, over the core's `lib/tune`, returning every trial
 * and the best configuration.
 */

import { type Json, meanOf, WhiteboxError } from '../../lib/base/index.ts'
import {
  type CrossValidationCurve,
  crossValidationCurve,
  type Explained,
  type Exposition,
  type HyperparameterSurface,
  hyperparameterSurface,
  learningCurve,
  type SearchTrials,
  searchTrials,
} from '../../lib/exposition/index.ts'
import type { LearnData } from '../../lib/learn/index.ts'
import type { Splitter } from '../../lib/learn/model-selection/index.ts'
import { key as makeKey } from '../../lib/random/index.ts'
import type { Space } from '../../lib/space/index.ts'
import { type RunOptions, type RunStatus, runAlgorithm } from '../../lib/trace/index.ts'
import {
  bestTrial,
  contourData,
  type EstimatorSource,
  estimatorObjective,
  gpSampler,
  gridSampler,
  medianPruner,
  optimisationHistory,
  type Pruner,
  randomSampler,
  type Sampler,
  type ScoringChoice,
  type Study,
  type StudyState,
  studyAlgorithm,
  successiveHalvingPruner,
  type Trial,
  tpeSampler,
} from '../../lib/tune/index.ts'

/** The samplers {@link tuneEstimator} knows by name. */
export type SamplerName = 'grid' | 'random' | 'tpe' | 'gp'

/** The pruners {@link tuneEstimator} knows by name: none, the median rule, or successive halving over the folds. */
export type PrunerName = 'none' | 'median' | 'halving'

/** What {@link TuneEstimatorOptions.onTrial} receives after every trial. */
export interface TuneProgress {
  /** Trials finished. */
  readonly trials: number
  /** The share of the trial budget spent. */
  readonly fraction: number
  readonly study: Study
  /** The exposition of the trials so far. */
  exposition(): Exposition[]
}

/** Options of {@link tuneEstimator}. */
export interface TuneEstimatorOptions
  extends ScoringChoice,
    Partial<Pick<RunOptions, 'key' | 'seed' | 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** The estimator (re-created with each configuration by `with`), or a factory from configurations. */
  readonly estimator: EstimatorSource
  /** The data, scored by cross-validation. */
  readonly data: LearnData
  /** The search space (default the estimator's `hyperSpace`). */
  readonly space?: Space
  /** Folds (default 5, stratified for classifiers) or a splitter. */
  readonly cv?: number | Splitter
  /** The sampler, by name (default `'tpe'`) or given. */
  readonly sampler?: SamplerName | Sampler
  /** The trial budget (default 30; a grid search runs the largest full grid within it). */
  readonly trials?: number
  /** Points per dimension of a grid search (default the most whose full grid fits in `trials`, at least 2). */
  readonly gridPoints?: number
  /** Random trials before TPE or the Gaussian process take over (default a third of the budget, 3 to 10). */
  readonly startupTrials?: number
  /** A pruner by name (default `'none'`) or given; it sees the running mean score after each fold. */
  readonly pruner?: PrunerName | Pruner
  /** The two parameters of the surface (default the first two numeric ones). */
  readonly surface?: readonly [string, string]
  /** Grid points per axis of the surface (default 20). */
  readonly surfaceSize?: number
  /** Called after every trial (for progress bars and live views; it must not change anything). */
  readonly onTrial?: (progress: TuneProgress) => void
}

/** The result of {@link tuneEstimator}. */
export interface TuneEstimatorResult {
  readonly study: Study
  readonly best: Trial | undefined
  /** The best configuration and its mean cross-validated score. */
  readonly bestParams: Json | undefined
  readonly bestScore: number
  readonly higherIsBetter: boolean
  readonly sampler: string
  readonly status: RunStatus
  readonly runId: string
}

function numericParameters(space: Space): string[] {
  if (space.kind !== 'dict') return []
  return Object.entries(space.spaces)
    .filter(([, s]) => s.kind === 'real' || s.kind === 'int')
    .map(([k]) => k)
}

function samplerOf(options: TuneEstimatorOptions, space: Space, trials: number): Sampler {
  const s = options.sampler ?? 'tpe'
  if (typeof s !== 'string') return s
  const startup = options.startupTrials ?? Math.min(10, Math.max(3, Math.round(trials / 3)))
  switch (s) {
    case 'grid': {
      const d = Math.max(1, space.kind === 'dict' ? Object.keys(space.spaces).length : 1)
      return gridSampler({ points: options.gridPoints ?? Math.max(2, Math.floor(trials ** (1 / d) + 1e-9)) })
    }
    case 'random':
      return randomSampler()
    case 'tpe':
      return tpeSampler({ nStartupTrials: startup })
    case 'gp':
      return gpSampler({ nStartupTrials: startup })
    default:
      throw new WhiteboxError('value', `tuneEstimator: unknown sampler '${String(s)}'`)
  }
}

function prunerOf(p: PrunerName | Pruner | undefined): Pruner | undefined {
  if (p === undefined || p === 'none') return undefined
  if (typeof p !== 'string') return p
  if (p === 'median') return medianPruner({ nStartupTrials: 5, nWarmupSteps: 1 })
  if (p === 'halving') return successiveHalvingPruner({ minResource: 1, reductionFactor: 3 })
  throw new WhiteboxError('value', `tuneEstimator: unknown pruner '${String(p)}'`)
}

/** A trial's value as the trials record shows it: its score, a pruned trial's last running mean, else NaN. */
function shownValue(t: Trial): number {
  if (t.state === 'complete') return t.values[0] ?? Number.NaN
  return t.intermediate.at(-1)?.value ?? Number.NaN
}

function trialsRecord(study: Study, sampler: string, higherIsBetter: boolean, scoreName: string): SearchTrials {
  const best = bestTrial(study)
  const parameters = study.space.kind === 'dict' ? Object.keys(study.space.spaces) : ['x']
  return searchTrials({
    method: sampler,
    parameters,
    trials: study.trials.map((t) => ({
      params: (study.space.kind === 'dict' ? t.params : { x: t.params }) as { readonly [name: string]: Json },
      value: shownValue(t),
      intermediate: t.intermediate.map((v) => v.value),
      state: t.state,
    })),
    higherIsBetter,
    ...(best === undefined ? {} : { best: study.trials.indexOf(best) }),
    scoreName,
  })
}

function surfaceRecord(
  study: Study,
  axes: readonly [string, string],
  size: number,
  objective: string,
  higherIsBetter: boolean,
): HyperparameterSurface | undefined {
  if (study.trials.filter((t) => t.state === 'complete').length < 3) return undefined
  const c = contourData(study, axes[0], axes[1], { gridSize: size })
  return hyperparameterSurface({
    parameters: [axes[0], axes[1]],
    x: c.x.ticks,
    y: c.y.ticks,
    values: c.z,
    path: c.points.map((p) => [p.x, p.y] as const),
    logScale: [c.x.log, c.y.log],
    objective,
    higherIsBetter,
  })
}

/** For a one-parameter search: the mean score and its standard error at every value tried. */
function curveRecord(
  study: Study,
  parameter: string,
  higherIsBetter: boolean,
  scoreName: string,
): CrossValidationCurve | undefined {
  const rows = study.trials
    .filter((t) => t.state === 'complete')
    .map((t) => {
      const value = (t.params as Readonly<Record<string, unknown>>)[parameter]
      const folds = t.attrs?.['foldScores']
      return { value, score: t.values[0] ?? Number.NaN, folds: Array.isArray(folds) ? (folds as number[]) : [] }
    })
    .filter((r): r is { value: number; score: number; folds: number[] } => typeof r.value === 'number')
    .sort((a, b) => a.value - b.value)
  if (rows.length < 2) return undefined
  const se = (f: readonly number[]): number => {
    if (f.length < 2) return 0
    const m = meanOf(f)
    let ss = 0
    for (const x of f) ss += (x - m) ** 2
    return Math.sqrt(ss / (f.length - 1) / f.length)
  }
  let best = 0
  for (const [i, r] of rows.entries()) {
    const b = rows[best]?.score ?? Number.NaN
    if (higherIsBetter ? r.score > b : r.score < b) best = i
  }
  const sameFolds = rows.every((r) => r.folds.length === rows[0]?.folds.length && r.folds.length > 0)
  return crossValidationCurve({
    parameter,
    values: rows.map((r) => r.value),
    meanScores: rows.map((r) => r.score),
    standardErrors: rows.map((r) => se(r.folds)),
    ...(sameFolds ? { foldScores: rows.map((r) => r.folds) } : {}),
    higherIsBetter,
    best,
    scoreName,
  })
}

/** Every trial's score beside the best score so far, by trial. */
function historyRecord(study: Study, scoreName: string): Exposition {
  const h = optimisationHistory(study)
  const values = study.trials.map(shownValue)
  return learningCurve({
    sizes: h.number.map((n) => n + 1),
    train: values,
    validation: [...h.best],
    resource: 'trial',
    labels: { train: 'each trial', validation: 'best so far' },
    scoreName,
  })
}

/**
 * Tune an estimator's hyperparameters by cross-validation (scikit-learn's `GridSearchCV` and
 * `RandomizedSearchCV`, Optuna's samplers and pruners) over `lib/tune`: each configuration is scored by its mean
 * score over the folds (`estimatorObjective`; the running mean is reported after each fold, so a pruner can stop a
 * hopeless configuration early), and the sampler (grid, random, TPE or Gaussian-process Bayesian optimisation)
 * proposes the next. Returns the study with its best configuration and, as exposition: every trial with its fold
 * means and fate (`searchTrials`), the score over two numeric parameters as a Gaussian-process surrogate predicts it
 * with the visited points in order (`hyperparameterSurface`, `contourData` of `tune`), the score against the one
 * parameter of a one-parameter search with standard errors (`crossValidationCurve`), and each trial's score beside
 * the best so far (`learningCurve`). The run is `studyAlgorithm` of `tune` through `runAlgorithm`, traced.
 *
 * @param options - The estimator, data, space, folds, scoring, sampler, budget, pruner, surface axes and run options.
 * @returns The study, the best configuration and the exposition records.
 * @throws WhiteboxError with code `value` for an unknown sampler or pruner, or a space with nothing to tune.
 */
export function tuneEstimator(options: TuneEstimatorOptions): Explained<TuneEstimatorResult> {
  const space = options.space ?? (typeof options.estimator === 'function' ? undefined : options.estimator.hyperSpace)
  if (space === undefined) throw new WhiteboxError('value', 'tuneEstimator: give a space (the estimator has none)')
  const trials = options.trials ?? 30
  const sampler = samplerOf(options, space, trials)
  const pruner = prunerOf(options.pruner)
  const objective = estimatorObjective(options.estimator, options.data, {
    space,
    cv: options.cv ?? 5,
    ...(options.metric === undefined ? {} : { metric: options.metric }),
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.response === undefined ? {} : { response: options.response }),
  })
  const higherIsBetter = objective.directions[0] === 'maximize'
  // The score's name: the scoring asked for, else the estimator's default (accuracy or R²).
  const kind = typeof options.estimator === 'function' ? undefined : options.estimator.kind
  const scoreName =
    typeof options.scoring === 'string'
      ? options.scoring
      : options.metric !== undefined || options.scoring !== undefined
        ? 'score'
        : kind === 'classifier'
          ? 'accuracy'
          : kind === 'regressor'
            ? 'R²'
            : 'score'
  const cvName = `cross-validated ${scoreName}`
  const numeric = numericParameters(space)
  const axes: readonly [string, string] | undefined =
    options.surface ?? (numeric.length >= 2 ? [numeric[0] as string, numeric[1] as string] : undefined)
  const size = options.surfaceSize ?? 20
  const explain = (study: Study): Exposition[] => {
    const out: Exposition[] = [trialsRecord(study, sampler.name, higherIsBetter, cvName)]
    const surface = axes === undefined ? undefined : surfaceRecord(study, axes, size, cvName, higherIsBetter)
    if (surface !== undefined) out.push(surface)
    const dims = space.kind === 'dict' ? Object.keys(space.spaces) : []
    if (dims.length === 1 && numeric.length === 1) {
      const curve = curveRecord(study, numeric[0] as string, higherIsBetter, cvName)
      if (curve !== undefined) out.push(curve)
    }
    out.push(historyRecord(study, cvName))
    return out
  }
  const inner = studyAlgorithm(objective, space, {
    sampler,
    ...(pruner === undefined ? {} : { pruner }),
    name: 'tuneEstimator',
  })
  const onTrial = options.onTrial
  const algorithm =
    onTrial === undefined
      ? inner
      : {
          ...inner,
          step: (s: StudyState, ctx: Parameters<typeof inner.step>[1]): StudyState => {
            const next = inner.step(s, ctx)
            const n = next.study.trials.length
            onTrial({
              trials: n,
              fraction: Math.min(1, n / trials),
              study: next.study,
              exposition: () => explain(next.study),
            })
            return next
          },
        }
  const { key, seed, tracer, policy, parent, runId } = options
  const result = runAlgorithm(algorithm, {
    maxSteps: trials,
    key: key ?? makeKey(seed ?? 0),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  })
  const study = result.state.study
  const best = bestTrial(study)
  return {
    study,
    best,
    bestParams: best === undefined ? undefined : (best.params as Json),
    bestScore: best?.values[0] ?? Number.NaN,
    higherIsBetter,
    sampler: sampler.name,
    status: result.status,
    runId: result.runId,
    exposition: explain(study),
  }
}
