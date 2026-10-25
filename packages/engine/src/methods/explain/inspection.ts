/**
 * Global inspection of fitted models as exposition records: permutation, conditional, drop-column, impurity and SAGE
 * importances (`importances`), partial dependence with ICE curves and accumulated local effects (`dependence`).
 */

import { MdmlError } from '../../lib/base/index.ts'
import {
  accumulatedLocalEffects,
  accumulatedLocalEffects2D,
  centredIce,
  conditionalPermutationImportance,
  dropColumnImportance,
  iceDerivative,
  partialDependence,
  permutationImportance,
  sage,
} from '../../lib/explain/index.ts'
import {
  type DependenceCurve,
  type DependenceSurface,
  dependence,
  type Explained,
  importances,
} from '../../lib/exposition/index.ts'
import {
  columnOf,
  type Estimator,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  type Scoring,
} from '../../lib/learn/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { modelOutput } from './_models.ts'

/** Options of {@link featureImportance}. */
export interface FeatureImportanceOptions {
  /**
   * `permutation` (default: any model, on the data given), `conditional` (permuted within strata of correlated
   * features) or `impurity` (a tree model's mean impurity decrease).
   */
  readonly method?: 'permutation' | 'conditional' | 'impurity'
  /** The scorer of permutation importance (default the estimator's score: accuracy or R²). */
  readonly scoring?: Scoring
  /** Shuffles per feature (default 5). */
  readonly nRepeats?: number
  /** Which data the importances are measured on, for the record (`training`, `held out`). */
  readonly on?: string
  readonly key?: Key
}

/** Feature importances: one mean per feature and its spread across repeats. */
export interface FeatureImportanceResult {
  readonly mean: readonly number[]
  readonly spread: readonly number[]
  /** The score on the unshuffled data (permutation importance). */
  readonly baseline?: number
}

function metricName(fitted: Fitted, scoring: Scoring): string {
  if (typeof scoring === 'string') return scoring
  if (scoring === undefined) return fitted.kind === 'classifier' ? 'accuracy' : 'R²'
  return 'score'
}

/**
 * How much a fitted model relies on each feature: the drop in its score when the feature's values are shuffled
 * across rows (permutation importance, repeated with the spread recorded) or, for trees and their ensembles, the
 * impurity decrease of the splits on it.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data to measure on (features and targets; held-out data shows what generalises).
 * @param options - method, scoring, nRepeats, on and key.
 * @returns The importances with an `importances` record.
 */
export function featureImportance(
  fitted: Fitted,
  data: LearnData,
  options: FeatureImportanceOptions = {},
): Explained<FeatureImportanceResult> {
  const x = matrixView(data.x, 'featureImportance', { allowNaN: true })
  const features = featureNamesOf(data, x.d)
  const on = options.on === undefined ? {} : { on: options.on }
  if (options.method === 'impurity') {
    const given = (fitted as { readonly featureImportances?: readonly number[] }).featureImportances
    if (given === undefined)
      throw new MdmlError('value', `featureImportance: ${fitted.estimator} has no impurity importances`)
    const mean = [...given]
    const spread = mean.map(() => 0)
    return {
      mean,
      spread,
      exposition: [
        importances({ method: 'impurity', features, mean, metric: 'impurity decrease', direction: 'higher', ...on }),
      ],
    }
  }
  const common = {
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.nRepeats === undefined ? {} : { nRepeats: options.nRepeats }),
    ...(options.key === undefined ? {} : { key: options.key }),
  }
  const r =
    options.method === 'conditional'
      ? conditionalPermutationImportance(fitted, data, common)
      : permutationImportance(fitted, data, common)
  const metric = metricName(fitted, options.scoring)
  return {
    mean: r.mean,
    spread: r.std,
    baseline: r.baseline,
    exposition: [
      importances({
        method: options.method === 'conditional' ? 'conditional permutation' : 'permutation',
        features,
        mean: r.mean,
        spread: r.std,
        repeats: r.importances,
        metric,
        // Scorers are larger-is-better (errors are negated), so a drop is always a loss.
        direction: 'higher',
        baseline: r.baseline,
        ...on,
      }),
    ],
  }
}

/** Options of {@link dropColumnImportances}. */
export interface DropColumnImportanceOptions {
  /** The scorer (default the estimator's score: accuracy or R²). */
  readonly scoring?: Scoring
  /** Held-out data to score the refits on (default the training data). */
  readonly test?: LearnData
  readonly key?: Key
}

/**
 * How much a model needs each feature: refit without it and measure the drop in score (drop-column importance). Where
 * permutation importance asks what the fitted model uses, this asks what the data hold that the other features cannot
 * replace; correlated features share little of it.
 *
 * @param estimator - The unfitted estimator (refit once per feature).
 * @param train - The training data.
 * @param options - scoring, test and key.
 * @returns The importances with an `importances` record.
 */
export function dropColumnImportances(
  estimator: Estimator,
  train: LearnData,
  options: DropColumnImportanceOptions = {},
): Explained<FeatureImportanceResult> {
  const x = matrixView(train.x, 'dropColumnImportances', { allowNaN: true })
  const features = featureNamesOf(train, x.d)
  const r = dropColumnImportance(estimator, train, {
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.test === undefined ? {} : { test: options.test }),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const metric =
    typeof options.scoring === 'string' ? options.scoring : estimator.kind === 'classifier' ? 'accuracy' : 'R²'
  const spread = r.importances.map(() => 0)
  return {
    mean: r.importances,
    spread,
    baseline: r.baseline,
    exposition: [
      importances({
        method: 'drop-column',
        features,
        mean: r.importances,
        metric,
        direction: 'higher',
        baseline: r.baseline,
        on: options.test === undefined ? 'training' : 'held out',
      }),
    ],
  }
}

/** Options of {@link dependencePlot}. */
export interface DependencePlotOptions {
  /** The features (default the first three). */
  readonly features?: readonly number[]
  /** Pairs for two-way partial dependence surfaces. */
  readonly pairs?: readonly (readonly [number, number])[]
  /** What the ICE curves hold: the output, curves centred at the grid's start, or their derivatives (default none). */
  readonly transform?: 'none' | 'centred' | 'derivative'
  /** Grid points per feature (default 30). */
  readonly gridResolution?: number
  /** Rows whose ICE curves are kept (default 100, spread over the data; 0 for none). */
  readonly maxRows?: number
  /** The class explained, for classifiers (default the positive class of a binary problem, else the last). */
  readonly classIndex?: number
}

/** Partial dependence of each feature asked for: its grid and averaged curve. */
export interface DependencePlotResult {
  readonly grids: readonly (readonly number[])[]
  readonly averages: readonly (readonly number[])[]
}

function rugOf(col: Float64Array, max = 300): number[] {
  const out: number[] = []
  const step = Math.max(1, col.length / max)
  for (let t = 0; t < col.length && out.length < max; t += step) out.push(col[Math.floor(t)] ?? 0)
  return out
}

function rowsFor(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i)
  return Array.from({ length: max }, (_, i) => Math.floor((i * n) / max))
}

/**
 * Partial dependence with ICE curves (Friedman 2001; Goldstein et al. 2015; sklearn's PartialDependenceDisplay) for a
 * fitted model: per feature the averaged prediction as the feature is swept over a grid, every (kept) row's own curve,
 * optionally centred (c-ICE) or differentiated (d-ICE), the data's rug, and two-way surfaces for pairs.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data the curves average over.
 * @param options - features, pairs, transform, gridResolution, maxRows and classIndex.
 * @returns The grids and averaged curves with a `dependence` record.
 */
export function dependencePlot(
  fitted: Fitted,
  data: LearnData,
  options: DependencePlotOptions = {},
): Explained<DependencePlotResult> {
  const x = matrixView(data.x, 'dependencePlot', { allowNaN: true })
  const names = featureNamesOf(data, x.d)
  const output = modelOutput(fitted, data, 'auto', options.classIndex)
  const features = options.features ?? Array.from({ length: Math.min(3, x.d) }, (_, j) => j)
  const keep = rowsFor(x.n, options.maxRows ?? 100)
  const resolution = options.gridResolution ?? 30
  const transform = options.transform ?? 'none'
  const pick = (k: number): number => (k === 1 ? 0 : (output.classIndex ?? k - 1))
  const curves: DependenceCurve[] = features.map((f) => {
    const r = partialDependence(fitted, data.x, [f], { gridResolution: resolution })
    const o = pick(r.average.length)
    const grid = [...(r.values[0] ?? [])]
    const average = [...(r.average[o] ?? [])]
    const ice = (r.individual[o] ?? []).filter((_, i) => keep.includes(i))
    const shown =
      transform === 'centred'
        ? centredIce(ice)
        : transform === 'derivative' && grid.length > 1
          ? iceDerivative(grid, ice)
          : ice
    const avg =
      transform === 'centred'
        ? centredIce([average])[0]
        : transform === 'derivative' && grid.length > 1
          ? iceDerivative(grid, [average])[0]
          : average
    return {
      feature: f,
      grid,
      average: avg ?? average,
      ...((options.maxRows ?? 100) > 0 ? { individual: shown } : {}),
      rug: rugOf(columnOf(x, f)),
      ...(grid.length < resolution ? { categorical: grid.length <= 10 } : {}),
    }
  })
  const surfaces: DependenceSurface[] = (options.pairs ?? []).map(([a, b]) => {
    const r = partialDependence(fitted, data.x, [a, b], { gridResolution: Math.min(resolution, 20) })
    const gx = r.values[0] ?? []
    const gy = r.values[1] ?? []
    const avg = r.average[pick(r.average.length)] ?? []
    return {
      features: [a, b] as const,
      gridX: [...gx],
      gridY: [...gy],
      values: gx.map((_, i) => gy.map((_, j) => avg[i * gy.length + j] ?? 0)),
    }
  })
  return {
    grids: curves.map((c) => c.grid),
    averages: curves.map((c) => c.average),
    exposition: [
      dependence({
        method: 'partial dependence',
        features: names,
        output: output.name,
        curves,
        ...(surfaces.length === 0 ? {} : { surfaces }),
        transform,
        ...((options.maxRows ?? 100) > 0 ? { indices: keep } : {}),
      }),
    ],
  }
}

/** Options of {@link aleEffects}. */
export interface AleEffectsOptions {
  /** The features (default the first three). */
  readonly features?: readonly number[]
  /** Pairs for second-order effects (pure interactions). */
  readonly pairs?: readonly (readonly [number, number])[]
  /** Quantile bins per feature (default 10). */
  readonly bins?: number
  readonly classIndex?: number
}

/** The first-order ALE of each feature: its bin edges and centred effects. */
export interface AleEffectsResult {
  readonly edges: readonly (readonly number[])[]
  readonly effects: readonly (readonly number[])[]
}

/**
 * Accumulated local effects (Apley and Zhu 2020; alibi's ALE, R's ALEPlot) for a fitted model: per feature the effect
 * of moving it within its quantile bins, accumulated and centred, which stays faithful when features are correlated
 * (partial dependence then averages over impossible rows); for pairs, the second-order effect left once both features'
 * own effects are removed.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data.
 * @param options - features, pairs, bins and classIndex.
 * @returns The edges and effects with a `dependence` record.
 */
export function aleEffects(
  fitted: Fitted,
  data: LearnData,
  options: AleEffectsOptions = {},
): Explained<AleEffectsResult> {
  const x = matrixView(data.x, 'aleEffects')
  const names = featureNamesOf(data, x.d)
  const output = modelOutput(fitted, data, 'auto', options.classIndex)
  const features = options.features ?? Array.from({ length: Math.min(3, x.d) }, (_, j) => j)
  const bins = options.bins ?? 10
  const pick = (k: number): number => (k === 1 ? 0 : (output.classIndex ?? k - 1))
  const curves: DependenceCurve[] = features.map((f) => {
    const r = accumulatedLocalEffects(fitted, data.x, f, { bins })
    return {
      feature: f,
      grid: [...r.edges],
      average: [...(r.effects[pick(r.effects.length)] ?? [])],
      counts: [...r.counts],
      rug: rugOf(columnOf(x, f)),
    }
  })
  const surfaces: DependenceSurface[] = (options.pairs ?? []).map(([a, b]) => {
    const r = accumulatedLocalEffects2D(fitted, data.x, [a, b], { bins: Math.min(bins, 8) })
    return {
      features: [a, b] as const,
      gridX: [...r.edges[0]],
      gridY: [...r.edges[1]],
      values: r.effects,
      counts: r.counts,
    }
  })
  return {
    edges: curves.map((c) => c.grid),
    effects: curves.map((c) => c.average),
    exposition: [
      dependence({
        method: 'ale',
        features: names,
        output: output.name,
        curves,
        ...(surfaces.length === 0 ? {} : { surfaces }),
      }),
    ],
  }
}

/** Options of {@link sageImportance}. */
export interface SageImportanceOptions {
  /** Background rows for imputation, drawn from the data (default 30). */
  readonly background?: number
  /** Permutations sampled (default 200). */
  readonly nPermutations?: number
  /** Evaluation rows used, the first of the data (default 300). */
  readonly maxRows?: number
  readonly key?: Key
}

/**
 * SAGE global importance (Covert, Lundberg and Lee 2020) for a fitted model: each feature's Shapley share of the drop
 * in loss (squared error for regressors, cross-entropy of the class probabilities for classifiers) from predicting
 * with no features to predicting with all, missing features imputed from background rows. The record's repeats are
 * absent; the spread is the sampling standard error.
 *
 * @param fitted - A fitted estimator.
 * @param data - Evaluation data with targets.
 * @param options - background, nPermutations, maxRows and key.
 * @returns The values with an `importances` record.
 */
export function sageImportance(
  fitted: Fitted,
  data: LearnData,
  options: SageImportanceOptions = {},
): Explained<FeatureImportanceResult> {
  const x = matrixView(data.x, 'sageImportance')
  const features = featureNamesOf(data, x.d)
  const n = Math.min(x.n, options.maxRows ?? 300)
  const rows = Array.from({ length: n }, (_, i) => Array.from(x.data.subarray(i * x.d, (i + 1) * x.d)))
  const yAll = toFlat(asTensor(data.y as TensorLike))
  const classify = fitted.kind === 'classifier'
  const classes = (fitted as { readonly classes?: readonly number[] }).classes ?? []
  const y = yAll.slice(0, n).map((v) => (classify ? Math.max(0, classes.indexOf(v)) : v))
  const predict = (t: Tensor): Tensor => {
    const f = classify ? fitted.predictProba : fitted.predict
    if (f === undefined) throw new MdmlError('value', `sageImportance: ${fitted.estimator} cannot predict`)
    return f(t)
  }
  const bgSize = Math.min(x.n, options.background ?? 30)
  const step = x.n / bgSize
  const background = Array.from({ length: bgSize }, (_, b) => {
    const i = Math.floor(b * step)
    return Array.from(x.data.subarray(i * x.d, (i + 1) * x.d))
  })
  const r = sage(predict, { x: rows, y }, background, {
    loss: classify ? 'cross-entropy' : 'mse',
    nPermutations: options.nPermutations ?? 200,
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const spread = r.standardErrors.map((v) => (Number.isFinite(v) ? v : 0))
  return {
    mean: r.values,
    spread,
    baseline: r.lossFull,
    exposition: [
      importances({
        method: 'SAGE',
        features,
        mean: r.values,
        spread,
        metric: classify ? 'cross-entropy removed' : 'squared error removed',
        direction: 'higher',
        baseline: r.lossEmpty,
      }),
    ],
  }
}
