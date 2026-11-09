/**
 * Choosing a covariance estimator by held-out Gaussian log likelihood (the rule of MNE's `auto` noise covariance and of
 * scikit-learn's `GridSearchCV` over covariance estimators): every candidate is fitted on the training folds of the
 * data and scored on the held-out fold, the candidate with the best mean score wins and is refitted on all the data.
 * The default candidates are the empirical covariance, the linear shrinkages to each target, the analytical nonlinear
 * shrinkage and the spectral cleaning of the correlation matrix. The selection is a stepped algorithm, one candidate per
 * step, and an estimator on the covariance pattern.
 */

import { OmaiError } from '../base/index.ts'
import { type Exposition, importances } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { dict, int } from '../space/index.ts'
import { empiricalCovariance, ledoitWolf, oas } from '../stats/index.ts'
import { type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { gaussianLogLikelihood, moments, type Rows } from './_matrix.ts'
import { quadraticInverseShrinkage } from './nonlinear-shrinkage.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'
import { type ShrinkageTarget, targetShrinkage } from './shrinkage-targets.ts'
import { spectralCleaningAlgorithm } from './spectral.ts'

/** A covariance estimator under comparison: a name and how to fit a location and a covariance to rows. */
export interface CovarianceCandidate {
  readonly name: string
  readonly fit: (x: TensorLike) => { readonly location: number[]; readonly covariance: number[][] }
}

/**
 * The mean Gaussian log likelihood of held-out rows under a fitted location and covariance (the score of the selection),
 * `-Infinity` for a singular covariance.
 *
 * @param x - Held-out rows `[n, d]`.
 * @param location - The fitted mean `[d]`.
 * @param covariance - The fitted covariance `[d, d]`.
 * @returns The mean log likelihood per row.
 * @index function Mean Gaussian log likelihood of held-out rows under a location and covariance.
 * @example
 * ```ts
 * import { heldOutLogLikelihood } from 'omai-engine/covariance'
 * heldOutLogLikelihood([[0, 0]], [0, 0], [[1, 0], [0, 1]]) // => -1.8378770664093453
 * ```
 */
export function heldOutLogLikelihood(
  x: TensorLike,
  location: readonly number[],
  covariance: readonly (readonly number[])[],
): number {
  return gaussianLogLikelihood(x, location, covariance)
}

const fromMoments = (x: TensorLike): { location: number[]; covariance: Rows } => {
  const m = moments(x, 'covarianceCandidate')
  return { location: m.location, covariance: m.covariance }
}

const shrunk = (name: string, target: ShrinkageTarget): CovarianceCandidate => ({
  name,
  fit: (x) => {
    const r = targetShrinkage(x, { target })
    return { location: r.location, covariance: r.covariance }
  },
})

/**
 * The default candidates of the selection: `empirical`, `ledoit-wolf`, `oas`, `diagonal`, `constant-correlation`,
 * `single-factor` (the last three the linear shrinkages of `shrinkage-targets` with their analytic intensities),
 * `nonlinear` (analytical nonlinear shrinkage) and `denoised` (the correlation matrix cleaned by the Marchenko-Pastur
 * edge).
 *
 * @returns The candidates, in that order.
 * @index table Default covariance candidates compared by held-out likelihood.
 * @example
 * ```ts
 * import { defaultCovarianceCandidates } from 'omai-engine/covariance'
 * defaultCovarianceCandidates().map((c) => c.name).slice(0, 3) // => ['empirical', 'ledoit-wolf', 'oas']
 * ```
 */
export function defaultCovarianceCandidates(): CovarianceCandidate[] {
  return [
    {
      name: 'empirical',
      fit: (x) => ({ location: fromMoments(x).location, covariance: toArray(empiricalCovariance(x)) as number[][] }),
    },
    {
      name: 'ledoit-wolf',
      fit: (x) => ({ location: fromMoments(x).location, covariance: toArray(ledoitWolf(x).covariance) as number[][] }),
    },
    {
      name: 'oas',
      fit: (x) => ({ location: fromMoments(x).location, covariance: toArray(oas(x).covariance) as number[][] }),
    },
    shrunk('diagonal', 'diagonal'),
    shrunk('constant-correlation', 'constant-correlation'),
    shrunk('single-factor', 'single-factor'),
    {
      name: 'nonlinear',
      fit: (x) => {
        const r = quadraticInverseShrinkage(x)
        return { location: r.location, covariance: r.covariance }
      },
    },
    {
      name: 'denoised',
      fit: (x) => {
        const s = runAlgorithm(spectralCleaningAlgorithm(x, { margin: 0.1 }), { maxSteps: 2 }).state
        return { location: s.location, covariance: s.covariance }
      },
    },
  ]
}

/** Options of {@link selectCovariance}. */
export interface SelectCovarianceOptions {
  /** The number of cross-validation folds (default 5, at most the number of rows). */
  readonly folds?: number
  /** A key that shuffles the rows before dealing them to folds (default none: rows are dealt in turn). */
  readonly key?: Key
}

/** The cross-validated score of one candidate. */
export interface CandidateScore {
  readonly name: string
  /** The held-out log likelihood of each fold (`-Infinity` for a singular fit, `NaN` when the fit failed). */
  readonly folds: number[]
  /** The mean over folds, weighted by their rows: `-Infinity` if any fold is singular, `NaN` if any fit failed. */
  readonly mean: number
  /** The standard error of the fold scores (0 when they are not all finite). */
  readonly standardError: number
}

/** The state of {@link selectCovarianceAlgorithm}: the candidates scored so far and, at the end, the refit of the best. */
export interface SelectionState {
  /** How many candidates have been scored. */
  readonly scored: number
  readonly scores: CandidateScore[]
  /** The best candidate so far (`''` before any scored finitely). */
  readonly best: string
  /** The best candidate refitted on all the data, once the scores are in. */
  readonly location: number[]
  readonly covariance: number[][]
  readonly done: boolean
}

function dealFolds(n: number, folds: number, key: Key | undefined): number[] {
  const order = key === undefined ? Array.from({ length: n }, (_, i) => i) : Array.from(permutation(key, n).data)
  const fold = new Array<number>(n).fill(0)
  order.forEach((row, rank) => {
    fold[row] = rank % folds
  })
  return fold
}

function scoreCandidate(
  c: CovarianceCandidate,
  rows: readonly (readonly number[])[],
  fold: readonly number[],
  folds: number,
): CandidateScore {
  const scores: number[] = []
  const weights: number[] = []
  for (let k = 0; k < folds; k++) {
    const train = rows.filter((_, i) => fold[i] !== k)
    const test = rows.filter((_, i) => fold[i] === k)
    if (test.length === 0) continue
    weights.push(test.length)
    try {
      const f = c.fit(train as number[][])
      scores.push(gaussianLogLikelihood(test as number[][], f.location, f.covariance))
    } catch {
      scores.push(Number.NaN)
    }
  }
  const total = weights.reduce((a, v) => a + v, 0)
  const mean = scores.reduce((a, v, i) => a + v * ((weights[i] ?? 0) / total), 0)
  const finite = scores.every((v) => Number.isFinite(v))
  const m = scores.length
  const mu = finite ? scores.reduce((a, v) => a + v, 0) / m : 0
  const se = finite && m > 1 ? Math.sqrt(scores.reduce((a, v) => a + (v - mu) ** 2, 0) / (m - 1) / m) : 0
  return { name: c.name, folds: scores, mean, standardError: se }
}

/**
 * The selection of a covariance estimator as a stepped {@link Algorithm}: each step cross-validates one candidate (fits it
 * on the training folds, scores the held-out fold by Gaussian log likelihood), recording its mean score and the best so
 * far; the last step refits the best candidate on all the data.
 *
 * @param x - Observations `[n, d]`.
 * @param candidates - The estimators to compare (default {@link defaultCovarianceCandidates}).
 * @param options - The number of folds and an optional shuffling key.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { selectCovarianceAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1], [2, 3], [3, 1]]
 * runAlgorithm(selectCovarianceAlgorithm(x, undefined, { folds: 2 }), { maxSteps: 20 }).state.scores.length // => 8
 * ```
 */
export function selectCovarianceAlgorithm(
  x: TensorLike,
  candidates: readonly CovarianceCandidate[] = defaultCovarianceCandidates(),
  options: SelectCovarianceOptions = {},
): Algorithm<SelectionState> {
  const m = moments(x, 'selectCovariance', { minSamples: 4 })
  if (candidates.length === 0) throw new OmaiError('value', 'selectCovariance: needs at least one candidate')
  const folds = Math.min(options.folds ?? 5, m.n)
  if (!(folds >= 2)) throw new OmaiError('value', 'selectCovariance: needs at least 2 folds')
  const rows = Array.from({ length: m.n }, (_, t) =>
    Array.from({ length: m.d }, (_, j) => (m.centred[t * m.d + j] ?? 0) + (m.location[j] ?? 0)),
  )
  const fold = dealFolds(m.n, folds, options.key)
  const bestOf = (scores: readonly CandidateScore[]): string => {
    let best = ''
    let top = Number.NEGATIVE_INFINITY
    for (const s of scores)
      if (Number.isFinite(s.mean) && s.mean > top) {
        top = s.mean
        best = s.name
      }
    return best
  }
  return {
    name: 'covariance-selection',
    config: { folds, candidates: candidates.map((c) => c.name) },
    schema: channels({
      metrics: {
        score: {
          description: 'the mean held-out log likelihood of the candidate scored in this step',
          direction: 'higher',
        },
        best: { description: 'the best mean held-out log likelihood so far', direction: 'higher' },
      },
      tensors: {
        covariance: {
          role: 'param',
          description: 'the selected covariance, refitted on all the data',
          shape: ['d', 'd'],
        },
      },
    }),
    init: () => ({ scored: 0, scores: [], best: '', location: [], covariance: [], done: false }),
    step: (s) => {
      if (s.scored < candidates.length) {
        const c = candidates[s.scored] as CovarianceCandidate
        const scores = [...s.scores, scoreCandidate(c, rows, fold, folds)]
        return { ...s, scored: s.scored + 1, scores, best: bestOf(scores) }
      }
      const winner = candidates.find((c) => c.name === s.best) ?? (candidates[0] as CovarianceCandidate)
      const f = winner.fit(rows)
      return { ...s, best: winner.name, location: f.location, covariance: f.covariance, done: true }
    },
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => {
      const last = s.scores[s.scores.length - 1]
      const top = s.scores.find((c) => c.name === s.best)
      return {
        metrics: {
          ...(last !== undefined && Number.isFinite(last.mean) ? { score: last.mean } : {}),
          ...(top !== undefined && Number.isFinite(top.mean) ? { best: top.mean } : {}),
        },
        tensors:
          s.covariance.length === 0
            ? {}
            : { covariance: { role: 'param', value: s.covariance.flat(), shape: [m.d, m.d] } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SelectionState,
  }
}

/** The result of {@link selectCovariance}. */
export interface CovarianceSelection {
  /** The name of the candidate with the best mean held-out log likelihood. */
  readonly best: string
  /** Every candidate's cross-validated score, in the order given. */
  readonly scores: CandidateScore[]
  /** The winner refitted on all the data. */
  readonly location: number[]
  readonly covariance: number[][]
}

/**
 * Select a covariance estimator by cross-validated held-out Gaussian log likelihood: each candidate is fitted on the
 * training folds and scored on the held-out fold; the best mean score wins and is refitted on all the data.
 * {@link selectCovarianceAlgorithm} run to its end.
 *
 * @param x - Observations `[n, d]` (at least 4 rows).
 * @param candidates - The estimators to compare (default {@link defaultCovarianceCandidates}).
 * @param options - The number of folds (default 5) and an optional shuffling key.
 * @returns The winner's name, every candidate's scores and the refitted winner.
 * @throws OmaiError with code `value` for no candidates or fewer than 2 folds.
 * @index function Choose a covariance estimator by cross-validated held-out Gaussian log likelihood.
 * @example
 * ```ts
 * import { selectCovariance } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1], [2, 3], [3, 1]]
 * selectCovariance(x, undefined, { folds: 2 }).scores.length // => 8
 * ```
 */
export function selectCovariance(
  x: TensorLike,
  candidates: readonly CovarianceCandidate[] = defaultCovarianceCandidates(),
  options: SelectCovarianceOptions = {},
): CovarianceSelection {
  const out = runAlgorithm(selectCovarianceAlgorithm(x, candidates, options), { maxSteps: candidates.length + 2 })
  return {
    best: out.state.best,
    scores: out.state.scores,
    location: out.state.location,
    covariance: out.state.covariance,
  }
}

/** Hyperparameters of {@link covarianceSelectionEstimator}. */
export interface CovarianceSelectionHyper {
  /** The number of cross-validation folds (default 5). */
  readonly folds?: number
}

/** The fitted parameters of the selection estimator: the winner's covariance and the scores. */
export type CovarianceSelectionParams = CovarianceParams & {
  /** The name of the selected candidate. */
  readonly selected: string
  /** The candidates' names. */
  readonly candidates: string[]
  /** Each candidate's mean cross-validated log likelihood (`null` for a singular or failed fit). */
  readonly scores: (number | null)[]
}

function selectionRecord(s: SelectionState): Exposition {
  const usable = s.scores.filter((c) => Number.isFinite(c.mean))
  return importances({
    method: 'cross-validation',
    features: usable.map((c) => c.name),
    mean: usable.map((c) => c.mean),
    spread: usable.map((c) => c.standardError),
    repeats: usable.map((c) => c.folds),
    metric: 'held-out log likelihood',
    direction: 'higher',
    on: 'held out',
  })
}

/**
 * The selection of a covariance estimator by cross-validated held-out likelihood as a covariance estimator on the
 * iterative base: it scores the default candidates (one per step), refits the best on all the data and exposes it as
 * location, covariance, precision (pseudo-inverse), `mahalanobis` and `score`. The exposition is the candidates' held-out
 * log likelihoods with their standard errors, and the Mahalanobis anomaly scores.
 *
 * @param hyper - The number of folds.
 * @returns The iterative estimator.
 * @index estimator Covariance estimator chosen by cross-validated held-out likelihood.
 * @example
 * ```ts
 * import { covarianceSelectionEstimator } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1], [2, 3], [3, 1]]
 * covarianceSelectionEstimator({ folds: 2 }).fit({ x }).params.candidates.length // => 8
 * ```
 */
export function covarianceSelectionEstimator(
  hyper: CovarianceSelectionHyper = {},
): IterativeEstimator<Required<CovarianceSelectionHyper>, SelectionState, CovarianceSelectionParams, CovarianceVerbs> {
  return iterative<Required<CovarianceSelectionHyper>, SelectionState, CovarianceSelectionParams, CovarianceVerbs>({
    name: 'covariance-selection',
    kind: 'density',
    defaults: { folds: hyper.folds ?? 5 },
    hyperSpace: (h) =>
      dict({
        folds: int(2, 20, { label: 'folds', description: 'The number of cross-validation folds.', default: h.folds }),
      }),
    algorithm: (data: LearnData, h) => selectCovarianceAlgorithm(data.x, undefined, { folds: h.folds }),
    maxSteps: () => defaultCovarianceCandidates().length + 2,
    params: (s) => ({
      location: s.location,
      covariance: s.covariance,
      precision: toArray(linalg.pinv(s.covariance)) as number[][],
      shrinkage: 0,
      selected: s.best,
      candidates: s.scores.map((c) => c.name),
      scores: s.scores.map((c) => (Number.isFinite(c.mean) ? c.mean : null)),
    }),
    verbs: covarianceVerbs,
    records: (s, data) => [
      selectionRecord(s),
      mahalanobisRecord('covarianceSelection', data.x, {
        location: s.location,
        covariance: s.covariance,
        precision: toArray(linalg.pinv(s.covariance)) as number[][],
        shrinkage: 0,
      }),
    ],
    logs: { 'solver/score': Log(), 'solver/best': Log() },
  })
}

/**
 * The covariance selection estimator as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { CovarianceSelectionEstimator } from 'omai-engine/covariance'
 * new CovarianceSelectionEstimator({ folds: 2 }).fit([[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1], [2, 3], [3, 1]]).model.params.location.length // => 2
 * ```
 */
export class CovarianceSelectionEstimator extends ExplainedEstimatorClass<
  Required<CovarianceSelectionHyper>,
  Fitted<CovarianceSelectionParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted covariance selection model.
   *
   * @param hyper - The hyperparameters of {@link covarianceSelectionEstimator}.
   */
  constructor(hyper: CovarianceSelectionHyper = {}) {
    super(covarianceSelectionEstimator(hyper))
  }
}
