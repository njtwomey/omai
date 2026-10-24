// TODO: co-training (two feature views) and tri-training (three bootstrapped learners voting).
import { argmaxOf, type Json, WhiteboxError } from '../../lib/base/index.ts'
import { clusteringSteps, type Exposition, learningCurve } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type Estimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  fitRunOptions,
  type LearnData,
  matrixView,
  requireVerb,
  takeData,
} from '../../lib/learn/index.ts'
import { foldIn, type Key } from '../../lib/random/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { kNeighborsClassifier } from '../neighbours/index.ts'
import { semiSupervisedCodes, UNLABELLED } from './labels.ts'

/** Hyperparameters of {@link selfTraining}. */
export interface SelfTrainingHyper {
  /** The base classifier (it must predict probabilities; default 3-nearest neighbours, distance-weighted). */
  readonly estimator: Estimator
  /** Pseudo-label rows whose top probability exceeds a threshold, or the k most confident each round. */
  readonly criterion: 'threshold' | 'kBest'
  /** The confidence threshold (default 0.75). */
  readonly threshold: number
  /** Rows pseudo-labelled per round by `kBest` (default 10). */
  readonly kBest: number
  /** Rounds (default 10). */
  readonly maxIter: number
}

/** Parameters of a fitted self-training classifier: the base model's and the pseudo-labelling history. */
export type SelfTrainingParams = {
  readonly classes: number[]
  /** The base model fitted on the final labelled set, as its serialised parameters. */
  readonly base: Json
  /** The labels the base model was finally fitted on (pseudo-labels included; -1 never labelled). */
  readonly transduction: number[]
  /** The round in which each row was labelled (0 given, -1 never). */
  readonly labeledIter: number[]
  readonly nIter: number
  readonly terminationCondition: 'max_iter' | 'no_change' | 'all_labeled'
}

/** Verbs of a fitted self-training classifier (the base model's). */
export interface SelfTrainingVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted self-training classifier. */
export type FittedSelfTraining = ExplainedFitted<SelfTrainingParams, SelfTrainingVerbs>

interface SelfTrainingState {
  /** Class codes per row (-1 unlabelled). */
  readonly codes: readonly number[]
  readonly labeledIter: readonly number[]
  readonly fitted?: Fitted
  readonly added: number
}

/**
 * Self-training (sklearn.semi_supervised.SelfTrainingClassifier; Yarowsky 1995): fit the base classifier on the
 * labelled rows (targets -1 are unlabelled), give the unlabelled rows it is most confident about its predicted labels,
 * and refit, round by round, until no row qualifies, every row is labelled or the rounds run out. Each round is a
 * traced step whose base fit nests as a child run. Exposition: a `clusteringSteps` record (every row's label after
 * each round, -1 while unlabelled, with the labelled count as the objective), a `learningCurve` of labelled rows and
 * the base model's training accuracy on the given labels per round, and the final decision surface for two features.
 *
 * @param hyper - The base classifier, criterion, threshold, k and rounds.
 * @returns The estimator.
 */
export function selfTraining(
  hyper: Partial<SelfTrainingHyper> = {},
): ExplainedEstimator<SelfTrainingHyper, FittedSelfTraining> {
  const full: SelfTrainingHyper = {
    estimator: hyper.estimator ?? kNeighborsClassifier({ nNeighbors: 3, weights: 'distance' }),
    criterion: hyper.criterion ?? 'threshold',
    threshold: hyper.threshold ?? 0.75,
    kBest: hyper.kBest ?? 10,
    maxIter: hyper.maxIter ?? 10,
  }
  const est: ExplainedEstimator<SelfTrainingHyper, FittedSelfTraining> = defineExplainedEstimator<
    SelfTrainingHyper,
    SelfTrainingParams,
    SelfTrainingVerbs
  >(
    {
      name: 'self-training',
      kind: 'classifier',
      hyperSpace: (h: SelfTrainingHyper): Space =>
        dict({
          criterion: choice(['threshold', 'kBest'], {
            label: 'selection',
            description: 'Add rows above a confidence threshold, or the k most confident each round.',
            default: h.criterion,
          }),
          threshold: real(0.5, 0.999, { default: h.threshold, label: 'confidence threshold' }),
          kBest: int(1, 100, { default: h.kBest, label: 'rows per round (kBest)' }),
          maxIter: int(1, 50, { default: h.maxIter, label: 'rounds' }),
        }),
      fit: (data, h, options) => fitSelfTraining(data, h, options),
      methods: (p: SelfTrainingParams, h: SelfTrainingHyper) => selfTrainingVerbs(h.estimator, p),
    },
    full,
  )
  return est
}

function selfTrainingVerbs(estimator: Estimator, p: SelfTrainingParams): SelfTrainingVerbs {
  const base = estimator.revive(p.base)
  const predict = requireVerb(base, 'predict', 'selfTraining')
  const predictProba = requireVerb(base, 'predictProba', 'selfTraining')
  return { predict, predictProba, classes: p.classes, nFeaturesIn: base.nFeaturesIn ?? 0 }
}

function fitSelfTraining(
  data: LearnData,
  h: SelfTrainingHyper,
  options: FitOptions,
): { params: SelfTrainingParams; exposition: Exposition[] } {
  const m = matrixView(data.x, 'selfTraining')
  const { classes, codes } = semiSupervisedCodes(data.y ?? [], 'selfTraining', m.n)
  if (!(h.threshold > 0 && h.threshold < 1))
    throw new WhiteboxError('value', 'selfTraining: threshold must be in (0, 1)')
  const fitOn = (s: Pick<SelfTrainingState, 'codes'>, key: Key, parent: FitOptions['parent']): Fitted => {
    const rows = s.codes.flatMap((c, i) => (c >= 0 ? [i] : []))
    const part = takeData({ x: data.x }, rows)
    return h.estimator.fit(
      { x: part.x, y: rows.map((i) => classes[s.codes[i] ?? 0] ?? 0) },
      { key, ...(parent === undefined ? {} : { parent }) },
    )
  }
  const given = codes.map((c) => c >= 0)
  const history: { codes: number[]; accuracy: number }[] = []
  const accuracyOnGiven = (f: Fitted): number => {
    const pred = toFlat(requireVerb(f, 'predict', 'selfTraining')(data.x))
    let right = 0
    let total = 0
    for (let i = 0; i < m.n; i++)
      if (given[i]) {
        total++
        if (pred[i] === classes[codes[i] ?? 0]) right++
      }
    return total === 0 ? 0 : right / total
  }
  const algorithm: Algorithm<SelfTrainingState> = {
    name: 'self-training',
    config: { estimator: h.estimator.name, criterion: h.criterion, threshold: h.threshold, kBest: h.kBest },
    schema: channels({
      metrics: {
        labeled: { description: 'rows labelled (given and pseudo)' },
        added: { description: 'rows pseudo-labelled this round' },
      },
    }),
    init: () => ({ codes, labeledIter: codes.map((c) => (c >= 0 ? 0 : -1)), added: 0 }),
    step: (s, ctx) => {
      const fitted = fitOn(s, foldIn(ctx.key, 0), ctx.run)
      if (history.length === 0) history.push({ codes: [...s.codes], accuracy: accuracyOnGiven(fitted) })
      const proba = requireVerb(fitted, 'predictProba', 'selfTraining')(data.x)
      const flat = toFlat(proba)
      const k = proba.shape[1] ?? classes.length
      const fittedClasses = fitted.classes ?? classes
      const candidates = s.codes.flatMap((c, i) => {
        if (c >= 0) return []
        const row = flat.slice(i * k, (i + 1) * k)
        const best = argmaxOf(row)
        return [{ i, p: row[best] ?? 0, code: classes.indexOf(fittedClasses[best] ?? 0) }]
      })
      const chosen =
        h.criterion === 'threshold'
          ? candidates.filter((c) => c.p > h.threshold)
          : [...candidates].sort((a, b) => b.p - a.p || a.i - b.i).slice(0, h.kBest)
      const next = [...s.codes]
      const iter = [...s.labeledIter]
      for (const c of chosen) {
        next[c.i] = c.code
        iter[c.i] = ctx.t + 1
      }
      const refit = chosen.length > 0 ? fitOn({ codes: next }, foldIn(ctx.key, 1), ctx.run) : fitted
      history.push({ codes: next, accuracy: accuracyOnGiven(refit) })
      return { codes: next, labeledIter: iter, fitted: refit, added: chosen.length }
    },
    done: (s, t) => (t > 0 && s.added === 0 ? 'converged' : s.codes.every((c) => c >= 0) ? 'completed' : undefined),
    observe: (s) => ({ metrics: { labeled: s.codes.filter((c) => c >= 0).length, added: s.added } }),
  }
  const run = runAlgorithm(algorithm, { ...fitRunOptions(options, 'selfTraining'), maxSteps: h.maxIter })
  const s = run.state
  const fitted = s.fitted ?? fitOn(s, foldIn(fitRunOptions(options, 'selfTraining').key as Key, 9), options.parent)
  const params: SelfTrainingParams = {
    classes,
    base: fitted.params,
    transduction: s.codes.map((c) => (c >= 0 ? (classes[c] ?? 0) : UNLABELLED)),
    labeledIter: [...s.labeledIter],
    nIter: run.steps,
    terminationCondition: s.codes.every((c) => c >= 0)
      ? 'all_labeled'
      : run.status === 'converged'
        ? 'no_change'
        : 'max_iter',
  }
  const labelled = history.map((r) => r.codes.filter((c) => c >= 0).length)
  const records: Exposition[] = [
    clusteringSteps({
      method: 'self-training',
      assignments: history.map((r) => r.codes),
      objective: labelled,
      marked: {
        initial: (history[0]?.codes ?? []).flatMap((c, i) => (c >= 0 ? [i] : [])),
        added: history.map((r, t) =>
          r.codes.flatMap((c, i) => (c >= 0 && (history[t - 1]?.codes[i] ?? 0) < 0 ? [i] : [])),
        ),
        name: 'pseudo-labelled',
      },
    }),
    learningCurve({
      sizes: history.map((_, t) => t),
      train: labelled.map((l) => l / m.n),
      validation: history.map((r) => r.accuracy),
      resource: 'round',
      labels: { train: 'fraction labelled', validation: 'accuracy on the given labels' },
    }),
  ]
  if (m.d === 2) {
    const verbs = selfTrainingVerbs(h.estimator, params)
    records.push(
      decisionSurfaceOf({ estimator: 'self-training', kind: 'classifier', params, ...verbs }, data.x, {
        featureNames: featureNamesOf(data, m.d),
        classNames: classNamesOf(data, classes),
      }),
    )
  }
  return { params, exposition: records }
}

/** {@link selfTraining} as a class. */
export class SelfTrainingClassifier extends ExplainedEstimatorClass<SelfTrainingHyper, FittedSelfTraining> {
  /** @param hyper - As {@link selfTraining}. */
  constructor(hyper: Partial<SelfTrainingHyper> = {}) {
    super(selfTraining(hyper))
  }
}
