/**
 * Bagging (bootstrap aggregation, as scikit-learn's `BaggingClassifier` and `BaggingRegressor`) on the iterative base:
 * any estimator of the library as the member (a decision tree by default), fitted on bootstrap samples of the rows, one
 * member per step, with predictions averaged (probabilities for classifiers, values for regressors). Pasting (samples
 * without replacement) and random subspaces (each member sees a random subset of the features) are modes. The session
 * logs the out-of-bag error as members are added, which is the variance reduction made visible.
 */

import type { Json } from '../base/index.ts'
import { argmaxOf } from '../base/index.ts'
import { categorical, type Distribution, normal } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import {
  type Estimator,
  type Fitted,
  type FittedFacts,
  type LearnData,
  type Verbs,
  withDefaultScore,
} from '../learn/estimator.ts'
import { classNamesOf, decisionSurfaceOf, featureNamesOf } from '../learn/explained.ts'
import { type BalancedBagOptions, balancedBag } from '../learn/preprocessing/resampling.ts'
import { encodeLabels, fromView, matrixView, takeColumns, takeData, targetVector } from '../learn/validation.ts'
import { foldIn, type Key, subsample } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Contract, defineContract, derived, Log } from '../trace/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/decision-tree.ts'
import { memberProba, stagesFromRun } from './_stages.ts'
import {
  type AveragingMember,
  type AveragingPath,
  type AveragingState,
  averagingAlgorithm,
  averagingSources,
} from './averaging.ts'

/** Hyperparameters of the bagging ensembles. */
export interface BaggingHyper {
  /** The member estimator (default a decision tree). */
  readonly estimator: Estimator
  readonly nEstimators: number
  /** Rows per sample, as a fraction (default 1). */
  readonly maxSamples: number
  /** Features per member, as a fraction, or a count when above 1 (random subspaces; default all). */
  readonly maxFeatures: number
  /** Sample with replacement (otherwise every member sees all rows, or a sample without replacement when `maxSamples` < 1). */
  readonly bootstrap: boolean
  /** Balance each member's sample by random under-sampling (as imbalanced-learn's balanced bagging), see {@link balancedBag}. */
  readonly balance?: BalancedBagOptions
}

/** One member of a bagged ensemble: the features it saw and its parameters. */
export type BaggedMember = { readonly features: number[]; readonly params: Json }

/** Parameters of a fitted bagging ensemble. */
export type BaggingParams = {
  readonly members: BaggedMember[]
  readonly classes: number[]
  readonly nFeatures: number
  /** The out-of-bag score (accuracy or R^2) of the whole ensemble, when members were sampled. */
  readonly oobScore: number | null
}

/** Verbs and facts of a fitted bagging ensemble. */
export interface BaggingVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/**
 * The paths bagging logs and declares: the averaging ensemble's and the derived `model/memberSurface`.
 *
 * @example
 * ```ts
 * import type { BaggingPath } from 'omai-engine/ensembles'
 * const path: BaggingPath = 'model/memberSurface'
 * path // => 'model/memberSurface'
 * ```
 */
export type BaggingPath = AveragingPath | 'model/memberSurface'

/** A bagging estimator: the iterative base over the averaging state. */
export type BaggingEstimator = IterativeEstimator<
  BaggingHyper,
  AveragingState<BaggedMember>,
  BaggingParams,
  BaggingVerbs,
  BaggingPath
>

/** The columns of learning data at some feature indices. */
function columnsOf(data: LearnData, features: readonly number[]): LearnData {
  const m = matrixView(data.x, 'bagging', { allowNaN: true })
  return { ...data, x: fromView(takeColumns(m, features)) }
}

/** A member as a fitted value over all the features (it looks at its own subset). */
function memberFitted(estimator: Estimator, member: BaggedMember): Fitted {
  const inner = estimator.revive(member.params)
  const project = (x: TensorLike): Tensor =>
    fromView(takeColumns(matrixView(x, 'bagging', { allowNaN: true }), member.features))
  return {
    ...inner,
    ...(inner.predict === undefined
      ? {}
      : { predict: (x: TensorLike) => (inner.predict as Verbs['predict'] & object)(project(x)) }),
    ...(inner.predictProba === undefined
      ? {}
      : { predictProba: (x: TensorLike) => (inner.predictProba as Verbs['predictProba'] & object)(project(x)) }),
  }
}

/** The mean and standard deviation across members of their outputs. */
function spread(outs: readonly Float64Array[]): { readonly mean: Float64Array; readonly sd: Float64Array } {
  const len = outs[0]?.length ?? 0
  const mean = new Float64Array(len)
  for (const o of outs) for (let i = 0; i < len; i++) mean[i] = (mean[i] ?? 0) + (o[i] ?? 0) / outs.length
  const sd = new Float64Array(len)
  for (const o of outs)
    for (let i = 0; i < len; i++) sd[i] = (sd[i] ?? 0) + ((o[i] ?? 0) - (mean[i] ?? 0)) ** 2 / outs.length
  return { mean, sd: sd.map(Math.sqrt) }
}

/**
 * The bagging estimator under a name: the engine of {@link baggingClassifier}, {@link baggingRegressor} and the
 * balanced ensembles built on it.
 *
 * @param name - The estimator's name.
 * @param classifier - Classification (probabilities averaged) or regression.
 * @param defaults - The hyperparameters in full.
 * @returns The estimator.
 * @example
 * ```ts
 * import { baggingEstimator } from 'omai-engine/ensembles'
 * import { decisionTreeClassifier } from 'omai-engine/trees'
 * const estimator = baggingEstimator('my-bagging', true, {
 *   estimator: decisionTreeClassifier({ record: 0 }),
 *   nEstimators: 3,
 *   maxSamples: 1,
 *   maxFeatures: 1,
 *   bootstrap: true,
 * })
 * estimator.name // => 'my-bagging'
 * ```
 */
export function baggingEstimator(name: string, classifier: boolean, defaults: BaggingHyper): BaggingEstimator {
  const classesOf = (data: LearnData): readonly number[] => (classifier ? encodeLabels(data.y ?? [], name).classes : [])
  const memberOf = (data: LearnData, h: BaggingHyper): AveragingMember<BaggedMember> => {
    const d = matrixView(data.x, name, { allowNaN: true }).d
    const classes = classesOf(data)
    const nf = h.maxFeatures <= 1 ? Math.max(1, Math.round(h.maxFeatures * d)) : Math.min(d, Math.round(h.maxFeatures))
    return {
      fit: (counts, key: Key) => {
        const rows: number[] = []
        counts.forEach((c, i) => {
          for (let r = 0; r < c; r++) rows.push(i)
        })
        const features =
          nf >= d ? Array.from({ length: d }, (_, j) => j) : Array.from(toFlat(subsample(foldIn(key, 2), d, nf)))
        const fitted = h.estimator.fit(columnsOf(takeData(data, rows), features), {
          key: foldIn(key, 3),
          explain: false,
        })
        return { features, params: fitted.params }
      },
      outputs: (m) => {
        const f = memberFitted(h.estimator, m)
        return classifier
          ? memberProba(f, data.x, classes)
          : Float64Array.from(toFlat((f.predict as NonNullable<Verbs['predict']>)(data.x)))
      },
    }
  }
  return iterative<BaggingHyper, AveragingState<BaggedMember>, BaggingParams, BaggingVerbs, BaggingPath>(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, Math.max(200, h.nEstimators), { default: h.nEstimators, label: 'members' }),
          maxSamples: real(0.1, 1, { default: h.maxSamples, label: 'sample fraction' }),
        }),
      algorithm: (data, h) => {
        const y = classifier ? encodeLabels(data.y ?? [], name).codes : targetVector(data.y ?? [], name)
        const labels = Array.from(y, (code) => classesOf(data)[code] ?? code)
        return averagingAlgorithm(
          name,
          { y, k: classesOf(data).length },
          {
            nEstimators: h.nEstimators,
            bootstrap: h.bootstrap,
            maxSamples: h.maxSamples <= 1 ? h.maxSamples * y.length : h.maxSamples,
            ...(h.balance === undefined
              ? {}
              : {
                  balance: (counts, key) => balancedBag(labels, counts, h.balance as BalancedBagOptions, key),
                }),
          },
          memberOf(data, h),
        )
      },
      params: (state, data) => ({
        members: [...state.members],
        classes: [...classesOf(data)],
        nFeatures: matrixView(data.x, name, { allowNaN: true }).d,
        oobScore: Number.isFinite(state.oobScores.at(-1)) ? (state.oobScores.at(-1) as number) : null,
      }),
      verbs: (p, h) => {
        const members = p.members.map((m) => memberFitted(h.estimator, m))
        const k = p.classes.length
        const proba = (x: TensorLike): Tensor => {
          const outs = members.map((f) => memberProba(f, x, p.classes))
          const { mean } = spread(outs)
          return fromStorage(mean, [mean.length / k, k], 'float64')
        }
        const kind = classifier ? 'classifier' : 'regressor'
        if (classifier)
          return withDefaultScore(kind, {
            predict: (x: TensorLike) => {
              const pr = proba(x)
              const m = pr.shape[0] ?? 0
              return fromStorage(
                Float64Array.from(
                  { length: m },
                  (_, i) => p.classes[argmaxOf(pr.data.subarray(i * k, (i + 1) * k))] ?? 0,
                ),
                [m],
                'float64',
              )
            },
            predictProba: proba,
            predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
            classes: p.classes,
            nFeaturesIn: p.nFeatures,
          })
        return withDefaultScore(kind, {
          predict: (x: TensorLike) => {
            const { mean } = spread(
              members.map((f) => Float64Array.from(toFlat((f.predict as NonNullable<Verbs['predict']>)(x)))),
            )
            return fromStorage(mean, [mean.length], 'float64')
          },
          predictDistribution: (x: TensorLike) => {
            const { mean, sd } = spread(
              members.map((f) => Float64Array.from(toFlat((f.predict as NonNullable<Verbs['predict']>)(x)))),
            )
            return normal(
              fromStorage(mean, [mean.length], 'float64'),
              fromStorage(
                sd.map((v) => Math.max(v, 1e-12)),
                [mean.length],
                'float64',
              ),
            )
          },
          nFeaturesIn: p.nFeatures,
        })
      },
      maxSteps: (h) => h.nEstimators,
      contract: (data, h): Contract<AveragingState<BaggedMember>, BaggingPath> => {
        const features = featureNamesOf(data, matrixView(data.x, name, { allowNaN: true }).d)
        const classes = classifier ? classNamesOf(data, classesOf(data)) : undefined
        return defineContract({
          ...averagingSources(classifier),
          'model/memberSurface': derived<AveragingState<BaggedMember>>({
            description: 'the decision surface of one member (args: member, resolution), beside the average’s',
            cost: 'dear',
            compute: (state, args) => {
              const a = (args ?? {}) as { readonly member?: number; readonly resolution?: number }
              const m = state.members[Math.min(a.member ?? 0, state.members.length - 1)] as BaggedMember
              return decisionSurfaceOf(memberFitted(h.estimator, m), data.x, {
                featureNames: features,
                ...(a.resolution === undefined ? {} : { resolution: a.resolution }),
                ...(classes === undefined ? {} : { classNames: classes }),
                ...(data.y === undefined ? {} : { y: data.y }),
              })
            },
          }),
        })
      },
      projections: {
        'ensemble/stages': (run) =>
          stagesFromRun(run, {
            method: name,
            scoreName: classifier ? 'accuracy' : 'R²',
            higherIsBetter: true,
            weight: 'ensemble/weight',
            train: 'ensemble/trainScore',
            valid: 'ensemble/oobScore',
            bootstrap: 'ensemble/bootstrap',
          }) as Exposition | undefined,
      },
      logs: {
        'ensemble/member': Log(),
        'ensemble/weight': Log(),
        'ensemble/trainScore': Log(),
        'ensemble/oobScore': Log(),
        'ensemble/oobError': Log(),
        'ensemble/bootstrap': Log({ detail: 'full' }),
      },
    },
    defaults,
  )
}

function baggingDefaults(hyper: Partial<BaggingHyper>, estimator: Estimator): BaggingHyper {
  return {
    estimator: hyper.estimator ?? estimator,
    nEstimators: hyper.nEstimators ?? 10,
    maxSamples: hyper.maxSamples ?? 1,
    maxFeatures: hyper.maxFeatures ?? 1,
    bootstrap: hyper.bootstrap ?? true,
    ...(hyper.balance === undefined ? {} : { balance: hyper.balance }),
  }
}

/**
 * Bagging for classification (sklearn.ensemble.BaggingClassifier): any classifier fitted on keyed bootstrap samples,
 * probabilities averaged (votes for members without probabilities), one member per step. The session logs
 * `ensemble/member`, `ensemble/oobError`, `ensemble/memberError`, `ensemble/bootstrap` and the training and out-of-bag
 * scores, answers the probe `model/memberSurface`, and its fit records the stages (`ensemble/stages`: training and
 * out-of-bag accuracy as members are added, each member's bootstrap rows).
 *
 * @param hyper - Member estimator (default a decision tree), members (default 10), sample fraction, features per
 *   member (random subspaces) and bootstrap (false with a fraction below 1 is pasting).
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Bagging with out-of-bag bookkeeping.
 * @example
 * ```ts
 * import { baggingClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40, ((i * 7) % 11) / 11])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * const fitted = baggingClassifier({ nEstimators: 5 }).fit({ x, y }, { key: key(0) })
 * fitted.params.members.length // => 5
 * ```
 */
export function baggingClassifier(hyper: Partial<BaggingHyper> = {}): BaggingEstimator {
  return baggingEstimator('bagging-classifier', true, baggingDefaults(hyper, decisionTreeClassifier({ record: 0 })))
}

/**
 * Bagging for regression (sklearn.ensemble.BaggingRegressor): member predictions averaged; `predictDistribution` is
 * normal with the members' mean and spread. Logging as {@link baggingClassifier} with R^2 and mean squared errors.
 *
 * @param hyper - Member estimator (default a regression tree), members, sample fraction, features and bootstrap.
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { baggingRegressor } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 30])
 * baggingRegressor({ nEstimators: 4 }).fit({ x, y: x.map(([a = 0]) => 2 * a) }, { key: key(0) }).params.members.length // => 4
 * ```
 */
export function baggingRegressor(hyper: Partial<BaggingHyper> = {}): BaggingEstimator {
  return baggingEstimator('bagging-regressor', false, baggingDefaults(hyper, decisionTreeRegressor({ record: 0 })))
}
