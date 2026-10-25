/**
 * Combining different estimators, as in scikit-learn: voting ensembles (a weighted majority of labels, or an average of
 * probabilities or predictions) and stacking, where a final estimator learns how to combine the members' out-of-fold
 * predictions. Classifier and regressor functions, classes and their exposition.
 */

import { argmaxOf, type Json, MdmlError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type Exposition, ensembleStages } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  type LearnData,
  matrixView,
  requireVerb,
  targetVector,
} from '../../lib/learn/index.ts'
import { type CvOption, crossValPredict } from '../../lib/learn/model-selection/index.ts'
import { foldIn } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression, ridgeCV } from '../linear/index.ts'
import { scoreOf, stagedAverageScores } from './_stages.ts'

/** A named member of a voting or stacking ensemble. */
export type NamedEstimator = readonly [string, Estimator]

/** Hyperparameters of the voting ensembles. */
export interface VotingHyper {
  readonly estimators: readonly NamedEstimator[]
  /** Hard (majority of labels) or soft (average of probabilities) voting, for classifiers. */
  readonly voting: 'hard' | 'soft'
  /** One weight per member (default equal). */
  readonly weights?: readonly number[]
}

/** Parameters of a fitted voting or stacking ensemble: each member's parameters, and the final estimator's. */
export type CombinationParams = {
  readonly members: Json[]
  readonly final: Json | null
  readonly classes: number[]
  readonly nFeatures: number
}

/** Verbs of a fitted voting or stacking ensemble. */
export interface CombinationVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  /** Each member's predictions `[members][m]`. */
  readonly memberPredictions: (x: TensorLike) => number[][]
  readonly classes?: number[]
  readonly nFeaturesIn: number
}

/** A fitted voting or stacking ensemble. */
export type FittedCombination = Fitted<CombinationParams> & CombinationVerbs

function probaOf(f: Fitted, x: TensorLike, classes: readonly number[], soft = true): Float64Array {
  const k = classes.length
  if (soft && f.predictProba !== undefined && f.classes !== undefined) {
    const p = toFlat(f.predictProba(x))
    const own = f.classes
    const m = p.length / own.length
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++)
      for (const [c, label] of own.entries()) out[i * k + classes.indexOf(label)] = p[i * own.length + c] ?? 0
    return out
  }
  const pred = toFlat(requireVerb(f, 'predict', 'voting')(x))
  const out = new Float64Array(pred.length * k)
  for (const [i, v] of pred.entries()) out[i * k + classes.indexOf(v)] = 1
  return out
}

function targetsOf(
  data: LearnData,
  classifier: boolean,
  who: string,
): { readonly y: Float64Array; readonly classes: number[] } {
  const n = matrixView(data.x, who).n
  if (!classifier) return { y: targetVector(data.y ?? [], who, n), classes: [] }
  const e = encodeLabels(data.y ?? [], who, n)
  return { y: Float64Array.from(e.codes), classes: [...e.classes] }
}

function votingEstimator(
  name: string,
  classifier: boolean,
  hyper: VotingHyper,
): ExplainedEstimator<VotingHyper, FittedCombination> {
  return defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      fit: (data: LearnData, h: VotingHyper, options: FitOptions) => {
        if (h.estimators.length === 0) throw new MdmlError('value', `${name}: needs at least one estimator`)
        const { y, classes } = targetsOf(data, classifier, name)
        const key = resolveKey(options, name)
        const fitted = h.estimators.map(([, est], m) => est.fit(data, { key: foldIn(key, m) }))
        const weights = h.estimators.map((_, m) => h.weights?.[m] ?? 1)
        const k = classes.length
        const outputs = fitted.map((f) =>
          classifier
            ? h.voting === 'soft'
              ? probaOf(f, data.x, classes)
              : probaOf(f, data.x, classes, false)
            : Float64Array.from(toFlat(requireVerb(f, 'predict', name)(data.x))),
        )
        const params: CombinationParams = {
          members: fitted.map((f) => f.params),
          final: null,
          classes,
          nFeatures: matrixView(data.x, name).d,
        }
        const exposition: Exposition[] = [
          ensembleStages({
            method: `${name} (${h.estimators.map(([n]) => n).join(', ')})`,
            weights,
            trainScores: stagedAverageScores(outputs, y, k, weights),
            scoreName: classifier ? 'training accuracy' : 'training R²',
            higherIsBetter: true,
          }),
        ]
        return { params, exposition }
      },
      methods: (p: CombinationParams, h: VotingHyper): CombinationVerbs => {
        const fitted = p.members.map((m, i) => (h.estimators[i] as NamedEstimator)[1].revive(m))
        const weights = h.estimators.map((_, m) => h.weights?.[m] ?? 1)
        const total = weights.reduce((a, b) => a + b, 0)
        const combined = (x: TensorLike, soft: boolean): Float64Array => {
          const outs = fitted.map((f) => probaOf(f, x, p.classes, soft))
          const sum = new Float64Array(outs[0]?.length ?? 0)
          for (const [m, o] of outs.entries())
            for (let i = 0; i < sum.length; i++) sum[i] = (sum[i] ?? 0) + ((weights[m] ?? 1) * (o[i] ?? 0)) / total
          return sum
        }
        const memberPredictions = (x: TensorLike): number[][] =>
          fitted.map((f) => toFlat(requireVerb(f, 'predict', name)(x)))
        if (!classifier)
          return {
            predict: (x) => {
              const preds = memberPredictions(x)
              const m = preds[0]?.length ?? 0
              return fromStorage(
                Float64Array.from(
                  { length: m },
                  (_, i) => preds.reduce((a, pr, j) => a + (weights[j] ?? 1) * (pr[i] ?? 0), 0) / total,
                ),
                [m],
                'float64',
              )
            },
            memberPredictions,
            nFeaturesIn: p.nFeatures,
          }
        const k = p.classes.length
        const proba = (x: TensorLike): Tensor => {
          const s = combined(x, true)
          return fromStorage(s, [s.length / k, k], 'float64')
        }
        return {
          predict: (x) => {
            const s = combined(x, h.voting === 'soft')
            const m = s.length / k
            return fromStorage(
              Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(s.subarray(i * k, (i + 1) * k))] ?? 0),
              [m],
              'float64',
            )
          },
          ...(h.voting === 'soft'
            ? { predictProba: proba, predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }) }
            : {}),
          memberPredictions,
          classes: p.classes,
          nFeaturesIn: p.nFeatures,
        }
      },
    },
    hyper,
  )
}

/**
 * A voting classifier (sklearn.ensemble.VotingClassifier): members fitted on all the data, combined by a weighted
 * majority of labels (hard) or average of probabilities (soft). Exposition: the stages (member weights and the
 * training accuracy as members are added).
 *
 * @param hyper - Named members, voting rule (default soft) and weights.
 * @returns The estimator.
 */
export function votingClassifier(
  hyper: Partial<VotingHyper> & Pick<VotingHyper, 'estimators'>,
): ExplainedEstimator<VotingHyper, FittedCombination> {
  return votingEstimator('voting-classifier', true, { voting: 'soft', ...hyper })
}

/**
 * A voting regressor (sklearn.ensemble.VotingRegressor): the weighted average of the members' predictions.
 *
 * @param hyper - Named members and weights.
 * @returns The estimator.
 */
export function votingRegressor(
  hyper: Partial<VotingHyper> & Pick<VotingHyper, 'estimators'>,
): ExplainedEstimator<VotingHyper, FittedCombination> {
  return votingEstimator('voting-regressor', false, { voting: 'soft', ...hyper })
}

/** Hyperparameters of the stacking ensembles. */
export interface StackingHyper {
  readonly estimators: readonly NamedEstimator[]
  /** The model fitted on the members' out-of-fold outputs (default logistic regression or ridge CV). */
  readonly finalEstimator: Estimator
  /** Folds for the out-of-fold outputs (default 5). */
  readonly cv: CvOption
  /** Also pass the original features to the final estimator. */
  readonly passthrough: boolean
}

function stackingEstimator(
  name: string,
  classifier: boolean,
  hyper: StackingHyper,
): ExplainedEstimator<StackingHyper, FittedCombination> {
  const features = (
    members: readonly Fitted[],
    x: TensorLike,
    classes: readonly number[],
    passthrough: boolean,
  ): Tensor => {
    const blocks = members.map((f) => {
      if (!classifier) return { width: 1, data: Float64Array.from(toFlat(requireVerb(f, 'predict', name)(x))) }
      const p = probaOf(f, x, classes)
      // Binary problems keep the positive class's probability only (scikit-learn's stack_method='auto').
      if (classes.length === 2)
        return { width: 1, data: Float64Array.from({ length: p.length / 2 }, (_, i) => p[2 * i + 1] ?? 0) }
      return { width: classes.length, data: p }
    })
    return assemble(blocks, x, passthrough)
  }
  return defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      fit: (data: LearnData, h: StackingHyper, options: FitOptions) => {
        if (h.estimators.length === 0) throw new MdmlError('value', `${name}: needs at least one estimator`)
        const { y, classes } = targetsOf(data, classifier, name)
        const key = resolveKey(options, name)
        const oof = h.estimators.map(([, est], m) => {
          const out = crossValPredict(est, data, {
            cv: h.cv,
            key: foldIn(key, m),
            method: classifier && est.kind === 'classifier' ? 'predictProba' : 'predict',
          })
          return out
        })
        const blocks = oof.map((t) => {
          const flat = Float64Array.from(toFlat(t))
          if (t.shape.length === 1) return { width: 1, data: flat }
          const width = t.shape[1] ?? 1
          if (width === 2)
            return { width: 1, data: Float64Array.from({ length: flat.length / 2 }, (_, i) => flat[2 * i + 1] ?? 0) }
          return { width, data: flat }
        })
        const z = assemble(blocks, data.x, h.passthrough)
        const final = h.finalEstimator.fit(
          { x: z, ...(data.y === undefined ? {} : { y: data.y }) },
          { key: foldIn(key, 99) },
        )
        const members = h.estimators.map(([, est], m) => est.fit(data, { key: foldIn(key, 100 + m) }))
        const k = classes.length
        const memberScores = blocks.map((b) => {
          if (!classifier) return scoreOf(b.data, 1, y, 0)
          const probs =
            b.width === 1
              ? Float64Array.from({ length: b.data.length * 2 }, (_, i) =>
                  i % 2 === 1 ? (b.data[(i - 1) / 2] ?? 0) : 1 - (b.data[i / 2] ?? 0),
                )
              : b.data
          return scoreOf(probs, 1, y, k)
        })
        const params: CombinationParams = {
          members: members.map((f) => f.params),
          final: final.params,
          classes,
          nFeatures: matrixView(data.x, name).d,
        }
        const finalExposition = (final as { readonly exposition?: readonly Exposition[] }).exposition ?? []
        return {
          params,
          exposition: [
            ensembleStages({
              method: `${name} (${h.estimators.map(([n]) => n).join(', ')})`,
              weights: h.estimators.map(() => 1),
              trainScores: memberScores,
              scoreName: classifier ? 'member training accuracy' : 'member training R²',
              higherIsBetter: true,
            }),
            ...finalExposition.filter((r) => r.kind === 'coefficientTable'),
          ],
        }
      },
      methods: (p: CombinationParams, h: StackingHyper): CombinationVerbs => {
        const members = p.members.map((m, i) => (h.estimators[i] as NamedEstimator)[1].revive(m))
        const final = h.finalEstimator.revive(p.final)
        const memberPredictions = (x: TensorLike): number[][] =>
          members.map((f) => toFlat(requireVerb(f, 'predict', name)(x)))
        const predict = (x: TensorLike): Tensor =>
          requireVerb(final, 'predict', name)(features(members, x, p.classes, h.passthrough))
        if (!classifier) return { predict, memberPredictions, nFeaturesIn: p.nFeatures }
        const proba = (x: TensorLike): Tensor =>
          requireVerb(final, 'predictProba', name)(features(members, x, p.classes, h.passthrough))
        return {
          predict,
          predictProba: proba,
          predictDistribution: (x) => categorical({ probs: proba(x) }),
          memberPredictions,
          classes: p.classes,
          nFeaturesIn: p.nFeatures,
        }
      },
    },
    hyper,
  )
}

/** Member outputs side by side (and the original features when passing through), as a matrix. */
function assemble(
  blocks: readonly { readonly width: number; readonly data: Float64Array }[],
  x: TensorLike,
  passthrough: boolean,
): Tensor {
  const v = passthrough ? matrixView(x, 'stacking') : undefined
  const m = blocks[0] === undefined ? 0 : blocks[0].data.length / blocks[0].width
  const width = blocks.reduce((a, b) => a + b.width, 0) + (v?.d ?? 0)
  const out = new Float64Array(m * width)
  for (let i = 0; i < m; i++) {
    let c = 0
    for (const b of blocks) for (let j = 0; j < b.width; j++) out[i * width + c++] = b.data[i * b.width + j] ?? 0
    if (v !== undefined) for (let j = 0; j < v.d; j++) out[i * width + c++] = v.data[i * v.d + j] ?? 0
  }
  return fromStorage(out, [m, width], 'float64')
}

/**
 * A stacking classifier (sklearn.ensemble.StackingClassifier): the final estimator (default logistic regression) is
 * fitted on the members' out-of-fold probabilities (the positive class's only for two classes), the members then
 * refitted on all the data. Exposition: the stages (each member's out-of-fold accuracy) and the final estimator's
 * coefficient table, when it has one.
 *
 * @param hyper - Named members, final estimator, folds (default 5) and passthrough.
 * @returns The estimator.
 */
export function stackingClassifier(
  hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>,
): ExplainedEstimator<StackingHyper, FittedCombination> {
  return stackingEstimator('stacking-classifier', true, {
    finalEstimator: logisticRegression(),
    cv: 5,
    passthrough: false,
    ...hyper,
  })
}

/**
 * A stacking regressor (sklearn.ensemble.StackingRegressor): the final estimator (default ridge with leave-one-out
 * alpha) fitted on the members' out-of-fold predictions; exposition as {@link stackingClassifier} with R^2.
 *
 * @param hyper - Named members, final estimator, folds (default 5) and passthrough.
 * @returns The estimator.
 */
export function stackingRegressor(
  hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>,
): ExplainedEstimator<StackingHyper, FittedCombination> {
  return stackingEstimator('stacking-regressor', false, {
    finalEstimator: ridgeCV({ alphas: [0.1, 1, 10] }),
    cv: 5,
    passthrough: false,
    ...hyper,
  })
}

/** {@link votingClassifier} as a class. */
export class VotingClassifier extends ExplainedEstimatorClass<VotingHyper, FittedCombination> {
  /** @param hyper - As {@link votingClassifier}. */
  constructor(hyper: Partial<VotingHyper> & Pick<VotingHyper, 'estimators'>) {
    super(votingClassifier(hyper))
  }
}

/** {@link stackingClassifier} as a class. */
export class StackingClassifier extends ExplainedEstimatorClass<StackingHyper, FittedCombination> {
  /** @param hyper - As {@link stackingClassifier}. */
  constructor(hyper: Partial<StackingHyper> & Pick<StackingHyper, 'estimators'>) {
    super(stackingClassifier(hyper))
  }
}
