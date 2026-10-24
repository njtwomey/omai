import { argmaxOf, type Json } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { ensembleStages } from '../../lib/exposition/index.ts'
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
  takeData,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, randint } from '../../lib/random/index.ts'
import { dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/index.ts'
import { stagedAverageScores, stagedOobScores } from './_stages.ts'

/** Hyperparameters of the bagging ensembles. */
export interface BaggingHyper {
  /** The member estimator (default a decision tree). */
  readonly estimator: Estimator
  readonly nEstimators: number
  /** Rows per bootstrap, as a fraction (default 1). */
  readonly maxSamples: number
  readonly bootstrap: boolean
}

/** Parameters of a fitted bagging ensemble: each member's parameters (revived with the member estimator). */
export type BaggingParams = {
  readonly members: Json[]
  readonly classes: number[]
  readonly nFeatures: number
  readonly oobScore: number | null
}

/** Verbs of a fitted bagging ensemble. */
export interface BaggingVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly classes?: number[]
  readonly nFeaturesIn: number
}

/** A fitted bagging ensemble. */
export type FittedBagging = Fitted<BaggingParams> & BaggingVerbs

/** A member's class probabilities `[m * k]` (one-hot of its predictions when it has no `predictProba`). */
function memberProba(f: Fitted, x: TensorLike, classes: readonly number[]): Float64Array {
  const k = classes.length
  if (f.predictProba !== undefined && f.classes !== undefined) {
    const p = toFlat(f.predictProba(x))
    const own = f.classes
    const m = p.length / own.length
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++)
      for (const [c, label] of own.entries()) out[i * k + classes.indexOf(label)] = p[i * own.length + c] ?? 0
    return out
  }
  const pred = toFlat(requireVerb(f, 'predict', 'bagging')(x))
  const out = new Float64Array(pred.length * k)
  pred.forEach((v, i) => {
    out[i * k + classes.indexOf(v)] = 1
  })
  return out
}

function baggingEstimator(
  name: string,
  classifier: boolean,
  hyper: BaggingHyper,
): ExplainedEstimator<BaggingHyper, FittedBagging> {
  return defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: BaggingHyper): Space =>
        dict({
          nEstimators: int(1, 200, { default: h.nEstimators, label: 'members' }),
          maxSamples: real(0.1, 1, { default: h.maxSamples, label: 'bootstrap fraction' }),
        }),
      update: (h, changes) => {
        const own = Object.fromEntries(Object.entries(changes).filter(([k]) => !k.startsWith('estimator.')))
        const inner = Object.fromEntries(
          Object.entries(changes).flatMap(([k, v]) => (k.startsWith('estimator.') ? [[k.slice(10), v]] : [])),
        )
        return {
          ...h,
          ...own,
          estimator: Object.keys(inner).length > 0 ? h.estimator.with(inner) : h.estimator,
        } as BaggingHyper
      },
      fit: (data: LearnData, h: BaggingHyper, options: FitOptions) => {
        const n = matrixView(data.x, name).n
        const d = matrixView(data.x, name).d
        const classes = classifier ? [...encodeLabels(data.y ?? [], name, n).classes] : []
        const y = classifier ? encodeLabels(data.y ?? [], name, n).codes : targetVector(data.y ?? [], name, n)
        const key = resolveKey(options, name)
        const size = Math.max(1, Math.round(h.maxSamples * n))
        const members = Array.from({ length: h.nEstimators }, (_, m) => {
          const rows = h.bootstrap
            ? Array.from(toFlat(randint(foldIn(key, m), [size], 0, n)))
            : Array.from({ length: n }, (_, i) => i)
          const fitted = h.estimator.fit(takeData(data, rows), { key: foldIn(foldIn(key, m), 1) })
          const inBag = new Uint8Array(n)
          for (const r of rows) inBag[r] = 1
          return { fitted, rows, inBag }
        })
        const outputs = members.map((m) =>
          classifier
            ? memberProba(m.fitted, data.x, classes)
            : Float64Array.from(toFlat(requireVerb(m.fitted, 'predict', 'bagging')(data.x))),
        )
        const k = classes.length
        const trainScores = stagedAverageScores(outputs, y, k)
        const oob = h.bootstrap
          ? stagedOobScores(
              outputs,
              members.map((m) => m.inBag),
              y,
              k,
            )
          : undefined
        const params: BaggingParams = {
          members: members.map((m) => m.fitted.params),
          classes,
          nFeatures: d,
          oobScore: oob === undefined ? null : (oob[oob.length - 1] ?? null),
        }
        return {
          params,
          exposition: [
            ensembleStages({
              method: name,
              weights: members.map(() => 1),
              trainScores,
              ...(oob === undefined ? {} : { validationScores: oob }),
              bootstraps: members.map((m) => m.rows),
              scoreName: classifier ? 'accuracy' : 'R²',
              higherIsBetter: true,
            }),
          ],
        }
      },
      methods: (p: BaggingParams, h: BaggingHyper): BaggingVerbs => {
        const fitted = p.members.map((m) => h.estimator.revive(m))
        const proba = (x: TensorLike): Tensor => {
          const outs = fitted.map((f) => memberProba(f, x, p.classes))
          const len = outs[0]?.length ?? 0
          const mean = new Float64Array(len)
          for (const o of outs) for (let i = 0; i < len; i++) mean[i] = (mean[i] ?? 0) + (o[i] ?? 0) / outs.length
          return fromStorage(mean, [len / p.classes.length, p.classes.length], 'float64')
        }
        if (classifier)
          return {
            predict: (x) => {
              const pr = proba(x)
              const k = p.classes.length
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
            predictDistribution: (x) => categorical({ probs: proba(x) }),
            classes: p.classes,
            nFeaturesIn: p.nFeatures,
          }
        return {
          predict: (x) => {
            const outs = fitted.map((f) => toFlat(requireVerb(f, 'predict', 'bagging')(x)))
            const m = outs[0]?.length ?? 0
            return fromStorage(
              Float64Array.from({ length: m }, (_, i) => outs.reduce((a, o) => a + (o[i] ?? 0), 0) / outs.length),
              [m],
              'float64',
            )
          },
          nFeaturesIn: p.nFeatures,
        }
      },
    },
    hyper,
  )
}

/**
 * Bagging for classification (sklearn.ensemble.BaggingClassifier): any classifier fitted on keyed bootstrap samples,
 * probabilities averaged (votes for members without probabilities). Exposition: the stages (training and out-of-bag
 * accuracy as members are added, each member's bootstrap rows).
 *
 * @param hyper - Member estimator (default a decision tree), members (default 10), bootstrap fraction and bootstrap.
 * @returns The estimator.
 */
export function baggingClassifier(hyper: Partial<BaggingHyper> = {}): ExplainedEstimator<BaggingHyper, FittedBagging> {
  return baggingEstimator('bagging-classifier', true, {
    estimator: hyper.estimator ?? decisionTreeClassifier(),
    nEstimators: hyper.nEstimators ?? 10,
    maxSamples: hyper.maxSamples ?? 1,
    bootstrap: hyper.bootstrap ?? true,
  })
}

/**
 * Bagging for regression (sklearn.ensemble.BaggingRegressor): predictions averaged; exposition as
 * {@link baggingClassifier} with R^2.
 *
 * @param hyper - Member estimator (default a regression tree), members, bootstrap fraction and bootstrap.
 * @returns The estimator.
 */
export function baggingRegressor(hyper: Partial<BaggingHyper> = {}): ExplainedEstimator<BaggingHyper, FittedBagging> {
  return baggingEstimator('bagging-regressor', false, {
    estimator: hyper.estimator ?? decisionTreeRegressor(),
    nEstimators: hyper.nEstimators ?? 10,
    maxSamples: hyper.maxSamples ?? 1,
    bootstrap: hyper.bootstrap ?? true,
  })
}

/** {@link baggingClassifier} as a class. */
export class BaggingClassifier extends ExplainedEstimatorClass<BaggingHyper, FittedBagging> {
  /** @param hyper - As {@link baggingClassifier}. */
  constructor(hyper: Partial<BaggingHyper> = {}) {
    super(baggingClassifier(hyper))
  }
}

/** {@link baggingRegressor} as a class. */
export class BaggingRegressor extends ExplainedEstimatorClass<BaggingHyper, FittedBagging> {
  /** @param hyper - As {@link baggingRegressor}. */
  constructor(hyper: Partial<BaggingHyper> = {}) {
    super(baggingRegressor(hyper))
  }
}
