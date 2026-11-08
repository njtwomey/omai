/**
 * The estimator kinds as add-ons to a base: what a classifier, a regressor, a clusterer, a transformer or an outlier
 * detector brings to every method of its kind, whatever way it runs. Each kind contributes standard derived paths,
 * computed from any state through the fitted value it yields: a decision surface over the plane and the predictions
 * on the training rows for classifiers and regressors, the assignments for clusterers, the embedding for transformers,
 * the scores for outlier detectors. Adding a probe to every classifier is one line here.
 */

import { type Json, OmaiError, toRows } from '../../base/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { type DerivedSource, derived } from '../../trace/index.ts'
import type { EstimatorKind, Fitted, LearnData } from '../estimator.ts'
import { decisionSurfaceOf } from '../explained.ts'

/** What the kind add-ons need: the training data and the fitted value a state yields. */
export interface KindContext<S> {
  readonly data: LearnData
  fitted(state: S): Fitted
}

/** The standard derived paths of a kind. */
export type KindProbes<S> = Readonly<Record<string, DerivedSource<S>>>

/** The paths the kinds add (see {@link kindProbes}), as a type: the bases include them in the paths of their sessions. */
export type KindPath =
  | 'model/decision-surface'
  | 'model/predictions'
  | 'model/assignments'
  | 'model/embedding'
  | 'model/scores'

const rows = (t: Tensor): number[][] => {
  const d = t.shape[1] ?? 1
  const flat = toFlat(t)
  return toRows(flat, Math.ceil(flat.length / d), d)
}

function verb<K extends 'predict' | 'transform' | 'decisionFunction'>(fitted: Fitted, name: K): NonNullable<Fitted[K]> {
  const f = fitted[name]
  if (f === undefined) throw new OmaiError('value', `${fitted.estimator}: no ${name} to derive from`)
  return f
}

/**
 * The standard derived paths of an estimator kind.
 *
 * @param kind - The kind.
 * @param ctx - The data and the fitted value of a state.
 * @returns Derived sources by path (`model/decision-surface`, `model/predictions`, `model/assignments`,
 *   `model/embedding`, `model/scores`), empty for kinds with none.
 * @concept log-contract
 * @example
 * ```ts
 * import { kindProbes } from 'omai-engine/learn'
 * const fitted = () => ({ estimator: 'x', kind: 'classifier', params: {} })
 * Object.keys(kindProbes('classifier', { data: { x: [[0, 0]] }, fitted })) // => ['model/decision-surface', 'model/predictions']
 * ```
 */
export function kindProbes<S>(kind: EstimatorKind, ctx: KindContext<S>): KindProbes<S> {
  const { data } = ctx
  const names = (data as { readonly featureNames?: readonly string[] }).featureNames
  const predictions = derived<S>({
    description: 'the predictions on the training rows',
    compute: (s) => toFlat(verb(ctx.fitted(s), 'predict')(data.x)),
  })
  switch (kind) {
    case 'classifier':
    case 'regressor':
      return {
        'model/decision-surface': derived<S>({
          description: 'the prediction over the plane of the first two features, with the training rows',
          cost: 'dear',
          compute: (s, args) => {
            const res = (args as { readonly resolution?: number } | undefined)?.resolution
            return decisionSurfaceOf(ctx.fitted(s), data.x, {
              ...(res === undefined ? {} : { resolution: res }),
              ...(data.y === undefined ? {} : { y: data.y }),
              ...(names === undefined ? {} : { featureNames: names }),
            })
          },
        }),
        'model/predictions': predictions,
      }
    case 'clusterer':
      return {
        'model/assignments': derived<S>({
          description: 'the cluster of each training row',
          compute: (s) => toFlat(verb(ctx.fitted(s), 'predict')(data.x)),
        }),
      }
    case 'transformer':
      return {
        'model/embedding': derived<S>({
          description: 'the training rows mapped forwards',
          cost: 'dear',
          compute: (s) => rows(verb(ctx.fitted(s), 'transform')(data.x)) as Json,
        }),
      }
    case 'outlier':
      return {
        'model/scores': derived<S>({
          description: 'the outlier score of each training row',
          compute: (s) => toFlat(verb(ctx.fitted(s), 'decisionFunction')(data.x)),
        }),
      }
    default:
      return {}
  }
}
