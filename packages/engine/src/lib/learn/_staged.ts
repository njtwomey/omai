// Ensembles of base-estimator fits, one fit per step (not exported from the domain): the algorithm the multiclass,
// multilabel and multi-output wrappers share, and the small readers of a fitted binary base they all use.
import type { Json } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import type { Estimator, Fitted, LearnData } from './estimator.ts'

/** The state of a staged ensemble: the plan of the problems, and the fitted parameters of those solved so far. */
export type StagedState<Plan extends Json> = { readonly plan: Plan; readonly models: Json[] }

/** What a wrapper declares about its problems. */
export interface StagedSpec<Plan extends Json> {
  readonly name: string
  /** The plan (which problems, in which order), chosen from the run's key. */
  plan(key: Key): Plan
  /** The number of problems in a plan. */
  total(plan: Plan): number
  /** The data of problem `index`. */
  stage(plan: Plan, index: number): LearnData
  /** A problem with nothing to learn (a constant label) answered directly, as the parameters standing in for a fit. */
  fixed?(plan: Plan, index: number): Json | undefined
}

/**
 * A staged ensemble as an algorithm: the plan at the start with its first problem solved (so there is a model to
 * look at), then one problem fitted by the base estimator per step. It logs the metric `problems`.
 */
export function stagedAlgorithm<Plan extends Json>(
  estimator: Estimator,
  spec: StagedSpec<Plan>,
): Algorithm<StagedState<Plan>> {
  const solve = (plan: Plan, index: number, key: Key): Json =>
    spec.fixed?.(plan, index) ?? estimator.fit(spec.stage(plan, index), { key, explain: false }).params
  return {
    name: spec.name,
    config: { base: estimator.name },
    schema: channels({ metrics: { problems: { description: 'binary problems solved so far', direction: 'higher' } } }),
    init: (key) => {
      const plan = spec.plan(key)
      return { plan, models: [solve(plan, 0, foldIn(key, 0))] }
    },
    step: (s, ctx) => ({ plan: s.plan, models: [...s.models, solve(s.plan, s.models.length, ctx.key)] }),
    done: (s) => (s.models.length >= spec.total(s.plan) ? 'converged' : undefined),
    observe: (s) => ({ metrics: { problems: s.models.length } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StagedState<Plan>,
  }
}

/** Whether a fitted base reports a decision function (so its scores are margins rather than probabilities). */
export const hasDecision = (f: Fitted): boolean => typeof f.decisionFunction === 'function'

/**
 * The confidence of a fitted binary base in the positive class, as scikit-learn's `_predict_binary`: its decision
 * function when it has one, else the probability of the second class, else its prediction.
 */
export function confidence(f: Fitted, x: TensorLike): Float64Array {
  if (f.decisionFunction !== undefined) return Float64Array.from(toFlat(f.decisionFunction(x)))
  if (f.predictProba !== undefined) {
    const p = f.predictProba(x)
    const k = p.shape[1] ?? 1
    const flat = toFlat(p)
    return Float64Array.from({ length: flat.length / k }, (_, i) => flat[i * k + (k - 1)] ?? 0)
  }
  return Float64Array.from(toFlat((f.predict as (x: TensorLike) => Tensor)(x)))
}

/** The labels a fitted base predicts for rows. */
export function predicted(f: Fitted, x: TensorLike): Float64Array {
  if (f.predict === undefined) return confidence(f, x)
  return Float64Array.from(toFlat(f.predict(x)))
}

/** A float64 tensor from row-major data and a shape. */
export const matrix = (data: ArrayLike<number>, shape: readonly number[]): Tensor =>
  fromStorage(Float64Array.from(data), [...shape], 'float64')
