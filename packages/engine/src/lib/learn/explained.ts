import type { Json } from '../base/index.ts'
import { resolveKey, resolvePolicy, resolveTracer } from '../context/index.ts'
import type { Explained, Exposition } from '../exposition/index.ts'
import type { RunOptions } from '../trace/index.ts'
import {
  defineEstimator,
  type Estimator,
  type EstimatorSpec,
  type FitOptions,
  type Fitted,
  type FittedFacts,
  type Hyper,
  type LearnData,
  type Verbs,
} from './estimator.ts'

/** What an explained fit learns: the parameters (plain JSON) and the exposition records that explain them. */
export interface ExplainedFit<P extends Json> {
  readonly params: P
  readonly exposition: readonly Exposition[]
}

/** A fitted value that also carries its exposition records (empty when revived from parameters). */
export type ExplainedFitted<P extends Json = Json, V = object> = Explained<Fitted<P> & V>

/** How to build an explained estimator: an {@link EstimatorSpec} whose `fit` also returns exposition records. */
export interface ExplainedEstimatorSpec<H extends Hyper, P extends Json, V extends Verbs & FittedFacts>
  extends Omit<EstimatorSpec<H, P, V>, 'fit' | 'fitTransform'> {
  /** Learn the parameters and the records that explain the fit. */
  fit(data: LearnData, hyper: H, options: FitOptions): ExplainedFit<P>
}

/**
 * Build an estimator whose fitted values carry exposition records (section 5.0's contract) beside their verbs: `fit`
 * returns `{ ...fitted, exposition }`, `revive` rebuilds the verbs from the parameters with no records, and `with`
 * re-applies the spec, so the estimator still clones, tunes and cross-validates as any other.
 *
 * @param spec - Name, kind, hyperparameter space, an explained `fit` and `methods`.
 * @param hyper - The hyperparameters.
 * @returns The estimator.
 * @concept exposition
 * @example
 * ```ts
 * import { defineExplainedEstimator } from 'whitebox-engine/learn'
 * import { anomalyScores } from 'whitebox-engine/exposition'
 * import { full } from 'whitebox-engine/tensor'
 * const zero = defineExplainedEstimator(
 *   {
 *     name: 'zero',
 *     kind: 'outlier',
 *     fit: () => ({ params: { t: 1 }, exposition: [anomalyScores({ method: 'zero', scores: [0], threshold: 1 })] }),
 *     methods: (p) => ({ predict: () => full([1], p.t) }),
 *   },
 *   {},
 * )
 * zero.fit({ x: [[0]] }).exposition.length // => 1
 * zero.revive({ t: 1 }).exposition.length // => 0
 * ```
 */
export function defineExplainedEstimator<H extends Hyper, P extends Json, V extends Verbs & FittedFacts>(
  spec: ExplainedEstimatorSpec<H, P, V>,
  hyper: H,
): Estimator<H, ExplainedFitted<P, V>> {
  const plain = defineEstimator<H, P, V>(
    { ...spec, fit: (data, h, options) => spec.fit(data, h, options).params },
    hyper,
  )
  return {
    name: plain.name,
    kind: plain.kind,
    hyper,
    ...(plain.hyperSpace === undefined ? {} : { hyperSpace: plain.hyperSpace }),
    fit: (data, options = {}) => {
      const { params, exposition } = spec.fit(data, hyper, options)
      return { ...plain.revive(params), exposition }
    },
    revive: (params) => ({ ...plain.revive(params), exposition: [] }),
    with: (changes) => defineExplainedEstimator(spec, spec.update?.(hyper, changes) ?? ({ ...hyper, ...changes } as H)),
  }
}

/**
 * Run options for a traced fit (section 3.2's rules): the caller's parent run, or the resolved key, tracer and policy.
 *
 * @param options - The fit options.
 * @param name - The name the key is derived under when none is given.
 * @returns Options to spread into `runAlgorithm`.
 * @example
 * ```ts
 * import { fitRunOptions } from 'whitebox-engine/learn'
 * import { key } from 'whitebox-engine/random'
 * fitRunOptions({ key: key(0) }, 'kmeans').key !== undefined // => true
 * ```
 */
export function fitRunOptions(
  options: FitOptions,
  name: string,
): Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'> {
  const key = resolveKey(options, name)
  if (options.parent !== undefined) return { key, parent: options.parent }
  return { key, tracer: resolveTracer(options), policy: resolvePolicy(options) }
}
