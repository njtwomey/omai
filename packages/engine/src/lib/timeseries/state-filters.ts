/**
 * Filters of state-space models as estimators on the iterative base. `nonlinearFilter` runs the extended, unscented
 * or ensemble Kalman filter over a model; `particleFilter` runs the bootstrap or auxiliary particle filter with a
 * resampling scheme. The model (a table of state-space models with their noise as knobs, or one the data carries), the
 * filter, the sigma-point scheme or the resampling and the particle count are hyperparameters filled from the option
 * tables of `dynamics/filtering`. The fit steps through the observations one at a time, so its `session` runs live and
 * `fittedAt(t)` is the filter after t observations; the fitted value carries the filtered means and covariances or the
 * weighted means, the effective sample sizes and the evidence.
 */

import {
  type AnyNonlinearFilterPath,
  auxiliaryParticleAlgorithm,
  bootstrapParticleAlgorithm,
  gaussianParticleModel,
  type NonlinearFilterState,
  type NonlinearModel,
  nonlinearFilterContract,
  nonlinearFilters,
  type ParticleModel,
  type PfPath,
  type PfState,
  particleProposals,
  pfContract,
  resamplers,
  stateSpaceModels,
} from '../dynamics/filtering/index.ts'
import {
  completeOption,
  type FittedFacts,
  type IterativeEstimator,
  iterative,
  type LearnData,
  type Verbs,
} from '../learn/index.ts'
import { dict, int, type SpaceValue } from '../space/index.ts'
import { asTensor, reshape, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Contract, defineContract, Log, tensor } from '../trace/index.ts'

/** Observations for a filter, and the model they come from when it is not one of the table's. */
export interface StateSpaceData extends LearnData {
  /** The model to filter with (default: the estimator's `model` slot). */
  readonly model?: NonlinearModel
  /** A particle model, for a non-Gaussian one (default: the Gaussian form of the model). */
  readonly particleModel?: ParticleModel
  /** The true states `[T, n]`, when known (for the error). */
  readonly states?: TensorLike
}

/**
 * Data for a filter: observations `[T, p]`, with the model they come from and the true states when known.
 *
 * @param observations - `[T, p]` (a plain series is one column).
 * @param extra - The model, a particle model, and the true states.
 * @returns Data the filters fit.
 * @example
 * ```ts
 * import { stateSpaceData } from 'mdml-engine/timeseries'
 * stateSpaceData([0.1, 0.2, 0.3]).x.shape // => [3, 1]
 * ```
 */
export function stateSpaceData(
  observations: TensorLike,
  extra: { readonly model?: NonlinearModel; readonly particleModel?: ParticleModel; readonly states?: TensorLike } = {},
): StateSpaceData {
  const t = asTensor(observations, 'float64')
  return { x: t.shape.length === 1 ? reshape(t, [t.shape[0] ?? 0, 1]) : t, ...extra }
}

function modelOf(data: LearnData, slot: SpaceValue): NonlinearModel {
  return (data as StateSpaceData).model ?? stateSpaceModels.make(slot)
}

/** The error of filtered means against the true states, when the data has them. */
function rmse(means: readonly (readonly number[])[], states: TensorLike | undefined): number | undefined {
  if (states === undefined) return undefined
  const truth = asTensor(states)
  const d = truth.shape[1] ?? 1
  const flat = toFlat(truth)
  let sum = 0
  let count = 0
  means.forEach((m, t) => {
    m.forEach((v, j) => {
      const e = v - (flat[t * d + j] ?? 0)
      sum += e * e
      count++
    })
  })
  return count === 0 ? undefined : Math.sqrt(sum / count)
}

/**
 * The hyperparameters of {@link nonlinearFilter}. A slot's value is whole (the option and every knob of it); fill one
 * from a name and some knobs with `completeOption(table, { kind, ...knobs })` from `mdml-engine/learn`.
 */
export interface NonlinearFilterHyper {
  /** The state-space model, a value of `stateSpaceModels.space` (ignored when the data carries a model). */
  readonly model: SpaceValue
  /** The filter, a value of `nonlinearFilters.space`: `{ kind: 'unscented', alpha, beta, kappa }` and so on. */
  readonly filter: SpaceValue
}

/** The fitted parameters of {@link nonlinearFilter}: the filtered moments per step, as plain numbers. */
export type NonlinearFilterParams = {
  readonly filter: string
  readonly steps: number
  /** Filtered means, one row per step. */
  readonly means: number[][]
  /** Filtered covariances, each flattened row-major, one row per step. */
  readonly covariances: number[][]
  readonly logLikelihood: number
  readonly stateSize: number
}

/** What a fitted {@link nonlinearFilter} offers beyond its parameters. */
export interface NonlinearFilterVerbs extends Verbs, FittedFacts {
  /** The filtered mean of one state component at every step. */
  component(i: number): number[]
  /** The root mean square error of the filtered means against true states (`undefined` without them). */
  error(states: TensorLike): number | undefined
}

/** The paths of {@link nonlinearFilter} beyond the base's. */
export type NonlinearFilterEstimatorPath = AnyNonlinearFilterPath

const filterContract = (): Contract<NonlinearFilterState, AnyNonlinearFilterPath> =>
  defineContract({
    ...nonlinearFilterContract,
    'filter/sigma-points': tensor({
      role: 'state',
      shape: ['point', 'state'],
      description: 'the sigma points of the posterior (unscented filters)',
    }),
    'filter/ensemble': tensor({
      role: 'state',
      shape: ['member', 'state'],
      description: 'the analysis ensemble (ensemble filter)',
    }),
  })

/**
 * A nonlinear Kalman filter as an estimator of a state-space model: choose the `model` (a random walk, a pendulum seen
 * through the sine of its angle, a body tracked by its range, with the noise as knobs, or a model the data carries) and
 * the `filter` (extended with Jacobians by autodiff, unscented with Van der Merwe or Julier sigma points and their
 * knobs, ensemble with a size and inflation). `fit` takes observations (see {@link stateSpaceData}); `session` steps
 * through them live, logging the paths of `nonlinearFilterContract` (and the sigma points or the ensemble). The fitted
 * value has the filtered means and covariances, the log-likelihood and the error against true states when given.
 *
 * @concept unscented-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { completeOption } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * import { nonlinearFilter, stateSpaceData } from 'mdml-engine/timeseries'
 * const model = filtering.stateSpaceModels.make({ kind: 'range' })
 * const { observations } = filtering.simulateStateSpace(model, 6, key(0))
 * const unscented = completeOption(filtering.nonlinearFilters, { kind: 'unscented' })
 * const fitted = nonlinearFilter.with({ model: completeOption(filtering.stateSpaceModels, { kind: 'range' }), filter: unscented })
 * fitted.fit(stateSpaceData(observations)).params.steps // => 6
 * ```
 */
export const nonlinearFilter: IterativeEstimator<
  NonlinearFilterHyper,
  NonlinearFilterState,
  NonlinearFilterParams,
  NonlinearFilterVerbs,
  NonlinearFilterEstimatorPath
> = iterative<
  NonlinearFilterHyper,
  NonlinearFilterState,
  NonlinearFilterParams,
  NonlinearFilterVerbs,
  NonlinearFilterEstimatorPath
>({
  name: 'timeseries/nonlinear-filter',
  kind: 'other',
  defaults: { model: completeOption(stateSpaceModels, { kind: 'random-walk' }), filter: { kind: 'extended' } },
  hyperSpace: () => dict({ model: stateSpaceModels.space, filter: nonlinearFilters.space }),
  algorithm: (data, h, key) => nonlinearFilters.make(h.filter)(modelOf(data, h.model), data.x, key),
  maxSteps: (_, data) => asTensor(data.x).shape[0] ?? 0,
  params: (s, _data, h) => {
    const n = s.x.shape.reduce((a, b) => a * b, 1)
    return {
      filter: nonlinearFilters.optionOf(h.filter).name,
      steps: s.t,
      means: s.means.map((m) => Array.from(toFlat(m))),
      covariances: s.covs.map((c) => Array.from(toFlat(c))),
      logLikelihood: toFlat(s.logLik)[0] ?? 0,
      stateSize: n,
    }
  },
  verbs: (p) => ({
    component: (i) => p.means.map((m) => m[i] ?? 0),
    error: (states) => rmse(p.means, states),
  }),
  contract: () => filterContract(),
  records: () => [],
  logs: {
    'filter/mean': Log(),
    'filter/cov': Log(),
    'filter/innovation': Log(),
    'filter/log-likelihood': Log(),
  },
})

/** The hyperparameters of {@link particleFilter}. */
export interface ParticleFilterHyper {
  /** The state-space model, a value of `stateSpaceModels.space` (ignored when the data carries a model). */
  readonly model: SpaceValue
  /** The proposal, a value of `particleProposals.space`: `{ kind: 'bootstrap', essThreshold }` or `{ kind: 'auxiliary' }`. */
  readonly proposal: SpaceValue
  /** The resampling scheme, a value of `resamplers.space`. */
  readonly resampling: SpaceValue
  /** The number of particles. */
  readonly particles: number
}

/** The fitted parameters of {@link particleFilter}: what the cloud said at every step, as plain numbers. */
export type ParticleFilterParams = {
  readonly proposal: string
  readonly resampling: string
  readonly steps: number
  readonly particles: number
  /** Weighted means, one row per step. */
  readonly means: number[][]
  /** The effective sample size before resampling at each step. */
  readonly ess: number[]
  /** Whether each step resampled. */
  readonly resampled: boolean[]
  readonly logEvidence: number
}

/** What a fitted {@link particleFilter} offers beyond its parameters. */
export interface ParticleFilterVerbs extends Verbs, FittedFacts {
  /** The weighted mean of one state component at every step. */
  component(i: number): number[]
  /** The root mean square error of the weighted means against true states (`undefined` without them). */
  error(states: TensorLike): number | undefined
  /** The steps at which the filter resampled. */
  resamplingSteps(): number[]
}

const particleExtras: Contract<PfState, PfPath> = pfContract

/**
 * A particle filter as an estimator of a state-space model: choose the `model` (as for {@link nonlinearFilter}, or a
 * non-Gaussian particle model the data carries), the `proposal` (bootstrap, which resamples when the effective sample
 * size falls below a threshold, or auxiliary, which resamples first by a point prediction), the `resampling` scheme
 * (multinomial, systematic, stratified, residual) and the number of `particles`. The session logs the effective sample
 * size, the evidence, whether each step resampled, the weights, the particles and their ancestry; the fitted value has
 * the weighted means, the ESS per step, the resampling steps and the evidence.
 *
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { key } from 'mdml-engine/random'
 * import { particleFilter, stateSpaceData } from 'mdml-engine/timeseries'
 * const model = filtering.stateSpaceModels.make({ kind: 'random-walk' })
 * const { observations } = filtering.simulateStateSpace(model, 5, key(0))
 * particleFilter.with({ particles: 50 }).fit(stateSpaceData(observations), { key: key(1) }).params.steps // => 5
 * ```
 */
export const particleFilter: IterativeEstimator<
  ParticleFilterHyper,
  PfState,
  ParticleFilterParams,
  ParticleFilterVerbs,
  PfPath
> = iterative<ParticleFilterHyper, PfState, ParticleFilterParams, ParticleFilterVerbs, PfPath>({
  name: 'timeseries/particle-filter',
  kind: 'other',
  defaults: {
    model: completeOption(stateSpaceModels, { kind: 'random-walk' }),
    proposal: { kind: 'bootstrap', essThreshold: 0.5 },
    resampling: { kind: 'systematic' },
    particles: 200,
  },
  hyperSpace: (h) =>
    dict({
      model: stateSpaceModels.space,
      proposal: particleProposals.space,
      resampling: resamplers.space,
      particles: int(10, 5000, { log: true, default: h.particles, label: 'particles' }),
    }),
  algorithm: (data, h, key) => {
    const given = (data as StateSpaceData).particleModel
    const model = given ?? gaussianParticleModel(modelOf(data, h.model))
    const proposal = particleProposals.make(h.proposal)
    const options = { particles: h.particles, resampling: resamplers.make(h.resampling), key }
    return proposal.auxiliary
      ? auxiliaryParticleAlgorithm(model, data.x, options)
      : bootstrapParticleAlgorithm(model, data.x, { ...options, essThreshold: proposal.essThreshold })
  },
  maxSteps: (_, data) => asTensor(data.x).shape[0] ?? 0,
  params: (s, _data, h) => ({
    proposal: particleProposals.optionOf(h.proposal).name,
    resampling: resamplers.optionOf(h.resampling).name,
    steps: s.t,
    particles: h.particles,
    means: s.means.map((m) => Array.from(m)),
    ess: [...s.ess],
    resampled: [...s.resampled],
    logEvidence: s.logEvidence,
  }),
  verbs: (p) => ({
    component: (i) => p.means.map((m) => m[i] ?? 0),
    error: (states) => rmse(p.means, states),
    resamplingSteps: () => p.resampled.flatMap((r, t) => (r ? [t] : [])),
  }),
  contract: () => particleExtras,
  logs: {
    'pf/ess': Log(),
    'pf/log-evidence': Log(),
    'pf/resampled': Log(),
    'pf/mean': Log(),
    'pf/weights': Log(),
  },
})
