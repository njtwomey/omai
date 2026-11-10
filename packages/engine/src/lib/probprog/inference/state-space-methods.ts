/**
 * The particle filter of a numeric state-space model as an inference method with the slots of the particle machine as
 * knobs: the number of particles, the resampling scheme (multinomial, systematic, stratified, residual), the effective-sample-size
 * threshold that triggers resampling, and the proposal (the bootstrap filter, the auxiliary filter with its first stage
 * guided by a point prediction, or a guided filter whose proposal sees the observation). One observation is a step, so a
 * page watches the effective sample size and the running evidence and reads the filtering posterior at any step.
 * `particleFilterMethod` is the same for a program's `markov` loop; this one takes the model as functions over tensors
 * (`ParticleModel` of `dynamics/filtering`), which is what `gaussianGuidedModel` builds for a Gaussian model.
 */

import type { Json } from '../../base/index.ts'
import { OmaiError } from '../../base/index.ts'
import {
  auxiliaryParticleAlgorithm,
  bootstrapParticleAlgorithm,
  type ParticleModel,
  type PfPath,
  type PfState,
  pfContract,
  type Resampling,
} from '../../dynamics/filtering/index.ts'
import type { Key } from '../../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { asTensor, primalOf, type TensorLike } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { type Posterior, weightedPosterior } from '../posterior.ts'
import { no, num, pick, posteriorJson, yes } from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the state-space filter method logs: those of the particle filters and the snapshot. */
export type StateSpaceFilterPath = PfPath | 'pf/snapshot'

const SCHEMES = ['systematic', 'multinomial', 'stratified', 'residual'] as const

/** The knobs of {@link stateSpaceFilterMethod} (the budget and snapshot knobs are added to them). */
export const stateSpaceFilterKnobs: DictSpace = dict({
  particles: int(2, 10_000_000, { default: 500, label: 'particles' }),
  resampling: choice([...SCHEMES], { default: 'systematic', label: 'resampling' }),
  essThreshold: real(0.01, 1, {
    default: 0.5,
    label: 'resampling threshold (ESS / N)',
    description: 'resample when the effective sample size falls below this fraction of the particles (1: every step)',
  }),
  proposal: choice(['bootstrap', 'auxiliary', 'guided'], {
    default: 'bootstrap',
    label: 'proposal',
    description:
      'bootstrap: move by the transition; auxiliary: resample first by the likelihood of a point prediction (needs `predict`); guided: move by a proposal that sees the observation (needs `guide`)',
  }),
})

/**
 * The particle filter of a state-space model as a method with a fine step: one observation per step. The tape is the
 * particle filters' (`pf/ess`, `pf/log-evidence`, `pf/resampled`, `pf/mean`, `pf/weights`, `pf/particles`, `pf/ancestors`);
 * `posterior(state)` is the weighted filtering posterior of the state at the current step with the running evidence. The
 * evidence estimate is unbiased on the natural scale whichever proposal is chosen; the proposals differ in its variance
 * (the guided filter's is smallest when the observation is informative). The run ends `converged` after the last
 * observation, and the state serialises, so a run resumes and forks exactly.
 *
 * @param model - The particle model; `predict` makes the auxiliary proposal available and `guide` the guided one.
 * @param observations - `[T, ...]`.
 * @param options - The name.
 * @returns The method, with `check` and `posterior`.
 * @concept particle-filter
 * @index estimator Particle filter of a state-space model: bootstrap, auxiliary or guided proposal, one observation per step.
 * @example
 * ```ts
 * import { stateSpaceFilterMethod } from 'omai-engine/probprog/inference'
 * import { filtering } from 'omai-engine/dynamics'
 * import { session } from 'omai-engine/trace'
 * const model = filtering.gaussianGuidedModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] }, { observationMatrix: [[1]] })
 * const method = stateSpaceFilterMethod(model, [[0.5], [0.2], [0.4]])
 * const s = session(method, { steps: 10, mode: 'eager', config: { particles: 100, proposal: 'guided' } })
 * [s.status, s.t] // => ['converged', 3]
 * ```
 */
export function stateSpaceFilterMethod(
  model: ParticleModel,
  observations: TensorLike,
  options: { readonly name?: string } = {},
): InferenceMethod<PfState, Config, StateSpaceFilterPath, ParticleModel> {
  const name = options.name ?? 'state-space-filter'
  const T = asTensor(observations).shape[0] ?? 0
  const check = (m: ParticleModel): Verdict => {
    const available = [
      'bootstrap',
      ...(m.predict === undefined ? [] : ['auxiliary']),
      ...(m.guide === undefined ? [] : ['guided']),
    ]
    return T > 0
      ? yes(`proposals available: ${available.join(', ')}`)
      : no('no observations', ['observations'], 'give at least one observation')
  }
  const posterior = (s: PfState): Posterior => {
    const n = s.logW.length
    return weightedPosterior({ x: primalOf(s.particles) }, s.logW, {
      engine: 'state-space-filter',
      status: s.t >= T ? 'converged' : 'stopped',
      logEvidence: { value: s.logEvidence, standardError: Number.NaN, method: 'particle filter' },
      diagnostics: {
        method: 'particle filter',
        step: s.t,
        particles: n,
        ess: s.ess.at(-1) ?? null,
        resamplings: s.resampled.filter(Boolean).length,
        partial: s.t < T,
      },
    })
  }
  return defineInferenceMethod<PfState, Config, StateSpaceFilterPath, ParticleModel>({
    name,
    family: 'smc',
    description:
      'particle filter of a state-space model with a bootstrap, auxiliary or guided proposal, one observation per step',
    config: stateSpaceFilterKnobs,
    check,
    posterior,
    defaults: { 'pf/ess': Log(), 'pf/log-evidence': Log(), 'pf/resampled': Log() },
    contract: (): Contract<PfState, StateSpaceFilterPath> =>
      defineContract({
        ...pfContract,
        'pf/snapshot': derived<PfState>({
          cost: 'dear',
          description: 'the weighted filtering posterior at the current step as JSON',
          compute: (s) => posteriorJson(posterior(s)),
        }),
      }),
    make: (config, key: Key): Algorithm<PfState, StateSpaceFilterPath> => {
      const proposal = pick(config, 'proposal')
      if (proposal === 'guided' && model.guide === undefined)
        throw new OmaiError('value', `${name}: the guided proposal needs the model's guide (see gaussianGuidedModel)`)
      if (proposal === 'auxiliary' && model.predict === undefined)
        throw new OmaiError('value', `${name}: the auxiliary proposal needs the model's predict`)
      // The bootstrap and auxiliary proposals move by the transition, whatever guide the model carries.
      const { guide: _guide, ...unguided } = model
      const options = {
        particles: num(config, 'particles'),
        resampling: pick(config, 'resampling') as Resampling,
        key,
      }
      const inner =
        proposal === 'auxiliary'
          ? auxiliaryParticleAlgorithm(unguided, observations, options)
          : bootstrapParticleAlgorithm(proposal === 'guided' ? model : unguided, observations, {
              ...options,
              essThreshold: num(config, 'essThreshold'),
            })
      return {
        ...inner,
        step: (s, ctx) => inner.step(s, ctx as never),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as PfState,
      } as Algorithm<PfState, StateSpaceFilterPath>
    },
  })
}
