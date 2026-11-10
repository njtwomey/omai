/**
 * The ensemble samplers as inference methods: the affine-invariant stretch (and differential evolution) move of NumPyro's
 * `AIES` (`aiesMethod`) and ensemble slice sampling (`ensembleSliceMethod`), on programs and bound specs. A population of
 * walkers moves together on the unconstrained scale and each move is built from the other walkers, so there is no gradient,
 * no step size and no proposal shape to tune; every walker of an ensemble is a chain of the posterior (so R-hat and the
 * effective sample size compare walkers), and one step is one transition of every ensemble (half the walkers, then the other
 * half). The slice sampler tunes the scale of its directions in warm-up.
 */

import type { Json } from '../../base/index.ts'
import {
  chainFromJson,
  chainToJson,
  type EnsembleState,
  ensembleSliceKernel,
  type MCMCKernel,
  type MCMCState,
  stretchMoveKernel,
  type TransitionInfo,
} from '../../sampling/index.ts'
import { bool, choice, dict, int, real } from '../../space/index.ts'
import { type Config, defineContract, scalar, stateFromJson, stateToJson } from '../../trace/index.ts'
import type { McmcRunState } from './_mcmc-driver.ts'
import { type McmcPath, type McmcPlan, mcmcMethodOf, type Warmer } from './_mcmc-driver.ts'
import { flag, type MethodModel, type ModelAnalysis, num, pick } from './_support.ts'
import type { InferenceMethod } from './method.ts'

/** The type of an ensemble method: the MCMC state, the shared paths and the scale of the slice directions. */
export type EnsembleMethod = InferenceMethod<McmcRunState, Config, McmcPath | 'ensemble/scale', ModelAnalysis>

const walkerKnobs = (defaultWalkers: number) => ({
  numChains: int(1, 64, { default: 1, label: 'ensembles' }),
  numWarmup: int(0, 1_000_000, { default: 200, label: 'warm-up iterations' }),
  numDraws: int(1, 1_000_000, { default: 500, label: 'draws per walker' }),
  numWalkers: int(4, 1024, {
    default: defaultWalkers,
    label: 'walkers',
    description:
      'the walkers of one ensemble: each is a chain of the posterior (more than twice the dimension is the usual advice)',
  }),
})

/**
 * The affine-invariant ensemble sampler as a method (NumPyro's `AIES`): `numWalkers` walkers per ensemble, each moved by the
 * stretch move of Goodman and Weare (or the differential evolution move) built from the walkers of the other half. It needs
 * no gradient and has nothing to tune, and it is unaffected by linear transformations of the coordinates, so correlated and
 * badly scaled posteriors need no preconditioner. Every walker is a chain of the posterior; one transition of every ensemble
 * per step, warm-up is burn-in.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `aies`).
 * @returns The method.
 * @concept ensemble-sampling
 * @index estimator Affine-invariant ensemble sampler (stretch and differential moves) as a steppable method.
 * @example
 * ```ts
 * import { aiesMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const m = aiesMethod(model)
 * const s = session(m, { config: { numWalkers: 6, numWarmup: 20, numDraws: 20 }, steps: 500, mode: 'eager' })
 * m.posterior?.(s.state, s.config).chains?.numChains // => 6
 * ```
 */
export function aiesMethod(model: MethodModel, options: { readonly name?: string } = {}): EnsembleMethod {
  type T = Record<string, never>
  return mcmcMethodOf<T, 'ensemble/scale'>(
    {
      name: 'aies',
      description: 'the affine-invariant ensemble sampler, one transition of every ensemble per step',
      knobs: dict({
        ...walkerKnobs(16),
        move: choice(['stretch', 'differential'], { default: 'stretch', label: 'move' }),
        a: real(1.1, 10, { default: 2, label: 'stretch scale' }),
      }),
      plan: (b) => {
        const walkers = num(b.config, 'numWalkers')
        const kernel = (): MCMCKernel =>
          stretchMoveKernel(b.um.logDensity, {
            numWalkers: walkers,
            move: pick(b.config, 'move') === 'differential' ? 'differential' : 'stretch',
            a: num(b.config, 'a'),
          })
        return {
          kernelDim: walkers * b.um.dim,
          walkers,
          kernel,
          tuned0: {},
          tunedToJson: (t) => stateToJson(t),
          tunedFromJson: (j) => stateFromJson(j) as T,
          tunedSummary: () => ({ numWalkers: walkers }),
        } satisfies McmcPlan<T>
      },
    },
    model,
    options,
  ) as unknown as EnsembleMethod
}

/**
 * Ensemble slice sampling as a method (NumPyro's `ESS`): each walker of one half does a slice-sampling step along a
 * direction built from the other half (the difference of two walkers, or a draw from the Gaussian fitted to them), scaled by
 * the scale mu. Every walker moves every transition and there is no gradient; the only parameter, mu, is tuned in warm-up
 * from the ratio of stepping-out to shrinking steps (`tuneMu`) and then fixed, so the draws come from an exact Markov chain.
 * The tape adds `ensemble/scale`, mu per ensemble. Every walker is a chain of the posterior.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `ensemble-slice`).
 * @returns The method.
 * @concept ensemble-sampling
 * @index estimator Ensemble slice sampling with a tuned direction scale as a steppable method.
 * @example
 * ```ts
 * import { ensembleSliceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(ensembleSliceMethod(model), { config: { numWalkers: 6, numWarmup: 20, numDraws: 20 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function ensembleSliceMethod(model: MethodModel, options: { readonly name?: string } = {}): EnsembleMethod {
  type T = { readonly mu: number }
  type W = { readonly chain: EnsembleState; readonly info: TransitionInfo | undefined }
  return mcmcMethodOf<T, 'ensemble/scale'>(
    {
      name: 'ensemble-slice',
      description: 'ensemble slice sampling with a tuned direction scale, one transition of every ensemble per step',
      extraPaths: defineContract({
        'ensemble/scale': scalar({
          description: 'the scale mu of the slice directions of each ensemble (tuned in warm-up)',
        }),
      }) as never,
      knobs: dict({
        ...walkerKnobs(16),
        move: choice(['differential', 'gaussian'], { default: 'differential', label: 'direction' }),
        mu: real(0.01, 100, { log: true, default: 1, label: 'initial scale' }),
        tuneMu: bool({ default: true, label: 'tune the scale in warm-up' }),
      }),
      plan: (b) => {
        const walkers = num(b.config, 'numWalkers')
        const move = pick(b.config, 'move') === 'gaussian' ? 'gaussian' : 'differential'
        const mu0 = num(b.config, 'mu')
        const numWarmup = num(b.config, 'numWarmup')
        const kernelAt = (mu: number, tuneMu: boolean): MCMCKernel<EnsembleState> =>
          ensembleSliceKernel(b.um.logDensity, { numWalkers: walkers, move, mu, tuneMu })
        const tuning = kernelAt(mu0, true)
        const warm: Warmer<T> | undefined =
          numWarmup > 0 && flag(b.config, 'tuneMu')
            ? {
                init: (start) => ({ chain: tuning.init(start), info: undefined }) satisfies W,
                step: (w, ctx) => {
                  const r = tuning.step(ctx.key, (w as W).chain)
                  return { chain: r.state, info: r.info } satisfies W
                },
                chain: (w) => (w as W).chain,
                info: (w) => (w as W).info,
                result: (w) => ({ mu: (w as W).chain.mu }),
                toJson: (w) => ({ chain: chainToJson((w as W).chain) }),
                fromJson: (j) =>
                  ({ chain: chainFromJson<EnsembleState>((j as { chain: Json }).chain), info: undefined }) satisfies W,
              }
            : undefined
        return {
          kernelDim: walkers * b.um.dim,
          walkers,
          kernel: (t: T) => kernelAt(t.mu, false),
          tuned0: { mu: mu0 },
          ...(warm === undefined ? {} : { warm }),
          tunedToJson: (t) => stateToJson(t),
          tunedFromJson: (j) => stateFromJson(j) as T,
          tunedSummary: (t) => ({ numWalkers: walkers, mu: t.mu }),
          extraTape: (ctx, states) => {
            ctx.log(
              'ensemble/scale',
              states.map((s: MCMCState) => (s as EnsembleState).mu),
            )
          },
        } satisfies McmcPlan<T>
      },
    },
    model,
    options,
  ) as unknown as EnsembleMethod
}
