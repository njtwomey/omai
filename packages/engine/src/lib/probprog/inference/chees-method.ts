/**
 * ChEES-HMC as an inference method: Hamiltonian Monte Carlo for an ensemble of chains with a jittered trajectory, whose trajectory length
 * is adapted in warm-up by ascent on the ChEES criterion across the chains and whose step size is adapted by dual averaging on their
 * harmonic-mean acceptance (the library's `cheesAdaptationAlgorithm`). It is the trajectory-length adaptation of HMC: where
 * `hmcMethod` fixes the number of leapfrog steps and adapts the step size, this method learns how far to go, and it suits many
 * chains in parallel (the case of the original, run on a GPU). Every chain of an ensemble is a chain of the posterior.
 */

import {
  type CheesParameters,
  type CheesState,
  type CheesWarmupState,
  cheesAdaptationAlgorithm,
  cheesAdaptationResult,
  cheesKernel,
  type MCMCKernel,
} from '../../sampling/index.ts'
import { bool, dict, int, real } from '../../space/index.ts'
import { type Config, defineContract, scalar, stateFromJson, stateToJson, tensor } from '../../trace/index.ts'
import { type McmcPath, type McmcPlan, type McmcRunState, mcmcMethodOf, type Warmer } from './_mcmc-driver.ts'
import { flag, type MethodModel, type ModelAnalysis, num } from './_support.ts'
import type { InferenceMethod } from './method.ts'

/** The paths ChEES-HMC adds to the shared ones. */
export type CheesPath = 'chees/trajectoryLength' | 'chees/numSteps'

/** The type of the ChEES-HMC method. */
export type CheesMethod = InferenceMethod<McmcRunState, Config, McmcPath | CheesPath, ModelAnalysis>

/**
 * ChEES-HMC as a method: `numWalkers` chains per ensemble share a step size and a jittered trajectory length, both tuned in warm-up
 * from all of them (the trajectory length by Adam on the ChEES criterion, with a quasi-random Halton jitter of the number of
 * leapfrog steps; the step size by dual averaging towards a harmonic-mean acceptance of `targetAcceptance`, 0.651). The mass
 * matrix is the identity. Every chain is a chain of the posterior, and the tape adds the trajectory length and the number of leapfrog
 * steps of every ensemble. One transition of every ensemble per step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `chees`).
 * @returns The method.
 * @concept chees-hmc
 * @index estimator ChEES-HMC: jittered HMC over many chains with an adapted trajectory length, as a steppable method.
 * @example
 * ```ts
 * import { cheesMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(cheesMethod(model), { config: { numWalkers: 4, numWarmup: 30, numDraws: 10 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function cheesMethod(model: MethodModel, options: { readonly name?: string } = {}): CheesMethod {
  type W = CheesWarmupState
  return mcmcMethodOf<CheesParameters, CheesPath>(
    {
      name: 'chees',
      description:
        'ChEES-HMC: jittered HMC over many chains with an adapted trajectory length, one transition of every ensemble per step',
      extraPaths: defineContract({
        'chees/trajectoryLength': tensor({
          role: 'output',
          shape: ['chains'],
          description: 'the trajectory length of each ensemble',
        }),
        'chees/numSteps': scalar({
          description: 'the leapfrog steps of the last trajectory (the jitter is shared by the chains)',
        }),
      }) as never,
      knobs: dict({
        numChains: int(1, 64, { default: 1, label: 'ensembles' }),
        numWarmup: int(0, 1_000_000, { default: 400, label: 'warm-up iterations' }),
        numDraws: int(1, 1_000_000, { default: 500, label: 'draws per chain' }),
        numWalkers: int(2, 1024, { default: 16, label: 'chains per ensemble' }),
        targetAcceptance: real(0.3, 0.95, { default: 0.651, label: 'target acceptance' }),
        learningRate: real(0.001, 1, { log: true, default: 0.025, label: 'trajectory length learning rate' }),
        initialStepSize: real(0.001, 10, { log: true, default: 0.1, label: 'initial step size' }),
        maxLeapfrogSteps: int(2, 10_000, { default: 1000, label: 'most leapfrog steps' }),
        jitter: bool({ default: true, label: 'jitter the trajectory' }),
      }),
      plan: (b) => {
        const walkers = num(b.config, 'numWalkers')
        const jitter = flag(b.config, 'jitter')
        const numWarmup = num(b.config, 'numWarmup')
        const eps0 = num(b.config, 'initialStepSize')
        const kernel = (t: CheesParameters): MCMCKernel =>
          cheesKernel(b.um.logDensity, { numChains: walkers, ...t, jitter })
        const options = {
          numChains: walkers,
          numWarmup,
          initialStepSize: eps0,
          targetAcceptance: num(b.config, 'targetAcceptance'),
          learningRate: num(b.config, 'learningRate'),
          maxLeapfrogSteps: num(b.config, 'maxLeapfrogSteps'),
          jitter,
        }
        const shared = cheesAdaptationAlgorithm(b.um.logDensity, new Float64Array(walkers * b.um.dim), options)
        const warm: Warmer<CheesParameters> | undefined =
          numWarmup > 0
            ? {
                init: (start, key) => cheesAdaptationAlgorithm(b.um.logDensity, start, options).init(key),
                step: (w, ctx) => shared.step(w as W, ctx),
                chain: (w) => (w as W).chain,
                info: (w) => ({
                  acceptanceRate: (w as W).chain.acceptProbs.reduce((a, v) => a + v, 0) / walkers,
                  accepted: true,
                  stepSize: (w as W).stepSize,
                  numSteps: (w as W).chain.numSteps,
                }),
                result: (w) => cheesAdaptationResult(w as W),
                toJson: (w) => shared.serialise?.(w as W) ?? null,
                fromJson: (j) => shared.deserialise?.(j),
              }
            : undefined
        return {
          kernelDim: walkers * b.um.dim,
          walkers,
          kernel,
          tuned0: { stepSize: eps0, trajectoryLength: eps0 * 10 },
          ...(warm === undefined ? {} : { warm }),
          tunedToJson: (t) => stateToJson(t),
          tunedFromJson: (j) => stateFromJson(j) as CheesParameters,
          tunedSummary: (t) => ({ numWalkers: walkers, ...t }),
          extraTape: (ctx, states) => {
            ctx.log(
              'chees/trajectoryLength',
              states.map((s) => (s as CheesState).trajectoryLength),
            )
            ctx.log('chees/numSteps', (states[0] as CheesState | undefined)?.numSteps ?? 0)
          },
        } satisfies McmcPlan<CheesParameters>
      },
    },
    model,
    options,
  ) as unknown as CheesMethod
}
