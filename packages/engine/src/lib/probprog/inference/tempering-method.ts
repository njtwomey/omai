/**
 * Parallel tempering as an inference method on programs and bound specs: replicas of a chain at a ladder of inverse
 * temperatures (the model's density raised to the power beta), the hot ones crossing the barriers between modes that the cold
 * one cannot, with swaps between neighbours, over a kernel of the user's choice (HMC, NUTS, MALA, Barker or a random walk).
 * The cold replica's draws are the posterior; the tape shows the ladder, the swap rate of every adjacent pair and the round
 * trips of the walkers between the hottest and the coldest replica, which say whether the ladder is good. Warm-up adapts the
 * cold replica's step size (and mass matrix) with the library's window adaptation, and the hotter replicas take a step size scaled
 * by the temperature.
 */

import {
  type HamiltonianParameters,
  type MCMCKernel,
  type MCMCState,
  type PTState,
  parallelTemperingKernel,
  temperatureLadder,
} from '../../sampling/index.ts'
import { bool, choice, dict, int, real } from '../../space/index.ts'
import { type Config, defineContract, scalar, stateFromJson, stateToJson, tensor } from '../../trace/index.ts'
import { type McmcPath, type McmcPlan, type McmcRunState, mcmcMethodOf } from './_mcmc-driver.ts'
import { hamiltonianFamilies, windowWarmer } from './_mcmc-families.ts'
import { flag, type MethodModel, type ModelAnalysis, num, pick } from './_support.ts'
import type { InferenceMethod } from './method.ts'

/** The paths parallel tempering adds to the shared ones. */
export type TemperingPath =
  | 'pt/temperature'
  | 'pt/swapRate'
  | 'pt/swapAccepted'
  | 'pt/replicaLogDensity'
  | 'pt/roundTrips'

/** The type of the tempering method. */
export type TemperingMethod = InferenceMethod<McmcRunState, Config, McmcPath | TemperingPath, ModelAnalysis>

/**
 * Parallel tempering as a method: `numTemperatures` replicas at inverse temperatures from 1 down to `minBeta` (geometric or
 * linear), each moved by the chosen kernel (`hmc`, `nuts`, `mala`, `barker`, `rwm`) on the model's density raised to the power
 * of its temperature, with swaps between neighbours every `swapEvery` steps on the even and the odd pairs in turn (or one
 * random pair). The posterior is the cold replica's draws. Warm-up adapts the cold replica's step size and mass matrix by
 * Stan's windows; the hotter replicas take the step size scaled by their temperature. The tape adds the ladder
 * (`pt/temperature`), the swap rate of every adjacent pair (`pt/swapRate`) and the last swap outcomes, the log density of every
 * replica and the round trips of walkers between the hottest and the coldest replica (`pt/roundTrips`). The whole density is
 * tempered: the hot end flattens the prior as well as the likelihood, so the model needs a proper prior.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `parallel-tempering`).
 * @returns The method.
 * @concept parallel-tempering
 * @index estimator Parallel tempering (replica exchange) over HMC, NUTS, MALA, Barker or a random walk, with swap diagnostics.
 * @example
 * ```ts
 * import { temperingMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(temperingMethod(model), { config: { numChains: 1, numWarmup: 30, numDraws: 20, numTemperatures: 3 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function temperingMethod(model: MethodModel, options: { readonly name?: string } = {}): TemperingMethod {
  return mcmcMethodOf<HamiltonianParameters, TemperingPath>(
    {
      name: 'parallel-tempering',
      description: 'parallel tempering over a kernel, one transition of every replica and the swaps per step',
      extraPaths: defineContract({
        'pt/temperature': tensor({
          role: 'output',
          shape: ['temperatures'],
          description: 'the inverse temperature of each replica',
        }),
        'pt/swapRate': tensor({
          role: 'output',
          shape: ['chains', 'pairs'],
          description: 'the fraction of proposed swaps accepted so far, for each adjacent pair of replicas',
        }),
        'pt/swapAccepted': tensor({
          role: 'output',
          shape: ['chains', 'pairs'],
          description:
            'the last swap proposal of each adjacent pair: 1 accepted, 0 rejected, NaN not proposed this step',
        }),
        'pt/replicaLogDensity': tensor({
          role: 'output',
          shape: ['chains', 'temperatures'],
          description: "the target's log density at each replica (not tempered)",
        }),
        'pt/roundTrips': scalar({
          description: 'trips of a walker from the hottest to the coldest replica, all chains, so far',
        }),
      }) as never,
      knobs: dict({
        targetAcceptance: real(0, 0.99, {
          default: 0,
          label: 'target acceptance',
          description:
            "of the cold replica; 0: the kernel's usual (0.8 HMC and NUTS, 0.574 MALA, 0.4 Barker, 0.234 random walk)",
        }),
        dense: bool({ default: false, label: 'dense mass matrix' }),
        kernel: choice(['hmc', 'nuts', 'mala', 'barker', 'rwm'], { default: 'hmc', label: 'kernel' }),
        numSteps: int(1, 1000, { default: 10, label: 'HMC steps' }),
        maxTreeDepth: int(1, 15, { default: 8, label: 'maximum tree depth' }),
        numTemperatures: int(2, 64, { default: 5, label: 'temperatures' }),
        minBeta: real(0.001, 0.95, { log: true, default: 0.05, label: 'hottest inverse temperature' }),
        ladder: choice(['geometric', 'linear'], { default: 'geometric', label: 'ladder' }),
        swapEvery: int(1, 1000, { default: 1, label: 'swap interval' }),
        swapScheme: choice(['even-odd', 'random'], { default: 'even-odd', label: 'swap scheme' }),
      }),
      plan: (b) => {
        const dim = b.um.dim
        const c = b.config
        const shape = { numSteps: num(c, 'numSteps'), maxTreeDepth: num(c, 'maxTreeDepth') }
        const inner = hamiltonianFamilies[pick(c, 'kernel')] ?? hamiltonianFamilies['hmc']
        if (inner === undefined) throw new Error('parallel-tempering: no inner kernel')
        const betas = temperatureLadder(
          num(c, 'numTemperatures'),
          num(c, 'minBeta'),
          pick(c, 'ladder') === 'linear' ? 'linear' : 'geometric',
        )
        const kernelFor = (p: HamiltonianParameters): MCMCKernel =>
          parallelTemperingKernel(b.um.logDensity, {
            betas,
            kernelFor: (target, beta) =>
              inner.make(target, { ...p, stepSize: p.stepSize / beta ** inner.power }, dim, shape),
            swapEvery: num(c, 'swapEvery'),
            swapScheme: pick(c, 'swapScheme') === 'random' ? 'random' : 'even-odd',
          })
        const dense = flag(c, 'dense')
        const goal = num(c, 'targetAcceptance') > 0 ? num(c, 'targetAcceptance') : inner.accept
        const identity = dense
          ? Float64Array.from({ length: dim * dim }, (_, k) => (k % (dim + 1) === 0 ? 1 : 0))
          : new Float64Array(dim).fill(1)
        return {
          kernelDim: dim,
          walkers: 1,
          kernel: kernelFor,
          tuned0: { stepSize: 1, inverseMassMatrix: identity },
          warm: windowWarmer(dim, kernelFor, { numWarmup: num(c, 'numWarmup'), dense, targetAcceptance: goal }),
          tunedToJson: (t) => stateToJson(t),
          tunedFromJson: (j) => stateFromJson(j) as HamiltonianParameters,
          tunedSummary: (t) => ({ stepSize: t.stepSize, inverseMassMatrix: Array.from(t.inverseMassMatrix), betas }),
          extraTape: (ctx, states: readonly MCMCState[]) => {
            const pt = states as readonly PTState[]
            ctx.log('pt/temperature', Array.from(betas))
            ctx.log(
              'pt/swapRate',
              pt.flatMap((s) => Array.from(s.swapSuccesses, (a, i) => a / Math.max(1, s.swapAttempts[i] ?? 0))),
            )
            ctx.log(
              'pt/swapAccepted',
              pt.flatMap((s) => Array.from(s.swapAccepted)),
            )
            ctx.log(
              'pt/replicaLogDensity',
              pt.flatMap((s) => s.replicas.map((r, i) => r.logDensity / (s.betas[i] ?? 1))),
            )
            ctx.log(
              'pt/roundTrips',
              pt.reduce((a, s) => a + s.roundTrips, 0),
            )
          },
        } satisfies McmcPlan<HamiltonianParameters>
      },
    },
    model,
    options,
  ) as unknown as TemperingMethod
}
