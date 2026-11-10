/**
 * Marginal-likelihood estimators as inference methods, each its own estimator with its own knobs and tape (`evidence/estimate`
 * and `evidence/se` beside what is particular to it): the stepping-stone estimator and thermodynamic integration over the
 * ladder of tempered SMC (one temperature per step), bridge sampling (a slice-sampling chain for the posterior and a fitted
 * Gaussian for the bridge, one of each per step), and importance sampling with a Gaussian proposal fitted by pilot batches, with
 * the Pareto k-hat of its weights. `evidenceEstimators` is a lookup of what exists, with the harmonic mean listed as refused
 * and why; it chooses nothing. Each method reads the evidence so far at any step, so a page sees the estimate settle.
 */

import type { Json } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import {
  type BridgeResult,
  bridgeSamplingFromRatios,
  fitGaussian,
  type GaussianFit,
  gaussianProposal,
  importanceEss,
  logEvidence as importanceLogEvidence,
  type LadderRung,
  numericTarget,
  type SMCState,
  sliceSampler,
  smcAlgorithm,
  steppingStone,
  thermodynamicIntegration,
  weightDiagnostics,
} from '../../sampling/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { constrainDraws } from '../density.ts'
import { type TemperedProblem, temperedPosterior, temperedProblem } from '../engines/smc.ts'
import { type Posterior, weightedPosterior } from '../posterior.ts'
import {
  guard,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  pick,
  posteriorJson,
  programOf,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'
import { nestedMethod } from './nested-method.ts'
import { temperedSmcMethod } from './smc-methods.ts'

const SCHEMES = ['systematic', 'multinomial', 'stratified', 'residual'] as const

/** The check every evidence method here shares: a likelihood, continuous latent sites only (the samplers move on the unconstrained scale). */
function continuousCheck(a: ModelAnalysis): Verdict {
  if (!a.hasLikelihood) return no('observed nodes have no likelihood', ['likelihood'], 'add an observed node')
  const discrete = isProgramAnalysis(a) ? a.discrete.sites : [...a.finiteDiscrete, ...a.unboundedDiscrete]
  if (a.continuous.length === 0) return no('no continuous latent nodes', ['continuous latents'], 'use enumeration')
  if (discrete.length > 0)
    return no(
      `discrete latent nodes (${discrete.join(', ')}) cannot be moved on the unconstrained scale`,
      ['continuous latents only'],
      'sum the discrete nodes out, or use enumeration',
    )
  return yes('continuous latents with a proper prior: the evidence is the integral of prior times likelihood')
}

function problemOf(model: MethodModel): () => TemperedProblem {
  const program = programOf(model)
  let cached: TemperedProblem | undefined
  return () => {
    cached ??= temperedProblem(program, undefined, makeKey(0))
    return cached
  }
}

/** The paths the ladder estimators (stepping stone, thermodynamic integration) log. */
export type LadderEvidencePath =
  | 'evidence/estimate'
  | 'evidence/se'
  | 'evidence/temperature'
  | 'evidence/meanLogLik'
  | 'evidence/ess'
  | 'evidence/snapshot'

/** The state of the ladder estimators: the tempered SMC state, and the temperatures and per-particle log likelihoods of every rung so far. */
export interface LadderState {
  readonly smc: SMCState
  readonly betas: readonly number[]
  readonly logLiks: readonly Float64Array[]
}

function ladderMethod(
  model: MethodModel,
  options: { readonly name?: string },
  spec: {
    readonly name: string
    readonly description: string
    readonly estimator: (rungs: readonly LadderRung[]) => { readonly value: number; readonly standardError: number }
    readonly label: string
  },
): InferenceMethod<LadderState, Config, LadderEvidencePath, ModelAnalysis> {
  const name = options.name ?? spec.name
  const problem = problemOf(model)
  const rungsOf = (s: LadderState): LadderRung[] =>
    s.betas.map((beta, k) => ({ beta, logLikelihoods: s.logLiks[k] as Float64Array }))
  const estimate = (s: LadderState): { value: number; standardError: number } =>
    s.betas.length < 2 ? { value: 0, standardError: Number.NaN } : spec.estimator(rungsOf(s))
  const posterior = (s: LadderState): Posterior => {
    const e = estimate(s)
    return temperedPosterior(problem(), s.smc.particles, {
      logEvidence: e.value,
      status: s.smc.temperature >= 1 ? 'converged' : 'stopped',
      diagnostics: {
        method: spec.label,
        temperature: s.smc.temperature,
        rungs: s.betas.length,
        standardError: Number.isFinite(e.standardError) ? e.standardError : null,
        partial: s.smc.temperature < 1,
      },
    })
  }
  const withError = (s: LadderState): Posterior => {
    const p = posterior(s)
    const e = estimate(s)
    return { ...p, logEvidence: { value: e.value, standardError: e.standardError, method: spec.label } }
  }
  const ensure = guard(name, model, continuousCheck)
  return defineInferenceMethod<LadderState, Config, LadderEvidencePath, ModelAnalysis>({
    name,
    family: 'smc',
    description: spec.description,
    config: dict({
      numParticles: int(2, 1_000_000, { default: 500, label: 'particles' }),
      mcmcSteps: int(0, 1000, { default: 5, label: 'MCMC steps per rung' }),
      targetEss: real(0.05, 0.99, {
        default: 0.9,
        label: 'ladder density (target ESS)',
        description: 'the fraction of particles the ESS is held at between temperatures: higher means more rungs',
      }),
      resampling: choice([...SCHEMES], { default: 'systematic', label: 'resampling' }),
    }),
    check: continuousCheck,
    posterior: withError,
    defaults: { 'evidence/estimate': Log(), 'evidence/se': Log(), 'evidence/temperature': Log() },
    contract: (): Contract<LadderState, LadderEvidencePath> =>
      defineContract({
        'evidence/estimate': scalar({
          description: `the ${spec.label} estimate of the log evidence of the tempered posterior at the current temperature (the evidence at 1)`,
          direction: 'higher',
        }),
        'evidence/se': scalar({
          description: "the Monte Carlo error of the estimate (the rule's discretisation error is not in it)",
          direction: 'lower',
        }),
        'evidence/temperature': scalar({
          description: 'the temperature reached, 0 at the prior and 1 at the posterior',
        }),
        'evidence/meanLogLik': scalar({
          description:
            'the mean log likelihood of the particles at this temperature: the integrand of thermodynamic integration',
        }),
        'evidence/ess': scalar({ description: 'the ESS of the incremental weights before resampling' }),
        'evidence/snapshot': derived<LadderState>({
          cost: 'dear',
          description: 'the particles at the current temperature as JSON with the evidence estimate and its error',
          compute: (s) => posteriorJson(withError(s)),
        }),
      }),
    make: (config): Algorithm<LadderState, LadderEvidencePath> => {
      ensure()
      const p = problem()
      const inner = smcAlgorithm(p.prior, p.logLikelihood, {
        numParticles: num(config, 'numParticles'),
        mcmcSteps: num(config, 'mcmcSteps'),
        targetEss: num(config, 'targetEss'),
        resampling: pick(config, 'resampling') as (typeof SCHEMES)[number],
      })
      return {
        name: `probprog/${spec.name}`,
        config: inner.config,
        init: (key: Key) => {
          ensure()
          const s = inner.init(key)
          return { smc: s, betas: [0], logLiks: [s.logLikelihood] }
        },
        step: (s, ctx) => {
          const next = inner.step(s.smc, ctx as never)
          const state: LadderState = {
            smc: next,
            betas: [...s.betas, next.temperature],
            logLiks: [...s.logLiks, next.logLikelihood],
          }
          const e = estimate(state)
          ctx.log('evidence/estimate', e.value)
          if (Number.isFinite(e.standardError)) ctx.log('evidence/se', e.standardError)
          ctx.log('evidence/temperature', next.temperature)
          ctx.log('evidence/ess', next.ess)
          ctx.log('evidence/meanLogLik', next.logLikelihood.reduce((a, b) => a + b, 0) / next.logLikelihood.length)
          return state
        },
        done: (s) => (s.smc.temperature >= 1 ? 'converged' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as LadderState,
      }
    },
  })
}

/**
 * The stepping-stone estimator of the evidence over the ladder of tempered SMC, as a method with a fine step: one
 * temperature per step, the ladder chosen adaptively to hold the ESS, the log evidence of the tempered posterior so far on the
 * tape with its error, and the particles at the current temperature as the posterior. The ladder is the one of
 * `temperedSmcMethod` (its evidence is this estimator computed on its resampled particles); what this method adds is the
 * estimate over the recorded rungs with its error and the tape of the integrand, and its sibling
 * `thermodynamicIntegrationMethod` the trapezoid integral of the same rungs.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @param options - The name.
 * @returns The method, with `check` and `posterior`.
 * @concept thermodynamic-integration
 * @index estimator Stepping-stone evidence over the adaptive tempering ladder, one temperature per step.
 * @example
 * ```ts
 * import { steppingStoneMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = steppingStoneMethod(model)
 * const s = session(method, { steps: 200, mode: 'eager', config: { numParticles: 300 } })
 * [s.status, Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - (-1.5155)) < 0.3] // => ['converged', true]
 * ```
 */
export function steppingStoneMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<LadderState, Config, LadderEvidencePath, ModelAnalysis> {
  return ladderMethod(model, options, {
    name: 'stepping-stone',
    description: 'stepping-stone evidence over the adaptive tempering ladder, one temperature per step',
    estimator: steppingStone,
    label: 'stepping stone',
  })
}

/**
 * Thermodynamic integration of the evidence over the ladder of tempered SMC, as a method with a fine step: one temperature per
 * step, the mean log likelihood of the particles at each rung integrated over the temperature by the trapezoid rule. It is
 * the same ladder as `steppingStoneMethod` with the older estimator, which has a discretisation bias the stepping stone
 * does not (it vanishes as the ladder is refined: raise `targetEss`).
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @param options - The name.
 * @returns The method, with `check` and `posterior`.
 * @concept thermodynamic-integration
 * @index estimator Thermodynamic integration of the mean log likelihood over the tempering ladder.
 * @example
 * ```ts
 * import { thermodynamicIntegrationMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = thermodynamicIntegrationMethod(model)
 * const s = session(method, { steps: 200, mode: 'eager', config: { numParticles: 300, targetEss: 0.8 } })
 * [s.status, Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - (-1.5155)) < 0.3] // => ['converged', true]
 * ```
 */
export function thermodynamicIntegrationMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<LadderState, Config, LadderEvidencePath, ModelAnalysis> {
  return ladderMethod(model, options, {
    name: 'thermodynamic-integration',
    description: 'thermodynamic integration of the mean log likelihood over the adaptive tempering ladder',
    estimator: thermodynamicIntegration,
    label: 'thermodynamic integration',
  })
}

/** The paths the bridge sampling method logs. */
export type BridgePath =
  | 'evidence/estimate'
  | 'evidence/se'
  | 'evidence/draws'
  | 'evidence/posteriorEss'
  | 'evidence/logPosterior'
  | 'evidence/snapshot'

/** The state of {@link bridgeSamplingMethod}. */
export interface BridgeState {
  readonly t: number
  /** The chain's current point and its log posterior. */
  readonly position: Float64Array
  readonly logDensity: number
  /** The draws after warm-up, row-major `[m, dim]` (the first `fitDraws` fit the proposal). */
  readonly draws: Float64Array
  /** The proposal, once fitted. */
  readonly fit: GaussianFit | undefined
  /** The log ratios `log p~ - log q` at the posterior draws and at the proposal draws taken after the fit. */
  readonly l1: Float64Array
  readonly l2: Float64Array
}

/** The knobs of {@link bridgeSamplingMethod} (the budget and snapshot knobs are added to them). */
export const bridgeKnobs: DictSpace = dict({
  numWarmup: int(0, 1_000_000, {
    default: 100,
    label: 'warm-up sweeps',
    description: 'slice-sampling sweeps discarded before the draws',
  }),
  fitDraws: int(8, 1_000_000, {
    default: 200,
    label: 'draws for the proposal',
    description: 'posterior draws used to fit the Gaussian proposal',
  }),
  numDraws: int(8, 100_000_000, {
    default: 1000,
    label: 'bridge draws',
    description: 'posterior and proposal draws after the fit; the run ends after this many',
  }),
  inflate: real(0.1, 100, {
    default: 1,
    label: 'proposal inflation',
    description: 'a factor on the fitted covariance',
  }),
})

/**
 * Bridge sampling of the evidence as a method with a fine step: a slice-sampling chain on the unconstrained posterior (a
 * sweep per step) supplies the posterior draws, a Gaussian fitted to the first of them is the proposal, and each later step
 * adds one posterior draw and one proposal draw to the optimal-bridge iteration, so the estimate and its error are available
 * at every step after the fit and tighten as the run goes on. The tape has the estimate, its error, the draws and the
 * effective size of the posterior draws; the posterior is the chain. It suits a unimodal, near-Gaussian posterior on
 * the unconstrained scale; where the Gaussian misses the posterior (several modes, heavy skew) the error is an underestimate.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @param options - The name.
 * @returns The method, with `check` and `posterior`.
 * @concept bridge-sampling
 * @index estimator Bridge sampling evidence: slice-sampling posterior draws and a fitted Gaussian, one pair per step.
 * @example
 * ```ts
 * import { bridgeSamplingMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = bridgeSamplingMethod(model)
 * const s = session(method, { steps: 2000, mode: 'eager', config: { numDraws: 500 } })
 * [s.status, Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - (-1.5155)) < 0.2] // => ['completed', true]
 * ```
 */
export function bridgeSamplingMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<BridgeState, Config, BridgePath, ModelAnalysis> {
  const name = options.name ?? 'bridge-sampling'
  const problem = problemOf(model)
  const result = (s: BridgeState): BridgeResult | undefined =>
    s.l1.length >= 8 ? bridgeSamplingFromRatios(s.l1, s.l2) : undefined
  const posterior = (s: BridgeState, config: Config): Posterior => {
    const p = problem()
    const dim = p.prior.dim
    const r = result(s)
    const m = s.draws.length / dim
    const done = s.t >= num(config, 'numWarmup') + num(config, 'fitDraws') + num(config, 'numDraws')
    if (m < 2)
      return weightedPosterior({}, [], {
        engine: 'bridge-sampling',
        status: 'failed',
        diagnostics: { reason: 'no draws yet' },
      })
    return {
      ...temperedPosterior(p, s.draws, {
        logEvidence: r?.logEvidence ?? Number.NaN,
        status: done ? 'completed' : 'stopped',
        diagnostics: {
          method: 'bridge sampling',
          draws: m,
          bridgeDraws: s.l1.length,
          iterations: r?.iterations ?? null,
          converged: r?.converged ?? null,
          posteriorEss: r?.posteriorEss ?? null,
          partial: !done,
        },
      }),
      ...(r === undefined
        ? {}
        : { logEvidence: { value: r.logEvidence, standardError: r.standardError, method: 'bridge sampling' } }),
    }
  }
  const ensure = guard(name, model, continuousCheck)
  return defineInferenceMethod<BridgeState, Config, BridgePath, ModelAnalysis>({
    name,
    family: 'mcmc',
    description: 'bridge sampling: slice-sampling posterior draws and a fitted Gaussian, one pair per step',
    config: bridgeKnobs,
    check: continuousCheck,
    posterior,
    defaults: { 'evidence/estimate': Log(), 'evidence/se': Log() },
    contract: (config): Contract<BridgeState, BridgePath> =>
      defineContract({
        'evidence/estimate': scalar({
          description: 'the bridge sampling estimate of the log evidence (after the proposal is fitted)',
          direction: 'higher',
        }),
        'evidence/se': scalar({
          description: "the delta-method error of the estimate, with the posterior draws' autocorrelation",
          direction: 'lower',
        }),
        'evidence/draws': scalar({
          description: 'bridge draws taken so far (each a posterior draw and a proposal draw)',
        }),
        'evidence/posteriorEss': scalar({
          description: 'the effective size of the posterior draws the error used',
          direction: 'higher',
        }),
        'evidence/logPosterior': scalar({ description: "the unnormalised log posterior at the chain's current point" }),
        'evidence/snapshot': derived<BridgeState>({
          cost: 'dear',
          description: 'the chain so far as JSON with the evidence estimate and its error',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<BridgeState, BridgePath> => {
      ensure()
      const warm = num(config, 'numWarmup')
      const fitN = num(config, 'fitDraws')
      const total = warm + fitN + num(config, 'numDraws')
      const inflate = num(config, 'inflate')
      const kernelFor = (fit: GaussianFit | undefined): ReturnType<typeof sliceSampler> => {
        const p = problem()
        const target = numericTarget((x) => {
          const lp = p.prior.logProb(x)
          return lp === Number.NEGATIVE_INFINITY ? lp : lp + p.logLikelihood(x)
        }, p.prior.dim)
        const dim = p.prior.dim
        const width =
          fit === undefined
            ? 1
            : Float64Array.from(
                { length: dim },
                (_, a) => 2 * Math.sqrt(Math.max(1e-12, fit.covariance[a * dim + a] ?? 1)),
              )
        return sliceSampler(target, { width })
      }
      const logTarget = (x: Float64Array): number => {
        const p = problem()
        const lp = p.prior.logProb(x)
        return lp === Number.NEGATIVE_INFINITY ? lp : lp + p.logLikelihood(x)
      }
      return {
        name: 'probprog/bridge-sampling',
        config: { numWarmup: warm, fitDraws: fitN, numDraws: num(config, 'numDraws') },
        init: () => {
          ensure()
          const x = problem().prior.sample(foldIn(key, 0))
          return {
            t: 0,
            position: x,
            logDensity: logTarget(x),
            draws: new Float64Array(0),
            fit: undefined,
            l1: new Float64Array(0),
            l2: new Float64Array(0),
          }
        },
        step: (s, ctx) => {
          const dim = problem().prior.dim
          const k = kernelFor(s.fit)
          const r = k.step(ctx.key, { position: s.position, logDensity: s.logDensity })
          let draws = s.draws
          let fit = s.fit
          let l1 = s.l1
          let l2 = s.l2
          const t = s.t + 1
          if (t > warm) {
            draws = new Float64Array(s.draws.length + dim)
            draws.set(s.draws)
            draws.set(r.state.position, s.draws.length)
          }
          const kept = draws.length / dim
          if (fit === undefined && kept >= fitN) fit = fitGaussian(draws, dim)
          else if (s.fit !== undefined) {
            const q = gaussianProposal(s.fit, { inflate })
            const y = q.sample(foldIn(ctx.key, 7))
            const ratio = (x: Float64Array): number => {
              const v = logTarget(x) - q.logProb(x)
              return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
            }
            l1 = new Float64Array(s.l1.length + 1)
            l1.set(s.l1)
            l1[s.l1.length] = r.state.logDensity - q.logProb(r.state.position)
            l2 = new Float64Array(s.l2.length + 1)
            l2.set(s.l2)
            l2[s.l2.length] = ratio(y)
          }
          const next: BridgeState = {
            t,
            position: r.state.position,
            logDensity: r.state.logDensity,
            draws,
            fit,
            l1,
            l2,
          }
          ctx.log('evidence/logPosterior', r.state.logDensity)
          ctx.log('evidence/draws', l1.length)
          const e = result(next)
          if (e !== undefined) {
            ctx.log('evidence/estimate', e.logEvidence)
            ctx.log('evidence/se', e.standardError)
            ctx.log('evidence/posteriorEss', e.posteriorEss)
          }
          return next
        },
        done: (s) => (s.t >= total ? 'completed' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as BridgeState,
      }
    },
  })
}

/** The paths the importance evidence method logs. */
export type ImportanceEvidencePath =
  | 'evidence/estimate'
  | 'evidence/se'
  | 'evidence/draws'
  | 'evidence/ess'
  | 'evidence/paretoK'
  | 'evidence/snapshot'

/** The state of {@link importanceEvidenceMethod}. */
export interface ImportanceEvidenceState {
  /** Batches taken. */
  readonly t: number
  /** The Gaussian proposal fitted by the pilot batches (absent while the prior is the proposal). */
  readonly fit: GaussianFit | undefined
  /** The pilot draws and their log weights (kept to refit the proposal). */
  readonly pilotDraws: Float64Array
  readonly pilotLogWeights: Float64Array
  /** The main draws (row-major `[m, dim]`) and their log weights: what the estimate and the posterior use. */
  readonly draws: Float64Array
  readonly logWeights: Float64Array
}

/** The knobs of {@link importanceEvidenceMethod} (the budget and snapshot knobs are added to them). */
export const importanceEvidenceKnobs: DictSpace = dict({
  proposal: choice(['adaptive-normal', 'prior'], {
    default: 'adaptive-normal',
    label: 'proposal',
    description: 'the prior, or a Gaussian refitted after each pilot batch from the weighted draws so far',
  }),
  batchSize: int(2, 1_000_000, { default: 200, label: 'batch size', description: 'draws per step' }),
  pilotBatches: int(0, 1000, {
    default: 4,
    label: 'pilot batches',
    description: 'steps that refit the proposal; their draws are not in the estimate',
  }),
  numBatches: int(1, 1_000_000, {
    default: 20,
    label: 'estimate batches',
    description: 'steps that draw from the fitted proposal; the run ends after them',
  }),
  inflate: real(0.1, 100, {
    default: 1.5,
    label: 'proposal inflation',
    description: 'a factor on the fitted covariance: above 1 thickens the tails',
  }),
  defensive: real(0, 0.9, {
    default: 0.05,
    label: 'defensive weight',
    description: 'the weight of the prior in the proposal, which bounds the importance weights',
  }),
  smooth: bool({
    default: false,
    label: 'Pareto smoothing',
    description: 'use Pareto-smoothed weights for the posterior',
  }),
})

/**
 * Importance sampling of the evidence as a method with a fine step: a batch of draws per step from a proposal that the first
 * (pilot) batches fit to the weighted draws so far (a Gaussian on the unconstrained scale, inflated, with a defensive share
 * of the prior), then batches from the fixed proposal whose weights give the estimate (unbiased for the evidence given the
 * proposal) and its error. The tape carries the estimate, the error, Kish's ESS and the Pareto k-hat of the weights: a
 * k-hat above 0.7 says the estimate cannot be trusted, whatever its stated error. `proposal: 'prior'` is plain importance
 * sampling from the prior, which fails exactly when the posterior is narrow in the prior's volume.
 *
 * @param model - A bound spec, or a program closed over its data (continuous latent sites).
 * @param options - The name.
 * @returns The method, with `check` and `posterior`.
 * @concept importance-sampling
 * @index estimator Importance-sampling evidence with a pilot-fitted Gaussian proposal and Pareto k-hat.
 * @example
 * ```ts
 * import { importanceEvidenceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * const method = importanceEvidenceMethod(model)
 * const s = session(method, { steps: 100, mode: 'eager' })
 * [s.status, Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - (-1.5155)) < 0.1] // => ['completed', true]
 * ```
 */
export function importanceEvidenceMethod(
  model: MethodModel,
  options: { readonly name?: string } = {},
): InferenceMethod<ImportanceEvidenceState, Config, ImportanceEvidencePath, ModelAnalysis> {
  const name = options.name ?? 'importance-evidence'
  const problem = problemOf(model)
  const proposalOf = (
    config: Config,
    fit: GaussianFit | undefined,
  ): ReturnType<typeof gaussianProposal> | ReturnType<typeof problem>['prior'] => {
    const p = problem()
    return fit === undefined
      ? p.prior
      : gaussianProposal(fit, {
          inflate: num(config, 'inflate'),
          defensive: { prior: p.prior, weight: num(config, 'defensive') },
        })
  }
  const posterior = (s: ImportanceEvidenceState, config: Config): Posterior => {
    const n = s.logWeights.length
    if (n < 2)
      return weightedPosterior({}, [], {
        engine: 'importance-evidence',
        status: 'failed',
        diagnostics: { reason: 'no draws yet' },
      })
    const p = problem()
    const ev = importanceLogEvidence(s.logWeights)
    let diag: Record<string, Json> = { method: 'importance sampling', draws: n, ess: importanceEss(s.logWeights) }
    try {
      const d = weightDiagnostics(s.logWeights)
      diag = { ...diag, paretoK: Number.isFinite(d.paretoK) ? d.paretoK : null, reliable: d.reliable }
    } catch {
      diag = { ...diag, paretoK: null }
    }
    const done = s.t >= num(config, 'pilotBatches') + num(config, 'numBatches')
    return weightedPosterior(constrainDraws(p.um, s.draws), s.logWeights, {
      engine: 'importance-evidence',
      status: done ? 'completed' : 'stopped',
      logEvidence: { value: ev.value, standardError: ev.standardError ?? Number.NaN, method: 'importance sampling' },
      diagnostics: diag,
    })
  }
  const ensure = guard(name, model, continuousCheck)
  return defineInferenceMethod<ImportanceEvidenceState, Config, ImportanceEvidencePath, ModelAnalysis>({
    name,
    family: 'smc',
    description: 'importance-sampling evidence with a pilot-fitted Gaussian proposal, with the Pareto k-hat',
    config: importanceEvidenceKnobs,
    check: continuousCheck,
    posterior,
    defaults: { 'evidence/estimate': Log(), 'evidence/se': Log(), 'evidence/paretoK': Log() },
    contract: (config): Contract<ImportanceEvidenceState, ImportanceEvidencePath> =>
      defineContract({
        'evidence/estimate': scalar({
          description: 'the importance-sampling estimate of the log evidence from the batches after the pilot',
          direction: 'higher',
        }),
        'evidence/se': scalar({ description: 'the delta-method error of the log evidence', direction: 'lower' }),
        'evidence/draws': scalar({ description: 'draws in the estimate so far' }),
        'evidence/ess': scalar({ description: "Kish's effective sample size of the weights", direction: 'higher' }),
        'evidence/paretoK': scalar({
          description: 'the Pareto k-hat of the weights: above 0.7 the estimate is not to be trusted',
          direction: 'lower',
        }),
        'evidence/snapshot': derived<ImportanceEvidenceState>({
          cost: 'dear',
          description: 'the weighted posterior of the draws so far as JSON with the evidence estimate and k-hat',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<ImportanceEvidenceState, ImportanceEvidencePath> => {
      ensure()
      const batch = num(config, 'batchSize')
      const pilot = pick(config, 'proposal') === 'prior' ? 0 : num(config, 'pilotBatches')
      const total = pilot + num(config, 'numBatches')
      const algorithm: Algorithm<ImportanceEvidenceState, ImportanceEvidencePath> = {
        name: 'probprog/importance-evidence',
        config: { batchSize: batch, pilotBatches: pilot, numBatches: num(config, 'numBatches') },
        init: () => {
          ensure()
          return {
            t: 0,
            fit: undefined,
            pilotDraws: new Float64Array(0),
            pilotLogWeights: new Float64Array(0),
            draws: new Float64Array(0),
            logWeights: new Float64Array(0),
          }
        },
        step: (s, ctx) => {
          const p = problem()
          const dim = p.prior.dim
          const q = proposalOf(config, s.fit)
          const xs = q.sampleMany(foldIn(key, s.t), batch)
          const lq = q.logProbMany(xs)
          const lw = new Float64Array(batch)
          for (let i = 0; i < batch; i++) {
            const x = xs.subarray(i * dim, (i + 1) * dim)
            const lp = p.prior.logProb(x)
            const v = lp === Number.NEGATIVE_INFINITY ? lp : lp + p.logLikelihood(x) - (lq[i] ?? 0)
            lw[i] = Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
          }
          const concat = (a: Float64Array, b: Float64Array): Float64Array => {
            const out = new Float64Array(a.length + b.length)
            out.set(a)
            out.set(b, a.length)
            return out
          }
          let next: ImportanceEvidenceState
          if (s.t < pilot) {
            const pilotDraws = concat(s.pilotDraws, xs)
            const pilotLogWeights = concat(s.pilotLogWeights, lw)
            let fit = s.fit
            // Refit from the weighted pilot draws so far, when there are weights to fit with.
            if (pilotLogWeights.some((v) => Number.isFinite(v)) && importanceEss(pilotLogWeights) >= 3) {
              fit = fitGaussian(pilotDraws, dim, pilotLogWeights)
            }
            next = { ...s, t: s.t + 1, fit, pilotDraws, pilotLogWeights }
          } else next = { ...s, t: s.t + 1, draws: concat(s.draws, xs), logWeights: concat(s.logWeights, lw) }
          ctx.log('evidence/draws', next.logWeights.length)
          if (next.logWeights.length >= 2 && next.logWeights.some((v) => Number.isFinite(v))) {
            const ev = importanceLogEvidence(next.logWeights)
            ctx.log('evidence/estimate', ev.value)
            if (ev.standardError !== undefined && Number.isFinite(ev.standardError))
              ctx.log('evidence/se', ev.standardError)
            ctx.log('evidence/ess', importanceEss(next.logWeights))
            if (next.logWeights.length >= 25) {
              const d = weightDiagnostics(next.logWeights)
              if (Number.isFinite(d.paretoK)) ctx.log('evidence/paretoK', d.paretoK)
            }
          }
          return next
        },
        done: (s) => (s.t >= total ? 'completed' : undefined),
        serialise: (s) => stateToJson(s),
        deserialise: (j: Json) => stateFromJson(j) as ImportanceEvidenceState,
      }
      return algorithm
    },
  })
}

/** One entry of {@link evidenceEstimators}. */
export interface EvidenceEstimatorEntry {
  /** What the estimator does and what it needs, in a sentence. */
  readonly description: string
  /** Why the estimator is refused, when it is: it is listed so that it is known, and has no `make`. */
  readonly refused?: string
  /** Make the method for a model. */
  readonly make?: (
    model: MethodModel,
    options?: { readonly name?: string },
  ) => InferenceMethod<never, Config, string, ModelAnalysis>
}

/**
 * The evidence estimators that exist, by name, as a lookup: what an app lists when it offers the user a choice, with the
 * harmonic mean included as refused and the reason. It selects nothing: no estimator is better in general (bridge sampling
 * wants a unimodal near-Gaussian posterior, the ladder estimators and nested sampling cost more but do not, importance
 * sampling is cheapest and fails on narrow posteriors, which its k-hat reports).
 *
 * @concept marginal-likelihood
 * @index table Evidence estimators by name: bridge, stepping stone, thermodynamic, importance, nested, tempered SMC.
 * @example
 * ```ts
 * import { evidenceEstimators } from 'omai-engine/probprog/inference'
 * [Object.keys(evidenceEstimators).includes('bridge-sampling'), evidenceEstimators['harmonic-mean']?.make === undefined] // => [true, true]
 * ```
 */
export const evidenceEstimators: Readonly<Record<string, EvidenceEstimatorEntry>> = {
  'bridge-sampling': {
    description:
      'bridge sampling from slice-sampling posterior draws and a fitted Gaussian; unimodal, near-Gaussian posteriors',
    make: bridgeSamplingMethod as never,
  },
  'stepping-stone': {
    description:
      'stepping-stone estimator over the adaptive tempering ladder; any continuous posterior, costs a ladder of MCMC moves',
    make: steppingStoneMethod as never,
  },
  'thermodynamic-integration': {
    description:
      'thermodynamic integration of the mean log likelihood over the tempering ladder; has a discretisation bias',
    make: thermodynamicIntegrationMethod as never,
  },
  'importance-evidence': {
    description:
      'importance sampling with a pilot-fitted Gaussian proposal and the Pareto k-hat of its weights; cheapest, fails on narrow or multimodal posteriors',
    make: importanceEvidenceMethod as never,
  },
  nested: {
    description:
      'nested sampling: the evidence with an error bar and the posterior; costs live points times information',
    make: nestedMethod as never,
  },
  'tempered-smc': {
    description:
      'the evidence of adaptive-tempering SMC (the stepping stone over resampled particles), with the posterior',
    make: temperedSmcMethod as never,
  },
  'harmonic-mean': {
    description: 'the harmonic mean of the likelihood over posterior draws',
    refused:
      'the harmonic mean estimator has infinite variance for the usual models, is typically biased far upwards by a few draws, and its error estimate is worthless (Neal 1994); use bridge sampling, nested sampling or the stepping stone. `harmonicMeanEvidence` of `omai-engine/sampling` computes it only when the infinite variance is acknowledged, to show the failure',
  },
}
