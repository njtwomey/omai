/**
 * Exact inference for discrete-state chain programs (hidden Markov models written as a markov loop): the program is
 * turned into a chain of potentials, and forward-backward gives posterior marginals and the evidence, while forward
 * filtering backward sampling gives exact joint draws. Also the engine that offers it to the planner.
 */

import { OmaiError } from '../../base/index.ts'
import type { Distribution } from '../../dists/index.ts'
import { backwardSample, type ChainModel, forwardBackward, viterbi } from '../../pgm/index.ts'
import type { Key } from '../../random/index.ts'
import {
  asTensor,
  fromStorage,
  primalOf,
  type Tensor,
  type TensorLike,
  toArray,
  toFlat,
  transpose,
  zeros,
} from '../../tensor/index.ts'
import { type NamedFactor, productOver, traceFactors } from '../_contract.ts'
import { shiftedCategorical } from '../_discrete.ts'
import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { enumerate } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** A discrete-state chain program as a pgm chain: the step sites, their states and the log potentials. */
export interface ChainForm {
  /** The latent site of each step. */
  readonly sites: readonly string[]
  readonly low: number
  readonly states: number
  readonly chain: ChainModel
  /** log of the factors without latent sites. */
  readonly constant: number
}

function chainProblem(a: ProgramAnalysis): string | undefined {
  if (a.markov.length !== 1) return 'the program needs exactly one markov loop'
  const loop = a.markov[0]
  if (loop === undefined) return 'no markov loop'
  if (loop.history !== 1) return 'the markov loop must have history 1'
  if (loop.outsideLatents.length > 0) return `latent sites outside the loop (${loop.outsideLatents.join(', ')})`
  if (a.continuous.length > 0) return 'continuous latent sites'
  const latents = a.sites.filter((s) => s.role === 'latent')
  const steps = new Map<number, number>()
  for (const s of latents) {
    if (s.cardinality === undefined) return `site ${s.name} is not finite and discrete`
    if (s.elements !== 1) return `site ${s.name} is not a single variable per step`
    const t = s.markov?.step ?? -1
    steps.set(t, (steps.get(t) ?? 0) + 1)
  }
  if ([...steps.values()].some((c) => c !== 1)) return 'each step needs exactly one latent site'
  if (new Set(latents.map((s) => s.cardinality)).size > 1) return 'every step needs the same number of states'
  return undefined
}

/**
 * Turn a discrete-state chain program (one markov loop, one finite discrete latent site per step depending on the
 * previous one, observations depending on the current one) into a pgm chain model: the initial, transition and
 * emission log potentials, read off the site log densities with every state enumerated.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param analysis - Its analysis (default: analysed here).
 * @returns The chain form.
 * @throws OmaiError with code `value` when the program is not such a chain.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { chainForm, markov, observe, sample } from 'omai-engine/probprog'
 * import { bernoulli, categorical } from 'omai-engine/dists'
 * import { take, tensor, type Value } from 'omai-engine/tensor'
 * const hmm = () => markov('t', 3, (t, z: Value | undefined) => {
 *   const s = sample(`z${t}`, categorical({ probs: z === undefined ? [0.5, 0.5] : take(tensor([[0.9, 0.1], [0.1, 0.9]]), z) }))
 *   observe(`y${t}`, bernoulli({ probs: take(tensor([0.2, 0.8]), s) }), 1)
 *   return s
 * }, undefined as Value | undefined)
 * chainForm(hmm, undefined).chain.logEmission.length // => 3
 * ```
 */
export function chainForm<D = void>(model: Model<D>, data: D, analysis?: ProgramAnalysis): ChainForm {
  const a = analysis ?? analyse(model, data)
  const problem = chainProblem(a)
  if (problem !== undefined) throw new OmaiError('value', `chainForm: ${problem}`)
  const latents = a.sites.filter((s) => s.role === 'latent')
  latents.sort((x, y) => (x.markov?.step ?? 0) - (y.markov?.step ?? 0))
  const sites = latents.map((s) => s.name)
  const first = latents[0]
  const states = first?.cardinality ?? 0
  const support = first?.support
  const low = support?.kind === 'integerInterval' ? support.low : 0
  const step = new Map(sites.map((n, t) => [n, t]))
  const r = logJoint(enumerate(model), data)
  const depth = a.maxPlateNesting
  const { factors, scales } = traceFactors(r.trace, depth)
  const T = sites.length
  const logInitial = toFlat(zeros([states]))
  const logTransition = toArray(zeros([Math.max(0, T - 1), states, states])) as number[][][]
  const logEmission = toArray(zeros([T, states])) as number[][]
  let constant = 0
  for (const raw of factors) {
    const f: NamedFactor = productOver(raw, raw.plates, scales)
    const vars = f.dims
    const values = toFlat(primalOf(f.value))
    if (vars.length === 0) {
      constant += values[0] ?? 0
      continue
    }
    if (vars.length === 1) {
      const t = step.get(vars[0] ?? '') ?? 0
      const target = raw.site === sites[0] ? logInitial : (logEmission[t] as number[])
      values.forEach((v, k) => {
        target[k] = (target[k] ?? 0) + v
      })
      continue
    }
    if (vars.length === 2) {
      const [u, v] = vars as [string, string]
      const tu = step.get(u) ?? 0
      const tv = step.get(v) ?? 0
      if (Math.abs(tu - tv) !== 1) throw new OmaiError('value', `chainForm: a factor links steps ${tu} and ${tv}`)
      const later = Math.max(tu, tv)
      const m = tu < tv ? values : toFlat(primalOf(transpose(primalOf(f.value))))
      const A = logTransition[later - 1] as number[][]
      for (let i = 0; i < states; i++)
        for (let j = 0; j < states; j++) (A[i] as number[])[j] = (A[i]?.[j] ?? 0) + (m[i * states + j] ?? 0)
      continue
    }
    throw new OmaiError('value', `chainForm: a factor of ${raw.site ?? '?'} involves ${vars.length} steps`)
  }
  return { sites, low, states, chain: { logInitial, logTransition, logEmission }, constant }
}

/** Options of {@link chainInference}. */
export interface ChainInferenceOptions {
  readonly analysis?: ProgramAnalysis
  readonly tracer?: EngineRunOptions['tracer']
  readonly policy?: EngineRunOptions['policy']
}

/**
 * Exact inference for discrete-state chain programs (HMM-like) by forward-backward (posterior marginals and the
 * evidence), forward filtering and backward sampling (exact joint draws) and Viterbi (the MAP path, in diagnostics),
 * all from `pgm`, with one traced message per step.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The analysis, tracer and policy.
 * @returns The exact posterior.
 * @concept hidden-markov-models
 * @example
 * ```ts
 * import { chainInference, markov, observe, sample } from 'omai-engine/probprog'
 * import { bernoulli, categorical } from 'omai-engine/dists'
 * import { take, tensor, type Value } from 'omai-engine/tensor'
 * const hmm = () => markov('t', 3, (t, z: Value | undefined) => {
 *   const s = sample(`z${t}`, categorical({ probs: z === undefined ? [0.5, 0.5] : take(tensor([[0.9, 0.1], [0.1, 0.9]]), z) }))
 *   observe(`y${t}`, bernoulli({ probs: take(tensor([0.2, 0.8]), s) }), 1)
 *   return s
 * }, undefined as Value | undefined)
 * chainInference(hmm, undefined).diagnostics['mapPath'] // => [1, 1, 1]
 * ```
 */
export function chainInference<D = void>(model: Model<D>, data: D, options: ChainInferenceOptions = {}): Posterior {
  const form = chainForm(model, data, options.analysis)
  const trace = {
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  const fb = forwardBackward(form.chain, { method: 'log', ...trace })
  const map = viterbi(form.chain, trace)
  const marginals: Record<string, Distribution> = {}
  form.sites.forEach((name, t) => {
    marginals[name] = shiftedCategorical(
      fromStorage(Float64Array.from(fb.posteriors[t] ?? []), [form.states], 'float64'),
      form.low,
    )
  })
  const logZ = fb.logLikelihood + form.constant
  return definePosterior({
    form: 'exact',
    engine: 'chain',
    sites: form.sites,
    marginals,
    logEvidence: { value: logZ, standardError: 0, method: 'forward-backward' },
    diagnostics: { steps: form.sites.length, states: form.states, mapPath: map.path.map((k) => k + form.low) },
    sampler: (key: Key, n: number) => {
      const paths = backwardSample(form.chain, key, n)
      const out: Record<string, Tensor> = {}
      form.sites.forEach((name, t) => {
        out[name] = fromStorage(
          Float64Array.from(paths, (p) => (p[t] ?? 0) + form.low),
          [n],
          'float64',
        )
      })
      return out
    },
    logProb: (values) => {
      const path = form.sites.map((name) => {
        const v = values[name]
        if (v === undefined) throw new OmaiError('value', `chain posterior: no value for ${name}`)
        return (toFlat(primalOf(asTensor(v as TensorLike)))[0] ?? 0) - form.low
      })
      let lp = form.constant
      path.forEach((k, t) => {
        lp += form.chain.logEmission[t]?.[k] ?? Number.NEGATIVE_INFINITY
        if (t === 0) lp += form.chain.logInitial[k] ?? Number.NEGATIVE_INFINITY
        else {
          const A = (form.chain.logTransition as readonly (readonly (readonly number[])[])[])[t - 1]
          lp += A?.[path[t - 1] ?? 0]?.[k] ?? Number.NEGATIVE_INFINITY
        }
      })
      return lp - logZ
    },
  })
}

/**
 * The chain engine: forward-backward and FFBS for HMM-like programs (cost: steps times squared states).
 *
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { chainEngine } from 'omai-engine/probprog'
 * chainEngine().name // => 'chain'
 * ```
 */
export function chainEngine(): InferenceEngine {
  return {
    name: 'chain',
    form: 'exact',
    description: 'forward-backward, forward filtering backward sampling and Viterbi for discrete-state chains',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'exact', 0, 'observed sites have no likelihood')
      const problem = chainProblem(a)
      if (problem !== undefined) return feasibility(false, 'exact', 0, problem)
      const loop = a.markov[0]
      const k = Object.values(a.discrete.cardinalities).reduce((m, c) => Math.max(m, c), 1)
      const T = loop?.length ?? 1
      return feasibility(true, 'exact', T * k * k, `a chain of ${T} discrete steps with ${k} states`)
    },
    run: <D>(model: Model<D>, data: D, a: ProgramAnalysis, run: EngineRunOptions) =>
      chainInference(model, data, {
        analysis: a,
        ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
        ...(run.policy === undefined ? {} : { policy: run.policy }),
      }),
  }
}
