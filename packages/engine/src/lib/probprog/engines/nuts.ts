/**
 * Hamiltonian Monte Carlo with the No-U-Turn Sampler on probabilistic programs (NumPyro's `MCMC(NUTS(model))`):
 * sampling the unconstrained density with Jacobian corrections and enumerated discrete sites summed out, then drawing
 * those discrete sites back with `inferDiscrete`. Also the engine for the planner.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { categorical as drawCategorical, foldIn, type Key } from '../../random/index.ts'
import { type Chains, diagnose, hmc, nuts } from '../../sampling/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { enumeratedMarginals, maxPlateNesting, traceFactors } from '../_contract.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import {
  constrainDraws,
  findInitialParams,
  type InitStrategy,
  initToUniform,
  logJoint,
  potentialEnergy,
  type UnconstrainedModel,
} from '../density.ts'
import { enumerate } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/**
 * Draw the enumerated discrete sites of a program given values of its other sites (NumPyro's `infer_discrete` with
 * temperature 1): each site in execution order is drawn from its exact conditional marginal given the values and the
 * sites drawn before it (elementwise for plated sites), by differentiating the enumerated log density.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - Values of the non-enumerated latent sites.
 * @param key - The key.
 * @returns The values plus a draw of every enumerated site.
 * @concept exact-inference
 * @example
 * ```ts
 * import { inferDiscrete, observe, sample } from 'mdml-engine/probprog'
 * import { bernoulli, normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * import { item } from 'mdml-engine/tensor'
 * const model = () => { const c = sample('c', bernoulli({ probs: 0.5 })); observe('y', normal(c, 0.01), 1) }
 * item(inferDiscrete(model, undefined, {}, key(0))['c'] ?? 0) // => 1
 * ```
 */
export function inferDiscrete<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>>,
  key: Key,
): Record<string, TensorLike> {
  const m = enumerate(model)
  const fixed: Record<string, TensorLike> = { ...values }
  for (let step = 0; step < 10000; step++) {
    const r = logJoint(m, data, fixed)
    const t = r.trace
    const next = [...t.sites.values()].find((s) => s.type === 'sample' && s.enumDim !== undefined)
    if (next === undefined || next.type !== 'sample') return fixed
    const depth = maxPlateNesting(t)
    const { factors, sumVars, scales } = traceFactors(t, depth)
    const { marginals } = enumeratedMarginals(factors, sumVars, scales)
    const mg = marginals.get(next.name)
    if (mg === undefined) throw new MdmlError('value', `inferDiscrete: no marginal for ${next.name}`)
    const probs = primalOf(mg.probs)
    const k = probs.shape[probs.shape.length - 1] ?? 1
    const batch = probs.shape.slice(0, -1)
    const count = batch.reduce((a, b) => a * b, 1)
    const flat = toFlat(probs)
    const support = next.fn.support
    const low = support.kind === 'integerInterval' ? support.low : 0
    const out = new Float64Array(count)
    for (let i = 0; i < count; i++) {
      const w = flat.slice(i * k, (i + 1) * k).map((p) => Math.max(p, 0))
      out[i] = low + (toFlat(drawCategorical(foldIn(foldIn(key, step), i), w, [1]))[0] ?? 0)
    }
    // The site's own batch shape: plates of the marginal, padded with the singleton plate axes of its batch.
    const own = next.fn.batchShape.slice(
      next.fn.batchShape.length - next.plates.reduce((a, f) => Math.max(a, -f.dim), 0),
    )
    fixed[next.name] = fromStorage(out, own.reduce((a, b) => a * b, 1) === count ? own : batch, 'float64')
  }
  throw new MdmlError('value', 'inferDiscrete: too many enumerated sites')
}

/** Options of {@link mcmc}. */
export interface McmcOptions {
  /** `nuts` (default) or `hmc` with a fixed number of integration steps. */
  readonly algorithm?: 'nuts' | 'hmc'
  readonly numChains?: number
  readonly numDraws?: number
  readonly numWarmup?: number
  readonly thin?: number
  /** HMC integration steps per transition (default 10). */
  readonly numSteps?: number
  readonly maxTreeDepth?: number
  readonly targetAcceptance?: number
  /** Adapt a dense mass matrix (default diagonal). */
  readonly dense?: boolean
  /** How chains start (default {@link initToUniform}, radius 2). */
  readonly init?: InitStrategy
  /** Draw enumerated discrete sites for every draw (default true when there are any). */
  readonly inferDiscrete?: boolean
}

function rowOf(d: Tensor, i: number): Tensor {
  const shape = d.shape.slice(1)
  const size = shape.reduce((a, b) => a * b, 1)
  return fromStorage(d.data.slice(i * size, (i + 1) * size) as Float64Array, shape, 'float64')
}

function failed(engine: string, reason: string, extra: Readonly<Record<string, Json>> = {}): Posterior {
  return definePosterior({
    form: 'samples',
    engine,
    sites: [],
    status: 'failed',
    draws: {},
    diagnostics: { reason, ...extra },
  })
}

/**
 * Hamiltonian Monte Carlo on a program (NumPyro's `MCMC(NUTS(model))`): the unconstrained potential (Jacobians
 * added, enumerated discrete sites summed out), valid starting points per chain, window adaptation of step size and
 * mass matrix, NUTS or fixed-length HMC as a traced run per chain, then draws mapped back to every latent and
 * deterministic site, with enumerated sites recovered by {@link inferDiscrete}. Programs without a valid starting
 * point end with status `failed` and the reason; divergences, R-hat and ESS are in the diagnostics.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Sampler options, key and tracing.
 * @returns The posterior of draws (chains laid end to end, with `chains` on unconstrained coordinates).
 * @concept nuts
 * @example
 * ```ts
 * import { mcmc, observe, sample } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const post = mcmc(model, undefined, { key: key(0), numChains: 1, numWarmup: 100, numDraws: 100 })
 * post.draws?.['mu']?.shape // => [100]
 * ```
 */
export function mcmc<D = void>(model: Model<D>, data: D, options: McmcOptions & EngineRunOptions): Posterior {
  const engine = options.algorithm ?? 'nuts'
  let um: UnconstrainedModel
  try {
    um = potentialEnergy(model, data, { key: foldIn(options.key, 0) })
  } catch (e) {
    return failed(engine, e instanceof Error ? e.message : String(e))
  }
  const numChains = options.numChains ?? 2
  const numDraws = options.numDraws ?? 500
  const numWarmup = options.numWarmup ?? 500
  const starts: Float64Array[] = []
  for (let c = 0; c < numChains; c++) {
    const init = findInitialParams(um, model, data, {
      key: foldIn(foldIn(options.key, 1), c),
      strategy: options.init ?? initToUniform(),
    })
    if (!init.valid)
      return failed(engine, 'no starting point with a finite log density and gradient', { tries: init.tries })
    starts.push(init.unconstrained)
  }
  const common = {
    key: foldIn(options.key, 2),
    numChains,
    numDraws,
    numWarmup,
    init: starts,
    ...(options.thin === undefined ? {} : { thin: options.thin }),
    ...(options.targetAcceptance === undefined ? {} : { targetAcceptance: options.targetAcceptance }),
    ...(options.dense === undefined ? {} : { dense: options.dense }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  const chains: Chains =
    engine === 'hmc'
      ? hmc(um.logDensity, { ...common, numIntegrationSteps: options.numSteps ?? 10 })
      : nuts(um.logDensity, {
          ...common,
          ...(options.maxTreeDepth === undefined ? {} : { maxTreeDepth: options.maxTreeDepth }),
        })
  const flat = chains.draws.data
  const draws = constrainDraws(um, flat)
  const total = numChains * chains.numDraws
  if (um.enumerated.length > 0 && options.inferDiscrete !== false) {
    const discrete: Record<string, number[]> = {}
    const shapes: Record<string, readonly number[]> = {}
    for (let i = 0; i < total; i++) {
      const values: Record<string, Value> = {}
      for (const l of um.layout) {
        const d = draws[l.name]
        if (d !== undefined) values[l.name] = rowOf(d, i)
      }
      const r = inferDiscrete(model, data, values, foldIn(foldIn(options.key, 3), i))
      for (const name of um.enumerated) {
        const v = primalOf(r[name] as Tensor)
        shapes[name] = v.shape
        discrete[name] = [...(discrete[name] ?? []), ...toFlat(v)]
      }
    }
    for (const name of um.enumerated)
      draws[name] = fromStorage(Float64Array.from(discrete[name] ?? []), [total, ...(shapes[name] ?? [])], 'float64')
  }
  const report = diagnose(chains, options.maxTreeDepth === undefined ? {} : { maxTreeDepth: options.maxTreeDepth })
  const nonFinite = toFlat(chains.draws).some((v) => !Number.isFinite(v))
  return definePosterior({
    form: 'samples',
    engine,
    sites: Object.keys(draws),
    draws,
    chainShape: [numChains, chains.numDraws],
    chains,
    status: nonFinite ? 'failed' : 'completed',
    diagnostics: {
      divergences: report.divergences?.count ?? 0,
      highRhat: [...report.highRhat],
      lowEss: [...report.lowEss],
      messages: [...report.messages],
      stepSizes: chains.parameters.map((p) => (p as { stepSize?: number }).stepSize ?? null),
      enumerated: um.enumerated,
    },
  })
}

/**
 * The NUTS engine: Hamiltonian Monte Carlo on the unconstrained density with enumerated discrete sites summed out
 * (cost: chains x (warm-up + draws) x about 2^5 gradient evaluations x observations).
 *
 * @param options - Sampler options.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { nutsEngine } from 'mdml-engine/probprog'
 * nutsEngine({ numDraws: 200 }).name // => 'nuts'
 * ```
 */
export function nutsEngine(options: McmcOptions = {}): InferenceEngine {
  const name = options.algorithm ?? 'nuts'
  return {
    name,
    form: 'samples',
    description: 'NUTS and HMC on the unconstrained log density, discrete sites summed out by enumeration',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'mcmc', 0, 'observed sites have no likelihood')
      if (a.continuous.length === 0) return feasibility(false, 'mcmc', 0, 'no continuous latent sites')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'mcmc', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      if (!a.differentiable) return feasibility(false, 'mcmc', 0, 'the log density is not differentiable')
      if (a.plates.some((p) => p.subsampleSize < p.size))
        return feasibility(false, 'mcmc', 0, 'subsampled plates need stochastic-gradient methods')
      const iterations = (options.numChains ?? 2) * ((options.numWarmup ?? 500) + (options.numDraws ?? 500))
      const enumCost = Math.max(1, Math.min(a.discrete.jointSize, 1e6))
      const cost = iterations * 32 * Math.max(1, a.observations + a.continuousDim) * enumCost
      return feasibility(true, 'mcmc', cost, `${a.continuousDim} continuous coordinates with a differentiable density`)
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      mcmc(model, data, { ...options, ...run }),
  }
}
