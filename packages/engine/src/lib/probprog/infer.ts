import { type Json, WhiteboxError } from '../base/index.ts'
import { type ContextOptions, resolveKey, resolvePolicy, resolveTracer } from '../context/index.ts'
import { startRun } from '../trace/index.ts'
import { analyse, type ProgramAnalysis } from './analysis.ts'
import { abcEngine } from './engines/abc.ts'
import { chainEngine } from './engines/chain.ts'
import { conjugateEngine } from './engines/conjugate.ts'
import { emEngine } from './engines/em.ts'
import { type Feasibility, type InferenceEngine, QUALITY_RANK } from './engines/engine.ts'
import { enumerationEngine } from './engines/enumeration.ts'
import { gaussianEngine } from './engines/gaussian.ts'
import { gibbsEngine } from './engines/gibbs.ts'
import { importanceEngine } from './engines/importance.ts'
import { laplaceEngine } from './engines/laplace.ts'
import { nutsEngine } from './engines/nuts.ts'
import { smcEngine } from './engines/smc.ts'
import { sviEngine } from './engines/svi.ts'
import type { Posterior } from './posterior.ts'
import type { Model } from './primitives.ts'

/**
 * Every engine the library has, with default options: exact engines (enumeration, conjugate, Gaussian, chain), MCMC
 * (NUTS, Gibbs), weighting (SMC, importance), approximations (Laplace, SVI, EM) and ABC.
 *
 * @returns The engines.
 * @concept inference-engines
 * @example
 * ```ts
 * import { defaultEngines } from 'whitebox-engine/probprog'
 * defaultEngines().map((e) => e.name).length // => 12
 * ```
 */
export function defaultEngines(): InferenceEngine[] {
  return [
    enumerationEngine(),
    conjugateEngine(),
    gaussianEngine(),
    chainEngine(),
    nutsEngine(),
    gibbsEngine(),
    smcEngine(),
    importanceEngine(),
    laplaceEngine(),
    sviEngine(),
    emEngine(),
    abcEngine(),
  ]
}

/** One engine in a plan: its verdict and its place in the ranking. */
export interface PlanEntry {
  readonly engine: InferenceEngine
  readonly feasibility: Feasibility
  /** Whether the cost is within the budget. */
  readonly withinBudget: boolean
}

/** The ranked engines for a program, with the analysis behind them. */
export interface InferencePlan {
  readonly analysis: ProgramAnalysis
  /** Engines that apply and fit the budget first (by quality, then cost), then the rest. */
  readonly entries: readonly PlanEntry[]
  readonly budget: number
}

/** Options of {@link plan}. */
export interface PlanOptions {
  /** The engines to consider (default {@link defaultEngines}). */
  readonly engines?: readonly InferenceEngine[]
  /** The largest acceptable estimated cost (default Infinity). */
  readonly budget?: number
  /** An analysis already made. */
  readonly analysis?: ProgramAnalysis
}

/**
 * Rank the inference engines for a program (section 3.6 of the design): analyse it once, ask every engine whether it
 * applies and at what cost, and order the applicable engines within the budget by the quality of their answer (exact,
 * MCMC, weighting, approximation, likelihood-free) and then by cost. The order is predictable: an exact engine always
 * wins when one applies.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Engines, budget and an existing analysis.
 * @returns The plan.
 * @concept inference-engines
 * @example
 * ```ts
 * import { observe, plan, plate, sample } from 'whitebox-engine/probprog'
 * import { beta, bernoulli } from 'whitebox-engine/dists'
 * const coin = () => { const p = sample('p', beta(1, 1)); plate('n', 3, () => observe('y', bernoulli({ probs: p }), [1, 0, 1])) }
 * plan(coin, undefined).entries[0]?.engine.name // => 'conjugate'
 * ```
 */
export function plan<D>(model: Model<D>, data: D, options: PlanOptions = {}): InferencePlan {
  const analysis = options.analysis ?? analyse(model, data)
  const budget = options.budget ?? Number.POSITIVE_INFINITY
  const entries = (options.engines ?? defaultEngines()).map((engine) => {
    let f: Feasibility
    try {
      f = engine.applies(analysis)
    } catch (e) {
      f = {
        applies: false,
        cost: Number.POSITIVE_INFINITY,
        quality: 'approximate',
        reason: `could not decide: ${e instanceof Error ? e.message : String(e)}`,
      }
    }
    return { engine, feasibility: f, withinBudget: f.applies && f.cost <= budget }
  })
  entries.sort((a, b) => {
    const ua = a.withinBudget ? 0 : a.feasibility.applies ? 1 : 2
    const ub = b.withinBudget ? 0 : b.feasibility.applies ? 1 : 2
    if (ua !== ub) return ua - ub
    const qa = QUALITY_RANK[a.feasibility.quality]
    const qb = QUALITY_RANK[b.feasibility.quality]
    if (qa !== qb) return qa - qb
    return a.feasibility.cost - b.feasibility.cost
  })
  return { analysis, entries, budget }
}

/**
 * A plan as a table: one line per engine with its verdict, quality, cost and reason, the chosen engine first.
 *
 * @param p - The plan.
 * @returns The explanation.
 * @concept inference-engines
 * @example
 * ```ts
 * import { explainPlan, observe, plan, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * explainPlan(plan(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }, undefined)).split('\n')[0] // => '1. conjugate (exact, cost 1): every latent site is a conjugate prior (normal-normal)'
 * ```
 */
export function explainPlan(p: InferencePlan): string {
  return p.entries
    .map((e, i) => {
      const f = e.feasibility
      const verdict = e.withinBudget
        ? `${f.quality}, cost ${formatCost(f.cost)}`
        : f.applies
          ? `over budget, cost ${formatCost(f.cost)}`
          : 'does not apply'
      return `${i + 1}. ${e.engine.name} (${verdict}): ${f.reason}`
    })
    .join('\n')
}

function formatCost(c: number): string {
  return c < 1e5 ? String(Math.round(c)) : c.toExponential(1)
}

/** Options of {@link infer}. */
export interface InferOptions extends ContextOptions {
  /** `auto` (default), an engine name, or an engine. */
  readonly engine?: 'auto' | string | InferenceEngine
  /** The largest acceptable estimated cost for `auto`. */
  readonly budget?: number
  /** The engines `auto` chooses from. */
  readonly engines?: readonly InferenceEngine[]
}

/**
 * Posterior inference on a program (`infer(model, data)`, section 3.6): plan, pick the best engine (or the one asked
 * for), run it as a traced child of an `infer` run, and return its posterior. If the chosen engine fails at run time
 * (a degenerate model, a failed start), the next engine in the plan is tried; `diagnostics.reason` records why the
 * engine was chosen and `diagnostics.tried` every attempt.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Engine choice, budget, engines, and key, context, tracer and policy.
 * @returns The posterior.
 * @throws WhiteboxError with code `value` for an unknown engine name, or when no engine applies.
 * @concept inference-engines
 * @example
 * ```ts
 * import { infer, observe, plate, sample } from 'whitebox-engine/probprog'
 * import { gamma, poisson } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const model = () => { const r = sample('rate', gamma(2, 1)); plate('n', 4, () => observe('k', poisson(r), [3, 1, 4, 2])) }
 * const post = infer(model, undefined)
 * post.engine // => 'conjugate'
 * item(post.mean('rate')) // => 2.4
 * ```
 */
export function infer<D>(model: Model<D>, data: D, options: InferOptions = {}): Posterior {
  const key = resolveKey(options, 'infer')
  const tracer = resolveTracer(options)
  const policy = resolvePolicy(options)
  const engines = options.engines ?? defaultEngines()
  const chosen = options.engine ?? 'auto'
  const p = plan(model, data, { engines, ...(options.budget === undefined ? {} : { budget: options.budget }) })
  let candidates: InferenceEngine[]
  if (chosen === 'auto') candidates = p.entries.filter((e) => e.withinBudget).map((e) => e.engine)
  else if (typeof chosen === 'string') {
    const e = engines.find((x) => x.name === chosen)
    if (e === undefined) throw new WhiteboxError('value', `infer: no engine named ${chosen}`)
    candidates = [e]
  } else candidates = [chosen]
  if (candidates.length === 0)
    throw new WhiteboxError('value', `infer: no engine applies within the budget\n${explainPlan(p)}`)
  const run = startRun(tracer, {
    name: 'infer',
    seed: null,
    config: { engine: typeof chosen === 'string' ? chosen : chosen.name, budget: p.budget } as Json,
    policy,
  })
  run.message(0, 'plan', {
    ranking: p.entries.map((e) => ({
      engine: e.engine.name,
      applies: e.feasibility.applies,
      withinBudget: e.withinBudget,
      quality: e.feasibility.quality,
      cost: Number.isFinite(e.feasibility.cost) ? e.feasibility.cost : null,
      reason: e.feasibility.reason,
    })),
  } as Json)
  const tried: Json[] = []
  for (const engine of candidates) {
    const entry = p.entries.find((e) => e.engine === engine)
    const reason = entry === undefined ? 'requested' : entry.feasibility.reason
    let post: Posterior
    try {
      post = engine.run(model, data, p.analysis, {
        key,
        tracer,
        policy,
        parent: run,
        ...(options.budget === undefined ? {} : { budget: options.budget }),
      })
    } catch (e) {
      tried.push({ engine: engine.name, error: e instanceof Error ? e.message : String(e) })
      if (chosen !== 'auto') throw e
      continue
    }
    if (post.status === 'failed' && chosen === 'auto') {
      tried.push({ engine: engine.name, error: String(post.diagnostics['reason'] ?? 'failed') })
      continue
    }
    tried.push({ engine: engine.name, status: post.status })
    run.end(post.status, 1, {})
    return withDiagnostics(post, { reason, tried: [...tried] })
  }
  run.end('failed', 1, {})
  throw new WhiteboxError('value', `infer: every engine failed: ${JSON.stringify(tried)}`)
}

/**
 * A posterior with fixed diagnostics replaced (a convenience for engines and wrappers).
 *
 * @param post - The posterior.
 * @param diagnostics - Diagnostics to merge in.
 * @returns The posterior with the merged diagnostics.
 * @example
 * ```ts
 * import { pointPosterior, withDiagnostics } from 'whitebox-engine/probprog'
 * withDiagnostics(pointPosterior({ a: 1 }, { engine: 'x' }), { note: 'hi' }).diagnostics['note'] // => 'hi'
 * ```
 */
export function withDiagnostics(post: Posterior, diagnostics: Readonly<Record<string, Json>>): Posterior {
  return { ...post, diagnostics: { ...post.diagnostics, ...diagnostics } }
}
