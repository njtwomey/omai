/**
 * Stochastic variational inference on probabilistic programs (NumPyro's `SVI`): fit a guide's parameters by stochastic
 * gradient ascent on the ELBO, as a traced `sviAlgorithm` and the `svi` function that returns the parameters, the ELBO
 * trace and the variational posterior. Also the engine for the planner, which uses an automatic guide.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import { type Json, MdmlError } from '../../base/index.ts'
import { bijectors, type Support } from '../../dists/index.ts'
import { adam, applyUpdates, type GradientTransformation } from '../../optim/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { type ElboEstimator, traceElbo } from '../elbo.ts'
import { type AutoGuide, autoNormal } from '../guides.ts'
import { enumerate, replay, seed, substitute, traceOf } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** A learnable parameter: its support and its value on the unconstrained scale. */
interface ParamEntry {
  readonly support: Support
  readonly bijector: bijectors.Bijector
}

/** The state of {@link sviAlgorithm}: unconstrained parameters, optimiser and estimator states, the last ELBO. */
export interface SviState {
  readonly params: Readonly<Record<string, Value>>
  readonly optState: unknown
  readonly elboState: unknown
  readonly elbo: number
  /** An exponential moving average of the ELBO (for the convergence test). */
  readonly smoothed: number
  readonly previous: number
}

/** Options of {@link sviAlgorithm} and {@link svi}. */
export interface SviOptions<D> {
  /** The guide (default `autoNormal(model)`). */
  readonly guide?: Model<D>
  readonly elbo?: ElboEstimator
  /** The optimiser on unconstrained parameters (default Adam with learning rate 0.01). */
  readonly optimizer?: GradientTransformation
  /** Steps (default 2000). */
  readonly numSteps?: number
  /** Stop with `converged` when the smoothed ELBO changes by less than this relative amount over a step (default 0). */
  readonly tol?: number
  /** Initial parameter values (constrained) by name, overriding the programs' initialisers. */
  readonly initParams?: Readonly<Record<string, TensorLike>>
}

/**
 * Initial parameters of a model and guide: run the guide (seeded) and the model replayed on it, and collect every
 * param site with its support (NumPyro's `SVI.init`).
 *
 * @param key - The key of the initialisers.
 * @param model - The model.
 * @param guide - The guide.
 * @param data - The data.
 * @returns Constrained initial values and supports by name.
 * @concept variational-inference
 * @example
 * ```ts
 * import { autoNormal, initialParams, sample } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * const model = () => { sample('x', normal(0, 1)) }
 * Object.keys(initialParams(key(0), model, autoNormal(model), undefined).values) // => ['x_auto_loc', 'x_auto_scale']
 * ```
 */
export function initialParams<D = void>(
  key: Key,
  model: Model<D>,
  guide: Model<D>,
  data: D,
): { readonly values: Record<string, Value>; readonly supports: Record<string, Support> } {
  const values: Record<string, Value> = {}
  const supports: Record<string, Support> = {}
  const g = traceOf(seed(guide, foldIn(key, 0)), data)
  const m = traceOf(seed(replay(enumerate(model), g), foldIn(key, 1)), data)
  for (const t of [g, m])
    for (const site of t.sites.values()) {
      if (site.type !== 'param' || values[site.name] !== undefined) continue
      values[site.name] = primalOf(site.value)
      supports[site.name] = site.support
    }
  return { values, supports }
}

/**
 * Stochastic variational inference as a traced algorithm (NumPyro's `SVI`): each step estimates the ELBO's gradient
 * with respect to the unconstrained parameters (constrained parameters are their images through the bijectors of
 * their supports) and applies the optimiser. Records the ELBO, the loss and the gradient norm; stops with `diverged`
 * when the loss is not finite and `converged` when the smoothed ELBO settles (with `tol`).
 *
 * @param model - The model.
 * @param data - The data.
 * @param options - Guide, estimator, optimiser, tolerance and initial parameters.
 * @returns The algorithm (run it with `runAlgorithm` and a step budget).
 * @concept variational-inference
 * @example
 * ```ts
 * import { observe, sample, sviAlgorithm } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1) }
 * runAlgorithm(sviAlgorithm(model, undefined), { seed: 0, maxSteps: 5 }).steps // => 5
 * ```
 */
export function sviAlgorithm<D = void>(model: Model<D>, data: D, options: SviOptions<D> = {}): Algorithm<SviState> {
  const guide = options.guide ?? autoNormal(model)
  const estimator = options.elbo ?? traceElbo()
  const optimizer = options.optimizer ?? adam({ learningRate: 0.01 })
  const tol = options.tol ?? 0
  let entries: Record<string, ParamEntry> = {}
  const constrain = (u: Readonly<Record<string, Value>>): Record<string, Value> => {
    const out: Record<string, Value> = {}
    for (const [name, v] of Object.entries(u)) {
      const e = entries[name]
      out[name] = e === undefined ? v : e.bijector.forward(v)
    }
    return out
  }
  return {
    name: 'svi',
    config: { elbo: estimator.name, optimizer: optimizer.name, tol } as Json,
    schema: channels({
      metrics: {
        elbo: { description: 'estimated evidence lower bound', direction: 'higher' },
        loss: { description: 'surrogate loss (minus the ELBO)', direction: 'lower' },
      },
      tensors: { params: { role: 'param', description: 'unconstrained parameters' } },
    }),
    init: (key) => {
      const init = initialParams(key, model, guide, data)
      entries = {}
      const params: Record<string, Value> = {}
      for (const [name, v0] of Object.entries(init.values)) {
        const support = init.supports[name] ?? { kind: 'real' }
        const bijector = bijectors.bijectorForSupport(support)
        entries[name] = { support, bijector }
        const given = options.initParams?.[name]
        params[name] = primalOf(bijector.inverse(given === undefined ? v0 : given))
      }
      return {
        params,
        optState: optimizer.init(params),
        elboState: estimator.init(),
        elbo: Number.NaN,
        smoothed: Number.NaN,
        previous: Number.NaN,
      }
    },
    step: (s, ctx) => {
      let est: { readonly elbo: number; readonly state: unknown } = { elbo: Number.NaN, state: s.elboState }
      const { grad } = valueAndGrad((u: Record<string, Value>) => {
        const r = estimator.estimate(ctx.key, model, guide, data, constrain(u), s.elboState)
        est = r
        return r.loss
      })(s.params as Record<string, Value>)
      const { updates, state } = optimizer.update(grad, s.optState, s.params)
      const params = applyUpdates(s.params, updates) as Record<string, Value>
      const smoothed = Number.isNaN(s.smoothed) ? est.elbo : 0.9 * s.smoothed + 0.1 * est.elbo
      return { params, optState: state, elboState: est.state, elbo: est.elbo, smoothed, previous: s.smoothed }
    },
    done: (s) => {
      if (Number.isNaN(s.elbo)) return undefined
      if (!Number.isFinite(s.elbo)) return 'diverged'
      for (const v of Object.values(s.params)) if (toFlat(v).some((x) => !Number.isFinite(x))) return 'diverged'
      if (
        tol > 0 &&
        Number.isFinite(s.previous) &&
        Math.abs(s.smoothed - s.previous) <= tol * Math.max(1, Math.abs(s.smoothed))
      )
        return 'converged'
      return undefined
    },
    observe: (s) => ({
      metrics: Number.isNaN(s.elbo) ? {} : { elbo: s.elbo, loss: -s.elbo },
      tensors: Object.fromEntries(
        Object.entries(s.params).map(([k, v]) => [`params.${k}`, { role: 'param' as const, value: v }]),
      ),
    }),
  }
}

/** The result of {@link svi}. */
export interface SviResult {
  /** Fitted parameters on the constrained scale. */
  readonly params: Record<string, Tensor>
  /** The ELBO estimate at every step. */
  readonly elbos: readonly number[]
  readonly status: Posterior['status']
  readonly steps: number
  readonly posterior: Posterior
}

/**
 * Fit a guide by stochastic variational inference and return the fitted parameters, the ELBO trace and the
 * variational posterior (from the automatic guide, or Monte Carlo runs of a custom guide).
 *
 * @param model - The model.
 * @param data - The data.
 * @param options - SVI options, key and tracing.
 * @returns Parameters, ELBOs, status and posterior.
 * @concept variational-inference
 * @example
 * ```ts
 * import { observe, sample, svi } from 'mdml-engine/probprog'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * import { item } from 'mdml-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const r = svi(model, undefined, { key: key(0), numSteps: 1500 })
 * Math.abs(item(r.params['m_auto_loc'] ?? 0) - 1) < 0.2 // => true
 * ```
 */
export function svi<D = void>(model: Model<D>, data: D, options: SviOptions<D> & EngineRunOptions): SviResult {
  const guide = options.guide ?? autoNormal(model)
  const alg = sviAlgorithm(model, data, { ...options, guide })
  const elbos: number[] = []
  const recording: Algorithm<SviState> = {
    ...alg,
    step: (s, ctx) => {
      const next = alg.step(s, ctx)
      elbos.push(next.elbo)
      return next
    },
  }
  const r = runAlgorithm(recording, {
    key: options.key,
    maxSteps: options.numSteps ?? 2000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const init = initialParams(options.key, model, guide, data)
  const params: Record<string, Tensor> = {}
  for (const [name, u] of Object.entries(r.state.params)) {
    const b = bijectors.bijectorForSupport(init.supports[name] ?? { kind: 'real' })
    params[name] = primalOf(b.forward(u))
  }
  const status = r.status
  const diagnostics: Record<string, Json> = {
    steps: r.steps,
    finalElbo: elbos[elbos.length - 1] ?? null,
    guide: isAutoGuide(guide) ? guide.kind : 'custom',
  }
  let posterior: Posterior
  if (isAutoGuide(guide)) {
    const p = guide.posterior(params, data)
    posterior = definePosterior({
      form: p.form,
      engine: 'svi',
      sites: p.sites,
      status,
      diagnostics,
      sampler: (k, n) => p.sample(k, n),
      ...(p.logProb === undefined ? {} : { logProb: p.logProb }),
      marginals: Object.fromEntries(
        p.sites.flatMap((s) => {
          try {
            return [[s, p.marginal(s)]]
          } catch {
            return []
          }
        }),
      ),
    })
  } else {
    const sites = [...traceOf(seed(substitute(guide, params), foldIn(options.key, 9)), data).sites.values()]
      .filter((s) => s.type === 'sample' && !s.isObserved)
      .map((s) => s.name)
    posterior = definePosterior({
      form: 'variational',
      engine: 'svi',
      sites,
      status,
      diagnostics,
      sampler: (k, n) => {
        const out: Record<string, number[]> = {}
        const shapes: Record<string, readonly number[]> = {}
        for (let i = 0; i < n; i++) {
          const t = traceOf(seed(substitute(guide, params), foldIn(k, i)), data)
          for (const name of sites) {
            const s = t.sites.get(name)
            if (s?.type !== 'sample') continue
            shapes[name] = s.value.shape
            out[name] = [...(out[name] ?? []), ...toFlat(primalOf(s.value))]
          }
        }
        return Object.fromEntries(
          sites.map((name) => [
            name,
            fromStorage(Float64Array.from(out[name] ?? []), [n, ...(shapes[name] ?? [])], 'float64'),
          ]),
        )
      },
    })
  }
  return { params, elbos, status, steps: r.steps, posterior }
}

function isAutoGuide<D = void>(g: Model<D>): g is AutoGuide<D> {
  return typeof (g as Partial<AutoGuide<D>>).posterior === 'function'
}

/** Options of {@link sviEngine}. */
export type SviEngineOptions = Omit<SviOptions<unknown>, 'guide'> & {
  /** Which automatic guide (default `autoNormal`). */
  readonly guide?: <D>(model: Model<D>) => AutoGuide<D>
}

/**
 * The SVI engine: stochastic variational inference with an automatic guide (cost: steps x observations).
 *
 * @param options - Steps, estimator, optimiser and the guide family.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { sviEngine } from 'mdml-engine/probprog'
 * sviEngine({ numSteps: 500 }).form // => 'variational'
 * ```
 */
export function sviEngine(options: SviEngineOptions = {}): InferenceEngine {
  const { guide: family, ...rest } = options
  return {
    name: 'svi',
    form: 'variational',
    description: 'stochastic variational inference with an automatic guide',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'approximate', 0, 'observed sites have no likelihood')
      if (a.continuous.length === 0) return feasibility(false, 'approximate', 0, 'no continuous latent sites')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'approximate', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      if (!a.differentiable) return feasibility(false, 'approximate', 0, 'the log density is not differentiable')
      const steps = options.numSteps ?? 2000
      const enumCost = Math.max(1, Math.min(a.discrete.jointSize, 1e6))
      return feasibility(
        true,
        'approximate',
        steps * 3 * Math.max(1, a.observations + a.continuousDim) * enumCost,
        a.plates.some((p) => p.subsampleSize < p.size)
          ? 'differentiable density with subsampled plates'
          : 'differentiable density',
      )
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      svi(model, data, {
        ...(rest as Omit<SviOptions<D>, 'guide'>),
        ...(family === undefined ? {} : { guide: family(model) }),
        ...run,
      }).posterior,
  }
}

/**
 * The ELBO of a fitted SVI run: the mean of the last tenth of its trace (single estimates are noisy).
 *
 * @param result - The run.
 * @returns The averaged ELBO.
 * @throws MdmlError with code `value` for a run without steps.
 * @example
 * ```ts
 * import { finalElbo } from 'mdml-engine/probprog'
 * finalElbo({ elbos: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] }) // => 10
 * ```
 */
export function finalElbo(result: Pick<SviResult, 'elbos'>): number {
  const n = result.elbos.length
  if (n === 0) throw new MdmlError('value', 'finalElbo: no steps')
  const tail = result.elbos.slice(Math.floor(n * 0.9))
  return tail.reduce((a, b) => a + b, 0) / tail.length
}
