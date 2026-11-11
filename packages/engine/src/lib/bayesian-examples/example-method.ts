/**
 * The scaffold every canonical Bayesian example is made on. An example is a trace `Method` that a worker or a page runs by
 * name and a few knobs: it rebuilds its problem (data, features, model) from the knobs, takes one inference method from
 * the table of `methodKinds` with that method's own knobs and a time and step budget, delegates the run to it, and adds
 * derived paths to the contract that read the posterior at any step (what the example's page draws and checks). This file
 * holds what the examples share: the table of inference methods on offer and their short-run defaults, the union space
 * that offers them, the budget knobs, the steps a run is expected to take, the curve a method has on the tape, the draws of
 * a posterior as rows, a reference configuration (a long run of the same problem), and {@link defineExample}, which turns
 * a problem and its derived paths into the `Method`. An example file declares its knobs, its problem and its derived
 * paths, nothing else.
 *
 * @index structure Example scaffold: a problem from knobs, one chosen inference method, derived posterior paths.
 */

import { anytimeKnobs, defineInferenceMethod, type InferenceMethod } from '../probprog/inference/method.ts'
import { type MethodSpace, methodConfig, methodSpace } from '../probprog/inference/method-space.ts'
import {
  type AnyInferenceMethod,
  inferenceMethodFor,
  methodKinds,
  modelRegistry,
} from '../probprog/inference/models.ts'
import type { DataInput, ModelDefinition } from '../probprog/model/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
import { key } from '../random/index.ts'
import { type DictSpace, defaults, dict, int, type RealSpace, type UnionSpace } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Config, type Contract, type DerivedSource, session } from '../trace/index.ts'

/**
 * Where the controls of each inference method open (short runs: an answer in a second or two); a method not listed opens
 * on its own defaults. Nothing is removed: every knob of the method is still there.
 */
export const exampleStarting: Readonly<Record<string, Readonly<Record<string, number | string | boolean>>>> = {
  nuts: { numChains: 2, numWarmup: 60, numDraws: 120 },
  hmc: { numChains: 2, numWarmup: 60, numDraws: 120, numSteps: 10 },
  mala: { numChains: 2, numWarmup: 100, numDraws: 250 },
  svi: { guide: 'autoNormal', learningRate: 0.05 },
  ep: { damping: 0 },
  pathfinder: { numDraws: 400 },
}

// Knobs of a method fixed for every run (when it has them).
const FIXED: Readonly<Record<string, Readonly<Record<string, number | string | boolean>>>> = { ep: { tol: 1e-6 } }

// The variational kinds (`svi`, `svi-stl`, ...) run for a number of steps the page sets (a knob added to their options).
const isVariational = (kind: string): boolean => kind === 'svi' || kind.startsWith('svi-')
const STEPS = 600

/** The names of the kinds that run a number of optimiser steps the page sets. */
const hasStepsKnob = isVariational

/** The names of the knobs that make a method a sampler of chains (its curve is the chains, read by the diagnostics). */
const isSampler = (knobs: Config): boolean => 'numDraws' in knobs && 'numWarmup' in knobs

/**
 * The number in a configuration, or 0 when it is not one.
 *
 * @param config - A configuration.
 * @param name - The knob.
 * @returns The number.
 * @example
 * ```ts
 * import { exampleNumber } from 'omai-engine/bayesian-examples'
 * exampleNumber({ rows: 5 }, 'rows') // => 5
 * ```
 */
export function exampleNumber(config: Config, name: string): number {
  const v = config[name]
  return typeof v === 'number' ? v : 0
}

/**
 * The string in a configuration, or the empty string when it is not one.
 *
 * @param config - A configuration.
 * @param name - The knob.
 * @returns The string.
 * @example
 * ```ts
 * import { exampleString } from 'omai-engine/bayesian-examples'
 * exampleString({ link: 'probit' }, 'link') // => 'probit'
 * ```
 */
export function exampleString(config: Config, name: string): string {
  const v = config[name]
  return typeof v === 'string' ? v : ''
}

/** The inference kind and the knobs of a configuration (`nuts` when it names none); the union's tag is `method`. */
function inferenceOf(config: Config): { readonly kind: string; readonly knobs: Config } {
  const given = config['inference']
  const value = (typeof given === 'object' && given !== null && !Array.isArray(given) ? given : {}) as Record<
    string,
    unknown
  >
  const { kind, config: knobs } = methodConfig('method' in value ? value : { method: 'nuts', ...value })
  return { kind: kind in methodKinds ? kind : 'nuts', knobs }
}

/**
 * The knobs of the budget shown with the inference: the time budget (2 s by default) and the step budget (0: none).
 * A method of the library stops at them (the producers and the session enforce them).
 */
export const exampleBudgetKnobs: DictSpace = dict({
  budgetMs: { ...(anytimeKnobs.spaces['budgetMs'] as RealSpace), default: 2000, label: 'time budget' },
  budgetSteps: anytimeKnobs.spaces['budgetSteps'] as DictSpace['spaces'][string],
})

/** What an example's problem has to give: the model and its data. The rest is the example's own. */
export interface ExampleProblem {
  readonly definition: ModelDefinition
  readonly data: DataInput
}

/** The methods an example offers on a problem: every kind that runs on it as a union, and the others with the library's reason. */
export type ExampleOffer = MethodSpace

/**
 * Every inference method the library can run on a problem, as one union of their knobs (tagged `method`, opening on NUTS
 * with the short-run knobs of {@link exampleStarting}) and the reason for each kind that cannot (`methodSpace`). Where a
 * variational kind runs, a `steps` knob is added to its options. This is the one place that chooses which methods an
 * example offers: all of them that apply.
 *
 * @param problem - The problem to bind the methods to.
 * @param modelName - The name the model is registered under (the example's own).
 * @returns The union space, the applicable kinds and the reasons for the others.
 * @example
 * ```ts
 * import { exampleInferenceSpace, classifierProblem } from 'omai-engine/bayesian-examples'
 * const p = classifierProblem({ dataset: 'moons', rows: 60, noise: 0.3, dataSeed: 1, link: 'probit', degree: 1, priorScale: 1 })
 * const offer = exampleInferenceSpace(p, 'classifier')
 * [offer.applicable.includes('nuts'), 'enumeration' in offer.unavailable] // => [true, true]
 * ```
 */
export function exampleInferenceSpace(problem: ExampleProblem, modelName: string): ExampleOffer {
  const registry = modelRegistry({ [modelName]: { definition: problem.definition, data: problem.data } })
  const model = registry.need(modelName)
  const first = methodSpace(model, { omit: ['inferDiscrete'], first: 'nuts' })
  // Short runs: a sampler that opens with thousands of draws opens with at most a few hundred (the table has its own).
  const starting: Record<string, Readonly<Record<string, number | string | boolean>>> = {}
  for (const kind of first.applicable) {
    const spaces = first.space.options[kind]?.spaces ?? {}
    const cap = (knob: string, most: number): [string, number][] => {
      const space = spaces[knob] as { default?: unknown } | undefined
      return typeof space?.default === 'number' ? [[knob, Math.min(space.default, most)]] : []
    }
    starting[kind] = Object.fromEntries([...cap('numChains', 2), ...cap('numWarmup', 100), ...cap('numDraws', 200)])
  }
  const offer = methodSpace(model, {
    starting: Object.fromEntries(first.applicable.map((k) => [k, { ...starting[k], ...exampleStarting[k] }])),
    omit: ['inferDiscrete'],
    first: 'nuts',
  })
  const options: Record<string, DictSpace> = { ...offer.space.options }
  for (const kind of offer.applicable) {
    const o = options[kind]
    if (o !== undefined && hasStepsKnob(kind))
      options[kind] = dict(
        {
          ...o.spaces,
          steps: int(50, 5000, {
            default: STEPS,
            log: true,
            label: 'steps',
            description: 'iterations of the optimiser',
          }),
        },
        { description: o.description ?? '' },
      )
  }
  return { ...offer, space: { ...offer.space, options, label: 'inference method' } as UnionSpace }
}

/**
 * How many steps a configuration's run is expected to take: what the session's `steps` should be so the run ends where
 * the method does and progress reads right (a sampler's warm-up and draws, a variational method's steps), capped by the
 * step budget.
 *
 * @param config - The example's configuration.
 * @returns The steps.
 * @example
 * ```ts
 * import { exampleSteps } from 'omai-engine/bayesian-examples'
 * exampleSteps({ inference: { method: 'nuts', numWarmup: 100, numDraws: 200 } }) // => 302
 * ```
 */
export function exampleSteps(config: Config): number {
  const { kind, knobs } = inferenceOf(config)
  const own = hasStepsKnob(kind)
    ? exampleNumber(knobs, 'steps') || STEPS
    : isSampler(knobs)
      ? exampleNumber(knobs, 'numWarmup') + exampleNumber(knobs, 'numDraws') + 2
      : 400
  const cap = exampleNumber(config, 'budgetSteps')
  return cap > 0 ? Math.min(own, cap) : own
}

/**
 * The objective of the chosen inference method as a curve on the tape: its path and name (variational inference's ELBO,
 * the Laplace fit's log density), or `undefined` for a method with none to show (a sampler has chains, which the
 * example's diagnostics path shows; expectation propagation and Pathfinder end in a few steps).
 *
 * @param config - The example's configuration.
 * @returns The path and the label.
 * @example
 * ```ts
 * import { exampleObjective } from 'omai-engine/bayesian-examples'
 * exampleObjective({ inference: { method: 'svi' } }) // => { path: 'svi/elbo', label: 'ELBO' }
 * ```
 */
export function exampleObjective(config: Config): { readonly path: string; readonly label: string } | undefined {
  const { kind } = inferenceOf(config)
  if (isVariational(kind)) return { path: 'svi/elbo', label: 'ELBO' }
  if (kind === 'laplace') return { path: 'laplace/logDensity', label: 'log density at the iterate' }
  return undefined
}

/** The draws of a posterior as rows, with the chain and the index within the chain of each. */
export interface DrawRows {
  /** `rows[k]` holds the draw's coordinates: the sites in order, a vector site flattened (a scalar site is one). */
  readonly rows: number[][]
  readonly chain: number[]
  readonly index: number[]
  /** The name of each coordinate (`site`, or `site[j]` for a vector site). */
  readonly names: string[]
}

/**
 * The draws of some sites of a posterior as rows (at most `n`, evenly spaced through the draws), whatever its form: the
 * samples of a chain method, or `n` draws of a fitted normal. `undefined` when a site is missing or a draw is not finite.
 *
 * @param post - The posterior.
 * @param sites - The sites, whose coordinates are joined in this order.
 * @param n - The most draws to keep.
 * @returns The rows, or `undefined`.
 * @example
 * ```ts
 * import { drawRows } from 'omai-engine/bayesian-examples'
 * drawRows({ sites: [] } as never, ['w'], 3) // => undefined
 * ```
 */
export function drawRows(post: Posterior, sites: readonly string[], n: number): DrawRows | undefined {
  try {
    if (!sites.every((s) => post.sites.includes(s))) return undefined
    const stored = post.form === 'samples' ? post.draws : undefined
    const form = stored !== undefined && sites.every((s) => stored[s] !== undefined)
    const drawn = form ? undefined : post.sample(key(1), n)
    const columns: { names: string[]; width: number; flat: ArrayLike<number> }[] = []
    let total = Number.POSITIVE_INFINITY
    for (const site of sites) {
      const tensor = form ? stored?.[site] : drawn?.[site]
      if (tensor === undefined) return undefined
      const width = tensor.shape.length <= 1 ? 1 : tensor.shape.slice(1).reduce((a, b) => a * b, 1)
      const flat = toFlat(tensor)
      total = Math.min(total, Math.floor(flat.length / width))
      columns.push({
        names: tensor.shape.length <= 1 ? [site] : Array.from({ length: width }, (_, j) => `${site}[${j}]`),
        width,
        flat,
      })
    }
    if (total === 0 || !Number.isFinite(total)) return undefined
    const perChain = form && post.chains !== undefined ? post.chains.numDraws : total
    const keep = Math.min(n, total)
    const rows: number[][] = []
    const chain: number[] = []
    const index: number[] = []
    for (let k = 0; k < keep; k++) {
      const i = keep === total ? k : Math.round((k * (total - 1)) / (keep - 1 || 1))
      rows.push(
        columns.flatMap((c) => Array.from({ length: c.width }, (_, j) => c.flat[i * c.width + j] ?? Number.NaN)),
      )
      chain.push(Math.floor(i / perChain))
      index.push(i % perChain)
    }
    return rows.every((r) => r.every(Number.isFinite))
      ? { rows, chain, index, names: columns.flatMap((c) => c.names) }
      : undefined
  } catch {
    return undefined
  }
}

/**
 * The mean and the standard deviation of each column of some rows.
 *
 * @param rows - The rows.
 * @param p - The number of columns.
 * @returns The means and the sample standard deviations.
 * @example
 * ```ts
 * import { columnMoments } from 'omai-engine/bayesian-examples'
 * columnMoments([[1], [3]], 1).mean // => [2]
 * ```
 */
export function columnMoments(
  rows: readonly (readonly number[])[],
  p: number,
): { readonly mean: number[]; readonly sd: number[] } {
  const n = rows.length
  const mean = Array.from({ length: p }, (_, j) => rows.reduce((a, r) => a + (r[j] ?? 0), 0) / n)
  const sd = Array.from({ length: p }, (_, j) =>
    Math.sqrt(rows.reduce((a, r) => a + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / Math.max(1, n - 1)),
  )
  return { mean, sd }
}

/** The state of an example is the inner method's own. */
type State = unknown

/** What the derived paths of an example get: the problem, the configuration, the chosen kind and the posterior reader. */
export interface ExampleContext<P extends ExampleProblem> {
  readonly problem: P
  readonly config: Config
  /** The chosen inference kind (a name of `methodKinds`). */
  readonly kind: string
  /** The posterior a state of the run stands for. */
  readonly posterior: (state: State) => Posterior
  /**
   * The posterior of another run of the same inference method (its knobs) on the problem of another configuration, run to
   * its end on the spot: how a derived path shows a fit to a part of the data (the flips seen so far). The seed is fixed.
   */
  readonly fit: (config: Config) => Posterior
}

/** What an example declares: its knobs, its problem and the derived paths that read the posterior. */
export interface ExampleSpec<P extends ExampleProblem> {
  /** The method's name (`classify`). */
  readonly name: string
  readonly description: string
  /** The name the model is registered under for the inner method (the example's own). */
  readonly modelName: string
  /** The knobs of the data (a generator and its sizes, noise and seed) and of the model (its prior, its link). */
  readonly dataKnobs: DictSpace
  readonly modelKnobs: DictSpace
  /** The knobs the problem depends on (what a reference of it depends on): usually every data and model knob. */
  readonly problemKnobs: readonly string[]
  /** The problem a configuration stands for (a pure function of the problem's knobs). */
  readonly problem: (config: Config) => P
  /** The derived paths, given the problem and the way to read the posterior of a state. */
  readonly derived: (context: ExampleContext<P>) => Readonly<Record<string, DerivedSource<State>>>
}

/** A declared example: its method and what a page needs around it. */
export interface Example<P extends ExampleProblem> {
  /** The trace method: knobs, the chosen method's tape and the derived paths. */
  readonly method: InferenceMethod<State, Config, string>
  readonly dataKnobs: DictSpace
  readonly modelKnobs: DictSpace
  readonly problemKnobs: readonly string[]
  /** The knobs of the budget (the time budget and the step budget). */
  readonly budgetKnobs: DictSpace
  /** Every knob but the budget's. */
  readonly knobs: DictSpace
  /** The problem a configuration stands for. */
  readonly problem: (config: Config) => P
  /**
   * Every method that runs on the problem a configuration stands for (the verdicts are of the model it makes: a link or a
   * prior can change them), as a union of knobs with the reasons for the others; the last answer is kept.
   */
  readonly offer: (config: Config) => ExampleOffer
  /** The steps a run is expected to take, see {@link exampleSteps}. */
  readonly steps: (config: Config) => number
  /** The objective curve of the chosen method, see {@link exampleObjective}. */
  readonly objective: (config: Config) => { readonly path: string; readonly label: string } | undefined
  /**
   * The configuration of a reference run of the same problem: the knobs of the problem kept, the inference replaced by
   * `inference` (a long run), and a generous time budget with no step budget.
   */
  readonly reference: (config: Config, inference: Config, budgetMs?: number) => Config
}

/**
 * Declare an example: the problem from knobs, the chosen inference method delegated to the library's `methodKinds`
 * (rebuilt from the knobs, so only the configuration crosses a worker boundary) and the example's derived paths added
 * to its contract.
 *
 * @param spec - The example's knobs, problem and derived paths.
 * @returns The method and what a page needs around it.
 * @example
 * ```ts
 * import { classifierExample } from 'omai-engine/bayesian-examples'
 * classifierExample.method.name // => 'classify'
 * ```
 */
export function defineExample<P extends ExampleProblem>(spec: ExampleSpec<P>): Example<P> {
  const problemKey = (config: Config): string => JSON.stringify(spec.problemKnobs.map((k) => config[k]))
  const own = dict({ ...spec.dataKnobs.spaces, ...spec.modelKnobs.spaces })
  const offerFor = (config: Config): ExampleOffer => exampleInferenceSpace(spec.problem(config), spec.modelName)
  const offer = offerFor(defaults(own) as Config)
  let offered: { readonly id: string; readonly offer: ExampleOffer } | undefined
  const knobs = dict({ ...own.spaces, inference: offer.space })

  // A built problem with its inner method, kept for the last configuration (the session asks for it several times).
  interface Built {
    readonly problem: P
    readonly inner: AnyInferenceMethod
    readonly innerConfig: Config
    readonly kind: string
  }
  let last: { readonly id: string; readonly built: Built } | undefined
  const construct = (config: Config): Built => {
    const { kind, knobs: given } = inferenceOf(config)
    const problem = spec.problem(config)
    const registry = modelRegistry({ [spec.modelName]: { definition: problem.definition, data: problem.data } })
    const inner = inferenceMethodFor(registry, kind, spec.modelName)
    const { steps: _steps, ...rest } = given
    const fixed = Object.entries(FIXED[kind] ?? {}).filter(([k]) => k in inner.config.spaces)
    return {
      problem,
      inner,
      innerConfig: { ...(defaults(inner.config) as Config), ...Object.fromEntries(fixed), ...rest },
      kind,
    }
  }
  const build = (config: Config): Built => {
    const { kind, knobs: given } = inferenceOf(config)
    const id = JSON.stringify([problemKey(config), kind, given])
    if (last?.id === id) return last.built
    const built = construct(config)
    last = { id, built }
    return built
  }
  const innerPosterior =
    (b: Built) =>
    (state: State): Posterior =>
      (b.inner as InferenceMethod<State, Config, string>).posterior?.(state, b.innerConfig) as Posterior

  const fitOf = (config: Config): Posterior => {
    const b = construct(config)
    const run = session(b.inner as never, {
      config: b.innerConfig as never,
      seed: 1,
      steps: exampleSteps(config),
      mode: 'eager',
    })
    return innerPosterior(b)(run.state)
  }

  const declared: InferenceMethod<State, Config, string> = defineInferenceMethod<State, Config, string>({
    name: spec.name,
    family: 'composite',
    description: spec.description,
    config: dict(knobs.spaces),
    posterior: (state, config) => innerPosterior(build(config))(state),
    defaults: {},
    contract: (config, k): Contract<State, string> => {
      const b = build(config)
      const inner = (
        typeof b.inner.contract === 'function' ? b.inner.contract(b.innerConfig, k) : b.inner.contract
      ) as Contract<State, string>
      return {
        ...inner,
        ...spec.derived({ problem: b.problem, config, kind: b.kind, posterior: innerPosterior(b), fit: fitOf }),
      }
    },
    make: (config, k) => {
      const b = build(config)
      const algorithm = b.inner.make(b.innerConfig, k)
      const steps = hasStepsKnob(b.kind) ? exampleSteps(config) : undefined
      return steps === undefined
        ? algorithm
        : {
            ...algorithm,
            done: (s: State, t: number) => algorithm.done?.(s, t) ?? (t >= steps ? 'converged' : undefined),
          }
    },
  }) as never
  const method: InferenceMethod<State, Config, string> = {
    ...declared,
    config: { ...declared.config, spaces: { ...declared.config.spaces, ...exampleBudgetKnobs.spaces } },
  }
  return {
    method,
    dataKnobs: spec.dataKnobs,
    modelKnobs: spec.modelKnobs,
    problemKnobs: spec.problemKnobs,
    budgetKnobs: exampleBudgetKnobs,
    knobs,
    offer: (config) => {
      const id = problemKey(config)
      if (offered?.id !== id) offered = { id, offer: offerFor(config) }
      return offered.offer
    },
    problem: spec.problem,
    steps: exampleSteps,
    objective: exampleObjective,
    reference: (config, inferenceOverrides, budgetMs = 20000) => ({
      ...config,
      inference: { ...((defaults(method.config) as Config)['inference'] as object), ...inferenceOverrides },
      budgetMs,
      budgetSteps: 0,
    }),
  }
}
