/**
 * Effect handlers, the mechanism that gives a probabilistic program its meanings, NumPyro style: each handler
 * intercepts the messages a program's sites send and rewrites or records them. Tracing, seeding, conditioning,
 * substituting and replaying values, blocking, scaling and masking log densities, Pearl's do-operator, enumeration
 * marks for discrete sites, `uncondition`, and `handle` for custom handlers.
 */

import { WhiteboxError } from '../base/index.ts'
import type { Distribution, Support } from '../dists/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { asTensor, logicalAnd, type TensorLike, type Value } from '../tensor/index.ts'
import { applyStack, withHandler } from './_stack.ts'
import type { InferConfig, MarkovFrame, Message, Messenger, Model, PlateFrame } from './primitives.ts'

/** A recorded sample site: its distribution (after plate broadcasting), value and how it is scored. */
export interface SampleSite {
  readonly type: 'sample'
  readonly name: string
  readonly fn: Distribution
  readonly value: Value
  readonly isObserved: boolean
  readonly scale: number
  readonly mask: Value | undefined
  readonly plates: readonly PlateFrame[]
  readonly markov: readonly MarkovFrame[]
  readonly infer: InferConfig
  readonly isFactor: boolean
  /** The batch dimension of an enumerated site (its value holds every state along it). */
  readonly enumDim: number | undefined
}

/** A recorded param site. */
export interface ParamSite {
  readonly type: 'param'
  readonly name: string
  readonly value: Value
  readonly support: Support
  readonly eventDim: number
  readonly plates: readonly PlateFrame[]
  readonly markov: readonly MarkovFrame[]
}

/** A recorded deterministic site. */
export interface DeterministicSite {
  readonly type: 'deterministic'
  readonly name: string
  readonly value: Value
  readonly plates: readonly PlateFrame[]
  readonly markov: readonly MarkovFrame[]
}

/** A recorded plate: its size, subsample size, dimension and the subsample indices it used. */
export interface PlateSite {
  readonly type: 'plate'
  readonly name: string
  readonly size: number
  readonly subsampleSize: number
  readonly value: Value
  /** The enclosing plates. */
  readonly plates: readonly PlateFrame[]
}

/** A recorded markov loop. */
export interface MarkovSite {
  readonly type: 'markov'
  readonly name: string
  readonly length: number
  readonly history: number
  readonly plates: readonly PlateFrame[]
  readonly markov: readonly MarkovFrame[]
}

/** Any recorded site. */
export type Site = SampleSite | ParamSite | DeterministicSite | PlateSite | MarkovSite

/** One run of a program: its sites in execution order and its return value. */
export interface ProgramTrace {
  readonly sites: ReadonlyMap<string, Site>
  readonly returnValue: unknown
}

/** A program wrapped by {@link trace}: callable as the program, plus `getTrace`. */
export interface TracedModel<D> {
  (data: D): unknown
  getTrace(data: D): ProgramTrace
}

function wrap<D = void>(model: Model<D>, make: () => Messenger): Model<D> {
  return (data: D) => withHandler(make(), () => model(data))
}

function record(msg: Message): Site | undefined {
  const plates = [...msg.plates]
  const markov = [...msg.markov]
  switch (msg.type) {
    case 'sample':
      if (msg.fn === undefined || msg.value === undefined) return undefined
      return {
        type: 'sample',
        name: msg.name,
        fn: msg.fn,
        value: msg.value,
        isObserved: msg.isObserved,
        scale: msg.scale,
        mask: msg.mask,
        plates,
        markov,
        infer: { ...msg.infer },
        isFactor: msg.isFactor,
        enumDim: msg.enumDim,
      }
    case 'param':
      if (msg.value === undefined) return undefined
      return {
        type: 'param',
        name: msg.name,
        value: msg.value,
        support: msg.support ?? { kind: 'real' },
        eventDim: msg.eventDim,
        plates,
        markov,
      }
    case 'deterministic':
      if (msg.value === undefined) return undefined
      return { type: 'deterministic', name: msg.name, value: msg.value, plates, markov }
    case 'plate':
      if (msg.value === undefined) return undefined
      return {
        type: 'plate',
        name: msg.name,
        size: msg.size ?? 0,
        subsampleSize: msg.subsampleSize ?? msg.size ?? 0,
        value: msg.value,
        plates,
      }
    case 'markov':
      return {
        type: 'markov',
        name: msg.name,
        length: msg.length ?? 0,
        history: msg.history ?? 1,
        plates,
        markov,
      }
  }
}

/**
 * Record every site of a run (NumPyro's `trace`). The wrapped program runs as usual; `getTrace(data)` runs it and
 * returns its sites in execution order, with the return value.
 *
 * @param model - The program.
 * @returns The traced program.
 * @throws WhiteboxError with code `value` when two sample or deterministic sites share a name.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { sample, seed, trace } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const model = () => { const mu = sample('mu', normal(0, 1)); sample('y', normal(mu, 1), { obs: 0.5 }) }
 * const t = trace(seed(model, key(0))).getTrace()
 * [...t.sites.keys()] // => ['mu', 'y']
 * t.sites.get('y')?.type === 'sample' && t.sites.get('y')?.isObserved // => true
 * ```
 */
export function trace<D = void>(model: Model<D>): TracedModel<D> {
  const getTrace = (data: D): ProgramTrace => {
    // Messages are kept by reference and recorded when the run ends, so handlers inside this one that finish a
    // message in their own postprocess (plates slicing params) are seen, as in NumPyro.
    const messages = new Map<string, Message>()
    const handler: Messenger = {
      postprocess: (msg) => {
        if ((msg.type === 'sample' || msg.type === 'deterministic') && messages.has(msg.name))
          throw new WhiteboxError('value', `trace: the site name ${msg.name} is used twice`, { site: msg.name })
        messages.delete(msg.name)
        messages.set(msg.name, msg)
      },
    }
    const returnValue = withHandler(handler, () => model(data))
    const sites = new Map<string, Site>()
    for (const [name, msg] of messages) {
      const site = record(msg)
      if (site !== undefined) sites.set(name, site)
    }
    return { sites, returnValue }
  }
  const run = (data: D): unknown => getTrace(data).returnValue
  return Object.assign(run, { getTrace })
}

/**
 * Run a program and return its trace: `trace(model).getTrace(data)`.
 *
 * @param model - The program.
 * @param data - Its data.
 * @returns The trace.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { deterministic, traceOf } from 'whitebox-engine/probprog'
 * traceOf(() => deterministic('d', 1), undefined).sites.size // => 1
 * ```
 */
export function traceOf<D = void>(model: Model<D>, data: D): ProgramTrace {
  return trace(model).getTrace(data)
}

/**
 * Supply keys to a program's draws (NumPyro's `seed`): the i-th draw of a run uses `foldIn(key, i)`, so every call of
 * the seeded program makes the same draws.
 *
 * @param model - The program.
 * @param key - The key.
 * @returns The seeded program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { sample, seed } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item, type Value } from 'whitebox-engine/tensor'
 * const draw = seed(() => sample('x', normal(0, 1)), key(3))
 * item(draw() as Value) === item(draw() as Value) // => true
 * ```
 */
export function seed<D = void>(model: Model<D>, key: Key): Model<D> {
  return wrap(model, () => {
    let count = 0
    return {
      process: (msg) => {
        if (msg.key !== undefined || msg.value !== undefined) return
        const needs =
          (msg.type === 'sample' && !msg.isObserved) ||
          (msg.type === 'param' && typeof msg.init === 'function') ||
          (msg.type === 'plate' && (msg.subsampleSize ?? 0) < (msg.size ?? 0))
        if (!needs) return
        msg.key = foldIn(key, count)
        count += 1
      },
    }
  })
}

/** Values by site name, or a function of the message returning a value (or undefined to leave the site alone). */
export type SiteValues = Readonly<Record<string, TensorLike | undefined>> | ((msg: Message) => TensorLike | undefined)

function lookup(values: SiteValues, msg: Message): Value | undefined {
  const v = typeof values === 'function' ? values(msg) : values[msg.name]
  return v === undefined ? undefined : asTensor(v)
}

/**
 * Condition sample sites on values (NumPyro's `condition`): the named sites become observed with the given values.
 *
 * @param model - The program.
 * @param data - Values by site name, or a function of the message.
 * @returns The conditioned program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { condition, sample, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * const site = traceOf(condition(() => sample('x', normal(0, 1)), { x: 2 }), undefined).sites.get('x')
 * site?.type === 'sample' && site.isObserved // => true
 * ```
 */
export function condition<D = void>(model: Model<D>, data: SiteValues): Model<D> {
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample' || msg.isObserved) return
      const v = lookup(data, msg)
      if (v === undefined) return
      msg.value = v
      msg.isObserved = true
    },
  }))
}

/**
 * Substitute values at sites (NumPyro's `substitute`): sample sites keep their latent status but take the given
 * values; param and deterministic sites take them too, and plates take subsample indices.
 *
 * @param model - The program.
 * @param data - Values by site name, or a function of the message.
 * @returns The program with substituted values.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { sample, substitute, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * const site = traceOf(substitute(() => sample('x', normal(0, 1)), { x: 2 }), undefined).sites.get('x')
 * site?.type === 'sample' && !site.isObserved && toArray(site.value) === 2 // => true
 * ```
 */
export function substitute<D = void>(model: Model<D>, data: SiteValues): Model<D> {
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type === 'markov' || (msg.type === 'sample' && msg.isObserved)) return
      const v = lookup(data, msg)
      if (v !== undefined) msg.value = v
    },
  }))
}

/**
 * Replay latent sample sites (and plate subsamples) from a trace (NumPyro's `replay`): how a model is run against the
 * draws of a guide.
 *
 * @param model - The program.
 * @param guideTrace - The trace to take values from.
 * @returns The replaying program.
 * @throws WhiteboxError with code `value` when a replayed site has another type in the trace.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { replay, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item } from 'whitebox-engine/tensor'
 * const model = () => sample('x', normal(0, 1))
 * const first = traceOf(seed(model, key(1)), undefined)
 * const again = traceOf(replay(model, first), undefined)
 * item(again.sites.get('x')?.value ?? 0) === item(first.sites.get('x')?.value ?? 1) // => true
 * ```
 */
export function replay<D = void>(model: Model<D>, guideTrace: ProgramTrace): Model<D> {
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample' && msg.type !== 'plate') return
      if (msg.type === 'sample' && msg.isObserved) return
      const site = guideTrace.sites.get(msg.name)
      if (site === undefined) return
      if (site.type !== msg.type)
        throw new WhiteboxError(
          'value',
          `replay: site ${msg.name} is a ${msg.type} here but a ${site.type} in the trace`,
        )
      msg.value = site.value
      if (site.type === 'sample') msg.infer = { ...site.infer }
    },
  }))
}

/** Options of {@link block}: which sites outer handlers stop seeing (default all of them). */
export interface BlockOptions {
  readonly hide?: readonly string[]
  readonly expose?: readonly string[]
  readonly hideFn?: (msg: Message) => boolean
  readonly exposeTypes?: readonly Message['type'][]
}

/**
 * Hide sites from outer handlers (NumPyro's `block`): hidden sites still run, but handlers outside the block (an
 * outer `trace`, `condition` or `seed`) never see them.
 *
 * @param model - The program.
 * @param options - Sites to hide or expose, a predicate, or types to expose; by default every site is hidden.
 * @returns The program with blocked sites.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { block, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const model = () => { sample('a', normal(0, 1)); sample('b', normal(0, 1)) }
 * [...traceOf(block(seed(model, key(0)), { hide: ['a'] }), undefined).sites.keys()] // => ['b']
 * ```
 */
export function block<D = void>(model: Model<D>, options: BlockOptions = {}): Model<D> {
  const hidden = (msg: Message): boolean => {
    if (options.hideFn !== undefined) return options.hideFn(msg)
    if (options.exposeTypes?.includes(msg.type) === true) return false
    if (options.hide !== undefined) return options.hide.includes(msg.name)
    if (options.expose !== undefined) return !options.expose.includes(msg.name)
    return true
  }
  return wrap(model, () => ({
    process: (msg) => {
      if (hidden(msg)) msg.stop = true
    },
  }))
}

/**
 * Scale the log densities of sample sites (NumPyro's `scale`): a tempered likelihood, or a correction for
 * subsampled data.
 *
 * @param model - The program.
 * @param factor - A positive multiplier.
 * @returns The scaled program.
 * @throws WhiteboxError with code `value` for a factor that is not positive.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { logJoint, observe, scale } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const model = () => { observe('y', normal(0, 1), 0) }
 * item(logJoint(scale(model, 2), undefined).logJoint) === 2 * item(logJoint(model, undefined).logJoint) // => true
 * ```
 */
export function scale<D = void>(model: Model<D>, factor: number): Model<D> {
  if (!(factor > 0)) throw new WhiteboxError('value', `scale: the factor must be positive, got ${factor}`)
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type === 'sample') msg.scale *= factor
    },
  }))
}

/**
 * Mask the log densities of sample sites (NumPyro's `mask`): only elements where the mask is true count, which is how
 * missing observations are skipped.
 *
 * @param model - The program.
 * @param mask - A boolean, or a boolean value broadcasting against each site's batch shape.
 * @returns The masked program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { logJoint, mask, observe, plate } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const model = () => plate('n', 2, () => observe('y', normal(0, 1), [0, 0]))
 * const half = item(logJoint(mask(model, [true, false]), undefined).logJoint)
 * Math.abs(half - item(logJoint(model, undefined).logJoint) / 2) < 1e-15 // => true
 * ```
 */
export function mask<D = void>(model: Model<D>, mask: boolean | TensorLike): Model<D> {
  const m = asTensor(mask)
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample') return
      msg.mask = msg.mask === undefined ? m : logicalAnd(msg.mask, m)
    },
  }))
}

/**
 * Intervene on sites (Pearl's do-operator, NumPyro's `do`): each named site still draws its own value (recorded as
 * usual), but the program downstream sees the intervention value instead, so its descendants follow the mutilated
 * model while its ancestors are unaffected.
 *
 * @param model - The program.
 * @param data - Intervention values by site name.
 * @returns The intervened program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { doIntervention, sample, seed } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * import { item, type Value } from 'whitebox-engine/tensor'
 * const model = () => { const z = sample('z', normal(0, 1)); return sample('y', normal(z, 1e-9)) }
 * Math.abs(item(seed(doIntervention(model, { z: 5 }), key(0))() as Value) - 5) < 1e-6 // => true
 * ```
 */
export function doIntervention<D = void>(model: Model<D>, data: Readonly<Record<string, TensorLike>>): Model<D> {
  return wrap(model, () => {
    const id = {}
    return {
      process: (msg) => {
        if (msg.type !== 'sample' || msg.interveners.includes(id)) return
        const v = data[msg.name]
        if (v === undefined) return
        msg.interveners = [...msg.interveners, id]
        const original = { ...msg, plates: [...msg.plates], markov: [...msg.markov] }
        applyStack(original)
        msg.name = `${msg.name}__CF`
        msg.value = asTensor(v)
        msg.isObserved = true
        msg.stop = true
      },
    }
  })
}

/** Options of {@link enumerate}. */
export interface EnumerateOptions {
  /** Only these sites (default every discrete latent site with finite support). */
  readonly sites?: readonly string[]
}

/**
 * Whether a support is discrete with finitely many values (so a site on it can be enumerated).
 *
 * @param support - The support.
 * @returns True for integer intervals and booleans.
 * @example
 * ```ts
 * import { isFiniteDiscrete } from 'whitebox-engine/probprog'
 * isFiniteDiscrete({ kind: 'boolean' }) // => true
 * ```
 */
export function isFiniteDiscrete(support: Support): boolean {
  if (support.kind === 'boolean') return true
  return support.kind === 'integerInterval' && Number.isFinite(support.low) && Number.isFinite(support.high)
}

/**
 * Whether a support is discrete (integers, booleans, counts or one-hot vectors), finite or not.
 *
 * @param support - The support.
 * @returns True for discrete supports.
 * @example
 * ```ts
 * import { isDiscrete } from 'whitebox-engine/probprog'
 * isDiscrete({ kind: 'integerInterval', low: 0, high: Infinity }) // => true
 * ```
 */
export function isDiscrete(support: Support): boolean {
  return ['boolean', 'integerInterval', 'counts', 'oneHot'].includes(support.kind)
}

/**
 * Mark discrete latent sites for exact summation (NumPyro's `config_enumerate`): marked sites get
 * `infer.enumerate = 'parallel'`, and every density, ELBO and sampler then sums them out exactly instead of sampling
 * them.
 *
 * @param model - The program.
 * @param options - The sites to mark (default every latent site with a finite discrete support).
 * @returns The marked program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { enumerate, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { bernoulli } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const site = traceOf(seed(enumerate(() => sample('c', bernoulli({ probs: 0.3 }))), key(0)), undefined).sites.get('c')
 * site?.type === 'sample' ? site.infer.enumerate : undefined // => 'parallel'
 * ```
 */
export function enumerate<D = void>(model: Model<D>, options: EnumerateOptions = {}): Model<D> {
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample' || msg.isObserved || msg.fn === undefined || msg.isFactor) return
      if (options.sites !== undefined ? !options.sites.includes(msg.name) : !isFiniteDiscrete(msg.fn.support)) return
      if (msg.infer.enumerate === undefined) msg.infer = { ...msg.infer, enumerate: 'parallel' }
    },
  }))
}

/**
 * Turn observed sample sites back into latent ones (Pyro's `uncondition`), so a program draws replicated data: the
 * basis of prior and posterior predictive checks.
 *
 * @param model - The program.
 * @param options - The sites to uncondition (default every observed site except factors).
 * @returns The unconditioned program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { observe, seed, traceOf, uncondition } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const site = traceOf(seed(uncondition(() => observe('y', normal(0, 1), 3)), key(0)), undefined).sites.get('y')
 * site?.type === 'sample' && !site.isObserved // => true
 * ```
 */
export function uncondition<D = void>(model: Model<D>, options: { readonly sites?: readonly string[] } = {}): Model<D> {
  return wrap(model, () => ({
    process: (msg) => {
      if (msg.type !== 'sample' || !msg.isObserved || msg.isFactor) return
      if (options.sites !== undefined && !options.sites.includes(msg.name)) return
      msg.isObserved = false
      msg.value = undefined
    },
  }))
}

/**
 * Wrap a program with a custom handler made fresh for each call (the general form of every handler above).
 *
 * @param model - The program.
 * @param make - Builds the handler for one call.
 * @returns The handled program.
 * @concept effect-handlers
 * @example
 * ```ts
 * import { handle, sample, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { tensor } from 'whitebox-engine/tensor'
 * const fixed = handle(() => sample('x', normal(0, 1)), () => ({ process: (m) => { m.value = tensor(1) } }))
 * traceOf(fixed, undefined).sites.size // => 1
 * ```
 */
export function handle<D = void>(model: Model<D>, make: () => Messenger): Model<D> {
  return wrap(model, make)
}
