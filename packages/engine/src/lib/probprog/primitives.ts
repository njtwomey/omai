/**
 * The primitives probabilistic programs are written with, NumPyro style: `sample` and `observe` for random variables,
 * `param` for learnable parameters, `deterministic` for recorded values, `factor` for extra log-density terms, `plate`
 * for conditionally independent repetitions and `markov` for loops with sequential dependence. Also the message and
 * handler stack (`send`, `withMessenger`) they all go through.
 *
 * @index function sample, observe and param: the vocabulary programs are written in.
 */

import { OmaiError } from '../base/index.ts'
import { type Distribution, defineDistribution, type Support } from '../dists/index.ts'
import type { Key } from '../random/index.ts'
import {
  asTensor,
  broadcastShapes,
  ones,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { expandDistribution } from './_expand.ts'
import { applyStack, withHandler } from './_stack.ts'

/**
 * A probabilistic program (NumPyro's model): a function of its data that calls {@link sample}, {@link observe},
 * {@link param}, {@link deterministic}, {@link factor}, {@link plate} and {@link markov}. Run plainly it needs a
 * key for its draws; under effect handlers (`seed`, `condition`, `substitute`, `trace`, ...) the same function is
 * simulated, conditioned, scored and inferred.
 *
 * @concept probabilistic-programs
 */
export type Model<D = void> = (data: D) => unknown

/** What a site message is about. */
export type SiteType = 'sample' | 'param' | 'deterministic' | 'plate' | 'markov'

/**
 * One plate around a site: its name, full size, subsample size and the batch dimension it occupies (negative,
 * counted from the right of the batch shape, as in NumPyro).
 */
export interface PlateFrame {
  readonly name: string
  readonly size: number
  readonly subsampleSize: number
  readonly dim: number
}

/** One step of a {@link markov} loop around a site. */
export interface MarkovFrame {
  readonly name: string
  readonly step: number
  readonly history: number
}

/**
 * Inference hints on a sample site: `enumerate: 'parallel'` marks a discrete latent site for exact summation,
 * `isAuxiliary` marks a site that is not a model variable (factors), `blockPlates` names plates that should not
 * broadcast the site, and `method` and `block` (set by `inferHint`) say which inference method and which block of a
 * composite plan the site belongs to. Nothing reads `method` and `block` on its own: they are annotations a plan the
 * user writes can honour.
 */
export interface InferConfig {
  readonly enumerate?: 'parallel' | undefined
  readonly isAuxiliary?: boolean
  readonly blockPlates?: readonly string[]
  readonly method?: string
  readonly block?: string
}

/**
 * The message a primitive sends through the handler stack (NumPyro's msg dict). Handlers read and rewrite it in
 * `process` (on the way in, innermost handler first) and `postprocess` (on the way out); it is mutable only while it
 * travels.
 */
export interface Message {
  type: SiteType
  name: string
  /** The distribution of a sample site. */
  fn: Distribution | undefined
  /** The value: given by `obs`, a handler, or drawn by the default behaviour. */
  value: Value | undefined
  isObserved: boolean
  /** A multiplier of the site's log density (from `scale` and subsampled plates). */
  scale: number
  /** Which elements of the site's log density count (a boolean value), or all of them. */
  mask: Value | undefined
  plates: PlateFrame[]
  markov: MarkovFrame[]
  infer: InferConfig
  /** The key for a draw, set by `seed`. */
  key: Key | undefined
  sampleShape: readonly number[]
  stop: boolean
  /** The initial value of a param site, or a function of a key. */
  init: Value | ((key: Key) => Value) | undefined
  /** The support of a param site. */
  support: Support | undefined
  /** The number of event dimensions of a param site (for slicing under subsampled plates). */
  eventDim: number
  /** Plate size, subsample size and dimension. */
  size: number | undefined
  subsampleSize: number | undefined
  dim: number | undefined
  /** Markov loop length and history. */
  length: number | undefined
  history: number | undefined
  /** A handler that runs a markov loop itself stores the loop's result here and sets `handled`. */
  result: unknown
  handled: boolean
  /** The markov loop body and its initial carry. */
  body: ((t: number, carry: unknown) => unknown) | undefined
  carry: unknown
  /** Whether this sample site is a {@link factor}. */
  isFactor: boolean
  /** The batch dimension given to an enumerated site. */
  enumDim: number | undefined
  /** Handlers that have intervened on the site (`doIntervention`). */
  interveners: readonly object[]
}

/**
 * An effect handler: functions that rewrite messages on the way in (`process`) and observe them on the way out
 * (`postprocess`). Handlers are plain objects made fresh for every call of the program they wrap.
 */
export interface Messenger {
  process?(msg: Message): void
  postprocess?(msg: Message): void
}

/**
 * A new message with default fields.
 *
 * @param type - The site type.
 * @param name - The site name.
 * @param fields - Fields to set.
 * @returns The message.
 * @example
 * ```ts
 * import { message } from 'omai-engine/probprog'
 * message('sample', 'x').scale // => 1
 * ```
 */
export function message(type: SiteType, name: string, fields: Partial<Message> = {}): Message {
  return {
    type,
    name,
    fn: undefined,
    value: undefined,
    isObserved: false,
    scale: 1,
    mask: undefined,
    plates: [],
    markov: [],
    infer: {},
    key: undefined,
    sampleShape: [],
    stop: false,
    init: undefined,
    support: undefined,
    eventDim: 0,
    size: undefined,
    subsampleSize: undefined,
    dim: undefined,
    length: undefined,
    history: undefined,
    result: undefined,
    handled: false,
    body: undefined,
    carry: undefined,
    isFactor: false,
    enumDim: undefined,
    interveners: [],
    ...fields,
  }
}

/**
 * Send a message through the active effect handlers and return it (NumPyro's `apply_stack`): the building block of
 * every primitive, and of custom primitives.
 *
 * @param msg - The message.
 * @returns The processed message.
 * @example
 * ```ts
 * import { message, send } from 'omai-engine/probprog'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(send(message('deterministic', 'd', { value: tensor(2) })).value ?? tensor(0)) // => 2
 * ```
 */
export function send(msg: Message): Message {
  return applyStack(msg)
}

/**
 * Run `fn` with a handler active (pushed on the stack for the duration of the call): how handlers wrap programs.
 *
 * @param handler - The handler.
 * @param fn - The function to run under it.
 * @returns What `fn` returns.
 * @example
 * ```ts
 * import { sample, withMessenger } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { toArray } from 'omai-engine/tensor'
 * const x = withMessenger({ process: (m) => { m.key = key(0) } }, () => sample('x', normal(0, 1)))
 * Number.isFinite(toArray(x) as number) // => true
 * ```
 */
export function withMessenger<T>(handler: Messenger, fn: () => T): T {
  return withHandler(handler, fn)
}

/** Options of {@link sample}. */
export interface SampleOptions {
  /** An observed value: the site is conditioned on it (its log density scores the value). */
  readonly obs?: TensorLike | undefined
  /** Draw this many independent values per batch element (left of the batch shape). */
  readonly sampleShape?: readonly number[]
  readonly infer?: InferConfig
}

function siteValue(msg: Message): Value {
  if (msg.value === undefined) throw new OmaiError('value', `${msg.type} ${msg.name}: no value was produced`)
  return msg.value
}

/**
 * A random variable of a program (NumPyro's `sample`): its distribution, and either an observed value or a draw.
 * Handlers decide what happens: `seed` supplies the key, `condition` and `substitute` the value, `trace` records the
 * site, plates broadcast the distribution.
 *
 * @param name - The site name (unique within one run of the program).
 * @param fn - The distribution.
 * @param options - An observed value, a sample shape and inference hints.
 * @returns The site's value.
 * @throws OmaiError with code `value` when a draw is needed but no handler supplied a key.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { sample, seed, trace } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => sample('x', normal(0, 1), { sampleShape: [3] })
 * trace(seed(model, key(0))).getTrace().sites.get('x')?.value.shape // => [3]
 * ```
 */
export function sample(name: string, fn: Distribution, options: SampleOptions = {}): Value {
  const msg = message('sample', name, {
    fn,
    sampleShape: options.sampleShape ?? [],
    infer: options.infer ?? {},
    ...(options.obs === undefined ? {} : { value: asTensor(options.obs), isObserved: true }),
  })
  return siteValue(applyStack(msg))
}

/**
 * An observed random variable: `sample(name, fn, { obs: value })`.
 *
 * @param name - The site name.
 * @param fn - The distribution of the observation.
 * @param value - The observed value.
 * @param options - Inference hints.
 * @returns The observed value (or a substituted one).
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { observe } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(observe('y', normal(0, 1), 1.5)) // => 1.5
 * ```
 */
export function observe(
  name: string,
  fn: Distribution,
  value: TensorLike,
  options: Omit<SampleOptions, 'obs'> = {},
): Value {
  return sample(name, fn, { ...options, obs: value })
}

/**
 * Record a value computed from other sites (NumPyro's `deterministic`), so traces and posteriors report it.
 *
 * @param name - The site name.
 * @param value - The value.
 * @returns The value (or a substituted one).
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { deterministic, trace } from 'omai-engine/probprog'
 * import { toArray } from 'omai-engine/tensor'
 * const model = () => deterministic('d', [1, 2])
 * toArray(trace(model).getTrace().sites.get('d')?.value ?? [0]) // => [1, 2]
 * ```
 */
export function deterministic(name: string, value: TensorLike): Value {
  return siteValue(applyStack(message('deterministic', name, { value: asTensor(value) })))
}

/** Options of {@link param}. */
export interface ParamOptions {
  /** The set the parameter lives in (default the real line); optimisers work on its unconstrained image. */
  readonly support?: Support
  /** Event dimensions on the right of the value (for slicing under subsampled plates; default 0). */
  readonly eventDim?: number
}

/**
 * A learnable parameter (NumPyro's `param`): an initial value, or an initialiser of a key, replaced by the current
 * value when the program runs under an optimiser (`substitute` with the parameter store).
 *
 * @param name - The site name.
 * @param init - The initial value (constrained), or a function of a key returning it.
 * @param options - Support and event dimensions.
 * @returns The parameter's value.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { param } from 'omai-engine/probprog'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(param('scale', 2, { support: { kind: 'positive' } })) // => 2
 * ```
 */
export function param(name: string, init: TensorLike | ((key: Key) => Value), options: ParamOptions = {}): Value {
  const msg = message('param', name, {
    init: typeof init === 'function' ? init : asTensor(init),
    support: options.support ?? { kind: 'real' },
    eventDim: options.eventDim ?? 0,
  })
  return siteValue(applyStack(msg))
}

function unitDistribution(logFactor: Value): Distribution {
  return defineDistribution({
    name: 'unit',
    params: { logFactor },
    support: { kind: 'real' },
    batchShape: logFactor.shape,
    eventShape: [0],
    logProb: () => logFactor,
    sample: (_key, shape) => zeros([...shape, ...logFactor.shape, 0]),
    mean: () => zeros([...logFactor.shape, 0]),
    variance: () => zeros([...logFactor.shape, 0]),
    mode: () => zeros([...logFactor.shape, 0]),
    entropy: () => zeros(logFactor.shape),
  })
}

/**
 * Add a term to the program's log density (NumPyro's `factor`): an observed site of a `unit` distribution whose log
 * density is the given value, for potentials, soft constraints and hand-written likelihoods.
 *
 * @param name - The site name.
 * @param logFactor - The log factor (any shape; it is summed, and plates broadcast it).
 * @returns Nothing.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { factor, logJoint } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * item(logJoint(() => factor('penalty', [-1, -2]), undefined).logJoint) // => -3
 * ```
 */
export function factor(name: string, logFactor: TensorLike): void {
  const lf = asTensor(logFactor)
  const fn = unitDistribution(lf)
  applyStack(
    message('sample', name, {
      fn,
      value: zeros([...lf.shape, 0]),
      isObserved: true,
      isFactor: true,
      infer: { isAuxiliary: true },
    }),
  )
}

/** Options of {@link plate}. */
export interface PlateOptions {
  /** Draw a random subset of this many indices (scaling log densities by size / subsampleSize). */
  readonly subsampleSize?: number
  /** Use these indices as the subsample. */
  readonly subsample?: TensorLike
  /** The batch dimension (negative); by default the first one free of enclosing plates. */
  readonly dim?: number
}

function batchShapeOf(plates: readonly PlateFrame[]): number[] {
  const depth = plates.reduce((m, f) => Math.max(m, -f.dim), 0)
  const shape = toFlat(ones([depth]))
  for (const f of plates) shape[depth + f.dim] = f.subsampleSize
  return shape
}

function plateHandler(frame: PlateFrame, indices: Tensor<'int32'>): Messenger {
  return {
    process: (msg) => {
      if (msg.type === 'markov') return
      if (msg.infer.blockPlates?.includes(frame.name) === true) return
      if (msg.plates.some((f) => f.name === frame.name)) return
      msg.plates.push(frame)
      if (msg.type !== 'sample') return
      const fn = msg.fn
      if (fn === undefined) return
      const expected = batchShapeOf(msg.plates)
      const own = [...msg.sampleShape, ...fn.batchShape]
      const overlap = Math.max(expected.length - own.length, 0)
      let trailing: number[]
      try {
        trailing = broadcastShapes(expected.slice(overlap), own)
      } catch {
        throw new OmaiError(
          'shape',
          `plate ${frame.name}: site ${msg.name} has batch shape [${own.join(', ')}], which does not fit plates of shape [${expected.join(', ')}]`,
          { site: msg.name, plate: frame.name },
        )
      }
      msg.fn = expandDistribution(fn, [...expected.slice(0, overlap), ...trailing])
      msg.sampleShape = []
      if (frame.subsampleSize !== frame.size) msg.scale *= frame.size / frame.subsampleSize
    },
    postprocess: (msg) => {
      if (msg.type !== 'param' || frame.subsampleSize === frame.size || msg.value === undefined) return
      const axis = frame.dim - msg.eventDim
      const v = msg.value
      if (v.shape.length >= -axis && v.shape[v.shape.length + axis] === frame.size)
        msg.value = take(v, indices, v.shape.length + axis)
    },
  }
}

/**
 * A plate: conditionally independent repetitions (NumPyro's `plate`). Sites inside get the plate's batch dimension
 * (their distributions are broadcast so each element is its own variable), and with `subsampleSize` a random subset of
 * indices is used and log densities are scaled by size / subsampleSize, which is how stochastic variational inference
 * subsamples data.
 *
 * @param name - The plate name.
 * @param size - The number of elements.
 * @param fn - The body, given the (sub)sample indices.
 * @param options - Subsample size or indices, and the batch dimension.
 * @returns What the body returns.
 * @throws OmaiError with code `value` for an invalid size or subsample, or `shape` when a site's batch shape
 *   cannot fit the plates.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { plate, sample, seed, trace } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => plate('data', 4, () => sample('x', normal(0, 1)))
 * trace(seed(model, key(0))).getTrace().sites.get('x')?.value.shape // => [4]
 * ```
 */
export function plate<R>(
  name: string,
  size: number,
  fn: (indices: Tensor<'int32'>) => R,
  options: PlateOptions = {},
): R {
  if (!Number.isInteger(size) || size < 0)
    throw new OmaiError('value', `plate ${name}: size must be a non-negative integer`)
  const given = options.subsample === undefined ? undefined : asTensor(options.subsample)
  const subsampleSize = given?.shape[0] ?? options.subsampleSize ?? size
  if (!Number.isInteger(subsampleSize) || subsampleSize < 0 || subsampleSize > size)
    throw new OmaiError('value', `plate ${name}: subsample size must be an integer in [0, ${size}]`)
  const msg = applyStack(
    message('plate', name, {
      size,
      subsampleSize,
      ...(given === undefined ? {} : { value: given }),
    }),
  )
  const occupied = new Set(msg.plates.map((f) => f.dim))
  let dim = options.dim ?? -1
  if (options.dim === undefined) while (occupied.has(dim)) dim -= 1
  else if (occupied.has(dim) || dim >= 0)
    throw new OmaiError('value', `plate ${name}: dim ${dim} is negative and free of enclosing plates`)
  const indices = siteValue(msg) as Tensor<'int32'>
  const frame: PlateFrame = { name, size, subsampleSize: indices.shape[0] ?? subsampleSize, dim }
  return withHandler(plateHandler(frame, indices), () => fn(indices))
}

/** Options of {@link markov}. */
export interface MarkovOptions {
  /** How many previous steps a step may depend on (default 1); enumeration reuses dimensions beyond it. */
  readonly history?: number
}

/**
 * A loop with sequential dependence (NumPyro's `markov` and `scan`): `step(t, carry)` runs for t = 0, ..., length - 1,
 * each step's sites carrying a markov frame, so analysis sees the chain, enumeration recycles dimensions beyond the
 * history, and sequential Monte Carlo can filter step by step. Site names must be unique per step (`z_${t}`).
 *
 * @param name - The loop name.
 * @param length - The number of steps.
 * @param step - The step: the time and the carry in, the new carry out.
 * @param init - The initial carry.
 * @param options - The history (default 1).
 * @returns The final carry.
 * @throws OmaiError with code `value` for a negative length or history.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { markov, sample, seed, trace } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import type { Value } from 'omai-engine/tensor'
 * const walk = () => markov('t', 3, (t, x: Value | number) => sample(`x${t}`, normal(x, 1)), 0)
 * [...trace(seed(walk, key(0))).getTrace().sites.keys()] // => ['t', 'x0', 'x1', 'x2']
 * ```
 */
export function markov<C>(
  name: string,
  length: number,
  step: (t: number, carry: C) => C,
  init: C,
  options: MarkovOptions = {},
): C {
  const history = options.history ?? 1
  if (!Number.isInteger(length) || length < 0)
    throw new OmaiError('value', `markov ${name}: length must be a non-negative integer`)
  if (!Number.isInteger(history) || history < 0)
    throw new OmaiError('value', `markov ${name}: history must be a non-negative integer`)
  const msg = applyStack(
    message('markov', name, {
      length,
      history,
      body: step as (t: number, carry: unknown) => unknown,
      carry: init,
    }),
  )
  if (msg.handled) return msg.result as C
  let carry = init
  for (let t = 0; t < length; t++) {
    const frame: MarkovFrame = { name, step: t, history }
    carry = withHandler(
      {
        process: (m) => {
          if (m.type !== 'plate' && m.type !== 'markov') m.markov.push(frame)
        },
      },
      () => step(t, carry),
    )
  }
  return carry
}
