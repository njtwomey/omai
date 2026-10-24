import { quantileOf, WhiteboxError } from '../base/index.ts'
import { bijectors, type Support } from '../dists/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../random/index.ts'
import { type LogDensity, target } from '../sampling/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  concat,
  fromStorage,
  isfinite,
  isValue,
  item,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  contractFactors,
  enumerationHandler,
  hasEnumerated,
  maskedLogProb,
  maxPlateNesting,
  traceFactors,
} from './_contract.ts'
import {
  enumerate,
  handle,
  isDiscrete,
  isFiniteDiscrete,
  type ProgramTrace,
  type SampleSite,
  seed,
  substitute,
  traceOf,
} from './handlers.ts'
import type { Message, Model } from './primitives.ts'

type Bijector = bijectors.Bijector

const PLATE_GUESS = 8

/** Options of {@link logJoint}. */
export interface LogJointOptions {
  /** A key for latent sites without values (by default every latent site needs a value). */
  readonly key?: Key
  /** The plate nesting, when known (saves a tracing pass for programs with enumerated sites). */
  readonly maxPlateNesting?: number
}

/** The result of {@link logJoint}. */
export interface LogJointResult {
  /** log p(values, observations), with enumerated sites summed out. */
  readonly logJoint: Value
  readonly trace: ProgramTrace
}

/**
 * The log joint density of a program at given latent values (NumPyro's `log_density`): the sum over sample sites of
 * scale x masked log density, with sites marked for enumeration summed out exactly by plated sum-product contraction.
 * Differentiable in the values and in any parameters the program uses.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - Values of latent sites (and params) by name.
 * @param options - A key for any latent site left without a value, and the plate nesting.
 * @returns The log joint and the trace it was computed from.
 * @throws WhiteboxError with code `value` when a latent site has no value and no key was given.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { logJoint, observe, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * // log N(0; 0, 1) + log N(1; 0, 1) = -log(2 pi) - 1/2
 * Math.abs(item(logJoint(model, undefined, { mu: 0 }).logJoint) + Math.log(2 * Math.PI) + 0.5) < 1e-14 // => true
 * ```
 */
export function logJoint<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>> = {},
  options: LogJointOptions = {},
): LogJointResult {
  let m = Object.keys(values).length === 0 ? model : substitute(model, values)
  if (options.key !== undefined) m = seed(m, options.key)
  const run = (depth: number): ProgramTrace =>
    traceOf(
      handle(m, () => enumerationHandler(-depth - 1)),
      data,
    )
  let depth = options.maxPlateNesting ?? PLATE_GUESS
  let t = run(depth)
  if (options.maxPlateNesting === undefined && hasEnumerated(t)) {
    const exact = maxPlateNesting(t)
    if (exact !== depth) {
      depth = exact
      t = run(depth)
    }
  }
  return { logJoint: traceLogJoint(t, depth), trace: t }
}

/**
 * The log joint of a recorded trace: sum of scaled, masked site log densities, contracting enumerated sites.
 *
 * @param t - The trace.
 * @param plateDepth - The plate nesting enumeration dimensions were placed beyond (default from the trace).
 * @returns The log joint.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { observe, traceLogJoint, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const t = traceOf(() => observe('y', normal(0, 1), 0), undefined)
 * Math.abs(item(traceLogJoint(t)) + 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function traceLogJoint(t: ProgramTrace, plateDepth: number = maxPlateNesting(t)): Value {
  if (hasEnumerated(t)) {
    const { factors, sumVars, scales } = traceFactors(t, plateDepth)
    return contractFactors(factors, sumVars, scales)
  }
  let total: Value = tensor(0)
  for (const site of t.sites.values()) {
    if (site.type !== 'sample') continue
    const s = sum(maskedLogProb(site))
    total = add(total, site.scale === 1 ? s : mul(s, site.scale))
  }
  return total
}

/**
 * The scaled, masked log density of every sample site of a trace (NumPyro's `compute_log_probs`), summed per site.
 *
 * @param t - The trace (without enumerated sites).
 * @returns The log density by site name.
 * @concept probabilistic-programs
 * @example
 * ```ts
 * import { observe, siteLogProbs, traceOf } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * const t = traceOf(() => observe('y', normal(0, 1), 0), undefined)
 * Object.keys(siteLogProbs(t)) // => ['y']
 * ```
 */
export function siteLogProbs(t: ProgramTrace): Record<string, Value> {
  const out: Record<string, Value> = {}
  for (const site of t.sites.values()) {
    if (site.type !== 'sample') continue
    const s = sum(maskedLogProb(site))
    out[site.name] = site.scale === 1 ? s : mul(s, site.scale)
  }
  return out
}

/** Where one latent site sits in the unconstrained vector, with its shapes and bijector. */
export interface LatentLayout {
  readonly name: string
  /** The shape of the constrained value. */
  readonly shape: readonly number[]
  /** The shape of the unconstrained value. */
  readonly unconstrainedShape: readonly number[]
  readonly support: Support
  /** From the unconstrained to the constrained value (at the prototype trace). */
  readonly bijector: Bijector
  readonly offset: number
  readonly size: number
}

/** A program's posterior density on unconstrained reals, as samplers and optimisers need it. */
export interface UnconstrainedModel {
  readonly dim: number
  readonly layout: readonly LatentLayout[]
  /** Discrete latent sites summed out by enumeration. */
  readonly enumerated: readonly string[]
  readonly maxPlateNesting: number
  /** The sampler target: log p(constrain(u), data) + log |det d constrain / du|. */
  readonly logDensity: LogDensity
  /** The same log density as a differentiable function of a `[dim]` value. */
  fn(u: Value): Value
  /** The potential energy -log density (NumPyro's `potential_energy`); NaN reads as +Infinity. */
  potential(u: ArrayLike<number>): number
  /** Constrained values of every latent and deterministic site at u (NumPyro's `constrain_fn`). */
  constrain(u: ArrayLike<number> | Value): Record<string, Value>
  /** The unconstrained vector of latent values (NumPyro's `unconstrain_fn`). */
  unconstrain(values: Readonly<Record<string, TensorLike>>): Float64Array
}

/** Options of {@link potentialEnergy}. */
export interface PotentialOptions {
  /** The key for the prototype run (default key(0); only shapes are taken from it). */
  readonly key?: Key
  /** Sum finite discrete latent sites out by enumeration (default true); otherwise they are an error. */
  readonly enumerate?: boolean
}

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

function coordinateNames(layout: readonly LatentLayout[]): string[] {
  const names: string[] = []
  for (const l of layout) {
    if (l.size === 1 && l.unconstrainedShape.length === 0) names.push(l.name)
    else for (let i = 0; i < l.size; i++) names.push(`${l.name}[${i}]`)
  }
  return names
}

/**
 * The latent sites a sampler or optimiser moves, from a prototype trace: continuous latent sample sites in execution
 * order, with their unconstrained shapes.
 *
 * @param t - A trace of the program.
 * @returns The layout and the enumerated sites.
 * @throws WhiteboxError with code `value` for discrete latent sites that are not marked for enumeration.
 * @example
 * ```ts
 * import { latentLayout, sample, seed, traceOf } from 'whitebox-engine/probprog'
 * import { dirichlet } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const t = traceOf(seed(() => sample('p', dirichlet([1, 1, 1])), key(0)), undefined)
 * latentLayout(t).layout[0]?.unconstrainedShape // => [2]
 * ```
 */
export function latentLayout(t: ProgramTrace): {
  readonly layout: LatentLayout[]
  readonly enumerated: string[]
} {
  const layout: LatentLayout[] = []
  const enumerated: string[] = []
  let offset = 0
  for (const site of t.sites.values()) {
    if (site.type !== 'sample' || site.isObserved) continue
    if (isDiscrete(site.fn.support)) {
      if (site.infer.enumerate === 'parallel' && isFiniteDiscrete(site.fn.support)) {
        enumerated.push(site.name)
        continue
      }
      throw new WhiteboxError(
        'value',
        isFiniteDiscrete(site.fn.support)
          ? `site ${site.name} is a discrete latent variable; mark it with enumerate(...) to sum it out`
          : `site ${site.name} is a discrete latent variable with unbounded support, which cannot be summed out`,
        { site: site.name },
      )
    }
    const shape = [...site.value.shape]
    const bijector = bijectors.bijectorForSupport(site.fn.support)
    const unconstrainedShape = bijector.inverseShape(shape)
    const size = product(unconstrainedShape)
    layout.push({ name: site.name, shape, unconstrainedShape, support: site.fn.support, bijector, offset, size })
    offset += size
  }
  return { layout, enumerated }
}

/**
 * A program's unconstrained log density (NumPyro's `potential_energy` and `initialize_model`): every continuous latent
 * site is reached from unconstrained reals through the bijector of its support (re-evaluated at each run, so supports
 * that depend on other sites work), the log-Jacobian is added (scaled like the site), and discrete latent sites are
 * summed out by enumeration. The result is a sampling target for HMC and NUTS and an objective for MAP and Laplace.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The prototype key and whether to enumerate discrete sites.
 * @returns The unconstrained model.
 * @throws WhiteboxError with code `value` for discrete latent sites when enumeration is off, or for a program without
 *   continuous latent sites.
 * @concept log-density
 * @example
 * ```ts
 * import { observe, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { exponential, normal } from 'whitebox-engine/dists'
 * const model = () => { const s = sample('s', exponential(1)); observe('y', normal(0, s), 0) }
 * const pe = potentialEnergy(model, undefined)
 * // At u = 0: s = 1, so -(log Exp(1; 1) + log N(0; 0, 1) + log|ds/du|) = 1 + log(2 pi) / 2.
 * Math.abs(pe.potential([0]) - (1 + 0.5 * Math.log(2 * Math.PI))) < 1e-14 // => true
 * ```
 */
export function potentialEnergy<D = void>(
  model: Model<D>,
  data: D,
  options: PotentialOptions = {},
): UnconstrainedModel {
  const m = options.enumerate === false ? model : enumerate(model)
  const proto = traceOf(seed(m, options.key ?? makeKey(0)), data)
  const depth = maxPlateNesting(proto)
  const { layout, enumerated } = latentLayout(proto)
  const dim = layout.reduce((a, l) => a + l.size, 0)
  const byName = new Map(layout.map((l) => [l.name, l]))

  const runAt = (u: Value): { readonly logDensity: Value; readonly values: Record<string, Value> } => {
    let ldj: Value = tensor(0)
    const values: Record<string, Value> = {}
    const placed = handle(m, () => ({
      process: (msg: Message) => {
        if (msg.type !== 'sample' || msg.isObserved || msg.value !== undefined || msg.fn === undefined) return
        const l = byName.get(msg.name)
        if (l === undefined) return
        const piece = reshape(sliceAxis(u, 0, l.offset, l.offset + l.size), l.unconstrainedShape)
        const b = bijectors.bijectorForSupport(msg.fn.support)
        const x = b.forward(piece)
        const j = sum(b.forwardLogDetJacobian(piece))
        ldj = add(ldj, msg.scale === 1 ? j : mul(j, msg.scale))
        msg.value = x
        values[msg.name] = x
      },
    }))
    const r = logJoint(placed, data, {}, { maxPlateNesting: depth })
    for (const site of r.trace.sites.values())
      if (site.type === 'deterministic' && values[site.name] === undefined) values[site.name] = site.value
    return { logDensity: add(r.logJoint, ldj), values }
  }
  const fn = (u: Value): Value => runAt(u).logDensity
  const asVector = (u: ArrayLike<number> | Value): Value =>
    isValue(u) ? u : fromStorage(Float64Array.from(u), [dim], 'float64')
  const unconstrain = (values: Readonly<Record<string, TensorLike>>): Float64Array => {
    const out = new Float64Array(dim)
    for (const l of layout) {
      const v = values[l.name]
      if (v === undefined) throw new WhiteboxError('value', `unconstrain: no value for site ${l.name}`)
      const x = broadcastTo(asTensor(v), l.shape)
      out.set(toFlat(primalOf(l.bijector.inverse(x))), l.offset)
    }
    return out
  }
  const constrainFlat = (x: ArrayLike<number>): Float64Array => {
    const values = runAt(asVector(x)).values
    return Float64Array.from(layout.flatMap((l) => toFlat(primalOf(values[l.name] ?? tensor(Number.NaN)))))
  }
  const unconstrainFlat = (y: ArrayLike<number>): Float64Array => {
    const record: Record<string, Value> = {}
    let at = 0
    for (const l of layout) {
      const n = product(l.shape)
      record[l.name] = reshape(fromStorage(Float64Array.from(Array.from(y).slice(at, at + n)), [n], 'float64'), l.shape)
      at += n
    }
    return unconstrain(record)
  }
  if (dim === 0)
    throw new WhiteboxError('value', 'potentialEnergy: the program has no continuous latent sites', {
      enumerated: enumerated.length,
    })
  const base = target(fn, dim, { names: coordinateNames(layout) })
  const logDensity: LogDensity = { ...base, constrain: constrainFlat, unconstrain: unconstrainFlat }
  return {
    dim,
    layout,
    enumerated,
    maxPlateNesting: depth,
    logDensity,
    fn,
    potential: (u) => {
      const v = logDensity.logDensity(u)
      return Number.isNaN(v) ? Number.POSITIVE_INFINITY : -v
    },
    constrain: (u) => runAt(asVector(u)).values,
    unconstrain,
  }
}

/**
 * Map unconstrained values of latent sites to constrained ones by running the program (NumPyro's `constrain_fn`), so
 * supports that depend on other sites are respected.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - Unconstrained values by site name.
 * @returns Constrained values of those sites and every deterministic site.
 * @concept log-density
 * @example
 * ```ts
 * import { constrainSites, sample } from 'whitebox-engine/probprog'
 * import { gamma } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(constrainSites(() => sample('s', gamma(2, 1)), undefined, { s: 0 })['s'] ?? 0) // => 1
 * ```
 */
export function constrainSites<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>>,
): Record<string, Value> {
  const out: Record<string, Value> = {}
  const placed = handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type !== 'sample' || msg.isObserved || msg.fn === undefined) return
      const u = values[msg.name]
      if (u === undefined) return
      const x = bijectors.bijectorForSupport(msg.fn.support).forward(u)
      msg.value = x
      out[msg.name] = x
    },
  }))
  const t = traceOf(placed, data)
  for (const site of t.sites.values()) if (site.type === 'deterministic') out[site.name] = site.value
  return out
}

/**
 * Map constrained values of latent sites to unconstrained reals by running the program (NumPyro's `unconstrain_fn`).
 *
 * @param model - The program.
 * @param data - Its data.
 * @param values - Constrained values by site name.
 * @returns Unconstrained values of those sites.
 * @concept log-density
 * @example
 * ```ts
 * import { sample, unconstrainSites } from 'whitebox-engine/probprog'
 * import { gamma } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(unconstrainSites(() => sample('s', gamma(2, 1)), undefined, { s: 1 })['s'] ?? 1) // => 0
 * ```
 */
export function unconstrainSites<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, TensorLike>>,
): Record<string, Value> {
  const out: Record<string, Value> = {}
  const placed = handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type !== 'sample' || msg.isObserved || msg.fn === undefined) return
      const x = values[msg.name]
      if (x === undefined) return
      const v = asTensor(x)
      out[msg.name] = bijectors.bijectorForSupport(msg.fn.support).inverse(v)
      msg.value = v
    },
  }))
  traceOf(placed, data)
  return out
}

/**
 * How to pick a starting value for a latent site (NumPyro's `init_to_*` strategies): a function of the site (with its
 * distribution and the prototype value) and a key, returning a constrained value, or undefined to fall back to a draw.
 */
export type InitStrategy = (site: SampleSite, key: Key) => Value | undefined

function sampleShapeOf(site: SampleSite): number[] {
  const own = site.fn.batchShape.length + site.fn.eventShape.length
  return site.value.shape.slice(0, site.value.shape.length - own)
}

/**
 * Start at a prior draw (NumPyro's `init_to_sample`).
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToSample } from 'whitebox-engine/probprog'
 * typeof initToSample() // => 'function'
 * ```
 */
export function initToSample(): InitStrategy {
  return (site, key) => site.fn.sample(key, sampleShapeOf(site))
}

/**
 * Start at the median of prior draws, elementwise (NumPyro's `init_to_median`, the default of its samplers).
 *
 * @param options - The number of draws (default 15).
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToMedian } from 'whitebox-engine/probprog'
 * typeof initToMedian({ numSamples: 5 }) // => 'function'
 * ```
 */
export function initToMedian(options: { readonly numSamples?: number } = {}): InitStrategy {
  const n = options.numSamples ?? 15
  return (site, key) => {
    const draws = primalOf(site.fn.sample(key, [n, ...sampleShapeOf(site)]))
    const shape = draws.shape.slice(1)
    const size = product(shape)
    const flat = toFlat(draws)
    const out = new Float64Array(size)
    for (let j = 0; j < size; j++)
      out[j] = quantileOf(
        Array.from({ length: n }, (_, i) => flat[i * size + j] ?? 0),
        0.5,
      )
    return fromStorage(out, shape, 'float64')
  }
}

/**
 * Start at the distribution's mean, falling back to the median of draws where the mean is not finite (NumPyro's
 * `init_to_mean`).
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToMean } from 'whitebox-engine/probprog'
 * typeof initToMean() // => 'function'
 * ```
 */
export function initToMean(): InitStrategy {
  const fallback = initToMedian()
  return (site, key) => {
    try {
      const mean = broadcastTo(site.fn.mean(), site.value.shape)
      if (toFlat(isfinite(mean)).every((v) => v === 1)) return mean
    } catch {
      return fallback(site, key)
    }
    return fallback(site, key)
  }
}

/**
 * Start uniformly in (-radius, radius) on the unconstrained scale (NumPyro's `init_to_uniform`, Stan's default with
 * radius 2).
 *
 * @param options - The radius (default 2).
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToUniform } from 'whitebox-engine/probprog'
 * typeof initToUniform({ radius: 1 }) // => 'function'
 * ```
 */
export function initToUniform(options: { readonly radius?: number } = {}): InitStrategy {
  const r = options.radius ?? 2
  return (site, key) => {
    const b = bijectors.bijectorForSupport(site.fn.support)
    const shape = b.inverseShape(site.value.shape)
    const u = r === 0 ? broadcastTo(tensor(0), shape) : uniform(key, shape, { low: -r, high: r })
    return b.forward(u)
  }
}

/**
 * Start at unconstrained zero: the centre of each support (NumPyro's `init_to_feasible`).
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToFeasible } from 'whitebox-engine/probprog'
 * typeof initToFeasible() // => 'function'
 * ```
 */
export function initToFeasible(): InitStrategy {
  return initToUniform({ radius: 0 })
}

/**
 * Start at given values, with another strategy for the remaining sites (NumPyro's `init_to_value`).
 *
 * @param values - Constrained values by site name.
 * @param fallback - The strategy for other sites (default {@link initToUniform}).
 * @returns The strategy.
 * @example
 * ```ts
 * import { initToValue } from 'whitebox-engine/probprog'
 * typeof initToValue({ mu: 0 }) // => 'function'
 * ```
 */
export function initToValue(
  values: Readonly<Record<string, TensorLike>>,
  fallback: InitStrategy = initToUniform(),
): InitStrategy {
  return (site, key) => {
    const v = values[site.name]
    return v === undefined ? fallback(site, key) : broadcastTo(asTensor(v), site.value.shape)
  }
}

/** The result of {@link findInitialParams}. */
export interface InitialParams {
  /** The unconstrained starting point. */
  readonly unconstrained: Float64Array
  /** Constrained values of the latent sites there. */
  readonly constrained: Readonly<Record<string, Value>>
  /** The potential energy there. */
  readonly potential: number
  /** Whether the potential and its gradient are finite. */
  readonly valid: boolean
  readonly tries: number
}

/**
 * Find a starting point with a finite potential energy and gradient (NumPyro's `find_valid_initial_params`): apply the
 * strategy site by site in a run of the program, and retry with new keys up to `maxTries` times.
 *
 * @param um - The unconstrained model.
 * @param model - The program.
 * @param data - Its data.
 * @param options - Key, strategy (default {@link initToUniform}) and the number of tries (default 100).
 * @returns The point, whether it is valid, and the number of tries.
 * @concept log-density
 * @example
 * ```ts
 * import { findInitialParams, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * import { key } from 'whitebox-engine/random'
 * const model = () => { sample('x', normal(0, 1)) }
 * findInitialParams(potentialEnergy(model, undefined), model, undefined, { key: key(0) }).valid // => true
 * ```
 */
export function findInitialParams<D = void>(
  um: UnconstrainedModel,
  model: Model<D>,
  data: D,
  options: { readonly key: Key; readonly strategy?: InitStrategy; readonly maxTries?: number },
): InitialParams {
  const strategy = options.strategy ?? initToUniform()
  const maxTries = options.maxTries ?? 100
  const names = new Set(um.layout.map((l) => l.name))
  let last: InitialParams | undefined
  for (let i = 0; i < maxTries; i++) {
    const k = foldIn(options.key, i)
    let count = 0
    const values: Record<string, Value> = {}
    const placed = handle(seed(enumerate(model), foldIn(k, 1 << 20)), () => ({
      process: (msg: Message) => {
        if (msg.type !== 'sample' || msg.isObserved || msg.value !== undefined || msg.fn === undefined) return
        if (!names.has(msg.name)) return
        const proto: SampleSite = {
          type: 'sample',
          name: msg.name,
          fn: msg.fn,
          value: msg.fn.sample(foldIn(k, 1 << 21), msg.sampleShape),
          isObserved: false,
          scale: msg.scale,
          mask: msg.mask,
          plates: [...msg.plates],
          markov: [...msg.markov],
          infer: msg.infer,
          isFactor: false,
          enumDim: undefined,
        }
        const v = strategy(proto, foldIn(k, count))
        count += 1
        if (v !== undefined) {
          msg.value = v
          values[msg.name] = v
        }
      },
    }))
    let u: Float64Array
    try {
      const t = traceOf(placed, data)
      for (const site of t.sites.values())
        if (site.type === 'sample' && names.has(site.name) && values[site.name] === undefined)
          values[site.name] = site.value
      u = um.unconstrain(values)
    } catch (e) {
      if (i === maxTries - 1) throw e
      continue
    }
    let ok = toFlat(fromStorage(u, [u.length], 'float64')).every(Number.isFinite)
    let potential = Number.POSITIVE_INFINITY
    if (ok) {
      const r = um.logDensity.valueAndGrad(u)
      potential = -r.value
      ok = Number.isFinite(r.value) && r.grad.every(Number.isFinite)
    }
    last = { unconstrained: u, constrained: values, potential, valid: ok, tries: i + 1 }
    if (ok) return last
  }
  return (
    last ?? {
      unconstrained: new Float64Array(um.dim),
      constrained: {},
      potential: Number.POSITIVE_INFINITY,
      valid: false,
      tries: maxTries,
    }
  )
}

/**
 * The constrained values of an unconstrained draw as tensors, for many draws at once: `[n, dim]` to site values
 * `[n, ...shape]`.
 *
 * @param um - The unconstrained model.
 * @param draws - Unconstrained draws, row-major `[n, dim]`.
 * @returns Stacked constrained values by site (latent and deterministic).
 * @example
 * ```ts
 * import { constrainDraws, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { exponential } from 'whitebox-engine/dists'
 * const um = potentialEnergy(() => { sample('s', exponential(1)) }, undefined)
 * constrainDraws(um, new Float64Array([0, 0]))['s']?.shape // => [2]
 * ```
 */
export function constrainDraws(um: UnconstrainedModel, draws: Float64Array): Record<string, Tensor> {
  const n = um.dim === 0 ? 0 : draws.length / um.dim
  const columns: Record<string, number[][]> = {}
  const shapes: Record<string, readonly number[]> = {}
  for (let i = 0; i < n; i++) {
    const values = um.constrain(draws.subarray(i * um.dim, (i + 1) * um.dim))
    for (const [name, v] of Object.entries(values)) {
      const t = primalOf(v)
      shapes[name] = t.shape
      const col = columns[name] ?? []
      col.push(toFlat(t))
      columns[name] = col
    }
  }
  const out: Record<string, Tensor> = {}
  for (const [name, rows] of Object.entries(columns)) {
    const shape = shapes[name] ?? []
    out[name] = fromStorage(Float64Array.from(rows.flat()), [rows.length, ...shape], 'float64')
  }
  return out
}

/**
 * The value of a scalar log density at a point, as a number (NaN reads as -Infinity).
 *
 * @param um - The unconstrained model.
 * @param u - The unconstrained point.
 * @returns log density at u.
 * @example
 * ```ts
 * import { logDensityAt, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * const um = potentialEnergy(() => { sample('x', normal(0, 1)) }, undefined)
 * Math.abs(logDensityAt(um, [0]) + 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function logDensityAt(um: UnconstrainedModel, u: ArrayLike<number>): number {
  const v = item(um.fn(fromStorage(Float64Array.from(u), [um.dim], 'float64')))
  return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
}

/**
 * Concatenate unconstrained site values into one flat vector, in layout order (a differentiable helper for guides).
 *
 * @param layout - The layout.
 * @param values - Unconstrained values by site.
 * @returns The `[dim]` vector.
 * @example
 * ```ts
 * import { packLatents, potentialEnergy, sample } from 'whitebox-engine/probprog'
 * import { normal } from 'whitebox-engine/dists'
 * const um = potentialEnergy(() => { sample('x', normal(0, 1), { sampleShape: [2] }) }, undefined)
 * packLatents(um.layout, { x: [1, 2] }).shape // => [2]
 * ```
 */
export function packLatents(layout: readonly LatentLayout[], values: Readonly<Record<string, TensorLike>>): Value {
  return concat(
    layout.map((l) => {
      const v = values[l.name]
      if (v === undefined) throw new WhiteboxError('value', `packLatents: no value for ${l.name}`)
      return reshape(asTensor(v), [l.size])
    }),
  )
}
