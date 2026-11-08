/**
 * Associative memories: the classical Hopfield network (patterns stored by the Hebb or pseudo-inverse rule, recalled by
 * asynchronous updates that never raise the energy, as a stepped `Algorithm`), the dense associative memory of Krotov
 * and Hopfield (polynomial or exponential interaction, which stores far more patterns), and the modern continuous
 * Hopfield network of Ramsauer et al., whose one-step update is attention over the stored patterns. The restricted
 * Boltzmann machine, the other energy-based block, is in `rbm.ts`.
 */

import { maxOf, OmaiError } from '../../base/index.ts'
import { pinv } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, randint } from '../../random/index.ts'
import {
  add,
  asTensor,
  eye,
  logsumexp,
  matmul,
  mul,
  neg,
  softmax,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels } from '../../trace/index.ts'
import { normalInit } from '../init.ts'
import { initKey, Module, type Params } from '../module/module.ts'

/** How the weights of a classical Hopfield network are set from the patterns. */
export type HopfieldRule = 'hebb' | 'pseudo-inverse'

/**
 * The weights of a classical Hopfield network storing `+-1` patterns: the Hebb rule `W = X^T X / n` or the
 * pseudo-inverse (projection) rule `W = X^+ X`, which stores correlated patterns exactly. The diagonal is zeroed, so
 * asynchronous updates never raise the energy.
 *
 * @param patterns - The patterns `[p, n]`, entries `+-1`.
 * @param rule - The learning rule (default Hebb).
 * @returns The symmetric weights `[n, n]`.
 * @example
 * ```ts
 * import { hopfieldWeights } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hopfieldWeights([[1, -1]])) // => [[0, -0.5], [-0.5, 0]]
 * ```
 */
export function hopfieldWeights(patterns: TensorLike, rule: HopfieldRule = 'hebb'): Value {
  const x = asTensor(patterns)
  if (x.shape.length !== 2) throw new OmaiError('shape', 'hopfieldWeights: patterns must be [p, n]')
  const n = x.shape[1] ?? 0
  const w = rule === 'hebb' ? matmul(transpose(x), x) : matmul(pinv(x), x)
  const scaled = rule === 'hebb' ? mul(w, 1 / n) : w
  return mul(scaled, sub(1, eye(n)))
}

/**
 * The energy of a classical Hopfield network, `E(s) = -1/2 s^T W s - b^T s`; a state is a vector or a batch of rows.
 *
 * @param weights - The symmetric weights `[n, n]`.
 * @param state - The state `[n]` or `[B, n]`.
 * @param bias - Optional thresholds `[n]`.
 * @returns The energy (a scalar, or `[B]`).
 * @example
 * ```ts
 * import { hopfieldEnergy } from 'omai-engine/nn/blocks'
 * import { item } from 'omai-engine/tensor'
 * item(hopfieldEnergy([[0, 1], [1, 0]], [1, 1])) // => -1
 * ```
 */
export function hopfieldEnergy(weights: TensorLike, state: TensorLike, bias?: TensorLike): Value {
  const s = asTensor(state)
  let e = mul(sum(mul(matmul(s, weights), s), -1), -0.5)
  if (bias !== undefined) e = sub(e, sum(mul(s, bias), -1))
  return e
}

/**
 * An associative memory over `n` units with a discrete state: an energy and the update of one unit. The classical
 * network and the dense memory both are one, so one recall `Algorithm` runs either.
 */
export interface AssociativeMemory {
  /** The number of units. */
  readonly size: number
  /** The energy of a state. */
  energy(state: ArrayLike<number>): number
  /** The value of unit `i` that minimises the energy with the others fixed (its current value on a tie). */
  update(state: ArrayLike<number>, i: number): number
}

/**
 * The memory of a classical Hopfield network: the unit takes the sign of its local field (the other units' weighted
 * sum plus the bias) and keeps its value when the field is zero.
 *
 * @param weights - The symmetric weights `[n, n]`.
 * @param bias - Optional thresholds `[n]`.
 * @returns The memory.
 * @example
 * ```ts
 * import { classicalMemory, hopfieldWeights } from 'omai-engine/nn/blocks'
 * const m = classicalMemory(hopfieldWeights([[1, -1, 1, -1]]))
 * m.update([1, 1, 1, -1], 1) // => -1
 * ```
 */
export function classicalMemory(weights: TensorLike, bias?: TensorLike): AssociativeMemory {
  const w = asTensor(weights)
  const n = w.shape[0] ?? 0
  const wd = Float64Array.from(toFlat(w))
  const b = bias === undefined ? new Float64Array(n) : Float64Array.from(toFlat(asTensor(bias)))
  return {
    size: n,
    energy: (s) => {
      let e = 0
      for (let i = 0; i < n; i++) {
        let f = 0
        for (let j = 0; j < n; j++) f += (wd[i * n + j] as number) * (s[j] as number)
        e -= 0.5 * f * (s[i] as number) + (b[i] as number) * (s[i] as number)
      }
      return e
    },
    update: (s, i) => {
      let f = b[i] as number
      for (let j = 0; j < n; j++) if (j !== i) f += (wd[i * n + j] as number) * (s[j] as number)
      return f > 0 ? 1 : f < 0 ? -1 : (s[i] as number)
    },
  }
}

/** The interaction function of a dense associative memory. */
export type DenseInteraction = 'polynomial' | 'exponential'

/** Options of {@link denseAssociativeMemory}. */
export interface DenseMemoryOptions {
  /** The interaction (default polynomial). */
  readonly interaction?: DenseInteraction
  /** The degree of the polynomial `n (m_+)^a` in the overlap `m = xi . s / n` (default 3). */
  readonly degree?: number
  /** The inverse temperature of the exponential `exp(beta m)` (default 8). */
  readonly beta?: number
}

/**
 * The dense associative memory of Krotov and Hopfield (2016): the energy `E(s) = -sum_mu F(xi_mu . s)` with a
 * rectified polynomial (`F = n (m_+)^a`, `m = x / n`) or an exponential (`F = exp(beta m)`) interaction instead of the
 * quadratic one, so the capacity grows like `n^(a - 1)` or exponentially. A unit takes the sign that gives the lower
 * energy with the others fixed, so asynchronous updates never raise the energy.
 *
 * @param patterns - The stored patterns `[p, n]`, entries `+-1`.
 * @param options - The interaction and its degree or inverse temperature.
 * @returns The memory.
 * @reference Krotov and Hopfield (2016), Dense associative memory for pattern recognition.
 * @example
 * ```ts
 * import { denseAssociativeMemory } from 'omai-engine/nn/blocks'
 * const m = denseAssociativeMemory([[1, -1, 1, -1], [1, 1, -1, -1]])
 * m.update([1, -1, 1, 1], 3) // => -1
 * ```
 */
export function denseAssociativeMemory(patterns: TensorLike, options: DenseMemoryOptions = {}): AssociativeMemory {
  const x = asTensor(patterns)
  if (x.shape.length !== 2) throw new OmaiError('shape', 'denseAssociativeMemory: patterns must be [p, n]')
  const [p, n] = x.shape as [number, number]
  const xi = Float64Array.from(toFlat(x))
  const a = options.degree ?? 3
  const beta = options.beta ?? 8
  const f =
    (options.interaction ?? 'polynomial') === 'polynomial'
      ? (v: number): number => (v > 0 ? n * (v / n) ** a : 0)
      : (v: number): number => Math.exp((beta * v) / n)
  const overlaps = (s: ArrayLike<number>): Float64Array => {
    const o = new Float64Array(p)
    for (let m = 0; m < p; m++) {
      let d = 0
      for (let j = 0; j < n; j++) d += (xi[m * n + j] as number) * (s[j] as number)
      o[m] = d
    }
    return o
  }
  return {
    size: n,
    energy: (s) => {
      let e = 0
      for (const o of overlaps(s)) e -= f(o)
      return e
    },
    update: (s, i) => {
      const o = overlaps(s)
      let diff = 0
      for (let m = 0; m < p; m++) {
        const c = xi[m * n + i] as number
        const rest = (o[m] as number) - c * (s[i] as number)
        diff += f(rest + c) - f(rest - c)
      }
      return diff > 0 ? 1 : diff < 0 ? -1 : (s[i] as number)
    },
  }
}

/** The order in which a recall visits the units. */
export type RecallOrder = 'sequential' | 'random'

/** Options of {@link hopfieldRecallAlgorithm}. */
export interface HopfieldRecallOptions {
  /** The order of the asynchronous updates (default sequential sweeps). */
  readonly order?: RecallOrder
  /** The stored patterns `[p, n]`, for the overlap with the nearest one on the tape. */
  readonly patterns?: TensorLike
}

/** The state of {@link hopfieldRecallAlgorithm}: the state of the network and how long it has been stable. */
export interface HopfieldRecallState {
  readonly state: readonly number[]
  readonly energy: number
  /** Updates since the last unit changed. */
  readonly quiet: number
  /** The number of updates that changed a unit. */
  readonly flips: number
}

/** The paths of {@link hopfieldRecallAlgorithm}. */
export type HopfieldRecallPath = 'recall/state' | 'recall/energy' | 'recall/overlap' | 'recall/flips'

const isStable = (memory: AssociativeMemory, s: ArrayLike<number>): boolean => {
  for (let i = 0; i < memory.size; i++) if (memory.update(s, i) !== s[i]) return false
  return true
}

/**
 * Recall from a corrupted pattern, one asynchronous update per step: a unit (in sweeps or at random) takes the value
 * that lowers the energy, so the energy never rises and the run ends at a fixed point (a stored pattern, its mirror or
 * a spurious state). The energy, the overlap with the nearest stored pattern and the state are on the tape.
 *
 * @param memory - A {@link classicalMemory} or {@link denseAssociativeMemory}.
 * @param start - The initial state `[n]`, entries `+-1`.
 * @param options - The update order and the patterns to measure overlap against.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { classicalMemory, hopfieldRecallAlgorithm, hopfieldWeights } from 'omai-engine/nn/blocks'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const stored = [[1, -1, 1, -1, 1, -1]]
 * const algo = hopfieldRecallAlgorithm(classicalMemory(hopfieldWeights(stored)), [1, 1, 1, -1, 1, -1])
 * runAlgorithm(algo, { maxSteps: 50 }).state.state // => [1, -1, 1, -1, 1, -1]
 * ```
 */
export function hopfieldRecallAlgorithm(
  memory: AssociativeMemory,
  start: TensorLike,
  options: HopfieldRecallOptions = {},
): Algorithm<HopfieldRecallState> {
  const initial = toFlat(asTensor(start))
  if (initial.length !== memory.size) throw new OmaiError('shape', 'hopfieldRecallAlgorithm: start has the wrong size')
  const stored = options.patterns === undefined ? undefined : asTensor(options.patterns)
  const order = options.order ?? 'sequential'
  const n = memory.size
  const overlap = (s: readonly number[]): number => {
    if (stored === undefined) return Number.NaN
    const [p] = stored.shape as [number, number]
    const d = toFlat(stored)
    let best = -Infinity
    for (let m = 0; m < p; m++) {
      let o = 0
      for (let j = 0; j < n; j++) o += (d[m * n + j] as number) * (s[j] as number)
      best = Math.max(best, o / n)
    }
    return best
  }
  return {
    name: 'hopfield-recall',
    config: { order, size: n },
    schema: channels({
      metrics: {
        'recall/energy': { description: 'the energy of the network', direction: 'lower' },
        'recall/overlap': { description: 'the largest overlap with a stored pattern', direction: 'higher' },
        'recall/flips': { description: 'updates that changed a unit so far' },
      },
      tensors: { 'recall/state': { role: 'state', description: 'the state of the units', shape: [n] } },
    }),
    init: () => ({ state: initial, energy: memory.energy(initial), quiet: 0, flips: 0 }),
    step: (s, ctx) => {
      const i = order === 'sequential' ? ctx.t % n : (toFlat(randint(foldIn(ctx.key, 0), [1], 0, n))[0] as number)
      const next = memory.update(s.state, i)
      if (next === s.state[i]) return { ...s, quiet: s.quiet + 1 }
      const state = s.state.map((v, j) => (j === i ? next : v))
      return { state, energy: memory.energy(state), quiet: 0, flips: s.flips + 1 }
    },
    done: (s) => (s.quiet >= n && isStable(memory, s.state) ? 'converged' : undefined),
    observe: (s) => {
      const metrics: Record<string, number> = { 'recall/energy': s.energy, 'recall/flips': s.flips }
      if (stored !== undefined) metrics['recall/overlap'] = overlap(s.state)
      return { metrics, tensors: { 'recall/state': { role: 'state', value: s.state } } }
    },
    serialise: (s) => ({ state: [...s.state], energy: s.energy, quiet: s.quiet, flips: s.flips }),
    deserialise: (j) => {
      const o = j as { state: number[]; energy: number; quiet: number; flips: number }
      return { state: o.state, energy: o.energy, quiet: o.quiet, flips: o.flips }
    },
  }
}

/**
 * One update of the modern (continuous) Hopfield network: `x <- X^T softmax(beta X x)` over the stored patterns, which
 * is attention with the patterns as keys and values.
 *
 * @param patterns - The stored patterns `[p, n]`.
 * @param state - The state `[n]` or `[B, n]`.
 * @param beta - The inverse temperature (default 1).
 * @returns The updated state, the shape of `state`.
 * @reference Ramsauer et al. (2021), Hopfield networks is all you need.
 * @example
 * ```ts
 * import { modernHopfieldUpdate } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(modernHopfieldUpdate([[1, 0], [0, 1]], [0.9, 0.1], 50)).map((v) => Math.round(v)) // => [1, 0]
 * ```
 */
export function modernHopfieldUpdate(patterns: TensorLike, state: TensorLike, beta = 1): Value {
  const x = asTensor(patterns)
  return matmul(softmax(mul(matmul(state, transpose(x)), beta), -1), x)
}

/**
 * The energy of the modern Hopfield network, `E(x) = -lse(beta, X x) + 1/2 x.x + log(p) / beta + 1/2 M^2`, with `M`
 * the largest pattern norm; the update of {@link modernHopfieldUpdate} is the concave-convex step on it, so it never
 * raises it.
 *
 * @param patterns - The stored patterns `[p, n]`.
 * @param state - The state `[n]` or `[B, n]`.
 * @param beta - The inverse temperature (default 1).
 * @returns The energy (a scalar, or `[B]`).
 * @example
 * ```ts
 * import { modernHopfieldEnergy } from 'omai-engine/nn/blocks'
 * import { item } from 'omai-engine/tensor'
 * item(modernHopfieldEnergy([[1, 0], [0, 1]], [1, 0], 100)) > item(modernHopfieldEnergy([[1, 0], [0, 1]], [1, 0], 1)) // => false
 * ```
 */
export function modernHopfieldEnergy(patterns: TensorLike, state: TensorLike, beta = 1): Value {
  const x = asTensor(patterns)
  const p = x.shape[0] ?? 1
  const m2 = maxOf(toFlat(sum(mul(x, x), -1)))
  const s = asTensor(state)
  const scores = mul(matmul(s, transpose(x)), beta)
  return add(add(neg(div1(logsumexp(scores, -1), beta)), mul(sum(mul(s, s), -1), 0.5)), Math.log(p) / beta + 0.5 * m2)
}

const div1 = (x: Value, b: number): Value => mul(x, 1 / b)

/** The state of {@link modernHopfieldAlgorithm}. */
export interface ModernHopfieldState {
  readonly state: readonly number[]
  readonly energy: number
  /** The largest change of a coordinate in the last step. */
  readonly change: number
}

/**
 * Retrieval in the modern Hopfield network as a stepped algorithm: each step applies {@link modernHopfieldUpdate} to
 * the state, until it stops moving. The energy and the overlap (cosine) with the nearest stored pattern are on the
 * tape.
 *
 * @param patterns - The stored patterns `[p, n]`.
 * @param start - The query `[n]`.
 * @param beta - The inverse temperature (default 1).
 * @param tolerance - Stop when no coordinate moves more than this (default 1e-9).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { modernHopfieldAlgorithm } from 'omai-engine/nn/blocks'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const a = modernHopfieldAlgorithm([[1, 0, 0], [0, 1, 0]], [0.8, 0.3, 0.1], 20)
 * runAlgorithm(a, { maxSteps: 20 }).state.state.map((v) => Math.round(v)) // => [1, 0, 0]
 * ```
 */
export function modernHopfieldAlgorithm(
  patterns: TensorLike,
  start: TensorLike,
  beta = 1,
  tolerance = 1e-9,
): Algorithm<ModernHopfieldState> {
  const x = asTensor(patterns)
  const initial = toFlat(asTensor(start))
  const energyOf = (s: readonly number[]): number => toFlat(modernHopfieldEnergy(x, s, beta))[0] as number
  const cosineBest = (s: readonly number[]): number => {
    const [p, n] = x.shape as [number, number]
    const d = toFlat(x)
    const ns = Math.hypot(...s)
    let best = -Infinity
    for (let m = 0; m < p; m++) {
      let dot = 0
      let nm = 0
      for (let j = 0; j < n; j++) {
        dot += (d[m * n + j] as number) * (s[j] as number)
        nm += (d[m * n + j] as number) ** 2
      }
      best = Math.max(best, dot / (Math.sqrt(nm) * ns || 1))
    }
    return best
  }
  return {
    name: 'modern-hopfield',
    config: { beta, tolerance },
    init: () => ({ state: initial, energy: energyOf(initial), change: Infinity }),
    step: (s) => {
      const next = toFlat(modernHopfieldUpdate(x, s.state, beta))
      const change = maxOf(next.map((v, j) => Math.abs(v - (s.state[j] as number))))
      return { state: next, energy: energyOf(next), change }
    },
    done: (s) => (s.change <= tolerance ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { 'recall/energy': s.energy, 'recall/overlap': cosineBest(s.state) },
      tensors: { 'recall/state': { role: 'state', value: s.state } },
    }),
    serialise: (s) => ({ state: [...s.state], energy: s.energy, change: Number.isFinite(s.change) ? s.change : null }),
    deserialise: (j) => {
      const o = j as { state: number[]; energy: number; change: number | null }
      return { state: o.state, energy: o.energy, change: o.change ?? Infinity }
    },
  }
}

/** Options of {@link ModernHopfield}. */
export interface ModernHopfieldOptions {
  /** The inverse temperature (default 1). */
  readonly beta?: number
  /** The initial patterns `[p, n]` (default Gaussian). */
  readonly patterns?: TensorLike
  readonly key?: Key
}

/**
 * A Hopfield layer (Ramsauer et al. 2021) with learnable stored patterns: `x -> softmax(beta x X^T) X`. Its forward
 * pass is attention of the input over the patterns, differentiable in the input and in the patterns.
 *
 * @example
 * ```ts
 * import { ModernHopfield } from 'omai-engine/nn/blocks'
 * import { key } from 'omai-engine/random'
 * new ModernHopfield(4, 3, { key: key(0) }).forward([[1, 0, 0, 0]]).shape // => [1, 4]
 * ```
 */
export class ModernHopfield extends Module {
  readonly kind = 'ModernHopfield'
  readonly features: number
  readonly stored: number
  readonly beta: number
  private readonly initial: Tensor | undefined

  constructor(features: number, stored: number, options: ModernHopfieldOptions = {}) {
    super()
    this.features = features
    this.stored = stored
    this.beta = options.beta ?? 1
    this.initial = options.patterns === undefined ? undefined : (asTensor(options.patterns) as Tensor)
    this.params = this.initParams(initKey('ModernHopfield', options.key))
  }

  override initParams(key: Key): Params {
    return { patterns: this.initial ?? normalInit(foldIn(key, 0), [this.stored, this.features]) }
  }

  apply(params: Params, x: Value): Value {
    return modernHopfieldUpdate(params['patterns'] as Value, x, this.beta)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `features=${this.features}, stored=${this.stored}, beta=${this.beta}`
  }
}
