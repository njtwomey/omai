/**
 * Reservoir sampling: a uniform (or weighted) sample of fixed size from a stream of unknown length in one pass. Vitter's
 * Algorithm R and Li's skip-based Algorithm L for the unweighted case, Efraimidis and Spirakis's A-Res and its jump
 * variant A-ExpJ for weights. Every draw comes from a `Key`, so a sample is reproducible; the sampler object takes items
 * one at a time and the algorithm runs the same rules one item per step with a state that serialises.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Key, uniform, uniformStream } from '../random/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** The rule a {@link ReservoirSampler} follows: unweighted `'R'` and `'L'`, weighted `'A-Res'` and `'A-ExpJ'`. */
export type ReservoirMethod = 'R' | 'L' | 'A-Res' | 'A-ExpJ'

/** Configuration of {@link reservoirSampler}. */
export interface ReservoirOptions {
  /** The reservoir size `k >= 1`. */
  readonly k: number
  /** The key all draws come from. */
  readonly key: Key
  /** The rule (default `'R'`; `'A-Res'` and `'A-ExpJ'` take weights). */
  readonly method?: ReservoirMethod
}

/** A reservoir sampler over items of type `T`. */
export interface ReservoirSampler<T> {
  /** The reservoir size. */
  readonly k: number
  /** The rule. */
  readonly method: ReservoirMethod
  /** Items offered so far. */
  readonly seen: number
  /** Offer an item (with its weight, which the weighted rules require and the unweighted ones ignore). */
  add(item: T, weight?: number): void
  /** The current sample (at most `k` items, in no particular order). */
  sample(): readonly T[]
}

// u in (0, 1]: safe under log.
const open01 = (u: number): number => (u === 0 ? 2 ** -53 : u)

/**
 * A reservoir sampler. After `n` items, every size-`k` subset is equally likely to be the sample (`'R'`, `'L'`), or
 * the sample is the weighted sampling without replacement of Efraimidis and Spirakis, where each item gets key
 * `u^(1/w)` and the `k` largest keys are kept (`'A-Res'`, `'A-ExpJ'`; for `k = 1` item `i` is chosen with
 * probability `w_i / sum w`). `'L'` and `'A-ExpJ'` skip over the items that cannot enter, drawing O(k log(n/k))
 * random numbers instead of O(n).
 *
 * @param options - `k`, the `key` and the `method`.
 * @returns The sampler.
 * @throws MdmlError with code `value` for `k < 1`.
 * @concept reservoir-sampling
 * @reference J. S. Vitter, Random sampling with a reservoir (ACM Transactions on Mathematical Software, 1985); K.-H. Li,
 *   Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))) (1994); P. S. Efraimidis and P. G. Spirakis,
 *   Weighted random sampling with a reservoir (Information Processing Letters, 2006).
 * @example
 * ```ts
 * import { reservoirSampler } from 'mdml-engine/sketches'
 * import { key } from 'mdml-engine/random'
 * const r = reservoirSampler<number>({ k: 5, key: key(1) })
 * for (let i = 0; i < 1000; i++) r.add(i)
 * [r.sample().length, r.seen] // => [5, 1000]
 * ```
 */
export function reservoirSampler<T>(options: ReservoirOptions): ReservoirSampler<T> {
  const { k } = options
  const method = options.method ?? 'R'
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `reservoirSampler: k must be a positive integer, got ${k}`)
  const next = uniformStream(options.key)
  const items: T[] = []
  const keys: number[] = [] // the weighted rules' priorities
  let seen = 0
  // Algorithm L: skip is the number of items to pass over before the next replacement; w is its running scale.
  let skip = 0
  let w = 0
  // A-ExpJ: the weight still to pass before the next replacement.
  let jump = 0
  let jumpFresh = true
  const minAt = (): number => {
    let m = 0
    for (let i = 1; i < keys.length; i++) if ((keys[i] ?? 0) < (keys[m] ?? 0)) m = i
    return m
  }
  const scheduleL = (): void => {
    w *= Math.exp(Math.log(open01(next())) / k)
    skip = Math.floor(Math.log(open01(next())) / Math.log1p(-w))
  }
  return {
    k,
    method,
    get seen() {
      return seen
    },
    add: (item, weight = 1) => {
      seen += 1
      if (method === 'R') {
        if (items.length < k) items.push(item)
        else {
          const j = Math.floor(next() * seen)
          if (j < k) items[j] = item
        }
      } else if (method === 'L') {
        if (items.length < k) {
          items.push(item)
          if (items.length === k) {
            w = Math.exp(Math.log(open01(next())) / k)
            skip = Math.floor(Math.log(open01(next())) / Math.log1p(-w))
          }
        } else if (skip > 0) skip -= 1
        else {
          items[Math.floor(next() * k)] = item
          scheduleL()
        }
      } else {
        if (!(weight > 0 && Number.isFinite(weight))) {
          throw new MdmlError('value', `reservoirSampler.add: weights must be positive and finite, got ${weight}`)
        }
        if (items.length < k) {
          items.push(item)
          keys.push(open01(next()) ** (1 / weight))
        } else if (method === 'A-Res') {
          const key = open01(next()) ** (1 / weight)
          const m = minAt()
          if (key > (keys[m] ?? 0)) {
            items[m] = item
            keys[m] = key
          }
        } else {
          // A-ExpJ: pass items until the accumulated weight exceeds X = log(r) / log(T), T the smallest key.
          const m = minAt()
          const threshold = keys[m] ?? 0
          if (jumpFresh) {
            jump = Math.log(open01(next())) / Math.log(threshold)
            jumpFresh = false
          }
          jump -= weight
          if (jump <= 0) {
            const t = threshold ** weight
            const r2 = t + (1 - t) * next()
            items[m] = item
            keys[m] = open01(r2) ** (1 / weight)
            jumpFresh = true
          }
        }
      }
    },
    sample: () => items.slice(),
  }
}

/** Options for {@link reservoirSample}. */
export interface ReservoirSampleOptions extends ReservoirOptions {
  /** Item weights (positive), for the weighted rules; the default method then becomes `'A-Res'`. */
  readonly weights?: readonly number[]
}

/**
 * Sample `k` items from an array in one pass with a reservoir rule.
 *
 * @param items - The stream.
 * @param options - `k`, `key`, `method` and optional `weights`.
 * @returns The sample and the number of items seen.
 * @throws MdmlError with code `shape` when `weights` and `items` differ in length.
 * @concept reservoir-sampling
 * @example
 * ```ts
 * import { reservoirSample } from 'mdml-engine/sketches'
 * import { key } from 'mdml-engine/random'
 * const out = reservoirSample([10, 20, 30, 40, 50], { k: 3, key: key(42) })
 * [out.sample.length, out.seen] // => [3, 5]
 * ```
 */
export function reservoirSample<T>(
  items: readonly T[],
  options: ReservoirSampleOptions,
): { readonly sample: readonly T[]; readonly seen: number } {
  if (options.weights !== undefined && options.weights.length !== items.length) {
    throw new MdmlError('shape', `reservoirSample: ${options.weights.length} weights for ${items.length} items`)
  }
  const sampler = reservoirSampler<T>({
    k: options.k,
    key: options.key,
    method: options.method ?? (options.weights === undefined ? 'R' : 'A-Res'),
  })
  items.forEach((item, i) => {
    sampler.add(item, options.weights?.[i])
  })
  return { sample: sampler.sample(), seen: sampler.seen }
}

/** The state of {@link reservoirAlgorithm}. */
export interface ReservoirState {
  /** Indices (into the stream) of the sampled items. */
  readonly indices: readonly number[]
  /** The weighted rule's priorities `u^(1/w)` (empty for the unweighted rule). */
  readonly priorities: readonly number[]
  /** Items seen. */
  readonly seen: number
  /** Replacements made after the reservoir filled. */
  readonly replacements: number
  /** Whether the last item entered the reservoir. */
  readonly entered: boolean
}

/** Input of {@link reservoirAlgorithm}. */
export interface ReservoirAlgorithmOptions {
  /** The stream's length: items are identified by their index, one per step. */
  readonly length: number
  /** The reservoir size. */
  readonly k: number
  /** Item weights; when given the rule is A-Res, otherwise Algorithm R. */
  readonly weights?: readonly number[]
}

/**
 * Reservoir sampling as an algorithm, one stream position per step (the sample holds indices, so it applies to any
 * stream). Step `t` draws from the step's own key, so a paused or replayed run takes the same decisions. It logs the
 * replacement count and whether the item just seen entered, and the entry probability `k / t` is the thing to watch
 * fall.
 *
 * @param options - The stream length, `k` and optional weights.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for `k < 1`; `shape` for weights of the wrong length.
 * @concept reservoir-sampling
 * @example
 * ```ts
 * import { reservoirAlgorithm } from 'mdml-engine/sketches'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const res = runAlgorithm(reservoirAlgorithm({ length: 100, k: 5 }), { seed: 7, maxSteps: 200 })
 * [res.state.indices.length, res.state.seen] // => [5, 100]
 * ```
 */
export function reservoirAlgorithm(options: ReservoirAlgorithmOptions): Algorithm<ReservoirState> {
  const { length, k, weights } = options
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `reservoirAlgorithm: k must be a positive integer, got ${k}`)
  if (weights !== undefined && weights.length !== length) {
    throw new MdmlError('shape', `reservoirAlgorithm: ${weights.length} weights for ${length} items`)
  }
  return {
    name: 'reservoir-sample',
    config: { length, k, weighted: weights !== undefined },
    schema: channels({
      metrics: {
        replacements: { description: 'replacements made after the reservoir filled' },
        entered: { description: '1 when the item just seen entered the reservoir' },
      },
    }),
    init: () => ({ indices: [], priorities: [], seen: 0, replacements: 0, entered: false }),
    step: (s, ctx) => {
      const i = s.seen
      const u = open01(uniform(ctx.key, [1]).data[0] ?? 0.5)
      const indices = [...s.indices]
      const priorities = [...s.priorities]
      let entered = false
      let replacements = s.replacements
      if (weights === undefined) {
        if (indices.length < k) {
          indices.push(i)
          entered = true
        } else {
          const j = Math.floor(u * (i + 1))
          if (j < k) {
            indices[j] = i
            entered = true
            replacements += 1
          }
        }
      } else {
        const weight = weights[i] ?? 1
        if (!(weight > 0)) throw new MdmlError('value', `reservoirAlgorithm: weights must be positive, got ${weight}`)
        const priority = u ** (1 / weight)
        if (indices.length < k) {
          indices.push(i)
          priorities.push(priority)
          entered = true
        } else {
          let m = 0
          for (let j = 1; j < priorities.length; j++) if ((priorities[j] ?? 0) < (priorities[m] ?? 0)) m = j
          if (priority > (priorities[m] ?? 0)) {
            indices[m] = i
            priorities[m] = priority
            entered = true
            replacements += 1
          }
        }
      }
      return { indices, priorities, seen: i + 1, replacements, entered }
    },
    done: (s) => (s.seen >= length ? 'completed' : undefined),
    observe: (s) => ({ metrics: { replacements: s.replacements, entered: s.entered ? 1 : 0 } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as ReservoirState,
  }
}
