/**
 * Experience storage for reinforcement learning, as persistent (immutable) values: a replay buffer as in
 * stable-baselines3, a sum tree and prioritised replay with importance weights (Schaul et al.), n-step transition
 * accumulation, an on-policy rollout buffer with GAE advantages and shuffled minibatches, and an episode store that
 * samples subsequences for sequence models.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, permutation, randint, uniform } from '../random/index.ts'
import type { Transition } from './agent.ts'
import { gae } from './returns.ts'

// ------------------------------------------------------------------------------------------------- replay buffer

/**
 * A persistent ring buffer of experience (stable-baselines3's ReplayBuffer, as an immutable value): items live in
 * fixed-size chunks, so adding one copies a single chunk and the chunk index rather than the whole buffer, and every
 * earlier version stays valid (agents keep their buffers in their state and can be checkpointed and replayed).
 *
 * @concept experience-replay
 */
export interface ReplayBuffer<T> {
  readonly capacity: number
  readonly chunkSize: number
  readonly chunks: readonly (readonly T[])[]
  /** The number of stored items (at most the capacity). */
  readonly size: number
  /** The slot the next item is written to. */
  readonly next: number
  /** The number of items ever added. */
  readonly added: number
}

/**
 * An empty replay buffer.
 *
 * @param capacity - The maximum number of items; older items are overwritten first.
 * @param options - The chunk size (default the square root of the capacity, at least 16).
 * @returns The buffer.
 * @throws MdmlError with code `value` for a capacity below 1.
 * @concept experience-replay
 * @index structure Replay buffer for off-policy learners.
 * @example
 * ```ts
 * import { replayBuffer } from 'mdml-engine/rl'
 * replayBuffer<number>(100).size // => 0
 * ```
 */
export function replayBuffer<T>(capacity: number, options: { readonly chunkSize?: number } = {}): ReplayBuffer<T> {
  if (!(Number.isInteger(capacity) && capacity >= 1)) {
    throw new MdmlError('value', 'replayBuffer: capacity must be a positive integer')
  }
  const chunkSize = options.chunkSize ?? Math.max(16, Math.ceil(Math.sqrt(capacity)))
  return { capacity, chunkSize, chunks: [], size: 0, next: 0, added: 0 }
}

/**
 * Add an item, overwriting the oldest when the buffer is full.
 *
 * @param buffer - The buffer.
 * @param item - The item.
 * @returns The new buffer (the old one is unchanged).
 * @concept experience-replay
 * @example
 * ```ts
 * import { bufferAdd, bufferItems, replayBuffer } from 'mdml-engine/rl'
 * let b = replayBuffer<number>(2)
 * for (const x of [1, 2, 3]) b = bufferAdd(b, x)
 * bufferItems(b) // => [2, 3]
 * ```
 */
export function bufferAdd<T>(buffer: ReplayBuffer<T>, item: T): ReplayBuffer<T> {
  const c = Math.floor(buffer.next / buffer.chunkSize)
  const j = buffer.next % buffer.chunkSize
  const chunk = [...(buffer.chunks[c] ?? [])]
  chunk[j] = item
  const chunks = [...buffer.chunks]
  chunks[c] = chunk
  return {
    ...buffer,
    chunks,
    size: Math.min(buffer.capacity, buffer.size + 1),
    next: (buffer.next + 1) % buffer.capacity,
    added: buffer.added + 1,
  }
}

/**
 * The item in a slot.
 *
 * @param buffer - The buffer.
 * @param slot - A slot in `[0, size)`.
 * @returns The item.
 * @throws MdmlError with code `value` for an empty slot.
 * @concept experience-replay
 * @example
 * ```ts
 * import { bufferAdd, bufferGet, replayBuffer } from 'mdml-engine/rl'
 * bufferGet(bufferAdd(replayBuffer<string>(4), 'a'), 0) // => 'a'
 * ```
 */
export function bufferGet<T>(buffer: ReplayBuffer<T>, slot: number): T {
  if (!(slot >= 0 && slot < buffer.size)) throw new MdmlError('value', `bufferGet: slot ${slot} is empty`)
  return (buffer.chunks[Math.floor(slot / buffer.chunkSize)] as readonly T[])[slot % buffer.chunkSize] as T
}

/**
 * Every stored item, oldest first.
 *
 * @param buffer - The buffer.
 * @returns The items.
 * @concept experience-replay
 * @example
 * ```ts
 * import { bufferAdd, bufferItems, replayBuffer } from 'mdml-engine/rl'
 * bufferItems(bufferAdd(bufferAdd(replayBuffer<number>(3), 1), 2)) // => [1, 2]
 * ```
 */
export function bufferItems<T>(buffer: ReplayBuffer<T>): T[] {
  const start = buffer.size < buffer.capacity ? 0 : buffer.next
  return Array.from({ length: buffer.size }, (_, i) => bufferGet(buffer, (start + i) % buffer.capacity))
}

/**
 * A uniform sample with replacement (stable-baselines3's `ReplayBuffer.sample`).
 *
 * @param buffer - A non-empty buffer.
 * @param key - The key.
 * @param n - The batch size.
 * @returns The sampled slots and items.
 * @throws MdmlError with code `value` for an empty buffer.
 * @concept experience-replay
 * @example
 * ```ts
 * import { bufferAdd, bufferSample, replayBuffer } from 'mdml-engine/rl'
 * import { key } from 'mdml-engine/random'
 * bufferSample(bufferAdd(replayBuffer<number>(4), 7), key(0), 3).items // => [7, 7, 7]
 * ```
 */
export function bufferSample<T>(
  buffer: ReplayBuffer<T>,
  key: Key,
  n: number,
): { readonly slots: number[]; readonly items: T[] } {
  if (buffer.size === 0) throw new MdmlError('value', 'bufferSample: the buffer is empty')
  const slots = Array.from(randint(key, [n], 0, buffer.size).data)
  return { slots, items: slots.map((s) => bufferGet(buffer, s)) }
}

// ------------------------------------------------------------------------------------------------------ sum trees

interface SumNode {
  readonly sum: number
  readonly min: number
  readonly left?: SumNode
  readonly right?: SumNode
}

/**
 * A persistent sum tree over `capacity` non-negative values (with their minimum), for proportional sampling in
 * O(log n): setting a value copies one root-to-leaf path, so older trees stay valid.
 *
 * @concept experience-replay
 */
export interface SumTree {
  readonly capacity: number
  readonly depth: number
  readonly root: SumNode
}

const EMPTY_LEAF: SumNode = { sum: 0, min: Infinity }

/**
 * A sum tree of zeros (empty leaves do not count towards the minimum).
 *
 * @param capacity - The number of leaves.
 * @returns The tree.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeTotal } from 'mdml-engine/rl'
 * sumTreeTotal(sumTree(8)) // => 0
 * ```
 */
export function sumTree(capacity: number): SumTree {
  if (!(Number.isInteger(capacity) && capacity >= 1)) throw new MdmlError('value', 'sumTree: capacity must be positive')
  const depth = Math.ceil(Math.log2(Math.max(capacity, 1)))
  let node = EMPTY_LEAF
  for (let d = 0; d < depth; d++) node = { sum: 0, min: Infinity, left: node, right: node }
  return { capacity, depth, root: node }
}

/**
 * Set the value of a leaf.
 *
 * @param tree - The tree.
 * @param index - The leaf in `[0, capacity)`.
 * @param value - A non-negative value.
 * @returns The new tree.
 * @throws MdmlError with code `value` for an index out of range or a negative value.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeSet, sumTreeTotal } from 'mdml-engine/rl'
 * sumTreeTotal(sumTreeSet(sumTreeSet(sumTree(4), 0, 1), 3, 2)) // => 3
 * ```
 */
export function sumTreeSet(tree: SumTree, index: number, value: number): SumTree {
  if (!(index >= 0 && index < tree.capacity)) throw new MdmlError('value', `sumTreeSet: index ${index} out of range`)
  if (!(value >= 0)) throw new MdmlError('value', 'sumTreeSet: values must be non-negative')
  const set = (node: SumNode, depth: number, offset: number): SumNode => {
    if (depth === 0) return { sum: value, min: value }
    const half = 2 ** (depth - 1)
    const left = node.left ?? EMPTY_LEAF
    const right = node.right ?? EMPTY_LEAF
    const l = index < offset + half ? set(left, depth - 1, offset) : left
    const r = index < offset + half ? right : set(right, depth - 1, offset + half)
    return { sum: l.sum + r.sum, min: Math.min(l.min, r.min), left: l, right: r }
  }
  return { ...tree, root: set(tree.root, tree.depth, 0) }
}

/**
 * The value of a leaf.
 *
 * @param tree - The tree.
 * @param index - The leaf.
 * @returns Its value.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeGet, sumTreeSet } from 'mdml-engine/rl'
 * sumTreeGet(sumTreeSet(sumTree(4), 2, 5), 2) // => 5
 * ```
 */
export function sumTreeGet(tree: SumTree, index: number): number {
  let node = tree.root
  let offset = 0
  for (let d = tree.depth; d > 0; d--) {
    const half = 2 ** (d - 1)
    if (index < offset + half) node = node.left ?? EMPTY_LEAF
    else {
      node = node.right ?? EMPTY_LEAF
      offset += half
    }
  }
  return node.sum
}

/**
 * The sum of all leaves.
 *
 * @param tree - The tree.
 * @returns The total.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeSet, sumTreeTotal } from 'mdml-engine/rl'
 * sumTreeTotal(sumTreeSet(sumTree(2), 1, 4)) // => 4
 * ```
 */
export function sumTreeTotal(tree: SumTree): number {
  return tree.root.sum
}

/**
 * The smallest value among the leaves that have been set (`Infinity` for an empty tree).
 *
 * @param tree - The tree.
 * @returns The minimum.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeMin, sumTreeSet } from 'mdml-engine/rl'
 * sumTreeMin(sumTreeSet(sumTreeSet(sumTree(4), 0, 3), 1, 2)) // => 2
 * ```
 */
export function sumTreeMin(tree: SumTree): number {
  return tree.root.min
}

/**
 * The leaf at which the running sum first exceeds `mass` (proportional sampling: draw `mass` uniformly in
 * `[0, total)`).
 *
 * @param tree - A tree with a positive total.
 * @param mass - A number in `[0, total)` (clamped).
 * @returns The leaf index.
 * @concept experience-replay
 * @example
 * ```ts
 * import { sumTree, sumTreeFind, sumTreeSet } from 'mdml-engine/rl'
 * const t = sumTreeSet(sumTreeSet(sumTree(4), 0, 1), 2, 3)
 * [sumTreeFind(t, 0.5), sumTreeFind(t, 1.5)] // => [0, 2]
 * ```
 */
export function sumTreeFind(tree: SumTree, mass: number): number {
  let m = Math.min(Math.max(mass, 0), tree.root.sum * (1 - 1e-15))
  let node = tree.root
  let index = 0
  for (let d = tree.depth; d > 0; d--) {
    const left = node.left ?? EMPTY_LEAF
    if (m < left.sum || (node.right ?? EMPTY_LEAF).sum === 0) node = left
    else {
      m -= left.sum
      node = node.right ?? EMPTY_LEAF
      index += 2 ** (d - 1)
    }
  }
  return Math.min(index, tree.capacity - 1)
}

// ---------------------------------------------------------------------------------------------- prioritised replay

/**
 * A prioritised replay buffer (Schaul et al. 2016, proportional variant): item `i` is sampled with probability
 * `p_i^alpha / sum_j p_j^alpha`, and importance weights `(N P(i))^-beta / max_j (N P(j))^-beta` correct the bias.
 * New items get the largest priority seen so far.
 *
 * @concept experience-replay
 */
export interface PrioritizedReplayBuffer<T> {
  readonly buffer: ReplayBuffer<T>
  readonly tree: SumTree
  readonly alpha: number
  readonly epsilon: number
  readonly maxPriority: number
}

/**
 * An empty prioritised replay buffer.
 *
 * @param capacity - The capacity.
 * @param options - `alpha` (default 0.6), the priority offset `epsilon` (default 1e-6) and the chunk size.
 * @returns The buffer.
 * @concept experience-replay
 * @example
 * ```ts
 * import { prioritizedReplay } from 'mdml-engine/rl'
 * prioritizedReplay<number>(8).alpha // => 0.6
 * ```
 */
export function prioritizedReplay<T>(
  capacity: number,
  options: { readonly alpha?: number; readonly epsilon?: number; readonly chunkSize?: number } = {},
): PrioritizedReplayBuffer<T> {
  return {
    buffer: replayBuffer<T>(capacity, options.chunkSize === undefined ? {} : { chunkSize: options.chunkSize }),
    tree: sumTree(capacity),
    alpha: options.alpha ?? 0.6,
    epsilon: options.epsilon ?? 1e-6,
    maxPriority: 1,
  }
}

/**
 * Add an item with the maximal priority.
 *
 * @param buffer - The buffer.
 * @param item - The item.
 * @returns The new buffer.
 * @concept experience-replay
 * @example
 * ```ts
 * import { prioritizedAdd, prioritizedReplay } from 'mdml-engine/rl'
 * prioritizedAdd(prioritizedReplay<number>(4), 1).buffer.size // => 1
 * ```
 */
export function prioritizedAdd<T>(buffer: PrioritizedReplayBuffer<T>, item: T): PrioritizedReplayBuffer<T> {
  const slot = buffer.buffer.next
  return {
    ...buffer,
    buffer: bufferAdd(buffer.buffer, item),
    tree: sumTreeSet(buffer.tree, slot, buffer.maxPriority ** buffer.alpha),
  }
}

/** A prioritised sample. */
export interface PrioritizedSample<T> {
  readonly slots: number[]
  readonly items: T[]
  /** Importance-sampling weights, normalised so the largest possible weight is 1. */
  readonly weights: number[]
  /** The sampling probabilities of the slots. */
  readonly probabilities: number[]
}

/**
 * Sample by priority with stratification (one draw from each of `n` equal slices of the total priority), with
 * importance weights for exponent `beta`.
 *
 * @param buffer - A non-empty buffer.
 * @param key - The key.
 * @param n - The batch size.
 * @param beta - The importance-sampling exponent (default 0.4; annealed to 1 in the paper).
 * @returns Slots, items, weights and probabilities.
 * @throws MdmlError with code `value` for an empty buffer.
 * @concept experience-replay
 * @example
 * ```ts
 * import { prioritizedAdd, prioritizedReplay, prioritizedSample } from 'mdml-engine/rl'
 * import { key } from 'mdml-engine/random'
 * prioritizedSample(prioritizedAdd(prioritizedReplay<number>(4), 9), key(0), 2).weights // => [1, 1]
 * ```
 */
export function prioritizedSample<T>(
  buffer: PrioritizedReplayBuffer<T>,
  key: Key,
  n: number,
  beta = 0.4,
): PrioritizedSample<T> {
  const size = buffer.buffer.size
  if (size === 0) throw new MdmlError('value', 'prioritizedSample: the buffer is empty')
  const total = sumTreeTotal(buffer.tree)
  const u = uniform(key, [n]).data
  const segment = total / n
  const slots = Array.from({ length: n }, (_, i) =>
    Math.min(sumTreeFind(buffer.tree, (i + (u[i] ?? 0)) * segment), size - 1),
  )
  const probabilities = slots.map((s) => sumTreeGet(buffer.tree, s) / total)
  const pMin = sumTreeMin(buffer.tree) / total
  const maxWeight = (size * pMin) ** -beta
  return {
    slots,
    items: slots.map((s) => bufferGet(buffer.buffer, s)),
    weights: probabilities.map((p) => (size * p) ** -beta / maxWeight),
    probabilities,
  }
}

/**
 * Update the priorities of sampled slots from their new TD errors: `p = (|delta| + epsilon)^alpha`.
 *
 * @param buffer - The buffer.
 * @param slots - The slots sampled.
 * @param errors - Their TD errors (or any priority signal).
 * @returns The new buffer.
 * @concept experience-replay
 * @example
 * ```ts
 * import { prioritizedAdd, prioritizedReplay, prioritizedUpdate } from 'mdml-engine/rl'
 * prioritizedUpdate(prioritizedAdd(prioritizedReplay<number>(4), 1), [0], [3]).maxPriority // => 3.000001
 * ```
 */
export function prioritizedUpdate<T>(
  buffer: PrioritizedReplayBuffer<T>,
  slots: readonly number[],
  errors: readonly number[],
): PrioritizedReplayBuffer<T> {
  let tree = buffer.tree
  let maxPriority = buffer.maxPriority
  for (const [i, s] of slots.entries()) {
    const p = Math.abs(errors[i] ?? 0) + buffer.epsilon
    maxPriority = Math.max(maxPriority, p)
    tree = sumTreeSet(tree, s, p ** buffer.alpha)
  }
  return { ...buffer, tree, maxPriority }
}

// ------------------------------------------------------------------------------------------------- n-step buffers

/** An n-step transition: rewards summed over up to `n` steps, and the discount of the bootstrap. */
export interface NStepTransition<O, A> extends Transition<O, A> {
  /** The number of steps summed. */
  readonly steps: number
  /** `gamma^steps`: the factor of `V(nextObservation)` in the target (0 when terminated is used instead). */
  readonly discount: number
}

/** The pending transitions of {@link nStepAdd}. */
export interface NStepBuffer<O, A> {
  readonly n: number
  readonly gamma: number
  readonly pending: readonly Transition<O, A>[]
}

/**
 * An empty n-step accumulator (as in Rainbow and stable-baselines3's `n_steps` replay).
 *
 * @param n - The number of steps.
 * @param gamma - The discount.
 * @returns The accumulator.
 * @concept experience-replay
 * @example
 * ```ts
 * import { nStepBuffer } from 'mdml-engine/rl'
 * nStepBuffer(3, 0.99).pending.length // => 0
 * ```
 */
export function nStepBuffer<O, A>(n: number, gamma: number): NStepBuffer<O, A> {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'nStepBuffer: n must be a positive integer')
  return { n, gamma, pending: [] }
}

function combine<O, A>(items: readonly Transition<O, A>[], gamma: number): NStepTransition<O, A> {
  const first = items[0] as Transition<O, A>
  const last = items[items.length - 1] as Transition<O, A>
  let reward = 0
  let discount = 1
  for (const t of items) {
    reward += discount * t.reward
    discount *= gamma
  }
  return {
    observation: first.observation,
    action: first.action,
    reward,
    nextObservation: last.nextObservation,
    terminated: last.terminated,
    truncated: last.truncated,
    steps: items.length,
    discount,
  }
}

/**
 * Add a one-step transition and collect the n-step transitions it completes: once `n` are pending the oldest is
 * emitted with the discounted sum of `n` rewards; at the end of an episode every pending transition is flushed with
 * its shortened horizon.
 *
 * @param buffer - The accumulator.
 * @param transition - The new one-step transition.
 * @returns The new accumulator and the completed n-step transitions.
 * @concept experience-replay
 * @example
 * ```ts
 * import { nStepAdd, nStepBuffer } from 'mdml-engine/rl'
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 1, terminated: false, truncated: false }
 * const { buffer } = nStepAdd(nStepBuffer<number, number>(2, 0.5), t)
 * nStepAdd(buffer, { ...t, observation: 1, nextObservation: 2 }).ready[0]?.reward // => 1.5
 * ```
 */
export function nStepAdd<O, A>(
  buffer: NStepBuffer<O, A>,
  transition: Transition<O, A>,
): { readonly buffer: NStepBuffer<O, A>; readonly ready: NStepTransition<O, A>[] } {
  const pending = [...buffer.pending, transition]
  if (transition.terminated || transition.truncated) {
    const ready = pending.map((_, i) => combine(pending.slice(i), buffer.gamma))
    return { buffer: { ...buffer, pending: [] }, ready }
  }
  if (pending.length < buffer.n) return { buffer: { ...buffer, pending }, ready: [] }
  return { buffer: { ...buffer, pending: pending.slice(1) }, ready: [combine(pending, buffer.gamma)] }
}

// --------------------------------------------------------------------------------------------- on-policy rollouts

/** One step of an on-policy rollout. */
export interface RolloutStep<O, A> {
  readonly observation: O
  readonly action: A
  readonly reward: number
  /** The value estimate `V(s_t)` when the step was taken. */
  readonly value: number
  /** The log-probability of the action under the behaviour policy. */
  readonly logProb: number
  readonly terminated: boolean
  readonly truncated: boolean
  /** For truncated steps: `V(s_{t+1})` of the final observation, added as `gamma V` to the reward (bootstrap). */
  readonly bootstrapValue?: number
}

/**
 * An on-policy rollout buffer (stable-baselines3's RolloutBuffer): the steps of the current policy since the last
 * update, with their values and log-probabilities, turned into GAE advantages and returns.
 *
 * @concept policy-gradient
 */
export interface RolloutBuffer<O, A> {
  readonly steps: readonly RolloutStep<O, A>[]
}

/**
 * An empty rollout buffer.
 *
 * @returns The buffer.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { rolloutBuffer } from 'mdml-engine/rl'
 * rolloutBuffer().steps.length // => 0
 * ```
 */
export function rolloutBuffer<O, A>(): RolloutBuffer<O, A> {
  return { steps: [] }
}

/**
 * Append a step.
 *
 * @param buffer - The buffer.
 * @param step - The step.
 * @returns The new buffer.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { rolloutAdd, rolloutBuffer } from 'mdml-engine/rl'
 * const s = { observation: 0, action: 1, reward: 1, value: 0, logProb: 0, terminated: false, truncated: false }
 * rolloutAdd(rolloutBuffer<number, number>(), s).steps.length // => 1
 * ```
 */
export function rolloutAdd<O, A>(buffer: RolloutBuffer<O, A>, step: RolloutStep<O, A>): RolloutBuffer<O, A> {
  return { steps: [...buffer.steps, step] }
}

/**
 * GAE advantages and returns for a rollout (stable-baselines3's `compute_returns_and_advantage`): truncated steps add
 * `gamma V(final observation)` to their reward before the episode boundary, as stable-baselines3 does.
 *
 * @param buffer - The rollout.
 * @param lastValue - `V` of the observation after the last step (0 if it ended an episode).
 * @param gamma - The discount (default 0.99).
 * @param lambda - The GAE parameter (default 0.95).
 * @returns Advantages and returns per step.
 * @concept advantage-estimation
 * @example
 * ```ts
 * import { rolloutAdd, rolloutAdvantages, rolloutBuffer } from 'mdml-engine/rl'
 * const s = { observation: 0, action: 0, reward: 1, value: 0, logProb: 0, terminated: true, truncated: false }
 * rolloutAdvantages(rolloutAdd(rolloutBuffer<number, number>(), s), 0).advantages // => [1]
 * ```
 */
export function rolloutAdvantages<O, A>(
  buffer: RolloutBuffer<O, A>,
  lastValue: number,
  gamma = 0.99,
  lambda = 0.95,
): { readonly advantages: number[]; readonly returns: number[] } {
  const steps = buffer.steps
  const rewards = steps.map((s) => s.reward + (s.truncated && !s.terminated ? gamma * (s.bootstrapValue ?? 0) : 0))
  return gae(
    rewards,
    steps.map((s) => s.value),
    steps.map((s) => s.terminated || s.truncated),
    lastValue,
    gamma,
    lambda,
  )
}

/**
 * Shuffled minibatches of indices for epochs of updates (stable-baselines3's `RolloutBuffer.get(batch_size)`).
 *
 * @param n - The number of items.
 * @param batchSize - The minibatch size (the last batch may be smaller).
 * @param key - The key for the permutation.
 * @returns The minibatches of indices.
 * @concept policy-gradient
 * @example
 * ```ts
 * import { minibatchIndices } from 'mdml-engine/rl'
 * import { key } from 'mdml-engine/random'
 * minibatchIndices(5, 2, key(0)).map((b) => b.length) // => [2, 2, 1]
 * ```
 */
export function minibatchIndices(n: number, batchSize: number, key: Key): number[][] {
  const order = Array.from(permutation(key, n).data)
  const out: number[][] = []
  for (let i = 0; i < n; i += batchSize) out.push(order.slice(i, i + batchSize))
  return out
}

// --------------------------------------------------------------------------------------------- episode storage

/**
 * Whole episodes for sequence models (recurrent agents, decision transformers, world models): items are appended to
 * the current episode, closed at its end, and the oldest episodes are dropped beyond the capacity.
 *
 * @concept experience-replay
 */
export interface EpisodeStore<T> {
  readonly capacity: number
  readonly episodes: readonly (readonly T[])[]
  readonly current: readonly T[]
}

/**
 * An empty episode store.
 *
 * @param capacity - The number of finished episodes kept.
 * @returns The store.
 * @concept experience-replay
 * @example
 * ```ts
 * import { episodeStore } from 'mdml-engine/rl'
 * episodeStore<number>(10).episodes.length // => 0
 * ```
 */
export function episodeStore<T>(capacity: number): EpisodeStore<T> {
  if (!(Number.isInteger(capacity) && capacity >= 1))
    throw new MdmlError('value', 'episodeStore: capacity must be positive')
  return { capacity, episodes: [], current: [] }
}

/**
 * Append an item to the current episode, closing it when `done`.
 *
 * @param store - The store.
 * @param item - The item.
 * @param done - Whether the episode ended with this item.
 * @returns The new store.
 * @concept experience-replay
 * @example
 * ```ts
 * import { episodeAdd, episodeStore } from 'mdml-engine/rl'
 * episodeAdd(episodeAdd(episodeStore<number>(5), 1, false), 2, true).episodes // => [[1, 2]]
 * ```
 */
export function episodeAdd<T>(store: EpisodeStore<T>, item: T, done: boolean): EpisodeStore<T> {
  const current = [...store.current, item]
  if (!done) return { ...store, current }
  return { ...store, episodes: [...store.episodes, current].slice(-store.capacity), current: [] }
}

/**
 * Sample subsequences of finished episodes: `batch` windows of `length` consecutive items, each from an episode drawn
 * in proportion to its number of windows (episodes shorter than `length` give themselves whole).
 *
 * @param store - A store with at least one finished episode.
 * @param key - The key.
 * @param batch - The number of sequences.
 * @param length - The window length.
 * @returns The sequences.
 * @throws MdmlError with code `value` when no episode has finished.
 * @concept experience-replay
 * @example
 * ```ts
 * import { episodeAdd, episodeStore, sampleSequences } from 'mdml-engine/rl'
 * import { key } from 'mdml-engine/random'
 * let s = episodeStore<number>(5)
 * for (const x of [1, 2, 3]) s = episodeAdd(s, x, x === 3)
 * sampleSequences(s, key(0), 2, 3) // => [[1, 2, 3], [1, 2, 3]]
 * ```
 */
export function sampleSequences<T>(store: EpisodeStore<T>, key: Key, batch: number, length: number): T[][] {
  const eps = store.episodes
  if (eps.length === 0) throw new MdmlError('value', 'sampleSequences: no finished episodes')
  const windows = eps.map((e) => Math.max(1, e.length - length + 1))
  let total = 0
  for (const w of windows) total += w
  const u = uniform(foldIn(key, 0), [batch]).data
  return Array.from({ length: batch }, (_, b) => {
    let r = Math.floor((u[b] ?? 0) * total)
    let e = 0
    while (r >= (windows[e] ?? 0)) {
      r -= windows[e] ?? 0
      e += 1
    }
    const ep = eps[e] as readonly T[]
    return ep.slice(r, r + length)
  })
}
