/**
 * Pugh's probabilistic skip list: a sorted linked list with randomly promoted express lanes, giving expected O(log n)
 * search, insertion and deletion without rebalancing. Tower heights are geometric draws from a `Key`, the search
 * returns the staircase it walked level by level, and the stepped algorithm inserts one key per step while counting the
 * comparisons a search would take.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Key, uniformStream } from '../random/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** Configuration of {@link skipList}. */
export interface SkipListOptions<K = number> {
  /** The key the tower heights come from. */
  readonly key: Key
  /** Promotion probability `p` in `(0, 1)` (default 0.5). */
  readonly p?: number
  /** Tallest tower allowed (default 16). */
  readonly maxLevel?: number
  /** Ordering of the keys: negative, zero or positive (default `<`/`>` on numbers and strings). */
  readonly compare?: (a: K, b: K) => number
  /** Tower heights to use for the first insertions, in order, instead of random draws (to reproduce a given structure). */
  readonly heights?: readonly number[]
}

/** The keys inspected at one level of a search. */
export interface SkipListLevelStep<K = number> {
  /** The level (0 is the full list). */
  readonly level: number
  /** The keys compared at this level before dropping down (the last one is not below the target). */
  readonly visitedKeys: readonly K[]
}

/** The path a search took. */
export interface SkipListSearchTrace<K = number, V = K> {
  /** The key searched for. */
  readonly target: K
  /** Whether it is in the list. */
  readonly found: boolean
  /** Its value if found. */
  readonly value: V | undefined
  /** Key comparisons made over all levels. */
  readonly comparisons: number
  /** The levels visited from the top down. */
  readonly levels: readonly SkipListLevelStep<K>[]
}

/** A node of the skip list as seen from outside. */
export interface SkipListNodeSnapshot<K = number, V = K> {
  /** The key. */
  readonly key: K
  /** The value. */
  readonly value: V
  /** The tower height. */
  readonly height: number
}

/** A skip list keeping `(key, value)` pairs in key order. */
export interface SkipList<K = number, V = K> {
  /** Number of pairs. */
  readonly size: number
  /** The tallest tower currently present. */
  readonly level: number
  /** Tallest tower allowed. */
  readonly maxLevel: number
  /** Promotion probability. */
  readonly p: number
  /** Insert a pair (replacing the value of an existing key); returns the tower height of the key's node. */
  insert(key: K, value?: V): number
  /** The value of a key, or `undefined`. */
  get(key: K): V | undefined
  /** Whether a key is present. */
  has(key: K): boolean
  /** Remove a key; `false` if it was absent. */
  delete(key: K): boolean
  /** Search for a key, recording the keys compared at each level. */
  searchPath(target: K): SkipListSearchTrace<K, V>
  /** The pairs in key order with their tower heights. */
  entries(): readonly SkipListNodeSnapshot<K, V>[]
}

interface Node<K, V> {
  readonly key: K
  value: V
  readonly forward: (Node<K, V> | null)[]
}

function defaultCompare<K>(a: K, b: K): number {
  return a < b ? -1 : a > b ? 1 : 0
}

/**
 * Create a skip list (Pugh 1990). A new node's height is `1 +` the number of consecutive promotions, each with
 * probability `p`, capped at `maxLevel`, so the expected number of nodes at level `h` is `n p^h` and a search compares
 * about `log_{1/p}(n) / p` keys.
 *
 * @param options - The `key` for the heights, `p`, `maxLevel` and an optional `compare`.
 * @returns The list.
 * @throws OmaiError with code `value` for `p` outside `(0, 1)` or a non-positive `maxLevel`.
 * @concept skip-list
 * @reference W. Pugh, Skip lists: a probabilistic alternative to balanced trees (Communications of the ACM, 1990).
 * @example
 * ```ts
 * import { skipList } from 'omai-engine/sketches'
 * import { key } from 'omai-engine/random'
 * const sl = skipList<number>({ key: key(42) })
 * for (const x of [10, 5, 20]) sl.insert(x)
 * [sl.has(10), sl.has(7), sl.entries().map((e) => e.key)] // => [true, false, [5, 10, 20]]
 * ```
 */
export function skipList<K = number, V = K>(options: SkipListOptions<K>): SkipList<K, V> {
  const p = options.p ?? 0.5
  const maxLevel = options.maxLevel ?? 16
  const compare = options.compare ?? defaultCompare
  if (!(p > 0 && p < 1)) throw new OmaiError('value', `skipList: p must lie in (0, 1), got ${p}`)
  if (!(Number.isInteger(maxLevel) && maxLevel >= 1)) {
    throw new OmaiError('value', `skipList: maxLevel must be a positive integer, got ${maxLevel}`)
  }
  const draw = uniformStream(options.key)
  const head: (Node<K, V> | null)[] = Array.from({ length: maxLevel }, () => null)
  let level = 1
  let size = 0
  let created = 0

  // The predecessor of `target` at every level (null is the head), and the node after the one at level 0.
  const descend = (target: K): { readonly update: (Node<K, V> | null)[]; readonly candidate: Node<K, V> | null } => {
    const update: (Node<K, V> | null)[] = Array.from({ length: maxLevel }, () => null)
    let x: Node<K, V> | null = null
    for (let lvl = level - 1; lvl >= 0; lvl--) {
      for (;;) {
        const next: Node<K, V> | null = x === null ? (head[lvl] ?? null) : (x.forward[lvl] ?? null)
        if (next === null || compare(next.key, target) >= 0) break
        x = next
      }
      update[lvl] = x
    }
    return { update, candidate: x === null ? (head[0] ?? null) : (x.forward[0] ?? null) }
  }
  const searchPath = (target: K): SkipListSearchTrace<K, V> => {
    const levels: SkipListLevelStep<K>[] = []
    let comparisons = 0
    let x: Node<K, V> | null = null
    for (let lvl = level - 1; lvl >= 0; lvl--) {
      const visitedKeys: K[] = []
      for (;;) {
        const next: Node<K, V> | null = x === null ? (head[lvl] ?? null) : (x.forward[lvl] ?? null)
        if (next === null) break
        comparisons += 1
        visitedKeys.push(next.key)
        if (compare(next.key, target) >= 0) break
        x = next
      }
      levels.push({ level: lvl, visitedKeys })
    }
    const candidate: Node<K, V> | null = x === null ? (head[0] ?? null) : (x.forward[0] ?? null)
    const found = candidate !== null && compare(candidate.key, target) === 0
    return { target, found, value: found && candidate !== null ? candidate.value : undefined, comparisons, levels }
  }
  return {
    get size() {
      return size
    },
    get level() {
      return level
    },
    maxLevel,
    p,
    insert: (key, value) => {
      const val = value === undefined ? (key as unknown as V) : value
      const { update, candidate } = descend(key)
      if (candidate !== null && compare(candidate.key, key) === 0) {
        candidate.value = val
        return candidate.forward.length
      }
      let height = 1
      const given = options.heights?.[created]
      if (given !== undefined) height = Math.min(maxLevel, Math.max(1, given))
      else while (height < maxLevel && draw() < p) height += 1
      created += 1
      level = Math.max(level, height)
      const node: Node<K, V> = { key, value: val, forward: Array.from({ length: height }, () => null) }
      for (let lvl = 0; lvl < height; lvl++) {
        const prev = update[lvl] ?? null
        if (prev === null) {
          node.forward[lvl] = head[lvl] ?? null
          head[lvl] = node
        } else {
          node.forward[lvl] = prev.forward[lvl] ?? null
          prev.forward[lvl] = node
        }
      }
      size += 1
      return height
    },
    get: (key) => searchPath(key).value,
    has: (key) => searchPath(key).found,
    delete: (key) => {
      const { update, candidate } = descend(key)
      if (candidate === null || compare(candidate.key, key) !== 0) return false
      for (let lvl = 0; lvl < candidate.forward.length; lvl++) {
        const prev = update[lvl] ?? null
        if (prev === null) head[lvl] = candidate.forward[lvl] ?? null
        else prev.forward[lvl] = candidate.forward[lvl] ?? null
      }
      while (level > 1 && head[level - 1] === null) level -= 1
      size -= 1
      return true
    },
    searchPath,
    entries: () => {
      const out: SkipListNodeSnapshot<K, V>[] = []
      for (let node = head[0] ?? null; node !== null; node = node.forward[0] ?? null) {
        out.push({ key: node.key, value: node.value, height: node.forward.length })
      }
      return out
    },
  }
}

/**
 * The search path for a key in a skip list: the keys compared at each level from the top, and the comparison count.
 *
 * @param list - The list.
 * @param target - The key.
 * @returns The trace.
 * @concept skip-list
 * @example
 * ```ts
 * import { skipList, skipListSearchPath } from 'omai-engine/sketches'
 * import { key } from 'omai-engine/random'
 * const sl = skipList<number>({ key: key(99) })
 * for (const x of [10, 20, 30, 40]) sl.insert(x)
 * const trace = skipListSearchPath(sl, 30)
 * [trace.found, trace.levels.at(-1)?.level] // => [true, 0]
 * ```
 */
export function skipListSearchPath<K, V = K>(list: SkipList<K, V>, target: K): SkipListSearchTrace<K, V> {
  return list.searchPath(target)
}

/** The state of {@link skipListAlgorithm}: the keys in order with the height of each tower. */
export interface SkipListState {
  /** The stored keys, sorted. */
  readonly keys: readonly number[]
  /** The tower height of each. */
  readonly heights: readonly number[]
  /** Keys offered so far. */
  readonly inserted: number
  /** Comparisons the search for the last key took before it was inserted. */
  readonly comparisons: number
}

/** Input of {@link skipListAlgorithm}. */
export interface SkipListAlgorithmOptions {
  /** The keys to insert, one per step. */
  readonly keys: readonly number[]
  /** Promotion probability (default 0.5). */
  readonly p?: number
  /** Tallest tower allowed (default 16). */
  readonly maxLevel?: number
}

// The comparisons of a search in the list given by sorted keys and their heights: level `h` is the sublist of nodes
// taller than `h`, so each level's walk is a scan over those nodes only.
function searchComparisons(keys: readonly number[], heights: readonly number[], target: number): number {
  let at = -1
  let comparisons = 0
  const top = heights.reduce((m, h) => Math.max(m, h), 0)
  for (let lvl = top - 1; lvl >= 0; lvl--) {
    for (let j = at + 1; j < keys.length; j++) {
      if ((heights[j] ?? 0) <= lvl) continue
      comparisons += 1
      if ((keys[j] ?? 0) >= target) break
      at = j
    }
  }
  return comparisons
}

/**
 * Skip list construction as an algorithm, one key per step: the step searches for the key (logging the comparisons),
 * draws its tower height from the step's key, and inserts it. The comparisons stay near `log_2(n) / p` however the keys
 * arrive, in sorted order included, which is the point of randomising the heights.
 *
 * @param options - The keys, `p` and `maxLevel`.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for `p` outside `(0, 1)` or a non-positive `maxLevel`.
 * @concept skip-list
 * @example
 * ```ts
 * import { skipListAlgorithm } from 'omai-engine/sketches'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const res = runAlgorithm(skipListAlgorithm({ keys: [30, 10, 20, 40] }), { seed: 11, maxSteps: 10 })
 * res.state.keys // => [10, 20, 30, 40]
 * ```
 */
export function skipListAlgorithm(options: SkipListAlgorithmOptions): Algorithm<SkipListState> {
  const p = options.p ?? 0.5
  const maxLevel = options.maxLevel ?? 16
  if (!(p > 0 && p < 1)) throw new OmaiError('value', `skipListAlgorithm: p must lie in (0, 1), got ${p}`)
  if (!(Number.isInteger(maxLevel) && maxLevel >= 1)) {
    throw new OmaiError('value', `skipListAlgorithm: maxLevel must be a positive integer, got ${maxLevel}`)
  }
  const keys = options.keys
  return {
    name: 'skip-list',
    config: { p, maxLevel, keys: keys.length },
    schema: channels({
      metrics: {
        size: { description: 'keys stored' },
        height: { description: 'the tallest tower' },
        comparisons: { description: 'key comparisons the search for the inserted key took' },
      },
    }),
    init: () => ({ keys: [], heights: [], inserted: 0, comparisons: 0 }),
    step: (s, ctx) => {
      const k = keys[s.inserted]
      if (k === undefined) return s
      const comparisons = searchComparisons(s.keys, s.heights, k)
      const draw = uniformStream(ctx.key, 16)
      let height = 1
      while (height < maxLevel && draw() < p) height += 1
      let at = 0
      while (at < s.keys.length && (s.keys[at] ?? 0) < k) at += 1
      if (s.keys[at] === k) return { ...s, inserted: s.inserted + 1, comparisons }
      return {
        keys: [...s.keys.slice(0, at), k, ...s.keys.slice(at)],
        heights: [...s.heights.slice(0, at), height, ...s.heights.slice(at)],
        inserted: s.inserted + 1,
        comparisons,
      }
    },
    done: (s) => (s.inserted >= keys.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        size: s.keys.length,
        height: s.heights.reduce((m, h) => Math.max(m, h), 0),
        comparisons: s.comparisons,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as SkipListState,
  }
}
