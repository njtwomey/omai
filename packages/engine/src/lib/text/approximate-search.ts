/**
 * Approximate (fuzzy) string search: finding the strings or substrings within a small edit distance of a query. Q-gram
 * indexes with the count filter, BK-trees that prune by the triangle inequality, tries searched with an incremental
 * Levenshtein row, Sellers' approximate substring matching, and top-k fuzzy matching with length and early-termination
 * pruning as a traced algorithm.
 */

import { heap, type Json, MdmlError } from '../base/index.ts'
import { type Algorithm, runAlgorithm, type TracePolicy, type Tracer } from '../trace/index.ts'
import { levenshtein } from './similarity.ts'

/** A string with its distance to a query. */
export interface FuzzyMatch {
  readonly value: string
  readonly distance: number
}

function checkK(k: number, name: string): void {
  if (!Number.isInteger(k) || k < 0) throw new MdmlError('value', `${name}: k must be a non-negative integer`)
}

// ---------------------------------------------------------------------------------------------------------------
// q-gram index with count filtering.

/** A q-gram index: the padded q-grams of each string, and for each q-gram the strings containing it with counts. */
export interface QgramIndex {
  readonly q: number
  readonly strings: readonly string[]
  readonly lengths: readonly number[]
  readonly postings: ReadonlyMap<string, readonly (readonly [number, number])[]>
}

/** The q-grams of a string padded with q - 1 sentinels on both sides (so every character is in q of them). */
function paddedQgrams(s: string, q: number): Map<string, number> {
  const chars = [
    ...Array.from({ length: q - 1 }, () => '\u0001'),
    ...Array.from(s),
    ...Array.from({ length: q - 1 }, () => '\u0002'),
  ]
  const out = new Map<string, number>()
  for (let i = 0; i + q <= chars.length; i++) {
    const g = chars.slice(i, i + q).join('')
    out.set(g, (out.get(g) ?? 0) + 1)
  }
  return out
}

/**
 * Index strings by their padded q-grams, for approximate search with the q-gram count filter.
 *
 * @param strings - The strings.
 * @param q - The gram length (default 2).
 * @returns The index.
 * @throws MdmlError with code `value` for q below 1.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { qgramIndex } from 'mdml-engine/text'
 * qgramIndex(['abc', 'abd']).postings.get('ab') // => [[0, 1], [1, 1]]
 * ```
 */
export function qgramIndex(strings: readonly string[], q = 2): QgramIndex {
  if (!Number.isInteger(q) || q < 1) throw new MdmlError('value', 'qgramIndex: q must be a positive integer')
  const postings = new Map<string, [number, number][]>()
  for (const [i, s] of strings.entries())
    for (const [g, c] of paddedQgrams(s, q)) {
      const list = postings.get(g) ?? []
      list.push([i, c])
      postings.set(g, list)
    }
  return { q, strings: [...strings], lengths: strings.map((s) => Array.from(s).length), postings }
}

/**
 * Approximate search in a q-gram index (Ukkonen 1992; Gravano et al. 2001): a string within edit distance k of the
 * query shares at least max(|x|, |y|) + q - 1 - kq padded q-grams with it, so only strings passing that count (and
 * the length filter ||x| - |y|| <= k) are verified with the edit distance.
 *
 * @param index - The index.
 * @param query - The query.
 * @param k - The largest edit distance.
 * @returns The matches (sorted by distance, then by index order) and the number of candidates verified.
 * @throws MdmlError with code `value` for a negative k.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { qgramIndex, qgramSearch } from 'mdml-engine/text'
 * qgramSearch(qgramIndex(['kitten', 'sitting', 'mitten', 'bitter']), 'kitten', 1).matches // => [{ value: 'kitten', distance: 0 }, { value: 'mitten', distance: 1 }]
 * ```
 */
export function qgramSearch(
  index: QgramIndex,
  query: string,
  k: number,
): { readonly matches: FuzzyMatch[]; readonly candidates: number } {
  checkK(k, 'qgramSearch')
  const qlen = Array.from(query).length
  const shared = new Map<number, number>()
  for (const [g, c] of paddedQgrams(query, index.q))
    for (const [i, ci] of index.postings.get(g) ?? []) shared.set(i, (shared.get(i) ?? 0) + Math.min(c, ci))
  const matches: (FuzzyMatch & { i: number })[] = []
  let candidates = 0
  for (const [i, len] of index.lengths.entries()) {
    if (Math.abs(len - qlen) > k) continue
    const needed = Math.max(len, qlen) + index.q - 1 - k * index.q
    if ((shared.get(i) ?? 0) < needed) continue
    candidates++
    const value = index.strings[i] ?? ''
    const d = levenshtein(query, value)
    if (d <= k) matches.push({ value, distance: d, i })
  }
  matches.sort((a, b) => a.distance - b.distance || a.i - b.i)
  return { matches: matches.map(({ value, distance }) => ({ value, distance })), candidates }
}

// ---------------------------------------------------------------------------------------------------------------
// BK-trees.

/** A node of a BK-tree: a word and its children by distance. */
export interface BkNode {
  readonly word: string
  readonly children: Map<number, BkNode>
}

/** A BK-tree (Burkhard and Keller 1973) over a metric. */
export interface BkTree {
  readonly root: BkNode | undefined
  readonly distance: (a: string, b: string) => number
  readonly size: number
}

/**
 * Build a BK-tree: each child of a node holds the words at one distance from it, so the triangle inequality bounds
 * which subtrees can contain words near a query.
 *
 * @param words - The words (repeats ignored).
 * @param distance - A metric (default Levenshtein).
 * @returns The tree.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { bkTree } from 'mdml-engine/text'
 * bkTree(['book', 'books', 'cake', 'boo']).size // => 4
 * ```
 */
export function bkTree(
  words: Iterable<string>,
  distance: (a: string, b: string) => number = (a, b) => levenshtein(a, b),
): BkTree {
  let root: BkNode | undefined
  let size = 0
  for (const w of words) {
    if (root === undefined) {
      root = { word: w, children: new Map() }
      size++
      continue
    }
    let node = root
    for (;;) {
      const d = distance(w, node.word)
      if (d === 0) break
      const child = node.children.get(d)
      if (child === undefined) {
        node.children.set(d, { word: w, children: new Map() })
        size++
        break
      }
      node = child
    }
  }
  return { root, distance, size }
}

/**
 * The words of a BK-tree within distance k of a query, visiting only children whose distance from their parent is
 * within k of the query's.
 *
 * @param tree - The tree.
 * @param query - The query.
 * @param k - The largest distance.
 * @returns The matches (sorted by distance, then by word) and the number of distances computed.
 * @throws MdmlError with code `value` for a negative k.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { bkTree, bkSearch } from 'mdml-engine/text'
 * bkSearch(bkTree(['book', 'books', 'cake', 'boo', 'cape']), 'bool', 1).matches // => [{ value: 'boo', distance: 1 }, { value: 'book', distance: 1 }]
 * ```
 */
export function bkSearch(
  tree: BkTree,
  query: string,
  k: number,
): { readonly matches: FuzzyMatch[]; readonly evaluations: number } {
  checkK(k, 'bkSearch')
  const matches: FuzzyMatch[] = []
  let evaluations = 0
  const stack = tree.root === undefined ? [] : [tree.root]
  while (stack.length > 0) {
    const node = stack.pop() as BkNode
    const d = tree.distance(query, node.word)
    evaluations++
    if (d <= k) matches.push({ value: node.word, distance: d })
    for (const [cd, child] of node.children) if (cd >= d - k && cd <= d + k) stack.push(child)
  }
  matches.sort((a, b) => a.distance - b.distance || (a.value < b.value ? -1 : a.value > b.value ? 1 : 0))
  return { matches, evaluations }
}

// ---------------------------------------------------------------------------------------------------------------
// Tries with fuzzy lookup.

/** A trie node over code points. */
export interface TrieNode {
  readonly children: Map<string, TrieNode>
  word?: string
}

/**
 * A trie of words (by code point).
 *
 * @param words - The words.
 * @returns The root.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { trie } from 'mdml-engine/text'
 * [...trie(['to', 'tea', 'ten']).children.get('t')?.children.keys() ?? []] // => ['o', 'e']
 * ```
 */
export function trie(words: Iterable<string>): TrieNode {
  const root: TrieNode = { children: new Map() }
  for (const w of words) {
    let node = root
    for (const ch of w) {
      let next = node.children.get(ch)
      if (next === undefined) {
        next = { children: new Map() }
        node.children.set(ch, next)
      }
      node = next
    }
    node.word = w
  }
  return root
}

/**
 * Fuzzy lookup in a trie: the Levenshtein DP row is extended one character at a time down each path, and a branch is
 * abandoned as soon as its row's minimum exceeds k (shared prefixes are computed once).
 *
 * @param root - The trie.
 * @param query - The query.
 * @param k - The largest edit distance.
 * @returns The matches (sorted by distance, then by word) and the number of trie nodes visited.
 * @throws MdmlError with code `value` for a negative k.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { trie, trieSearch } from 'mdml-engine/text'
 * trieSearch(trie(['tea', 'ten', 'to', 'inn']), 'tan', 1).matches // => [{ value: 'ten', distance: 1 }]
 * ```
 */
export function trieSearch(
  root: TrieNode,
  query: string,
  k: number,
): { readonly matches: FuzzyMatch[]; readonly visited: number } {
  checkK(k, 'trieSearch')
  const q = Array.from(query)
  const matches: FuzzyMatch[] = []
  let visited = 0
  const first = Int32Array.from({ length: q.length + 1 }, (_, j) => j)
  if (root.word !== undefined && q.length <= k) matches.push({ value: root.word, distance: q.length })
  const stack: [TrieNode, string, Int32Array][] = [...root.children].map(([ch, n]) => [n, ch, first])
  while (stack.length > 0) {
    const [node, ch, prev] = stack.pop() as [TrieNode, string, Int32Array]
    visited++
    const row = new Int32Array(q.length + 1)
    row[0] = (prev[0] ?? 0) + 1
    let min = row[0] ?? 0
    for (let j = 1; j <= q.length; j++) {
      row[j] = Math.min((row[j - 1] ?? 0) + 1, (prev[j] ?? 0) + 1, (prev[j - 1] ?? 0) + (q[j - 1] === ch ? 0 : 1))
      if ((row[j] ?? 0) < min) min = row[j] ?? 0
    }
    if (node.word !== undefined && (row[q.length] ?? 0) <= k)
      matches.push({ value: node.word, distance: row[q.length] ?? 0 })
    if (min <= k) for (const [c, n] of node.children) stack.push([n, c, row])
  }
  matches.sort((a, b) => a.distance - b.distance || (a.value < b.value ? -1 : a.value > b.value ? 1 : 0))
  return { matches, visited }
}

// ---------------------------------------------------------------------------------------------------------------
// Approximate substring matching.

/** An approximate occurrence of a pattern in a text: `[start, end)` in code points, and its edit distance. */
export interface ApproximateOccurrence {
  readonly start: number
  readonly end: number
  readonly distance: number
}

/**
 * Approximate substring matching (Sellers 1980): the edit-distance DP with a free start anywhere in the text, so
 * D[m][j] is the least distance of the pattern to a substring ending at j; each end with D[m][j] <= k is reported
 * with the start of a best alignment (the longest such substring, from the traceback preferring diagonal moves).
 *
 * @param pattern - The pattern.
 * @param text - The text.
 * @param k - The largest edit distance.
 * @returns The occurrences, by end position.
 * @throws MdmlError with code `value` for a negative k.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { sellers } from 'mdml-engine/text'
 * sellers('survey', 'surgery', 2).map((o) => [o.start, o.end, o.distance]) // => [[0, 5, 2], [0, 6, 2], [0, 7, 2]]
 * ```
 */
export function sellers(pattern: string, text: string, k: number): ApproximateOccurrence[] {
  checkK(k, 'sellers')
  const p = Array.from(pattern)
  const t = Array.from(text)
  const m = p.length
  const n = t.length
  const w = n + 1
  const D = new Int32Array((m + 1) * w)
  for (let i = 1; i <= m; i++) D[i * w] = i
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      D[i * w + j] = Math.min(
        (D[(i - 1) * w + j] ?? 0) + 1,
        (D[i * w + j - 1] ?? 0) + 1,
        (D[(i - 1) * w + j - 1] ?? 0) + (p[i - 1] === t[j - 1] ? 0 : 1),
      )
  const out: ApproximateOccurrence[] = []
  for (let j = 0; j <= n; j++) {
    const d = D[m * w + j] ?? 0
    if (d > k) continue
    let i = m
    let c = j
    while (i > 0) {
      const here = D[i * w + c] ?? 0
      if (c > 0 && here === (D[(i - 1) * w + c - 1] ?? 0) + (p[i - 1] === t[c - 1] ? 0 : 1)) {
        i--
        c--
      } else if (here === (D[(i - 1) * w + c] ?? 0) + 1) i--
      else c--
    }
    out.push({ start: c, end: j, distance: d })
  }
  return out
}

// ---------------------------------------------------------------------------------------------------------------
// Top-k fuzzy matching with traced pruning.

/** The state of a top-k fuzzy search: the next candidate, the best matches so far, and the pruning counts. */
export interface TopKState {
  readonly next: number
  readonly best: {
    push: (m: FuzzyMatch & { i: number }) => void
    pop: () => (FuzzyMatch & { i: number }) | undefined
    peek: () => (FuzzyMatch & { i: number }) | undefined
    readonly size: number
    toArray: () => (FuzzyMatch & { i: number })[]
  }
  readonly prunedByLength: number
  readonly prunedEarly: number
  readonly verified: number
  readonly last?: {
    readonly value: string
    readonly outcome: 'length' | 'early' | 'kept' | 'rejected'
    readonly distance?: number
  }
}

/** The edit distance, or undefined as soon as every entry of a DP row exceeds `limit` (Ukkonen's cut-off). */
function boundedLevenshtein(a: readonly string[], b: readonly string[], limit: number): number | undefined {
  let prev = Int32Array.from({ length: b.length + 1 }, (_, j) => j)
  let cur = new Int32Array(b.length + 1)
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i
    let min = i
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min((prev[j] ?? 0) + 1, (cur[j - 1] ?? 0) + 1, (prev[j - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1))
      if ((cur[j] ?? 0) < min) min = cur[j] ?? 0
    }
    if (min > limit) return undefined
    ;[prev, cur] = [cur, prev]
  }
  return prev[b.length] ?? 0
}

/**
 * Top-k fuzzy matching as an {@link Algorithm}: each step examines one candidate, pruning it when the length
 * difference alone exceeds the current k-th best distance, or abandoning its DP early when a row exceeds that bound;
 * each step records what happened, so a trace shows the pruning.
 *
 * @param query - The query.
 * @param candidates - The candidate strings.
 * @param k - How many matches to keep.
 * @param maxDistance - The largest distance of interest (default unbounded).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a k below 1.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { topKFuzzyAlgorithm } from 'mdml-engine/text'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const r = runAlgorithm(topKFuzzyAlgorithm('apple', ['apply', 'ape', 'maple', 'pineapple', 'a'], 2), { maxSteps: 10 })
 * [r.state.verified, r.state.prunedByLength + r.state.prunedEarly] // => [3, 2]
 * ```
 */
export function topKFuzzyAlgorithm(
  query: string,
  candidates: readonly string[],
  k: number,
  maxDistance: number = Number.POSITIVE_INFINITY,
): Algorithm<TopKState> {
  if (!Number.isInteger(k) || k < 1) throw new MdmlError('value', 'topKFuzzy: k must be a positive integer')
  const q = Array.from(query)
  return {
    name: 'topKFuzzy',
    config: { query, k, candidates: candidates.length } as Json,
    init: () => ({
      next: 0,
      best: heap<FuzzyMatch & { i: number }>((a, b) => b.distance - a.distance || b.i - a.i),
      prunedByLength: 0,
      prunedEarly: 0,
      verified: 0,
    }),
    step: (s) => {
      const i = s.next
      const value = candidates[i] ?? ''
      const c = Array.from(value)
      const full = s.best.size >= k
      const bound = Math.min(
        maxDistance,
        full ? (s.best.peek()?.distance ?? Number.POSITIVE_INFINITY) : Number.POSITIVE_INFINITY,
      )
      if (Math.abs(c.length - q.length) > bound)
        return { ...s, next: i + 1, prunedByLength: s.prunedByLength + 1, last: { value, outcome: 'length' } }
      const d = boundedLevenshtein(q, c, bound)
      if (d === undefined)
        return {
          ...s,
          next: i + 1,
          prunedEarly: s.prunedEarly + 1,
          verified: s.verified + 1,
          last: { value, outcome: 'early' },
        }
      if (d > bound || (full && d >= bound))
        return { ...s, next: i + 1, verified: s.verified + 1, last: { value, outcome: 'rejected', distance: d } }
      s.best.push({ value, distance: d, i })
      if (s.best.size > k) s.best.pop()
      return { ...s, next: i + 1, verified: s.verified + 1, last: { value, outcome: 'kept', distance: d } }
    },
    done: (s) => (s.next >= candidates.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { prunedByLength: s.prunedByLength, prunedEarly: s.prunedEarly, verified: s.verified },
      ...(s.last === undefined ? {} : { messages: [{ kind: 'candidate', payload: { ...s.last } as Json }] }),
    }),
  }
}

/**
 * The k candidates nearest a query by edit distance (ties by candidate order), with pruning by length and early
 * DP termination ({@link topKFuzzyAlgorithm}, traced when a tracer is given).
 *
 * @param query - The query.
 * @param candidates - The candidates.
 * @param k - How many.
 * @param options - The largest distance, tracer and policy.
 * @returns The matches, nearest first.
 * @concept approximate-string-matching
 * @example
 * ```ts
 * import { topKFuzzy } from 'mdml-engine/text'
 * topKFuzzy('apple', ['apply', 'ape', 'maple', 'pineapple', 'a'], 2) // => [{ value: 'apply', distance: 1 }, { value: 'ape', distance: 2 }]
 * ```
 */
export function topKFuzzy(
  query: string,
  candidates: readonly string[],
  k: number,
  options: { readonly maxDistance?: number; readonly tracer?: Tracer; readonly policy?: TracePolicy } = {},
): FuzzyMatch[] {
  if (candidates.length === 0) return []
  const r = runAlgorithm(topKFuzzyAlgorithm(query, candidates, k, options.maxDistance), {
    maxSteps: candidates.length,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return r.state.best
    .toArray()
    .sort((a, b) => a.distance - b.distance || a.i - b.i)
    .map(({ value, distance }) => ({ value, distance }))
}
