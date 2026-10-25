/**
 * Refinement search: exploring a lattice of patterns from most general to more specific, the search form of rule
 * learning, subgroup discovery and ILP. Top-k result sets with redundancy filtering, exhaustive search with
 * optimistic-estimate pruning, best-first and beam searches as traced algorithms, and the language of conjunctions of
 * selectors.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { BinaryHeap } from '../graph/index.ts'
import { metricFunction } from '../numerics/geometry/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { type SearchRunOptions, type StateKey, stateKeyOf } from './problem.ts'

/**
 * A refinement search problem (the search form of rule learning, subgroup discovery, exceptional model mining and
 * ILP): a most general pattern, a refinement operator giving its specialisations, a quality to maximise, and
 * optionally an optimistic estimate (an upper bound on the quality of a pattern and of all its refinements), an
 * anti-monotone validity constraint (such as minimum support: an invalid pattern's refinements are invalid too), and
 * a coverage (0/1 per record) for redundancy filtering.
 *
 * @concept refinement-search
 */
export interface RefinementProblem<P> {
  readonly root: P
  refine(pattern: P): readonly P[]
  quality(pattern: P): number
  optimisticEstimate?(pattern: P): number
  isValid?(pattern: P): boolean
  coverage?(pattern: P): ArrayLike<number>
  key?(pattern: P): StateKey
  label?(pattern: P): string
}

/** A pattern with its quality, depth (refinement steps from the root) and coverage when known. */
export interface ScoredPattern<P> {
  readonly pattern: P
  readonly quality: number
  readonly depth: number
  readonly coverage?: ArrayLike<number>
}

/**
 * The Jaccard similarity of two coverages (0/1 vectors): |a and b| / |a or b|, 1 for two empty coverages. Computed
 * as one minus numerics' Jaccard distance.
 *
 * @param a - A coverage.
 * @param b - Another of the same length.
 * @returns The similarity in [0, 1].
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept refinement-search
 * @example
 * ```ts
 * import { coverageSimilarity } from 'mdml-engine/search'
 * coverageSimilarity([1, 1, 0, 0], [1, 0, 1, 0]) // => 1 / 3
 * ```
 */
export function coverageSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new MdmlError('shape', 'coverageSimilarity: coverages differ in length')
  let any = false
  for (let i = 0; i < a.length; i++) if ((a[i] ?? 0) !== 0 || (b[i] ?? 0) !== 0) any = true
  if (!any) return 1
  return 1 - metricFunction('jaccard', a.length)(a, 0, b, 0)
}

/**
 * Keep the best patterns that are not redundant: in order of quality, a pattern is kept unless its similarity to a
 * kept one reaches `maxSimilarity` (greedy diverse top-k, as in pysubgroup and Leeuwen and Knobbe's DSSD).
 *
 * @param patterns - Scored patterns.
 * @param options - The similarity (default coverage Jaccard), the redundancy threshold (default 0.9) and k.
 * @returns The kept patterns, best first.
 * @concept refinement-search
 * @example
 * ```ts
 * import { redundancyFilter } from 'mdml-engine/search'
 * const scored = [
 *   { pattern: 'a', quality: 3, depth: 1, coverage: [1, 1, 0, 0] },
 *   { pattern: 'a and b', quality: 2.9, depth: 2, coverage: [1, 1, 0, 0] },
 *   { pattern: 'c', quality: 1, depth: 1, coverage: [0, 0, 1, 1] },
 * ]
 * redundancyFilter(scored).map((s) => s.pattern) // => ['a', 'c']
 * ```
 */
export function redundancyFilter<P>(
  patterns: readonly ScoredPattern<P>[],
  options: {
    readonly similarity?: (a: ScoredPattern<P>, b: ScoredPattern<P>) => number
    readonly maxSimilarity?: number
    readonly k?: number
  } = {},
): ScoredPattern<P>[] {
  const max = options.maxSimilarity ?? 0.9
  const k = options.k ?? Number.POSITIVE_INFINITY
  const similarity =
    options.similarity ??
    ((a: ScoredPattern<P>, b: ScoredPattern<P>) =>
      a.coverage === undefined || b.coverage === undefined ? 0 : coverageSimilarity(a.coverage, b.coverage))
  const kept: ScoredPattern<P>[] = []
  const ordered = patterns.map((p, i) => ({ p, i })).sort((x, y) => y.p.quality - x.p.quality || x.i - y.i)
  for (const { p } of ordered) {
    if (kept.length >= k) break
    if (kept.every((q) => similarity(p, q) < max)) kept.push(p)
  }
  return kept
}

/**
 * A top-k result set: the k best patterns offered so far (optionally without redundant ones), with the quality a new
 * pattern must beat to enter (the pruning threshold of optimistic-estimate search).
 *
 * @concept refinement-search
 * @example
 * ```ts
 * import { TopK } from 'mdml-engine/search'
 * const top = new TopK<string>(2)
 * top.add({ pattern: 'a', quality: 1, depth: 1 })
 * top.add({ pattern: 'b', quality: 3, depth: 1 })
 * top.add({ pattern: 'c', quality: 2, depth: 1 })
 * [top.items.map((x) => x.pattern), top.threshold] // => [['b', 'c'], 2]
 * ```
 */
export class TopK<P> {
  readonly k: number
  readonly #redundancy: { readonly maxSimilarity: number } | undefined
  #items: ScoredPattern<P>[] = []

  /**
   * An empty result set.
   *
   * @param k - The capacity.
   * @param options - Redundancy filtering by coverage similarity (off by default).
   * @returns The set.
   * @throws MdmlError with code `value` when k is not a positive integer.
   * @example
   * ```ts
   * import { TopK } from 'mdml-engine/search'
   * new TopK(3).threshold // => -Infinity
   * ```
   */
  constructor(k: number, options: { readonly maxSimilarity?: number } = {}) {
    if (!(Number.isInteger(k) && k >= 1)) throw new MdmlError('value', `TopK: k = ${k} must be a positive integer`)
    this.k = k
    this.#redundancy = options.maxSimilarity === undefined ? undefined : { maxSimilarity: options.maxSimilarity }
  }

  /** The patterns kept, best first. */
  get items(): readonly ScoredPattern<P>[] {
    return this.#items
  }

  /** The quality to beat: the k-th best quality, or -Infinity while fewer than k are held. */
  get threshold(): number {
    return this.#items.length < this.k
      ? Number.NEGATIVE_INFINITY
      : (this.#items[this.k - 1]?.quality ?? Number.NEGATIVE_INFINITY)
  }

  /**
   * Offer a pattern.
   *
   * @param item - The scored pattern.
   * @returns True when it is kept.
   * @example
   * ```ts
   * import { TopK } from 'mdml-engine/search'
   * new TopK<string>(1).add({ pattern: 'a', quality: 1, depth: 0 }) // => true
   * ```
   */
  add(item: ScoredPattern<P>): boolean {
    if (this.#items.length >= this.k && item.quality <= this.threshold) return false
    const all = [...this.#items, item]
    this.#items =
      this.#redundancy === undefined
        ? all.sort((a, b) => b.quality - a.quality).slice(0, this.k)
        : redundancyFilter(all, { maxSimilarity: this.#redundancy.maxSimilarity, k: this.k })
    return this.#items.includes(item)
  }
}

/** Options shared by the refinement searches. */
export interface RefinementOptions extends SearchRunOptions {
  /** The size of the result set (default 10). */
  readonly k?: number
  /** The deepest refinement (default 3). */
  readonly maxDepth?: number
  /** Patterns below this quality are not kept (default -Infinity). */
  readonly minQuality?: number
  /**
   * Filter redundant results by coverage similarity at this threshold (default off). Pruning then compares estimates
   * with the k-th quality of the filtered set, so a pruned search may keep a slightly different diverse set than an
   * unpruned one.
   */
  readonly maxSimilarity?: number
  /** Include the root pattern among the results (default false). */
  readonly includeRoot?: boolean
  /** Stop after this many expansions (default 100000). */
  readonly maxExpansions?: number
}

/** The result of a refinement search. */
export interface RefinementResult<P> {
  readonly status: 'complete' | 'budget'
  /** The top-k patterns, best first. */
  readonly top: readonly ScoredPattern<P>[]
  /** Patterns refined. */
  readonly expanded: number
  /** Patterns whose quality was computed. */
  readonly evaluated: number
  /** Patterns not refined because their optimistic estimate could not beat the result set. */
  readonly pruned: number
}

/** The state of a traced refinement search. */
export interface RefinementState<P> {
  readonly top: readonly ScoredPattern<P>[]
  readonly threshold: number
  readonly expanded: number
  readonly evaluated: number
  readonly pruned: number
  readonly frontier: number
  readonly done: boolean
  /** The pattern refined in the last step. */
  readonly current: string | null
}

const REFINEMENT_SCHEMA: TraceSchema = channels({
  metrics: {
    threshold: { description: 'the quality to beat (k-th best)', direction: 'higher' },
    best: { description: 'best quality so far', direction: 'higher' },
    expanded: { description: 'patterns refined' },
    pruned: { description: 'patterns pruned by optimistic estimates' },
    frontier: { description: 'patterns waiting' },
  },
  messages: { refine: { description: 'the pattern refined', payload: { pattern: 'label', depth: 'depth' } } },
})

interface Entry<P> {
  readonly pattern: P
  readonly depth: number
  readonly bound: number
  readonly quality: number
  readonly seq: number
}

interface Internal<P> extends RefinementState<P> {
  top: readonly ScoredPattern<P>[]
  threshold: number
  expanded: number
  evaluated: number
  pruned: number
  frontier: number
  done: boolean
  current: string | null
  readonly results: TopK<P>
  readonly seen: Set<StateKey>
  readonly stack: Entry<P>[]
  readonly heap: BinaryHeap<Entry<P>>
  beam: Entry<P>[]
  seq: number
}

type Strategy = 'exhaustive' | 'best-first' | 'beam'

function refinementEngine<P>(
  strategy: Strategy,
  problem: RefinementProblem<P>,
  options: RefinementOptions & { readonly width?: number },
): Algorithm<Internal<P>> {
  const k = options.k ?? 10
  const maxDepth = options.maxDepth ?? 3
  const minQuality = options.minQuality ?? Number.NEGATIVE_INFINITY
  const width = options.width ?? 10
  if (strategy === 'beam' && !(Number.isInteger(width) && width >= 1)) {
    throw new MdmlError('value', `refinementBeamSearch: width ${width} must be a positive integer`)
  }
  const key = stateKeyOf(problem)
  const label = (p: P): string => problem.label?.(p) ?? String(key(p))
  const valid = (p: P): boolean => problem.isValid?.(p) ?? true
  const bound = (p: P, q: number): number =>
    problem.optimisticEstimate?.(p) ?? (strategy === 'best-first' ? q : Number.POSITIVE_INFINITY)
  /** Evaluate and offer a pattern; return its entry for further refinement. */
  const visit = (s: Internal<P>, pattern: P, depth: number): Entry<P> => {
    const quality = problem.quality(pattern)
    s.evaluated += 1
    if (quality >= minQuality && (depth > 0 || options.includeRoot === true)) {
      const coverage = problem.coverage?.(pattern)
      s.results.add({ pattern, quality, depth, ...(coverage === undefined ? {} : { coverage }) })
    }
    return { pattern, depth, quality, bound: bound(pattern, quality), seq: s.seq++ }
  }
  const children = (s: Internal<P>, e: Entry<P>): P[] => {
    s.expanded += 1
    s.current = label(e.pattern)
    const out: P[] = []
    for (const c of problem.refine(e.pattern)) {
      const kc = key(c)
      if (s.seen.has(kc) || !valid(c)) continue
      s.seen.add(kc)
      out.push(c)
    }
    return out
  }
  const promising = (s: Internal<P>, e: Entry<P>): boolean => {
    if (e.depth >= maxDepth) return false
    if (problem.optimisticEstimate !== undefined && e.bound <= Math.max(s.results.threshold, minQuality)) {
      s.pruned += 1
      return false
    }
    return true
  }
  const sync = (s: Internal<P>): void => {
    s.top = s.results.items
    s.threshold = s.results.threshold
    s.frontier = strategy === 'exhaustive' ? s.stack.length : strategy === 'best-first' ? s.heap.size : s.beam.length
  }
  return {
    name: `refinement-${strategy}`,
    config: { strategy, k, maxDepth, width } as Json,
    schema: REFINEMENT_SCHEMA,
    init: () => {
      const s: Internal<P> = {
        top: [],
        threshold: Number.NEGATIVE_INFINITY,
        expanded: 0,
        evaluated: 0,
        pruned: 0,
        frontier: 0,
        done: false,
        current: null,
        results: new TopK<P>(k, options.maxSimilarity === undefined ? {} : { maxSimilarity: options.maxSimilarity }),
        seen: new Set([key(problem.root)]),
        stack: [],
        heap: new BinaryHeap<Entry<P>>((a, b) => b.bound - a.bound || a.seq - b.seq),
        beam: [],
        seq: 0,
      }
      const root = visit(s, problem.root, 0)
      if (strategy === 'exhaustive') s.stack.push(root)
      else if (strategy === 'best-first') s.heap.push(root)
      else s.beam = [root]
      sync(s)
      return s
    },
    step: (s) => {
      s.current = null
      if (strategy === 'exhaustive') {
        // Depth-first in refinement order, pruning subtrees whose optimistic estimate cannot enter the result set.
        const e = s.stack.pop()
        if (e === undefined) s.done = true
        else if (promising(s, e)) {
          const next = children(s, e).map((c) => visit(s, c, e.depth + 1))
          for (let i = next.length - 1; i >= 0; i--) s.stack.push(next[i] as Entry<P>)
        }
      } else if (strategy === 'best-first') {
        // The most promising pattern first; once the best bound cannot beat the result set, the search is complete.
        const e = s.heap.pop()
        if (e === undefined) s.done = true
        else if (problem.optimisticEstimate !== undefined && e.bound <= Math.max(s.results.threshold, minQuality)) {
          s.pruned += 1 + s.heap.size
          s.done = true
        } else if (promising(s, e)) for (const c of children(s, e)) s.heap.push(visit(s, c, e.depth + 1))
      } else {
        // One level: refine the whole beam, keep the best `width` refinements by quality.
        const level: Entry<P>[] = []
        for (const e of s.beam) {
          if (!promising(s, e)) continue
          for (const c of children(s, e)) level.push(visit(s, c, e.depth + 1))
        }
        s.beam = level.sort((a, b) => b.quality - a.quality || a.seq - b.seq).slice(0, width)
        if (s.beam.length === 0) s.done = true
      }
      sync(s)
      return s
    },
    done: (s) => (s.done ? 'converged' : s.expanded >= (options.maxExpansions ?? 100_000) ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: {
        threshold: s.threshold,
        best: s.top[0]?.quality ?? Number.NEGATIVE_INFINITY,
        expanded: s.expanded,
        pruned: s.pruned,
        frontier: s.frontier,
      },
      messages: s.current === null ? [] : [{ kind: 'refine', payload: { pattern: s.current } }],
    }),
  }
}

function runRefinement<P>(algorithm: Algorithm<Internal<P>>, options: RefinementOptions): RefinementResult<P> {
  const {
    k: _k,
    maxDepth: _d,
    minQuality: _m,
    maxSimilarity: _s,
    includeRoot: _r,
    maxExpansions = 100_000,
    ...run
  } = options
  const { width: _w, ...rest } = run as typeof run & { readonly width?: number }
  const { state } = runAlgorithm(algorithm, { ...rest, maxSteps: maxExpansions + 1 })
  return {
    status: state.done ? 'complete' : 'budget',
    top: state.top,
    expanded: state.expanded,
    evaluated: state.evaluated,
    pruned: state.pruned,
  }
}

/**
 * Exhaustive refinement search as a traced algorithm: depth-first through the refinement tree to `maxDepth`, keeping
 * the top k, and skipping the refinements of any pattern whose optimistic estimate cannot beat the k-th best (the
 * pruning of SD-Map and pysubgroup's DFS). With an admissible estimate the result equals a search without pruning.
 *
 * @param problem - The refinement problem.
 * @param options - k, depth, a quality floor, redundancy filtering and the expansion budget.
 * @returns The algorithm.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementExhaustiveAlgorithm } from 'mdml-engine/search'
 * refinementExhaustiveAlgorithm({ root: '', refine: () => [], quality: () => 0 }).name // => 'refinement-exhaustive'
 * ```
 */
export function refinementExhaustiveAlgorithm<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions = {},
): Algorithm<RefinementState<P>> {
  return refinementEngine('exhaustive', problem, options)
}

/**
 * Exhaustive refinement search with optimistic-estimate pruning.
 *
 * @param problem - The refinement problem.
 * @param options - k (default 10), maxDepth (default 3), a quality floor, redundancy filtering, budget and tracing.
 * @returns The top-k patterns and the effort.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementExhaustive } from 'mdml-engine/search'
 * // Patterns are sets of letters; quality rewards 'a' and 'c' and penalises length.
 * const letters = ['a', 'b', 'c']
 * const p = {
 *   root: '',
 *   refine: (s: string) => letters.filter((l) => l > (s.at(-1) ?? '')).map((l) => s + l),
 *   quality: (s: string) => (s.includes('a') ? 2 : 0) + (s.includes('c') ? 1 : 0) - 0.1 * s.length,
 * }
 * refinementExhaustive(p, { k: 2 }).top.map((t) => t.pattern) // => ['ac', 'abc']
 * ```
 */
export function refinementExhaustive<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions = {},
): RefinementResult<P> {
  return runRefinement(refinementEngine('exhaustive', problem, options), options)
}

/**
 * Best-first refinement search as a traced algorithm: refine the pattern with the highest optimistic estimate (or
 * quality) first, and stop once no waiting pattern's estimate can beat the k-th best, so with an admissible estimate
 * the top k are exact, usually after fewer expansions than depth-first search.
 *
 * @param problem - The refinement problem.
 * @param options - k, depth, a quality floor, redundancy filtering and the expansion budget.
 * @returns The algorithm.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementBestFirstAlgorithm } from 'mdml-engine/search'
 * refinementBestFirstAlgorithm({ root: '', refine: () => [], quality: () => 0 }).name // => 'refinement-best-first'
 * ```
 */
export function refinementBestFirstAlgorithm<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions = {},
): Algorithm<RefinementState<P>> {
  return refinementEngine('best-first', problem, options)
}

/**
 * Best-first refinement search.
 *
 * @param problem - The refinement problem.
 * @param options - k (default 10), maxDepth (default 3), a quality floor, redundancy filtering, budget and tracing.
 * @returns The top-k patterns and the effort.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementBestFirst } from 'mdml-engine/search'
 * const letters = ['a', 'b', 'c']
 * const p = {
 *   root: '',
 *   refine: (s: string) => letters.filter((l) => l > (s.at(-1) ?? '')).map((l) => s + l),
 *   quality: (s: string) => (s.includes('a') ? 2 : 0) + (s.includes('c') ? 1 : 0) - 0.1 * s.length,
 *   optimisticEstimate: (s: string) => (s === '' || s.includes('a') ? 2 : 0) + 1,
 * }
 * refinementBestFirst(p, { k: 1 }).top[0]?.pattern // => 'ac'
 * ```
 */
export function refinementBestFirst<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions = {},
): RefinementResult<P> {
  return runRefinement(refinementEngine('best-first', problem, options), options)
}

/**
 * Beam refinement search as a traced algorithm (the beam search of CN2 and DSSD): level by level, refine every
 * pattern of the beam and keep the `width` best refinements; all evaluated patterns are offered to the top k. Fast,
 * not exhaustive.
 *
 * @param problem - The refinement problem.
 * @param options - The beam width (default 10), k, depth, a quality floor, redundancy filtering and the budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when the width is not a positive integer.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementBeamSearchAlgorithm } from 'mdml-engine/search'
 * refinementBeamSearchAlgorithm({ root: '', refine: () => [], quality: () => 0 }).name // => 'refinement-beam'
 * ```
 */
export function refinementBeamSearchAlgorithm<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions & { readonly width?: number } = {},
): Algorithm<RefinementState<P>> {
  return refinementEngine('beam', problem, options)
}

/**
 * Beam refinement search.
 *
 * @param problem - The refinement problem.
 * @param options - The beam width (default 10), k (default 10), maxDepth (default 3), a quality floor, redundancy
 * filtering, budget and tracing.
 * @returns The top-k patterns and the effort.
 * @throws MdmlError with code `value` when the width is not a positive integer.
 * @concept refinement-search
 * @example
 * ```ts
 * import { refinementBeamSearch } from 'mdml-engine/search'
 * const letters = ['a', 'b', 'c']
 * const p = {
 *   root: '',
 *   refine: (s: string) => letters.filter((l) => l > (s.at(-1) ?? '')).map((l) => s + l),
 *   quality: (s: string) => (s.includes('a') ? 2 : 0) + (s.includes('c') ? 1 : 0) - 0.1 * s.length,
 * }
 * refinementBeamSearch(p, { width: 1, k: 1 }).top[0]?.pattern // => 'ac'
 * ```
 */
export function refinementBeamSearch<P>(
  problem: RefinementProblem<P>,
  options: RefinementOptions & { readonly width?: number } = {},
): RefinementResult<P> {
  return runRefinement(refinementEngine('beam', problem, options), options)
}

/** A selector of a conjunctive pattern language: a named test, its attribute, and the records it covers (0/1). */
export interface Selector {
  readonly name: string
  /** Selectors on the same attribute are not combined (default: each selector its own attribute). */
  readonly attribute?: string
  readonly covers: ArrayLike<number>
}

/**
 * The refinement problem of conjunctions of selectors (the language of subgroup discovery and propositional rule
 * learning): a pattern is a sorted list of selector indices, refined by appending a later selector on a new
 * attribute, so each conjunction is generated once. The quality and its optimistic estimate see the pattern's
 * coverage; a minimum support prunes rare patterns.
 *
 * @param selectors - The selectors (all with coverages over the same records).
 * @param quality - The quality of a pattern from its coverage.
 * @param options - An optimistic estimate from the coverage and a minimum support (default 1 record).
 * @returns The refinement problem; patterns are labelled `s1 & s2`.
 * @throws MdmlError with code `shape` when coverages differ in length.
 * @concept refinement-search
 * @example
 * ```ts
 * import { conjunctions, refinementExhaustive } from 'mdml-engine/search'
 * // Four records with a binary target; find the conjunction that best isolates it.
 * const target = [1, 1, 0, 0]
 * const selectors = [
 *   { name: 'x=1', attribute: 'x', covers: [1, 1, 1, 0] },
 *   { name: 'y=1', attribute: 'y', covers: [1, 1, 0, 1] },
 *   { name: 'z=1', attribute: 'z', covers: [0, 1, 1, 1] },
 * ]
 * const precision = (c: ArrayLike<number>) => Array.from(c).reduce((s, v, i) => s + v * (target[i] ?? 0), 0) / Math.max(1, Array.from(c).reduce((s, v) => s + v, 0))
 * const p = conjunctions(selectors, precision)
 * p.label?.(refinementExhaustive(p, { k: 1 }).top[0]?.pattern ?? []) // => 'x=1 & y=1'
 * ```
 */
export function conjunctions(
  selectors: readonly Selector[],
  quality: (coverage: Uint8Array, pattern: readonly number[]) => number,
  options: {
    readonly optimisticEstimate?: (coverage: Uint8Array, pattern: readonly number[]) => number
    readonly minSupport?: number
  } = {},
): RefinementProblem<readonly number[]> {
  const n = selectors[0]?.covers.length ?? 0
  if (selectors.some((s) => s.covers.length !== n))
    throw new MdmlError('shape', 'conjunctions: coverages differ in length')
  const minSupport = options.minSupport ?? 1
  const cache = new Map<string, Uint8Array>()
  const coverage = (pattern: readonly number[]): Uint8Array => {
    const id = pattern.join(',')
    const hit = cache.get(id)
    if (hit !== undefined) return hit
    const out = new Uint8Array(n).fill(1)
    for (const j of pattern) {
      const c = selectors[j]?.covers
      for (let i = 0; i < n; i++) if ((c?.[i] ?? 0) === 0) out[i] = 0
    }
    if (cache.size < 100_000) cache.set(id, out)
    return out
  }
  const attribute = (j: number): string => selectors[j]?.attribute ?? `#${j}`
  const estimate = options.optimisticEstimate
  return {
    root: [],
    refine: (pattern) => {
      const used = new Set(pattern.map(attribute))
      const last = pattern[pattern.length - 1] ?? -1
      const out: number[][] = []
      for (let j = last + 1; j < selectors.length; j++) if (!used.has(attribute(j))) out.push([...pattern, j])
      return out
    },
    quality: (pattern) => quality(coverage(pattern), pattern),
    ...(estimate === undefined
      ? {}
      : { optimisticEstimate: (pattern: readonly number[]) => estimate(coverage(pattern), pattern) }),
    isValid: (pattern) => coverage(pattern).reduce((s, v) => s + v, 0) >= minSupport,
    coverage,
    key: (pattern) => pattern.join(','),
    label: (pattern) => (pattern.length === 0 ? 'true' : pattern.map((j) => selectors[j]?.name ?? '?').join(' & ')),
  }
}
