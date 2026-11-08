/**
 * Sequential pattern mining over sequences of itemsets: PrefixSpan (depth-first growth of patterns by projected
 * databases) and GSP (level-wise candidate generation with the apriori prune, with minimum and maximum gap and
 * maximum span constraints), each as a traced algorithm, and an exhaustive enumeration the two are checked against.
 * A sequence is a list of itemsets (events, in order); a pattern is contained in a sequence when its itemsets are
 * subsets of distinct, ordered events of it.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'

/** A sequence of events, each event a set of item ids (sorted ascending). */
export type Sequence = readonly (readonly number[])[]

/** A sequential pattern and the number of sequences that contain it. */
export interface SequentialPattern {
  readonly pattern: Sequence
  readonly count: number
}

/** Gap and span constraints on where a pattern's events may sit in a sequence (counted in events). */
export interface GapConstraints {
  /** The least distance between the events matching consecutive pattern events (default 1). */
  readonly minGap?: number
  /** The greatest such distance (default unlimited). */
  readonly maxGap?: number
  /** The greatest distance from the first matched event to the last (default unlimited). */
  readonly maxSpan?: number
}

/** Options of the sequence miners. */
export interface SequenceOptions {
  /** The smallest support as a fraction of the sequences, in (0, 1]. */
  readonly minSupport: number
  /** The most items in a pattern (default no limit). */
  readonly maxLength?: number
}

/** Tracing options. */
export type SequenceRunOptions = SequenceOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>>

function neededCount(n: number, minSupport: number): number {
  if (!(minSupport > 0 && minSupport <= 1)) throw new OmaiError('value', `minSupport = ${minSupport} must be in (0, 1]`)
  return Math.max(1, Math.ceil(minSupport * n - 1e-9))
}

function normalise(sequences: readonly Sequence[]): number[][][] {
  return sequences.map((s) => s.map((e) => [...new Set(e)].sort((a, b) => a - b)))
}

const isSubset = (a: readonly number[], b: readonly number[]): boolean => {
  let j = 0
  for (const x of a) {
    while (j < b.length && (b[j] as number) < x) j++
    if (j >= b.length || b[j] !== x) return false
    j++
  }
  return true
}

/**
 * The text of a pattern: events in parentheses, `(0 1)(2)`.
 *
 * @param pattern - The pattern.
 * @returns The text.
 * @example
 * ```ts
 * import { patternText } from 'omai-engine/patterns'
 * patternText([[0, 1], [2]]) // => '(0 1)(2)'
 * ```
 */
export function patternText(pattern: Sequence): string {
  return pattern.map((e) => `(${e.join(' ')})`).join('')
}

/**
 * Whether a sequence contains a pattern, under gap and span constraints. Without constraints this is the usual
 * subsequence test (greedy earliest match); with them it searches the placements of each pattern event.
 *
 * @param sequence - The sequence.
 * @param pattern - The pattern.
 * @param constraints - Minimum and maximum gap between consecutive matched events and maximum span.
 * @returns True when the pattern occurs.
 * @example
 * ```ts
 * import { containsPattern } from 'omai-engine/patterns'
 * const s = [[1], [2, 3], [4], [1, 5]]
 * [containsPattern(s, [[1], [4]]), containsPattern(s, [[1], [4]], { maxGap: 1 }), containsPattern(s, [[2], [5]], { maxGap: 2 })] // => [true, false, true]
 * ```
 */
export function containsPattern(sequence: Sequence, pattern: Sequence, constraints: GapConstraints = {}): boolean {
  const minGap = Math.max(1, constraints.minGap ?? 1)
  const maxGap = constraints.maxGap ?? Number.POSITIVE_INFINITY
  const maxSpan = constraints.maxSpan ?? Number.POSITIVE_INFINITY
  if (pattern.length === 0) return true
  // Place pattern event p+1 after the event matched at `pos` (the first match was at `first`).
  const go = (p: number, pos: number, first: number): boolean => {
    if (p === pattern.length - 1) return true
    const hi = Math.min(sequence.length - 1, pos + maxGap, first + maxSpan)
    for (let q = pos + minGap; q <= hi; q++) {
      if (isSubset(pattern[p + 1] as readonly number[], sequence[q] as readonly number[]) && go(p + 1, q, first)) {
        return true
      }
    }
    return false
  }
  for (let pos = 0; pos < sequence.length; pos++) {
    if (isSubset(pattern[0] as readonly number[], sequence[pos] as readonly number[]) && go(0, pos, pos)) return true
  }
  return false
}

/**
 * The number of sequences that contain a pattern.
 *
 * @param sequences - The sequences.
 * @param pattern - The pattern.
 * @param constraints - Gap and span constraints.
 * @returns The count.
 * @example
 * ```ts
 * import { sequenceSupport } from 'omai-engine/patterns'
 * sequenceSupport([[[1], [2]], [[2], [1]], [[1, 2]]], [[1], [2]]) // => 1
 * ```
 */
export function sequenceSupport(
  sequences: readonly Sequence[],
  pattern: Sequence,
  constraints: GapConstraints = {},
): number {
  let c = 0
  for (const s of sequences) if (containsPattern(s, pattern, constraints)) c++
  return c
}

const patternLength = (p: Sequence): number => p.reduce((s, e) => s + e.length, 0)

const patternKey = (p: Sequence): string => p.map((e) => e.join(' ')).join('|')

function byPattern(a: SequentialPattern, b: SequentialPattern): number {
  const la = patternLength(a.pattern)
  const lb = patternLength(b.pattern)
  if (la !== lb) return la - lb
  return patternKey(a.pattern) < patternKey(b.pattern) ? -1 : patternKey(a.pattern) > patternKey(b.pattern) ? 1 : 0
}

/**
 * Patterns in canonical order: by number of items, then by their text.
 *
 * @param patterns - Patterns in any order.
 * @returns A sorted copy.
 * @example
 * ```ts
 * import { canonicalPatterns } from 'omai-engine/patterns'
 * canonicalPatterns([{ pattern: [[1], [2]], count: 1 }, { pattern: [[3]], count: 2 }]).map((p) => p.count) // => [2, 1]
 * ```
 */
export function canonicalPatterns(patterns: readonly SequentialPattern[]): SequentialPattern[] {
  return [...patterns].sort(byPattern)
}

/**
 * Every frequent sequential pattern by enumerating every subsequence (itemset subsequences) of every sequence and
 * counting each distinct one: exponential, the reference the miners are checked against.
 *
 * @param sequences - The sequences.
 * @param options - The support threshold and the largest pattern.
 * @param constraints - Gap and span constraints.
 * @returns The frequent patterns in canonical order.
 * @example
 * ```ts
 * import { exhaustivePatterns } from 'omai-engine/patterns'
 * exhaustivePatterns([[[1], [2]], [[1], [2]], [[2]]], { minSupport: 0.6 }).map((p) => p.count) // => [2, 3, 2]
 * ```
 */
export function exhaustivePatterns(
  sequences: readonly Sequence[],
  options: SequenceOptions,
  constraints: GapConstraints = {},
): SequentialPattern[] {
  const data = normalise(sequences)
  const need = neededCount(data.length, options.minSupport)
  const max = options.maxLength ?? Number.POSITIVE_INFINITY
  const candidates = new Map<string, number[][]>()
  for (const s of data) {
    // All subsequences: choose a non-empty subset of items from a subset of events.
    const rec = (e: number, current: number[][], length: number): void => {
      if (e === s.length) {
        if (current.length > 0)
          candidates.set(
            patternKey(current),
            current.map((x) => [...x]),
          )
        return
      }
      rec(e + 1, current, length)
      const ev = s[e] as number[]
      for (let mask = 1; mask < 1 << ev.length; mask++) {
        const part = ev.filter((_, b) => (mask >> b) & 1)
        if (length + part.length <= max) rec(e + 1, [...current, part], length + part.length)
      }
    }
    rec(0, [], 0)
  }
  const out: SequentialPattern[] = []
  for (const p of candidates.values()) {
    const count = sequenceSupport(data, p, constraints)
    if (count >= need) out.push({ pattern: p, count })
  }
  return canonicalPatterns(out)
}

// ------------------------------------------------------------------------------------------------ PrefixSpan

/** A sequence in a projected database: its index, and where the pattern's last two events sit in it (earliest match). */
type Projection = readonly [sequence: number, previous: number, last: number]

interface PrefixNode {
  readonly pattern: Sequence
  readonly projected: readonly Projection[]
}

/** The state of {@link prefixSpanAlgorithm}. */
export interface PrefixSpanState {
  readonly stack: readonly PrefixNode[]
  readonly found: readonly SequentialPattern[]
  /** The pattern extended in the last step and the size of its projected database. */
  readonly current: Sequence
  readonly projectedSize: number
  readonly started: boolean
}

const PS_SCHEMA: TraceSchema = channels({
  metrics: {
    projected: { description: 'sequences in the projected database of the pattern just extended' },
    found: { description: 'frequent patterns found so far' },
    pending: { description: 'patterns waiting to be extended' },
  },
  messages: {
    extend: { description: 'a pattern extended', payload: { pattern: 'the pattern', projected: 'database size' } },
  },
})

/**
 * PrefixSpan (Pei et al. 2001) as a traced algorithm, one prefix extension per step. A pattern's projected database
 * keeps, for each sequence containing it, where its earliest match ends; one step counts, over that database, the items
 * that can extend the pattern by a new event after it (sequence extension) or by joining its last event (itemset
 * extension), keeps the frequent ones as new patterns and queues them with their own projections.
 *
 * @param sequences - The sequences.
 * @param options - The support threshold and the largest pattern.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { prefixSpanAlgorithm } from 'omai-engine/patterns'
 * prefixSpanAlgorithm([[[1], [2]]], { minSupport: 1 }).name // => 'prefixspan'
 * ```
 */
export function prefixSpanAlgorithm(
  sequences: readonly Sequence[],
  options: SequenceOptions,
): Algorithm<PrefixSpanState> {
  const data = normalise(sequences)
  const need = neededCount(data.length, options.minSupport)
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  return {
    name: 'prefixspan',
    config: {
      minSupport: options.minSupport,
      minCount: need,
      ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    },
    schema: PS_SCHEMA,
    init: () => ({ stack: [], found: [], current: [], projectedSize: 0, started: false }),
    step: (s) => {
      if (!s.started) {
        const root: PrefixNode = { pattern: [], projected: data.map((_, i) => [i, -1, -1] as const) }
        return { ...s, started: true, stack: [root] }
      }
      const stack = [...s.stack]
      const node = stack.pop() as PrefixNode
      const found = [...s.found]
      const length = patternLength(node.pattern)
      if (length < maxLength) {
        const last = node.pattern[node.pattern.length - 1]
        const lastMax = last === undefined ? -1 : (last[last.length - 1] as number)
        // Sequence extensions: items in events after the earliest match; itemset extensions: items larger than the last item, in events from the match on that contain the last event.
        const seqSupport = new Map<number, Projection[]>()
        const setSupport = new Map<number, Projection[]>()
        for (const [si, prev, end] of node.projected) {
          const seq = data[si] as number[][]
          const seenSeq = new Set<number>()
          for (let e = end + 1; e < seq.length; e++) {
            for (const it of seq[e] as number[]) {
              if (seenSeq.has(it)) continue
              seenSeq.add(it)
              const list = seqSupport.get(it)
              const proj: Projection = [si, end, e]
              if (list === undefined) seqSupport.set(it, [proj])
              else list.push(proj)
            }
          }
          if (last !== undefined) {
            const seenSet = new Set<number>()
            for (let e = Math.max(end, 0); e < seq.length; e++) {
              if (!isSubset(last, seq[e] as number[])) continue
              for (const it of seq[e] as number[]) {
                if (it <= lastMax || seenSet.has(it)) continue
                seenSet.add(it)
                const list = setSupport.get(it)
                const proj: Projection = [si, prev, e]
                if (list === undefined) setSupport.set(it, [proj])
                else list.push(proj)
              }
            }
          }
        }
        const add = (map: Map<number, Projection[]>, make: (it: number) => number[][]): void => {
          for (const [it, projected] of [...map.entries()].sort((a, b) => b[0] - a[0])) {
            if (projected.length < need) continue
            const pattern = make(it)
            found.push({ pattern, count: projected.length })
            stack.push({ pattern, projected })
          }
        }
        add(seqSupport, (it) => [...node.pattern.map((e) => [...e]), [it]])
        add(setSupport, (it) => [...node.pattern.slice(0, -1).map((e) => [...e]), [...(last as number[]), it]])
      }
      return { stack, found, current: node.pattern, projectedSize: node.projected.length, started: true }
    },
    done: (s) => (s.started && s.stack.length === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { projected: s.projectedSize, found: s.found.length, pending: s.stack.length },
      messages: s.started
        ? [{ kind: 'extend', payload: { pattern: s.current.map((e) => [...e]), projected: s.projectedSize } }]
        : [],
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as PrefixSpanState,
  }
}

/**
 * Frequent sequential patterns by PrefixSpan.
 *
 * @param sequences - The sequences.
 * @param options - The support threshold, the largest pattern and tracing.
 * @returns The frequent patterns in canonical order.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { patternText, prefixSpan } from 'omai-engine/patterns'
 * const data = [[[1], [2, 3]], [[1, 2], [3]], [[1], [3]]]
 * prefixSpan(data, { minSupport: 1 }).map((p) => patternText(p.pattern)) // => ['(1)', '(3)', '(1)(3)']
 * ```
 */
export function prefixSpan(sequences: readonly Sequence[], options: SequenceRunOptions): SequentialPattern[] {
  const { minSupport, maxLength, ...run } = options
  const alg = prefixSpanAlgorithm(sequences, { minSupport, ...(maxLength === undefined ? {} : { maxLength }) })
  const { state } = runAlgorithm(alg, { ...run, maxSteps: 10_000_000 })
  return canonicalPatterns(state.found)
}

// ------------------------------------------------------------------------------------------------ GSP

/** The state of {@link gspAlgorithm}: the level (pattern length in items) just counted. */
export interface GspState {
  readonly k: number
  readonly level: readonly SequentialPattern[]
  readonly found: readonly SequentialPattern[]
  readonly joined: number
  readonly pruned: number
  readonly candidates: number
  readonly done: boolean
}

const GSP_SCHEMA: TraceSchema = channels({
  metrics: {
    k: { description: 'the number of items in the patterns counted at this level' },
    candidates: { description: 'candidates counted after the join and the prune' },
    frequent: { description: 'frequent patterns at this level' },
    pruned: { description: 'candidates removed because a contiguous subsequence was infrequent' },
    total: { description: 'frequent patterns so far' },
  },
})

/** The pattern without its first item (the first event shrinks, or vanishes when it had one item). */
function dropFirst(p: Sequence): number[][] {
  const [head, ...rest] = p
  const h = (head as readonly number[]).slice(1)
  return [...(h.length > 0 ? [[...h]] : []), ...rest.map((e) => [...e])]
}

/** The pattern without its last item. */
function dropLast(p: Sequence): number[][] {
  const init = p.slice(0, -1).map((e) => [...e])
  const l = (p[p.length - 1] as readonly number[]).slice(0, -1)
  return [...init, ...(l.length > 0 ? [[...l]] : [])]
}

/** The patterns obtained by deleting one item from an event with a neighbour or from a larger event (contiguous subsequences). */
function contiguousSubpatterns(p: Sequence): number[][][] {
  const out: number[][][] = []
  p.forEach((e, ei) => {
    // An item may be dropped when its event has at least two items, or is the first or last event.
    if (e.length > 1 || ei === 0 || ei === p.length - 1) {
      e.forEach((_, ii) => {
        const q = p.map((x, xi) => (xi === ei ? x.filter((__, j) => j !== ii) : [...x])).filter((x) => x.length > 0)
        out.push(q)
      })
    }
  })
  return out
}

function gspJoin(level: readonly SequentialPattern[]): number[][][] {
  const byDropLast = new Map<string, SequentialPattern[]>()
  for (const s of level) {
    const key = patternKey(dropLast(s.pattern))
    const list = byDropLast.get(key)
    if (list === undefined) byDropLast.set(key, [s])
    else list.push(s)
  }
  const out = new Map<string, number[][]>()
  for (const a of level) {
    const key = patternKey(dropFirst(a.pattern))
    for (const b of byDropLast.get(key) ?? []) {
      const lastEvent = b.pattern[b.pattern.length - 1] as readonly number[]
      const item = lastEvent[lastEvent.length - 1] as number
      const candidate: number[][] =
        lastEvent.length === 1
          ? [...a.pattern.map((e) => [...e]), [item]]
          : [...a.pattern.slice(0, -1).map((e) => [...e]), [...(a.pattern[a.pattern.length - 1] as number[]), item]]
      out.set(patternKey(candidate), candidate)
    }
  }
  return [...out.values()]
}

/**
 * GSP (Srikant and Agrawal 1996) as a traced algorithm, one level per step: the frequent items first, then candidates
 * one item longer from joining patterns whose first-item-dropped and last-item-dropped forms agree, pruned when a
 * contiguous subsequence is not frequent, counted against the sequences under the gap and span constraints. Without
 * constraints it finds exactly the patterns PrefixSpan does.
 *
 * @param sequences - The sequences.
 * @param options - The support threshold and the largest pattern.
 * @param constraints - Minimum and maximum gap and maximum span, in events.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { gspAlgorithm } from 'omai-engine/patterns'
 * gspAlgorithm([[[1], [2]]], { minSupport: 1 }).name // => 'gsp'
 * ```
 */
export function gspAlgorithm(
  sequences: readonly Sequence[],
  options: SequenceOptions,
  constraints: GapConstraints = {},
): Algorithm<GspState> {
  const data = normalise(sequences)
  const need = neededCount(data.length, options.minSupport)
  const maxLength = options.maxLength ?? Number.POSITIVE_INFINITY
  return {
    name: 'gsp',
    config: {
      minSupport: options.minSupport,
      minCount: need,
      constraints: { ...constraints },
      ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    },
    schema: GSP_SCHEMA,
    init: () => ({ k: 0, level: [], found: [], joined: 0, pruned: 0, candidates: 0, done: false }),
    step: (s) => {
      const k = s.k + 1
      let candidates: number[][][]
      let joined: number
      if (k === 1) {
        const items = [...new Set(data.flat(2))].sort((a, b) => a - b)
        candidates = items.map((i) => [[i]])
        joined = candidates.length
      } else if (k === 2) {
        const items = s.level.map((p) => (p.pattern[0] as number[])[0] as number)
        const all: number[][][] = []
        for (const x of items) for (const y of items) all.push([[x], [y]])
        for (const x of items) for (const y of items) if (x < y) all.push([[x, y]])
        candidates = all
        joined = all.length
      } else {
        const joinedList = gspJoin(s.level)
        joined = joinedList.length
        const known = new Set(s.level.map((p) => patternKey(p.pattern)))
        candidates = joinedList.filter((c) => contiguousSubpatterns(c).every((q) => known.has(patternKey(q))))
      }
      const level: SequentialPattern[] = []
      for (const c of candidates) {
        const count = sequenceSupport(data, c, constraints)
        if (count >= need) level.push({ pattern: c, count })
      }
      return {
        k,
        level,
        found: [...s.found, ...level],
        joined,
        pruned: joined - candidates.length,
        candidates: candidates.length,
        done: level.length === 0 || k >= maxLength,
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { k: s.k, candidates: s.candidates, frequent: s.level.length, pruned: s.pruned, total: s.found.length },
    }),
    serialise: (s) => s as unknown as Json,
    deserialise: (j) => j as unknown as GspState,
  }
}

/**
 * Frequent sequential patterns by GSP.
 *
 * @param sequences - The sequences.
 * @param options - The support threshold, the largest pattern and tracing.
 * @param constraints - Minimum and maximum gap and maximum span, in events.
 * @returns The frequent patterns in canonical order.
 * @throws OmaiError with code `value` for a support outside (0, 1].
 * @example
 * ```ts
 * import { gsp, patternText } from 'omai-engine/patterns'
 * const data = [[[1], [2], [3]], [[1], [2], [3]], [[1], [3]]]
 * gsp(data, { minSupport: 0.6 }, { maxGap: 1 }).map((p) => patternText(p.pattern)) // => ['(1)', '(2)', '(3)', '(1)(2)', '(2)(3)', '(1)(2)(3)']
 * ```
 */
export function gsp(
  sequences: readonly Sequence[],
  options: SequenceRunOptions,
  constraints: GapConstraints = {},
): SequentialPattern[] {
  const { minSupport, maxLength, ...run } = options
  const alg = gspAlgorithm(sequences, { minSupport, ...(maxLength === undefined ? {} : { maxLength }) }, constraints)
  const { state } = runAlgorithm(alg, { ...run, maxSteps: 100_000 })
  return canonicalPatterns(state.found)
}
