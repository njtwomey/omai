/**
 * The dynamic-programming tables of the edit-distance family as one stepped algorithm: the weighted Levenshtein table,
 * the longest-common-subsequence table, the unit-cost table behind the edit counts of error rates (hits, substitutions,
 * deletions, insertions) and the Sellers table of approximate substring matching (a free start in the text). Each step
 * fills one row of the table; one more step traces back through it. The kind is a slot (`dpKinds`), the state is the
 * table and serialises.
 */

import { type Json, OmaiError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { elements, type Sequence } from './_sequence.ts'

/** Costs of the edit operations of the weighted Levenshtein table (rapidfuzz's `weights`). */
export interface EditWeights {
  readonly insertion?: number
  readonly deletion?: number
  readonly substitution?: number
}

/** The tables {@link dpAlgorithm} can fill. */
export type DpKind = 'levenshtein' | 'lcs' | 'edit-counts' | 'sellers'

/** The paths {@link dpAlgorithm} logs. */
export type DpPath = 'dp/row' | 'dp/cell' | 'dp/path'

/** What the table fills log: the row just filled, the cost in its last cell, and the traceback path at the end. */
export const dpContract: Contract<unknown, DpPath> = defineContract({
  'dp/row': tensor({ role: 'state', shape: ['columns'], description: 'the row of the table just filled' }),
  'dp/cell': scalar({ description: 'the last cell of the row just filled (the distance or length so far)' }),
  'dp/path': tensor({
    role: 'output',
    shape: ['pairs'],
    description: 'the traceback as flat (row, column) pairs, from the corner',
  }),
})

/** What {@link dpAlgorithm} computes. */
export interface DpOptions {
  /** The table (default `levenshtein`). */
  readonly kind?: DpKind
  /** Operation costs of `levenshtein` (default 1 each). */
  readonly weights?: EditWeights
  /** The largest edit distance reported by `sellers` (default 0). */
  readonly k?: number
}

/** Hits, substitutions, deletions and insertions of an alignment of a reference with a hypothesis. */
export interface EditCounts {
  readonly hits: number
  readonly substitutions: number
  readonly deletions: number
  readonly insertions: number
}

/** An approximate occurrence of a pattern in a text: `[start, end)` in elements, and its edit distance. */
export interface ApproximateOccurrence {
  readonly start: number
  readonly end: number
  readonly distance: number
}

/**
 * The state of {@link dpAlgorithm}: the table (row-major, `(rows + 1)` by `(columns + 1)`), how many rows are filled,
 * and what the traceback found. Steps fill the next row in place; keep a copy of a state you want to look at later.
 */
export interface DpState {
  readonly kind: DpKind
  readonly rows: number
  readonly columns: number
  /** The table, row-major. */
  readonly table: Float64Array
  /** The next row to fill (`rows + 1` once all are filled and the traceback is next). */
  row: number
  /** Elements shared at the start and the end, skipped by `edit-counts` (as the table is filled on the middle). */
  readonly prefix: number
  readonly suffix: number
  /** The traceback from the corner, as (row, column) pairs (`lcs` and `edit-counts`). */
  path: [number, number][]
  /** The indices into the first sequence of the common subsequence (`lcs`). */
  subsequence: number[]
  /** The alignment counts (`edit-counts`). */
  counts: EditCounts | null
  /** The occurrences (`sellers`). */
  occurrences: ApproximateOccurrence[]
  finished: boolean
}

/**
 * One of the edit-distance tables as a stepped algorithm, a row per step: the first sequence runs down the rows, the
 * second across the columns. `levenshtein` takes the cheapest of a deletion, an insertion and a (mis)match; `lcs` the
 * longer of the neighbours unless the elements match; `edit-counts` is the unit-cost table after skipping the shared
 * prefix and suffix, traced back to count hits, substitutions, deletions and insertions; `sellers` lets the match start
 * anywhere in the second sequence (the text) and reports each end within `k` edits. After the last row one step traces
 * back. It logs the row, its last cell and the path, and is done (`converged`) after the traceback. The state
 * serialises.
 *
 * @param a - The first sequence (the pattern, for `sellers`).
 * @param b - The second sequence (the text, for `sellers`).
 * @param options - The table, the weights and, for `sellers`, k.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a negative or fractional k.
 * @concept edit-distance
 * @example
 * ```ts
 * import { dpAlgorithm } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(dpAlgorithm('kitten', 'sitting'), { maxSteps: 20 })
 * [run.steps, run.state.table[6 * 8 + 7]] // => [7, 3]
 * ```
 */
export function dpAlgorithm<T>(a: Sequence<T>, b: Sequence<T>, options: DpOptions = {}): Algorithm<DpState, DpPath> {
  const kind = options.kind ?? 'levenshtein'
  const k = options.k ?? 0
  if (kind === 'sellers' && !(Number.isInteger(k) && k >= 0))
    throw new OmaiError('value', 'sellers: k must be a non-negative integer')
  const full = [elements(a), elements(b)] as const
  const ins = options.weights?.insertion ?? 1
  const del = options.weights?.deletion ?? 1
  const sub = options.weights?.substitution ?? 1
  // edit-counts skips the shared prefix and suffix.
  let prefix = 0
  let suffix = 0
  if (kind === 'edit-counts') {
    const [r, h] = full
    while (prefix < r.length && prefix < h.length && r[prefix] === h[prefix]) prefix++
    while (
      suffix < r.length - prefix &&
      suffix < h.length - prefix &&
      r[r.length - 1 - suffix] === h[h.length - 1 - suffix]
    )
      suffix++
  }
  const x = full[0].slice(prefix, full[0].length - suffix)
  const y = full[1].slice(prefix, full[1].length - suffix)
  const n = x.length
  const m = y.length
  const w = m + 1
  return {
    name: `text/dp-${kind}`,
    config: { kind, rows: n, columns: m, k },
    init: () => {
      const table = new Float64Array((n + 1) * w)
      for (let j = 0; j <= m; j++)
        table[j] = kind === 'levenshtein' ? j * ins : kind === 'lcs' || kind === 'sellers' ? 0 : j
      for (let i = 1; i <= n; i++) table[i * w] = kind === 'levenshtein' ? i * del : kind === 'lcs' ? 0 : i
      return {
        kind,
        rows: n,
        columns: m,
        table,
        row: 1,
        prefix,
        suffix,
        path: [],
        subsequence: [],
        counts: null,
        occurrences: [],
        finished: false,
      }
    },
    step: (s, ctx) => {
      const d = s.table
      const at = (i: number, j: number): number => d[i * w + j] ?? 0
      if (s.row <= n) {
        const i = s.row
        for (let j = 1; j <= m; j++) {
          const same = x[i - 1] === y[j - 1]
          d[i * w + j] =
            kind === 'levenshtein'
              ? Math.min(at(i - 1, j) + del, at(i, j - 1) + ins, at(i - 1, j - 1) + (same ? 0 : sub))
              : kind === 'lcs'
                ? same
                  ? at(i - 1, j - 1) + 1
                  : Math.max(at(i, j - 1), at(i - 1, j))
                : Math.min(at(i - 1, j) + 1, at(i, j - 1) + 1, at(i - 1, j - 1) + (same ? 0 : 1))
        }
        s.row = i + 1
        if (ctx.wants('dp/row')) ctx.log('dp/row', d.slice(i * w, (i + 1) * w))
        ctx.log('dp/cell', at(i, m))
        return s
      }
      // The traceback.
      if (kind === 'lcs') {
        let i = n
        let j = m
        s.path.push([i, j])
        while (i > 0 && j > 0) {
          if (x[i - 1] === y[j - 1]) {
            s.subsequence.push(i - 1)
            i--
            j--
          } else if (at(i - 1, j) >= at(i, j - 1)) i--
          else j--
          s.path.push([i, j])
        }
        s.subsequence.reverse()
      } else if (kind === 'edit-counts') {
        let hits = prefix + suffix
        let substitutions = 0
        let deletions = 0
        let insertions = 0
        let i = n
        let j = m
        s.path.push([i, j])
        while (i > 0 || j > 0) {
          const here = at(i, j)
          const same = i > 0 && j > 0 && x[i - 1] === y[j - 1]
          if (i > 0 && here === at(i - 1, j) + 1) {
            deletions++
            i--
          } else if (i > 0 && j > 0 && !same && here === at(i - 1, j - 1) + 1) {
            substitutions++
            i--
            j--
          } else if (j > 0 && here === at(i, j - 1) + 1) {
            insertions++
            j--
          } else {
            hits++
            i--
            j--
          }
          s.path.push([i, j])
        }
        s.counts = { hits, substitutions, deletions, insertions }
      } else if (kind === 'sellers') {
        for (let end = 0; end <= m; end++) {
          const dist = at(n, end)
          if (dist > k) continue
          let i = n
          let c = end
          while (i > 0) {
            const here = at(i, c)
            if (c > 0 && here === at(i - 1, c - 1) + (x[i - 1] === y[c - 1] ? 0 : 1)) {
              i--
              c--
            } else if (here === at(i - 1, c) + 1) i--
            else c--
          }
          s.occurrences.push({ start: c, end, distance: dist })
        }
      }
      s.finished = true
      ctx.log('dp/cell', at(n, m))
      if (s.path.length > 0) ctx.log('dp/path', s.path.flat())
      return s
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as DpState,
  }
}

/**
 * Run a table to the end.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @param options - The table and its settings.
 * @returns The finished state.
 * @concept edit-distance
 * @example
 * ```ts
 * import { dpTable } from 'omai-engine/text'
 * dpTable('ab', 'b', { kind: 'lcs' }).subsequence // => [1]
 * ```
 */
export function dpTable<T>(a: Sequence<T>, b: Sequence<T>, options: DpOptions = {}): DpState {
  return runAlgorithm(dpAlgorithm(a, b, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state
}
