import { WhiteboxError } from '../base/index.ts'
import type { Sequence } from './similarity.ts'

/**
 * Scores of a pairwise alignment. Gaps are linear (`gap` per position) or affine: a gap of length k scores
 * `gapOpen + (k - 1) gapExtend` (Biopython's PairwiseAligner convention). A `substitution` function overrides
 * `match` and `mismatch`.
 */
export interface AlignmentScoring<T = string> {
  /** Score of aligning equal elements (default 1). */
  readonly match?: number
  /** Score of aligning different elements (default -1). */
  readonly mismatch?: number
  /** Score of each gap position, for linear gaps (default -1). */
  readonly gap?: number
  /** Score of the first position of a gap (affine gaps; default `gap`). */
  readonly gapOpen?: number
  /** Score of each further position of a gap (affine gaps; default `gap`). */
  readonly gapExtend?: number
  /** A substitution score of two elements (a substitution matrix). */
  readonly substitution?: (x: T, y: T) => number
}

/** A pairwise alignment: its score and the aligned index pairs (-1 marks a gap), over `a[startA, endA)` and `b[startB, endB)`. */
export interface Alignment {
  readonly score: number
  readonly pairs: readonly (readonly [number, number])[]
  readonly startA: number
  readonly endA: number
  readonly startB: number
  readonly endB: number
}

const NEG = Number.NEGATIVE_INFINITY

function align<T>(a: Sequence<T>, b: Sequence<T>, scoring: AlignmentScoring<T | string>, local: boolean): Alignment {
  const x = typeof a === 'string' ? Array.from(a) : a
  const y = typeof b === 'string' ? Array.from(b) : b
  const n = x.length
  const m = y.length
  const gap = scoring.gap ?? -1
  const open = scoring.gapOpen ?? gap
  const extend = scoring.gapExtend ?? gap
  if (![open, extend, scoring.match ?? 1, scoring.mismatch ?? -1].every(Number.isFinite))
    throw new WhiteboxError('value', 'alignment: scores must be finite')
  const sub =
    scoring.substitution ??
    ((p: T | string, q: T | string): number => (p === q ? (scoring.match ?? 1) : (scoring.mismatch ?? -1)))
  const w = m + 1
  // M: ends with a pair; X: ends with a[i-1] against a gap; Y: ends with b[j-1] against a gap.
  const M = new Float64Array((n + 1) * w).fill(NEG)
  const X = new Float64Array((n + 1) * w).fill(NEG)
  const Y = new Float64Array((n + 1) * w).fill(NEG)
  M[0] = 0
  if (!local) {
    for (let i = 1; i <= n; i++) X[i * w] = open + (i - 1) * extend
    for (let j = 1; j <= m; j++) Y[j] = open + (j - 1) * extend
  }
  const at = (t: Float64Array, i: number, j: number): number => t[i * w + j] ?? NEG
  const max3 = (p: number, q: number, r: number): number => Math.max(p, q, r)
  let best = local ? 0 : NEG
  let bi = 0
  let bj = 0
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const s = sub(x[i - 1] as T | string, y[j - 1] as T | string)
      let mv = s + max3(at(M, i - 1, j - 1), at(X, i - 1, j - 1), at(Y, i - 1, j - 1))
      if (local) mv = Math.max(mv, s)
      M[i * w + j] = mv
      X[i * w + j] = max3(at(M, i - 1, j) + open, at(X, i - 1, j) + extend, at(Y, i - 1, j) + open)
      Y[i * w + j] = max3(at(M, i, j - 1) + open, at(Y, i, j - 1) + extend, at(X, i, j - 1) + open)
      if (local && mv > best) {
        best = mv
        bi = i
        bj = j
      }
    }
  }
  let i = local ? bi : n
  let j = local ? bj : m
  let state: 'M' | 'X' | 'Y'
  if (local) state = 'M'
  else {
    const mm = at(M, n, m)
    const xx = at(X, n, m)
    const yy = at(Y, n, m)
    best = max3(mm, xx, yy)
    state = n === 0 && m === 0 ? 'M' : mm === best ? 'M' : xx === best ? 'X' : 'Y'
  }
  const pairs: [number, number][] = []
  if (local && best <= 0) return { score: 0, pairs: [], startA: 0, endA: 0, startB: 0, endB: 0 }
  const endA = i
  const endB = j
  while (i > 0 || j > 0) {
    if (state === 'M') {
      if (i === 0 || j === 0) break
      const s = sub(x[i - 1] as T | string, y[j - 1] as T | string)
      const here = at(M, i, j)
      pairs.push([i - 1, j - 1])
      if (local && here === s) {
        i--
        j--
        break
      }
      state = here === s + at(M, i - 1, j - 1) ? 'M' : here === s + at(X, i - 1, j - 1) ? 'X' : 'Y'
      i--
      j--
    } else if (state === 'X') {
      const here = at(X, i, j)
      pairs.push([i - 1, -1])
      state =
        i === 1 && j === 0 && !local
          ? 'M'
          : here === at(M, i - 1, j) + open
            ? 'M'
            : here === at(X, i - 1, j) + extend
              ? 'X'
              : 'Y'
      i--
    } else {
      const here = at(Y, i, j)
      pairs.push([-1, j - 1])
      state =
        j === 1 && i === 0 && !local
          ? 'M'
          : here === at(M, i, j - 1) + open
            ? 'M'
            : here === at(Y, i, j - 1) + extend
              ? 'Y'
              : 'X'
      j--
    }
  }
  pairs.reverse()
  return { score: best, pairs, startA: i, endA, startB: j, endB }
}

/**
 * Global alignment (Needleman and Wunsch 1970; Gotoh 1982 for affine gaps): the best-scoring alignment of the whole
 * of both sequences. Ties in the traceback prefer a pair, then a gap in `b`, then a gap in `a`.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @param scoring - Match, mismatch and gap scores (default 1, -1, -1).
 * @returns The alignment.
 * @throws WhiteboxError with code `value` for non-finite scores.
 * @concept sequence-alignment
 * @example
 * ```ts
 * import { needlemanWunsch, formatAlignment } from 'whitebox-engine/text'
 * const r = needlemanWunsch('GATTACA', 'GCATGCU', { match: 1, mismatch: -1, gap: -1 })
 * [r.score, formatAlignment('GATTACA', 'GCATGCU', r)] // => [0, ['G-ATTACA', 'GCA-TGCU']]
 * ```
 */
export function needlemanWunsch<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T | string> = {},
): Alignment {
  return align(a, b, scoring, false)
}

/**
 * Local alignment (Smith and Waterman 1981; Gotoh's affine gaps): the best-scoring alignment of a part of each
 * sequence (the first best cell in row-major order, traced back to where its score starts).
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @param scoring - Match, mismatch and gap scores (default 1, -1, -1).
 * @returns The alignment (empty, score 0, when no pair scores positively).
 * @throws WhiteboxError with code `value` for non-finite scores.
 * @concept sequence-alignment
 * @example
 * ```ts
 * import { smithWaterman, formatAlignment } from 'whitebox-engine/text'
 * const r = smithWaterman('TGTTACGG', 'GGTTGACTA', { match: 3, mismatch: -3, gap: -2 })
 * [r.score, formatAlignment('TGTTACGG', 'GGTTGACTA', r)] // => [13, ['GTT-AC', 'GTTGAC']]
 * ```
 */
export function smithWaterman<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  scoring: AlignmentScoring<T | string> = {},
): Alignment {
  return align(a, b, scoring, true)
}

/**
 * The two rows of an alignment of strings, with a gap character.
 *
 * @param a - The first string.
 * @param b - The second string.
 * @param alignment - Their alignment.
 * @param gap - The gap character (default `-`).
 * @returns The aligned rows.
 * @concept sequence-alignment
 * @example
 * ```ts
 * import { formatAlignment } from 'whitebox-engine/text'
 * formatAlignment('ab', 'b', { score: 0, pairs: [[0, -1], [1, 0]], startA: 0, endA: 2, startB: 0, endB: 1 }) // => ['ab', '-b']
 * ```
 */
export function formatAlignment(a: string, b: string, alignment: Alignment, gap = '-'): [string, string] {
  const x = Array.from(a)
  const y = Array.from(b)
  let top = ''
  let bottom = ''
  for (const [i, j] of alignment.pairs) {
    top += i < 0 ? gap : (x[i] ?? '')
    bottom += j < 0 ? gap : (y[j] ?? '')
  }
  return [top, bottom]
}

/**
 * The score of a given alignment under a scoring scheme (to check an alignment, or score one made by hand).
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @param pairs - The aligned index pairs (-1 for a gap).
 * @param scoring - The scores.
 * @returns The total score.
 * @concept sequence-alignment
 * @example
 * ```ts
 * import { alignmentScore } from 'whitebox-engine/text'
 * alignmentScore('ab', 'b', [[0, -1], [1, 0]]) // => 0
 * ```
 */
export function alignmentScore<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  pairs: readonly (readonly [number, number])[],
  scoring: AlignmentScoring<T | string> = {},
): number {
  const x = typeof a === 'string' ? Array.from(a) : a
  const y = typeof b === 'string' ? Array.from(b) : b
  const gap = scoring.gap ?? -1
  const open = scoring.gapOpen ?? gap
  const extend = scoring.gapExtend ?? gap
  const sub =
    scoring.substitution ??
    ((p: T | string, q: T | string): number => (p === q ? (scoring.match ?? 1) : (scoring.mismatch ?? -1)))
  let score = 0
  let prev: 'M' | 'X' | 'Y' = 'M'
  for (const [i, j] of pairs) {
    if (i >= 0 && j >= 0) {
      score += sub(x[i] as T | string, y[j] as T | string)
      prev = 'M'
    } else if (j < 0) {
      score += prev === 'X' ? extend : open
      prev = 'X'
    } else {
      score += prev === 'Y' ? extend : open
      prev = 'Y'
    }
  }
  return score
}
