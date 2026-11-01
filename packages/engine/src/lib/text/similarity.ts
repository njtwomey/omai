/**
 * String and sequence similarity measures, as rapidfuzz and jellyfish compute them: Levenshtein (weighted and
 * normalised, with edit counts for error rates), Damerau-Levenshtein, Hamming, Jaro and Jaro-Winkler, longest common
 * subsequence and substring, Indel, set measures (Jaccard, containment, clipped overlap) and the fuzzy ratios (ratio,
 * token-sort, token-set).
 */

import { MdmlError } from '../base/index.ts'
import { compareCodePoints, PY_SPACE } from './_classes.ts'
import { elements, type Sequence } from './_sequence.ts'
import { dpTable, type EditCounts, type EditWeights } from './dp-tables.ts'

export type { Sequence } from './_sequence.ts'
export type { EditCounts, EditWeights } from './dp-tables.ts'

/**
 * The Levenshtein (edit) distance: the least total cost of insertions, deletions and substitutions turning `a` into
 * `b` (unit costs by default, as rapidfuzz's Levenshtein.distance).
 *
 * @param a - The source.
 * @param b - The target.
 * @param weights - Operation costs (default 1 each).
 * @returns The distance.
 * @concept edit-distance
 * @example
 * ```ts
 * import { levenshtein } from 'mdml-engine/text'
 * levenshtein('kitten', 'sitting') // => 3
 * levenshtein('kitten', 'sitting', { substitution: 2 }) // => 5
 * ```
 */
export function levenshtein<T>(a: Sequence<T>, b: Sequence<T>, weights: EditWeights = {}): number {
  const x = elements(a)
  const y = elements(b)
  const ins = weights.insertion ?? 1
  const del = weights.deletion ?? 1
  const sub = weights.substitution ?? 1
  let prev = Float64Array.from({ length: y.length + 1 }, (_, j) => j * ins)
  let cur = new Float64Array(y.length + 1)
  for (let i = 1; i <= x.length; i++) {
    cur[0] = i * del
    for (let j = 1; j <= y.length; j++) {
      const same = x[i - 1] === y[j - 1]
      cur[j] = Math.min((prev[j] ?? 0) + del, (cur[j - 1] ?? 0) + ins, (prev[j - 1] ?? 0) + (same ? 0 : sub))
    }
    ;[prev, cur] = [cur, prev]
  }
  return prev[y.length] ?? 0
}

/**
 * The normalised Levenshtein similarity 1 - d / max(|a|, |b|) (1 for two empty sequences), rapidfuzz's
 * Levenshtein.normalized_similarity with unit weights.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The similarity in [0, 1].
 * @concept edit-distance
 * @example
 * ```ts
 * import { levenshteinSimilarity } from 'mdml-engine/text'
 * levenshteinSimilarity('kitten', 'sitting') // => 0.5714285714285714
 * ```
 */
export function levenshteinSimilarity<T>(a: Sequence<T>, b: Sequence<T>): number {
  const n = Math.max(elements(a).length, elements(b).length)
  return n === 0 ? 1 : 1 - levenshtein(a, b) / n
}

/**
 * The counts of a minimal Levenshtein alignment of a reference with a hypothesis, with rapidfuzz's tie-breaking (as
 * jiwer reads its opcodes): the common prefix and suffix are hits, and the backtrace through the rest prefers a
 * deletion, then a substitution, then an insertion, then a match. The counts behind WER, MER, WIL and CER.
 *
 * @param reference - The reference sequence.
 * @param hypothesis - The hypothesis sequence.
 * @returns The hits and the three kinds of error.
 * @concept edit-distance
 * @example
 * ```ts
 * import { editCounts } from 'mdml-engine/text'
 * editCounts(['the', 'cat', 'sat'], ['the', 'cat', 'sit', 'down']) // => { hits: 2, substitutions: 1, deletions: 0, insertions: 1 }
 * ```
 */
export function editCounts<T>(reference: Sequence<T>, hypothesis: Sequence<T>): EditCounts {
  return dpTable(reference, hypothesis, { kind: 'edit-counts' }).counts as EditCounts
}

/**
 * The Damerau-Levenshtein distance: edits plus transpositions of adjacent elements. `restricted: true` is the optimal
 * string alignment distance (no substring is edited twice; rapidfuzz's OSA), the default is the unrestricted distance
 * of Lowrance and Wagner (rapidfuzz's DamerauLevenshtein).
 *
 * @param a - The source.
 * @param b - The target.
 * @param options - Whether to use the restricted (OSA) form.
 * @returns The distance.
 * @concept edit-distance
 * @example
 * ```ts
 * import { damerauLevenshtein } from 'mdml-engine/text'
 * [damerauLevenshtein('ca', 'abc'), damerauLevenshtein('ca', 'abc', { restricted: true })] // => [2, 3]
 * ```
 */
export function damerauLevenshtein<T>(
  a: Sequence<T>,
  b: Sequence<T>,
  options: { readonly restricted?: boolean } = {},
): number {
  const x = elements(a)
  const y = elements(b)
  const n = x.length
  const m = y.length
  if (options.restricted === true) {
    const d = Array.from({ length: n + 1 }, (_, i) =>
      Int32Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
    )
    const at = (i: number, j: number): number => d[i]?.[j] ?? 0
    for (let i = 1; i <= n; i++) {
      const row = d[i] as Int32Array
      for (let j = 1; j <= m; j++) {
        const cost = x[i - 1] === y[j - 1] ? 0 : 1
        let v = Math.min(at(i - 1, j) + 1, at(i, j - 1) + 1, at(i - 1, j - 1) + cost)
        if (i > 1 && j > 1 && x[i - 1] === y[j - 2] && x[i - 2] === y[j - 1]) v = Math.min(v, at(i - 2, j - 2) + 1)
        row[j] = v
      }
    }
    return at(n, m)
  }
  // Lowrance-Wagner with the last row at which each element was seen.
  const big = n + m
  const d = Array.from({ length: n + 2 }, () => new Int32Array(m + 2))
  const at = (i: number, j: number): number => d[i]?.[j] ?? 0
  const set = (i: number, j: number, v: number): void => {
    const row = d[i]
    if (row !== undefined) row[j] = v
  }
  set(0, 0, big)
  for (let i = 0; i <= n; i++) {
    set(i + 1, 0, big)
    set(i + 1, 1, i)
  }
  for (let j = 0; j <= m; j++) {
    set(0, j + 1, big)
    set(1, j + 1, j)
  }
  const lastRow = new Map<T | string, number>()
  for (let i = 1; i <= n; i++) {
    let lastCol = 0
    for (let j = 1; j <= m; j++) {
      const i1 = lastRow.get(y[j - 1] as T | string) ?? 0
      const j1 = lastCol
      const same = x[i - 1] === y[j - 1]
      if (same) lastCol = j
      set(
        i + 1,
        j + 1,
        Math.min(
          at(i, j) + (same ? 0 : 1),
          at(i + 1, j) + 1,
          at(i, j + 1) + 1,
          at(i1, j1) + (i - i1 - 1) + 1 + (j - j1 - 1),
        ),
      )
    }
    lastRow.set(x[i - 1] as T | string, i)
  }
  return at(n + 1, m + 1)
}

/**
 * The Hamming distance: the number of positions at which two equal-length sequences differ.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The number of mismatches.
 * @throws MdmlError with code `shape` for sequences of different lengths.
 * @concept edit-distance
 * @example
 * ```ts
 * import { hamming } from 'mdml-engine/text'
 * hamming('karolin', 'kathrin') // => 3
 * ```
 */
export function hamming<T>(a: Sequence<T>, b: Sequence<T>): number {
  const x = elements(a)
  const y = elements(b)
  if (x.length !== y.length) throw new MdmlError('shape', 'hamming: sequences must have equal length')
  let d = 0
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) d++
  return d
}

/**
 * The Jaro similarity (Jaro 1989) as rapidfuzz computes it: matches within a window of max(|a|, |b|) / 2 - 1, and
 * half the transposed matches; 1 for two empty sequences.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The similarity in [0, 1].
 * @concept string-similarity
 * @example
 * ```ts
 * import { jaro } from 'mdml-engine/text'
 * jaro('MARTHA', 'MARHTA') // => 0.9444444444444445
 * ```
 */
export function jaro<T>(a: Sequence<T>, b: Sequence<T>): number {
  let x = elements(a)
  let y = elements(b)
  if (x.length === 0 && y.length === 0) return 1
  const pl = x.length
  const tl = y.length
  if (pl === 0 || tl === 0) return 0
  if (pl === 1 && tl === 1) return x[0] === y[0] ? 1 : 0
  let bound: number
  if (tl > pl) {
    bound = Math.floor(tl / 2) - 1
    if (tl > pl + bound) y = y.slice(0, pl + bound)
  } else {
    bound = Math.floor(pl / 2) - 1
    if (pl > tl + bound) x = x.slice(0, tl + bound)
  }
  const xf = new Array<boolean>(pl).fill(false)
  const yf = new Array<boolean>(tl).fill(false)
  let common = 0
  for (const [i, ch] of x.entries()) {
    const lo = Math.max(0, i - bound)
    const hi = Math.min(i + bound, tl - 1)
    for (let j = lo; j <= hi; j++) {
      if (!yf[j] && y[j] === ch) {
        xf[i] = true
        yf[j] = true
        common++
        break
      }
    }
  }
  if (common === 0) return 0
  let k = 0
  let trans = 0
  for (let i = 0; i < pl; i++) {
    if (!xf[i]) continue
    let j = k
    for (; j < tl; j++) {
      if (yf[j]) {
        k = j + 1
        break
      }
    }
    if (x[i] !== y[j]) trans++
  }
  const t = Math.floor(trans / 2)
  return (common / pl + common / tl + (common - t) / common) / 3
}

/**
 * The Jaro-Winkler similarity (Winkler 1990): the Jaro similarity raised for a common prefix of up to four elements,
 * when it exceeds 0.7 (rapidfuzz's JaroWinkler.similarity).
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @param prefixWeight - The weight of each prefix element (default 0.1, at most 0.25).
 * @returns The similarity in [0, 1].
 * @throws MdmlError with code `value` for a prefix weight outside [0, 0.25].
 * @concept string-similarity
 * @example
 * ```ts
 * import { jaroWinkler } from 'mdml-engine/text'
 * jaroWinkler('MARTHA', 'MARHTA') // => 0.9611111111111111
 * ```
 */
export function jaroWinkler<T>(a: Sequence<T>, b: Sequence<T>, prefixWeight = 0.1): number {
  if (!(prefixWeight >= 0 && prefixWeight <= 0.25))
    throw new MdmlError('value', 'jaroWinkler: prefixWeight must be in [0, 0.25]')
  const x = elements(a)
  const y = elements(b)
  if (x.length === 0 && y.length === 0) return 1
  let prefix = 0
  const max = Math.min(x.length, y.length, 4)
  while (prefix < max && x[prefix] === y[prefix]) prefix++
  const sim = jaro(x, y)
  return sim > 0.7 ? sim + prefix * prefixWeight * (1 - sim) : sim
}

/**
 * The dynamic-programming table of the longest common subsequence: entry [i][j] is the LCS length of the first i
 * elements of `a` and the first j of `b`.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The (|a| + 1) x (|b| + 1) table.
 * @concept longest-common-subsequence
 * @example
 * ```ts
 * import { lcsTable } from 'mdml-engine/text'
 * lcsTable('ab', 'b').map((r) => Array.from(r)) // => [[0, 0], [0, 0], [0, 1]]
 * ```
 */
export function lcsTable<T>(a: Sequence<T>, b: Sequence<T>): Int32Array[] {
  const x = elements(a)
  const y = elements(b)
  const t = Array.from({ length: x.length + 1 }, () => new Int32Array(y.length + 1))
  for (let i = 1; i <= x.length; i++) {
    const row = t[i] as Int32Array
    const prev = t[i - 1] as Int32Array
    for (let j = 1; j <= y.length; j++) {
      row[j] = x[i - 1] === y[j - 1] ? (prev[j - 1] ?? 0) + 1 : Math.max(row[j - 1] ?? 0, prev[j] ?? 0)
    }
  }
  return t
}

/**
 * A longest common subsequence: the elements, as a backtrace through {@link lcsTable} that prefers moving up when
 * the two neighbours tie.
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The subsequence (an array; join it for strings).
 * @concept longest-common-subsequence
 * @example
 * ```ts
 * import { longestCommonSubsequence } from 'mdml-engine/text'
 * longestCommonSubsequence('ABCBDAB', 'BDCABA').join('') // => 'BCBA'
 * ```
 */
export function longestCommonSubsequence<T>(a: Sequence<T>, b: Sequence<T>): (T | string)[] {
  const x = elements(a)
  return dpTable(a, b, { kind: 'lcs' }).subsequence.map((i) => x[i] as T | string)
}

/**
 * The longest common substring (contiguous): its start in each sequence and its length (the first such, scanning `a`
 * then `b`).
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns `{ startA, startB, length }`.
 * @concept longest-common-subsequence
 * @example
 * ```ts
 * import { longestCommonSubstring } from 'mdml-engine/text'
 * longestCommonSubstring('xabcdy', 'zbcdw') // => { startA: 2, startB: 1, length: 3 }
 * ```
 */
export function longestCommonSubstring<T>(
  a: Sequence<T>,
  b: Sequence<T>,
): { readonly startA: number; readonly startB: number; readonly length: number } {
  const x = elements(a)
  const y = elements(b)
  let prev = new Int32Array(y.length + 1)
  let cur = new Int32Array(y.length + 1)
  let best = { startA: 0, startB: 0, length: 0 }
  for (let i = 1; i <= x.length; i++) {
    for (let j = 1; j <= y.length; j++) {
      const v = x[i - 1] === y[j - 1] ? (prev[j - 1] ?? 0) + 1 : 0
      cur[j] = v
      if (v > best.length) best = { startA: i - v, startB: j - v, length: v }
    }
    ;[prev, cur] = [cur, prev]
    cur.fill(0)
  }
  return best
}

/**
 * The Indel distance (insertions and deletions only): |a| + |b| - 2 LCS(a, b).
 *
 * @param a - The first sequence.
 * @param b - The second sequence.
 * @returns The distance.
 * @concept edit-distance
 * @example
 * ```ts
 * import { indel } from 'mdml-engine/text'
 * indel('kitten', 'sitting') // => 5
 * ```
 */
export function indel<T>(a: Sequence<T>, b: Sequence<T>): number {
  const x = elements(a)
  const y = elements(b)
  return x.length + y.length - 2 * (lcsTable(x, y)[x.length]?.[y.length] ?? 0)
}

/** Set or multiset input of the overlap measures. */
type Items<T> = Iterable<T> | ReadonlyMap<T, number>

function counts<T>(items: Items<T>): Map<T, number> {
  if (items instanceof Map) return new Map(items)
  const out = new Map<T, number>()
  for (const t of items as Iterable<T>) out.set(t, (out.get(t) ?? 0) + 1)
  return out
}

/**
 * The Jaccard similarity |A ∩ B| / |A ∪ B| of two sets (of shingles, tokens, ...); with `weighted: true`, of two
 * multisets (counts or weights): sum min / sum max (Ruzicka). 1 for two empty inputs.
 *
 * @param a - The first collection (or a map of weights).
 * @param b - The second collection (or a map of weights).
 * @param options - Whether to use multiplicities.
 * @returns The similarity in [0, 1].
 * @concept set-similarity
 * @example
 * ```ts
 * import { jaccard } from 'mdml-engine/text'
 * [jaccard(['a', 'b', 'c'], ['b', 'c', 'd']), jaccard(['a', 'a', 'b'], ['a', 'b', 'b'], { weighted: true })] // => [0.5, 0.5]
 * ```
 */
export function jaccard<T>(a: Items<T>, b: Items<T>, options: { readonly weighted?: boolean } = {}): number {
  const x = counts(a)
  const y = counts(b)
  if (options.weighted !== true) {
    let inter = 0
    for (const k of x.keys()) if (y.has(k)) inter++
    const union = x.size + y.size - inter
    return union === 0 ? 1 : inter / union
  }
  let lo = 0
  let hi = 0
  for (const [k, v] of x) {
    const w = y.get(k) ?? 0
    lo += Math.min(v, w)
    hi += Math.max(v, w)
  }
  for (const [k, w] of y) if (!x.has(k)) hi += w
  return hi === 0 ? 1 : lo / hi
}

/**
 * The containment |A ∩ B| / |A| of a set in another (Broder 1997): how much of `a` occurs in `b` (1 for empty `a`).
 *
 * @param a - The contained collection.
 * @param b - The containing collection.
 * @returns The containment in [0, 1].
 * @concept set-similarity
 * @example
 * ```ts
 * import { containment } from 'mdml-engine/text'
 * containment(['a', 'b'], ['a', 'b', 'c', 'd']) // => 1
 * ```
 */
export function containment<T>(a: Iterable<T>, b: Iterable<T>): number {
  const x = new Set(a)
  const y = new Set(b)
  if (x.size === 0) return 1
  let inter = 0
  for (const k of x) if (y.has(k)) inter++
  return inter / x.size
}

/**
 * The clipped overlap of two count maps: the sum over keys of min(count in a, count in b) (BLEU's clipped matches,
 * ROUGE-N's hits).
 *
 * @param a - The first counts.
 * @param b - The second counts.
 * @returns The overlap.
 * @concept set-similarity
 * @example
 * ```ts
 * import { clippedOverlap } from 'mdml-engine/text'
 * clippedOverlap(new Map([['a', 2], ['b', 1]]), new Map([['a', 1], ['c', 4]])) // => 1
 * ```
 */
export function clippedOverlap<K>(a: ReadonlyMap<K, number>, b: ReadonlyMap<K, number>): number {
  let total = 0
  for (const [k, v] of a) total += Math.min(v, b.get(k) ?? 0)
  return total
}

/**
 * The total of a count map.
 *
 * @param a - The counts.
 * @returns The sum of the counts.
 * @example
 * ```ts
 * import { countTotal } from 'mdml-engine/text'
 * countTotal(new Map([['a', 2], ['b', 3]])) // => 5
 * ```
 */
export function countTotal<K>(a: ReadonlyMap<K, number>): number {
  let s = 0
  for (const v of a.values()) s += v
  return s
}

const SPLIT = new RegExp(`[${PY_SPACE}]+`, 'u')
const words = (s: string): string[] => s.split(SPLIT).filter((w) => w.length > 0)
const indelRatio = (dist: number, lensum: number): number => (lensum === 0 ? 100 : 100 - (100 * dist) / lensum)

/**
 * The fuzzy matching ratio 100 (1 - Indel(a, b) / (|a| + |b|)) (rapidfuzz's fuzz.ratio, python-Levenshtein's ratio).
 *
 * @param a - The first string.
 * @param b - The second string.
 * @returns The ratio in [0, 100].
 * @concept string-similarity
 * @example
 * ```ts
 * import { ratio } from 'mdml-engine/text'
 * ratio('this is a test', 'this is a test!') // => 96.55172413793103
 * ```
 */
export function ratio(a: string, b: string): number {
  const x = Array.from(a)
  const y = Array.from(b)
  return indelRatio(indel(x, y), x.length + y.length)
}

/**
 * The token-sort ratio (rapidfuzz's fuzz.token_sort_ratio): {@link ratio} of the whitespace-separated words of each
 * string, sorted and joined by single spaces.
 *
 * @param a - The first string.
 * @param b - The second string.
 * @returns The ratio in [0, 100].
 * @concept string-similarity
 * @example
 * ```ts
 * import { tokenSortRatio } from 'mdml-engine/text'
 * tokenSortRatio('fuzzy wuzzy was a bear', 'wuzzy fuzzy was a bear') // => 100
 * ```
 */
export function tokenSortRatio(a: string, b: string): number {
  return ratio(words(a).sort(compareCodePoints).join(' '), words(b).sort(compareCodePoints).join(' '))
}

/**
 * The token-set ratio (rapidfuzz's fuzz.token_set_ratio): 100 when one word set contains the other, otherwise the
 * best of the ratios between the sorted intersection with each sorted difference, and between the differences.
 *
 * @param a - The first string.
 * @param b - The second string.
 * @returns The ratio in [0, 100] (0 when either string has no words).
 * @concept string-similarity
 * @example
 * ```ts
 * import { tokenSetRatio } from 'mdml-engine/text'
 * [tokenSetRatio('fuzzy was a bear', 'fuzzy fuzzy was a bear'), tokenSetRatio('fuzzy was a bear but not a dog', 'fuzzy was a bear but not a cat')] // => [100, 92.3076923076923]
 * ```
 */
export function tokenSetRatio(a: string, b: string): number {
  const ta = new Set(words(a))
  const tb = new Set(words(b))
  if (ta.size === 0 || tb.size === 0) return 0
  const inter = [...ta].filter((w) => tb.has(w))
  const diffAb = [...ta].filter((w) => !tb.has(w)).sort(compareCodePoints)
  const diffBa = [...tb].filter((w) => !ta.has(w)).sort(compareCodePoints)
  if (inter.length > 0 && (diffAb.length === 0 || diffBa.length === 0)) return 100
  const ab = diffAb.join(' ')
  const ba = diffBa.join(' ')
  const abLen = Array.from(ab).length
  const baLen = Array.from(ba).length
  const sectLen = Array.from(inter.join(' ')).length
  const sep = sectLen === 0 ? 0 : 1
  const sectAb = sectLen + sep + abLen
  const sectBa = sectLen + sep + baLen
  const result = indelRatio(indel(ab, ba), sectAb + sectBa)
  if (sectLen === 0) return result
  return Math.max(result, indelRatio(sep + abLen, sectLen + sectAb), indelRatio(sep + baLen, sectLen + sectBa))
}
