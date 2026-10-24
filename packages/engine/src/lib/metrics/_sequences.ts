/**
 * Sequence alignment and n-gram counting shared by the text metrics. TODO: move the edit alignment and the longest
 * common subsequence to `text/similarity.ts` once the text domain exists.
 */

/** Hits, substitutions, deletions and insertions of an alignment of a reference with a hypothesis. */
export interface EditCounts {
  readonly hits: number
  readonly substitutions: number
  readonly deletions: number
  readonly insertions: number
}

/**
 * The Levenshtein alignment counts of reference against hypothesis with rapidfuzz's tie-breaking (as jiwer reads its
 * opcodes): the common prefix and suffix are hits, and the backtrace through the rest prefers a deletion, then a
 * substitution, then an insertion, then a match.
 */
export function editCounts<T>(reference: readonly T[], hypothesis: readonly T[]): EditCounts {
  let p = 0
  while (p < reference.length && p < hypothesis.length && reference[p] === hypothesis[p]) p++
  let s = 0
  while (
    s < reference.length - p &&
    s < hypothesis.length - p &&
    reference[reference.length - 1 - s] === hypothesis[hypothesis.length - 1 - s]
  )
    s++
  const a = reference.slice(p, reference.length - s)
  const b = hypothesis.slice(p, hypothesis.length - s)
  const n = a.length
  const m = b.length
  const d = Array.from({ length: n + 1 }, (_, x) =>
    Int32Array.from({ length: m + 1 }, (_, y) => (x === 0 ? y : y === 0 ? x : 0)),
  )
  const at = (x: number, y: number): number => d[x]?.[y] ?? 0
  for (let x = 1; x <= n; x++) {
    const row = d[x] as Int32Array
    for (let y = 1; y <= m; y++) {
      row[y] = Math.min(at(x - 1, y) + 1, at(x, y - 1) + 1, at(x - 1, y - 1) + (a[x - 1] === b[y - 1] ? 0 : 1))
    }
  }
  let hits = p + s
  let substitutions = 0
  let deletions = 0
  let insertions = 0
  let x = n
  let y = m
  while (x > 0 || y > 0) {
    const here = at(x, y)
    const same = x > 0 && y > 0 && a[x - 1] === b[y - 1]
    if (x > 0 && here === at(x - 1, y) + 1) {
      deletions++
      x--
    } else if (x > 0 && y > 0 && !same && here === at(x - 1, y - 1) + 1) {
      substitutions++
      x--
      y--
    } else if (y > 0 && here === at(x, y - 1) + 1) {
      insertions++
      y--
    } else {
      hits++
      x--
      y--
    }
  }
  return { hits, substitutions, deletions, insertions }
}

/** The dynamic-programming table of the longest common subsequence of two sequences. */
export function lcsTable<T>(a: readonly T[], b: readonly T[]): Int32Array[] {
  const t = Array.from({ length: a.length + 1 }, () => new Int32Array(b.length + 1))
  for (let i = 1; i <= a.length; i++) {
    const row = t[i] as Int32Array
    const prev = t[i - 1] as Int32Array
    for (let j = 1; j <= b.length; j++) {
      row[j] = a[i - 1] === b[j - 1] ? (prev[j - 1] ?? 0) + 1 : Math.max(row[j - 1] ?? 0, prev[j] ?? 0)
    }
  }
  return t
}

/** Counts of the n-grams of a token list, keyed by the tokens joined with a separator that cannot occur in them. */
export function ngramCounts(tokens: readonly string[], n: number, separator = '\u0000'): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i + n <= tokens.length; i++) {
    const key = tokens.slice(i, i + n).join(separator)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return counts
}

/** The clipped overlap sum over keys of min(count in a, count in b). */
export function overlap(a: ReadonlyMap<string, number>, b: ReadonlyMap<string, number>): number {
  let total = 0
  for (const [k, v] of a) total += Math.min(v, b.get(k) ?? 0)
  return total
}

/** The total count of a counter. */
export function total(a: ReadonlyMap<string, number>): number {
  let s = 0
  for (const v of a.values()) s += v
  return s
}
