/**
 * Counting and enumeration: exact factorials, binomial and multinomial coefficients, permutations and combinations as
 * Python's itertools lists them, ranking and unranking them in lexicographic order, derangements, Stirling numbers of
 * both kinds, Bell, Catalan and partition numbers, integer partitions, the inclusion-exclusion principle, and Pascal's
 * triangle (optionally modulo m).
 */

import { WhiteboxError } from '../base/index.ts'

function count(n: number, name: string): number {
  if (!(Number.isSafeInteger(n) && n >= 0))
    throw new WhiteboxError('value', `${name}: expected a non-negative integer, got ${n}`)
  return n
}

/**
 * The factorial `n! = 1 * 2 * ... * n`, exactly (a BigInt; `numerics/special` has the floating-point gamma form).
 *
 * @param n - A non-negative integer.
 * @returns n!.
 * @concept factorial
 * @example
 * ```ts
 * import { factorialExact } from 'whitebox-engine/foundations'
 * factorialExact(20) // => 2432902008176640000n
 * ```
 */
export function factorialExact(n: number): bigint {
  count(n, 'factorialExact')
  let f = 1n
  for (let k = 2n; k <= BigInt(n); k++) f *= k
  return f
}

/**
 * The binomial coefficient `C(n, k) = n! / (k! (n - k)!)`, exactly, by the multiplicative formula (zero for k > n).
 *
 * @param n - A non-negative integer.
 * @param k - A non-negative integer.
 * @returns C(n, k).
 * @concept binomial-coefficients
 * @example
 * ```ts
 * import { binomialExact } from 'whitebox-engine/foundations'
 * binomialExact(10, 3) // => 120n
 * binomialExact(100, 50) // => 100891344545564193334812497256n
 * ```
 */
export function binomialExact(n: number, k: number): bigint {
  count(n, 'binomialExact')
  count(k, 'binomialExact')
  if (k > n) return 0n
  const kk = Math.min(k, n - k)
  let c = 1n
  for (let i = 1; i <= kk; i++) c = (c * BigInt(n - kk + i)) / BigInt(i)
  return c
}

/**
 * The multinomial coefficient `(k_1 + ... + k_m)! / (k_1! ... k_m!)`: the number of ways to split `sum k_i` labelled
 * items into groups of the given sizes, exactly.
 *
 * @param ks - The group sizes.
 * @returns The coefficient.
 * @concept multinomial-coefficients
 * @example
 * ```ts
 * import { multinomialExact } from 'whitebox-engine/foundations'
 * // MISSISSIPPI: 11! / (1! 4! 4! 2!)
 * multinomialExact([1, 4, 4, 2]) // => 34650n
 * ```
 */
export function multinomialExact(ks: readonly number[]): bigint {
  let total = 0
  let c = 1n
  for (const k of ks) {
    total += count(k, 'multinomialExact')
    c *= binomialExact(total, k)
  }
  return c
}

/**
 * All r-permutations of the items in lexicographic order of positions (as Python's `itertools.permutations`).
 *
 * @param items - The items.
 * @param r - The length (default all of them).
 * @returns The arrangements.
 * @concept permutations
 * @example
 * ```ts
 * import { permutations } from 'whitebox-engine/foundations'
 * permutations(['a', 'b', 'c'], 2) // => [['a', 'b'], ['a', 'c'], ['b', 'a'], ['b', 'c'], ['c', 'a'], ['c', 'b']]
 * ```
 */
export function permutations<T>(items: readonly T[], r: number = items.length): T[][] {
  count(r, 'permutations')
  const out: T[][] = []
  const used = items.map(() => false)
  const current: T[] = []
  const walk = (): void => {
    if (current.length === r) {
      out.push([...current])
      return
    }
    for (let i = 0; i < items.length; i++) {
      if (used[i]) continue
      used[i] = true
      current.push(items[i] as T)
      walk()
      current.pop()
      used[i] = false
    }
  }
  if (r <= items.length) walk()
  return out
}

/**
 * All r-combinations of the items in lexicographic order of positions (as Python's `itertools.combinations`).
 *
 * @param items - The items.
 * @param r - The size of each combination.
 * @returns The subsets, each in the items' order.
 * @concept combinations
 * @example
 * ```ts
 * import { combinations } from 'whitebox-engine/foundations'
 * combinations([1, 2, 3, 4], 2) // => [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
 * ```
 */
export function combinations<T>(items: readonly T[], r: number): T[][] {
  count(r, 'combinations')
  const out: T[][] = []
  const idx: number[] = []
  const walk = (start: number): void => {
    if (idx.length === r) {
      out.push(idx.map((i) => items[i] as T))
      return
    }
    for (let i = start; i <= items.length - (r - idx.length); i++) {
      idx.push(i)
      walk(i + 1)
      idx.pop()
    }
  }
  walk(0)
  return out
}

/**
 * The lexicographic rank of a permutation of `0..n-1` among all n! (0 for the identity), from its Lehmer code
 * `rank = sum_i c_i (n - 1 - i)!` where c_i counts the later entries smaller than entry i.
 *
 * @param perm - A permutation of 0..n-1.
 * @returns The rank (a BigInt, since n! grows fast) and the Lehmer code.
 * @throws WhiteboxError with code `value` when the input is not a permutation.
 * @concept permutation-ranking
 * @example
 * ```ts
 * import { permutationRank } from 'whitebox-engine/foundations'
 * permutationRank([1, 0, 2]).rank // => 2n
 * permutationRank([2, 1, 0]).lehmer // => [2, 1, 0]
 * ```
 */
export function permutationRank(perm: readonly number[]): { readonly rank: bigint; readonly lehmer: number[] } {
  const n = perm.length
  const seen = new Set(perm)
  if (seen.size !== n || perm.some((p) => !Number.isInteger(p) || p < 0 || p >= n)) {
    throw new WhiteboxError('value', 'permutationRank: expected a permutation of 0..n-1')
  }
  const lehmer = perm.map((p, i) => perm.slice(i + 1).filter((q) => q < p).length)
  let rank = 0n
  for (let i = 0; i < n; i++) rank = rank * BigInt(n - i) + BigInt(lehmer[i] ?? 0)
  return { rank, lehmer }
}

/**
 * The permutation of `0..n-1` with a given lexicographic rank (the inverse of {@link permutationRank}): read the rank
 * in the factorial number system and pick the remaining elements by its digits.
 *
 * @param n - The size.
 * @param rank - The rank in `[0, n!)`.
 * @returns The permutation.
 * @throws WhiteboxError with code `value` for a rank out of range.
 * @concept permutation-ranking
 * @example
 * ```ts
 * import { permutationUnrank } from 'whitebox-engine/foundations'
 * permutationUnrank(4, 23) // => [3, 2, 1, 0]
 * ```
 */
export function permutationUnrank(n: number, rank: number | bigint): number[] {
  count(n, 'permutationUnrank')
  let r = BigInt(rank)
  if (r < 0n || r >= factorialExact(n)) throw new WhiteboxError('value', 'permutationUnrank: rank out of range')
  const digits: number[] = []
  for (let base = 1; base <= n; base++) {
    digits.push(Number(r % BigInt(base)))
    r /= BigInt(base)
  }
  digits.reverse()
  const pool = Array.from({ length: n }, (_, i) => i)
  return digits.map((d) => pool.splice(d, 1)[0] as number)
}

/**
 * The lexicographic rank of a k-subset of `0..n-1` (given in increasing order) among all `C(n, k)` subsets.
 *
 * @param subset - Increasing elements of 0..n-1.
 * @param n - The size of the ground set.
 * @returns The rank (a BigInt).
 * @concept combination-ranking
 * @example
 * ```ts
 * import { combinationRank } from 'whitebox-engine/foundations'
 * combinationRank([0, 1], 4) // => 0n
 * combinationRank([2, 3], 4) // => 5n
 * ```
 */
export function combinationRank(subset: readonly number[], n: number): bigint {
  count(n, 'combinationRank')
  const k = subset.length
  if (subset.some((v, i) => !Number.isInteger(v) || v < 0 || v >= n || (i > 0 && v <= (subset[i - 1] ?? -1)))) {
    throw new WhiteboxError('value', 'combinationRank: expected increasing elements of 0..n-1')
  }
  // Count the subsets that come before: for each position, those with a smaller element there.
  let rank = 0n
  let prev = -1
  for (let i = 0; i < k; i++) {
    const v = subset[i] ?? 0
    for (let u = prev + 1; u < v; u++) rank += binomialExact(n - 1 - u, k - 1 - i)
    prev = v
  }
  return rank
}

/**
 * The k-subset of `0..n-1` with a given lexicographic rank (the inverse of {@link combinationRank}).
 *
 * @param n - The size of the ground set.
 * @param k - The subset size.
 * @param rank - The rank in `[0, C(n, k))`.
 * @returns The subset in increasing order.
 * @throws WhiteboxError with code `value` for a rank out of range.
 * @concept combination-ranking
 * @example
 * ```ts
 * import { combinationUnrank } from 'whitebox-engine/foundations'
 * combinationUnrank(5, 3, 9) // => [2, 3, 4]
 * ```
 */
export function combinationUnrank(n: number, k: number, rank: number | bigint): number[] {
  count(n, 'combinationUnrank')
  count(k, 'combinationUnrank')
  let r = BigInt(rank)
  if (r < 0n || r >= binomialExact(n, k)) throw new WhiteboxError('value', 'combinationUnrank: rank out of range')
  const out: number[] = []
  let u = 0
  for (let i = 0; i < k; i++) {
    for (; ; u++) {
      const block = binomialExact(n - 1 - u, k - 1 - i)
      if (r < block) break
      r -= block
    }
    out.push(u)
    u++
  }
  return out
}

/**
 * The number of derangements of n items (permutations with no fixed point), the subfactorial
 * `!n = (n - 1)(!(n - 1) + !(n - 2))`, close to `n! / e`.
 *
 * @param n - A non-negative integer.
 * @returns !n.
 * @concept derangements
 * @example
 * ```ts
 * import { derangements } from 'whitebox-engine/foundations'
 * [0, 1, 2, 3, 4, 5].map(derangements) // => [1n, 0n, 1n, 2n, 9n, 44n]
 * ```
 */
export function derangements(n: number): bigint {
  count(n, 'derangements')
  let [a, b] = [1n, 0n]
  if (n === 0) return 1n
  for (let k = 2; k <= n; k++) [a, b] = [b, BigInt(k - 1) * (a + b)]
  return b
}

/**
 * Stirling numbers of the second kind `S(n, k)`: the ways to partition n labelled items into k non-empty blocks,
 * `S(n, k) = k S(n-1, k) + S(n-1, k-1)`.
 *
 * @param n - Items.
 * @param k - Blocks.
 * @returns S(n, k).
 * @concept stirling-numbers
 * @example
 * ```ts
 * import { stirling2 } from 'whitebox-engine/foundations'
 * stirling2(5, 2) // => 15n
 * ```
 */
export function stirling2(n: number, k: number): bigint {
  count(n, 'stirling2')
  count(k, 'stirling2')
  let row = [1n]
  for (let m = 1; m <= n; m++) {
    const next = [0n]
    for (let j = 1; j <= m; j++) next.push(BigInt(j) * (row[j] ?? 0n) + (row[j - 1] ?? 0n))
    row = next
  }
  return row[k] ?? 0n
}

/**
 * Stirling numbers of the first kind: unsigned `c(n, k)` counts the permutations of n items with k cycles,
 * `c(n, k) = (n-1) c(n-1, k) + c(n-1, k-1)`; the signed `s(n, k) = (-1)^(n-k) c(n, k)` are the coefficients of the
 * falling factorial.
 *
 * @param n - Items.
 * @param k - Cycles.
 * @param signed - Return s(n, k) instead of c(n, k) (default false).
 * @returns The Stirling number.
 * @concept stirling-numbers
 * @example
 * ```ts
 * import { stirling1 } from 'whitebox-engine/foundations'
 * stirling1(5, 2) // => 50n
 * stirling1(5, 2, true) // => -50n
 * ```
 */
export function stirling1(n: number, k: number, signed = false): bigint {
  count(n, 'stirling1')
  count(k, 'stirling1')
  let row = [1n]
  for (let m = 1; m <= n; m++) {
    const next = [0n]
    for (let j = 1; j <= m; j++) next.push(BigInt(m - 1) * (row[j] ?? 0n) + (row[j - 1] ?? 0n))
    row = next
  }
  const c = row[k] ?? 0n
  return signed && (n - k) % 2 === 1 ? -c : c
}

/**
 * The Bell number `B(n)`: all partitions of n labelled items, `sum_k S(n, k)`, by the Bell triangle.
 *
 * @param n - A non-negative integer.
 * @returns B(n).
 * @concept bell-numbers
 * @example
 * ```ts
 * import { bell } from 'whitebox-engine/foundations'
 * [0, 1, 2, 3, 4, 5].map(bell) // => [1n, 1n, 2n, 5n, 15n, 52n]
 * ```
 */
export function bell(n: number): bigint {
  count(n, 'bell')
  let row = [1n]
  for (let m = 1; m <= n; m++) {
    const next = [row[row.length - 1] ?? 1n]
    for (const v of row) next.push((next[next.length - 1] ?? 0n) + v)
    row = next
  }
  return row[0] ?? 1n
}

/**
 * The Catalan number `C_n = C(2n, n) / (n + 1)`: balanced bracketings, binary trees with n internal nodes,
 * triangulations of an (n + 2)-gon, monotone lattice paths below the diagonal.
 *
 * @param n - A non-negative integer.
 * @returns C_n.
 * @concept catalan-numbers
 * @example
 * ```ts
 * import { catalan } from 'whitebox-engine/foundations'
 * [0, 1, 2, 3, 4, 5].map(catalan) // => [1n, 1n, 2n, 5n, 14n, 42n]
 * ```
 */
export function catalan(n: number): bigint {
  count(n, 'catalan')
  return binomialExact(2 * n, n) / BigInt(n + 1)
}

/**
 * The partition number `p(n)`: the ways to write n as a sum of positive integers regardless of order, by Euler's
 * pentagonal number recurrence `p(n) = sum_k (-1)^(k+1) (p(n - k(3k-1)/2) + p(n - k(3k+1)/2))`.
 *
 * @param n - A non-negative integer.
 * @returns p(n).
 * @concept integer-partitions
 * @example
 * ```ts
 * import { partitionCount } from 'whitebox-engine/foundations'
 * partitionCount(100) // => 190569292n
 * ```
 */
export function partitionCount(n: number): bigint {
  count(n, 'partitionCount')
  const p: bigint[] = [1n]
  for (let m = 1; m <= n; m++) {
    let s = 0n
    for (let k = 1; ; k++) {
      const g1 = (k * (3 * k - 1)) / 2
      if (g1 > m) break
      const sign = k % 2 === 1 ? 1n : -1n
      s += sign * (p[m - g1] ?? 0n)
      const g2 = (k * (3 * k + 1)) / 2
      if (g2 <= m) s += sign * (p[m - g2] ?? 0n)
    }
    p.push(s)
  }
  return p[n] ?? 0n
}

/**
 * All partitions of n into positive parts, each with its parts in non-increasing order, listed in reverse
 * lexicographic order (from `[n]` down to `[1, ..., 1]`); optionally only those with exactly k parts or with parts at
 * most a bound.
 *
 * @param n - A non-negative integer.
 * @param options - Exact number of parts, and the largest allowed part.
 * @returns The partitions.
 * @concept integer-partitions
 * @example
 * ```ts
 * import { integerPartitions } from 'whitebox-engine/foundations'
 * integerPartitions(4) // => [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
 * integerPartitions(6, { parts: 2 }) // => [[5, 1], [4, 2], [3, 3]]
 * ```
 */
export function integerPartitions(
  n: number,
  options: { readonly parts?: number; readonly maxPart?: number } = {},
): number[][] {
  count(n, 'integerPartitions')
  const out: number[][] = []
  const current: number[] = []
  const walk = (rest: number, max: number): void => {
    if (rest === 0) {
      if (options.parts === undefined || current.length === options.parts) out.push([...current])
      return
    }
    if (options.parts !== undefined && current.length >= options.parts) return
    for (let part = Math.min(rest, max); part >= 1; part--) {
      current.push(part)
      walk(rest - part, part)
      current.pop()
    }
  }
  walk(n, options.maxPart ?? n)
  return out
}

/** Result of {@link inclusionExclusion}. */
export interface InclusionExclusion {
  /** The size of the union. */
  readonly union: number
  /** `terms[k - 1]` is the sum of the sizes of all k-fold intersections. */
  readonly terms: number[]
  /** The running alternating sums: the Bonferroni bounds, alternately above and below the union. */
  readonly partialSums: number[]
}

/**
 * The inclusion-exclusion principle: `|A_1 u ... u A_m| = sum |A_i| - sum |A_i n A_j| + ...`, from finite sets
 * (as arrays of elements) or from a function giving the size of the intersection of any subfamily. The partial sums
 * are the Bonferroni inequalities.
 *
 * @param sets - The sets as arrays, or `{ count, intersection }` with `intersection(indices)` the size of the
 *   intersection of those sets.
 * @returns The union size, the k-fold intersection sums and the Bonferroni partial sums.
 * @concept inclusion-exclusion
 * @example
 * ```ts
 * import { inclusionExclusion } from 'whitebox-engine/foundations'
 * // Integers 1..100 divisible by 2, 3 or 5.
 * const d = [2, 3, 5]
 * inclusionExclusion({ count: 3, intersection: (ix) => Math.floor(100 / ix.reduce((p, i) => p * (d[i] ?? 1), 1)) }).union // => 74
 * inclusionExclusion([[1, 2, 3], [3, 4], [4, 5, 1]]).terms // => [8, 3, 0]
 * ```
 */
export function inclusionExclusion(
  sets:
    | readonly (readonly unknown[])[]
    | { readonly count: number; readonly intersection: (indices: readonly number[]) => number },
): InclusionExclusion {
  const m = Array.isArray(sets) ? sets.length : (sets as { count: number }).count
  if (m > 20) throw new WhiteboxError('value', 'inclusionExclusion: 2^m subfamilies; at most 20 sets')
  const size = Array.isArray(sets)
    ? (ix: readonly number[]): number => {
        const families = sets as readonly (readonly unknown[])[]
        const [first, ...rest] = ix.map((i) => new Set(families[i] ?? []))
        if (first === undefined) return 0
        let common = [...first]
        for (const s of rest) common = common.filter((x) => s.has(x))
        return common.length
      }
    : (sets as { intersection: (indices: readonly number[]) => number }).intersection
  const terms = Array.from({ length: m }, () => 0)
  for (let mask = 1; mask < 1 << m; mask++) {
    const ix: number[] = []
    for (let i = 0; i < m; i++) if (mask & (1 << i)) ix.push(i)
    terms[ix.length - 1] = (terms[ix.length - 1] ?? 0) + size(ix)
  }
  const partialSums: number[] = []
  let s = 0
  for (const [k, t] of terms.entries()) {
    s += k % 2 === 0 ? t : -t
    partialSums.push(s)
  }
  return { union: s, terms, partialSums }
}

/**
 * Rows `0..rows-1` of Pascal's triangle, `C(n, k)` for `0 <= k <= n`, optionally reduced modulo m (mod 2 draws the
 * Sierpinski triangle). Without a modulus the entries must stay exact, so rows are limited to 57.
 *
 * @param rows - How many rows.
 * @param modulo - An optional modulus.
 * @returns The rows.
 * @throws WhiteboxError with code `value` beyond row 57 without a modulus.
 * @concept pascals-triangle
 * @example
 * ```ts
 * import { pascalTriangle } from 'whitebox-engine/foundations'
 * pascalTriangle(5) // => [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
 * pascalTriangle(5, 2)[4] // => [1, 0, 0, 0, 1]
 * ```
 */
export function pascalTriangle(rows: number, modulo?: number): number[][] {
  count(rows, 'pascalTriangle')
  if (modulo === undefined && rows > 57)
    throw new WhiteboxError('value', 'pascalTriangle: entries beyond row 57 are not exact; give a modulus')
  if (modulo !== undefined && !(Number.isSafeInteger(modulo) && modulo >= 1))
    throw new WhiteboxError('value', 'pascalTriangle: the modulus must be a positive integer')
  const one = modulo === undefined ? 1 : 1 % modulo
  const out: number[][] = []
  let row = [one]
  for (let n = 0; n < rows; n++) {
    out.push(row)
    const next = [one]
    for (let k = 1; k < row.length; k++) {
      const v = (row[k - 1] ?? 0) + (row[k] ?? 0)
      next.push(modulo === undefined ? v : v % modulo)
    }
    next.push(one)
    row = next
  }
  return out
}
