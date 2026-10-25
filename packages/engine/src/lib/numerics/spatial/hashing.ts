/**
 * Locality-sensitive hashing, where similar points are likely to share a hash bucket: random-hyperplane hashing for
 * cosine similarity, p-stable hashing for Lp distances, MinHash signatures with banding for Jaccard similarity of sets,
 * collision and banding probabilities (the S-curve), multi-probe sequences, and an E2LSH-style index that reports query
 * statistics.
 */

import { heap, MdmlError } from '../../base/index.ts'
import { cauchy, foldIn, type Key, normal, randint, uniform } from '../../random/index.ts'
import { concreteOrThrow, item, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Metric, metricFunction } from '../geometry/index.ts'
import { ndtr } from '../special/index.ts'

/**
 * A family of locality-sensitive hash functions: k functions per table, each mapping a point to an integer, with the
 * margin to the nearest bucket boundary in each direction (for multi-probe search).
 */
export interface HashFamily {
  readonly name: 'hyperplane' | 'pstable'
  readonly dim: number
  readonly k: number
  /** The k hash values of x (offset into a flat array). */
  readonly hash: (x: ArrayLike<number>, offset?: number) => number[]
  /**
   * Perturbations of the hash values with their squared distances to the boundary crossed, as (function, shift, score):
   * the probes worth trying first are those with the smallest score.
   */
  readonly perturbations: (
    x: ArrayLike<number>,
    offset?: number,
  ) => { readonly fn: number; readonly shift: number; readonly score: number }[]
}

/**
 * Random-hyperplane hashing for angular (cosine) similarity (Charikar 2002): h_i(x) = [r_i . x >= 0] with Gaussian r_i;
 * two vectors at angle theta collide on one bit with probability 1 - theta / pi.
 *
 * @param key - Random key.
 * @param dim - Dimension.
 * @param k - Bits per table.
 * @returns The family.
 * @concept locality-sensitive-hashing
 * @example
 * ```ts
 * import { randomHyperplanes } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * randomHyperplanes(key(0), 3, 4).hash([1, 2, 3]).length // => 4
 * ```
 */
export function randomHyperplanes(key: Key, dim: number, k: number): HashFamily {
  const r = Array.from(normal(key, [k, dim]).data)
  const project = (x: ArrayLike<number>, offset: number): number[] =>
    Array.from({ length: k }, (_, i) => {
      let s = 0
      for (let j = 0; j < dim; j++) s += (r[i * dim + j] ?? 0) * (x[offset + j] ?? 0)
      return s
    })
  return {
    name: 'hyperplane',
    dim,
    k,
    hash: (x, offset = 0) => project(x, offset).map((v) => (v >= 0 ? 1 : 0)),
    perturbations: (x, offset = 0) => project(x, offset).map((v, fn) => ({ fn, shift: v >= 0 ? -1 : 1, score: v * v })),
  }
}

/**
 * p-stable hashing for Lp distances (Datar, Immorlica, Indyk and Mirrokni 2004): h_i(x) = floor((a_i . x + b_i) / w) with
 * a_i Gaussian (p = 2) or Cauchy (p = 1) and b_i uniform in [0, w).
 *
 * @param key - Random key.
 * @param dim - Dimension.
 * @param k - Functions per table.
 * @param options - Bucket width w (default 4) and p (1 or 2, default 2).
 * @returns The family.
 * @example
 * ```ts
 * import { pStableHash } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * pStableHash(key(1), 2, 3, { w: 1 }).hash([0.5, -0.5]).length // => 3
 * ```
 */
export function pStableHash(
  key: Key,
  dim: number,
  k: number,
  options: { readonly w?: number; readonly p?: 1 | 2 } = {},
): HashFamily {
  const w = options.w ?? 4
  const a = Array.from((options.p === 1 ? cauchy(foldIn(key, 0), [k, dim]) : normal(foldIn(key, 0), [k, dim])).data)
  const b = Array.from(uniform(foldIn(key, 1), [k], { high: w }).data)
  const project = (x: ArrayLike<number>, offset: number): number[] =>
    Array.from({ length: k }, (_, i) => {
      let s = b[i] ?? 0
      for (let j = 0; j < dim; j++) s += (a[i * dim + j] ?? 0) * (x[offset + j] ?? 0)
      return s / w
    })
  return {
    name: 'pstable',
    dim,
    k,
    hash: (x, offset = 0) => project(x, offset).map(Math.floor),
    perturbations: (x, offset = 0) =>
      project(x, offset).flatMap((v, fn) => {
        const frac = v - Math.floor(v)
        // Squared distances (in units of w) to the lower and upper boundaries of the bucket.
        return [
          { fn, shift: -1, score: frac * frac },
          { fn, shift: 1, score: (1 - frac) * (1 - frac) },
        ]
      }),
  }
}

/**
 * The probability that two points collide under one function of a family: for hyperplanes at angle theta,
 * 1 - theta / pi; for p-stable hashing at distance c with width w and u = w / c (Datar et al.), Gaussian (p = 2):
 * 1 - 2 Phi(-u) - 2 / (sqrt(2 pi) u) (1 - exp(-u^2 / 2)); Cauchy (p = 1): 2 atan(u) / pi - log(1 + u^2) / (pi u).
 *
 * @param family - `hyperplane` or `pstable`.
 * @param distance - Angle (hyperplane) or distance (p-stable).
 * @param options - Width w and p for p-stable families.
 * @returns The probability.
 * @example
 * ```ts
 * import { collisionProbability } from 'mdml-engine/numerics/spatial'
 * collisionProbability('hyperplane', Math.PI / 2) // => 0.5
 * ```
 */
export function collisionProbability(
  family: 'hyperplane' | 'pstable',
  distance: number,
  options: { readonly w?: number; readonly p?: 1 | 2 } = {},
): number {
  if (family === 'hyperplane') return 1 - distance / Math.PI
  if (distance === 0) return 1
  const u = (options.w ?? 4) / distance
  if (options.p === 1) return (2 * Math.atan(u)) / Math.PI - Math.log1p(u * u) / (Math.PI * u)
  return 1 - 2 * item(ndtr(-u)) - (2 / (Math.sqrt(2 * Math.PI) * u)) * (1 - Math.exp((-u * u) / 2))
}

/**
 * The probability that two items become candidates under banding with b bands of r rows when one hash collides with
 * probability s: 1 - (1 - s^r)^b, the S-curve whose threshold is about (1 / b)^(1 / r).
 *
 * @param s - Single-function collision probability (Jaccard similarity for MinHash).
 * @param r - Rows (functions) per band.
 * @param b - Bands (tables).
 * @returns The candidate probability.
 * @example
 * ```ts
 * import { bandingProbability } from 'mdml-engine/numerics/spatial'
 * Math.round(bandingProbability(0.8, 5, 20) * 1000) / 1000 // => 1
 * ```
 */
export function bandingProbability(s: number, r: number, b: number): number {
  return 1 - (1 - s ** r) ** b
}

const MERSENNE = 2147483647

/**
 * MinHash signatures of sets of non-negative integer tokens (Broder 1997): for each of n universal hash functions
 * h(x) = (a x + b) mod p, the minimum over the set. The fraction of agreeing positions estimates the Jaccard
 * similarity.
 *
 * @param sets - Token sets.
 * @param key - Random key.
 * @param n - Signature length (default 128).
 * @returns One signature per set.
 * @concept minhash
 * @example
 * ```ts
 * import { estimateJaccard, minHash } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * const [a, b] = minHash([[1, 2, 3, 4], [1, 2, 3, 5]], key(0), 256)
 * Math.abs(estimateJaccard(a, b) - 0.6) < 0.15 // => true
 * ```
 */
export function minHash(sets: readonly (readonly number[])[], key: Key, n = 128): number[][] {
  const a = Array.from(randint(foldIn(key, 0), [n], 1, MERSENNE).data)
  const b = Array.from(randint(foldIn(key, 1), [n], 0, MERSENNE).data)
  // (a x + b) mod p without overflowing doubles: split a into 16-bit halves.
  const mulmod = (x: number, y: number): number => {
    const hi = Math.floor(y / 65536)
    const lo = y % 65536
    return (((x * hi) % MERSENNE) * 65536 + x * lo) % MERSENNE
  }
  return sets.map((set) =>
    Array.from({ length: n }, (_, i) => {
      let m = Number.POSITIVE_INFINITY
      for (const token of set) {
        const v = (mulmod(token % MERSENNE, a[i] ?? 1) + (b[i] ?? 0)) % MERSENNE
        if (v < m) m = v
      }
      return m
    }),
  )
}

/**
 * The Jaccard similarity estimated from two MinHash signatures: the fraction of equal positions.
 *
 * @param a - Signature.
 * @param b - Signature of the same length.
 * @returns The estimate.
 * @example
 * ```ts
 * import { estimateJaccard } from 'mdml-engine/numerics/spatial'
 * estimateJaccard([1, 2, 3, 4], [1, 2, 0, 4]) // => 0.75
 * ```
 */
export function estimateJaccard(a: readonly number[], b: readonly number[]): number {
  let same = 0
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++
  return a.length === 0 ? 0 : same / a.length
}

/**
 * Candidate pairs from MinHash signatures by banding: split each signature into b bands of r rows; items sharing a band
 * in any band are candidates.
 *
 * @param signatures - MinHash signatures (length at least b r).
 * @param bands - b.
 * @param rows - r.
 * @returns Candidate pairs (i < j), sorted.
 * @example
 * ```ts
 * import { minHashBanding } from 'mdml-engine/numerics/spatial'
 * minHashBanding([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 7, 7]], 2, 2) // => [[0, 1]]
 * ```
 */
export function minHashBanding(
  signatures: readonly (readonly number[])[],
  bands: number,
  rows: number,
): [number, number][] {
  const pairs = new Set<string>()
  for (let band = 0; band < bands; band++) {
    const buckets = new Map<string, number[]>()
    signatures.forEach((sig, i) => {
      const k = sig.slice(band * rows, band * rows + rows).join(',')
      const list = buckets.get(k) ?? []
      list.push(i)
      buckets.set(k, list)
    })
    for (const list of buckets.values()) {
      for (let x = 0; x < list.length; x++) for (let y = x + 1; y < list.length; y++) pairs.add(`${list[x]},${list[y]}`)
    }
  }
  return [...pairs].map((p) => p.split(',').map(Number) as [number, number]).sort((p, q) => p[0] - q[0] || p[1] - q[1])
}

/** Statistics of an approximate query. */
export interface ApproximateQueryStats {
  readonly candidates: number
  readonly probes: number
  readonly distanceEvaluations: number
}

/** An LSH index over points. */
export interface LshIndex {
  readonly tables: number
  /** The k nearest among the candidates (re-ranked exactly), with statistics. */
  readonly query: (
    q: ArrayLike<number>,
    k: number,
    options?: { readonly probes?: number },
  ) => { readonly indices: number[]; readonly distances: number[]; readonly stats: ApproximateQueryStats }
}

/**
 * Probe sequences for multi-probe LSH (Lv, Josephson, Wang, Charikar and Li 2007): perturbation sets of the hash values
 * in increasing order of total score, generated with a heap by the shift and expand operations over the sorted
 * single-coordinate perturbations; each set changes a coordinate at most once.
 *
 * @param perturbations - Candidate single perturbations with scores.
 * @param count - Number of perturbation sets wanted (excluding the empty set).
 * @returns Sets of indices into the given perturbation list, cheapest first.
 * @example
 * ```ts
 * import { probeSequence } from 'mdml-engine/numerics/spatial'
 * probeSequence([{ fn: 0, score: 0.1 }, { fn: 1, score: 0.2 }, { fn: 0, score: 0.9 }], 3) // => [[0], [1], [0, 1]]
 * ```
 */
export function probeSequence(
  perturbations: readonly { readonly fn: number; readonly score: number }[],
  count: number,
): number[][] {
  const sorted = perturbations.map((p, i) => ({ ...p, i })).sort((a, b) => a.score - b.score)
  const score = (set: readonly number[]): number => set.reduce((s, j) => s + (sorted[j]?.score ?? 0), 0)
  const valid = (set: readonly number[]): boolean => {
    const fns = new Set<number>()
    for (const j of set) {
      const fn = sorted[j]?.fn ?? -1
      if (fns.has(fn)) return false
      fns.add(fn)
    }
    return true
  }
  const out: number[][] = []
  const queue = heap((a: number[], b: number[]) => score(a) - score(b), sorted.length > 0 ? [[0]] : [])
  while (queue.size > 0 && out.length < count) {
    const set = queue.pop() as number[]
    const last = set[set.length - 1] ?? 0
    if (last + 1 < sorted.length) {
      queue.push([...set.slice(0, -1), last + 1])
      queue.push([...set, last + 1])
    }
    if (valid(set)) out.push(set.map((j) => sorted[j]?.i ?? 0))
  }
  return out
}

/**
 * A locality-sensitive hashing index (E2LSH style): L tables of k concatenated hash functions; a query gathers the
 * points sharing its bucket in any table (and, with multi-probe, in nearby buckets), then re-ranks them exactly.
 *
 * @param data - Points `[n, d]`.
 * @param options - Family (hyperplane for cosine, pstable for Lp), tables L, functions per table k, width w, p, metric
 *   for re-ranking (cosine or euclidean by default) and key.
 * @returns The index.
 * @example
 * ```ts
 * import { lshIndex } from 'mdml-engine/numerics/spatial'
 * import { key } from 'mdml-engine/random'
 * const index = lshIndex([[0, 0], [0.1, 0], [5, 5]], { family: 'pstable', tables: 4, k: 2, w: 2, key: key(0) })
 * index.query([0.05, 0], 1).indices.length // => 1
 * ```
 */
export function lshIndex(
  data: TensorLike,
  options: {
    readonly family?: 'hyperplane' | 'pstable'
    readonly tables?: number
    readonly k?: number
    readonly w?: number
    readonly p?: 1 | 2
    readonly metric?: Metric
    readonly key: Key
  },
): LshIndex {
  const t = concreteOrThrow(promoteFloat(data)[0] as Tensor, 'lshIndex')
  const flat = toFlat(t)
  const [n = 0, d = 0] = t.shape
  const familyName = options.family ?? 'pstable'
  const L = options.tables ?? 8
  const k = options.k ?? 8
  const metric =
    options.metric ?? (familyName === 'hyperplane' ? 'cosine' : options.p === 1 ? 'cityblock' : 'euclidean')
  const distance = metricFunction(metric, d)
  const families = Array.from({ length: L }, (_, l) =>
    familyName === 'hyperplane'
      ? randomHyperplanes(foldIn(options.key, l), d, k)
      : pStableHash(foldIn(options.key, l), d, k, {
          ...(options.w === undefined ? {} : { w: options.w }),
          ...(options.p === undefined ? {} : { p: options.p }),
        }),
  )
  const tables = families.map((family) => {
    const buckets = new Map<string, number[]>()
    for (let i = 0; i < n; i++) {
      const code = family.hash(flat, i * d).join(',')
      const list = buckets.get(code) ?? []
      list.push(i)
      buckets.set(code, list)
    }
    return buckets
  })
  return {
    tables: L,
    query: (q, kNearest, queryOptions = {}) => {
      if (q.length !== d) throw new MdmlError('shape', 'lshIndex: query dimension mismatch')
      const probes = queryOptions.probes ?? 0
      const candidates = new Set<number>()
      let probesUsed = 0
      families.forEach((family, l) => {
        const base = family.hash(q)
        const codes = [base]
        if (probes > 0) {
          const perts = family.perturbations(q)
          for (const set of probeSequence(perts, probes)) {
            const code = [...base]
            for (const j of set) {
              const p = perts[j]
              if (p === undefined) continue
              code[p.fn] = family.name === 'hyperplane' ? 1 - (code[p.fn] ?? 0) : (code[p.fn] ?? 0) + p.shift
            }
            codes.push(code)
          }
        }
        for (const code of codes) {
          probesUsed++
          for (const i of tables[l]?.get(code.join(',')) ?? []) candidates.add(i)
        }
      })
      const ranked = [...candidates]
        .map((i) => [distance(q, 0, flat, i * d), i] as const)
        .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      const top = ranked.slice(0, kNearest)
      return {
        indices: top.map((r) => r[1]),
        distances: top.map((r) => r[0]),
        stats: { candidates: candidates.size, probes: probesUsed, distanceEvaluations: candidates.size },
      }
    },
  }
}
