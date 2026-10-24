import { meanOf, pairwiseSum, varianceOf, WhiteboxError } from '../base/index.ts'
import { nextAfter } from '../numerics/approximation/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { kdTree } from '../numerics/spatial/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { countsOf, entropyOfCounts, float, type Labels, labelCodes, logOfBase } from './_shared.ts'
import { asBytes, type ByteInput, deflate, gzipCompress, zlibCompress } from './deflate.ts'

/** A compressor for the compression distance: a named format, or any function giving the compressed length. */
export type Compressor = 'zlib' | 'gzip' | 'deflate' | ((bytes: Uint8Array) => number)

function compressedLength(bytes: Uint8Array, compressor: Compressor, level: number): number {
  if (typeof compressor === 'function') return compressor(bytes)
  if (compressor === 'gzip') return gzipCompress(bytes, { level }).length
  if (compressor === 'deflate') return deflate(bytes, { level }).length
  return zlibCompress(bytes, { level }).length
}

/** Options of the compression distances. */
export interface CompressionOptions {
  /** The compressor (default `'zlib'`). */
  readonly compressor?: Compressor
  /** Compression level of the named compressors (default 9). */
  readonly level?: number
}

/**
 * The normalised compression distance (Cilibrasi and Vitanyi 2005), a computable stand-in for the normalised
 * information distance: NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y)), with C the compressed length. Near 0
 * for objects that share most of their information, near 1 (or slightly above) for unrelated ones.
 *
 * @param x - The first object (bytes, byte values or a string).
 * @param y - The second.
 * @param options - Compressor and level.
 * @returns The distance.
 * @concept compression-distance
 * @example
 * ```ts
 * import { normalisedCompressionDistance } from 'whitebox-engine/information'
 * const a = 'the quick brown fox jumps over the lazy dog. '.repeat(8)
 * normalisedCompressionDistance(a, a) < 0.1 // => true
 * ```
 */
export function normalisedCompressionDistance(x: ByteInput, y: ByteInput, options: CompressionOptions = {}): number {
  const a = asBytes(x)
  const b = asBytes(y)
  const compressor = options.compressor ?? 'zlib'
  const level = options.level ?? 9
  const ab = new Uint8Array(a.length + b.length)
  ab.set(a)
  ab.set(b, a.length)
  const ca = compressedLength(a, compressor, level)
  const cb = compressedLength(b, compressor, level)
  return (compressedLength(ab, compressor, level) - Math.min(ca, cb)) / Math.max(ca, cb)
}

/**
 * The matrix of normalised compression distances between items (each item compressed once), for clustering or k-NN
 * classification by compression.
 *
 * @param items - The objects.
 * @param options - Compressor and level.
 * @returns The `[n, n]` distances (zero on the diagonal).
 * @concept compression-distance
 * @example
 * ```ts
 * import { compressionDistanceMatrix } from 'whitebox-engine/information'
 * compressionDistanceMatrix(['abcabcabc', 'abcabcabc'])[0][0] // => 0
 * ```
 */
export function compressionDistanceMatrix(items: readonly ByteInput[], options: CompressionOptions = {}): number[][] {
  const bytes = items.map(asBytes)
  const compressor = options.compressor ?? 'zlib'
  const level = options.level ?? 9
  const single = bytes.map((b) => compressedLength(b, compressor, level))
  return bytes.map((a, i) =>
    bytes.map((b, j) => {
      if (i === j) return 0
      const ab = new Uint8Array(a.length + b.length)
      ab.set(a)
      ab.set(b, a.length)
      const ca = single[i] ?? 0
      const cb = single[j] ?? 0
      return (compressedLength(ab, compressor, level) - Math.min(ca, cb)) / Math.max(ca, cb)
    }),
  )
}

/**
 * The Lempel-Ziv (1976) complexity of a sequence: the number of new phrases met when the sequence is parsed from left
 * to right, each phrase the shortest one that has not occurred before (the Kaspar-Schuster algorithm, as antropy's
 * `lziv_complexity`). Normalised, c / (n / log_b n) with b the number of distinct symbols, it estimates the entropy
 * rate in base-b units for ergodic sources.
 *
 * @param sequence - The symbols (a string is taken character by character).
 * @param options - `normalise` (default false).
 * @returns The complexity (an integer unless normalised).
 * @concept lempel-ziv-complexity
 * @example
 * ```ts
 * import { lempelZivComplexity } from 'whitebox-engine/information'
 * // Phrases 1 / 0 / 01 / 1110 / 1100 / 0010.
 * lempelZivComplexity('1001111011000010') // => 6
 * lempelZivComplexity('1001111011000010', { normalise: true }) // => 1.5
 * ```
 */
export function lempelZivComplexity(sequence: string | Labels, options: { readonly normalise?: boolean } = {}): number {
  const s = typeof sequence === 'string' ? Array.from(sequence, (c) => c.codePointAt(0) ?? 0) : sequence
  const { codes, m } = labelCodes(s as Labels, 'lempelZivComplexity')
  const n = codes.length
  let complexity = 1
  let prefix = 1
  let length = 1
  let longest = 1
  let pointer = 0
  while (prefix + length <= n) {
    if (codes[pointer + length - 1] === codes[prefix + length - 1]) length++
    else {
      longest = Math.max(length, longest)
      pointer++
      if (pointer === prefix) {
        complexity++
        prefix += longest
        pointer = 0
        longest = 1
      }
      length = 1
    }
  }
  if (length !== 1) complexity++
  if (options.normalise !== true) return complexity
  const base = Math.max(2, m)
  return complexity / (n / (Math.log(n) / Math.log(base)))
}

/**
 * The plug-in block entropy H_L: the entropy of the empirical distribution of the overlapping blocks of L consecutive
 * symbols.
 *
 * @param sequence - The symbols.
 * @param length - The block length L >= 1.
 * @param options - The base (default e).
 * @returns H_L.
 * @concept entropy-rate
 * @example
 * ```ts
 * import { blockEntropy } from 'whitebox-engine/information'
 * blockEntropy([0, 1, 0, 1, 0, 1, 0, 1, 0], 2, { base: 2 }) // => 1
 * ```
 */
export function blockEntropy(sequence: Labels, length: number, options: { readonly base?: number } = {}): number {
  const { codes } = labelCodes(sequence, 'blockEntropy')
  if (!Number.isInteger(length) || length < 1 || length > codes.length)
    throw new WhiteboxError('value', 'blockEntropy: the block length must lie in [1, n]')
  const blocks = Array.from({ length: codes.length - length + 1 }, (_, i) =>
    Array.from(codes.subarray(i, i + length)).join(','),
  )
  const { codes: blockCodes, m } = labelCodes(blocks, 'blockEntropy')
  return entropyOfCounts(countsOf(blockCodes, m)) / logOfBase(options.base)
}

/**
 * The plug-in entropy-rate estimate H(X_(k+1) | X_1 ... X_k): the conditional entropy of the symbol that follows each
 * context of k symbols, from the counts of the (k + 1)-blocks and of their k-symbol prefixes (so it is never negative
 * and is zero for a sequence whose next symbol is determined by the last k). It decreases towards the entropy rate as k
 * grows, until the sample runs out.
 *
 * @param sequence - The symbols.
 * @param options - The context length k (default 1) and the base.
 * @returns The estimate.
 * @concept entropy-rate
 * @example
 * ```ts
 * import { entropyRate } from 'whitebox-engine/information'
 * // A periodic sequence is fully predictable from its last symbol.
 * entropyRate([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]) // => 0
 * ```
 */
export function entropyRate(
  sequence: Labels,
  options: { readonly order?: number; readonly base?: number } = {},
): number {
  const k = options.order ?? 1
  const { codes } = labelCodes(sequence, 'entropyRate')
  if (!Number.isInteger(k) || k < 0 || k >= codes.length)
    throw new WhiteboxError('value', 'entropyRate: order must be an integer in [0, n)')
  const starts = codes.length - k
  const key = (i: number, l: number): string => Array.from(codes.subarray(i, i + l)).join(',')
  const h = (l: number): number => {
    const { codes: c, m } = labelCodes(
      Array.from({ length: starts }, (_, i) => key(i, l)),
      'entropyRate',
    )
    return entropyOfCounts(countsOf(c, m))
  }
  return (h(k + 1) - (k === 0 ? 0 : h(k))) / logOfBase(options.base)
}

/**
 * The entropy rate of a stationary Markov chain, -sum_i pi_i sum_j P_ij log P_ij, with pi the stationary
 * distribution (solved from pi P = pi, sum pi = 1).
 *
 * @param transition - The transition matrix `[states, states]` (rows sum to one; the chain must be irreducible).
 * @param options - The base (default e).
 * @returns The entropy rate.
 * @concept entropy-rate
 * @example
 * ```ts
 * import { markovEntropyRate } from 'whitebox-engine/information'
 * markovEntropyRate([[0.5, 0.5], [0.5, 0.5]], { base: 2 }) // => 1
 * ```
 */
export function markovEntropyRate(transition: TensorLike, options: { readonly base?: number } = {}): number {
  const v = float(transition)
  const n = v.shape[0] ?? 0
  if (v.shape.length !== 2 || v.shape[1] !== n || n === 0)
    throw new WhiteboxError('shape', 'markovEntropyRate: expected a square transition matrix')
  const p = toFlat(v)
  for (let i = 0; i < n; i++) {
    const row = p.slice(i * n, (i + 1) * n)
    if (row.some((x) => !(x >= 0)) || Math.abs(pairwiseSum(row) - 1) > 1e-9)
      throw new WhiteboxError('value', `markovEntropyRate: row ${i} is not a distribution`)
  }
  // (P^T - I) pi = 0 with the last equation replaced by sum pi = 1.
  const a = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === n - 1 ? 1 : (p[j * n + i] ?? 0) - (i === j ? 1 : 0))),
  )
  const b = Array.from({ length: n }, (_, i) => (i === n - 1 ? 1 : 0))
  const pi = toFlat(solve(a, b))
  const terms: number[] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const pij = p[i * n + j] ?? 0
      if (pij > 0) terms.push(-(pi[i] ?? 0) * pij * Math.log(pij))
    }
  return pairwiseSum(terms) / logOfBase(options.base)
}

/** Options of approximate and sample entropy. */
export interface TemplateEntropyOptions {
  /** The embedding dimension m (default 2). */
  readonly order?: number
  /** The tolerance r (default 0.2 times the population standard deviation, as antropy). */
  readonly tolerance?: number
}

/** The m-dimensional delay vectors of x: rows [x_i, ..., x_(i+m-1)], the first `count` of them. */
function embed(x: readonly number[], m: number, count: number): number[][] {
  return Array.from({ length: count }, (_, i) => x.slice(i, i + m))
}

function templateSetup(
  x: TensorLike,
  options: TemplateEntropyOptions,
  name: string,
): { xs: number[]; m: number; r: number } {
  const xs = toFlat(float(x))
  const m = options.order ?? 2
  if (!Number.isInteger(m) || m < 1 || xs.length < m + 2)
    throw new WhiteboxError('value', `${name}: need an integer order >= 1 and at least order + 2 samples`)
  const r = options.tolerance ?? 0.2 * Math.sqrt(varianceOf(xs))
  if (!(r >= 0)) throw new WhiteboxError('value', `${name}: the tolerance must be non-negative`)
  return { xs, m, r }
}

/**
 * Pincus's approximate entropy ApEn(m, r) = phi_m - phi_(m+1), with phi_m the average log fraction of length-m
 * templates within max-norm distance r of each template (self-matches included, so the log is defined): a regularity
 * statistic, low for predictable series. Neighbours are counted with a KD tree, as antropy's `app_entropy`.
 *
 * @param x - The series.
 * @param options - Order m and tolerance r.
 * @returns ApEn in nats.
 * @concept approximate-entropy
 * @example
 * ```ts
 * import { approximateEntropy } from 'whitebox-engine/information'
 * // A period-two series is perfectly regular.
 * Math.abs(approximateEntropy([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])) < 0.05 // => true
 * ```
 */
export function approximateEntropy(x: TensorLike, options: TemplateEntropyOptions = {}): number {
  const { xs, m, r } = templateSetup(x, options, 'approximateEntropy')
  const phi = (dim: number): number => {
    const count = xs.length - dim + 1
    const t = embed(xs, dim, count)
    const counts = kdTree(t, { metric: 'chebyshev' }).countNeighbours(t, r)
    return meanOf(counts.map((c) => Math.log(c / count)))
  }
  return phi(m) - phi(m + 1)
}

/**
 * Richman and Moorman's sample entropy SampEn(m, r) = -log(A / B), with B the number of pairs of length-m templates
 * closer than r in the max norm and A the number of those pairs that still match at length m + 1 (self-matches
 * excluded; the same N - m templates for both, as antropy's `sample_entropy`). Infinity when no pair matches at
 * length m + 1; NaN when none matches at length m.
 *
 * @param x - The series.
 * @param options - Order m and tolerance r.
 * @returns SampEn in nats.
 * @concept sample-entropy
 * @example
 * ```ts
 * import { sampleEntropy } from 'whitebox-engine/information'
 * sampleEntropy([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]) // => 0
 * ```
 */
export function sampleEntropy(x: TensorLike, options: TemplateEntropyOptions = {}): number {
  const { xs, m, r } = templateSetup(x, options, 'sampleEntropy')
  const count = xs.length - m
  const strict = r > 0 ? nextAfter(r, 0) : -1
  const pairs = (dim: number): number => {
    if (strict < 0) return 0
    const t = embed(xs, dim, count)
    const counts = kdTree(t, { metric: 'chebyshev' }).countNeighbours(t, strict)
    return (pairwiseSum(counts) - count) / 2
  }
  const b = pairs(m)
  const a = pairs(m + 1)
  if (b === 0) return Number.NaN
  if (a === 0) return Number.POSITIVE_INFINITY
  return -Math.log(a / b)
}
