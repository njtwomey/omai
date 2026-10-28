/**
 * Descriptor matching, OpenCV's BFMatcher: brute-force nearest neighbours under the L2, L1 or Hamming distance, the
 * cross-check that keeps only mutual nearest neighbours, the k nearest matches of each query, and Lowe's ratio test
 * on the two nearest.
 */

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { valuesOf } from './_image.ts'

/** A match, OpenCV's cv::DMatch: the query and train descriptor indices and their distance. */
export interface DMatch {
  readonly queryIdx: number
  readonly trainIdx: number
  readonly distance: number
}

/** The distances of the matcher: Euclidean, city-block, or Hamming on bytes. */
export type MatchNorm = 'l2' | 'l1' | 'hamming'

/** Descriptors as rows: float rows, or byte strings for Hamming. */
export type DescriptorsLike = TensorLike | readonly ArrayLike<number>[]

function rowsOf(d: DescriptorsLike): ArrayLike<number>[] {
  if (Array.isArray(d)) return d as ArrayLike<number>[]
  const { data, shape } = valuesOf(d as TensorLike)
  const n = shape[0] ?? 0
  const m = shape[1] ?? 0
  return Array.from({ length: n }, (_, i) => data.subarray(i * m, (i + 1) * m))
}

const POPCOUNT = Uint8Array.from({ length: 256 }, (_, i) => {
  let c = 0
  for (let v = i; v > 0; v >>= 1) c += v & 1
  return c
})

/**
 * The distance between two descriptors: Euclidean, city-block, or Hamming (the number of differing bits of byte
 * strings).
 *
 * @param a - First descriptor.
 * @param b - Second descriptor (same length).
 * @param norm - The distance (default 'l2').
 * @returns The distance.
 * @concept descriptor-matching
 * @example
 * ```ts
 * import { descriptorDistance } from 'mdml-engine/vision'
 * descriptorDistance([0b1010], [0b0110], 'hamming') // => 2
 * ```
 */
export function descriptorDistance(a: ArrayLike<number>, b: ArrayLike<number>, norm: MatchNorm = 'l2'): number {
  if (a.length !== b.length) throw new MdmlError('shape', 'descriptorDistance: descriptors differ in length')
  let s = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0
    const y = b[i] ?? 0
    if (norm === 'hamming') s += POPCOUNT[(x ^ y) & 255] ?? 0
    else if (norm === 'l1') s += Math.abs(x - y)
    else s += (x - y) * (x - y)
  }
  return norm === 'l2' ? Math.sqrt(s) : s
}

/**
 * The k nearest train descriptors of every query (cv2.BFMatcher.knnMatch): brute force, nearest first, ties to the
 * lower train index.
 *
 * @param query - Query descriptors `[n, d]` (or byte strings).
 * @param train - Train descriptors `[m, d]`.
 * @param k - Neighbours per query.
 * @param norm - The distance (default 'l2').
 * @returns Per query, up to k matches.
 * @concept descriptor-matching
 * @example
 * ```ts
 * import { knnMatch } from 'mdml-engine/vision'
 * knnMatch([[0, 0]], [[3, 4], [1, 0], [0, 2]], 2)[0].map((m) => m.trainIdx) // => [1, 2]
 * ```
 */
export function knnMatch(
  query: DescriptorsLike,
  train: DescriptorsLike,
  k: number,
  norm: MatchNorm = 'l2',
): DMatch[][] {
  const q = rowsOf(query)
  const t = rowsOf(train)
  return q.map((a, qi) => {
    const all = t.map((b, ti) => ({ queryIdx: qi, trainIdx: ti, distance: descriptorDistance(a, b, norm) }))
    all.sort((m1, m2) => m1.distance - m2.distance || m1.trainIdx - m2.trainIdx)
    return all.slice(0, k)
  })
}

/**
 * Brute-force matching (cv2.BFMatcher.match): each query's nearest train descriptor; with crossCheck, only pairs that
 * are each other's nearest neighbour (cv2.BFMatcher(crossCheck=True)).
 *
 * @param query - Query descriptors `[n, d]`.
 * @param train - Train descriptors `[m, d]`.
 * @param options - The distance (default 'l2') and crossCheck (default false).
 * @returns The matches, by query index.
 * @concept descriptor-matching
 * @example
 * ```ts
 * import { matchDescriptors } from 'mdml-engine/vision'
 * matchDescriptors([[0, 0], [5, 5]], [[5, 4], [1, 0]]).map((m) => m.trainIdx) // => [1, 0]
 * ```
 */
export function matchDescriptors(
  query: DescriptorsLike,
  train: DescriptorsLike,
  options: { readonly norm?: MatchNorm; readonly crossCheck?: boolean } = {},
): DMatch[] {
  const norm = options.norm ?? 'l2'
  const forward = knnMatch(query, train, 1, norm).flatMap((m) => m)
  if (options.crossCheck !== true) return forward
  const backward = knnMatch(train, query, 1, norm).flatMap((m) => m)
  return forward.filter((m) => backward[m.trainIdx]?.trainIdx === m.queryIdx)
}

/**
 * Lowe's ratio test (Lowe 2004): keep a query's nearest match when it is closer than ratio times the second nearest.
 *
 * @param query - Query descriptors `[n, d]`.
 * @param train - Train descriptors `[m, d]` (at least two).
 * @param options - The ratio (default 0.75) and distance (default 'l2').
 * @returns The distinctive matches.
 * @concept descriptor-matching
 * @example
 * ```ts
 * import { ratioTestMatch } from 'mdml-engine/vision'
 * ratioTestMatch([[0, 0], [5, 5]], [[0, 0.1], [5, 4.9], [5, 5.1]]).map((m) => m.queryIdx) // => [0]
 * ```
 */
export function ratioTestMatch(
  query: DescriptorsLike,
  train: DescriptorsLike,
  options: { readonly ratio?: number; readonly norm?: MatchNorm } = {},
): DMatch[] {
  const ratio = options.ratio ?? 0.75
  return knnMatch(query, train, 2, options.norm ?? 'l2').flatMap((ms) => {
    const [a, b] = ms
    if (a === undefined) return []
    if (b === undefined || a.distance < ratio * b.distance) return [a]
    return []
  })
}
