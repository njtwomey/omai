/**
 * Relabelling the components of an EP fixed point: the permutations of a set of categories, the same permutation applied to the
 * discrete messages that start a run and to the records of every site that is indexed by the components (a vector of rates, a
 * probability vector over the labels, a matrix with a row and a column per component), and the canonical form of a fixed point,
 * its components sorted by one site's mean. A mixture or a regime chain written with exchangeable components has one fixed point
 * per labelling of the same reading of the data; sorting them makes those copies one mode, and summing their evidence is the
 * mode's share of the posterior mass.
 */

import type { SiteRecord } from './_site-mixture.ts'

/** The `k!` permutations of `0..k-1` in lexicographic order (the first is the identity). */
export function permutations(k: number): number[][] {
  if (k <= 1) return [[0]]
  const out: number[][] = []
  const rec = (prefix: number[], rest: number[]): void => {
    if (rest.length === 0) {
      out.push(prefix)
      return
    }
    for (const [i, v] of rest.entries()) rec([...prefix, v], [...rest.slice(0, i), ...rest.slice(i + 1)])
  }
  rec(
    [],
    Array.from({ length: k }, (_, i) => i),
  )
  return out
}

/** The values of `xs` (an axis of length `k`) moved so that the entry at `q` goes to position `perm[q]`. */
export function moveAxis<T>(xs: readonly T[], perm: readonly number[]): T[] {
  const out = xs.slice()
  for (const [q, to] of perm.entries()) out[to] = xs[q] as T
  return out
}

/** The same move applied to every axis of length `k` of a row-major array of the given shape. */
function moveAxes(values: readonly number[], shape: readonly number[], k: number, perm: readonly number[]): number[] {
  let cur = values.slice()
  const strides = shape.map((_, a) => shape.slice(a + 1).reduce((p, v) => p * v, 1))
  for (const [axis, len] of shape.entries()) {
    if (len !== k) continue
    const stride = strides[axis] as number
    const next = cur.slice()
    for (let i = 0; i < cur.length; i++) {
      const along = Math.floor(i / stride) % len
      next[i - along * stride + (perm[along] as number) * stride] = cur[i] as number
    }
    cur = next
  }
  return cur
}

/**
 * A site's record with the components relabelled: entry `q` of every axis of length `k` moves to `perm[q]`. A covariance is moved
 * along both of its axes, a categorical site's probabilities along their category axis.
 */
export function relabelRecord(r: SiteRecord, perm: readonly number[], k: number): SiteRecord {
  const moved = (v: readonly number[]): number[] => moveAxes(v, r.shape, k, perm)
  // The covariance of a vector site of shape [k] is a k by k block: both of its axes move.
  const covariance =
    r.covariance !== undefined && r.mean.length === k ? moveAxes(r.covariance, [k, k], k, perm) : r.covariance
  let probs: number[] | undefined
  if (r.probs !== undefined) {
    const rows = r.probs.length / k
    const flat = r.probs.slice()
    const out = flat.slice()
    for (let row = 0; row < rows; row++)
      for (let q = 0; q < k; q++) out[row * k + (perm[q] as number)] = flat[row * k + q] as number
    probs = out
  }
  if (probs !== undefined) {
    // The index of a categorical site is a label: its mean and variance are those of the relabelled probabilities.
    const rows = probs.length / k
    const mean = Array.from({ length: rows }, (_, row) =>
      probs.slice(row * k, (row + 1) * k).reduce((a, p, q) => a + p * q, 0),
    )
    const variance = mean.map(
      (m, row) => probs.slice(row * k, (row + 1) * k).reduce((a, p, q) => a + p * q * q, 0) - m * m,
    )
    return { ...r, mean, variance, probs }
  }
  return {
    ...r,
    mean: moved(r.mean),
    variance: moved(r.variance),
    ...(covariance === undefined ? {} : { covariance: covariance as readonly number[] }),
  }
}

/**
 * The site that sorts the components: the first (in the order of `sites`) whose shape is the vector of `k` components and whose
 * belief is a continuous one (a vector of rates or locations), or undefined when there is none.
 */
export function orderingSite(sites: Readonly<Record<string, SiteRecord>>, k: number): string | undefined {
  for (const [name, r] of Object.entries(sites))
    if (r.shape.length === 1 && r.shape[0] === k && r.kind !== 'categorical' && r.kind !== 'moments') return name
  return undefined
}

/** The permutation that sorts the components by the mean of `key`, ascending (entry `q` goes to its rank). */
export function sortingPermutation(r: SiteRecord, k: number): number[] {
  const order = Array.from({ length: k }, (_, q) => q).sort((a, b) => (r.mean[a] ?? 0) - (r.mean[b] ?? 0) || a - b)
  const perm = new Array<number>(k).fill(0)
  for (const [rank, q] of order.entries()) perm[q] = rank
  return perm
}

/** The record set of a fixed point in canonical form: every site relabelled so that the components are sorted by the mean of `key`. */
export function canonicalSites(
  sites: Readonly<Record<string, SiteRecord>>,
  key: string,
  k: number,
): Record<string, SiteRecord> {
  const keyed = sites[key]
  if (keyed === undefined) return { ...sites }
  const perm = sortingPermutation(keyed, k)
  const out: Record<string, SiteRecord> = {}
  for (const [name, r] of Object.entries(sites)) out[name] = relabelRecord(r, perm, k)
  return out
}
