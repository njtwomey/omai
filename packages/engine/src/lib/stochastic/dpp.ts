/**
 * Determinantal point processes (DPPs), random subsets whose items repel each other, so that samples are diverse: the
 * L-ensemble and its marginal kernel, subset and inclusion probabilities, k-DPPs of fixed size, exact spectral
 * sampling, greedy MAP inference by incremental Cholesky, conditioning on included and excluded items, and the expected
 * sample size.
 */

import { MdmlError } from '../base/index.ts'
import { eigh, inv, slogdet, solve } from '../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  exp,
  eye,
  fromStorage,
  mul,
  sub,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { primalNumbers } from './_shared.ts'

function kernelOf(l: TensorLike, where: string): { l: Value; n: number } {
  const v = asFloat(asTensor(l))
  if (v.shape.length !== 2 || v.shape[0] !== v.shape[1] || (v.shape[0] ?? 0) === 0) {
    throw new MdmlError('shape', `${where}: L must be a non-empty square matrix`)
  }
  return { l: v, n: v.shape[0] ?? 0 }
}

function subsetOf(subset: readonly number[], n: number, where: string): number[] {
  const seen = new Set<number>()
  for (const i of subset) {
    if (!(Number.isInteger(i) && i >= 0 && i < n) || seen.has(i)) {
      throw new MdmlError('value', `${where}: the subset must hold distinct items in [0, ${n})`)
    }
    seen.add(i)
  }
  return [...subset]
}

function principal(m: Value, subset: readonly number[]): Value {
  const idx = fromStorage(Int32Array.from(subset), [subset.length], 'int32')
  return take(take(m, idx, 0), idx, 1)
}

function logDet(m: Value): Value {
  return slogdet(m).logabsdet
}

/**
 * The marginal kernel K = L (L + I)^{-1} = I - (L + I)^{-1} of the L-ensemble with likelihood kernel L: P(A is in the
 * sample) = det(K_A). Differentiable in L.
 *
 * @param l - A symmetric positive semidefinite likelihood kernel `[n, n]`.
 * @returns K `[n, n]` (eigenvalues lambda / (1 + lambda) in [0, 1)).
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { marginalKernel } from 'mdml-engine/stochastic'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(marginalKernel([[1, 0], [0, 3]])) // => [[0.5, 0], [0, 0.75]]
 * ```
 */
export function marginalKernel(l: TensorLike): Value {
  const { l: m, n } = kernelOf(l, 'marginalKernel')
  return sub(eye(n), inv(add(m, eye(n))))
}

/**
 * The probability of a subset under the L-ensemble, log P(Y = A) = log det(L_A) - log det(L + I). Differentiable in L.
 *
 * @param l - The likelihood kernel.
 * @param subset - The items of A (empty for the empty set).
 * @returns log P(Y = A).
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { dppLogProbability } from 'mdml-engine/stochastic'
 * import { item } from 'mdml-engine/tensor'
 * Math.exp(item(dppLogProbability([[1, 0], [0, 3]], [1]))) // => 0.375
 * ```
 */
export function dppLogProbability(l: TensorLike, subset: readonly number[]): Value {
  const { l: m, n } = kernelOf(l, 'dppLogProbability')
  const a = subsetOf(subset, n, 'dppLogProbability')
  const norm = logDet(add(m, eye(n)))
  return a.length === 0 ? sub(0, norm) : sub(logDet(principal(m, a)), norm)
}

/**
 * The inclusion probability P(A is in Y) = det(K_A) of a subset, from the marginal kernel (all singletons: the
 * diagonal of K; pairs: K_ii K_jj - K_ij^2, below the product, so items repel). Differentiable in L.
 *
 * @param l - The likelihood kernel.
 * @param subset - The items of A.
 * @returns det(K_A).
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { inclusionProbability } from 'mdml-engine/stochastic'
 * import { item } from 'mdml-engine/tensor'
 * item(inclusionProbability([[1, 0], [0, 3]], [0, 1])) // => 0.375
 * ```
 */
export function inclusionProbability(l: TensorLike, subset: readonly number[]): Value {
  const k = marginalKernel(l)
  const a = subsetOf(subset, k.shape[0] ?? 0, 'inclusionProbability')
  if (a.length === 0) return fromStorage(Float64Array.of(1), [], 'float64')
  const { sign, logabsdet } = slogdet(principal(k, a))
  return mul(sign, exp(logabsdet))
}

/**
 * The elementary symmetric polynomials e_0, ..., e_k of numbers lambda_1, ..., lambda_N by the recursion
 * e_j(1..n) = e_j(1..n-1) + lambda_n e_{j-1}(1..n-1): the normalisers of k-DPPs (sum over |A| = k of det L_A = e_k of
 * the eigenvalues).
 *
 * @param values - lambda.
 * @param k - The largest degree.
 * @returns `[k + 1]` values e_0 = 1, ..., e_k.
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { elementarySymmetric } from 'mdml-engine/stochastic'
 * Array.from(elementarySymmetric([1, 2, 3], 3)) // => [1, 6, 11, 6]
 * ```
 */
export function elementarySymmetric(values: ArrayLike<number>, k: number): Float64Array {
  if (!(Number.isInteger(k) && k >= 0)) throw new MdmlError('value', 'elementarySymmetric: k must be >= 0')
  const e = new Float64Array(k + 1)
  e[0] = 1
  for (let n = 0; n < values.length; n++) {
    const lam = values[n] ?? 0
    for (let j = Math.min(k, n + 1); j >= 1; j--) e[j] = (e[j] ?? 0) + lam * (e[j - 1] ?? 0)
  }
  return e
}

/** The table E[j][n] = e_j(lambda_1..lambda_n) for the k-DPP eigenvector selection. */
function symmetricTable(lam: readonly number[], k: number): number[][] {
  const n = lam.length
  const e = Array.from({ length: k + 1 }, (_, j) => new Array<number>(n + 1).fill(j === 0 ? 1 : 0))
  for (let j = 1; j <= k; j++) {
    for (let m = 1; m <= n; m++) {
      const row = e[j] ?? []
      row[m] = (row[m - 1] ?? 0) + (lam[m - 1] ?? 0) * (e[j - 1]?.[m - 1] ?? 0)
    }
  }
  return e
}

/**
 * The log-probability of a subset of size k under the k-DPP (the L-ensemble conditioned on |Y| = k),
 * log det(L_A) - log e_k(lambda). Differentiable in L through det(L_A); the normaliser is computed from the eigenvalues.
 *
 * @param l - The likelihood kernel.
 * @param subset - A subset of size k >= 1.
 * @returns log P(Y = A | |Y| = k).
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { kDppLogProbability } from 'mdml-engine/stochastic'
 * import { item } from 'mdml-engine/tensor'
 * Math.exp(item(kDppLogProbability([[1, 0], [0, 3]], [1]))) // => 0.75
 * ```
 */
export function kDppLogProbability(l: TensorLike, subset: readonly number[]): Value {
  const { l: m, n } = kernelOf(l, 'kDppLogProbability')
  const a = subsetOf(subset, n, 'kDppLogProbability')
  if (a.length === 0) throw new MdmlError('value', 'kDppLogProbability: k must be at least 1')
  const lam = toFlat(eigh(m).eigenvalues).map((v) => Math.max(v, 0))
  const ek = elementarySymmetric(lam, a.length)[a.length] ?? 0
  return sub(logDet(principal(m, a)), Math.log(ek))
}

function spectral(l: Value): { values: number[]; vectors: number[][] } {
  const { eigenvalues, eigenvectors } = eigh(l)
  const values = toFlat(eigenvalues).map((v) => Math.max(v, 0))
  const n = values.length
  const v = toFlat(eigenvectors)
  const vectors = Array.from({ length: n }, (_, c) => Array.from({ length: n }, (_, r) => v[r * n + c] ?? 0))
  return { values, vectors }
}

/** The projection-DPP sampler of Hough et al. on chosen orthonormal vectors (Kulesza and Taskar, Algorithm 1). */
function projectionSample(key: Key, basis: number[][], n: number): number[] {
  let v = basis.map((b) => [...b])
  const out: number[] = []
  for (let step = 0; v.length > 0; step++) {
    // P(i) = sum_v v_i^2 / |V|
    const weights = Array.from({ length: n }, (_, i) => v.reduce((s, b) => s + (b[i] ?? 0) ** 2, 0))
    const total = weights.reduce((a, b) => a + b, 0)
    let u = (uniform(foldIn(key, step), [1]).data[0] ?? 0) * total
    let pick = n - 1
    for (let i = 0; i < n; i++) {
      u -= weights[i] ?? 0
      if (u < 0 && !out.includes(i)) {
        pick = i
        break
      }
    }
    out.push(pick)
    // remove the component along e_pick: subtract a vector with a non-zero pick coordinate, then re-orthonormalise
    const j = v.reduce((best, b, idx) => (Math.abs(b[pick] ?? 0) > Math.abs(v[best]?.[pick] ?? 0) ? idx : best), 0)
    const pivot = v[j] ?? []
    const rest = v
      .filter((_, idx) => idx !== j)
      .map((b) => {
        const f = (b[pick] ?? 0) / (pivot[pick] ?? 1)
        return b.map((x, i) => x - f * (pivot[i] ?? 0))
      })
    // Gram-Schmidt
    const ortho: number[][] = []
    for (const b of rest) {
      const w = [...b]
      for (const q of ortho) {
        let d = 0
        for (let i = 0; i < n; i++) d += (w[i] ?? 0) * (q[i] ?? 0)
        for (let i = 0; i < n; i++) w[i] = (w[i] ?? 0) - d * (q[i] ?? 0)
      }
      const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0))
      if (norm > 1e-12) ortho.push(w.map((x) => x / norm))
    }
    v = ortho
  }
  return out.sort((a, b) => a - b)
}

/**
 * An exact sample of the L-ensemble DPP by the spectral algorithm of Hough et al.: keep eigenvector i with probability
 * lambda_i / (1 + lambda_i), then draw one item per kept vector from the projection DPP they span.
 *
 * @param key - The key.
 * @param l - The likelihood kernel.
 * @returns The sampled items, increasing.
 * @concept determinantal-point-processes
 * @reference J. B. Hough et al., Determinantal processes and independence, Probability Surveys 3 (2006); A. Kulesza
 *   and B. Taskar, Determinantal point processes for machine learning (2012).
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { sampleDpp } from 'mdml-engine/stochastic'
 * sampleDpp(key(0), [[1e9, 0], [0, 1e9]]) // => [0, 1]
 * ```
 */
export function sampleDpp(key: Key, l: TensorLike): number[] {
  const { l: m, n } = kernelOf(l, 'sampleDpp')
  const { values, vectors } = spectral(m)
  const u = uniform(foldIn(key, 0), [n]).data
  const kept = vectors.filter((_, i) => (u[i] ?? 1) < (values[i] ?? 0) / (1 + (values[i] ?? 0)))
  return projectionSample(foldIn(key, 1), kept, n)
}

/**
 * An exact sample of the k-DPP: choose k eigenvectors with probabilities from the elementary symmetric polynomials of
 * the eigenvalues (Kulesza and Taskar, Algorithm 8), then sample the projection DPP they span.
 *
 * @param key - The key.
 * @param l - The likelihood kernel.
 * @param k - The sample size (at most the rank of L).
 * @returns k items, increasing.
 * @throws MdmlError with code `value` when k exceeds the rank.
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { sampleKDpp } from 'mdml-engine/stochastic'
 * sampleKDpp(key(0), [[2, 0.5, 0], [0.5, 1, 0.2], [0, 0.2, 1]], 2).length // => 2
 * ```
 */
export function sampleKDpp(key: Key, l: TensorLike, k: number): number[] {
  const { l: m, n } = kernelOf(l, 'sampleKDpp')
  if (!(Number.isInteger(k) && k >= 1 && k <= n)) throw new MdmlError('value', 'sampleKDpp: k must be in [1, n]')
  const { values, vectors } = spectral(m)
  const e = symmetricTable(values, k)
  const top = values.reduce((m, v) => Math.max(m, v), 0)
  if (!((e[k]?.[n] ?? 0) > 1e-10 * top ** k)) throw new MdmlError('value', 'sampleKDpp: k exceeds the rank of L')
  const chosen: number[][] = []
  let left = k
  const u = uniform(foldIn(key, 0), [n]).data
  for (let i = n; i >= 1 && left > 0; i--) {
    const p = ((values[i - 1] ?? 0) * (e[left - 1]?.[i - 1] ?? 0)) / (e[left]?.[i] ?? 1)
    if ((u[i - 1] ?? 1) < p) {
      chosen.push(vectors[i - 1] ?? [])
      left -= 1
    }
  }
  return projectionSample(foldIn(key, 1), chosen, n)
}

/**
 * Greedy MAP inference for a DPP (Chen, Zhang and Zhou 2018): repeatedly add the item with the largest gain in
 * log det(L_A), maintained by an incremental Cholesky factorisation in O(n k^2), until k items (or, without k, until no
 * item increases the determinant, gain d_i^2 <= 1).
 *
 * @param l - The likelihood kernel.
 * @param k - The number of items (default: stop when the determinant would not grow).
 * @returns The chosen items in the order they were added.
 * @concept determinantal-point-processes
 * @reference L. Chen, G. Zhang and H. Zhou, Fast greedy MAP inference for determinantal point process to improve
 *   recommendation diversity (NeurIPS 2018).
 * @example
 * ```ts
 * import { greedyMap } from 'mdml-engine/stochastic'
 * greedyMap([[3, 2.9, 0], [2.9, 3, 0], [0, 0, 2]], 2) // => [0, 2]
 * ```
 */
export function greedyMap(l: TensorLike, k?: number): number[] {
  const { l: m, n } = kernelOf(l, 'greedyMap')
  const cap = k ?? n
  if (!(Number.isInteger(cap) && cap >= 1 && cap <= n)) throw new MdmlError('value', 'greedyMap: k must be in [1, n]')
  const a = primalNumbers(m)
  const d2 = Array.from({ length: n }, (_, i) => a[i * n + i] ?? 0)
  const c: number[][] = Array.from({ length: n }, () => [])
  const chosen: number[] = []
  for (let step = 0; step < cap; step++) {
    let j = -1
    for (let i = 0; i < n; i++) if (!chosen.includes(i) && (j < 0 || (d2[i] ?? 0) > (d2[j] ?? 0))) j = i
    if (j < 0) break
    const best = d2[j] ?? 0
    if (best <= (k === undefined ? 1 : 1e-12)) break
    chosen.push(j)
    const dj = Math.sqrt(best)
    for (let i = 0; i < n; i++) {
      if (chosen.includes(i)) continue
      let dot = 0
      for (let t = 0; t < step; t++) dot += (c[j]?.[t] ?? 0) * (c[i]?.[t] ?? 0)
      const e = ((a[j * n + i] ?? 0) - dot) / dj
      c[i]?.push(e)
      d2[i] = (d2[i] ?? 0) - e * e
    }
    c[j]?.push(dj)
  }
  return chosen
}

/** The kernel of a DPP after conditioning, on the items that remain. */
export interface ConditionedDpp {
  /** The likelihood kernel on the remaining items. */
  readonly kernel: Value
  /** The original index of each remaining item. */
  readonly items: readonly number[]
}

/**
 * Condition an L-ensemble on including the items A and excluding the items B: exclusion restricts L to the other items;
 * inclusion gives the L-ensemble L^A = ([(L + I_{not A})^{-1}]_{not A})^{-1} - I on the items outside A (Kulesza and
 * Taskar, eq. 2.42), so that P(Y = A + C | A in Y) is proportional to det(L^A_C). Differentiable in L.
 *
 * @param l - The likelihood kernel.
 * @param options - The items to include and to exclude.
 * @returns The conditioned kernel and the original indices of its items.
 * @throws MdmlError with code `value` when the sets overlap.
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { conditionDpp } from 'mdml-engine/stochastic'
 * conditionDpp([[1, 0, 0], [0, 2, 0], [0, 0, 3]], { exclude: [1] }).items // => [0, 2]
 * ```
 */
export function conditionDpp(
  l: TensorLike,
  options: { readonly include?: readonly number[]; readonly exclude?: readonly number[] },
): ConditionedDpp {
  const { l: m, n } = kernelOf(l, 'conditionDpp')
  const inc = subsetOf(options.include ?? [], n, 'conditionDpp')
  const exc = subsetOf(options.exclude ?? [], n, 'conditionDpp')
  if (inc.some((i) => exc.includes(i))) throw new MdmlError('value', 'conditionDpp: include and exclude overlap')
  const keep = Array.from({ length: n }, (_, i) => i).filter((i) => !exc.includes(i))
  let kernel = principal(m, keep)
  let items = keep
  if (inc.length > 0) {
    const local = inc.map((i) => keep.indexOf(i))
    const others = keep.map((_, i) => i).filter((i) => !local.includes(i))
    const size = keep.length
    // L + I restricted to the complement of A on the diagonal
    const mask = fromStorage(
      Float64Array.from({ length: size * size }, (_, t) => {
        const r = Math.floor(t / size)
        return r === t % size && !local.includes(r) ? 1 : 0
      }),
      [size, size],
      'float64',
    )
    const inverse = inv(add(kernel, mask))
    kernel = sub(solve(principal(inverse, others), eye(others.length)), eye(others.length))
    items = others.map((i) => keep[i] ?? 0)
  }
  return { kernel, items }
}

/**
 * The expected sample size of an L-ensemble, E|Y| = trace(K) = sum_i lambda_i / (1 + lambda_i).
 *
 * @param l - The likelihood kernel.
 * @returns E|Y|.
 * @concept determinantal-point-processes
 * @example
 * ```ts
 * import { expectedSize } from 'mdml-engine/stochastic'
 * expectedSize([[1, 0], [0, 3]]) // => 1.25
 * ```
 */
export function expectedSize(l: TensorLike): number {
  const { l: m } = kernelOf(l, 'expectedSize')
  return toFlat(eigh(m).eigenvalues).reduce((s, v) => s + Math.max(v, 0) / (1 + Math.max(v, 0)), 0)
}
