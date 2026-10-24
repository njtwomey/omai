// Orthogonal staircase reductions and subspace bases for controllability, observability, minimal realisations and the
// Kalman decomposition. Everything here is on primal (concrete) matrices; the transformations are constants.
import { blockDiag, daxpy, ddot, norm, nullSpace, svd } from '../numerics/linalg/index.ts'
import {
  concat,
  eye,
  matmul,
  primalOf,
  sliceAxis,
  type Tensor,
  tensor,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { numbers, rows } from './_shared.ts'

/** The result of a staircase reduction: an orthogonal T whose first `rank` columns span the reachable subspace. */
export interface Staircase {
  readonly t: Value
  readonly rank: number
}

const fro = (x: Value): number => (numbers(x).length === 0 ? 0 : (numbers(norm(x)).at(0) ?? 0))

/** The default rank tolerance (as SLICOT TB01PD): n^2 eps max(|A|_F, |B|_F). */
export function defaultTolerance(a: Value, b: Value): number {
  const n = a.shape[0] ?? 0
  return n * n * Number.EPSILON * Math.max(fro(primalOf(a)), fro(primalOf(b)), Number.MIN_VALUE)
}

/**
 * The controllability staircase of (A, B): orthogonal transformations of the remaining coordinates that compress each
 * new block of B (then of A's sub-diagonal blocks) onto its range.
 */
export function controllabilityStaircase(a: Value, b: Value, tol?: number): Staircase {
  const n = a.shape[0] ?? 0
  const threshold = tol ?? defaultTolerance(a, b)
  let am: Value = primalOf(a)
  let t: Value = eye(n)
  let block: Value = primalOf(b)
  let start = 0
  while (start < n && (block.shape[1] ?? 0) > 0) {
    const r = n - start
    // Padding with r zero columns makes the reduced SVD return a full r x r U.
    const { u, s } = svd(concat([block, zeros([r, r])], 1))
    const rho = numbers(s).filter((v) => v > threshold).length
    if (rho === 0) break
    const full = start === 0 ? u : blockDiag(eye(start), u)
    am = matmul(transpose(full), matmul(am, full))
    t = matmul(t, full)
    const next = start + rho
    if (next >= n) {
      start = next
      break
    }
    block = sliceAxis(sliceAxis(am, 0, next, n), 1, start, next)
    start = next
  }
  return { t, rank: Math.min(start, n) }
}

/** Columns of a matrix value as number arrays. */
export function columns(x: Value): number[][] {
  const r = rows(x)
  const k = x.shape[1] ?? 0
  return Array.from({ length: k }, (_, j) => r.map((row) => row[j] ?? 0))
}

/** A matrix tensor `[n, k]` from k column vectors of length n. */
export function fromColumns(cols: readonly (readonly number[])[], n: number): Tensor {
  if (cols.length === 0) return zeros([n, 0]) as Tensor
  return tensor(Array.from({ length: n }, (_, i) => cols.map((c) => c[i] ?? 0)))
}

/** An orthonormal basis of the null space of a matrix given by rows (numbers), or all of R^n when there are none. */
function nullOf(rowsOf: readonly (readonly number[])[], n: number): number[][] {
  if (rowsOf.length === 0) return columns(eye(n))
  return columns(nullSpace(tensor(rowsOf.map((r) => [...r])), 1e-10))
}

/** The linear combination sum_j coeffs_j cols_j. */
function combine(cols: readonly (readonly number[])[], coeffs: readonly number[], n: number): number[] {
  let acc: ArrayLike<number> = new Float64Array(n)
  cols.forEach((c, j) => {
    acc = daxpy(coeffs[j] ?? 0, c, acc)
  })
  return Array.from(acc)
}

/** The intersection of two subspaces given by orthonormal columns. */
export function intersect(u: readonly (readonly number[])[], v: readonly (readonly number[])[], n: number): number[][] {
  if (u.length === 0 || v.length === 0) return []
  // u a = v b: the null space of [U, -V] gives the coefficients a.
  const stacked = Array.from({ length: n }, (_, i) => [...u.map((c) => c[i] ?? 0), ...v.map((c) => -(c[i] ?? 0))])
  const coeffs = columns(nullSpace(tensor(stacked), 1e-10))
  return orthonormal(
    coeffs.map((cf) => combine(u, cf, n)),
    n,
  )
}

/** An orthonormal basis of the span of some vectors. */
export function orthonormal(vectors: readonly (readonly number[])[], n: number): number[][] {
  if (vectors.length === 0) return []
  const { u, s } = svd(fromColumns(vectors, n))
  const sv = numbers(s)
  const top = sv[0] ?? 0
  const rank = sv.filter((x) => x > 1e-10 * Math.max(top, 1)).length
  return columns(u).slice(0, rank)
}

/** Vectors of span(within) orthogonal to span(of): an orthonormal complement of `of` inside `within`. */
export function complementIn(
  within: readonly (readonly number[])[],
  of: readonly (readonly number[])[],
  n: number,
): number[][] {
  if (within.length === 0) return []
  if (of.length === 0) return within.map((c) => [...c])
  // Coefficients a with of^T (W a) = 0.
  const gram = of.map((o) => within.map((w) => ddot(w, o)))
  return orthonormal(
    nullOf(gram, within.length).map((cf) => combine(within, cf, n)),
    n,
  )
}

/** An orthonormal basis of the orthogonal complement of a span in R^n. */
export function complement(of: readonly (readonly number[])[], n: number): number[][] {
  return nullOf(of, n)
}
