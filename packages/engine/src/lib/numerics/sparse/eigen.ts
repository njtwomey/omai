import { WhiteboxError } from '../../base/index.ts'
import { type Key, normal } from '../../random/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { ddot as dot, eigh, dnrm2 as norm } from '../linalg/index.ts'
import { asOperator, type CsrMatrix, type LinearOperator } from './csr.ts'

type Vec = Float64Array

function numeric(op: LinearOperator, transpose = false): (x: Vec) => Vec {
  const f = transpose ? op.rmatvec : op.matvec
  if (f === undefined) throw new WhiteboxError('value', 'the operator has no transpose product')
  return (x) => Float64Array.from(toFlat(primalOf(f(fromStorage(x, [x.length], 'float64')))))
}

function startVector(n: number, v0: ArrayLike<number> | undefined, key: Key | undefined): Vec {
  if (v0 !== undefined) return Float64Array.from(v0)
  if (key !== undefined) return Float64Array.from(normal(key, [n]).data)
  // A fixed, non-degenerate default.
  return Float64Array.from({ length: n }, (_, i) => 1 + Math.sin(i + 1) / 3)
}

/** Orthogonalise w against the basis twice (classical Gram-Schmidt with reorthogonalisation). */
function orthogonalise(w: Vec, basis: readonly Vec[]): Vec {
  let v = w
  for (let pass = 0; pass < 2; pass++) {
    for (const q of basis) {
      const c = dot(v, q)
      v = v.map((x, i) => x - c * (q[i] ?? 0))
    }
  }
  return v
}

/** A Lanczos factorisation A Q_m = Q_m T_m + beta_m q_{m+1} e_m^T. */
export interface LanczosResult {
  /** Diagonal of T. */
  readonly alpha: readonly number[]
  /** Off-diagonal of T (beta[j] couples q_j and q_{j+1}); the last is the residual coupling. */
  readonly beta: readonly number[]
  /** The orthonormal basis vectors. */
  readonly basis: readonly Vec[]
}

/**
 * The Lanczos process for a symmetric operator, with full reorthogonalisation (so the basis stays orthonormal in
 * floating point): m steps produce the tridiagonal T whose eigenvalues (Ritz values) approximate A's extremes.
 *
 * @param a - Symmetric matrix or operator.
 * @param m - Number of steps.
 * @param options - Start vector or key, and vectors to stay orthogonal to (deflation).
 * @returns T's diagonals and the basis.
 * @concept lanczos
 * @example
 * ```ts
 * import { lanczos } from 'whitebox-engine/numerics/sparse'
 * lanczos([[2, 1], [1, 2]], 2).alpha.length // => 2
 * ```
 */
export function lanczos(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: { readonly v0?: ArrayLike<number>; readonly key?: Key; readonly deflate?: readonly Float64Array[] } = {},
): LanczosResult {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  const deflate = options.deflate ?? []
  let q = orthogonalise(startVector(n, options.v0, options.key), deflate)
  if (norm(q) < 1e-10)
    q = orthogonalise(
      Float64Array.from({ length: n }, (_, i) => Math.cos(3.7 * i + 0.3)),
      deflate,
    )
  q = q.map((v) => v / norm(q))
  const basis: Vec[] = [q]
  const alpha: number[] = []
  const beta: number[] = []
  for (let j = 0; j < Math.min(m, n); j++) {
    const w0 = apply(basis[j] as Vec)
    alpha.push(dot(w0, basis[j] as Vec))
    const w = orthogonalise(w0, deflate.length === 0 ? basis : [...deflate, ...basis])
    const b = norm(w)
    beta.push(b)
    if (b < 1e-14 * Math.max(1, Math.abs(alpha[j] ?? 0)) || j + 1 >= Math.min(m, n)) break
    basis.push(w.map((v) => v / b))
  }
  return { alpha, beta, basis }
}

/** An Arnoldi factorisation A Q_m = Q_{m+1} H_m. */
export interface ArnoldiResult {
  /** The (m + 1) x m upper Hessenberg matrix, row-major as rows of columns. */
  readonly hessenberg: Tensor
  readonly basis: readonly Vec[]
}

/**
 * The Arnoldi process for a general operator (with reorthogonalisation): an orthonormal basis of the Krylov space and
 * the Hessenberg matrix of A restricted to it, whose eigenvalues approximate A's.
 *
 * @param a - Matrix or operator.
 * @param m - Number of steps.
 * @param options - Start vector or key.
 * @returns The Hessenberg matrix `[m + 1, m]` and the basis.
 * @example
 * ```ts
 * import { arnoldi } from 'whitebox-engine/numerics/sparse'
 * arnoldi([[1, 2], [0, 3]], 2).hessenberg.shape // => [3, 2]
 * ```
 */
export function arnoldi(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: { readonly v0?: ArrayLike<number>; readonly key?: Key } = {},
): ArnoldiResult {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  let q = startVector(n, options.v0, options.key)
  q = q.map((v) => v / norm(q))
  const basis: Vec[] = [q]
  const steps = Math.min(m, n)
  const h = new Float64Array((steps + 1) * steps)
  for (let j = 0; j < steps; j++) {
    let w = apply(basis[j] as Vec)
    for (let pass = 0; pass < 2; pass++) {
      basis.forEach((b, i) => {
        const c = dot(w, b)
        h[i * steps + j] = (h[i * steps + j] ?? 0) + c
        w = w.map((x, k) => x - c * (b[k] ?? 0))
      })
    }
    const hn = norm(w)
    h[(j + 1) * steps + j] = hn
    if (hn < 1e-14) break
    basis.push(w.map((v) => v / hn))
  }
  return { hessenberg: fromStorage(h, [steps + 1, steps], 'float64'), basis }
}

/** Which eigenvalues {@link eigsh} returns: largest magnitude, largest algebraic or smallest algebraic. */
export type Which = 'LM' | 'LA' | 'SA'

/** Result of {@link eigsh}. */
export interface EigshResult {
  /** Eigenvalues, ascending (as scipy). */
  readonly values: Tensor
  /** Eigenvectors as columns `[n, k]`. */
  readonly vectors: Tensor
  readonly restarts: number
  /** Residual norms |A v - lambda v| per pair. */
  readonly residuals: readonly number[]
}

function ritz(a: LanczosResult): { values: number[]; vectors: number[][] } {
  const m = a.alpha.length
  const t = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    t[i * m + i] = a.alpha[i] ?? 0
    if (i + 1 < m) {
      t[i * m + i + 1] = a.beta[i] ?? 0
      t[(i + 1) * m + i] = a.beta[i] ?? 0
    }
  }
  const { eigenvalues, eigenvectors } = eigh(fromStorage(t, [m, m], 'float64'))
  const w = toFlat(eigenvalues as Tensor)
  const v = toFlat(eigenvectors as Tensor)
  return { values: w, vectors: w.map((_, j) => Array.from({ length: m }, (_, i) => v[i * m + j] ?? 0)) }
}

/**
 * The k extreme eigenpairs of a symmetric operator (scipy.sparse.linalg.eigsh): Lanczos with full
 * reorthogonalisation in blocks of m steps, explicitly restarted from the wanted Ritz vectors, with converged pairs
 * (residual below tol |lambda|) locked and deflated so that repeated eigenvalues appear once per copy.
 *
 * @param a - Symmetric matrix or operator.
 * @param k - Number of eigenpairs.
 * @param options - Which end (default LM), tolerance (default 1e-10), Krylov size m (default max(2k + 1, 20)), restart
 *   cap and start vector.
 * @returns Eigenvalues (ascending), eigenvectors and residuals.
 * @concept lanczos
 * @example
 * ```ts
 * import { diagCsr, eigsh } from 'whitebox-engine/numerics/sparse'
 * Array.from(eigsh(diagCsr([1, 5, 3, 9, 2]), 2).values.data).map((v) => Math.round(v * 1e10) / 1e10) // => [5, 9]
 * ```
 */
export function eigsh(
  a: CsrMatrix | TensorLike | LinearOperator,
  k: number,
  options: {
    readonly which?: Which
    readonly tol?: number
    readonly ncv?: number
    readonly maxRestarts?: number
    readonly v0?: ArrayLike<number>
  } = {},
): EigshResult {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  if (!(k >= 1 && k < n)) throw new WhiteboxError('value', `eigsh: k must be in [1, ${n - 1}]`)
  const which = options.which ?? 'LM'
  const tol = options.tol ?? 1e-10
  const m = Math.min(n, options.ncv ?? Math.max(2 * k + 1, 20))
  const pick = (values: readonly number[]): number[] => {
    const idx = values.map((_, i) => i)
    const score = (i: number): number =>
      which === 'LM' ? Math.abs(values[i] ?? 0) : which === 'LA' ? (values[i] ?? 0) : -(values[i] ?? 0)
    return idx.sort((p, q) => score(q) - score(p)).slice(0, k)
  }
  const converged = (r: number, lambda: number): boolean => r <= tol * Math.max(1, Math.abs(lambda))
  // Lanczos with locking: converged Ritz pairs are locked and later runs stay orthogonal to them, so repeated
  // eigenvalues are found once per copy.
  const locked: Vec[] = []
  const lockedValues: number[] = []
  const lockedResiduals: number[] = []
  let start: ArrayLike<number> | undefined = options.v0
  let restarts = 0
  while (locked.length < k) {
    const fact = lanczos(op, Math.min(m, n - locked.length), {
      ...(start === undefined ? {} : { v0: start }),
      deflate: locked,
    })
    const { values, vectors } = ritz(fact)
    const want = k - locked.length
    const chosen = pick(values).slice(0, want)
    const vecs = chosen.map((j) => {
      const y = vectors[j] as number[]
      const v = new Float64Array(n)
      fact.basis.forEach((q, i) => {
        const c = y[i] ?? 0
        for (let r = 0; r < n; r++) v[r] = (v[r] ?? 0) + c * (q[r] ?? 0)
      })
      return v
    })
    const lambdas = chosen.map((j) => values[j] ?? 0)
    const residuals = vecs.map((v, i) => norm(apply(v).map((x, r) => x - (lambdas[i] ?? 0) * (v[r] ?? 0))))
    const exhausted = fact.basis.length + locked.length >= n
    // Lock the leading run of converged pairs (in order of preference).
    let lockedNow = 0
    for (let i = 0; i < vecs.length; i++) {
      if (!(exhausted || converged(residuals[i] ?? 0, lambdas[i] ?? 0))) break
      locked.push(vecs[i] as Vec)
      lockedValues.push(lambdas[i] ?? 0)
      lockedResiduals.push(residuals[i] ?? 0)
      lockedNow++
    }
    if (lockedNow === 0) {
      restarts++
      if (restarts > (options.maxRestarts ?? 300))
        throw new WhiteboxError('convergence', 'eigsh: no convergence within the restart limit')
      // Restart from the wanted Ritz vectors, the leading one weighted most.
      const next = new Float64Array(n)
      vecs.forEach((v, i) => {
        const w = 1 / (i + 1)
        for (let r = 0; r < n; r++) next[r] = (next[r] ?? 0) + w * (v[r] ?? 0)
      })
      start = next
    } else start = undefined
  }
  const order = lockedValues.map((_, i) => i).sort((p, q) => (lockedValues[p] ?? 0) - (lockedValues[q] ?? 0))
  const out = new Float64Array(n * k)
  order.forEach((src, col) => {
    const v = locked[src] as Vec
    // Sign convention: the largest-magnitude component positive.
    let big = 0
    for (let r = 0; r < n; r++) if (Math.abs(v[r] ?? 0) > Math.abs(v[big] ?? 0)) big = r
    const sign = (v[big] ?? 0) < 0 ? -1 : 1
    for (let r = 0; r < n; r++) out[r * k + col] = sign * (v[r] ?? 0)
  })
  return {
    values: fromStorage(
      Float64Array.from(order, (i) => lockedValues[i] ?? 0),
      [k],
      'float64',
    ),
    vectors: fromStorage(out, [n, k], 'float64'),
    restarts,
    residuals: order.map((i) => lockedResiduals[i] ?? 0),
  }
}

/**
 * The k largest singular triplets of a matrix (scipy.sparse.linalg.svds): eigsh on A^T A for the right vectors, then
 * u = A v / s. Suited to the top of the spectrum (small singular values lose accuracy through the squaring).
 *
 * @param a - Matrix or operator with a transpose product.
 * @param k - Number of triplets.
 * @param options - Tolerance.
 * @returns Singular values (ascending, as scipy), left and right vectors.
 * @example
 * ```ts
 * import { svds } from 'whitebox-engine/numerics/sparse'
 * Array.from(svds([[3, 0, 0], [0, 4, 0], [0, 0, 1]], 1).s.data).map((v) => Math.round(v * 1e10) / 1e10) // => [4]
 * ```
 */
export function svds(
  a: CsrMatrix | TensorLike | LinearOperator,
  k: number,
  options: { readonly tol?: number } = {},
): { s: Tensor; u: Tensor; vt: Tensor } {
  const op = asOperator(a)
  const [m, n] = op.shape
  const apply = numeric(op)
  const applyT = numeric(op, true)
  const gram: LinearOperator = {
    shape: [n, n],
    matvec: (x) => fromStorage(applyT(apply(Float64Array.from(toFlat(primalOf(x))))), [n], 'float64'),
  }
  const { values, vectors } = eigsh(gram, k, { which: 'LA', tol: options.tol ?? 1e-12 })
  const lambdas = toFlat(values)
  const v = toFlat(vectors)
  const s = lambdas.map((l) => Math.sqrt(Math.max(l, 0)))
  const u = new Float64Array(m * k)
  const vt = new Float64Array(k * n)
  s.forEach((sv, j) => {
    const col = Float64Array.from({ length: n }, (_, r) => v[r * k + j] ?? 0)
    const av = apply(col)
    for (let r = 0; r < m; r++) u[r * k + j] = sv === 0 ? 0 : (av[r] ?? 0) / sv
    for (let c = 0; c < n; c++) vt[j * n + c] = col[c] ?? 0
  })
  return {
    s: fromStorage(Float64Array.from(s), [k], 'float64'),
    u: fromStorage(u, [m, k], 'float64'),
    vt: fromStorage(vt, [k, n], 'float64'),
  }
}
