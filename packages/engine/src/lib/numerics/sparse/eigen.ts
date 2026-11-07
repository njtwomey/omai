/**
 * Krylov eigensolvers for large or implicit matrices: the Lanczos process for symmetric operators and the Arnoldi
 * process for general ones (both with reorthogonalisation), `eigsh` for a few extreme eigenpairs by restarted Lanczos,
 * and `svds` for the largest singular triplets, as scipy.sparse.linalg.
 */

import { MdmlError } from '../../base/index.ts'
import { type Key, normal } from '../../random/index.ts'
import {
  asTensor,
  concreteOrThrow,
  fromStorage,
  primalOf,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { vectorOf } from '../_vector.ts'
import { ddot as dot, eigh, eigvals, dnrm2 as norm } from '../linalg/index.ts'
import { asOperator, type CsrMatrix, type LinearOperator } from './csr.ts'

function numeric(op: LinearOperator, transpose = false): (x: Float64Array) => Float64Array {
  const f = transpose ? op.rmatvec : op.matvec
  if (f === undefined) throw new MdmlError('value', 'the operator has no transpose product')
  return (x) => Float64Array.from(toFlat(primalOf(f(fromStorage(x, [x.length], 'float64')))))
}

function startVector(n: number, v0: TensorLike | undefined, key: Key | undefined): Float64Array {
  if (v0 !== undefined) return Float64Array.from(toVector(v0, 'the start vector'))
  if (key !== undefined) return Float64Array.from(normal(key, [n]).data)
  // A fixed, non-degenerate default.
  return Float64Array.from({ length: n }, (_, i) => 1 + Math.sin(i + 1) / 3)
}

/** The vectors to stay orthogonal to: the rows of a matrix, or a list of vectors. */
function deflationRows(deflate: TensorLike | readonly Float64Array[] | undefined): readonly Float64Array[] {
  if (deflate === undefined) return []
  if (Array.isArray(deflate) && deflate.every((row) => row instanceof Float64Array))
    return deflate as readonly Float64Array[]
  const t = concreteOrThrow(asTensor(deflate as TensorLike) as Tensor, 'the deflation vectors')
  if (t.shape.length !== 2) throw new MdmlError('shape', 'the deflation vectors must be the rows of a matrix')
  const [k = 0, n = 0] = t.shape
  return Array.from({ length: k }, (_, i) => Float64Array.from(Array.from(t.data).slice(i * n, (i + 1) * n)))
}

/** Orthogonalise w against the basis twice (classical Gram-Schmidt with reorthogonalisation). */
function orthogonalise(w: Float64Array, basis: readonly Float64Array[]): Float64Array {
  let v = w
  for (let pass = 0; pass < 2; pass++) {
    for (const q of basis) {
      const c = dot(v, q)
      v = v.map((x, i) => x - c * (q[i] ?? 0))
    }
  }
  return v
}

/**
 * A Lanczos factorisation A Q_m = Q_m T_m + beta_m q_{m+1} e_m^T. Holds tensors (migration, D-009: `alpha` and `beta`
 * were `readonly number[]`, `basis` a list of `Float64Array`s).
 */
export interface LanczosResult {
  /** Diagonal of T, `[m]`. */
  readonly alpha: Tensor<'float64'>
  /** Off-diagonal of T, `[m]` (beta[j] couples q_j and q_{j+1}); the last is the residual coupling. */
  readonly beta: Tensor<'float64'>
  /** The orthonormal basis vectors as the rows of a `[m, n]` matrix. */
  readonly basis: Tensor<'float64'>
}

/** A Lanczos factorisation as plain numbers: the form the restarted eigensolver loops on. */
interface LanczosRows {
  readonly alpha: number[]
  readonly beta: number[]
  readonly basis: Float64Array[]
}

/**
 * The Lanczos process for a symmetric operator, with full reorthogonalisation (so the basis stays orthonormal in
 * floating point): m steps produce the tridiagonal T whose eigenvalues (Ritz values) approximate A's extremes. Takes the
 * start vector as a tensor-like and the deflation vectors as the rows of a matrix (migration, D-009: `v0` was
 * `ArrayLike<number>`, `deflate` a list of `Float64Array`s, which is still accepted); the result holds tensors (it
 * was plain arrays); the restarted eigensolvers run on the plain-number kernel.
 *
 * @param a - Symmetric matrix or operator.
 * @param m - Number of steps.
 * @param options - Start vector `[n]` or key, and vectors to stay orthogonal to (deflation) as `[k, n]` rows.
 * @returns T's diagonals (`[m]` tensors) and the basis (an `[m, n]` matrix).
 * @throws MdmlError with code `shape` unless the start vector is a vector and the deflation vectors a matrix, and
 *   `not-differentiable` for a traced value.
 * @concept lanczos
 * @example
 * ```ts
 * import { lanczos } from 'mdml-engine/numerics/sparse'
 * lanczos([[2, 1], [1, 2]], 2).alpha.shape[0] // => 2
 * ```
 */
export function lanczos(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: {
    readonly v0?: TensorLike
    readonly key?: Key
    readonly deflate?: TensorLike | readonly Float64Array[]
  } = {},
): LanczosResult {
  const { alpha, beta, basis } = lanczosRows(a, m, options)
  const n = basis[0]?.length ?? 0
  const q = new Float64Array(basis.length * n)
  basis.forEach((row, i) => {
    q.set(row, i * n)
  })
  return { alpha: vectorOf(alpha), beta: vectorOf(beta), basis: fromStorage(q, [basis.length, n], 'float64') }
}

/** The Lanczos process of {@link lanczos} on plain numbers. */
function lanczosRows(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: {
    readonly v0?: TensorLike
    readonly key?: Key
    readonly deflate?: TensorLike | readonly Float64Array[]
  },
): LanczosRows {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  const deflate = deflationRows(options.deflate)
  let q = orthogonalise(startVector(n, options.v0, options.key), deflate)
  if (norm(q) < 1e-10)
    q = orthogonalise(
      Float64Array.from({ length: n }, (_, i) => Math.cos(3.7 * i + 0.3)),
      deflate,
    )
  q = q.map((v) => v / norm(q))
  const basis: Float64Array[] = [q]
  const alpha: number[] = []
  const beta: number[] = []
  for (let j = 0; j < Math.min(m, n); j++) {
    const w0 = apply(basis[j] as Float64Array)
    alpha.push(dot(w0, basis[j] as Float64Array))
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
  /** The orthonormal basis vectors as the rows of a `[k, n]` matrix (migration, D-009: a list of `Float64Array`s). */
  readonly basis: Tensor<'float64'>
}

/** The paths the Krylov algorithms log, as a type. */
export type KrylovPath = 'krylov/basis' | 'krylov/ritz' | 'krylov/residual' | 'krylov/restart' | 'krylov/locked'

/**
 * What the Arnoldi process and the restarted eigensolvers log: the size of the Krylov basis, the Ritz values (the
 * real parts, for Arnoldi) and their residuals, and for the restarted solvers the restart count and the pairs locked.
 */
export const krylovContract: Contract<unknown, KrylovPath> = defineContract({
  'krylov/basis': scalar({ description: 'the vectors in the Krylov basis (Arnoldi) or in this restart round' }),
  'krylov/ritz': tensor({
    role: 'state',
    shape: ['ritz'],
    description: 'the Ritz values wanted (real parts for Arnoldi)',
  }),
  'krylov/residual': tensor({
    role: 'state',
    shape: ['ritz'],
    description: 'the residual norm of each wanted Ritz pair (Arnoldi: the coupling h[j+1, j] alone)',
  }),
  'krylov/restart': scalar({ description: 'restarts so far' }),
  'krylov/locked': scalar({ description: 'eigenpairs converged and locked so far' }),
})

/**
 * The state of {@link arnoldiAlgorithm}: the orthonormal basis so far and the columns of the Hessenberg matrix (column
 * `j` has `j + 2` entries, the last the norm of the new vector).
 */
export interface ArnoldiState {
  readonly basis: Float64Array[]
  readonly columns: number[][]
  /** True once a new vector had norm below 1e-14: the Krylov space is invariant. */
  readonly breakdown: boolean
}

/**
 * The Arnoldi process as a stepped algorithm, one Krylov vector per step: apply the operator, orthogonalise twice
 * against the basis (the entries are a column of the Hessenberg matrix), normalise. It logs the basis size, the norm of
 * the new vector (the residual coupling) and, on request, the real parts of the Ritz values (the eigenvalues of the
 * Hessenberg matrix so far). It is done when `m` vectors (at most n) are taken (`completed`) or the space is invariant
 * (`converged`). The state serialises.
 *
 * @param a - Matrix or operator.
 * @param m - Number of steps.
 * @param options - Start vector `[n]` (a tensor-like; migration, D-009: was `ArrayLike<number>`) or key.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` unless the start vector is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { arnoldiAlgorithm } from 'mdml-engine/numerics/sparse'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(arnoldiAlgorithm([[1, 2], [0, 3]], 2), { maxSteps: 10 })
 * [run.steps, run.state.basis.length] // => [2, 2]
 * ```
 */
export function arnoldiAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: { readonly v0?: TensorLike; readonly key?: Key } = {},
): Algorithm<ArnoldiState, KrylovPath> {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  const steps = Math.min(m, n)
  return {
    name: 'numerics/arnoldi',
    config: { m, n },
    init: () => {
      const q = startVector(n, options.v0, options.key)
      return { basis: [q.map((v) => v / norm(q))], columns: [], breakdown: false }
    },
    step: (s, ctx) => {
      const j = s.columns.length
      const col = new Array<number>(j + 2).fill(0)
      let w = apply(s.basis[j] as Float64Array)
      for (let pass = 0; pass < 2; pass++) {
        s.basis.forEach((b, i) => {
          const c = dot(w, b)
          col[i] = (col[i] ?? 0) + c
          w = w.map((x, k) => x - c * (b[k] ?? 0))
        })
      }
      const hn = norm(w)
      col[j + 1] = hn
      const basis = hn < 1e-14 ? s.basis : [...s.basis, w.map((v) => v / hn)]
      const columns = [...s.columns, col]
      ctx.log('krylov/basis', basis.length)
      ctx.log('krylov/residual', [hn])
      if (ctx.wants('krylov/ritz')) {
        const sz = columns.length
        const h = new Float64Array(sz * sz)
        columns.forEach((c, jj) => {
          for (let i = 0; i < sz && i <= jj + 1; i++) h[i * sz + jj] = c[i] ?? 0
        })
        ctx.log('krylov/ritz', toFlat(eigvals(fromStorage(h, [sz, sz], 'float64')).re))
      }
      return { basis, columns, breakdown: hn < 1e-14 }
    },
    done: (s) => (s.breakdown ? 'converged' : s.columns.length >= steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ArnoldiState,
  }
}

/**
 * The Arnoldi process for a general operator (with reorthogonalisation): an orthonormal basis of the Krylov space and
 * the Hessenberg matrix of A restricted to it, whose eigenvalues approximate A's. {@link arnoldiAlgorithm} run to the
 * end.
 *
 * @param a - Matrix or operator.
 * @param m - Number of steps.
 * @param options - Start vector `[n]` (a tensor-like; migration, D-009: was `ArrayLike<number>`) or key.
 * @returns The Hessenberg matrix `[m + 1, m]` and the basis (a `[k, n]` matrix, was a list of `Float64Array`s).
 * @throws MdmlError with code `shape` unless the start vector is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { arnoldi } from 'mdml-engine/numerics/sparse'
 * arnoldi([[1, 2], [0, 3]], 2).hessenberg.shape // => [3, 2]
 * ```
 */
export function arnoldi(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: { readonly v0?: TensorLike; readonly key?: Key } = {},
): ArnoldiResult {
  const n = asOperator(a).shape[0]
  const steps = Math.min(m, n)
  const { basis, columns } = runAlgorithm(arnoldiAlgorithm(a, m, options), { maxSteps: steps }).state
  const h = new Float64Array((steps + 1) * steps)
  columns.forEach((col, j) => {
    col.forEach((v, i) => {
      h[i * steps + j] = v
    })
  })
  const q = new Float64Array(basis.length * n)
  basis.forEach((row, i) => {
    q.set(row, i * n)
  })
  return {
    hessenberg: fromStorage(h, [steps + 1, steps], 'float64'),
    basis: fromStorage(q, [basis.length, n], 'float64'),
  }
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

function ritz(a: LanczosRows): { values: number[]; vectors: number[][] } {
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

/** Options of {@link eigsh} and {@link eigshAlgorithm}. */
export interface EigshOptions {
  readonly which?: Which
  readonly tol?: number
  readonly ncv?: number
  readonly maxRestarts?: number
  readonly v0?: ArrayLike<number>
}

/**
 * The state of {@link eigshAlgorithm}: the pairs locked so far, the vector the next round starts from, the Ritz values
 * and residuals of the last round, and the restarts made.
 */
export interface EigshState {
  readonly locked: Float64Array[]
  readonly lockedValues: number[]
  readonly lockedResiduals: number[]
  /** The start vector of the next round (the given one, or a restart vector); empty for the default. */
  readonly start: Float64Array | null
  readonly restarts: number
  /** The wanted Ritz values of the last round (empty before the first). */
  readonly ritzValues: number[]
  /** Their residual norms. */
  readonly ritzResiduals: number[]
  /** The Lanczos basis size of the last round. */
  readonly basisSize: number
  /** True when the restart limit was passed. */
  readonly failed: boolean
}

/**
 * The restarted Lanczos eigensolver of {@link eigsh} as a stepped algorithm: each step is one round, a Lanczos
 * factorisation of m steps (with full reorthogonalisation, staying orthogonal to the locked vectors), the wanted
 * Ritz pairs and their residuals, then either the converged leading pairs are locked, or the round restarts from the
 * wanted Ritz vectors (the leading one weighted most). It logs the basis size, the wanted Ritz values and residuals,
 * the restarts and the pairs locked, and is done (`converged`) when k pairs are locked, `failed` past the restart limit.
 * The state serialises.
 *
 * @param a - Symmetric matrix or operator.
 * @param k - Number of eigenpairs.
 * @param options - Which end (default LM), tolerance (default 1e-10), Krylov size m (default max(2k + 1, 20)), restart
 *   cap and start vector.
 * @returns The algorithm.
 * @throws MdmlError with code `value` unless 1 <= k < n.
 * @concept lanczos
 * @example
 * ```ts
 * import { diagCsr, eigshAlgorithm } from 'mdml-engine/numerics/sparse'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(eigshAlgorithm(diagCsr([1, 5, 3, 9, 2]), 2), { maxSteps: 20 })
 * [run.status, run.state.locked.length] // => ['converged', 2]
 * ```
 */
export function eigshAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  k: number,
  options: EigshOptions = {},
): Algorithm<EigshState, KrylovPath> {
  const op = asOperator(a)
  const apply = numeric(op)
  const n = op.shape[0]
  if (!(k >= 1 && k < n)) throw new MdmlError('value', `eigsh: k must be in [1, ${n - 1}]`)
  const which = options.which ?? 'LM'
  const tol = options.tol ?? 1e-10
  const m = Math.min(n, options.ncv ?? Math.max(2 * k + 1, 20))
  const maxRestarts = options.maxRestarts ?? 300
  const pick = (values: readonly number[]): number[] => {
    const idx = values.map((_, i) => i)
    const score = (i: number): number =>
      which === 'LM' ? Math.abs(values[i] ?? 0) : which === 'LA' ? (values[i] ?? 0) : -(values[i] ?? 0)
    return idx.sort((p, q) => score(q) - score(p)).slice(0, k)
  }
  const converged = (r: number, lambda: number): boolean => r <= tol * Math.max(1, Math.abs(lambda))
  return {
    name: 'numerics/eigsh',
    config: { k, which, tol, ncv: m, maxRestarts },
    init: () => ({
      locked: [],
      lockedValues: [],
      lockedResiduals: [],
      start: options.v0 === undefined ? null : Float64Array.from(options.v0),
      restarts: 0,
      ritzValues: [],
      ritzResiduals: [],
      basisSize: 0,
      failed: false,
    }),
    step: (s, ctx) => {
      // Lanczos with locking: converged Ritz pairs are locked and later runs stay orthogonal to them, so repeated
      // eigenvalues are found once per copy.
      const locked = [...s.locked]
      const lockedValues = [...s.lockedValues]
      const lockedResiduals = [...s.lockedResiduals]
      const fact = lanczosRows(op, Math.min(m, n - locked.length), {
        ...(s.start === null ? {} : { v0: s.start }),
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
        locked.push(vecs[i] as Float64Array)
        lockedValues.push(lambdas[i] ?? 0)
        lockedResiduals.push(residuals[i] ?? 0)
        lockedNow++
      }
      let restarts = s.restarts
      let start: Float64Array | null = null
      let failed = false
      if (lockedNow === 0) {
        restarts++
        if (restarts > maxRestarts) failed = true
        // Restart from the wanted Ritz vectors, the leading one weighted most.
        const next = new Float64Array(n)
        vecs.forEach((v, i) => {
          const w = 1 / (i + 1)
          for (let r = 0; r < n; r++) next[r] = (next[r] ?? 0) + w * (v[r] ?? 0)
        })
        start = next
      }
      ctx.log('krylov/basis', fact.basis.length)
      ctx.log('krylov/ritz', lambdas)
      ctx.log('krylov/residual', residuals)
      ctx.log('krylov/restart', restarts)
      ctx.log('krylov/locked', locked.length)
      return {
        locked,
        lockedValues,
        lockedResiduals,
        start,
        restarts,
        ritzValues: lambdas,
        ritzResiduals: residuals,
        basisSize: fact.basis.length,
        failed,
      }
    },
    done: (s) => (s.failed ? 'failed' : s.locked.length >= k ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EigshState,
  }
}

/**
 * The k extreme eigenpairs of a symmetric operator (scipy.sparse.linalg.eigsh): Lanczos with full
 * reorthogonalisation in blocks of m steps, explicitly restarted from the wanted Ritz vectors, with converged pairs
 * (residual below tol |lambda|) locked and deflated so that repeated eigenvalues appear once per copy.
 * {@link eigshAlgorithm} run to the end.
 *
 * @param a - Symmetric matrix or operator.
 * @param k - Number of eigenpairs.
 * @param options - Which end (default LM), tolerance (default 1e-10), Krylov size m (default max(2k + 1, 20)), restart
 *   cap and start vector.
 * @returns Eigenvalues (ascending), eigenvectors and residuals.
 * @throws MdmlError with code `convergence` past the restart limit.
 * @concept lanczos
 * @index function Extreme eigenpairs of a symmetric operator by Lanczos.
 * @example
 * ```ts
 * import { diagCsr, eigsh } from 'mdml-engine/numerics/sparse'
 * Array.from(eigsh(diagCsr([1, 5, 3, 9, 2]), 2).values.data).map((v) => Math.round(v * 1e10) / 1e10) // => [5, 9]
 * ```
 */
export function eigsh(a: CsrMatrix | TensorLike | LinearOperator, k: number, options: EigshOptions = {}): EigshResult {
  const n = asOperator(a).shape[0]
  const run = runAlgorithm(eigshAlgorithm(a, k, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  if (run.state.failed) throw new MdmlError('convergence', 'eigsh: no convergence within the restart limit')
  const { locked, lockedValues, lockedResiduals, restarts } = run.state
  const order = lockedValues.map((_, i) => i).sort((p, q) => (lockedValues[p] ?? 0) - (lockedValues[q] ?? 0))
  const out = new Float64Array(n * k)
  order.forEach((src, col) => {
    const v = locked[src] as Float64Array
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

/** The Gram operator A^T A of a matrix with a transpose product, and the products that build singular triplets. */
function gramOf(op: LinearOperator): {
  gram: LinearOperator
  apply: (x: Float64Array) => Float64Array
  m: number
  n: number
} {
  const [m, n] = op.shape
  const apply = numeric(op)
  const applyT = numeric(op, true)
  const gram: LinearOperator = {
    shape: [n, n],
    matvec: (x) => fromStorage(applyT(apply(Float64Array.from(toFlat(primalOf(x))))), [n], 'float64'),
  }
  return { gram, apply, m, n }
}

/**
 * The largest singular triplets as a stepped algorithm: {@link eigshAlgorithm} on A^T A for the largest eigenvalues (the
 * squares of the singular values), one restart round per step, with the same state and paths. The triplets are built
 * from the final state by {@link svds}.
 *
 * @param a - Matrix or operator with a transpose product.
 * @param k - Number of triplets.
 * @param options - Tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { svdsAlgorithm } from 'mdml-engine/numerics/sparse'
 * svdsAlgorithm([[3, 0, 0], [0, 4, 0], [0, 0, 1]], 1).name // => 'numerics/svds'
 * ```
 */
export function svdsAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  k: number,
  options: { readonly tol?: number } = {},
): Algorithm<EigshState, KrylovPath> {
  const { gram } = gramOf(asOperator(a))
  return { ...eigshAlgorithm(gram, k, { which: 'LA', tol: options.tol ?? 1e-12 }), name: 'numerics/svds' }
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
 * import { svds } from 'mdml-engine/numerics/sparse'
 * Array.from(svds([[3, 0, 0], [0, 4, 0], [0, 0, 1]], 1).s.data).map((v) => Math.round(v * 1e10) / 1e10) // => [4]
 * ```
 */
export function svds(
  a: CsrMatrix | TensorLike | LinearOperator,
  k: number,
  options: { readonly tol?: number } = {},
): { s: Tensor; u: Tensor; vt: Tensor } {
  const { gram, apply, m, n } = gramOf(asOperator(a))
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
