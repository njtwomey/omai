/**
 * The Lanczos process as a traced algorithm, one Krylov vector per step: the tridiagonal matrix grows by a row, the
 * Ritz values (its eigenvalues) converge from the extremes of the spectrum inwards, and with the reorthogonalisation
 * switched off the loss of orthogonality of the basis, and the ghost copies of converged eigenvalues it brings, show
 * in the state. The state serialises.
 */

import { type Json, maxOf, OmaiError } from '../../base/index.ts'
import { type Key, normal } from '../../random/index.ts'
import { fromStorage, primalOf, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { eighKernel } from '../linalg/_kernels.ts'
import { ddot as dot, dnrm2 as norm } from '../linalg/blas.ts'
import { asOperator, type CsrMatrix, type LinearOperator } from './csr.ts'

/** Options of the stepped Lanczos process. */
export interface LanczosStepsOptions {
  /** Orthogonalise each new vector against every earlier one (`'full'`, default) or only use the three-term recurrence (`'none'`). */
  readonly reorthogonalise?: 'full' | 'none'
  /** The starting vector (default: Gaussian from the run's key). */
  readonly v0?: ArrayLike<number>
}

/**
 * The state after `j` steps: the tridiagonal `T_j` (diagonal `alpha`, off-diagonal `beta`; the last `beta` couples to
 * the newest basis vector), the basis `q_1 .. q_{j+1}` (each a vector of length n), the Ritz values ascending with the
 * residual norm `beta_j |s_j|` of each (the last component of its eigenvector of `T_j`), and the departure of the basis
 * from orthonormality `max |Q^T Q - I|`.
 */
export interface LanczosStepsState {
  readonly alpha: readonly number[]
  readonly beta: readonly number[]
  readonly basis: readonly (readonly number[])[]
  readonly ritzValues: readonly number[]
  readonly ritzResiduals: readonly number[]
  readonly orthogonalityLoss: number
  /** True when the Krylov space is exhausted (beta is zero: an invariant subspace was found). */
  readonly invariant: boolean
}

const SCHEMA = channels({
  metrics: {
    orthogonalityLoss: { description: 'largest departure of the basis from orthonormality', direction: 'lower' },
    maxResidual: { description: 'largest residual norm of a Ritz pair', direction: 'lower' },
    beta: { description: 'the newest off-diagonal entry of T' },
  },
  tensors: {
    ritzValues: { role: 'state', description: 'the eigenvalues of the tridiagonal matrix (variable length)' },
    ritzResiduals: { role: 'state', description: 'the residual norm of each Ritz pair' },
    alpha: { role: 'state', description: 'the diagonal of T' },
    beta: { role: 'state', description: 'the off-diagonal of T' },
  },
})

function ritzOf(alpha: readonly number[], beta: readonly number[]): { values: number[]; residuals: number[] } {
  const j = alpha.length
  const t = new Float64Array(j * j)
  for (let i = 0; i < j; i++) {
    t[i * j + i] = alpha[i] ?? 0
    if (i + 1 < j) {
      t[i * j + i + 1] = beta[i] ?? 0
      t[(i + 1) * j + i] = beta[i] ?? 0
    }
  }
  const { w, v } = eighKernel(t, j)
  const last = beta[j - 1] ?? 0
  return {
    values: Array.from(w),
    residuals: Array.from({ length: j }, (_, c) => Math.abs(last * (v[(j - 1) * j + c] ?? 0))),
  }
}

function orthogonalityLossOf(basis: readonly (readonly number[])[]): number {
  let worst = 0
  for (let i = 0; i < basis.length; i++)
    for (let k = i; k < basis.length; k++)
      worst = Math.max(worst, Math.abs(dot(basis[i] as number[], basis[k] as number[]) - (i === k ? 1 : 0)))
  return worst
}

/**
 * The Lanczos process for a symmetric operator as a traced algorithm: step `j` multiplies the newest basis vector by
 * the operator, forms `alpha_j`, subtracts the components along the last two (or all) basis vectors, and normalises the
 * remainder to the next basis vector with `beta_j` as its norm. After `j` steps the Ritz values are the eigenvalues of
 * `T_j`; `reorthogonalise: 'none'` is the textbook three-term recurrence, whose basis loses orthogonality once a Ritz
 * value has converged.
 *
 * @param a - A symmetric matrix or operator.
 * @param options - Reorthogonalisation and the starting vector.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { lanczosAlgorithm } from 'omai-engine/numerics/sparse'
 * lanczosAlgorithm([[2, 1], [1, 2]], { v0: [1, 0] }).name // => 'lanczos'
 * ```
 */
export function lanczosAlgorithm(
  a: CsrMatrix | TensorLike | LinearOperator,
  options: LanczosStepsOptions = {},
): Algorithm<LanczosStepsState> {
  const op = asOperator(a)
  const n = op.shape[0]
  if (op.shape[1] !== n) throw new OmaiError('shape', 'lanczosAlgorithm needs a square operator')
  const apply = (x: ArrayLike<number>): Float64Array =>
    Float64Array.from(toFlat(primalOf(op.matvec(fromStorage(Float64Array.from(x), [n], 'float64')))))
  const mode = options.reorthogonalise ?? 'full'
  return {
    name: 'lanczos',
    config: { n, reorthogonalise: mode } as Json,
    schema: SCHEMA,
    init: (key: Key) => {
      const v0 = options.v0 !== undefined ? Float64Array.from(options.v0) : Float64Array.from(normal(key, [n]).data)
      if (v0.length !== n) throw new OmaiError('shape', 'lanczosAlgorithm: v0 must have one entry per row')
      const nv = norm(v0)
      if (!(nv > 0)) throw new OmaiError('value', 'lanczosAlgorithm: v0 must be nonzero')
      return {
        alpha: [],
        beta: [],
        basis: [Array.from(v0, (x) => x / nv)],
        ritzValues: [],
        ritzResiduals: [],
        orthogonalityLoss: 0,
        invariant: false,
      }
    },
    step: (s) => {
      const j = s.alpha.length
      const q = s.basis[j] as number[]
      let w = apply(q)
      const alpha = dot(w, q)
      if (mode === 'full') {
        for (let pass = 0; pass < 2; pass++)
          for (const b of s.basis) {
            const c = dot(w, b as number[])
            w = w.map((x, i) => x - c * (b[i] ?? 0))
          }
      } else {
        const prev = j > 0 ? (s.basis[j - 1] as number[]) : undefined
        const bPrev = j > 0 ? (s.beta[j - 1] ?? 0) : 0
        w = w.map((x, i) => x - alpha * (q[i] ?? 0) - (prev === undefined ? 0 : bPrev * (prev[i] ?? 0)))
      }
      const b = norm(w)
      const alphas = [...s.alpha, alpha]
      const betas = [...s.beta, b]
      const invariant = b < 1e-14 * Math.max(1, Math.abs(alpha))
      const basis = invariant || alphas.length >= n ? s.basis : [...s.basis, Array.from(w, (x) => x / b)]
      const { values, residuals } = ritzOf(alphas, betas)
      return {
        alpha: alphas,
        beta: betas,
        basis,
        ritzValues: values,
        ritzResiduals: residuals,
        orthogonalityLoss: orthogonalityLossOf(basis),
        invariant: invariant || alphas.length >= n,
      }
    },
    done: (s) => (s.invariant ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        orthogonalityLoss: s.orthogonalityLoss,
        maxResidual: Math.max(0, maxOf(s.ritzResiduals)),
        beta: s.beta[s.beta.length - 1] ?? 0,
      },
      tensors: {
        ritzValues: { role: 'state' as const, value: s.ritzValues },
        ritzResiduals: { role: 'state' as const, value: s.ritzResiduals },
        alpha: { role: 'state' as const, value: s.alpha },
        beta: { role: 'state' as const, value: s.beta },
      },
    }),
    serialise: (s) =>
      ({
        ...s,
        alpha: [...s.alpha],
        beta: [...s.beta],
        basis: s.basis.map((b) => [...b]),
        ritzValues: [...s.ritzValues],
        ritzResiduals: [...s.ritzResiduals],
      }) as Json,
    deserialise: (j) => j as unknown as LanczosStepsState,
  }
}

/**
 * Run the Lanczos process for `m` steps (or until the Krylov space is exhausted) and return its final state.
 *
 * @param a - A symmetric matrix or operator.
 * @param m - The number of steps.
 * @param options - Reorthogonalisation, the starting vector, seed or key.
 * @returns The state: `T`'s diagonals, the basis and the Ritz values with their residuals.
 * @example
 * ```ts
 * import { lanczosSteps } from 'omai-engine/numerics/sparse'
 * lanczosSteps([[2, 1], [1, 2]], 2, { v0: [1, 0] }).ritzValues // => [1, 3]
 * ```
 */
export function lanczosSteps(
  a: CsrMatrix | TensorLike | LinearOperator,
  m: number,
  options: LanczosStepsOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): LanczosStepsState {
  const { reorthogonalise, v0, ...run } = options
  const algorithm = lanczosAlgorithm(a, {
    ...(reorthogonalise === undefined ? {} : { reorthogonalise }),
    ...(v0 === undefined ? {} : { v0 }),
  })
  return runAlgorithm(algorithm, { ...run, maxSteps: m }).state
}
