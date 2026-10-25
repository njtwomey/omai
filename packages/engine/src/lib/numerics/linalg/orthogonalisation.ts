/**
 * Orthogonalisation: classical and modified Gram-Schmidt (optionally repeated) as a differentiable thin QR
 * factorisation, with per-projection step records for figures, and `orth`, an orthonormal basis of a matrix's range
 * from its SVD (scipy.linalg).
 */

import { maxOf, WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  div,
  item,
  matmul,
  mul,
  norm,
  primalOf,
  promoteFloat,
  sliceAxis,
  stack,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { svd } from './svd.ts'

/** Options of {@link gramSchmidt}. */
export interface GramSchmidtOptions {
  /** `modified` (default) subtracts projections one at a time against the updated vector; `classical` all at once. */
  readonly method?: 'classical' | 'modified'
  /** Orthogonalise twice ("twice is enough", Kahan and Parlett), restoring orthogonality lost to rounding. */
  readonly reorthogonalise?: boolean
}

/** Result of {@link gramSchmidt}. */
export interface GramSchmidtResult {
  readonly q: Value
  readonly r: Value
  /** The loss of orthogonality ||Q^T Q - I|| (Frobenius), which shows classical Gram-Schmidt's instability. */
  readonly orthogonalityError: number
  /** One record per projection: column j against q_i in a pass, with the coefficient and the vector after it. */
  readonly steps: readonly GramSchmidtStep[]
}

/** One projection step of Gram-Schmidt, for figures. */
export interface GramSchmidtStep {
  readonly column: number
  readonly against: number
  readonly pass: number
  /** The projection coefficient r_ij of this pass. */
  readonly coefficient: number
  /** The vector after subtracting the projection. */
  readonly vector: readonly number[]
}

const column = (a: Value, j: number): Value => sliceAxis(a, 1, j, j + 1)

/**
 * Gram-Schmidt orthogonalisation of the columns of a full-column-rank matrix, classical or modified, optionally
 * twice, as a thin QR factorisation (built from operations, so differentiable).
 *
 * @param a - Matrix `[m, n]` with m >= n and independent columns.
 * @param options - Method and reorthogonalisation.
 * @returns Q, R, the orthogonality error and the projection steps.
 * @concept orthogonalisation
 * @example
 * ```ts
 * import { gramSchmidt } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(gramSchmidt([[3, 1], [4, 1]]).r)[0] // => [5, 1.4]
 * ```
 */
export function gramSchmidt(a: TensorLike, options: GramSchmidtOptions = {}): GramSchmidtResult {
  const [m] = promoteFloat(a) as [Value]
  const n = m.shape[1] ?? 0
  const modified = (options.method ?? 'modified') === 'modified'
  const passes = options.reorthogonalise === true ? 2 : 1
  const qs: Value[] = []
  const r: Value[][] = Array.from({ length: n }, () => [])
  const steps: GramSchmidtStep[] = []
  for (let j = 0; j < n; j++) {
    let v = column(m, j)
    const coefficients: (Value | undefined)[] = new Array(j).fill(undefined)
    for (let pass = 0; pass < passes; pass++) {
      // Classical Gram-Schmidt projects the vector as it was at the start of the pass; modified projects the
      // vector as it is after each subtraction.
      const start = v
      for (let i = 0; i < j; i++) {
        const qi = qs[i] as Value
        const rij = sum(mul(qi, modified ? v : start))
        v = sub(v, mul(qi, rij))
        steps.push({ column: j, against: i, pass, coefficient: item(primalOf(rij)), vector: toFlat(primalOf(v)) })
        coefficients[i] = coefficients[i] === undefined ? rij : add(coefficients[i] as Value, rij)
      }
    }
    const rjj = norm(v)
    if (!(item(primalOf(rjj)) > 0)) throw new WhiteboxError('value', 'gramSchmidt: the columns are linearly dependent')
    qs.push(div(v, rjj))
    for (let i = 0; i < n; i++) r[i]?.push(i < j ? (coefficients[i] as Value) : i === j ? rjj : mul(rjj, 0))
  }
  const q = concat(qs, 1)
  const rm = stack(
    r.map((row) => stack(row, 0)),
    0,
  )
  const gram = toFlat(primalOf(matmul(swapaxes(q, 0, 1), q)))
  const err = Math.sqrt(gram.reduce((s2, v, k) => s2 + (v - (Math.floor(k / n) === k % n ? 1 : 0)) ** 2, 0))
  return { q, r: rm, orthogonalityError: err, steps }
}

/**
 * An orthonormal basis of the range of a matrix (scipy.linalg.orth): the left singular vectors of the singular values
 * above `rcond` times the largest.
 *
 * @param a - Matrix `[m, n]`.
 * @param rcond - Relative cutoff (default max(m, n) eps).
 * @returns The basis `[m, rank]`.
 * @example
 * ```ts
 * import { orth } from 'whitebox-engine/numerics/linalg'
 * orth([[1, 1], [1, 1]]).shape // => [2, 1]
 * ```
 */
export function orth(a: TensorLike, rcond?: number): Value {
  const [m] = promoteFloat(a) as [Value]
  const { u, s } = svd(m)
  const values = toFlat(primalOf(s))
  const tol = (rcond ?? maxOf(m.shape) * 2.220446049250313e-16) * (values[0] ?? 0)
  const rank = values.filter((v) => v > tol).length
  return sliceAxis(u, 1, 0, rank)
}
