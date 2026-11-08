/**
 * Dense matrices of the samplers as row-major Float64Arrays: validation, and Cholesky factors and inverses computed
 * once (through numerics/linalg) so that per-iteration work is plain dgemv.
 */
import { OmaiError } from '../base/index.ts'
import { choleskyJitter, inv } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** A square matrix `[dim, dim]` as a row-major buffer, checked for shape and finiteness. */
export function squareMatrix(x: TensorLike, dim: number, name: string): Float64Array {
  const t = asTensor(x) as Tensor
  if (t.shape.length !== 2 || t.shape[0] !== dim || t.shape[1] !== dim) {
    throw new OmaiError('shape', `${name}: expected a [${dim}, ${dim}] matrix, got [${t.shape.join(', ')}]`)
  }
  const a = Float64Array.from(toFlat(t))
  if (!a.every(Number.isFinite)) throw new OmaiError('value', `${name}: the matrix must be finite`)
  return a
}

/** A vector of length `dim`, or a scalar broadcast to one. */
export function vectorOf(x: TensorLike | number, dim: number, name: string): Float64Array {
  if (typeof x === 'number') return new Float64Array(dim).fill(x)
  const v = Float64Array.from(toFlat(asTensor(x)))
  if (v.length === 1 && dim > 1) return new Float64Array(dim).fill(v[0] ?? 0)
  if (v.length !== dim) throw new OmaiError('shape', `${name}: expected ${dim} values, got ${v.length}`)
  return v
}

/** The lower Cholesky factor of a symmetric positive-definite matrix (with jitter for near-singular ones). */
export function lowerCholesky(a: Float64Array, dim: number, name: string): Float64Array {
  const r = choleskyJitter(fromStorage(Float64Array.from(a), [dim, dim], 'float64'))
  const l = Float64Array.from(toFlat(r.l))
  if (!l.every(Number.isFinite)) throw new OmaiError('value', `${name}: the matrix is not positive definite`)
  return l
}

/** The inverse of a square matrix. */
export function inverse(a: Float64Array, dim: number): Float64Array {
  return Float64Array.from(toFlat(inv(fromStorage(Float64Array.from(a), [dim, dim], 'float64'))))
}

export { allFinite } from '../base/index.ts'
