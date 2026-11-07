/**
 * The NaN rule at the boundary of the sparse solvers and eigensolvers: a matrix given as CSR or dense is scanned once for
 * a non-finite entry and the routine throws `value` naming itself; an operator is opaque and is not scanned (a
 * non-finite result of its product shows as a `diverged` run). Private to the folder.
 */

import { primalOf, promoteFloat, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { assertFinite } from '../_vector.ts'
import { type CsrMatrix, type LinearOperator, primalValues } from './csr.ts'

/**
 * Throw a `value` error when a CSR or dense matrix has a NaN or infinite entry; do nothing for an operator.
 *
 * @param name - The calling function.
 * @param a - The matrix or operator.
 */
export function assertFiniteOperand(name: string, a: CsrMatrix | TensorLike | LinearOperator): void {
  if (typeof a === 'object' && a !== null && 'matvec' in a) return
  if (typeof a === 'object' && a !== null && 'indptr' in a) assertFinite(name, primalValues(a as CsrMatrix), 'matrix')
  else assertFinite(name, toFlat(primalOf(promoteFloat(a as TensorLike)[0] as Value)), 'matrix')
}
