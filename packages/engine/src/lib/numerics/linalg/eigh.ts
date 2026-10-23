import {
  add,
  arg,
  bind,
  definePrimitive,
  div,
  expandDims,
  eye,
  type FloatDType,
  matmul,
  mul,
  type Primitive,
  promoteFloat,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { eighKernel } from './_kernels.ts'
import { assemble, diagEmbed, diagPart, matrices, mT, squareSize, sym } from './_matrix.ts'

/** F_ij = 1 / (w_j - w_i) off the diagonal and 0 on it (infinite for repeated eigenvalues, where V is not unique). */
export function eigengapInverse(w: Value): Value {
  const n = w.shape[w.shape.length - 1] ?? 0
  const gaps = sub(expandDims(w, -2), expandDims(w, -1))
  const onDiagonal = eye(n)
  const safe = where(onDiagonal, 1, gaps)
  return where(onDiagonal, 0, div(1, safe))
}

const eighPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'eigh',
  abstractEval: (xs) => {
    const a = arg(xs, 0)
    return [{ shape: a.shape.slice(0, -1), dtype: a.dtype }, a]
  },
  impl: ([a]) => {
    const av = a as NonNullable<typeof a>
    const n = squareSize(av.shape, 'eigh')
    const batch = av.shape.slice(0, -2)
    const results = matrices(av).map((m) => eighKernel(m, n))
    const dtype = av.dtype as FloatDType
    return [
      assemble(
        av,
        dtype,
        batch,
        [n],
        results.map((r) => r.w),
      ),
      assemble(
        av,
        dtype,
        batch,
        [n, n],
        results.map((r) => r.v),
      ),
    ]
  },
  // With M = V^T sym(dA) V: dw = diag(M), dV = V (F o M).
  jvp: (_xs, [ta], _p, [w, v]) => {
    if (ta === null || ta === undefined) return [null, null]
    const vv = v as Value
    const m = matmul(matmul(mT(vv), sym(ta)), vv)
    return [diagPart(m), matmul(vv, mul(eigengapInverse(w as Value), m))]
  },
  // dA = sym(V (diag(dw) + F o (V^T dV)) V^T).
  vjp: ([gw, gv], _xs, _p, [w, v]) => {
    const vv = v as Value
    const hasW = gw !== null && gw !== undefined
    const hasV = gv !== null && gv !== undefined
    if (!hasW && !hasV) return [null]
    const fromV = hasV ? mul(eigengapInverse(w as Value), matmul(mT(vv), gv)) : undefined
    const fromW = hasW ? diagEmbed(gw) : undefined
    const inner = fromV === undefined ? (fromW as Value) : fromW === undefined ? fromV : add(fromW, fromV)
    return [sym(matmul(matmul(vv, inner), mT(vv)))]
  },
})

/** Result of {@link eigh}. */
export interface EighResult {
  /** Eigenvalues in ascending order `[..., n]`. */
  readonly eigenvalues: Value
  /** Orthonormal eigenvectors as columns `[..., n, n]`, each with its largest-magnitude entry positive. */
  readonly eigenvectors: Value
}

/**
 * Eigendecomposition of a symmetric matrix by the cyclic Jacobi method (numpy.linalg.eigh), of its symmetric part.
 * Eigenvector signs are fixed (largest-magnitude entry positive), where LAPACK leaves them arbitrary.
 *
 * @param a - Symmetric matrix `[..., n, n]`.
 * @returns `{ eigenvalues, eigenvectors }`, differentiable (eigenvectors only where eigenvalues are distinct).
 * @concept linear-algebra
 * @example
 * ```ts
 * import { eigh } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(eigh([[2, 1], [1, 2]]).eigenvalues) // => [1, 3]
 * ```
 */
export function eigh(a: TensorLike): EighResult {
  const [av] = promoteFloat(a) as [Value]
  const [eigenvalues, eigenvectors] = bind(eighPrimitive, [av], null) as [Value, Value]
  return { eigenvalues, eigenvectors }
}

/**
 * Eigenvalues of a symmetric matrix, ascending (numpy.linalg.eigvalsh).
 *
 * @param a - Symmetric matrix `[..., n, n]`.
 * @returns The eigenvalues, differentiable (also at repeated eigenvalues, for symmetric functions of them).
 * @example
 * ```ts
 * import { eigvalsh } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(eigvalsh([[4, 0], [0, 1]])) // => [1, 4]
 * ```
 */
export function eigvalsh(a: TensorLike): Value {
  return eigh(a).eigenvalues
}
