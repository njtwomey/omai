/**
 * Pairwise squared Euclidean distances for the stationary kernels, as a differentiable primitive that avoids the
 * `[m, n, d]` difference tensor of `cdist`. Private to the kernels domain.
 *
 * The kernel is one fused loop over the two point sets, sum_p (a_ip - b_jp)^2, the arithmetic of the broadcast form
 * without its three `[m, n, d]` intermediates, so its values are those of `cdist` to the bit: never negative, exactly
 * zero for coincident points, and for a set with itself (computed on the upper triangle and mirrored) exactly symmetric
 * with an exactly zero diagonal. The expansion |a|^2 + |b|^2 - 2 a.b through a matrix product, with a shared centre
 * and an exact recomputation of small results, was measured and is accurate (relative error 1e-15 on well separated
 * pairs) but, with this engine's matrix product, 1.5 to 2 times slower than the fused loop up to 50 dimensions, so the
 * loop is what runs.
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  type Aval,
  arg,
  bind1,
  concat,
  definePrimitive,
  expandDims,
  eye,
  type MaybeZero,
  matmul,
  mean,
  mul,
  type Primitive,
  stopGradient,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type Value,
} from '../tensor/index.ts'
import { safeSqrt } from './_safe.ts'

/**
 * Squared Euclidean distances between the rows of `a` `[m, d]` and `b` `[n, d]` (`b` is `a` itself when `same`), as a
 * row-major `[m, n]` array.
 *
 * @param a - Rows of the first set.
 * @param b - Rows of the second set.
 * @param m - Rows of a.
 * @param n - Rows of b.
 * @param d - Coordinates.
 * @param same - Whether b is a, so only the upper triangle is computed and mirrored.
 * @returns The distances.
 */
export function squaredDistanceArrays(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  m: number,
  n: number,
  d: number,
  same: boolean,
): Float64Array {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    const ia = i * d
    for (let j = same ? i : 0; j < n; j++) {
      const jb = j * d
      let s = 0
      for (let p = 0; p < d; p++) {
        const t = (a[ia + p] ?? 0) - (b[jb + p] ?? 0)
        s += t * t
      }
      out[i * n + j] = s
      if (same) out[j * n + i] = s
    }
  }
  return out
}

interface DistanceParams {
  /** Whether both inputs are the same point set (a Gram matrix). */
  readonly same: boolean
}

/** The centre of both point sets (the mean of all rows), a constant for the derivative rules. */
function centreOf(a: Value, b: Value, same: boolean): Value {
  return stopGradient(mean(same ? a : concat([a, b], 0), 0))
}

/** 1 off the diagonal, 0 on it: a Gram matrix has no dependence on the points on its diagonal. */
function offDiagonal(n: number): Value {
  return sub(1, eye(n))
}

const squaredDistancePrimitive: Primitive<DistanceParams> = definePrimitive<DistanceParams>({
  name: 'squaredDistance',
  abstractEval: (inputs: readonly Aval[]) => {
    const a = arg(inputs, 0)
    const b = arg(inputs, 1)
    return [{ shape: [a.shape[0] ?? 0, b.shape[0] ?? 0], dtype: a.dtype }]
  },
  impl: ([a, b], { same }) => {
    const x = a as Tensor
    const y = b as Tensor
    const m = x.shape[0] ?? 0
    const n = y.shape[0] ?? 0
    const out = squaredDistanceArrays(x.data, y.data, m, n, x.shape[1] ?? 0, same)
    if (x.dtype === 'float64') return [x.backend.wrap('float64', [m, n], out)]
    const narrowed = new Float32Array(out)
    return [x.backend.wrap('float32', [m, n], narrowed)]
  },
  // dD_ij = 2 (a_i - b_j) . (da_i - db_j), written with the centred points (the centre cancels) and a matrix product.
  jvp: ([a, b], [ta, tb], { same }) => {
    const c = centreOf(a as Value, b as Value, same)
    const ac = sub(a as Value, c)
    const bc = sub(b as Value, c)
    let out: MaybeZero = null
    if (ta !== null && ta !== undefined) {
      const own = expandDims(sum(mul(ac, ta), -1), 1)
      out = mul(sub(own, matmul(ta, swapaxes(bc, 0, 1))), 2)
    }
    if (tb !== null && tb !== undefined) {
      const own = expandDims(sum(mul(bc, tb), -1), 0)
      const term = mul(sub(own, matmul(ac, swapaxes(tb, 0, 1))), 2)
      out = out === null ? term : add(out, term)
    }
    if (out === null || !same) return [out]
    return [mul(out, offDiagonal((a as Value).shape[0] ?? 0))]
  },
  // dA_i = 2 sum_j G_ij (a_i - b_j) and dB_j = 2 sum_i G_ij (b_j - a_i), again with the centred points.
  vjp: ([g], [a, b], { same }) => {
    if (g === null || g === undefined) return [null, null]
    const c = centreOf(a as Value, b as Value, same)
    const ac = sub(a as Value, c)
    const bc = sub(b as Value, c)
    const cot = same ? mul(g, offDiagonal(g.shape[0] ?? 0)) : g
    const rows = expandDims(sum(cot, 1), 1)
    const cols = expandDims(sum(cot, 0), 1)
    const da = mul(sub(mul(rows, ac), matmul(cot, bc)), 2)
    const db = mul(sub(mul(cols, bc), matmul(swapaxes(cot, 0, 1), ac)), 2)
    return [da, db]
  },
  describe: ({ same }) => (same ? 'gram' : 'cross'),
})

/**
 * Squared Euclidean distances between the rows of `a` `[m, d]` and of `b` `[n, d]` (`[m, m]` of `a` with itself when
 * `b` is omitted: exactly symmetric, exactly zero on the diagonal), differentiable in both. Values agree with the
 * broadcast difference form to the relative accuracy of the sum itself; see the module comment.
 *
 * @param a - Rows `[m, d]`.
 * @param b - Rows `[n, d]`, or omitted for `a`.
 * @returns The squared distances `[m, n]`.
 */
export function squaredDistances(a: Value, b?: Value): Value {
  const other = b ?? a
  if (a.shape.length !== 2 || other.shape.length !== 2 || a.shape[1] !== other.shape[1])
    throw new MdmlError('shape', 'squaredDistances: inputs must be [m, d] and [n, d]')
  return bind1(squaredDistancePrimitive, [a, other], { same: b === undefined || b === a })
}

/**
 * Euclidean distances between the rows (zero value and zero gradient at coincident points, as `cdist`).
 *
 * @param a - Rows `[m, d]`.
 * @param b - Rows `[n, d]`, or omitted for `a`.
 * @returns The distances `[m, n]`.
 */
export function euclideanDistances(a: Value, b?: Value): Value {
  return safeSqrt(squaredDistances(a, b))
}
