import { WhiteboxError } from '../../base/index.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { broadcastShapes, formatShape, normaliseAxis } from '../shape.ts'
import type { TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { coerce2, val } from './_coerce.ts'
import { add, mul } from './arithmetic.ts'
import { take } from './indexing.ts'
import { sum } from './reduce.ts'
import { expandDims, moveaxis, reshape, squeeze, sumToShape, swapaxes, transpose } from './shape.ts'

/** Batched matrix product of operands with rank at least 2 (batch axes broadcast). @internal */
export const matmulPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'matmul',
  abstractEval: (xs) => {
    const a = arg(xs, 0)
    const b = arg(xs, 1)
    const batch = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2))
    return [{ shape: [...batch, a.shape[a.shape.length - 2] ?? 1, b.shape[b.shape.length - 1] ?? 1], dtype: a.dtype }]
  },
  impl: (xs) => [arg(xs, 0).backend.matmul(arg(xs, 0), arg(xs, 1))],
  jvp: (xs, ts) => {
    const a = arg(xs, 0)
    const b = arg(xs, 1)
    const ta = arg(ts, 0)
    const tb = arg(ts, 1)
    const da = ta === null ? null : bind1(matmulPrimitive, [ta, b], null)
    const db = tb === null ? null : bind1(matmulPrimitive, [a, tb], null)
    return [da === null ? db : db === null ? da : add(da, db)]
  },
  vjp: (gs, xs) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null]
    const a = arg(xs, 0)
    const b = arg(xs, 1)
    const ga = bind1(matmulPrimitive, [g, swapaxes(b, -1, -2)], null)
    const gb = bind1(matmulPrimitive, [swapaxes(a, -1, -2), g], null)
    return [sumToShape(ga, a.shape), sumToShape(gb, b.shape)]
  },
})

/**
 * Matrix product with numpy semantics: batch axes broadcast; a 1-D left operand is a row vector and a 1-D right operand
 * is a column vector (the added axis is removed from the result).
 *
 * @param a - Left operand `[..., m, k]` or `[k]`.
 * @param b - Right operand `[..., k, n]` or `[k]`.
 * @returns The product.
 * @throws WhiteboxError with code `shape` when the inner dimensions differ.
 * @example
 * ```ts
 * import { matmul, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(matmul(tensor([[1, 2], [3, 4]]), tensor([[1], [1]]))) // => [[3], [7]]
 * toArray(matmul(tensor([1, 2]), tensor([3, 4]))) // => 11
 * ```
 */
export function matmul(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length === 0 || y.shape.length === 0)
    throw new WhiteboxError('shape', 'matmul: operands must not be scalars')
  const xa = x.shape.length === 1 ? expandDims(x, 0) : x
  const yb = y.shape.length === 1 ? expandDims(y, -1) : y
  const k1 = xa.shape[xa.shape.length - 1]
  const k2 = yb.shape[yb.shape.length - 2]
  if (k1 !== k2) {
    throw new WhiteboxError('shape', `matmul: shapes ${formatShape(x.shape)} and ${formatShape(y.shape)} do not align`)
  }
  let out = bind1(matmulPrimitive, [xa, yb], null)
  if (y.shape.length === 1) out = squeeze(out, -1)
  if (x.shape.length === 1) out = squeeze(out, out.shape.length === 1 && y.shape.length === 1 ? 0 : -2)
  return out
}

/**
 * Dot product of vectors, or matrix product for 2-D operands.
 *
 * @param a - First operand (1-D or 2-D).
 * @param b - Second operand (1-D or 2-D).
 * @returns The product.
 * @example
 * ```ts
 * import { dot, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(dot(tensor([1, 2, 3]), tensor([4, 5, 6]))) // => 32
 * ```
 */
export function dot(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length > 2 || y.shape.length > 2)
    throw new WhiteboxError('shape', 'dot: use matmul or tensordot above 2-D')
  if (x.shape.length === 0 || y.shape.length === 0) return mul(x, y)
  return matmul(x, y)
}

/**
 * Dot product of flattened operands.
 *
 * @param a - First operand.
 * @param b - Second operand with the same number of elements.
 * @returns The scalar sum of elementwise products.
 * @example
 * ```ts
 * import { vdot, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(vdot(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [1, 1]]))) // => 10
 * ```
 */
export function vdot(a: TensorLike, b: TensorLike): Value {
  return dot(reshape(a, [-1]), reshape(b, [-1]))
}

/**
 * Outer product of two vectors (flattened).
 *
 * @param a - First vector.
 * @param b - Second vector.
 * @returns The matrix `a_i b_j`.
 * @example
 * ```ts
 * import { outer, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(outer(tensor([1, 2]), tensor([3, 4]))) // => [[3, 4], [6, 8]]
 * ```
 */
export function outer(a: TensorLike, b: TensorLike): Value {
  return mul(expandDims(reshape(a, [-1]), -1), expandDims(reshape(b, [-1]), 0))
}

/**
 * Inner product over the last axes (numpy `inner`).
 *
 * @param a - First operand.
 * @param b - Second operand with the same last-axis length.
 * @returns The contraction.
 * @example
 * ```ts
 * import { inner, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(inner(tensor([[1, 2], [3, 4]]), tensor([1, 1]))) // => [3, 7]
 * ```
 */
export function inner(a: TensorLike, b: TensorLike): Value {
  return tensordot(a, b, [[-1], [-1]])
}

/**
 * Contract axes of two operands (numpy `tensordot`), by transposing and reshaping into one matrix product.
 *
 * @param a - First operand.
 * @param b - Second operand.
 * @param axes - Number of trailing axes of `a` against leading axes of `b`, or explicit axis lists.
 * @returns The contraction: `a`'s free axes followed by `b`'s.
 * @example
 * ```ts
 * import { tensordot, ones } from 'whitebox-engine/tensor'
 * tensordot(ones([2, 3, 4]), ones([3, 4, 5]), 2).shape // => [2, 5]
 * ```
 */
export function tensordot(
  a: TensorLike,
  b: TensorLike,
  axes: number | readonly [readonly number[], readonly number[]] = 2,
): Value {
  const [x, y] = coerce2(a, b)
  const [ax, bx] =
    typeof axes === 'number'
      ? [Array.from({ length: axes }, (_, i) => x.shape.length - axes + i), Array.from({ length: axes }, (_, i) => i)]
      : axes
  const ca = ax.map((i) => normaliseAxis(i, x.shape.length))
  const cb = bx.map((i) => normaliseAxis(i, y.shape.length))
  if (ca.length !== cb.length || ca.some((i, k) => x.shape[i] !== y.shape[cb[k] ?? -1])) {
    throw new WhiteboxError('shape', 'tensordot: contracted axes do not match')
  }
  const freeA = x.shape.map((_, i) => i).filter((i) => !ca.includes(i))
  const freeB = y.shape.map((_, i) => i).filter((i) => !cb.includes(i))
  const k = ca.reduce((p, i) => p * (x.shape[i] ?? 1), 1)
  const ma = transpose(x, [...freeA, ...ca])
  const mb = transpose(y, [...cb, ...freeB])
  const out = matmul(reshape(ma, [-1, k]), reshape(mb, [k, -1]))
  return reshape(out, [...freeA.map((i) => x.shape[i] ?? 1), ...freeB.map((i) => y.shape[i] ?? 1)])
}

/**
 * Einstein summation over any number of operands (`'ij,jk->ik'`). Each operand is aligned to a common index order by
 * transposition and broadcasting, multiplied, and summed over indices absent from the output; so it is differentiable
 * like any composition of primitives.
 *
 * @param spec - The subscripts, with an explicit `->` output.
 * @param operands - The operands.
 * @returns The result.
 * @throws WhiteboxError with code `value` for malformed subscripts.
 * @example
 * ```ts
 * import { einsum, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(einsum('ij,jk->ik', tensor([[1, 2]]), tensor([[3], [4]]))) // => [[11]]
 * toArray(einsum('ii->', tensor([[1, 2], [3, 4]]))) // => 5
 * ```
 */
export function einsum(spec: string, ...operands: TensorLike[]): Value {
  const [lhs, rhs] = spec.replace(/\s/g, '').split('->')
  if (lhs === undefined || rhs === undefined)
    throw new WhiteboxError('value', `einsum: need an explicit output in '${spec}'`)
  const inputs = lhs.split(',')
  if (inputs.length !== operands.length)
    throw new WhiteboxError('value', 'einsum: operand count does not match subscripts')
  const letters = [...new Set([...inputs.join(''), ...rhs])]
  const sizes = new Map<string, number>()
  const aligned = inputs.map((subs, k) => {
    let v = val(operands[k] as TensorLike)
    if (subs.length !== v.shape.length)
      throw new WhiteboxError('value', `einsum: '${subs}' does not match rank ${v.shape.length}`)
    // Repeated letters in one operand take the diagonal.
    let s = subs
    for (let i = 0; i < s.length; i++) {
      const j = s.indexOf(s[i] as string, i + 1)
      if (j > i) {
        v = diagonal(v, 0, i, j)
        s = s.slice(0, i) + s.slice(i + 1, j) + s.slice(j + 1) + s[i]
        i = -1
      }
    }
    ;[...s].forEach((c, i) => {
      const n = v.shape[i] ?? 1
      const known = sizes.get(c)
      if (known !== undefined && known !== n && known !== 1 && n !== 1) {
        throw new WhiteboxError('shape', `einsum: index '${c}' has lengths ${known} and ${n}`)
      }
      sizes.set(c, Math.max(known ?? 1, n))
    })
    const present = letters.filter((c) => s.includes(c))
    const permuted = transpose(
      v,
      present.map((c) => s.indexOf(c)),
    )
    return reshape(
      permuted,
      letters.map((c) => (s.includes(c) ? (v.shape[s.indexOf(c)] ?? 1) : 1)),
    )
  })
  let product = aligned[0] as Value
  for (const v of aligned.slice(1)) product = mul(product, v)
  const summed = letters.map((c, i) => (rhs.includes(c) ? -1 : i)).filter((i) => i >= 0)
  let out = summed.length > 0 ? sum(product, summed) : product
  const remaining = letters.filter((c) => rhs.includes(c))
  out = transpose(
    out,
    [...rhs].map((c) => remaining.indexOf(c)),
  )
  return out
}

/**
 * Kronecker product of two matrices.
 *
 * @param a - First matrix `[m, n]`.
 * @param b - Second matrix `[p, q]`.
 * @returns The `[m p, n q]` block matrix.
 * @example
 * ```ts
 * import { kron, eye, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(kron(eye(2), tensor([[1, 2]]))) // => [[1, 2, 0, 0], [0, 0, 1, 2]]
 * ```
 */
export function kron(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length !== 2 || y.shape.length !== 2) throw new WhiteboxError('shape', 'kron: expects matrices')
  const [m, n] = x.shape as [number, number]
  const [p, q] = y.shape as [number, number]
  const prod = mul(reshape(x, [m, 1, n, 1]), reshape(y, [1, p, 1, q]))
  return reshape(prod, [m * p, n * q])
}

/**
 * Diagonal of a matrix (or of the two given axes), with an optional offset, as the last axis of the result.
 *
 * @param x - Input with at least two axes.
 * @param offset - Diagonal offset (positive above the main diagonal).
 * @param axis1 - First axis (default 0).
 * @param axis2 - Second axis (default 1).
 * @returns The diagonal.
 * @example
 * ```ts
 * import { diagonal, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(diagonal(tensor([[1, 2], [3, 4]]))) // => [1, 4]
 * ```
 */
export function diagonal(x: TensorLike, offset = 0, axis1 = 0, axis2 = 1): Value {
  const v = val(x)
  const rank = v.shape.length
  const a1 = normaliseAxis(axis1, rank)
  const a2 = normaliseAxis(axis2, rank)
  const moved = moveaxis(moveaxis(v, a1, -1), a2 > a1 ? a2 - 1 : a2, -1)
  const n = moved.shape[rank - 2] ?? 0
  const m = moved.shape[rank - 1] ?? 0
  const flat = reshape(moved, [...moved.shape.slice(0, -2), n * m])
  const indices: number[] = []
  for (let i = Math.max(0, -offset); i < n && i + offset < m; i++) indices.push(i * m + i + offset)
  return take(flat, indices, -1)
}

/**
 * Sum of the diagonal of a matrix.
 *
 * @param x - A matrix.
 * @param offset - Diagonal offset.
 * @returns The trace.
 * @example
 * ```ts
 * import { trace, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(trace(tensor([[1, 2], [3, 4]]))) // => 5
 * ```
 */
export function trace(x: TensorLike, offset = 0): Value {
  return sum(diagonal(x, offset), -1)
}
