/**
 * Linear algebra basics on tensors: `matmul` with numpy batching semantics, dot, inner and outer products, `tensordot`,
 * `einsum` over any number of operands, Kronecker products, diagonals and traces. Factorisations and solvers live in
 * `numerics/linalg`.
 *
 * @index function matmul, dot, tensordot, einsum and Kronecker products with batching.
 */

import { OmaiError } from '../../base/index.ts'
import { alignBatched } from '../batching.ts'
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
  vjp: (gs, xs, _p, _outs, needs) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null]
    const a = arg(xs, 0)
    const b = arg(xs, 1)
    const ga = needs?.[0] === false ? null : bind1(matmulPrimitive, [g, swapaxes(b, -1, -2)], null)
    const gb = needs?.[1] === false ? null : bind1(matmulPrimitive, [swapaxes(a, -1, -2), g], null)
    return [ga === null ? null : sumToShape(ga, a.shape), gb === null ? null : sumToShape(gb, b.shape)]
  },
  // Batch axes of a product broadcast, so a batched operand only needs its batch in front of its own batch axes.
  batch: (xs, ds) => {
    const { values, dim } = alignBatched(xs, ds, 2)
    return { outputs: [bind1(matmulPrimitive, values, null)], dims: [dim] }
  },
})

/**
 * Matrix product with numpy semantics: batch axes broadcast; a 1-D left operand is a row vector and a 1-D right operand
 * is a column vector (the added axis is removed from the result).
 *
 * @param a - Left operand `[..., m, k]` or `[k]`.
 * @param b - Right operand `[..., k, n]` or `[k]`.
 * @returns The product.
 * @throws OmaiError with code `shape` when the inner dimensions differ.
 * @example
 * ```ts
 * import { matmul, tensor, toArray } from 'omai-engine/tensor'
 * toArray(matmul(tensor([[1, 2], [3, 4]]), tensor([[1], [1]]))) // => [[3], [7]]
 * toArray(matmul(tensor([1, 2]), tensor([3, 4]))) // => 11
 * ```
 */
export function matmul(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length === 0 || y.shape.length === 0) throw new OmaiError('shape', 'matmul: operands must not be scalars')
  const xa = x.shape.length === 1 ? expandDims(x, 0) : x
  const yb = y.shape.length === 1 ? expandDims(y, -1) : y
  const k1 = xa.shape[xa.shape.length - 1]
  const k2 = yb.shape[yb.shape.length - 2]
  if (k1 !== k2) {
    throw new OmaiError('shape', `matmul: shapes ${formatShape(x.shape)} and ${formatShape(y.shape)} do not align`)
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
 * import { dot, tensor, toArray } from 'omai-engine/tensor'
 * toArray(dot(tensor([1, 2, 3]), tensor([4, 5, 6]))) // => 32
 * ```
 */
export function dot(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length > 2 || y.shape.length > 2) throw new OmaiError('shape', 'dot: use matmul or tensordot above 2-D')
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
 * import { vdot, tensor, toArray } from 'omai-engine/tensor'
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
 * import { outer, tensor, toArray } from 'omai-engine/tensor'
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
 * import { inner, tensor, toArray } from 'omai-engine/tensor'
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
 * import { tensordot, ones } from 'omai-engine/tensor'
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
    throw new OmaiError('shape', 'tensordot: contracted axes do not match')
  }
  const freeA = x.shape.map((_, i) => i).filter((i) => !ca.includes(i))
  const freeB = y.shape.map((_, i) => i).filter((i) => !cb.includes(i))
  const k = ca.reduce((p, i) => p * (x.shape[i] ?? 1), 1)
  const ma = transpose(x, [...freeA, ...ca])
  const mb = transpose(y, [...cb, ...freeB])
  const out = matmul(reshape(ma, [-1, k]), reshape(mb, [k, -1]))
  return reshape(out, [...freeA.map((i) => x.shape[i] ?? 1), ...freeB.map((i) => y.shape[i] ?? 1)])
}

/** An einsum operand with a name for each of its axes. */
interface Term {
  readonly v: Value
  readonly s: readonly string[]
}

const product = (xs: readonly number[]): number => xs.reduce((p, n) => p * n, 1)

// Contract two terms with one matrix product. Letters in both and still needed (the output or a later operand) are
// batch axes, letters in both and not needed are summed by the product, and letters in one only that are not needed
// are summed first. The result's axes are the batch letters, then a's own, then b's own.
function contractPair(a: Term, b: Term, keep: ReadonlySet<string>, sizes: ReadonlyMap<string, number>): Term {
  const presum = (t: Term, other: Term): Term => {
    const drop = t.s.flatMap((c, i) => (other.s.includes(c) || keep.has(c) ? [] : [i]))
    return drop.length === 0 ? t : { v: sum(t.v, drop), s: t.s.filter((_, i) => !drop.includes(i)) }
  }
  const x = presum(a, b)
  const y = presum(b, a)
  const batch = x.s.filter((c) => y.s.includes(c) && keep.has(c))
  const inner = x.s.filter((c) => y.s.includes(c) && !keep.has(c))
  const left = x.s.filter((c) => !y.s.includes(c))
  const right = y.s.filter((c) => !x.s.includes(c))
  const dims = (cs: readonly string[]): number[] => cs.map((c) => sizes.get(c) ?? 1)
  const arrange = (t: Term, order: readonly string[], shape: readonly number[]): Value => {
    const perm = order.map((c) => t.s.indexOf(c))
    return reshape(perm.every((p, i) => p === i) ? t.v : transpose(t.v, perm), shape)
  }
  const batched = batch.length > 0
  const lead = batched ? [product(dims(batch))] : []
  const m = product(dims(left))
  const k = product(dims(inner))
  const n = product(dims(right))
  const xa = arrange(x, [...batch, ...left, ...inner], [...lead, m, k])
  const yb = arrange(y, [...batch, ...inner, ...right], [...lead, k, n])
  const letters = [...batch, ...left, ...right]
  return { v: reshape(matmul(xa, yb), dims(letters)), s: letters }
}

/**
 * Einstein summation over any number of operands (`'ij,jk->ik'`). Operands are contracted pairwise from the left, each
 * pair by one matrix product (so the cost is that of the contraction, not the product of every index extent), with
 * indices that nothing else needs summed as early as possible; it is differentiable like any composition of
 * primitives. Operands whose shared indices differ in length by broadcasting a length of 1, and degenerate lengths of
 * 0, are instead aligned to a common index order, multiplied and summed.
 *
 * @param spec - The subscripts, with an explicit `->` output.
 * @param operands - The operands.
 * @returns The result.
 * @throws OmaiError with code `value` for malformed subscripts.
 * @example
 * ```ts
 * import { einsum, tensor, toArray } from 'omai-engine/tensor'
 * toArray(einsum('ij,jk->ik', tensor([[1, 2]]), tensor([[3], [4]]))) // => [[11]]
 * toArray(einsum('ii->', tensor([[1, 2], [3, 4]]))) // => 5
 * ```
 */
export function einsum(spec: string, ...operands: TensorLike[]): Value {
  const [lhs, rhs] = spec.replace(/\s/g, '').split('->')
  if (lhs === undefined || rhs === undefined)
    throw new OmaiError('value', `einsum: need an explicit output in '${spec}'`)
  const inputs = lhs.split(',')
  if (inputs.length !== operands.length) throw new OmaiError('value', 'einsum: operand count does not match subscripts')
  const letters = [...new Set([...inputs.join(''), ...rhs])]
  const sizes = new Map<string, number>()
  const terms = inputs.map((subs, k): Term => {
    let v = val(operands[k] as TensorLike)
    if (subs.length !== v.shape.length)
      throw new OmaiError('value', `einsum: '${subs}' does not match rank ${v.shape.length}`)
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
        throw new OmaiError('shape', `einsum: index '${c}' has lengths ${known} and ${n}`)
      }
      sizes.set(c, Math.max(known ?? 1, n))
    })
    return { v, s: [...s] }
  })
  const pairwise =
    terms.length >= 2 &&
    new Set(rhs).size === rhs.length &&
    [...rhs].every((c) => terms.some((t) => t.s.includes(c))) &&
    terms.every((t) => t.s.every((c, i) => (t.v.shape[i] ?? 1) === sizes.get(c) && (sizes.get(c) ?? 0) > 0))
  let out: Value
  let remaining: readonly string[]
  if (pairwise) {
    let acc = terms[0] as Term
    for (let i = 1; i < terms.length; i++) {
      const keep = new Set([...rhs, ...terms.slice(i + 1).flatMap((t) => t.s)])
      acc = contractPair(acc, terms[i] as Term, keep, sizes)
    }
    out = acc.v
    remaining = acc.s
  } else {
    const aligned = terms.map(({ v, s }) => {
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
    let prod = aligned[0] as Value
    for (const v of aligned.slice(1)) prod = mul(prod, v)
    const summed = letters.map((c, i) => (rhs.includes(c) ? -1 : i)).filter((i) => i >= 0)
    out = summed.length > 0 ? sum(prod, summed) : prod
    remaining = letters.filter((c) => rhs.includes(c))
  }
  return transpose(
    out,
    [...rhs].map((c) => remaining.indexOf(c)),
  )
}

/**
 * Kronecker product of two matrices.
 *
 * @param a - First matrix `[m, n]`.
 * @param b - Second matrix `[p, q]`.
 * @returns The `[m p, n q]` block matrix.
 * @example
 * ```ts
 * import { kron, eye, tensor, toArray } from 'omai-engine/tensor'
 * toArray(kron(eye(2), tensor([[1, 2]]))) // => [[1, 2, 0, 0], [0, 0, 1, 2]]
 * ```
 */
export function kron(a: TensorLike, b: TensorLike): Value {
  const [x, y] = coerce2(a, b)
  if (x.shape.length !== 2 || y.shape.length !== 2) throw new OmaiError('shape', 'kron: expects matrices')
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
 * import { diagonal, tensor, toArray } from 'omai-engine/tensor'
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
 * import { trace, tensor, toArray } from 'omai-engine/tensor'
 * toArray(trace(tensor([[1, 2], [3, 4]]))) // => 5
 * ```
 */
export function trace(x: TensorLike, offset = 0): Value {
  return sum(diagonal(x, offset), -1)
}
