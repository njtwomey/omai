/**
 * Helpers for writing batch rules, the `batch` field of a primitive that `vmap` uses: moving and aligning batch axes,
 * broadcasting shared operands, and ready-made rules for elementwise primitives and for those that already accept
 * leading batch axes (factorisations and solvers). Every export is a function declaration, so operation modules can use
 * them inside their primitive definitions whatever the module evaluation order.
 */

import { MdmlError } from '../base/index.ts'
import { broadcastTo, expandDims, moveaxis, reshape } from './ops/shape.ts'
import { type BatchDim, type BatchResult, bind, type Primitive } from './primitive.ts'
import { normaliseAxis } from './shape.ts'
import type { Value } from './traced.ts'

/**
 * Put the batch axis of a value at a given position. A value without a batch axis (`dim` is `null`) is broadcast to
 * one of length `size`, so the result always has the batch at `to`.
 *
 * @param x - The underlying value.
 * @param dim - Its batch axis, or `null`.
 * @param to - Where the batch axis goes (negative counts from the end of the result).
 * @param size - The batch size.
 * @returns The value with its batch axis at `to`.
 * @concept vectorisation
 * @example
 * ```ts
 * import { moveBatchAxis, zeros } from 'mdml-engine/tensor'
 * moveBatchAxis(zeros([2, 5]), 1, 0, 5).shape // => [5, 2]
 * moveBatchAxis(zeros([2]), null, 0, 3).shape // => [3, 2]
 * ```
 */
export function moveBatchAxis(x: Value, dim: BatchDim, to: number, size: number): Value {
  if (dim === null) {
    const at = normaliseAxis(to, x.shape.length + 1)
    const shape = [...x.shape]
    shape.splice(at, 0, size)
    return broadcastTo(expandDims(x, at), shape)
  }
  const at = normaliseAxis(to, x.shape.length)
  return dim === at ? x : moveaxis(x, dim, at)
}

/**
 * The axis of the underlying value that a per-example axis refers to, given the batch axis.
 *
 * @param axis - A non-negative axis of one example.
 * @param dim - The batch axis of the underlying value.
 * @returns The corresponding axis of the underlying value.
 * @concept vectorisation
 * @example
 * ```ts
 * import { batchedAxis } from 'mdml-engine/tensor'
 * batchedAxis(0, 0) // => 1
 * batchedAxis(0, 1) // => 0
 * ```
 */
export function batchedAxis(axis: number, dim: number): number {
  return axis < dim ? axis : axis + 1
}

/**
 * Bring every input to a batch axis at the front, broadcasting the shared ones, for rules that need all operands
 * batched alike (factorisations, gathers, concatenation).
 *
 * @param inputs - The underlying inputs.
 * @param dims - Their batch axes.
 * @param size - The batch size.
 * @returns The inputs, each with the batch on axis 0.
 * @concept vectorisation
 * @example
 * ```ts
 * import { batchAtFront, zeros } from 'mdml-engine/tensor'
 * batchAtFront([zeros([2, 4]), zeros([2])], [1, null], 4).map((x) => x.shape) // => [[4, 2], [4, 2]]
 * ```
 */
export function batchAtFront(inputs: readonly Value[], dims: readonly BatchDim[], size: number): Value[] {
  return inputs.map((x, i) => moveBatchAxis(x, dims[i] ?? null, 0, size))
}

/**
 * Align the operands of a broadcasting operation: batched operands get their batch axis in front and length-1 axes
 * so that their example axes line up with the shared operands', which broadcast unchanged. When every operand is
 * already batched on the same axis with the same rank, nothing moves.
 *
 * @param inputs - The underlying inputs.
 * @param dims - Their batch axes (at least one is not `null`).
 * @param coreRank - Trailing axes that must not hold the batch axis (2 for matrix products, 0 for elementwise).
 * @returns The aligned inputs and the batch axis of the result.
 * @throws MdmlError with code `value` when no input is batched.
 * @concept vectorisation
 * @example
 * ```ts
 * import { alignBatched, zeros } from 'mdml-engine/tensor'
 * const { values, dim } = alignBatched([zeros([3, 2]), zeros([4, 2])], [0, null])
 * values[0].shape // => [3, 1, 2]
 * dim // => 0
 * ```
 */
export function alignBatched(
  inputs: readonly Value[],
  dims: readonly BatchDim[],
  coreRank = 0,
): { readonly values: Value[]; readonly dim: number } {
  let rank = 0
  let first: number | null = null
  let same = true
  inputs.forEach((x, i) => {
    const d = dims[i] ?? null
    rank = Math.max(rank, d === null ? x.shape.length : x.shape.length - 1)
    if (d === null) same = false
    else if (first === null) first = d
    else if (first !== d) same = false
  })
  if (first === null) throw new MdmlError('value', 'alignBatched: no input is batched')
  const d0: number = first
  const full = (x: Value, i: number): boolean => dims[i] === null || x.shape.length === rank + 1
  if (same && inputs.every(full) && d0 < rank + 1 - coreRank) return { values: [...inputs], dim: d0 }
  if (d0 === 0 && inputs.every((x, i) => dims[i] === null || (dims[i] === 0 && x.shape.length === rank + 1))) {
    return { values: [...inputs], dim: 0 }
  }
  const values = inputs.map((x, i) => {
    const d = dims[i] ?? null
    if (d === null) return x
    const front = d === 0 ? x : moveaxis(x, d, 0)
    const missing = rank + 1 - front.shape.length
    if (missing === 0) return front
    const shape = [...front.shape]
    shape.splice(1, 0, ...new Array<number>(missing).fill(1))
    return reshape(front, shape)
  })
  return { values, dim: 0 }
}

/**
 * The batch rule of an elementwise primitive with broadcasting (any number of operands): align the operands and apply
 * the primitive once.
 *
 * @param primitive - The primitive.
 * @param inputs - The underlying inputs.
 * @param dims - Their batch axes.
 * @param params - The primitive's parameters.
 * @returns The batched outputs (every output carries the same batch axis).
 * @concept vectorisation
 * @example
 * ```ts
 * import { batchElementwise, addPrimitive, tensor, toArray } from 'mdml-engine/tensor'
 * const r = batchElementwise(addPrimitive, [tensor([[1, 2], [3, 4]]), tensor([10, 20])], [0, null], null)
 * toArray(r.outputs[0]) // => [[11, 22], [13, 24]]
 * ```
 */
export function batchElementwise<P>(
  primitive: Primitive<P>,
  inputs: readonly Value[],
  dims: readonly BatchDim[],
  params: P,
): BatchResult {
  const { values, dim } = alignBatched(inputs, dims)
  const outputs = bind(primitive, values, params)
  return { outputs, dims: outputs.map(() => dim) }
}

/**
 * The batch rule of a primitive that already accepts leading batch axes (factorisations and solvers on `[..., m, n]`
 * stacks): move every input's batch to the front, broadcasting shared inputs, and apply the primitive once.
 *
 * @param primitive - The primitive.
 * @param inputs - The underlying inputs.
 * @param dims - Their batch axes.
 * @param params - The primitive's parameters.
 * @param size - The batch size.
 * @returns The batched outputs, each with the batch on axis 0.
 * @concept vectorisation
 * @example
 * ```ts
 * import { batchLeading, matmulPrimitive, eye, zeros } from 'mdml-engine/tensor'
 * batchLeading(matmulPrimitive, [zeros([3, 2, 2]), eye(2)], [0, null], null, 3).outputs[0].shape // => [3, 2, 2]
 * ```
 */
export function batchLeading<P>(
  primitive: Primitive<P>,
  inputs: readonly Value[],
  dims: readonly BatchDim[],
  params: P,
  size: number,
): BatchResult {
  const outputs = bind(primitive, batchAtFront(inputs, dims, size), params)
  return { outputs, dims: outputs.map(() => 0) }
}
