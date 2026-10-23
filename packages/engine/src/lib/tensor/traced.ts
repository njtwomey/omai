import { WhiteboxError } from '../base/index.ts'
import type { DType } from './dtype.ts'
import type { Primitive } from './primitive.ts'
import type { Shape } from './shape.ts'
import type { Tensor } from './tensor.ts'

/** The abstract value of an array: what is known about it without its data. */
export interface Aval {
  readonly shape: Shape
  readonly dtype: DType
}

/**
 * A transformation in progress (reverse mode, forward mode, batching). Traces form a stack by `level`: the trace with
 * the highest level among an operation's inputs processes it, and lifts lower-level inputs into itself as constants.
 * Implementations live in `whitebox-engine/autodiff`; the tensor layer only dispatches to them.
 *
 * @concept function-transformations
 */
export interface Trace {
  readonly level: number
  /** Turn a lower-level value into a value of this trace (a constant with respect to it). */
  lift(x: Value): Traced
  /** Apply a primitive to values of this trace. */
  process<P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[]
}

/**
 * A value standing in for an array inside a transformation. It knows its shape and dtype; its trace knows the rest.
 *
 * @concept function-transformations
 */
export interface Traced<D extends DType = DType, S extends Shape = Shape> {
  readonly kind: 'traced'
  readonly dtype: D
  readonly shape: S
  readonly trace: Trace
}

/** Anything an operation accepts and returns: a concrete tensor, or a traced value inside a transformation. */
export type Value<D extends DType = DType, S extends Shape = Shape> = Tensor<D, S> | Traced<D, S>

/**
 * Whether a value is traced (inside a transformation) rather than concrete.
 *
 * @param x - Any value.
 * @returns `true` for traced values.
 * @example
 * ```ts
 * import { isTraced, tensor } from 'whitebox-engine/tensor'
 * isTraced(tensor([1, 2])) // => false
 * ```
 */
export function isTraced(x: unknown): x is Traced {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'traced'
}

/**
 * The abstract value (shape and dtype) of a value.
 *
 * @param x - A tensor or traced value.
 * @returns Its shape and dtype.
 * @example
 * ```ts
 * import { avalOf, zeros } from 'whitebox-engine/tensor'
 * avalOf(zeros([2, 3])) // => { shape: [2, 3], dtype: 'float64' }
 * ```
 */
export function avalOf(x: Value): Aval {
  return { shape: x.shape, dtype: x.dtype }
}

/**
 * Require a concrete tensor, with a clear error where an operation cannot work on traced values (for example when a
 * shape would depend on data).
 *
 * @param x - A value.
 * @param where - What needed the concrete value, for the message.
 * @returns The tensor.
 * @throws WhiteboxError with code `not-differentiable` when `x` is traced.
 * @example
 * ```ts
 * import { concreteOrThrow, tensor } from 'whitebox-engine/tensor'
 * concreteOrThrow(tensor([1]), 'example').shape // => [1]
 * ```
 */
export function concreteOrThrow<D extends DType>(x: Value<D>, where: string): Tensor<D> {
  if (isTraced(x)) {
    throw new WhiteboxError('not-differentiable', `${where} needs a concrete value, but received a traced value`, {
      shape: x.shape,
    })
  }
  return x
}
