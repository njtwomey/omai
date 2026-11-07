/**
 * Primitives, the operations everything else is built from: `definePrimitive` declares one with its kernel, its `jvp`
 * and `vjp` derivative rules and an optional batch rule, and `bind` applies it, dispatching to whichever transform
 * (reverse mode, forward mode, batching) is in progress. Symbolic zeros (`MaybeZero`) keep zero tangents and cotangents
 * from being materialised.
 */

import { MdmlError } from '../base/index.ts'
import type { Tensor } from './tensor.ts'
import { type Aval, isTraced, lift, type Trace, type Traced, type Value } from './traced.ts'

/** A cotangent or tangent that may be a symbolic zero (`null`), so zeros are never materialised needlessly. */
export type MaybeZero = Value | null

/** The batch axis of a value inside `vmap`: an axis of the underlying value, or `null` when every example shares it. */
export type BatchDim = number | null

/** What a batch rule returns: the batched outputs and the axis of each that indexes the examples. */
export interface BatchResult {
  readonly outputs: readonly Value[]
  readonly dims: readonly BatchDim[]
}

/**
 * The definition of a primitive operation: everything other operations are built from.
 *
 * - `abstractEval` gives output shapes and dtypes without computing anything.
 * - `impl` computes on concrete tensors (through the inputs' backend).
 * - `jvp` maps input tangents to output tangents; `vjp` maps output cotangents to input cotangents. Both are written
 *   with operations, so they are themselves differentiable (higher-order derivatives work). Reverse mode passes `vjp` a
 *   fifth argument, `needs`: `needs[j]` is `false` when nothing tracked depends on input `j` (a constant, or data), so
 *   its cotangent would be dropped; a rule may return `null` for those inputs without computing them, or ignore `needs`
 *   (a cotangent returned for an input that needs none is discarded).
 * - A primitive without `jvp`/`vjp` is not differentiable: differentiating through it raises `not-differentiable`,
 *   unless `zeroDerivative` says its derivative is zero everywhere it exists (comparisons, rounding, argmax).
 * - `batch` applies the primitive to a whole batch at once inside `vmap`: it receives the underlying inputs, the batch
 *   axis of each (`null` for inputs shared by every example; at least one is batched), the parameters and the batch
 *   size. Without it, `vmap` falls back to a loop over the examples (slice, apply, stack), which is always correct.
 *
 * @concept primitives
 */
export interface Primitive<P> {
  readonly name: string
  readonly abstractEval: (inputs: readonly Aval[], params: P) => readonly Aval[]
  readonly impl: (inputs: readonly Tensor[], params: P) => readonly Tensor[]
  readonly jvp?: (
    primals: readonly Value[],
    tangents: readonly MaybeZero[],
    params: P,
    outputs: readonly Value[],
  ) => readonly MaybeZero[]
  readonly vjp?: (
    cotangents: readonly MaybeZero[],
    inputs: readonly Value[],
    params: P,
    outputs: readonly Value[],
    needs?: readonly boolean[],
  ) => readonly MaybeZero[]
  readonly batch?: (inputs: readonly Value[], dims: readonly BatchDim[], params: P, size: number) => BatchResult
  readonly zeroDerivative?: boolean
  /**
   * Whether the primitive is linear (or affine) in its inputs jointly, as `add`, `sum`, `reshape` or `slice` are:
   * Taylor mode then maps each coefficient of the input series through the `jvp` rule instead of needing a rule of its
   * own.
   */
  readonly linear?: boolean
  /** Describe the parameters for computation-graph labels. */
  readonly describe?: (params: P) => string
}

/**
 * Define a primitive. This is the extension point for new differentiable operations (special functions,
 * factorisations, solvers): give a kernel and derivative rules, and every transformation handles it.
 *
 * @param spec - The primitive's definition.
 * @returns The primitive, to be applied with {@link bind}.
 * @concept primitives
 * @index function Declare a new operation with its kernel and derivative rules.
 * @example
 * ```ts
 * import { definePrimitive, bind1, tensor, mul, toArray } from 'mdml-engine/tensor'
 * const triple = definePrimitive<null>({
 *   name: 'triple',
 *   abstractEval: ([x]) => [x],
 *   impl: ([x]) => [x.backend.map(x, (v) => 3 * v, x.dtype)],
 *   jvp: (_p, [t]) => [t === null ? null : mul(t, 3)],
 *   vjp: ([g]) => [g === null ? null : mul(g, 3)],
 * })
 * toArray(bind1(triple, [tensor([1, 2])], null)) // => [3, 6]
 * ```
 */
export function definePrimitive<P>(spec: Primitive<P>): Primitive<P> {
  return Object.freeze({ ...spec })
}

/**
 * Apply a primitive. With concrete inputs it runs the kernel; with traced inputs it hands the operation to the trace of
 * the highest level among them, lifting the other inputs into that trace.
 *
 * @param primitive - The primitive.
 * @param inputs - Its inputs.
 * @param params - Its static parameters.
 * @returns Its outputs.
 * @concept primitives
 * @example
 * ```ts
 * import { bind, tensor } from 'mdml-engine/tensor'
 * import { addPrimitive } from 'mdml-engine/tensor'
 * bind(addPrimitive, [tensor(1), tensor(2)], null)[0].shape // => []
 * ```
 */
export function bind<P>(primitive: Primitive<P>, inputs: readonly Value[], params: P): readonly Value[] {
  let top: Trace | null = null
  for (const x of inputs) if (isTraced(x) && (top === null || x.trace.level > top.level)) top = x.trace
  if (top === null) return primitive.impl(inputs as readonly Tensor[], params)
  const trace = top
  const lifted: Traced[] = inputs.map((x) => lift(trace, x))
  return trace.process(primitive, lifted, params)
}

/**
 * Apply a single-output primitive (see {@link bind}).
 *
 * @param primitive - The primitive.
 * @param inputs - Its inputs.
 * @param params - Its static parameters.
 * @returns Its only output.
 * @example
 * ```ts
 * import { bind1, tensor, toArray, addPrimitive } from 'mdml-engine/tensor'
 * toArray(bind1(addPrimitive, [tensor([1]), tensor([2])], null)) // => [3]
 * ```
 */
export function bind1<P>(primitive: Primitive<P>, inputs: readonly Value[], params: P): Value {
  const out = bind(primitive, inputs, params)
  const first = out[0]
  if (out.length !== 1 || first === undefined) {
    throw new MdmlError('value', `primitive ${primitive.name} has ${out.length} outputs; use bind`)
  }
  return first
}

/**
 * The `i`th element of a primitive's inputs, outputs or (co)tangents, with a clear error if it is missing. Rules use
 * this instead of destructuring so every access is checked.
 *
 * @param xs - The list.
 * @param i - The index.
 * @returns The element.
 * @throws MdmlError with code `value` when the list is too short.
 * @example
 * ```ts
 * import { arg } from 'mdml-engine/tensor'
 * arg([10, 20], 1) // => 20
 * ```
 */
export function arg<T>(xs: readonly T[], i: number): T {
  if (i >= xs.length) throw new MdmlError('value', `expected at least ${i + 1} arguments, got ${xs.length}`)
  return xs[i] as T
}
