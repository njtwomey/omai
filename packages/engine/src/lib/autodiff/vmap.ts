/**
 * Vectorisation (like `jax.vmap`): the batching trace, in which each value carries the axis that indexes its examples,
 * and `vmap`, which maps a function over an axis of its arguments by applying each primitive's batch rule once for the
 * whole batch (looping only for primitives without one).
 */

import { MdmlError } from '../base/index.ts'
import {
  type BatchDim,
  type BatchResult,
  bind,
  isTraced,
  moveBatchAxis,
  normaliseAxis,
  type Primitive,
  sliceAxis,
  squeeze,
  stack,
  type Trace,
  type Traced,
  treeFlatten,
  treeUnflatten,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { withLevel } from './_level.ts'

/**
 * A value inside a batching trace: the underlying value (one level down) holding every example, and the axis that
 * indexes them (`null` when every example shares the value). Its own shape is the shape of one example.
 */
export interface BatchValue extends Traced {
  readonly value: Value
  readonly dim: BatchDim
}

/**
 * A batching trace (vectorisation) at one level.
 *
 * @concept vectorisation
 */
export interface BatchTrace extends Trace {
  readonly kind: 'batch'
  readonly size: number
  batched(value: Value, dim: BatchDim): BatchValue
}

function example(x: Value, dim: number, b: number): Value {
  return squeeze(sliceAxis(x, dim, b, b + 1), dim)
}

// The fallback for primitives without a batch rule: apply the primitive to each example and stack the results. It is
// correct for every primitive (the per-example calls go through the lower traces as usual), just not vectorised.
function loopBatch<P>(
  primitive: Primitive<P>,
  values: readonly Value[],
  dims: readonly BatchDim[],
  params: P,
  size: number,
): BatchResult {
  if (size === 0) {
    const avals = values.map((v, i) => {
      const d = dims[i] ?? null
      return { shape: d === null ? v.shape : v.shape.filter((_, a) => a !== d), dtype: v.dtype }
    })
    const outputs = primitive.abstractEval(avals, params).map((o) => zeros([0, ...o.shape], { dtype: o.dtype }))
    return { outputs, dims: outputs.map(() => 0) }
  }
  const per: (readonly Value[])[] = []
  for (let b = 0; b < size; b++) {
    const args = values.map((v, i) => {
      const d = dims[i] ?? null
      return d === null ? v : example(v, d, b)
    })
    per.push(bind(primitive, args, params))
  }
  const first = per[0] ?? []
  const outputs = first.map((_, k) => stack(per.map((outs) => outs[k] as Value)))
  return { outputs, dims: outputs.map(() => 0) }
}

/**
 * Create a batching trace. Use {@link vmap} rather than this directly.
 *
 * @param level - The trace's level.
 * @param size - The batch size.
 * @returns The trace.
 * @concept vectorisation
 * @example
 * ```ts
 * import { batchTrace } from 'mdml-engine/autodiff'
 * import { zeros } from 'mdml-engine/tensor'
 * batchTrace(1, 3).batched(zeros([3, 2]), 0).shape // => [2]
 * ```
 */
export function batchTrace(level: number, size: number): BatchTrace {
  const trace: BatchTrace = {
    kind: 'batch',
    level,
    size,
    lift: (x) => make(x, null),
    batched: (value, dim) => make(value, dim),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const bin = inputs as readonly BatchValue[]
      const values = bin.map((x) => x.value)
      const dims = bin.map((x) => x.dim)
      if (dims.every((d) => d === null)) return bind(primitive, values, params).map((o) => make(o, null))
      const rule = primitive.batch
      const result =
        rule === undefined ? loopBatch(primitive, values, dims, params, size) : rule(values, dims, params, size)
      return result.outputs.map((o, i) => make(o, result.dims[i] ?? null))
    },
  }
  function make(value: Value, dim: BatchDim): BatchValue {
    const shape = dim === null ? value.shape : value.shape.filter((_, a) => a !== dim)
    return { kind: 'traced', dtype: value.dtype, shape, trace, value, dim }
  }
  return trace
}

/** Which axis of each argument holds the examples: one axis for every argument, or one per argument (`null`: shared). */
export type InAxes = number | null | readonly (number | null)[]

/** Where the example axis goes in the outputs: one axis for every output leaf, or one per element of an output array. */
export type OutAxes = number | readonly number[]

// The out-axis of every leaf of a result, in flattening order.
function leafAxes(result: unknown, outAxes: OutAxes, count: number): number[] {
  if (typeof outAxes === 'number') return new Array<number>(count).fill(outAxes)
  if (!Array.isArray(result) || result.length !== outAxes.length) {
    throw new MdmlError('value', `vmap: ${outAxes.length} out-axes need an output array of that length`)
  }
  return result.flatMap((r: unknown, i: number) =>
    new Array<number>(treeFlatten(r).leaves.length).fill(outAxes[i] ?? 0),
  )
}

function isConcreteValueError(e: unknown): boolean {
  return e instanceof MdmlError && e.code === 'not-differentiable'
}

/**
 * Vectorise a function (like `jax.vmap`): the result maps `f` over an axis of its arguments, computing every example
 * at once through each primitive's batch rule (primitives without one loop over the examples). It is a trace like
 * `grad` and `jvp`, so the transformations compose in any order: `vmap(grad(f))` gives per-example gradients, the
 * gradient of a vectorised function works, and vmaps nest. A function that needs the concrete value of a batched
 * argument (data-dependent control flow) is mapped by a loop over the examples instead.
 *
 * @param f - The function; its arguments and output may be pytrees.
 * @param inAxes - The example axis of every argument, or one per argument (`null` passes an argument unchanged to
 *   every example). Default 0.
 * @param outAxes - Where the example axis goes in the output (one axis for all leaves, or one per element of an output
 *   array). Default 0.
 * @returns The vectorised function.
 * @throws MdmlError with code `shape` when batched arguments disagree on the batch size, and `value` when no
 *   argument is batched.
 * @concept vectorisation
 * @index function Vectorise a function over a batch axis, like jax.vmap.
 * @example
 * ```ts
 * import { vmap, grad } from 'mdml-engine/autodiff'
 * import { dot, mul, sum, tensor, toArray, type Value } from 'mdml-engine/tensor'
 * const rowDots = vmap((a: Value, b: Value) => dot(a, b), [0, null])
 * toArray(rowDots(tensor([[1, 2], [3, 4]]), tensor([1, 1]))) // => [3, 7]
 * const perExample = vmap(grad((x: Value) => sum(mul(x, x))))
 * toArray(perExample(tensor([[1, 2], [3, 4]]))) // => [[2, 4], [6, 8]]
 * ```
 */
export function vmap<A extends readonly unknown[], O>(
  f: (...args: A) => O,
  inAxes: InAxes = 0,
  outAxes: OutAxes = 0,
): (...args: A) => O {
  return (...args: A): O => {
    const axes = typeof inAxes === 'number' || inAxes === null ? args.map(() => inAxes) : inAxes
    if (axes.length !== args.length) {
      throw new MdmlError('value', `vmap: ${axes.length} in-axes for ${args.length} arguments`)
    }
    const sizes: number[] = []
    const flat = args.map((a, i) => {
      const axis = axes[i] ?? null
      if (axis === null) return null
      const { leaves, def } = treeFlatten(a)
      const dims = leaves.map((leaf) => {
        const d = normaliseAxis(axis, leaf.shape.length)
        sizes.push(leaf.shape[d] ?? 0)
        return d
      })
      return { leaves, def, dims }
    })
    const n = sizes[0]
    if (n === undefined) throw new MdmlError('value', 'vmap: no argument is batched')
    if (sizes.some((m) => m !== n)) throw new MdmlError('shape', `vmap: batch sizes differ ([${sizes.join(', ')}])`)
    const batchedArgs = (pick: (leaf: Value, dim: number) => Value): A =>
      args.map((a, i) => {
        const fl = flat[i] ?? null
        return fl === null
          ? a
          : treeUnflatten(
              fl.def,
              fl.leaves.map((leaf, j) => pick(leaf, fl.dims[j] ?? 0)),
            )
      }) as unknown as A
    try {
      return withLevel((level) => {
        const trace = batchTrace(level, n)
        const result = f(...batchedArgs((leaf, dim) => trace.batched(leaf, dim)))
        const { leaves, def } = treeFlatten(result)
        const outs = leafAxes(result, outAxes, leaves.length)
        return treeUnflatten(
          def,
          leaves.map((o, k) => {
            const ours = isTraced(o) && o.trace === trace
            const value = ours ? (o as BatchValue).value : o
            return moveBatchAxis(value, ours ? (o as BatchValue).dim : null, outs[k] ?? 0, n)
          }),
        ) as O
      })
    } catch (e) {
      // A function that needs the concrete value of a batched argument (data-dependent control flow, masks built in
      // plain arrays) cannot be traced as a batch, so it is mapped by a loop over the examples instead.
      if (!isConcreteValueError(e) || n === 0) throw e
    }
    const results = Array.from({ length: n }, (_, b) => f(...batchedArgs((leaf, dim) => example(leaf, dim, b))))
    const { def } = treeFlatten(results[0])
    const perLeaf = results.map((r) => treeFlatten(r).leaves)
    const outs = leafAxes(results[0], outAxes, perLeaf[0]?.length ?? 0)
    return treeUnflatten(
      def,
      (perLeaf[0] ?? []).map((_, k) =>
        stack(
          perLeaf.map((leaves) => leaves[k] as Value),
          outs[k] ?? 0,
        ),
      ),
    ) as O
  }
}
