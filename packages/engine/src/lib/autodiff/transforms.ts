/**
 * The JAX-style derivative transforms: `vjp`, `grad`, `valueAndGrad` and `gradAndAux` in reverse mode, `jvp` and
 * `linearize` in forward mode, Jacobians by either mode (`jacrev`, `jacfwd`, `jacobian`), and Hessians with
 * Hessian-vector and vector-Hessian products that never form the full matrix.
 *
 * @index function JAX-style grad, vjp, jvp, Jacobians and Hessians.
 */

import { OmaiError } from '../base/index.ts'
import {
  asTensor,
  isFloat,
  isTraced,
  mul,
  ones,
  reshape,
  sizeOf,
  stack,
  sum,
  type TensorLike,
  treeFlatten,
  treeUnflatten,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { withLevel } from './_level.ts'
import { type ForwardValue, forwardTrace } from './forward.ts'
import { backward, isReverseValueOf, type ReverseValue, reverseTrace } from './reverse.ts'

function checkFloatLeaves(leaves: readonly Value[], where: string): void {
  for (const leaf of leaves) {
    if (!isFloat(leaf.dtype)) {
      throw new OmaiError('dtype', `${where}: can only differentiate with respect to float values, got ${leaf.dtype}`)
    }
  }
}

/**
 * Run a function and return its output with a function that maps output cotangents to input cotangents (like
 * `jax.vjp`). Inputs and outputs may be pytrees.
 *
 * @param f - The function.
 * @param primals - Its input (a value or a pytree of values).
 * @returns The output and the pullback.
 * @concept reverse-mode-autodiff
 * @example
 * ```ts
 * import { vjp } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * const { out, pullback } = vjp((x: Value) => mul(x, x), tensor([1, 2]))
 * toArray(pullback(tensor([1, 1])) as Value) // => [2, 4]
 * ```
 */
export function vjp<P, O>(
  f: (primals: P) => O,
  primals: P,
): { readonly out: O; readonly pullback: (cotangents: O) => P } {
  const { leaves, def } = treeFlatten(primals)
  checkFloatLeaves(leaves, 'vjp')
  return withLevel((level) => {
    const trace = reverseTrace(level)
    const traced = leaves.map((l) => trace.leafOf(l))
    const result = f(treeUnflatten(def, traced) as P)
    const { leaves: outLeaves, def: outDef } = treeFlatten(result)
    const out = treeUnflatten(
      outDef,
      outLeaves.map((o) => (isReverseValueOf(o, trace) ? o.primal : o)),
    ) as O
    const pullback = (cotangents: O): P => {
      const cts = treeFlatten(cotangents).leaves
      const tracked = outLeaves.map((o) => (isReverseValueOf(o, trace) ? o : null))
      const outputs = tracked.filter((o): o is ReverseValue => o !== null)
      const seeds = tracked.map((o, i) => (o === null ? null : (cts[i] ?? null))).filter((_, i) => tracked[i] !== null)
      return treeUnflatten(def, backward(outputs, seeds, traced)) as P
    }
    return { out, pullback }
  })
}

/** Options for {@link valueAndGrad} and {@link grad}. */
export interface GradOptions {
  /** The function returns `[value, aux]`; aux is passed through untouched. */
  readonly hasAux?: boolean
}

/**
 * Turn a scalar function of a pytree into one that returns its value and gradient (like `jax.value_and_grad`).
 * Differentiation happens only during this call: outside it, tensors carry no gradient machinery.
 *
 * @param f - A function from a pytree of values to a scalar value (or `[scalar, aux]` with `hasAux`).
 * @param options - `hasAux`.
 * @returns A function from a pytree to `{ value, grad, aux }`, where `grad` mirrors the input.
 * @throws OmaiError with code `shape` when `f` does not return a single-element value.
 * @concept reverse-mode-autodiff
 * @index function A function's value and gradient in one pass.
 * @example
 * ```ts
 * import { valueAndGrad } from 'omai-engine/autodiff'
 * import { mul, sum, tensor, toArray, type Value } from 'omai-engine/tensor'
 * const f = valueAndGrad((p: { w: Value }) => sum(mul(p.w, p.w)))
 * const { value, grad } = f({ w: tensor([1, 2]) })
 * toArray(value) // => 5
 * toArray(grad.w) // => [2, 4]
 * ```
 */
export function valueAndGrad<P>(
  f: (params: P) => Value | readonly [Value, unknown],
  options: GradOptions = {},
): (params: P) => { readonly value: Value; readonly grad: P; readonly aux: unknown } {
  return (params) => {
    let aux: unknown
    const { out, pullback } = vjp((p: P) => {
      const r = f(p)
      if (options.hasAux === true) {
        const [v, a] = r as readonly [Value, unknown]
        aux = a
        return v
      }
      return r as Value
    }, params)
    if (sizeOf(out.shape) !== 1) {
      throw new OmaiError('shape', `grad: function must return a single value, got shape [${out.shape.join(', ')}]`)
    }
    const grad = pullback(ones(out.shape, { dtype: out.dtype }) as Value)
    return { value: out, grad, aux }
  }
}

/**
 * The gradient of a scalar function of a pytree (like `jax.grad`).
 *
 * @param f - A function from a pytree of values to a scalar value.
 * @param options - `hasAux`.
 * @returns A function from a pytree to its gradient (same structure).
 * @concept reverse-mode-autodiff
 * @index function Gradient of a scalar function of a pytree, like jax.grad.
 * @example
 * ```ts
 * import { grad } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(grad((x: Value) => mul(x, x))(tensor(3))) // => 6
 * ```
 */
export function grad<P>(
  f: (params: P) => Value | readonly [Value, unknown],
  options: GradOptions = {},
): (params: P) => P {
  const vg = valueAndGrad(f, options)
  return (params) => vg(params).grad
}

/**
 * The gradient of a function that also returns auxiliary data (like `jax.grad(f, has_aux=True)`): `f` returns
 * `[value, aux]`, the gradient is taken of `value`, and `aux` (any data: metrics, updated state) is passed through.
 *
 * @param f - A function from a pytree of values to `[scalar, aux]`.
 * @returns A function from a pytree to `{ grad, aux }`, where `grad` mirrors the input.
 * @concept reverse-mode-autodiff
 * @example
 * ```ts
 * import { gradAndAux } from 'omai-engine/autodiff'
 * import { mul, sum, tensor, toArray, type Value } from 'omai-engine/tensor'
 * const f = gradAndAux((x: Value) => [sum(mul(x, x)), { size: x.shape[0] }] as const)
 * const { grad, aux } = f(tensor([1, 2]))
 * toArray(grad) // => [2, 4]
 * aux // => { size: 2 }
 * ```
 */
export function gradAndAux<P, A>(
  f: (params: P) => readonly [Value, A],
): (params: P) => { readonly grad: P; readonly aux: A } {
  const vg = valueAndGrad(f, { hasAux: true })
  return (params) => {
    const { grad: g, aux } = vg(params)
    return { grad: g, aux: aux as A }
  }
}

/**
 * Forward-mode derivative: the output of `f` at `primals` and its directional derivative along `tangents` (like
 * `jax.jvp`). Inputs and outputs may be pytrees.
 *
 * @param f - The function.
 * @param primals - The point.
 * @param tangents - The direction, with the structure of `primals`.
 * @returns The output and its tangent.
 * @concept forward-mode-autodiff
 * @example
 * ```ts
 * import { jvp } from 'omai-engine/autodiff'
 * import { sin, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(jvp((x: Value) => sin(x), tensor(0), tensor(1)).tangent as Value) // => 1
 * ```
 */
export function jvp<P, O>(f: (primals: P) => O, primals: P, tangents: P): { readonly out: O; readonly tangent: O } {
  const { leaves, def } = treeFlatten(primals)
  const tleaves = treeFlatten(tangents).leaves
  checkFloatLeaves(leaves, 'jvp')
  return withLevel((level) => {
    const trace = forwardTrace(level)
    const traced = leaves.map((l, i) => trace.dual(l, tleaves[i] ?? null))
    const result = f(treeUnflatten(def, traced) as P)
    const { leaves: outLeaves, def: outDef } = treeFlatten(result)
    const ours = (o: Value): o is ForwardValue => isTraced(o) && o.trace === trace
    const out = treeUnflatten(
      outDef,
      outLeaves.map((o) => (ours(o) ? o.primal : o)),
    ) as O
    const tangent = treeUnflatten(
      outDef,
      outLeaves.map((o) => {
        const t = ours(o) ? o.tangent : null
        return t ?? zeros(o.shape, { dtype: o.dtype })
      }),
    ) as O
    return { out, tangent }
  })
}

function basis(shape: readonly number[], dtype: Value['dtype'], i: number): Value {
  const e = zeros(shape, { dtype })
  ;(e.data as Float64Array)[i] = 1
  return e
}

/**
 * The Jacobian by reverse mode: one pullback per output element (efficient when outputs are few).
 *
 * @param f - A function of one value.
 * @returns A function returning the Jacobian with shape `[...outShape, ...inShape]`.
 * @concept jacobian
 * @example
 * ```ts
 * import { jacrev } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(jacrev((x: Value) => mul(x, tensor([2, 3])))(tensor([1, 1]))) // => [[2, 0], [0, 3]]
 * ```
 */
export function jacrev(f: (x: Value) => Value): (x: TensorLike) => Value {
  return (input) => {
    const x = asTensor(input)
    const { out, pullback } = vjp(f, x)
    const n = sizeOf(out.shape)
    const rows = Array.from({ length: n }, (_, i) => reshape(pullback(basis(out.shape, out.dtype, i)), [-1]))
    return reshape(n === 0 ? zeros([0]) : stack(rows), [...out.shape, ...x.shape])
  }
}

/**
 * The Jacobian by forward mode: one tangent per input element (efficient when inputs are few).
 *
 * @param f - A function of one value.
 * @returns A function returning the Jacobian with shape `[...outShape, ...inShape]`.
 * @concept jacobian
 * @example
 * ```ts
 * import { jacfwd } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(jacfwd((x: Value) => mul(x, x))(tensor([1, 2]))) // => [[2, 0], [0, 4]]
 * ```
 */
export function jacfwd(f: (x: Value) => Value): (x: TensorLike) => Value {
  return (input) => {
    const x = asTensor(input)
    const n = sizeOf(x.shape)
    const columns = Array.from({ length: n }, (_, i) => reshape(jvp(f, x, basis(x.shape, x.dtype, i)).tangent, [-1]))
    const outShape = n === 0 ? [0] : jvp(f, x, zeros(x.shape, { dtype: x.dtype })).out.shape
    // Columns are d out / d x_i; stack them on the last axis.
    const jac = n === 0 ? zeros([...outShape, 0]) : stack(columns, -1)
    return reshape(jac, [...outShape, ...x.shape])
  }
}

/**
 * The Jacobian of a function of one value (forward mode when the input is smaller than the output, else reverse).
 *
 * @param f - A function of one value.
 * @returns A function returning the Jacobian with shape `[...outShape, ...inShape]`.
 * @concept jacobian
 * @example
 * ```ts
 * import { jacobian } from 'omai-engine/autodiff'
 * import { sum, mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(jacobian((x: Value) => sum(mul(x, x)))(tensor([1, 2]))) // => [2, 4]
 * ```
 */
export function jacobian(f: (x: Value) => Value): (x: TensorLike) => Value {
  return (input) => {
    const x = asTensor(input)
    const out = f(x)
    return sizeOf(x.shape) <= sizeOf(out.shape) ? jacfwd(f)(x) : jacrev(f)(x)
  }
}

/**
 * The Hessian of a scalar function of one value, by forward mode over reverse mode.
 *
 * @param f - A scalar function of one value.
 * @returns A function returning the Hessian with shape `[...inShape, ...inShape]`.
 * @concept hessian
 * @index function Hessian of a scalar function, forward over reverse.
 * @example
 * ```ts
 * import { hessian } from 'omai-engine/autodiff'
 * import { mul, sum, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(hessian((x: Value) => sum(mul(mul(x, x), x)))(tensor([1, 2]))) // => [[6, 0], [0, 12]]
 * ```
 */
export function hessian(f: (x: Value) => Value): (x: TensorLike) => Value {
  return jacfwd(grad(f) as (x: Value) => Value)
}

/**
 * A Hessian-vector product without forming the Hessian (forward over reverse).
 *
 * @param f - A scalar function of one value.
 * @param x - The point.
 * @param v - The vector, with the shape of `x`.
 * @returns `H(x) v`.
 * @concept hessian
 * @example
 * ```ts
 * import { hvp } from 'omai-engine/autodiff'
 * import { mul, sum, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(hvp((x: Value) => sum(mul(mul(x, x), x)), tensor([1, 2]), tensor([1, 0]))) // => [6, 0]
 * ```
 */
export function hvp(f: (x: Value) => Value, x: TensorLike, v: TensorLike): Value {
  return jvp(grad(f) as (x: Value) => Value, asTensor(x), asTensor(v)).tangent
}

/**
 * A vector-Hessian product `v^T H(x)` without forming the Hessian, by reverse mode over reverse mode (like
 * `torch.autograd.functional.vhp`): the gradient of `<grad f(x), v>`. For a twice continuously differentiable `f` the
 * Hessian is symmetric and this equals {@link hvp}; the two differ in cost profile (two reverse sweeps here, one
 * forward over one reverse there).
 *
 * @param f - A scalar function of one value.
 * @param x - The point.
 * @param v - The vector, with the shape of `x`.
 * @returns `v^T H(x)`, with the shape of `x`.
 * @concept hessian
 * @example
 * ```ts
 * import { vhp } from 'omai-engine/autodiff'
 * import { mul, sum, tensor, toArray, type Value } from 'omai-engine/tensor'
 * toArray(vhp((x: Value) => sum(mul(mul(x, x), x)), tensor([1, 2]), tensor([1, 0]))) // => [6, 0]
 * ```
 */
export function vhp(f: (x: Value) => Value, x: TensorLike, v: TensorLike): Value {
  const direction = asTensor(v)
  const gf = grad(f) as (y: Value) => Value
  return grad((y: Value) => sum(mul(gf(y), direction)))(asTensor(x))
}

/**
 * A function's output and its linear map at a point: the tangent function re-evaluates the forward pass along any
 * direction.
 *
 * @param f - A function of one value.
 * @param x - The point.
 * @returns The output and `tangentAt(t)`.
 * @concept forward-mode-autodiff
 * @example
 * ```ts
 * import { linearize } from 'omai-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'
 * const { tangentAt } = linearize((x: Value) => mul(x, x), tensor(3))
 * toArray(tangentAt(tensor(1))) // => 6
 * ```
 */
export function linearize(
  f: (x: Value) => Value,
  x: TensorLike,
): { readonly out: Value; readonly tangentAt: (t: TensorLike) => Value } {
  const point = asTensor(x)
  return { out: f(point), tangentAt: (t) => jvp(f, point, asTensor(t)).tangent }
}
