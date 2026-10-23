import {
  arg,
  bind,
  definePrimitive,
  type MaybeZero,
  type Primitive,
  stopGradient as stopGradientOp,
  type TensorLike,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** The definition of a function with a hand-written reverse-mode rule. */
export interface CustomVjpSpec {
  readonly name: string
  /** The function itself, on values (it may use any operations). */
  readonly forward: (...inputs: Value[]) => Value
  /** Cotangents of the inputs, given the output cotangent, the inputs and the output. `null` means zero. */
  readonly vjp: (cotangent: Value, inputs: readonly Value[], output: Value) => readonly MaybeZero[]
  /** An optional forward-mode rule; without it the function supports reverse mode only. */
  readonly jvp?: (inputs: readonly Value[], tangents: readonly MaybeZero[], output: Value) => MaybeZero
}

/**
 * Define a function with a custom derivative rule (like `jax.custom_vjp`). The forward pass runs as written; when
 * differentiated, the given rule replaces differentiation through its internals. This is how factorisations, solvers
 * and numerically delicate functions get exact, stable derivatives.
 *
 * @param spec - Name, forward function, vjp rule and optional jvp rule.
 * @returns The function, usable inside and outside transformations.
 * @concept custom-derivatives
 * @example
 * ```ts
 * import { customVjp, grad } from 'whitebox-engine/autodiff'
 * import { log1p, exp, mul, sigmoid, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const softplus = customVjp({
 *   name: 'softplus',
 *   forward: (x) => log1p(exp(x)),
 *   vjp: (g, [x]) => [mul(g, sigmoid(x as Value))],
 * })
 * toArray(grad(softplus)(tensor(0))) // => 0.5
 * ```
 */
export function customVjp(spec: CustomVjpSpec): (...inputs: Value[]) => Value {
  const primitive: Primitive<null> = definePrimitive<null>({
    name: spec.name,
    abstractEval: (avals) => {
      const out = spec.forward(...avals.map((a) => zeros(a.shape, { dtype: a.dtype })))
      return [{ shape: out.shape, dtype: out.dtype }]
    },
    impl: (inputs) => [spec.forward(...inputs) as never],
    vjp: (gs, inputs, _p, outs) => {
      const g = arg(gs, 0)
      return g === null ? inputs.map(() => null) : spec.vjp(g, inputs, arg(outs, 0))
    },
    ...(spec.jvp === undefined
      ? {}
      : {
          jvp: (inputs: readonly Value[], ts: readonly MaybeZero[], _p: null, outs: readonly Value[]) => [
            (spec.jvp as NonNullable<CustomVjpSpec['jvp']>)(inputs, ts, arg(outs, 0)),
          ],
        }),
  })
  return (...inputs) => arg(bind(primitive, inputs, null), 0)
}

/**
 * The identity function with a zero derivative (an alias of the tensor operation, here for discoverability).
 *
 * @param x - Input.
 * @returns `x`, treated as a constant by differentiation.
 * @concept stop-gradient
 * @example
 * ```ts
 * import { stopGradient, grad } from 'whitebox-engine/autodiff'
 * import { mul, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(grad((x: Value) => mul(x, stopGradient(x)))(tensor(3))) // => 3
 * ```
 */
export function stopGradient(x: TensorLike): Value {
  return stopGradientOp(x)
}
