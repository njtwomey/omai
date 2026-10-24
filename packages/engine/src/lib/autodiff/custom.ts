import {
  add,
  arg,
  asTensor,
  bind,
  definePrimitive,
  isFloat,
  isTraced,
  type MaybeZero,
  matmulPrimitive,
  type Primitive,
  stopGradient as stopGradientOp,
  sub,
  type TensorLike,
  type Trace,
  type Traced,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { withLevel } from './_level.ts'
import { jvp as jvpTransform, vjp as vjpTransform } from './transforms.ts'

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

/** The definition of a function with a hand-written forward-mode rule. */
export interface CustomJvpSpec {
  readonly name: string
  /** The function itself, on values (it may use any operations). */
  readonly forward: (...inputs: Value[]) => Value
  /** The output tangent, given the inputs, their tangents (`null` for zero) and the output; linear in the tangents. */
  readonly jvp: (inputs: readonly Value[], tangents: readonly MaybeZero[], output: Value) => MaybeZero
}

function zerosOf(xs: readonly { readonly shape: readonly number[]; readonly dtype: Value['dtype'] }[]): Value[] {
  return xs.map((a) => zeros(a.shape, { dtype: a.dtype }))
}

function floatIndices(xs: readonly Value[]): number[] {
  return xs.map((x, i) => (isFloat(x.dtype) ? i : -1)).filter((i) => i >= 0)
}

/**
 * Define a function with a custom forward-mode rule (like `jax.custom_jvp`). Forward mode uses the rule directly;
 * reverse mode transposes it: the rule is linear in the tangents, so its pullback (reverse mode over the rule itself)
 * is the vector-Jacobian product. One rule gives both modes, and higher orders, as long as it is written with
 * operations.
 *
 * @param spec - Name, forward function and jvp rule.
 * @returns The function, usable inside and outside transformations.
 * @concept custom-derivatives
 * @example
 * ```ts
 * import { customJvp, grad, jvp } from 'whitebox-engine/autodiff'
 * import { add, exp, log, mul, sigmoid, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const softplus = customJvp({
 *   name: 'softplus',
 *   forward: (x) => log(add(exp(x), 1)),
 *   jvp: ([x], [t]) => (t === null || t === undefined ? null : mul(t, sigmoid(x as Value))),
 * })
 * toArray(grad(softplus)(tensor(0))) // => 0.5
 * toArray(jvp(softplus, tensor(0), tensor(2)).tangent) // => 1
 * ```
 */
export function customJvp(spec: CustomJvpSpec): (...inputs: Value[]) => Value {
  const primitive: Primitive<null> = definePrimitive<null>({
    name: spec.name,
    abstractEval: (avals) => {
      const out = spec.forward(...zerosOf(avals))
      return [{ shape: out.shape, dtype: out.dtype }]
    },
    impl: (inputs) => [spec.forward(...inputs) as never],
    jvp: (inputs, ts, _p, outs) => [spec.jvp(inputs, ts, arg(outs, 0))],
    vjp: (gs, inputs, _p, outs) => {
      const g = arg(gs, 0)
      if (g === null) return inputs.map(() => null)
      const out = arg(outs, 0)
      const wrt = floatIndices(inputs)
      const linear = (ts: readonly Value[]): Value => {
        const full = inputs.map((_, i) => {
          const k = wrt.indexOf(i)
          return k < 0 ? null : (ts[k] ?? null)
        })
        return spec.jvp(inputs, full, out) ?? zeros(out.shape, { dtype: out.dtype })
      }
      const cts = vjpTransform(linear, zerosOf(wrt.map((i) => arg(inputs, i)))).pullback(g)
      return inputs.map((_, i) => {
        const k = wrt.indexOf(i)
        return k < 0 ? null : (cts[k] ?? null)
      })
    },
  })
  return (...inputs) => arg(bind(primitive, inputs, null), 0)
}

/**
 * Which intermediate results {@link checkpoint} keeps from the forward pass instead of recomputing: none (the default,
 * the most memory saved), the outputs of matrix products (the expensive part of most networks), all of them (no
 * rematerialisation), or those of the primitives a predicate selects.
 */
export type CheckpointPolicy = 'nothing' | 'matmuls' | 'everything' | ((primitive: Primitive<unknown>) => boolean)

/** Options for {@link checkpoint}. */
export interface CheckpointOptions {
  /** What to save from the forward pass (default `'nothing'`). */
  readonly policy?: CheckpointPolicy
  /** A name for computation graphs (default `'checkpoint'`). */
  readonly name?: string
}

function saveablePredicate(policy: CheckpointPolicy): (primitive: Primitive<unknown>) => boolean {
  if (typeof policy === 'function') return policy
  if (policy === 'matmuls') return (p) => p === (matmulPrimitive as Primitive<unknown>)
  return () => policy === 'everything'
}

interface ResidualValue extends Traced {
  readonly primal: Value
}

interface ResidualTrace extends Trace {
  replaying: boolean
  unwrap(x: Value): Value
}

// A trace that records the outputs of saveable primitives, or hands recorded outputs back in the same order while it is
// replaying. During the recomputation it sits below the reverse trace, so derivative rules still see the right primal
// values; it stops replaying once the forward pass has been recomputed. Its values carry their primal, so functions
// that read concrete values (`primalOf`) still work inside a checkpoint.
function residualTrace(
  level: number,
  saveable: (primitive: Primitive<unknown>) => boolean,
  saved: Value[],
  mode: 'record' | 'replay',
): ResidualTrace {
  let cursor = 0
  const trace: ResidualTrace = {
    level,
    replaying: mode === 'replay',
    lift: (x) => make(x),
    unwrap: (x) => (isTraced(x) && x.trace === trace ? (x as ResidualValue).primal : x),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const values = inputs.map((x) => (x as ResidualValue).primal)
      if (!saveable(primitive as Primitive<unknown>)) return bind(primitive, values, params).map(make)
      if (!trace.replaying) {
        const outs = bind(primitive, values, params)
        if (mode === 'record') saved.push(...outs)
        return outs.map(make)
      }
      const count = primitive.abstractEval(values, params).length
      const outs = saved.slice(cursor, cursor + count)
      cursor += count
      return outs.map(make)
    },
  }
  function make(primal: Value): ResidualValue {
    return { kind: 'traced', dtype: primal.dtype, shape: primal.shape, trace, primal }
  }
  return trace
}

/**
 * Rematerialisation (gradient checkpointing, like `jax.checkpoint`): the forward pass keeps only the inputs and the
 * output of `f`, plus whatever the policy saves, and the backward pass recomputes `f` to differentiate it. Memory falls
 * from every intermediate of `f` to its inputs, for the cost of a second forward pass; values and derivatives are
 * unchanged. Forward mode differentiates `f` directly.
 *
 * @param f - The function to rematerialise.
 * @param options - The policy for what to save, and a name.
 * @returns The checkpointed function.
 * @concept rematerialisation
 * @example
 * ```ts
 * import { checkpoint, grad } from 'whitebox-engine/autodiff'
 * import { matmul, sum, tanh, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const w = tensor([[1, 0], [0, 2]])
 * const layer = checkpoint((x) => tanh(matmul(x, w)), { policy: 'matmuls' })
 * toArray(grad((x: Value) => sum(layer(x)))(tensor([0, 0]))) // => [1, 2]
 * ```
 */
export function checkpoint(
  f: (...inputs: Value[]) => Value,
  options: CheckpointOptions = {},
): (...inputs: Value[]) => Value {
  const saveable = saveablePredicate(options.policy ?? 'nothing')
  // The forward pass with the residuals the policy keeps: [output, ...saved].
  const record = (inputs: readonly Value[]): Value[] =>
    withLevel((level) => {
      const saved: Value[] = []
      const trace = residualTrace(level, saveable, saved, 'record')
      const out = trace.unwrap(f(...inputs.map((x) => trace.lift(x))))
      return [out, ...saved]
    })
  // Substitute the differentiated inputs into the full argument list.
  const withInputs = (all: readonly Value[], wrt: readonly number[], xs: readonly Value[]): Value[] =>
    all.map((x, i) => {
      const k = wrt.indexOf(i)
      return k < 0 ? x : (xs[k] ?? x)
    })
  const primitive: Primitive<null> = definePrimitive<null>({
    name: options.name ?? 'checkpoint',
    abstractEval: (avals) => record(zerosOf(avals)).map((v) => ({ shape: v.shape, dtype: v.dtype })),
    impl: (inputs) => record(inputs) as never,
    jvp: (inputs, ts, _p, outs) => {
      const wrt = floatIndices(inputs).filter((i) => ts[i] !== null && ts[i] !== undefined)
      if (wrt.length === 0) return outs.map(() => null)
      const tangent = jvpTransform(
        (xs: readonly Value[]) => f(...withInputs(inputs, wrt, xs)),
        wrt.map((i) => arg(inputs, i)),
        wrt.map((i) => arg(ts, i) as Value),
      ).tangent
      return [tangent, ...outs.slice(1).map(() => null)]
    },
    vjp: (gs, inputs, _p, outs) => {
      const g = arg(gs, 0)
      if (g === null) return inputs.map(() => null)
      const wrt = floatIndices(inputs)
      return withLevel((level) => {
        const trace = residualTrace(level, saveable, outs.slice(1), 'replay')
        const lifted = inputs.map((x) => trace.lift(x))
        const { pullback } = vjpTransform(
          (xs: readonly Value[]) => f(...withInputs(lifted, wrt, xs)),
          wrt.map((i) => arg(lifted, i)),
        )
        trace.replaying = false
        const cts = pullback(g)
        return inputs.map((_, i) => {
          const k = wrt.indexOf(i)
          const c = k < 0 ? undefined : cts[k]
          return c === undefined ? null : trace.unwrap(c)
        })
      })
    },
  })
  return (...inputs) => arg(bind(primitive, inputs, null), 0)
}

/**
 * The straight-through estimator: the value of `f(x)` with the derivative of the identity, for functions whose own
 * derivative is zero or undefined almost everywhere (rounding, quantisation, choosing a discrete code). The result is
 * `x + stopGradient(f(x) - x)`, so it works in every mode and under `vmap`.
 *
 * @param f - A function whose output has the shape of its float input.
 * @returns The function with an identity derivative.
 * @concept straight-through-estimator
 * @example
 * ```ts
 * import { straightThrough, grad } from 'whitebox-engine/autodiff'
 * import { round, sum, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const roundST = straightThrough(round)
 * toArray(roundST(tensor([0.4, 1.6]))) // => [0, 2]
 * toArray(grad((x: Value) => sum(roundST(x)))(tensor([0.4, 1.6]))) // => [1, 1]
 * ```
 */
export function straightThrough(f: (x: Value) => Value): (x: TensorLike) => Value {
  return (input) => {
    const x = asTensor(input)
    return add(x, stopGradientOp(sub(f(x), x)))
  }
}
