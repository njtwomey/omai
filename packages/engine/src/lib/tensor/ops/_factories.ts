import { WhiteboxError } from '../../base/index.ts'
import type { BinaryKernel, UnaryKernel } from '../backend.ts'
import { arg, definePrimitive, type MaybeZero, type Primitive } from '../primitive.ts'
import { broadcastShapes, shapesEqual } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import type { Aval, Value } from '../traced.ts'
import { broadcastTo, sumToShape } from './shape.ts'

/** A tangent broadcast to the output shape (tangents of broadcast inputs have the input's shape). */
export function toShape(t: Value, shape: readonly number[]): Value {
  return shapesEqual(t.shape, shape) ? t : broadcastTo(t, shape)
}

/**
 * An elementwise unary primitive. `rule(t, x, out)` maps a tangent (or, since the Jacobian is diagonal, a cotangent)
 * through the derivative; omit it for functions whose derivative is zero (rounding, sign).
 */
export function unaryPrimitive(kernel: UnaryKernel, rule?: (t: Value, x: Value, out: Value) => Value): Primitive<null> {
  const base = {
    name: kernel,
    abstractEval: (xs: readonly Aval[]) => [arg(xs, 0)],
    impl: (xs: readonly Tensor[]) => [arg(xs, 0).backend.unary(kernel, arg(xs, 0))],
  }
  if (rule === undefined) return definePrimitive<null>({ ...base, zeroDerivative: true })
  return definePrimitive<null>({
    ...base,
    jvp: (xs, ts, _p, outs) => {
      const t = arg(ts, 0)
      return [t === null ? null : rule(t, arg(xs, 0), arg(outs, 0))]
    },
    vjp: (gs, xs, _p, outs) => {
      const g = arg(gs, 0)
      return [g === null ? null : rule(g, arg(xs, 0), arg(outs, 0))]
    },
  })
}

/** The rules of an elementwise binary primitive, on broadcast operands. */
export interface BinaryRules {
  jvp(a: Value, b: Value, ta: MaybeZero, tb: MaybeZero, out: Value): MaybeZero
  vjp(g: Value, a: Value, b: Value, out: Value): readonly [MaybeZero, MaybeZero]
}

/** An elementwise binary primitive with broadcasting; cotangents are summed back to each input's shape. */
export function binaryPrimitive(kernel: BinaryKernel, rules?: BinaryRules): Primitive<null> {
  const base = {
    name: kernel,
    abstractEval: (xs: readonly Aval[]) => {
      const a = arg(xs, 0)
      const b = arg(xs, 1)
      if (a.dtype !== b.dtype) throw new WhiteboxError('dtype', `${kernel}: operands must share a dtype`)
      return [{ shape: broadcastShapes(a.shape, b.shape), dtype: a.dtype }]
    },
    impl: (xs: readonly Tensor[]) => [arg(xs, 0).backend.binary(kernel, arg(xs, 0), arg(xs, 1))],
  }
  if (rules === undefined) return definePrimitive<null>({ ...base, zeroDerivative: true })
  return definePrimitive<null>({
    ...base,
    jvp: (xs, ts, _p, outs) => {
      const ta = arg(ts, 0)
      const tb = arg(ts, 1)
      if (ta === null && tb === null) return [null]
      return [rules.jvp(arg(xs, 0), arg(xs, 1), ta, tb, arg(outs, 0))]
    },
    vjp: (gs, xs, _p, outs) => {
      const g = arg(gs, 0)
      if (g === null) return [null, null]
      const a = arg(xs, 0)
      const b = arg(xs, 1)
      const [ga, gb] = rules.vjp(g, a, b, arg(outs, 0))
      return [ga === null ? null : sumToShape(ga, a.shape), gb === null ? null : sumToShape(gb, b.shape)]
    },
  })
}
