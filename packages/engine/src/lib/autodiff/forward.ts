import { WhiteboxError } from '../base/index.ts'
import { bind, type MaybeZero, type Primitive, type Trace, type Traced, type Value } from '../tensor/index.ts'

/** A value inside a forward-mode trace: a primal and its tangent (`null` for a zero tangent). */
export interface ForwardValue extends Traced {
  readonly primal: Value
  readonly tangent: MaybeZero
}

/**
 * A forward-mode trace (dual numbers) at one level.
 *
 * @concept forward-mode-autodiff
 */
export interface ForwardTrace extends Trace {
  readonly kind: 'forward'
  dual(primal: Value, tangent: MaybeZero): ForwardValue
}

/**
 * Create a forward-mode trace. Use `jvp` rather than this directly.
 *
 * @param level - The trace's level.
 * @returns The trace.
 * @example
 * ```ts
 * import { forwardTrace } from 'whitebox-engine/autodiff'
 * forwardTrace(1).kind // => 'forward'
 * ```
 */
export function forwardTrace(level: number): ForwardTrace {
  const trace: ForwardTrace = {
    kind: 'forward',
    level,
    lift: (x) => make(x, null),
    dual: (primal, tangent) => make(primal, tangent),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const fin = inputs as readonly ForwardValue[]
      const primals = fin.map((x) => x.primal)
      const tangents = fin.map((x) => x.tangent)
      const outputs = bind(primitive, primals, params)
      if (tangents.every((t) => t === null) || primitive.zeroDerivative === true)
        return outputs.map((o) => make(o, null))
      const rule = primitive.jvp
      if (rule === undefined) {
        throw new WhiteboxError('not-differentiable', `primitive ${primitive.name} has no derivative rule`, {
          primitive: primitive.name,
        })
      }
      const outTangents = rule(primals, tangents, params, outputs)
      return outputs.map((o, i) => make(o, outTangents[i] ?? null))
    },
  }
  function make(primal: Value, tangent: MaybeZero): ForwardValue {
    return { kind: 'traced', dtype: primal.dtype, shape: primal.shape, trace, primal, tangent }
  }
  return trace
}
