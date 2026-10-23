import { cpu } from '../cpu.ts'
import { allocate, type DType, isFloat, resultType, weakResultType } from '../dtype.ts'
import type { MaybeZero } from '../primitive.ts'
import { asTensor, isTensor, type TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { cast } from './arithmetic.ts'

/** Convert any input to a value. */
export function val(x: TensorLike): Value {
  return asTensor(x)
}

/** A 0-d tensor holding `x` with a given dtype, on the backend of `like` when it is concrete. */
export function scalar(x: number, dtype: DType, like?: Value): Value {
  const backend = like !== undefined && isTensor(like) ? like.backend : cpu
  const data = allocate(dtype, 1)
  data[0] = dtype === 'int32' ? Math.trunc(x) : dtype === 'bool' ? (x ? 1 : 0) : x
  return backend.wrap(dtype, [], data)
}

/** Cast a value to a dtype when it differs. */
export function asDtype(x: Value, dtype: DType): Value {
  return x.dtype === dtype ? x : cast(x, dtype)
}

/**
 * Bring two operands to a common dtype. Plain numbers are weak (they adopt the other operand's dtype when they fit),
 * bool is treated as int32 for arithmetic, and `float` forces a floating result (true division, transcendental ops).
 */
export function coerce2(
  a: TensorLike,
  b: TensorLike,
  mode: 'arithmetic' | 'float' | 'any' = 'arithmetic',
): [Value, Value] {
  const aNum = typeof a === 'number'
  const bNum = typeof b === 'number'
  const va = aNum ? undefined : val(a)
  const vb = bNum ? undefined : val(b)
  let dtype: DType
  if (va === undefined && vb === undefined) dtype = 'float64'
  else if (va === undefined) dtype = weakResultType((vb as Value).dtype, a as number)
  else if (vb === undefined) dtype = weakResultType(va.dtype, b as number)
  else dtype = resultType(va.dtype, vb.dtype)
  if (mode !== 'any' && dtype === 'bool') dtype = 'int32'
  if (mode === 'float' && !isFloat(dtype)) dtype = 'float64'
  const like = va ?? vb
  const x = va === undefined ? scalar(a as number, dtype, like) : asDtype(va, dtype)
  const y = vb === undefined ? scalar(b as number, dtype, like) : asDtype(vb, dtype)
  return [x, y]
}

/** Bring any number of operands to one floating dtype (plain numbers are weak, as in {@link coerce2}). */
export function coerceFloat(xs: readonly TensorLike[]): Value[] {
  const vs = xs.map((x) => (typeof x === 'number' ? undefined : val(x)))
  const strong = vs.filter((v): v is Value => v !== undefined)
  let dtype: DType = strong.length === 0 ? 'float64' : strong.map((v) => v.dtype).reduce((a, b) => resultType(a, b))
  if (!isFloat(dtype)) dtype = 'float64'
  const like = strong[0]
  return xs.map((x, i) => {
    const v = vs[i]
    return v === undefined ? scalar(x as number, dtype, like) : asDtype(v, dtype)
  })
}

/** Cast an integer or bool value to float64 (for operations defined on floats). */
export function floatify(x: TensorLike): Value {
  const v = val(x)
  return isFloat(v.dtype) ? v : cast(v, 'float64')
}

/** Apply `f` unless the tangent or cotangent is a symbolic zero. */
export function mapZero(x: MaybeZero, f: (v: Value) => Value): MaybeZero {
  return x === null ? null : f(x)
}
