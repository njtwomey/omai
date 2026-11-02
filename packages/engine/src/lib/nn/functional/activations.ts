/**
 * Activation functions as plain functions of tensors, torch.nn.functional style: ReLU and its variants, ELU, SELU,
 * GELU, SiLU, Mish, softplus, the hard and shrinkage functions, and the gated units (GLU, SwiGLU, GeGLU, ReGLU) of
 * modern feed-forward layers, with `activation` to pick one by name.
 *
 * @index function Activation functions as plain tensor functions, torch.nn.functional style.
 */

import { MdmlError } from '../../base/index.ts'
import { erf, logaddexp } from '../../numerics/special/index.ts'
import {
  abs as absOp,
  add,
  clip,
  div,
  expm1,
  greater,
  less,
  minimum,
  mul,
  pow,
  promoteFloat,
  relu as reluOp,
  reshape,
  sigmoid as sigmoidOp,
  sliceAxis,
  sub,
  type TensorLike,
  tanh as tanhOp,
  type Value,
  where,
} from '../../tensor/index.ts'

const f = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/**
 * ReLU, max(x, 0) (torch.nn.functional.relu).
 *
 * @param x - Input.
 * @returns The activation.
 * @concept activations
 * @example
 * ```ts
 * import { relu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(relu([-1, 2])) // => [0, 2]
 * ```
 */
export function relu(x: TensorLike): Value {
  return reluOp(f(x))
}

/**
 * Leaky ReLU (torch.nn.functional.leaky_relu): x for x > 0, slope x otherwise.
 *
 * @param x - Input.
 * @param slope - Negative slope (default 0.01).
 * @returns The activation.
 * @example
 * ```ts
 * import { leakyRelu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(leakyRelu([-2, 3], 0.1)) // => [-0.2, 3]
 * ```
 */
export function leakyRelu(x: TensorLike, slope = 0.01): Value {
  const v = f(x)
  return where(greater(v, 0), v, mul(v, slope))
}

/**
 * ELU (torch.nn.functional.elu): x for x > 0, alpha (exp(x) - 1) otherwise.
 *
 * @param x - Input.
 * @param alpha - Scale of the negative part (default 1).
 * @returns The activation.
 * @example
 * ```ts
 * import { elu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(elu([0, 1])) // => [0, 1]
 * ```
 */
export function elu(x: TensorLike, alpha = 1): Value {
  const v = f(x)
  return where(greater(v, 0), v, mul(expm1(minimum(v, 0)), alpha))
}

const SELU_ALPHA = 1.6732632423543772
const SELU_SCALE = 1.0507009873554805

/**
 * SELU (torch.nn.functional.selu): scale elu(x, alpha) with the self-normalising constants.
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { selu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(selu([0])) // => [0]
 * ```
 */
export function selu(x: TensorLike): Value {
  return mul(elu(x, SELU_ALPHA), SELU_SCALE)
}

/**
 * GELU (torch.nn.functional.gelu): x Phi(x) exactly, or the tanh approximation.
 *
 * @param x - Input.
 * @param approximate - 'none' (default) or 'tanh'.
 * @returns The activation.
 * @example
 * ```ts
 * import { gelu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gelu([0])) // => [0]
 * ```
 */
export function gelu(x: TensorLike, approximate: 'none' | 'tanh' = 'none'): Value {
  const v = f(x)
  if (approximate === 'tanh') {
    const inner = mul(add(v, mul(pow(v, 3), 0.044715)), Math.sqrt(2 / Math.PI))
    return mul(mul(v, 0.5), add(tanhOp(inner), 1))
  }
  return mul(mul(v, 0.5), add(erf(mul(v, Math.SQRT1_2)), 1))
}

/**
 * SiLU (swish), x sigmoid(x) (torch.nn.functional.silu).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { silu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(silu([0])) // => [0]
 * ```
 */
export function silu(x: TensorLike): Value {
  const v = f(x)
  return mul(v, sigmoidOp(v))
}

/**
 * Mish, x tanh(softplus(x)) (torch.nn.functional.mish).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { mish } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(mish([0])) // => [0]
 * ```
 */
export function mish(x: TensorLike): Value {
  const v = f(x)
  return mul(v, tanhOp(logaddexp(0, v)))
}

/**
 * Softplus with sharpness beta, reverting to the identity above `threshold` (torch.nn.functional.softplus).
 *
 * @param x - Input.
 * @param beta - Sharpness (default 1).
 * @param threshold - Linear above beta x > threshold (default 20).
 * @returns The activation.
 * @example
 * ```ts
 * import { softplus } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(softplus([100])) // => [100]
 * ```
 */
export function softplus(x: TensorLike, beta = 1, threshold = 20): Value {
  const v = f(x)
  const bx = mul(v, beta)
  return where(greater(bx, threshold), v, div(logaddexp(0, bx), beta))
}

/**
 * Hard tanh, x clipped to [min, max] (torch.nn.functional.hardtanh).
 *
 * @param x - Input.
 * @param min - Lower bound (default -1).
 * @param max - Upper bound (default 1).
 * @returns The activation.
 * @example
 * ```ts
 * import { hardtanh } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(hardtanh([-3, 0.5])) // => [-1, 0.5]
 * ```
 */
export function hardtanh(x: TensorLike, min = -1, max = 1): Value {
  return clip(f(x), min, max)
}

/**
 * Hard sigmoid, clip(x / 6 + 1/2, 0, 1) (torch.nn.functional.hardsigmoid).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { hardsigmoid } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(hardsigmoid([0])) // => [0.5]
 * ```
 */
export function hardsigmoid(x: TensorLike): Value {
  return clip(add(div(f(x), 6), 0.5), 0, 1)
}

/**
 * Gated linear unit: the first half of the last axis times the sigmoid of the second half (torch.nn.functional.glu).
 *
 * @param x - Input with an even last axis.
 * @param axis - The axis to split (default -1).
 * @returns The activation, half as wide.
 * @example
 * ```ts
 * import { glu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(glu([2, 0])) // => [1]
 * ```
 */
export function glu(x: TensorLike, axis = -1): Value {
  const v = f(x)
  const n = v.shape[axis < 0 ? v.shape.length + axis : axis] ?? 0
  if (n % 2 !== 0) throw new MdmlError('shape', 'glu: the split axis must have even length')
  return mul(sliceAxis(v, axis, 0, n / 2), sigmoidOp(sliceAxis(v, axis, n / 2, n)))
}

const halves = (x: TensorLike, axis: number, name: string): [Value, Value] => {
  const v = f(x)
  const n = v.shape[axis < 0 ? v.shape.length + axis : axis] ?? 0
  if (n % 2 !== 0) throw new MdmlError('shape', `${name}: the split axis must have even length`)
  return [sliceAxis(v, axis, 0, n / 2), sliceAxis(v, axis, n / 2, n)]
}

/**
 * SwiGLU (Shazeer 2020; LLaMA's feed-forward gate): SiLU of the first half of the axis times the second half.
 *
 * @param x - Input with an even split axis (the concatenated outputs of the gate and value projections).
 * @param axis - The axis to split (default -1).
 * @returns The activation, half as wide.
 * @reference Shazeer (2020), GLU variants improve transformer.
 * @example
 * ```ts
 * import { swiglu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(swiglu([0, 5])) // => [0]
 * ```
 */
export function swiglu(x: TensorLike, axis = -1): Value {
  const [a, b] = halves(x, axis, 'swiglu')
  return mul(silu(a), b)
}

/**
 * GeGLU (Shazeer 2020): GELU of the first half of the axis times the second half.
 *
 * @param x - Input with an even split axis.
 * @param axis - The axis to split (default -1).
 * @returns The activation, half as wide.
 * @example
 * ```ts
 * import { geglu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(geglu([0, 3])) // => [0]
 * ```
 */
export function geglu(x: TensorLike, axis = -1): Value {
  const [a, b] = halves(x, axis, 'geglu')
  return mul(gelu(a), b)
}

/**
 * ReGLU (Shazeer 2020): ReLU of the first half of the axis times the second half.
 *
 * @param x - Input with an even split axis.
 * @param axis - The axis to split (default -1).
 * @returns The activation, half as wide.
 * @example
 * ```ts
 * import { reglu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(reglu([2, 3])) // => [6]
 * ```
 */
export function reglu(x: TensorLike, axis = -1): Value {
  const [a, b] = halves(x, axis, 'reglu')
  return mul(reluOp(a), b)
}

/**
 * ReLU6, min(max(x, 0), 6) (torch.nn.functional.relu6; MobileNet).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { relu6 } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(relu6([-1, 3, 9])) // => [0, 3, 6]
 * ```
 */
export function relu6(x: TensorLike): Value {
  return clip(f(x), 0, 6)
}

/**
 * Hard swish, x relu6(x + 3) / 6 (torch.nn.functional.hardswish; MobileNetV3).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { hardswish } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(hardswish([-4, 0, 4])) // => [-0, 0, 4]
 * ```
 */
export function hardswish(x: TensorLike): Value {
  const v = f(x)
  return mul(v, div(clip(add(v, 3), 0, 6), 6))
}

/**
 * CELU (torch.nn.functional.celu): max(0, x) + min(0, alpha (exp(x / alpha) - 1)).
 *
 * @param x - Input.
 * @param alpha - Scale (default 1).
 * @returns The activation.
 * @example
 * ```ts
 * import { celu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(celu([0, 2])) // => [0, 2]
 * ```
 */
export function celu(x: TensorLike, alpha = 1): Value {
  const v = f(x)
  return where(greater(v, 0), v, mul(expm1(div(minimum(v, 0), alpha)), alpha))
}

/**
 * Softsign, x / (1 + |x|) (torch.nn.functional.softsign).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { softsign } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(softsign([1, -3])) // => [0.5, -0.75]
 * ```
 */
export function softsign(x: TensorLike): Value {
  const v = f(x)
  return div(v, add(absOp(v), 1))
}

/**
 * Log-sigmoid, log(sigmoid(x)) = -softplus(-x), computed stably (torch.nn.functional.logsigmoid).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { logsigmoid } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(logsigmoid([0])).map((v) => Math.round(v * 1e6) / 1e6) // => [-0.693147]
 * ```
 */
export function logsigmoid(x: TensorLike): Value {
  return mul(logaddexp(0, mul(f(x), -1)), -1)
}

/**
 * PReLU (torch.nn.functional.prelu, He et al. 2015): x for x > 0, a x otherwise, with a learnt slope per channel
 * (axis 1) or one shared slope.
 *
 * @param x - Input `[B, C, ...]` or `[B]`.
 * @param weight - Slopes `[C]` or `[1]`.
 * @returns The activation.
 * @example
 * ```ts
 * import { prelu } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(prelu([[-2, 3]], [0.25])) // => [[-0.5, 3]]
 * ```
 */
export function prelu(x: TensorLike, weight: TensorLike): Value {
  const v = f(x)
  const w = f(weight)
  const n = w.shape[0] ?? 1
  const slope = n === 1 || v.shape.length < 2 ? w : reshape(w, [1, n, ...new Array<number>(v.shape.length - 2).fill(1)])
  return where(greater(v, 0), v, mul(v, slope))
}

/**
 * Tanh shrink, x - tanh(x) (torch.nn.functional.tanhshrink).
 *
 * @param x - Input.
 * @returns The activation.
 * @example
 * ```ts
 * import { tanhshrink } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(tanhshrink([0])) // => [0]
 * ```
 */
export function tanhshrink(x: TensorLike): Value {
  const v = f(x)
  return sub(v, tanhOp(v))
}

/**
 * Soft shrinkage (torch.nn.functional.softshrink, the proximal map of the L1 norm): x - lambda above lambda, x +
 * lambda below -lambda, 0 between.
 *
 * @param x - Input.
 * @param lambda - Threshold (default 0.5).
 * @returns The activation.
 * @example
 * ```ts
 * import { softshrink } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(softshrink([-2, 0.2, 1])) // => [-1.5, 0, 0.5]
 * ```
 */
export function softshrink(x: TensorLike, lambda = 0.5): Value {
  const v = f(x)
  return where(greater(v, lambda), sub(v, lambda), where(less(v, -lambda), add(v, lambda), mul(v, 0)))
}

/**
 * Hard shrinkage (torch.nn.functional.hardshrink): x where |x| > lambda, else 0.
 *
 * @param x - Input.
 * @param lambda - Threshold (default 0.5).
 * @returns The activation.
 * @example
 * ```ts
 * import { hardshrink } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(hardshrink([-2, 0.2, 1])) // => [-2, 0, 1]
 * ```
 */
export function hardshrink(x: TensorLike, lambda = 0.5): Value {
  const v = f(x)
  return where(greater(absOp(v), lambda), v, mul(v, 0))
}

/**
 * Threshold (torch.nn.functional.threshold): x where x > threshold, else `value`.
 *
 * @param x - Input.
 * @param threshold - The threshold.
 * @param value - The replacement.
 * @returns The activation.
 * @example
 * ```ts
 * import { threshold } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(threshold([1, 3], 2, -1)) // => [-1, 3]
 * ```
 */
export function threshold(x: TensorLike, threshold: number, value: number): Value {
  const v = f(x)
  return where(greater(v, threshold), v, add(mul(v, 0), value))
}

/** The activations available by name. */
export type ActivationName =
  | 'relu'
  | 'relu6'
  | 'leakyRelu'
  | 'elu'
  | 'celu'
  | 'selu'
  | 'gelu'
  | 'geluTanh'
  | 'silu'
  | 'mish'
  | 'softplus'
  | 'softsign'
  | 'tanh'
  | 'sigmoid'
  | 'logsigmoid'
  | 'hardtanh'
  | 'hardsigmoid'
  | 'hardswish'
  | 'tanhshrink'
  | 'identity'

/**
 * An activation function by name (for configuration and serialised models).
 *
 * @param name - The activation.
 * @returns The function.
 * @example
 * ```ts
 * import { activation } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(activation('relu')([-1, 1])) // => [0, 1]
 * ```
 */
export function activation(name: ActivationName): (x: TensorLike) => Value {
  const table: Readonly<Record<ActivationName, (x: TensorLike) => Value>> = {
    relu,
    relu6,
    leakyRelu: (x) => leakyRelu(x),
    elu: (x) => elu(x),
    celu: (x) => celu(x),
    selu,
    gelu: (x) => gelu(x),
    geluTanh: (x) => gelu(x, 'tanh'),
    silu,
    mish,
    softplus: (x) => softplus(x),
    softsign,
    tanh: (x) => tanhOp(f(x)),
    sigmoid: (x) => sigmoidOp(f(x)),
    logsigmoid,
    hardtanh: (x) => hardtanh(x),
    hardsigmoid,
    hardswish,
    tanhshrink,
    identity: f,
  }
  return table[name]
}
