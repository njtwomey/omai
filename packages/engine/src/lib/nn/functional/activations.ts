import { WhiteboxError } from '../../base/index.ts'
import { erf, logaddexp } from '../../numerics/special/index.ts'
import {
  add,
  clip,
  div,
  expm1,
  greater,
  minimum,
  mul,
  pow,
  promoteFloat,
  relu as reluOp,
  sigmoid as sigmoidOp,
  sliceAxis,
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
 * import { relu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { leakyRelu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { elu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { selu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { gelu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { silu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { mish } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { softplus } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { hardtanh } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { hardsigmoid } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { glu } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(glu([2, 0])) // => [1]
 * ```
 */
export function glu(x: TensorLike, axis = -1): Value {
  const v = f(x)
  const n = v.shape[axis < 0 ? v.shape.length + axis : axis] ?? 0
  if (n % 2 !== 0) throw new WhiteboxError('shape', 'glu: the split axis must have even length')
  return mul(sliceAxis(v, axis, 0, n / 2), sigmoidOp(sliceAxis(v, axis, n / 2, n)))
}

/** The activations available by name. */
export type ActivationName =
  | 'relu'
  | 'leakyRelu'
  | 'elu'
  | 'selu'
  | 'gelu'
  | 'silu'
  | 'mish'
  | 'softplus'
  | 'tanh'
  | 'sigmoid'
  | 'hardtanh'
  | 'hardsigmoid'
  | 'identity'

/**
 * An activation function by name (for configuration and serialised models).
 *
 * @param name - The activation.
 * @returns The function.
 * @example
 * ```ts
 * import { activation } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(activation('relu')([-1, 1])) // => [0, 1]
 * ```
 */
export function activation(name: ActivationName): (x: TensorLike) => Value {
  const table: Readonly<Record<ActivationName, (x: TensorLike) => Value>> = {
    relu,
    leakyRelu: (x) => leakyRelu(x),
    elu: (x) => elu(x),
    selu,
    gelu: (x) => gelu(x),
    silu,
    mish,
    softplus: (x) => softplus(x),
    tanh: (x) => tanhOp(f(x)),
    sigmoid: (x) => sigmoidOp(f(x)),
    hardtanh: (x) => hardtanh(x),
    hardsigmoid,
    identity: f,
  }
  return table[name]
}
