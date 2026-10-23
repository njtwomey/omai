import { add, matmul, promoteFloat, swapaxes, type TensorLike, type Value } from '../../tensor/index.ts'

/**
 * Affine map y = x W^T + b with PyTorch's weight layout `[out, in]` (torch.nn.functional.linear).
 *
 * @param x - Input `[..., in]`.
 * @param weight - Weight `[out, in]`.
 * @param bias - Bias `[out]` (optional).
 * @returns Output `[..., out]`, differentiable in all arguments.
 * @concept layers
 * @example
 * ```ts
 * import { linear } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(linear([[1, 2]], [[1, 1], [0, 1]], [0.5, 0])) // => [[3.5, 2]]
 * ```
 */
export function linear(x: TensorLike, weight: TensorLike, bias?: TensorLike): Value {
  const [xv, w] = promoteFloat(x, weight) as [Value, Value]
  const y = matmul(xv, swapaxes(w, -1, -2))
  return bias === undefined ? y : add(y, promoteFloat(bias)[0] as Value)
}
