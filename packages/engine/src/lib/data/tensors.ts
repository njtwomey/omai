import { WhiteboxError } from '../base/index.ts'
import { asTensor, cast, type Tensor, type TensorLike } from '../tensor/index.ts'
import type { DatasetMeta } from './schema/common.ts'
import type { ClassificationData, RegressionData } from './schema/tabular.ts'

function concreteFloat(x: TensorLike, name: string): Tensor {
  const t = asTensor(x)
  if (!('data' in t)) throw new WhiteboxError('value', `${name}: datasets hold concrete tensors`)
  return (t.dtype === 'float64' || t.dtype === 'float32' ? t : cast(t, 'float64')) as Tensor
}

/**
 * A regression dataset from feature and target arrays.
 *
 * @param x - Features `[n, d]` (a 1-D array is one feature).
 * @param y - Targets `[n]` or `[n, k]`.
 * @param meta - Name, feature names and other metadata.
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { xy } from 'whitebox-engine/data'
 * xy([[1, 2], [3, 4]], [0.5, 1.5]).featureNames // => ['x0', 'x1']
 * ```
 */
export function xy(
  x: TensorLike,
  y: TensorLike,
  meta: Partial<DatasetMeta> & {
    readonly featureNames?: readonly string[]
    readonly targetNames?: readonly string[]
  } = {},
): RegressionData {
  let xs = concreteFloat(x, 'xy')
  if (xs.shape.length === 1) xs = xs.backend.wrap(xs.dtype, [xs.shape[0] ?? 0, 1], xs.data)
  const ys = concreteFloat(y, 'xy')
  const d = xs.shape[1] ?? 0
  const k = ys.shape.length === 2 ? (ys.shape[1] ?? 1) : 1
  const { featureNames, targetNames, ...rest } = meta
  return {
    kind: 'regression',
    x: xs,
    y: ys,
    featureNames: featureNames ?? Array.from({ length: d }, (_, i) => `x${i}`),
    targetNames: targetNames ?? (k === 1 ? ['y'] : Array.from({ length: k }, (_, i) => `y${i}`)),
    meta: { name: rest.name ?? 'data', description: rest.description ?? '', task: 'regression', ...rest },
  }
}

/**
 * A classification dataset from features and integer labels.
 *
 * @param x - Features `[n, d]`.
 * @param y - Class indices `[n]`.
 * @param meta - Class names, feature names and other metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { xyClasses } from 'whitebox-engine/data'
 * xyClasses([[0], [1]], [0, 1]).classes // => ['0', '1']
 * ```
 */
export function xyClasses(
  x: TensorLike,
  y: TensorLike,
  meta: Partial<DatasetMeta> & { readonly featureNames?: readonly string[]; readonly classes?: readonly string[] } = {},
): ClassificationData {
  const xs = concreteFloat(x, 'xyClasses')
  const ys = cast(asTensor(y), 'int32') as Tensor<'int32'>
  const k = Math.max(-1, ...Array.from(ys.data)) + 1
  const { featureNames, classes, ...rest } = meta
  return {
    kind: 'classification',
    x: xs,
    y: ys,
    classes: classes ?? Array.from({ length: k }, (_, i) => String(i)),
    featureNames: featureNames ?? Array.from({ length: xs.shape[1] ?? 0 }, (_, i) => `x${i}`),
    meta: { name: rest.name ?? 'data', description: rest.description ?? '', task: 'classification', ...rest },
  }
}
