// A grey image as a row-major plane of numbers, the form the image detectors and motion estimators compute on. Private
// to the image files.
import { OmaiError } from '../../base/index.ts'
import { primalOf, type TensorLike } from '../../tensor/index.ts'
import { float } from '../_shared.ts'

/** A grey image `[h, w]` as row-major storage. */
export interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

/** An image `[H, W]` as a plane of primal numbers; any other rank is a `shape` error. */
export function planeOf(x: TensorLike, name: string): Plane {
  const v = primalOf(float(x))
  if (v.shape.length !== 2) throw new OmaiError('shape', `${name}: images are [H, W]`)
  return { h: v.shape[0] ?? 0, w: v.shape[1] ?? 0, data: Float64Array.from(v.data, Number) }
}
