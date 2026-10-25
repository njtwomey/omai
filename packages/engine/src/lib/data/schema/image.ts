/**
 * The schema for image datasets: pixels as `[n, h, w, c]` tensors (channels last, values in [0, 1]) with optional class
 * labels, bounding boxes and segmentation masks. The `images` constructor checks that they agree.
 */

import { MdmlError } from '../../base/index.ts'
import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { checkRows, concrete, concreteFloat, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * Images `[n, h, w, c]` (channels last, values in [0, 1] by convention) with optional class labels, bounding boxes and
 * segmentation masks.
 */
export interface ImageData {
  readonly kind: 'image'
  readonly images: Tensor
  /** A class per image `[n]` (int32). */
  readonly labels?: Tensor<'int32'>
  readonly classes?: readonly string[]
  /** Boxes per image, each `[k, 4]` as `[x0, y0, x1, y1]` in pixels. */
  readonly boxes?: readonly Tensor[]
  /** A class per box, per image. */
  readonly boxLabels?: readonly Tensor<'int32'>[]
  /** A class per pixel `[n, h, w]` (int32). */
  readonly masks?: Tensor<'int32'>
  readonly maskClasses?: readonly string[]
  readonly meta: DatasetMeta
}

/** Options of {@link images}. */
export interface ImageOptions {
  readonly labels?: TensorLike
  readonly classes?: readonly string[]
  readonly boxes?: readonly TensorLike[]
  readonly boxLabels?: readonly TensorLike[]
  readonly masks?: TensorLike
  readonly maskClasses?: readonly string[]
  readonly meta?: Partial<DatasetMeta>
}

/**
 * An image dataset from `[n, h, w, c]` pixels (or `[n, h, w]` for one channel), checking labels, boxes and masks.
 *
 * @param pixels - The images.
 * @param options - Labels, classes, boxes, box labels, masks and metadata.
 * @returns The dataset.
 * @throws MdmlError with code `shape` when the images are not 3-D or 4-D, or a mask's size differs.
 * @example
 * ```ts
 * import { images } from 'mdml-engine/data'
 * images([[[0, 1], [1, 0]]], { labels: [1] }).images.shape // => [1, 2, 2, 1]
 * ```
 */
export function images(pixels: TensorLike, options: ImageOptions = {}): ImageData {
  let x = concreteFloat(pixels, 'images')
  if (x.shape.length === 3) x = x.backend.wrap(x.dtype, [...x.shape, 1], x.data)
  if (x.shape.length !== 4) throw new MdmlError('shape', `images: expected [n, h, w, c], got [${x.shape}]`)
  const [n = 0, h = 0, w = 0] = x.shape
  const labels = options.labels === undefined ? undefined : concrete(options.labels, 'int32', 'images')
  const boxes = options.boxes?.map((b) => {
    const t = concreteFloat(b, 'images')
    return (t.shape[0] ?? 0) === 0 ? t.backend.wrap(t.dtype, [0, 4], t.data) : t
  })
  const boxLabels = options.boxLabels?.map((b) => concrete(b, 'int32', 'images'))
  const masks = options.masks === undefined ? undefined : concrete(options.masks, 'int32', 'images')
  if (masks !== undefined && (masks.shape[1] !== h || masks.shape[2] !== w))
    throw new MdmlError('shape', `images: masks must be [${n}, ${h}, ${w}]`)
  checkRows('images', n, { labels, boxes, boxLabels, masks })
  return {
    kind: 'image',
    images: x,
    ...(labels === undefined ? {} : { labels }),
    ...(options.classes === undefined ? {} : { classes: options.classes }),
    ...(boxes === undefined ? {} : { boxes }),
    ...(boxLabels === undefined ? {} : { boxLabels }),
    ...(masks === undefined ? {} : { masks }),
    ...(options.maskClasses === undefined ? {} : { maskClasses: options.maskClasses }),
    meta: metaFor('image', options.meta),
  }
}
