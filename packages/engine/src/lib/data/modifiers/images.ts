/**
 * Image modifiers, the augmentations of vision pipelines with their parameters kept as truth: random crops (with
 * padding), flips, cutout, mixup and cutmix. Each takes images `[n, h, w, c]` with a key and returns the changed images
 * with the draws that made them (offsets, flip masks, boxes, mixing weights and partners, soft labels) in
 * `truth.latents`, and moves bounding boxes and masks with the pixels where it can.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { beta, foldIn, type Key, permutation, uniform } from '../../random/index.ts'
import { dict, int, real } from '../../space/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import type { ImageData } from '../schema/image.ts'

function requireImages(input: Generated, name: string): Generated<ImageData> {
  if (input.data.kind !== 'image') throw new MdmlError('value', `${name} needs image data, got ${input.data.kind}`)
  return input as Generated<ImageData>
}

function dims(data: ImageData): { n: number; h: number; w: number; c: number } {
  const [n = 0, h = 0, w = 0, c = 1] = data.images.shape
  return { n, h, w, c }
}

/** Boxes `[k, 4]` of one image mapped by `f` and kept when they still cover some area inside `[0, w] x [0, h]`. */
function mapBoxes(
  data: ImageData,
  i: number,
  w: number,
  h: number,
  f: (b: readonly [number, number, number, number]) => [number, number, number, number],
): { boxes: Tensor; labels: Tensor<'int32'> } | undefined {
  const boxes = data.boxes?.[i]
  if (boxes === undefined) return undefined
  const labels = data.boxLabels?.[i]
  const keep: number[][] = []
  const kept: number[] = []
  for (let k = 0; k < (boxes.shape[0] ?? 0); k++) {
    const [x0, y0, x1, y1] = f([
      boxes.data[k * 4] ?? 0,
      boxes.data[k * 4 + 1] ?? 0,
      boxes.data[k * 4 + 2] ?? 0,
      boxes.data[k * 4 + 3] ?? 0,
    ])
    const cx0 = Math.max(0, x0)
    const cy0 = Math.max(0, y0)
    const cx1 = Math.min(w, x1)
    const cy1 = Math.min(h, y1)
    if (cx1 > cx0 && cy1 > cy0) {
      keep.push([cx0, cy0, cx1, cy1])
      kept.push(labels?.data[k] ?? 0)
    }
  }
  return {
    boxes: fromStorage(Float64Array.from(keep.flat()), [keep.length, 4], 'float64'),
    labels: fromStorage(Int32Array.from(kept), [kept.length], 'int32'),
  }
}

/** The same pixels with a per-image box-and-label pass, rebuilt as image data without stale fields. */
function rebuild(
  data: ImageData,
  pixels: Float64Array,
  h: number,
  w: number,
  extra: {
    masks?: Int32Array | undefined
    boxes?: (Tensor | undefined)[] | undefined
    boxLabels?: (Tensor<'int32'> | undefined)[] | undefined
    labels?: Tensor<'int32'> | undefined
    dropAnnotations?: boolean
  } = {},
): ImageData {
  const { n, c } = dims(data)
  const { boxes: _b, boxLabels: _bl, masks: _m, labels: _l, ...rest } = data
  const labels = extra.labels ?? data.labels
  const out: ImageData = {
    ...rest,
    images: fromStorage(pixels, [n, h, w, c], 'float64'),
    ...(labels === undefined ? {} : { labels }),
  }
  if (extra.dropAnnotations === true) return out
  const boxes = extra.boxes
  return {
    ...out,
    ...(boxes === undefined || boxes.some((b) => b === undefined) ? {} : { boxes: boxes as Tensor[] }),
    ...(extra.boxLabels === undefined || extra.boxLabels.some((b) => b === undefined)
      ? {}
      : { boxLabels: extra.boxLabels as Tensor<'int32'>[] }),
    ...(extra.masks === undefined ? {} : { masks: fromStorage(extra.masks, [n, h, w], 'int32') }),
  }
}

function withLatents(generated: Generated, latents: Readonly<Record<string, unknown>>): Generated['truth'] {
  return { ...generated.truth, latents: { ...(generated.truth.latents ?? {}), ...latents } }
}

/** Options of {@link withRandomCrop}. */
export interface RandomCropOptions {
  /** The side of the crop in pixels, or `[height, width]` (default the image size, so only the padding shifts it). */
  readonly size?: number | readonly [number, number]
  /** Zeros added on every side before cropping (default 0). */
  readonly padding?: number
  readonly key: Key
}

/**
 * Random crops: each image is padded with zeros on every side and a window of the given size is cut at a uniformly
 * random position (the CIFAR recipe is `padding: 4` at the image size). Bounding boxes move and clip with the pixels and
 * those left without area are dropped; masks are cropped alike.
 *
 * @param input - Images and their truth.
 * @param options - Crop size, padding and key.
 * @returns The cropped images; `truth.latents.cropOffsets` is `[n, 2]` with the window's top-left corner in the
 *   original image's coordinates (negative where it reaches into the padding).
 * @example
 * ```ts
 * import { asGenerated, images, withRandomCrop } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(images([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]))
 * withRandomCrop(d, { size: 2, key: key(0) }).data.images.shape // => [1, 2, 2, 1]
 * ```
 */
export function withRandomCrop(input: Generated, options: RandomCropOptions): Generated<ImageData> {
  const { data } = requireImages(input, 'withRandomCrop')
  const { n, h, w, c } = dims(data)
  const pad = options.padding ?? 0
  const [oh, ow] = typeof options.size === 'number' ? [options.size, options.size] : (options.size ?? [h, w])
  if (pad < 0 || oh < 1 || ow < 1 || oh > h + 2 * pad || ow > w + 2 * pad)
    throw new MdmlError('value', `withRandomCrop: a ${oh} by ${ow} crop does not fit ${h} by ${w} with padding ${pad}`)
  const u = uniform(options.key, [n, 2]).data
  const pixels = new Float64Array(n * oh * ow * c)
  const masks = data.masks === undefined ? undefined : new Int32Array(n * oh * ow)
  const offsets = new Int32Array(n * 2)
  const boxes: (Tensor | undefined)[] = []
  const boxLabels: (Tensor<'int32'> | undefined)[] = []
  for (let i = 0; i < n; i++) {
    const top = Math.floor((u[i * 2] ?? 0) * (h + 2 * pad - oh + 1)) - pad
    const left = Math.floor((u[i * 2 + 1] ?? 0) * (w + 2 * pad - ow + 1)) - pad
    offsets[i * 2] = top
    offsets[i * 2 + 1] = left
    for (let y = 0; y < oh; y++)
      for (let x = 0; x < ow; x++) {
        const sy = top + y
        const sx = left + x
        if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue
        for (let k = 0; k < c; k++)
          pixels[((i * oh + y) * ow + x) * c + k] = data.images.data[((i * h + sy) * w + sx) * c + k] ?? 0
        if (masks !== undefined) masks[(i * oh + y) * ow + x] = data.masks?.data[(i * h + sy) * w + sx] ?? 0
      }
    const moved = mapBoxes(data, i, ow, oh, ([x0, y0, x1, y1]) => [x0 - left, y0 - top, x1 - left, y1 - top])
    boxes.push(moved?.boxes)
    boxLabels.push(moved?.labels)
  }
  return {
    data: rebuild(data, pixels, oh, ow, { masks, boxes, boxLabels }),
    truth: withLatents(input, { cropOffsets: fromStorage(offsets, [n, 2], 'int32') }),
  }
}

/** Options of {@link withRandomFlip}. */
export interface RandomFlipOptions {
  /** The chance of a left-right flip per image (default 0.5). */
  readonly horizontal?: number
  /** The chance of an up-down flip per image (default 0). */
  readonly vertical?: number
  readonly key: Key
}

/**
 * Random flips, each image independently mirrored left-right and up-down with the given chances. Boxes and masks flip
 * with the pixels.
 *
 * @param input - Images and their truth.
 * @param options - The two flip chances and key.
 * @returns The flipped images; `truth.latents.flippedHorizontal` and `flippedVertical` are `[n]` boolean masks.
 * @example
 * ```ts
 * import { asGenerated, images, withRandomFlip } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(images([[[1, 0], [1, 0]]]))
 * Array.from(withRandomFlip(d, { horizontal: 1, key: key(0) }).data.images.data) // => [0, 1, 0, 1]
 * ```
 */
export function withRandomFlip(input: Generated, options: RandomFlipOptions): Generated<ImageData> {
  const { data } = requireImages(input, 'withRandomFlip')
  const { n, h, w, c } = dims(data)
  const ph = options.horizontal ?? 0.5
  const pv = options.vertical ?? 0
  const u = uniform(options.key, [n, 2]).data
  const flipH = new Uint8Array(n)
  const flipV = new Uint8Array(n)
  const pixels = new Float64Array(n * h * w * c)
  const masks = data.masks === undefined ? undefined : new Int32Array(n * h * w)
  const boxes: (Tensor | undefined)[] = []
  const boxLabels: (Tensor<'int32'> | undefined)[] = []
  for (let i = 0; i < n; i++) {
    const fh = (u[i * 2] ?? 1) < ph
    const fv = (u[i * 2 + 1] ?? 1) < pv
    flipH[i] = fh ? 1 : 0
    flipV[i] = fv ? 1 : 0
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const sy = fv ? h - 1 - y : y
        const sx = fh ? w - 1 - x : x
        for (let k = 0; k < c; k++)
          pixels[((i * h + y) * w + x) * c + k] = data.images.data[((i * h + sy) * w + sx) * c + k] ?? 0
        if (masks !== undefined) masks[(i * h + y) * w + x] = data.masks?.data[(i * h + sy) * w + sx] ?? 0
      }
    const moved = mapBoxes(data, i, w, h, ([x0, y0, x1, y1]) => [
      fh ? w - x1 : x0,
      fv ? h - y1 : y0,
      fh ? w - x0 : x1,
      fv ? h - y0 : y1,
    ])
    boxes.push(moved?.boxes)
    boxLabels.push(moved?.labels)
  }
  return {
    data: rebuild(data, pixels, h, w, { masks, boxes, boxLabels }),
    truth: withLatents(input, {
      flippedHorizontal: fromStorage(flipH, [n], 'bool'),
      flippedVertical: fromStorage(flipV, [n], 'bool'),
    }),
  }
}

/** Options of {@link withCutout}. */
export interface CutoutOptions {
  /** The side of the square hole in pixels. */
  readonly size: number
  /** The value written into the hole (default 0). */
  readonly fill?: number
  readonly key: Key
}

/**
 * Cutout: one square hole per image, centred uniformly over the image (so near an edge only part of it lies inside),
 * filled with a constant. Labels, boxes and masks are unchanged.
 *
 * @param input - Images and their truth.
 * @param options - Hole size, fill value and key.
 * @returns The images with holes; `truth.latents.cutoutBoxes` is `[n, 4]` as `[x0, y0, x1, y1]` clipped to the image.
 * @example
 * ```ts
 * import { asGenerated, images, withCutout } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(images([[[1, 1], [1, 1]]]))
 * Math.min(...withCutout(d, { size: 2, key: key(0) }).data.images.data) // => 0
 * ```
 */
export function withCutout(input: Generated, options: CutoutOptions): Generated<ImageData> {
  const { data } = requireImages(input, 'withCutout')
  const { n, h, w, c } = dims(data)
  if (options.size < 1) throw new MdmlError('value', 'withCutout: the hole needs a positive size')
  const fill = options.fill ?? 0
  const u = uniform(options.key, [n, 2]).data
  const pixels = Float64Array.from(data.images.data)
  const holes = new Float64Array(n * 4)
  for (let i = 0; i < n; i++) {
    const cy = Math.floor((u[i * 2] ?? 0) * h)
    const cx = Math.floor((u[i * 2 + 1] ?? 0) * w)
    const y0 = Math.max(0, cy - Math.floor(options.size / 2))
    const x0 = Math.max(0, cx - Math.floor(options.size / 2))
    const y1 = Math.min(h, y0 + options.size)
    const x1 = Math.min(w, x0 + options.size)
    holes.set([x0, y0, x1, y1], i * 4)
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) for (let k = 0; k < c; k++) pixels[((i * h + y) * w + x) * c + k] = fill
  }
  return {
    data: { ...data, images: fromStorage(pixels, [n, h, w, c], 'float64') },
    truth: withLatents(input, { cutoutBoxes: fromStorage(holes, [n, 4], 'float64') }),
  }
}

/** Options of {@link withMixup} and {@link withCutmix}. */
export interface MixOptions {
  /** The Beta(alpha, alpha) concentration of the mixing weight (default 1; small values mix rarely and sharply). */
  readonly alpha?: number
  readonly key: Key
}

function mixingDraws(
  data: ImageData,
  options: MixOptions,
  name: string,
): { lambda: Float64Array; partner: Int32Array; labels: Int32Array | undefined; k: number } {
  const { n } = dims(data)
  const alpha = options.alpha ?? 1
  if (!(alpha > 0)) throw new MdmlError('value', `${name}: alpha must be positive`)
  const lambda = Float64Array.from(beta(foldIn(options.key, 0), [n], alpha, alpha).data)
  const partner = Int32Array.from(permutation(foldIn(options.key, 1), n).data)
  const labels = data.labels === undefined ? undefined : Int32Array.from(data.labels.data)
  const k = data.classes?.length ?? (labels === undefined ? 0 : maxOf(labels) + 1)
  return { lambda, partner, labels, k }
}

/** The soft label rows `lambda e_y + (1 - lambda) e_partner`, and the hard label of the larger share. */
function softLabels(
  labels: Int32Array,
  partner: Int32Array,
  lambda: ArrayLike<number>,
  k: number,
): { soft: Float64Array; hard: Int32Array } {
  const n = labels.length
  const soft = new Float64Array(n * k)
  const hard = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const own = labels[i] ?? 0
    const other = labels[partner[i] ?? 0] ?? 0
    const l = lambda[i] ?? 1
    soft[i * k + own] = (soft[i * k + own] ?? 0) + l
    soft[i * k + other] = (soft[i * k + other] ?? 0) + (1 - l)
    hard[i] = l >= 0.5 ? own : other
  }
  return { soft, hard }
}

/**
 * Mixup (Zhang et al.): each image is blended with a partner drawn by a random permutation, `lambda x_i + (1 - lambda)
 * x_partner`, with `lambda ~ Beta(alpha, alpha)` per image. Labels become the soft targets of the same blend. Boxes
 * and masks, which a blend leaves ambiguous, are dropped.
 *
 * @param input - Images with labels and their truth.
 * @param options - Concentration and key.
 * @returns The blended images, with labels set to the larger share; `truth.latents` holds `lambda` `[n]`, `partner`
 *   `[n]`, `originalLabels` `[n]` and `softLabels` `[n, classes]`.
 * @example
 * ```ts
 * import { asGenerated, images, withMixup } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(images([[[0]], [[1]]], { labels: [0, 1] }))
 * withMixup(d, { key: key(0) }).data.images.shape // => [2, 1, 1, 1]
 * ```
 */
export function withMixup(input: Generated, options: MixOptions): Generated<ImageData> {
  const { data } = requireImages(input, 'withMixup')
  const { n, h, w, c } = dims(data)
  const { lambda, partner, labels, k } = mixingDraws(data, options, 'withMixup')
  const per = h * w * c
  const pixels = new Float64Array(n * per)
  for (let i = 0; i < n; i++) {
    const l = lambda[i] ?? 1
    const p = partner[i] ?? 0
    for (let j = 0; j < per; j++)
      pixels[i * per + j] = l * (data.images.data[i * per + j] ?? 0) + (1 - l) * (data.images.data[p * per + j] ?? 0)
  }
  const mixed = labels === undefined ? undefined : softLabels(labels, partner, lambda, k)
  return {
    data: rebuild(data, pixels, h, w, {
      dropAnnotations: true,
      ...(mixed === undefined ? {} : { labels: fromStorage(mixed.hard, [n], 'int32') }),
    }),
    truth: withLatents(input, {
      lambda: fromStorage(lambda, [n], 'float64'),
      partner: fromStorage(partner, [n], 'int32'),
      ...(mixed === undefined || labels === undefined
        ? {}
        : {
            originalLabels: fromStorage(labels, [n], 'int32'),
            softLabels: fromStorage(mixed.soft, [n, k], 'float64'),
          }),
    }),
  }
}

/**
 * CutMix (Yun et al.): a rectangle of each image is replaced by the same rectangle of its partner. The rectangle has the
 * area fraction `1 - lambda` with `lambda ~ Beta(alpha, alpha)` and an aspect ratio equal to the image's, centred
 * uniformly (clipped at the edges); labels mix by the area actually pasted. Boxes and masks are dropped.
 *
 * @param input - Images with labels and their truth.
 * @param options - Concentration and key.
 * @returns The patched images, with labels set to the larger share; `truth.latents` holds `lambda` `[n]` (the area
 *   share kept after clipping), `partner`, `originalLabels`, `softLabels` and `cutBoxes` `[n, 4]` as `[x0, y0, x1, y1]`.
 * @example
 * ```ts
 * import { asGenerated, images, withCutmix } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(images([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], { labels: [0, 1] }))
 * withCutmix(d, { key: key(0) }).data.images.shape // => [2, 2, 2, 1]
 * ```
 */
export function withCutmix(input: Generated, options: MixOptions): Generated<ImageData> {
  const { data } = requireImages(input, 'withCutmix')
  const { n, h, w, c } = dims(data)
  const { lambda: drawn, partner, labels, k } = mixingDraws(data, options, 'withCutmix')
  const centres = uniform(foldIn(options.key, 2), [n, 2]).data
  const per = h * w * c
  const pixels = Float64Array.from(data.images.data)
  const lambda = new Float64Array(n)
  const cuts = new Float64Array(n * 4)
  for (let i = 0; i < n; i++) {
    const ratio = Math.sqrt(1 - (drawn[i] ?? 1))
    const ch = Math.floor(h * ratio)
    const cw = Math.floor(w * ratio)
    const cy = Math.floor((centres[i * 2] ?? 0) * h)
    const cx = Math.floor((centres[i * 2 + 1] ?? 0) * w)
    const y0 = Math.max(0, cy - Math.floor(ch / 2))
    const x0 = Math.max(0, cx - Math.floor(cw / 2))
    const y1 = Math.min(h, cy + Math.ceil(ch / 2))
    const x1 = Math.min(w, cx + Math.ceil(cw / 2))
    cuts.set([x0, y0, x1, y1], i * 4)
    lambda[i] = 1 - ((y1 - y0) * (x1 - x0)) / (h * w)
    const p = partner[i] ?? 0
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++)
        for (let q = 0; q < c; q++)
          pixels[i * per + (y * w + x) * c + q] = data.images.data[p * per + (y * w + x) * c + q] ?? 0
  }
  const mixed = labels === undefined ? undefined : softLabels(labels, partner, lambda, k)
  return {
    data: rebuild(data, pixels, h, w, {
      dropAnnotations: true,
      ...(mixed === undefined ? {} : { labels: fromStorage(mixed.hard, [n], 'int32') }),
    }),
    truth: withLatents(input, {
      lambda: fromStorage(lambda, [n], 'float64'),
      partner: fromStorage(partner, [n], 'int32'),
      cutBoxes: fromStorage(cuts, [n, 4], 'float64'),
      ...(mixed === undefined || labels === undefined
        ? {}
        : {
            originalLabels: fromStorage(labels, [n], 'int32'),
            softLabels: fromStorage(mixed.soft, [n, k], 'float64'),
          }),
    }),
  }
}

/**
 * {@link withRandomCrop} as a recipe modifier (same-size crops of zero-padded images).
 *
 * @example
 * ```ts
 * import { randomCropModifier } from 'mdml-engine/data'
 * randomCropModifier.name // => 'randomCrop'
 * ```
 */
export const randomCropModifier: Modifier = {
  name: 'randomCrop',
  description: 'Crop each image at a random position after zero padding.',
  knobs: dict({ padding: int(0, 64, { default: 2, label: 'padding in pixels' }) }),
  apply: (key, input, knobs) => withRandomCrop(input, { padding: knobs['padding'] as number, key }),
}

/**
 * {@link withRandomFlip} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { randomFlipModifier } from 'mdml-engine/data'
 * randomFlipModifier.name // => 'randomFlip'
 * ```
 */
export const randomFlipModifier: Modifier = {
  name: 'randomFlip',
  description: 'Mirror each image left-right and up-down with given chances.',
  knobs: dict({
    horizontal: real(0, 1, { default: 0.5, label: 'left-right flip chance' }),
    vertical: real(0, 1, { default: 0, label: 'up-down flip chance' }),
  }),
  apply: (key, input, knobs) =>
    withRandomFlip(input, { horizontal: knobs['horizontal'] as number, vertical: knobs['vertical'] as number, key }),
}

/**
 * {@link withCutout} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { cutoutModifier } from 'mdml-engine/data'
 * cutoutModifier.name // => 'cutout'
 * ```
 */
export const cutoutModifier: Modifier = {
  name: 'cutout',
  description: 'Blank a square hole at a random position in each image.',
  knobs: dict({ size: int(1, 64, { default: 8, label: 'hole side in pixels' }) }),
  apply: (key, input, knobs) => withCutout(input, { size: knobs['size'] as number, key }),
}

/**
 * {@link withMixup} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { mixupModifier } from 'mdml-engine/data'
 * mixupModifier.name // => 'mixup'
 * ```
 */
export const mixupModifier: Modifier = {
  name: 'mixup',
  description: 'Blend each image with a random partner, with Beta-distributed weights and soft labels.',
  knobs: dict({ alpha: real(0.01, 10, { default: 1, log: true, label: 'Beta concentration' }) }),
  apply: (key, input, knobs) => withMixup(input, { alpha: knobs['alpha'] as number, key }),
}

/**
 * {@link withCutmix} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { cutmixModifier } from 'mdml-engine/data'
 * cutmixModifier.name // => 'cutmix'
 * ```
 */
export const cutmixModifier: Modifier = {
  name: 'cutmix',
  description: 'Paste a random rectangle of a partner image, with labels mixed by area.',
  knobs: dict({ alpha: real(0.01, 10, { default: 1, log: true, label: 'Beta concentration' }) }),
  apply: (key, input, knobs) => withCutmix(input, { alpha: knobs['alpha'] as number, key }),
}
