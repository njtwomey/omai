/**
 * Tone and colour balance of images in [0, 1]: the sRGB transfer curve both ways, power-law gamma correction (as
 * skimage.exposure.adjust_gamma), white balance by the grey-world and white-patch assumptions, and splitting an
 * image into its channel planes and merging them back (cv2.split and cv2.merge).
 */

import { MdmlError } from '../base/index.ts'
import { quantile } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { imageTensor, rasterOf, rasterTensor, srgbDecode, srgbEncode, valuesOf } from './_image.ts'

const elementwise = (x: TensorLike, f: (v: number) => number): Tensor => {
  const { data, shape } = valuesOf(x)
  return fromStorage(
    Float64Array.from(data, (v) => f(v)),
    [...shape],
    'float64',
  )
}

/**
 * Decode the sRGB transfer curve (gamma expansion): v / 12.92 below 0.04045, ((v + 0.055) / 1.055)^2.4 above;
 * values are clipped to [0, 1] first.
 *
 * @param x - An sRGB-encoded image (any shape).
 * @returns The linear-light image.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { srgbToLinear } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(srgbToLinear([0, 0.5, 1])).map((v) => Math.round(v * 1e4) / 1e4) // => [0, 0.214, 1]
 * ```
 */
export function srgbToLinear(x: TensorLike): Tensor {
  return elementwise(x, (v) => srgbDecode(Math.min(1, Math.max(0, v))))
}

/**
 * Encode the sRGB transfer curve (gamma compression), the inverse of {@link srgbToLinear}; values are clipped to
 * [0, 1] first.
 *
 * @param x - A linear-light image (any shape).
 * @returns The sRGB-encoded image.
 * @concept gamma-correction
 * @example
 * ```ts
 * import { linearToSrgb } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(linearToSrgb([0, 0.214, 1])).map((v) => Math.round(v * 1e3) / 1e3) // => [0, 0.5, 1]
 * ```
 */
export function linearToSrgb(x: TensorLike): Tensor {
  return elementwise(x, (v) => srgbEncode(Math.min(1, Math.max(0, v))))
}

/**
 * Power-law gamma correction (skimage.exposure.adjust_gamma): gain * v^gamma for an image in [0, 1]; gamma below 1
 * brightens the shadows, above 1 darkens them.
 *
 * @param x - Image in [0, 1] (any shape).
 * @param gamma - The exponent (default 1).
 * @param gain - The multiplier (default 1).
 * @returns The corrected image.
 * @throws MdmlError with code `value` for a negative gamma or values outside [0, 1].
 * @concept gamma-correction
 * @example
 * ```ts
 * import { adjustGamma } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(adjustGamma([0.25, 1], 0.5)) // => [0.5, 1]
 * ```
 */
export function adjustGamma(x: TensorLike, gamma = 1, gain = 1): Tensor {
  if (!(gamma >= 0)) throw new MdmlError('value', 'adjustGamma: gamma must be non-negative')
  const { data } = valuesOf(x)
  if (data.some((v) => v < 0 || v > 1)) throw new MdmlError('value', 'adjustGamma: values must lie in [0, 1]')
  return elementwise(x, (v) => gain * v ** gamma)
}

/** Options of {@link whiteBalance}. */
export interface WhiteBalanceOptions {
  /**
   * 'grey-world' scales each channel so its mean matches the mean over channels (the scene averages to grey);
   * 'white-patch' scales each channel so its brightest value (or the given percentile) becomes 1 (default).
   */
  readonly method?: 'grey-world' | 'white-patch'
  /** For 'white-patch': the percentile of each channel taken as white (default 100, the maximum). */
  readonly percentile?: number
}

/**
 * White balance of a colour image by a per-channel gain: grey world (each channel's mean scaled to the mean of the
 * channel means) or white patch (each channel's maximum, or a high percentile, scaled to 1); results are clipped to
 * [0, 1].
 *
 * @param x - Image `[H, W, C]`.
 * @param options - Method and white-patch percentile.
 * @returns The balanced image.
 * @concept white-balance
 * @example
 * ```ts
 * import { whiteBalance } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(whiteBalance([[[0.4, 0.2, 0.3], [0.2, 0.1, 0.15]]], { method: 'grey-world' }))[0][0] // => [0.3, 0.3, 0.3]
 * ```
 */
export function whiteBalance(x: TensorLike, options: WhiteBalanceOptions = {}): Tensor {
  const r = rasterOf(x, 'whiteBalance')
  if (r.rank !== 3) throw new MdmlError('shape', 'whiteBalance: expected an image [H, W, C]')
  const n = r.h * r.w
  const channel = (k: number): Float64Array => Float64Array.from({ length: n }, (_, i) => r.data[i * r.c + k] ?? 0)
  const gains = new Float64Array(r.c)
  if ((options.method ?? 'white-patch') === 'grey-world') {
    const means = Array.from({ length: r.c }, (_, k) => channel(k).reduce((a, b) => a + b, 0) / n)
    const grey = means.reduce((a, b) => a + b, 0) / r.c
    for (let k = 0; k < r.c; k++) gains[k] = (means[k] ?? 0) > 0 ? grey / (means[k] ?? 1) : 1
  } else {
    const p = options.percentile ?? 100
    for (let k = 0; k < r.c; k++) {
      const white = quantile(channel(k), p / 100)
      gains[k] = white > 0 ? 1 / white : 1
    }
  }
  const out = Float64Array.from(r.data, (v, i) => Math.min(1, Math.max(0, v * (gains[i % r.c] ?? 1))))
  return rasterTensor({ ...r, data: out })
}

/**
 * Split an image into its channel planes (cv2.split).
 *
 * @param x - Image `[H, W, C]` (an `[H, W]` image is one plane).
 * @returns C planes `[H, W]`.
 * @concept colour-channels
 * @example
 * ```ts
 * import { splitChannels } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * splitChannels([[[1, 2, 3]]]).map((p) => toArray(p)) // => [[[1]], [[2]], [[3]]]
 * ```
 */
export function splitChannels(x: TensorLike): Tensor[] {
  const r = rasterOf(x, 'splitChannels')
  return Array.from({ length: r.c }, (_, k) =>
    imageTensor(
      Float64Array.from({ length: r.h * r.w }, (_, i) => r.data[i * r.c + k] ?? 0),
      r.h,
      r.w,
    ),
  )
}

/**
 * Merge channel planes into one image (cv2.merge).
 *
 * @param planes - Planes `[H, W]` of one shape.
 * @returns The image `[H, W, C]`.
 * @throws MdmlError with code `shape` when the planes differ in shape.
 * @concept colour-channels
 * @example
 * ```ts
 * import { mergeChannels } from 'mdml-engine/vision'
 * mergeChannels([[[1]], [[2]], [[3]]]).shape // => [1, 1, 3]
 * ```
 */
export function mergeChannels(planes: readonly TensorLike[]): Tensor {
  if (planes.length === 0) throw new MdmlError('value', 'mergeChannels: no planes')
  const rs = planes.map((p) => rasterOf(p, 'mergeChannels'))
  const { h, w } = rs[0] as { h: number; w: number }
  if (rs.some((r) => r.h !== h || r.w !== w || r.c !== 1))
    throw new MdmlError('shape', 'mergeChannels: planes must be [H, W] of one shape')
  const c = rs.length
  const out = new Float64Array(h * w * c)
  for (const [k, r] of rs.entries()) for (let i = 0; i < h * w; i++) out[i * c + k] = r.data[i] ?? 0
  return imageTensor(out, h, w, c, 3)
}
