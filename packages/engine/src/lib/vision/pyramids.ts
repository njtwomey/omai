/**
 * Image pyramids with OpenCV's 5-tap Burt-Adelson kernel [1, 4, 6, 4, 1] / 16: `pyrDown` (smooth and drop every other
 * row and column), `pyrUp` (insert zeros and smooth, with OpenCV's border rules), the Gaussian pyramid `buildPyramid`,
 * the Laplacian pyramid and its exact reconstruction, and multiresolution blending of two images under a mask (Burt
 * and Adelson 1983). scikit-image's Gaussian pyramids are `signal/image`'s.
 */

import { OmaiError } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { primalOf, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Raster, rasterOf, rasterTensor } from './_image.ts'
import { borderMode, type CvBorder } from './blur.ts'

const KERNEL = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]

/**
 * Downsample an image by two (cv2.pyrDown): the 5 x 5 kernel [1, 4, 6, 4, 1]^2 / 256 with OpenCV's reflect-101
 * border, keeping even rows and columns; the result is ceil(H / 2) x ceil(W / 2).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - The border (default 'reflect101').
 * @returns The reduced image.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { pyrDown } from 'omai-engine/vision'
 * pyrDown(Array.from({ length: 5 }, () => new Array(7).fill(1))).shape // => [3, 4]
 * ```
 */
export function pyrDown(x: TensorLike, options: { readonly border?: CvBorder } = {}): Tensor {
  const r = rasterOf(x, 'pyrDown')
  const mode = borderMode(options.border ?? 'reflect101')
  let v = image.correlate1d(rasterTensor(r), KERNEL, 0, { mode })
  v = image.correlate1d(v, KERNEL, 1, { mode })
  const s = primalOf(v).data
  const h = Math.ceil(r.h / 2)
  const w = Math.ceil(r.w / 2)
  const out = new Float64Array(h * w * r.c)
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++)
      for (let k = 0; k < r.c; k++) out[(y * w + xx) * r.c + k] = Number(s[(2 * y * r.w + 2 * xx) * r.c + k] ?? 0)
  return rasterTensor({ ...r, h, w, data: out })
}

/**
 * Upsample an image by two (cv2.pyrUp): even outputs take src[x - 1] + 6 src[x] + src[x + 1], odd ones
 * 4 (src[x] + src[x + 1]), along each axis, over 64; OpenCV reflects at the start and repeats at the end. The output
 * is 2H x 2W, or the given size (one less on an axis crops it).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param size - The output size [height, width] (default [2H, 2W]).
 * @returns The expanded image.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { pyrUp } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(pyrUp([[1, 1], [1, 1]])).flat().every((v) => v === 1) // => true
 * ```
 */
export function pyrUp(x: TensorLike, size?: readonly [number, number]): Tensor {
  const r = rasterOf(x, 'pyrUp')
  const [oh, ow] = size ?? [2 * r.h, 2 * r.w]
  if (Math.abs(oh - 2 * r.h) > 1 || Math.abs(ow - 2 * r.w) > 1)
    throw new OmaiError('value', 'pyrUp: the output size must be within one of twice the input')
  // Horizontal pass into rows of 2W values.
  const up = (line: (i: number) => number, n: number, out: Float64Array): void => {
    if (n === 1) {
      out[0] = 8 * line(0)
      out[1] = 8 * line(0)
      return
    }
    out[0] = 6 * line(0) + 2 * line(1)
    out[1] = 4 * (line(0) + line(1))
    for (let i = 1; i < n - 1; i++) {
      out[2 * i] = line(i - 1) + 6 * line(i) + line(i + 1)
      out[2 * i + 1] = 4 * (line(i) + line(i + 1))
    }
    out[2 * n - 2] = line(n - 2) + 7 * line(n - 1)
    out[2 * n - 1] = 8 * line(n - 1)
  }
  const W2 = 2 * r.w
  const rows = new Float64Array(r.h * W2 * r.c)
  const tmp = new Float64Array(W2)
  for (let y = 0; y < r.h; y++)
    for (let k = 0; k < r.c; k++) {
      up((i) => r.data[(y * r.w + i) * r.c + k] ?? 0, r.w, tmp)
      for (let i = 0; i < W2; i++) rows[(y * W2 + i) * r.c + k] = tmp[i] ?? 0
    }
  const H2 = 2 * r.h
  const full = new Float64Array(H2 * W2 * r.c)
  const col = new Float64Array(H2)
  for (let xx = 0; xx < W2; xx++)
    for (let k = 0; k < r.c; k++) {
      up((i) => rows[(i * W2 + xx) * r.c + k] ?? 0, r.h, col)
      for (let i = 0; i < H2; i++) full[(i * W2 + xx) * r.c + k] = (col[i] ?? 0) / 64
    }
  // A size one larger repeats, as OpenCV does, the last odd column and the last even row.
  const out = new Float64Array(oh * ow * r.c)
  for (let y = 0; y < oh; y++)
    for (let xx = 0; xx < ow; xx++) {
      const sy = y < H2 ? y : H2 - 2
      const sx = xx < W2 ? xx : W2 - 1
      for (let k = 0; k < r.c; k++) out[(y * ow + xx) * r.c + k] = full[(Math.max(0, sy) * W2 + sx) * r.c + k] ?? 0
    }
  return rasterTensor({ ...r, h: oh, w: ow, data: out })
}

/**
 * The Gaussian pyramid (cv2.buildPyramid): the image and maxLevel successive {@link pyrDown} reductions.
 *
 * @param x - Image.
 * @param maxLevel - Levels below the image.
 * @returns The levels, finest first.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { buildPyramid } from 'omai-engine/vision'
 * buildPyramid(Array.from({ length: 8 }, () => new Array(8).fill(0)), 2).map((l) => l.shape[0]) // => [8, 4, 2]
 * ```
 */
export function buildPyramid(x: TensorLike, maxLevel: number): Tensor[] {
  const levels = [rasterTensor(rasterOf(x, 'buildPyramid'))]
  for (let l = 0; l < maxLevel; l++) levels.push(pyrDown(levels[l] as Tensor))
  return levels
}

const subtract = (a: Raster, b: Raster): Raster => ({
  ...a,
  data: Float64Array.from(a.data, (v, i) => v - (b.data[i] ?? 0)),
})
const plus = (a: Raster, b: Raster): Raster => ({
  ...a,
  data: Float64Array.from(a.data, (v, i) => v + (b.data[i] ?? 0)),
})

/**
 * The Laplacian pyramid (Burt and Adelson 1983): each Gaussian level minus the expansion of the next (as OpenCV's
 * samples compute it, with pyrUp to the finer level's size), the coarsest Gaussian level last.
 *
 * @param x - Image.
 * @param levels - Band-pass levels (the result has levels + 1 images).
 * @returns The pyramid, finest first.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { laplacianPyramid } from 'omai-engine/vision'
 * laplacianPyramid(Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => i + j)), 2).map((l) => l.shape[0]) // => [8, 4, 2]
 * ```
 */
export function laplacianPyramid(x: TensorLike, levels: number): Tensor[] {
  const g = buildPyramid(x, levels)
  const out: Tensor[] = []
  for (let l = 0; l < levels; l++) {
    const fine = rasterOf(g[l] as Tensor, 'laplacianPyramid')
    const up = rasterOf(pyrUp(g[l + 1] as Tensor, [fine.h, fine.w]), 'laplacianPyramid')
    out.push(rasterTensor(subtract(fine, up)))
  }
  out.push(g[levels] as Tensor)
  return out
}

/**
 * Reconstruct an image from its Laplacian pyramid: from the coarsest level, expand and add each band; exact up to
 * rounding.
 *
 * @param pyramid - A Laplacian pyramid, finest first.
 * @returns The image.
 * @concept image-pyramid
 * @example
 * ```ts
 * import { laplacianPyramid, reconstructLaplacian } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i * j) % 5))
 * toFlat(reconstructLaplacian(laplacianPyramid(x, 2))).map((v) => Math.round(v * 1e9) / 1e9) // => x.flat()
 * ```
 */
export function reconstructLaplacian(pyramid: readonly TensorLike[]): Tensor {
  if (pyramid.length === 0) throw new OmaiError('value', 'reconstructLaplacian: the pyramid is empty')
  let cur = rasterOf(pyramid[pyramid.length - 1] as TensorLike, 'reconstructLaplacian')
  for (let l = pyramid.length - 2; l >= 0; l--) {
    const band = rasterOf(pyramid[l] as TensorLike, 'reconstructLaplacian')
    const up = rasterOf(pyrUp(rasterTensor(cur), [band.h, band.w]), 'reconstructLaplacian')
    cur = plus(band, up)
  }
  return rasterTensor(cur)
}

/**
 * Multiresolution blending (Burt and Adelson 1983): the Laplacian pyramids of the two images are mixed level by level
 * with the Gaussian pyramid of the mask (1 takes the first image), and the mixture reconstructed, so the seam is
 * smoothed at every scale.
 *
 * @param a - First image.
 * @param b - Second image, the same shape.
 * @param mask - Weights in [0, 1] `[H, W]` (or the image's shape).
 * @param levels - Pyramid levels (default 4).
 * @returns The blend.
 * @concept image-blending
 * @example
 * ```ts
 * import { pyramidBlend } from 'omai-engine/vision'
 * const a = Array.from({ length: 16 }, () => new Array(16).fill(1))
 * const b = Array.from({ length: 16 }, () => new Array(16).fill(0))
 * const m = Array.from({ length: 16 }, () => Array.from({ length: 16 }, (_, j) => (j < 8 ? 1 : 0)))
 * pyramidBlend(a, b, m, 2).shape // => [16, 16]
 * ```
 */
export function pyramidBlend(a: TensorLike, b: TensorLike, mask: TensorLike, levels = 4): Tensor {
  const ra = rasterOf(a, 'pyramidBlend')
  const rb = rasterOf(b, 'pyramidBlend')
  if (ra.h !== rb.h || ra.w !== rb.w || ra.c !== rb.c)
    throw new OmaiError('shape', 'pyramidBlend: images differ in shape')
  const la = laplacianPyramid(a, levels)
  const lb = laplacianPyramid(b, levels)
  const gm = buildPyramid(mask, levels)
  const mixed = la.map((pa, l) => {
    const A = rasterOf(pa, 'pyramidBlend')
    const B = rasterOf(lb[l] as Tensor, 'pyramidBlend')
    const M = rasterOf(gm[l] as Tensor, 'pyramidBlend')
    const data = Float64Array.from(A.data, (v, i) => {
      const m = M.data[M.c === A.c ? i : Math.floor(i / A.c)] ?? 0
      return m * v + (1 - m) * (B.data[i] ?? 0)
    })
    return rasterTensor({ ...A, data })
  })
  return reconstructLaplacian(mixed)
}
