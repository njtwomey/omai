/**
 * Seam carving (Avidan and Shamir 2007): the gradient energy of an image, the minimum-energy vertical seam by dynamic
 * programming (each row's pixel adjacent to the next's), its removal, and content-aware narrowing by removing seams
 * one at a time (transposing for height).
 */

import { MdmlError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { imageTensor, type Raster, rasterOf, rasterTensor } from './_image.ts'

/**
 * The seam-carving energy (Avidan and Shamir 2007): the sum over channels of |dI/dx| + |dI/dy| by central differences
 * (one-sided at the borders).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @returns The energy `[H, W]`.
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamEnergy } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(seamEnergy([[0, 0, 1, 1]])) // => [[0, 0.5, 0.5, 0]]
 * ```
 */
export function seamEnergy(x: TensorLike): Tensor {
  const r = rasterOf(x, 'seamEnergy')
  const { h, w, c } = r
  const v = (y: number, xx: number, k: number): number => r.data[(y * w + xx) * c + k] ?? 0
  const out = new Float64Array(h * w)
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++) {
      let e = 0
      for (let k = 0; k < c; k++) {
        const x0 = Math.max(0, xx - 1)
        const x1 = Math.min(w - 1, xx + 1)
        const y0 = Math.max(0, y - 1)
        const y1 = Math.min(h - 1, y + 1)
        if (x1 > x0) e += Math.abs(v(y, x1, k) - v(y, x0, k)) / (x1 - x0)
        if (y1 > y0) e += Math.abs(v(y1, xx, k) - v(y0, xx, k)) / (y1 - y0)
      }
      out[y * w + xx] = e
    }
  return imageTensor(out, h, w)
}

/**
 * The minimum-energy vertical seam: dynamic programming over M(y, x) = e(y, x) + min(M(y - 1, x - 1 .. x + 1)), then
 * the path traced back from the smallest M of the last row (ties to the left).
 *
 * @param energy - Energy `[H, W]`.
 * @returns The seam's column in every row, top to bottom.
 * @concept seam-carving
 * @example
 * ```ts
 * import { findVerticalSeam } from 'mdml-engine/vision'
 * findVerticalSeam([[5, 1, 5], [5, 5, 1], [5, 1, 5]]) // => [1, 2, 1]
 * ```
 */
export function findVerticalSeam(energy: TensorLike): number[] {
  const r = rasterOf(energy, 'findVerticalSeam')
  if (r.c !== 1) throw new MdmlError('shape', 'findVerticalSeam: the energy is [H, W]')
  const { h, w } = r
  const M = Float64Array.from(r.data)
  const back = new Int32Array(h * w)
  for (let y = 1; y < h; y++)
    for (let xx = 0; xx < w; xx++) {
      // The cheapest of the three pixels above, ties to the left.
      let best = -1
      for (const k of [xx - 1, xx, xx + 1])
        if (k >= 0 && k < w && (best < 0 || (M[(y - 1) * w + k] ?? 0) < (M[(y - 1) * w + best] ?? 0))) best = k
      back[y * w + xx] = best
      M[y * w + xx] = (M[y * w + xx] ?? 0) + (M[(y - 1) * w + best] ?? 0)
    }
  let col = 0
  for (let xx = 1; xx < w; xx++) if ((M[(h - 1) * w + xx] ?? 0) < (M[(h - 1) * w + col] ?? 0)) col = xx
  const seam = new Array<number>(h).fill(0)
  for (let y = h - 1; y >= 0; y--) {
    seam[y] = col
    col = back[y * w + col] ?? 0
  }
  return seam
}

function dropSeam(r: Raster, seam: readonly number[]): Raster {
  const { h, w, c } = r
  const out = new Float64Array(h * (w - 1) * c)
  for (let y = 0; y < h; y++) {
    let o = 0
    for (let xx = 0; xx < w; xx++) {
      if (xx === seam[y]) continue
      for (let k = 0; k < c; k++) out[(y * (w - 1) + o) * c + k] = r.data[(y * w + xx) * c + k] ?? 0
      o++
    }
  }
  return { ...r, w: w - 1, data: out }
}

/**
 * Remove a vertical seam: one pixel per row, the image one column narrower.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param seam - The column to remove in each row.
 * @returns The narrower image.
 * @concept seam-carving
 * @example
 * ```ts
 * import { removeVerticalSeam } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(removeVerticalSeam([[1, 2, 3], [4, 5, 6]], [0, 2])) // => [[2, 3], [4, 5]]
 * ```
 */
export function removeVerticalSeam(x: TensorLike, seam: readonly number[]): Tensor {
  const r = rasterOf(x, 'removeVerticalSeam')
  if (seam.length !== r.h || seam.some((v) => !(v >= 0 && v < r.w)))
    throw new MdmlError('value', 'removeVerticalSeam: one column per row, inside the image')
  return rasterTensor(dropSeam(r, seam))
}

const transposed = (r: Raster): Raster => {
  const out = new Float64Array(r.data.length)
  for (let y = 0; y < r.h; y++)
    for (let xx = 0; xx < r.w; xx++)
      for (let k = 0; k < r.c; k++) out[(xx * r.h + y) * r.c + k] = r.data[(y * r.w + xx) * r.c + k] ?? 0
  return { ...r, h: r.w, w: r.h, data: out }
}

/**
 * Content-aware resizing by seam carving: repeatedly remove the minimum-energy vertical seam (horizontal seams, by
 * transposition, for the height) until the image has the target size.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param size - The target [height, width], no larger than the image.
 * @returns The carved image.
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamCarve } from 'mdml-engine/vision'
 * seamCarve(Array.from({ length: 6 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i * j) % 3)), [5, 6]).shape // => [5, 6]
 * ```
 */
export function seamCarve(x: TensorLike, size: readonly [number, number]): Tensor {
  let r = rasterOf(x, 'seamCarve')
  const [th, tw] = size
  if (th > r.h || tw > r.w || th < 1 || tw < 1) throw new MdmlError('value', 'seamCarve: the target must fit the image')
  while (r.w > tw) r = dropSeam(r, findVerticalSeam(seamEnergy(rasterTensor(r))))
  if (r.h > th) {
    r = transposed(r)
    while (r.w > th) r = dropSeam(r, findVerticalSeam(seamEnergy(rasterTensor(r))))
    r = transposed(r)
  }
  return rasterTensor(r)
}
