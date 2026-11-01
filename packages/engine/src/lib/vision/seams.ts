/**
 * Seam carving (Avidan and Shamir 2007): the gradient energy of an image, the minimum-energy vertical seam by dynamic
 * programming (each row's pixel adjacent to the next's), its removal, and content-aware narrowing by removing seams
 * one at a time (transposing for height).
 */

import { MdmlError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
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

/** The paths {@link seamCarveAlgorithm} logs. */
export type SeamPath = 'seam/energy' | 'seam/cost' | 'seam/path' | 'seam/width'

/** What seam carving logs per seam: the energy under the seam, the seam itself, and the width left (in the carved direction). */
export const seamContract: Contract<unknown, SeamPath> = defineContract({
  'seam/energy': scalar({ description: 'the mean energy of the pixels on the seam removed', direction: 'lower' }),
  'seam/cost': scalar({
    description: 'the total energy of the seam removed (the minimum of the cumulative map)',
    direction: 'lower',
  }),
  'seam/path': tensor({ role: 'output', shape: ['rows'], description: 'the column removed in each row' }),
  'seam/width': scalar({ description: 'the width left in the direction being carved' }),
})

/** Options of {@link seamCarveAlgorithm} and {@link seamCarve}. */
export interface SeamCarveOptions {
  /** Pixels (nonzero in an `[H, W]` mask) whose removal is made very costly: the content to keep. */
  readonly protect?: TensorLike
  /** Pixels (nonzero in an `[H, W]` mask) whose removal is made cheap: the object to remove. */
  readonly remove?: TensorLike
  /** How strongly the masks bias the energy (default 1e6). */
  readonly bias?: number
}

/**
 * The state of {@link seamCarveAlgorithm}: the image as it is now (transposed while the height is being carved), the
 * energy bias of the masks carved along with it, and where the carving stands.
 */
export interface SeamCarveState {
  readonly h: number
  readonly w: number
  readonly c: number
  readonly rank: 2 | 3
  /** The image data, row-major `h` by `w` by `c`. */
  readonly data: Float64Array
  /** The mask bias per pixel, `h` by `w`. */
  readonly biasData: Float64Array
  /** `width` while removing columns, `height` while removing rows (the image is then transposed), `done` after. */
  readonly phase: 'width' | 'height' | 'done'
  readonly seams: number
}

/**
 * Content-aware resizing by seam carving as a stepped algorithm, one seam per step: the energy of the image (plus the bias
 * of the protect and remove masks), the minimum-energy vertical seam by dynamic programming, and its removal; horizontal
 * seams, by transposition, for the height. It logs the seam's mean energy and total cost, the seam and the width left,
 * and is done (`converged`) at the target size. The state serialises; read the image with {@link seamCarveImage}.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param size - The target [height, width], no larger than the image.
 * @param options - Masks to protect or remove, and their strength.
 * @returns The algorithm.
 * @throws MdmlError with code `value` unless the target fits the image, and `shape` for a mask of the wrong shape.
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamCarveAlgorithm, seamCarveImage } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const img = Array.from({ length: 6 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i * j) % 3))
 * const run = runAlgorithm(seamCarveAlgorithm(img, [5, 6]), { maxSteps: 20 })
 * [run.steps, seamCarveImage(run.state).shape] // => [3, [5, 6]]
 * ```
 */
export function seamCarveAlgorithm(
  x: TensorLike,
  size: readonly [number, number],
  options: SeamCarveOptions = {},
): Algorithm<SeamCarveState, SeamPath> {
  const r0 = rasterOf(x, 'seamCarve')
  const [th, tw] = size
  if (th > r0.h || tw > r0.w || th < 1 || tw < 1)
    throw new MdmlError('value', 'seamCarve: the target must fit the image')
  const strength = options.bias ?? 1e6
  const maskOf = (m: TensorLike | undefined, sign: number): Float64Array | undefined => {
    if (m === undefined) return undefined
    const mr = rasterOf(m, 'seamCarve')
    if (mr.h !== r0.h || mr.w !== r0.w || mr.c !== 1) throw new MdmlError('shape', 'seamCarve: a mask is [H, W]')
    return Float64Array.from(mr.data, (v) => (v !== 0 ? sign * strength : 0))
  }
  const keep = maskOf(options.protect, 1)
  const drop = maskOf(options.remove, -1)
  // Settle the phase: finish the width, turn to the height, finish.
  const settle = (s: SeamCarveState): SeamCarveState => {
    if (s.phase === 'width' && s.w <= tw) {
      if (s.h <= th) return { ...s, phase: 'done' }
      const t = transposed(s)
      return { ...s, h: t.h, w: t.w, data: t.data, biasData: transposeFlat(s.biasData, s.h, s.w), phase: 'height' }
    }
    if (s.phase === 'height' && s.w <= th) {
      const t = transposed(s)
      return { ...s, h: t.h, w: t.w, data: t.data, biasData: transposeFlat(s.biasData, s.h, s.w), phase: 'done' }
    }
    return s
  }
  return {
    name: 'vision/seam-carving',
    config: { target: [th, tw], height: r0.h, width: r0.w },
    init: () =>
      settle({
        h: r0.h,
        w: r0.w,
        c: r0.c,
        rank: r0.rank,
        data: Float64Array.from(r0.data),
        biasData: Float64Array.from({ length: r0.h * r0.w }, (_, i) => (keep?.[i] ?? 0) + (drop?.[i] ?? 0)),
        phase: 'width',
        seams: 0,
      }),
    step: (s, ctx) => {
      const raster: Raster = { h: s.h, w: s.w, c: s.c, rank: s.rank, data: s.data }
      const energy = rasterOf(seamEnergy(rasterTensor(raster)), 'seamCarve')
      for (let i = 0; i < energy.data.length; i++) energy.data[i] = (energy.data[i] ?? 0) + (s.biasData[i] ?? 0)
      const seam = findVerticalSeam(imageTensor(energy.data, s.h, s.w))
      const cost = seam.reduce((acc, col, y) => acc + (energy.data[y * s.w + col] ?? 0), 0)
      const carved = dropSeam(raster, seam)
      const bias = dropSeam({ h: s.h, w: s.w, c: 1, rank: 2, data: s.biasData }, seam)
      ctx.log('seam/cost', cost)
      ctx.log('seam/energy', cost / s.h)
      ctx.log('seam/path', seam)
      ctx.log('seam/width', carved.w)
      return settle({ ...s, w: carved.w, data: carved.data, biasData: bias.data, seams: s.seams + 1 })
    },
    done: (s) => (s.phase === 'done' ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SeamCarveState,
  }
}

function transposeFlat(data: Float64Array, h: number, w: number): Float64Array {
  return transposed({ h, w, c: 1, rank: 2, data }).data
}

/**
 * The carved image of a seam-carving state.
 *
 * @param state - A state of {@link seamCarveAlgorithm}.
 * @returns The image in its original orientation (a state part-way through the height is turned back).
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamCarveAlgorithm, seamCarveImage } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(seamCarveAlgorithm([[1, 2, 3], [4, 5, 6]], [2, 2]), { maxSteps: 5 })
 * seamCarveImage(run.state).shape // => [2, 2]
 * ```
 */
export function seamCarveImage(state: SeamCarveState): Tensor {
  const r: Raster = { h: state.h, w: state.w, c: state.c, rank: state.rank, data: state.data }
  return rasterTensor(state.phase === 'height' ? transposed(r) : r)
}

/**
 * Content-aware resizing by seam carving: repeatedly remove the minimum-energy vertical seam (horizontal seams, by
 * transposition, for the height) until the image has the target size. {@link seamCarveAlgorithm} run to the end.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param size - The target [height, width], no larger than the image.
 * @param options - Masks to protect or remove content.
 * @returns The carved image.
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamCarve } from 'mdml-engine/vision'
 * seamCarve(Array.from({ length: 6 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i * j) % 3)), [5, 6]).shape // => [5, 6]
 * ```
 */
export function seamCarve(x: TensorLike, size: readonly [number, number], options: SeamCarveOptions = {}): Tensor {
  const run = runAlgorithm(seamCarveAlgorithm(x, size, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return seamCarveImage(run.state)
}
