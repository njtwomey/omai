/**
 * Box aspect versus data aspect: a figure's shape is fixed, and a data aspect (`equal`, so a circle looks like a
 * circle, or a ratio) is met either by widening the short axis range about its centre (`fitDataAspect`, matplotlib's
 * `adjustable='datalim'`) or by shrinking the plotting box (`boxForAspect`, `adjustable='box'`).
 */

import { MdmlError } from 'mdml-engine/base'

/** A closed interval `[low, high]` of data values. */
export type Range = readonly [low: number, high: number]

/**
 * The data aspect of axes: `auto` (ranges fill the box however they stretch), `equal` (one unit on x is as long as one
 * unit on y, so a circle looks like a circle), or a number (the length of a y unit over the length of an x unit).
 */
export type DataAspect = 'auto' | 'equal' | number

/**
 * Adjust axis ranges so the data aspect holds in a box of the given size: the range that is too short for the box is
 * widened about its centre (never shrunk), as matplotlib's `adjustable='datalim'` does. The box aspect is independent:
 * with `equal` data aspect in a 2:1 box, the x range becomes twice the y range.
 *
 * @param x - The x range.
 * @param y - The y range.
 * @param box - The plotting area's width and height in pixels.
 * @param aspect - The data aspect.
 * @returns The adjusted ranges.
 * @throws MdmlError with code `value` for a non-positive numeric aspect.
 * @index function Widen an axis range to meet a data aspect: `fitDataAspect(x, y, box, 'equal')`.
 * @example
 * ```ts
 * import { fitDataAspect } from 'mdml-render/figure'
 * fitDataAspect([-1, 1], [-1, 1], { width: 400, height: 200 }, 'equal') // => { x: [-2, 2], y: [-1, 1] }
 * fitDataAspect([0, 4], [0, 1], { width: 100, height: 100 }, 'equal') // => { x: [0, 4], y: [-1.5, 2.5] }
 * ```
 */
export function fitDataAspect(
  x: Range,
  y: Range,
  box: { readonly width: number; readonly height: number },
  aspect: DataAspect,
): { readonly x: Range; readonly y: Range } {
  if (aspect === 'auto') return { x, y }
  const ratio = aspect === 'equal' ? 1 : aspect
  if (!(ratio > 0)) throw new MdmlError('value', 'fitDataAspect: the aspect must be positive')
  if (!(box.width > 0 && box.height > 0)) return { x, y }
  const dx = x[1] - x[0]
  const dy = y[1] - y[0]
  if (!(dx > 0) || !(dy > 0)) return { x, y }
  // Pixels per data unit on each axis; equal aspect wants pxPerY = ratio * pxPerX.
  const px = box.width / dx
  const py = box.height / dy
  if (py > ratio * px) {
    const want = box.height / (ratio * px)
    const c = (y[0] + y[1]) / 2
    return { x, y: [c - want / 2, c + want / 2] }
  }
  const want = (box.width * ratio) / py
  const c = (x[0] + x[1]) / 2
  return { x: [c - want / 2, c + want / 2], y }
}

/**
 * The plotting-area size that gives a data aspect exactly for fixed ranges, fitting inside a box (the other way of
 * satisfying an aspect: change the box instead of the ranges, as `adjustable='box'`).
 *
 * @param x - The x range.
 * @param y - The y range.
 * @param box - The largest allowed width and height.
 * @param aspect - The data aspect (`auto` returns the box).
 * @returns The width and height.
 * @example
 * ```ts
 * import { boxForAspect } from 'mdml-render/figure'
 * boxForAspect([0, 2], [0, 1], { width: 300, height: 300 }, 'equal') // => { width: 300, height: 150 }
 * ```
 */
export function boxForAspect(
  x: Range,
  y: Range,
  box: { readonly width: number; readonly height: number },
  aspect: DataAspect,
): { readonly width: number; readonly height: number } {
  if (aspect === 'auto') return box
  const ratio = aspect === 'equal' ? 1 : aspect
  const dx = x[1] - x[0]
  const dy = y[1] - y[0]
  if (!(dx > 0 && dy > 0 && ratio > 0)) return box
  const want = (dy * ratio) / dx
  return want > box.height / box.width
    ? { width: box.height / want, height: box.height }
    : { width: box.width, height: box.width * want }
}
