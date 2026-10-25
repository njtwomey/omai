// Internal helpers of the chart builders: converting inputs to plain numbers and resolving colours.
import { MdmlError } from 'mdml-engine/base'
import type { Paint } from 'mdml-engine/scene'
import { asTensor, type TensorLike, toFlat } from 'mdml-engine/tensor'
import { categoricalColour } from '../theme/palettes.ts'
import { type ColourMode, resolvePaint } from '../theme/tokens.ts'

/** Numbers from any array input (a tensor, a typed array or nested arrays, flattened). */
export function numbers(x: TensorLike | readonly number[]): number[] {
  if (Array.isArray(x) && x.every((v) => typeof v === 'number')) return x as number[]
  return toFlat(asTensor(x as TensorLike))
}

/** A 2-D grid of values `[rows, cols]` from any input, row-major. */
export interface Grid2D {
  readonly rows: number
  readonly cols: number
  readonly values: Float64Array
}

/** A 2-D grid from nested arrays or a 2-D tensor. */
export function grid2d(z: TensorLike, name: string): Grid2D {
  const t = asTensor(z)
  if (!('shape' in t) || t.shape.length !== 2) throw new MdmlError('shape', `${name}: the grid must be 2-D`)
  const [rows = 0, cols = 0] = t.shape
  return { rows, cols, values: Float64Array.from(toFlat(t)) }
}

/** The grid as nested arrays, row by row. */
export function gridRows(g: Grid2D): number[][] {
  return Array.from({ length: g.rows }, (_, r) => Array.from(g.values.subarray(r * g.cols, (r + 1) * g.cols)))
}

/** A CSS colour for a paint (tokens resolved for the mode), or the categorical colour of `index` when unset. */
export function colourFor(paint: string | undefined, mode: ColourMode, index = 0): string {
  return paint === undefined ? categoricalColour(index) : resolvePaint(paint as Paint, mode)
}

/** ECharts line type for a dash option. */
export function dashType(dash: 'solid' | 'dashed' | 'dotted' | readonly number[] | undefined): string | number[] {
  if (dash === undefined) return 'solid'
  return typeof dash === 'string' ? dash : [...dash]
}

/** Coordinates along an axis: given values, or `n` points spread evenly over a range. */
export function axisCoords(coords: readonly number[] | readonly [number, number] | undefined, n: number): number[] {
  if (coords === undefined) return Array.from({ length: n }, (_, i) => i)
  if (coords.length === n) return [...coords]
  if (coords.length === 2) {
    const [a = 0, b = 1] = coords
    return Array.from({ length: n }, (_, i) => (n === 1 ? (a + b) / 2 : a + ((b - a) * i) / (n - 1)))
  }
  throw new MdmlError('shape', 'axis coordinates: give one value per grid line or a [low, high] range')
}

/** The colour mode of a chart's options: the stated mode, else the figure's, else light. */
export function modeOf(options: {
  readonly mode?: ColourMode
  readonly figure?: { readonly mode: ColourMode }
}): ColourMode {
  return options.mode ?? options.figure?.mode ?? 'light'
}
