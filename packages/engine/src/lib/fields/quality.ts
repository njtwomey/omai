/**
 * Measures and pictures shared by the neural-field fits: the peak signal-to-noise ratio of a mean squared error, and the
 * decoding of a field on a regular grid (`FieldGrid`, a plain serialisable record of a line, an image or a slice of a
 * volume), which is what a probe of a fit logs so a player can show the field sharpening step by step.
 */

import type { Module, Params } from '../nn/index.ts'
import { asTensor, fromStorage, toFlat, type Value } from '../tensor/index.ts'

/**
 * The peak signal-to-noise ratio in decibels of a mean squared error, `10 log10(range^2 / mse)`.
 *
 * @param mse - The mean squared error.
 * @param dataRange - The signal's peak-to-peak range (default 1, for images in [0, 1]).
 * @returns The ratio (infinite at zero error).
 * @concept image-quality
 * @example
 * ```ts
 * import { psnrFromMse } from 'mdml-engine/fields'
 * [psnrFromMse(0.01), psnrFromMse(0.0001)] // => [20, 40]
 * ```
 */
export function psnrFromMse(mse: number, dataRange = 1): number {
  return 10 * Math.log10((dataRange * dataRange) / mse)
}

/** A field decoded on a regular grid of cell centres over a box: a line, an image or a slice of a volume. */
export interface FieldGrid {
  readonly kind: 'fieldGrid'
  /** The field's input dimension (1, 2 or 3). */
  readonly dim: number
  /** The cells per side. */
  readonly resolution: number
  /** The values per cell. */
  readonly channels: number
  /** The coordinate range the grid covers on every axis. */
  readonly bounds: readonly [number, number]
  /** For a 3-D field, the coordinate of the plane decoded (the last coordinate). */
  readonly slice?: number
  /** Row-major values, channels interleaved: `resolution^min(dim, 2) * channels` of them (row 0 first). */
  readonly values: readonly number[]
}

/** Options of {@link decodeField}. */
export interface DecodeOptions {
  readonly dim: number
  /** Cells per side (default 32). */
  readonly resolution?: number
  /** The coordinate range (default `[-1, 1]`). */
  readonly bounds?: readonly [number, number]
  /** The last coordinate of a 3-D field's plane (default the middle of the bounds). */
  readonly slice?: number
}

/**
 * The coordinates of a decoding grid: cell centres over the bounds, `u` across and `v` down (a 3-D field gets its
 * `slice` as the third coordinate), as a matrix `[cells, dim]`.
 *
 * @param options - The dimension, resolution, bounds and slice.
 * @returns The coordinates.
 * @example
 * ```ts
 * import { gridCoordinates } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gridCoordinates({ dim: 2, resolution: 2 })) // => [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function gridCoordinates(options: DecodeOptions): Value {
  const res = options.resolution ?? 32
  const [lo, hi] = options.bounds ?? [-1, 1]
  const d = options.dim
  const at = (i: number): number => lo + ((hi - lo) * (i + 0.5)) / res
  const rows = d === 1 ? res : res * res
  const out = new Float64Array(rows * d)
  for (let r = 0; r < rows; r++) {
    if (d === 1) out[r] = at(r)
    else {
      out[r * d] = at(r % res)
      out[r * d + 1] = at(Math.floor(r / res))
      if (d === 3) out[r * d + 2] = options.slice ?? (lo + hi) / 2
    }
  }
  return fromStorage(out, [rows, d], 'float64')
}

/**
 * A field decoded on a regular grid: its value at the cell centres of a line (one dimension), an image (two) or a plane
 * of a volume (three).
 *
 * @param model - The field.
 * @param params - Its parameters.
 * @param options - The dimension, resolution, bounds and plane.
 * @returns The grid, plain data.
 * @concept neural-fields
 * @example
 * ```ts
 * import { decodeField, neuralField } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const f = neuralField({ inputDim: 2, outputDim: 3, width: 8, depth: 1, key: key(0) })
 * decodeField(f, f.params, { dim: 2, resolution: 4 }).values.length // => 48
 * ```
 */
export function decodeField(model: Module, params: Params, options: DecodeOptions): FieldGrid {
  const res = options.resolution ?? 32
  const y = model.apply(params, gridCoordinates(options), { train: false }) as Value
  const values = Array.from(toFlat(asTensor(y)))
  return {
    kind: 'fieldGrid',
    dim: options.dim,
    resolution: res,
    channels: values.length / (options.dim === 1 ? res : res * res),
    bounds: options.bounds ?? [-1, 1],
    ...(options.dim === 3
      ? { slice: options.slice ?? ((options.bounds ?? [-1, 1])[0] + (options.bounds ?? [-1, 1])[1]) / 2 }
      : {}),
    values,
  }
}
