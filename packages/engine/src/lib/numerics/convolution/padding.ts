/**
 * Padding and border handling: numpy.pad modes (constant, edge, reflect, symmetric, wrap and line statistics),
 * scipy.ndimage border modes and the translation between their names, and index extension for filters that read past an
 * array's edges.
 */

import { MdmlError } from '../../base/index.ts'
import {
  concat,
  fromStorage,
  full,
  max,
  mean,
  min,
  promoteFloat,
  sliceAxis,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'

/** numpy.pad modes. */
export type PadMode = 'constant' | 'edge' | 'reflect' | 'symmetric' | 'wrap' | 'maximum' | 'minimum' | 'mean'

/** scipy.ndimage border modes. */
export type BorderMode =
  | 'reflect'
  | 'mirror'
  | 'nearest'
  | 'wrap'
  | 'constant'
  | 'grid-mirror'
  | 'grid-wrap'
  | 'grid-constant'

/**
 * The numpy.pad mode that extends an array as an ndimage border mode does: ndimage's `reflect` is numpy's `symmetric`
 * (the edge sample repeats), `mirror` is numpy's `reflect` (it does not), `nearest` is `edge`.
 *
 * @param mode - The ndimage mode.
 * @returns The numpy mode.
 * @example
 * ```ts
 * import { padModeOf } from 'mdml-engine/numerics/convolution'
 * padModeOf('mirror') // => 'reflect'
 * ```
 */
export function padModeOf(mode: BorderMode): PadMode {
  switch (mode) {
    case 'reflect':
    case 'grid-mirror':
      return 'symmetric'
    case 'mirror':
      return 'reflect'
    case 'nearest':
      return 'edge'
    case 'wrap':
    case 'grid-wrap':
      return 'wrap'
    case 'constant':
    case 'grid-constant':
      return 'constant'
  }
}

/**
 * The source index of position i (possibly outside [0, n)) under a padding mode, or -1 for a constant.
 *
 * @param i - Position.
 * @param n - Axis length.
 * @param mode - The mode.
 * @returns The index into the array, or -1.
 * @example
 * ```ts
 * import { extendIndex } from 'mdml-engine/numerics/convolution'
 * [-2, -1, 4, 5].map((i) => extendIndex(i, 4, 'symmetric')) // => [1, 0, 3, 2]
 * ```
 */
export function extendIndex(i: number, n: number, mode: PadMode): number {
  if (i >= 0 && i < n) return i
  switch (mode) {
    case 'constant':
    case 'maximum':
    case 'minimum':
    case 'mean':
      return -1
    case 'edge':
      return i < 0 ? 0 : n - 1
    case 'wrap':
      return ((i % n) + n) % n
    case 'reflect': {
      if (n === 1) return 0
      const period = 2 * (n - 1)
      const j = ((i % period) + period) % period
      return j < n ? j : period - j
    }
    case 'symmetric': {
      const period = 2 * n
      const j = ((i % period) + period) % period
      return j < n ? j : period - 1 - j
    }
  }
}

/**
 * Pad an array (numpy.pad): constant (a value), edge, reflect (mirror without repeating the edge), symmetric (with),
 * wrap, or the maximum, minimum or mean of each line. Built from gathers, so differentiable.
 *
 * @param x - Input.
 * @param widths - (before, after) per axis, or one pair for every axis.
 * @param mode - The mode (default constant).
 * @param value - The constant (default 0).
 * @returns The padded array.
 * @example
 * ```ts
 * import { padArray } from 'mdml-engine/numerics/convolution'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(padArray([1, 2, 3], [[2, 2]], 'reflect')) // => [3, 2, 1, 2, 3, 2, 1]
 * ```
 */
export function padArray(
  x: TensorLike,
  widths: readonly (readonly [number, number])[],
  mode: PadMode = 'constant',
  value = 0,
): Value {
  const [v] = promoteFloat(x) as [Value]
  const rank = v.shape.length
  const all = widths.length === 1 && rank > 1 ? new Array(rank).fill(widths[0]) : widths
  if (all.length !== rank) throw new MdmlError('shape', `padArray: ${rank} axes need ${rank} width pairs`)
  let out = v
  all.forEach(([before, after], axis) => {
    if (before === 0 && after === 0) return
    const n = out.shape[axis] ?? 0
    if (n === 0) throw new MdmlError('value', 'padArray: cannot pad an empty axis')
    if (mode === 'constant' || mode === 'maximum' || mode === 'minimum' || mode === 'mean') {
      const fill = (count: number): Value => {
        if (count === 0) return sliceAxis(out, axis, 0, 0)
        const stat =
          mode === 'constant'
            ? // A concrete block, not 0 * x + value: that is NaN where x is already an infinite padding value.
              full(
                out.shape.map((s, i) => (i === axis ? 1 : s)),
                value,
                { dtype: out.dtype },
              )
            : mode === 'maximum'
              ? max(out, axis, true)
              : mode === 'minimum'
                ? min(out, axis, true)
                : mean(out, axis, true)
        return take(stat, fromStorage(new Int32Array(count), [count], 'int32'), axis)
      }
      out = concat([fill(before), out, fill(after)], axis)
      return
    }
    const idx = Int32Array.from({ length: n + before + after }, (_, k) => extendIndex(k - before, n, mode))
    out = take(out, fromStorage(idx, [idx.length], 'int32'), axis)
  })
  return out
}
