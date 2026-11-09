/**
 * Stacked areas: series stacked on a common baseline as bands, from zero (a stacked area chart), as shares of the
 * total (expanded to one), or about a centred, wiggle-minimising baseline (a streamgraph).
 */

import type { ColourMode } from '../theme/tokens.ts'
import { deferMode, isModeless } from './_mode.ts'
import { band } from './lines.ts'
import type { Layer } from './plot.ts'

/** Where the stack starts: at zero, as shares of the total, or centred to minimise wiggle (a streamgraph). */
export type StackOffset = 'zero' | 'expand' | 'wiggle'

/**
 * Series stacked into bands over a shared x: each band from the sum of the series below it to that sum plus its own
 * values. `expand` divides by the total at each x; `wiggle` uses Byron and Wattenberg's baseline, which keeps the
 * layers as level as it can.
 *
 * @param x - The shared x values.
 * @param series - The series, bottom first: names and values (non-negative).
 * @param options - The offset (default `zero`), colour mode and opacity.
 * @returns The layer of bands.
 * @example
 * ```ts
 * import { stackedAreas } from 'omai-render/charts'
 * stackedAreas([0, 1], [{ name: 'a', y: [1, 1] }, { name: 'b', y: [2, 2] }]).extent.y // => [0, 3]
 * ```
 */
export function stackedAreas(
  x: readonly number[],
  series: readonly { readonly name: string; readonly y: readonly number[] }[],
  options: { readonly offset?: StackOffset; readonly mode?: ColourMode; readonly opacity?: number } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => stackedAreas(x, series, { ...options, mode }))
  const n = x.length
  const offset = options.offset ?? 'zero'
  const total = x.map((_, j) => series.reduce((s, ser) => s + Math.max(0, ser.y[j] ?? 0), 0))
  const value = (k: number, j: number): number => {
    const v = Math.max(0, series[k]?.y[j] ?? 0)
    return offset === 'expand' ? ((total[j] ?? 0) > 0 ? v / (total[j] ?? 1) : 0) : v
  }
  // The baseline: zero, or (wiggle) g0 = -(1 / (n + 1)) sum_k (n - k + 1) f_k, Byron and Wattenberg's weighted centre.
  const m = series.length
  const base = Array.from({ length: n }, (_, j) =>
    offset === 'wiggle' ? -series.reduce((s, _ser, k) => s + (m - k) * value(k, j), 0) / (m + 1) : 0,
  )
  const lower = Array.from({ length: n }, (_, j) => base[j] ?? 0)
  const bands = series.map((ser, k) => {
    const lo = [...lower]
    for (let j = 0; j < n; j++) lower[j] = (lower[j] ?? 0) + value(k, j)
    return { x, lower: lo, upper: [...lower], name: ser.name, opacity: options.opacity ?? 0.85, edges: false }
  })
  return band(bands, { mode: options.mode ?? 'light' })
}
