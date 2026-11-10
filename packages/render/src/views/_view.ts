import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import type { Layer } from '../charts/plot.ts'
import { type Figure, type FigureOptions, figure as makeFigure } from '../figure/figure.ts'
import { type GridSpec, gridSpec } from '../figure/grid.ts'
import { formatNumber } from '../theme/format.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'

/** Options every view takes: the colour mode and the figure it fills. */
export interface ViewOptions {
  readonly mode?: ColourMode
  /** The figure (default the view's own size and aspect). */
  readonly figure?: Figure
}

/**
 * A view that plays: a number of frames, a label per frame (for the player's stamp), and the chart option of each.
 * Steps of an iterative method (k-means iterations, EM steps, an embedding's optimisation) are views of this kind.
 */
export interface ViewFrames {
  readonly count: number
  readonly label: (frame: number) => string
  readonly option: (frame: number) => EChartsOption
}

/** The figure of a view: the one given, else the view's default size, in the given mode. */
export function viewFigure(options: ViewOptions, defaults: FigureOptions): Figure {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  return options.figure === undefined ? makeFigure({ ...defaults, mode }) : { ...options.figure, mode }
}

/** The colour mode of a view's options. */
export function viewMode(options: ViewOptions): ColourMode {
  return options.mode ?? options.figure?.mode ?? 'light'
}

/** Whether values are best shown on a log axis: all positive and spanning more than a decade and a half. */
export function wantsLog(values: readonly number[]): boolean {
  if (values.length < 2 || !values.every((v) => v > 0 && Number.isFinite(v))) return false
  return maxOf(values) / minOf(values) > 30
}

/** The low and high of finite values (`[0, 1]` when there are none). */
export function spanOf(values: Iterable<number>): [number, number] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of values)
    if (Number.isFinite(v)) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  return lo <= hi ? [lo, hi] : [0, 1]
}

/** Axis limits for values: whole decades on a log axis (as matplotlib pads one), the data's span otherwise. */
export function limitsOf(values: readonly number[], log: boolean): [number, number] {
  const [lo, hi] = spanOf(values)
  if (!log) return [lo, hi]
  return [10 ** Math.floor(Math.log10(lo) + 1e-9), 10 ** Math.ceil(Math.log10(hi) - 1e-9)]
}

/** Short number text for labels and readouts (3 significant digits). */
export function short(value: number): string {
  return formatNumber(value, { digits: 3 })
}

/** The index of the value nearest a target (on a log scale when `log`). */
export function nearestIndex(values: readonly number[], target: number, log = false): number {
  let best = 0
  let gap = Number.POSITIVE_INFINITY
  const f = (v: number): number => (log ? Math.log(Math.max(v, 1e-300)) : v)
  for (const [i, v] of values.entries()) {
    const d = Math.abs(f(v) - f(target))
    if (d < gap) {
      gap = d
      best = i
    }
  }
  return best
}

/**
 * A layer's markers outlined in the foreground colour, so a point never melts into a region or a cluster of its own
 * colour (training rows over decision regions, centres over their clusters).
 */
export function outlined(l: Layer, mode: ColourMode, width = 0.9): Layer {
  const fg = tokens(mode).fg
  return {
    ...l,
    series: l.series.map((s) => {
      const style = (s as { itemStyle?: Record<string, unknown> }).itemStyle ?? {}
      return { ...s, itemStyle: { ...style, borderColor: fg, borderWidth: width } } as SeriesOption
    }),
  }
}

/**
 * The option fragment that gives a chart its description for assistive technology: ECharts' `aria` component with a
 * stated label, so a figure is announced as what it shows and not only as an image.
 *
 * @param description - What the figure shows, in a sentence.
 * @returns The fragment to spread into the option.
 * @example
 * ```ts
 * import { ariaOf } from 'omai-render/views'
 * ariaOf('A vector field').aria.label.description // => 'A vector field'
 * ```
 */
export function ariaOf(description: string): {
  readonly aria: { readonly enabled: true; readonly label: { readonly description: string } }
} {
  return { aria: { enabled: true, label: { description } } }
}

/**
 * A grid of `n` equal panels in `cols` columns, filled row by row (ids `q0`, `q1`, ...): the layout of the small
 * multiples of the chain and calibration views.
 *
 * @param n - The number of panels.
 * @param cols - The columns.
 * @param hspace - The vertical gap between rows in pixels (default 40; more where each panel carries an x label).
 * @returns The grid.
 * @example
 * ```ts
 * import { panelGrid } from 'omai-render/views'
 * panelGrid(5, 2).cells.map((c) => c.id) // => ['q0', 'q1', 'q2', 'q3', 'q4']
 * ```
 */
export function panelGrid(n: number, cols: number, hspace = 40): GridSpec {
  const rows = Math.ceil(n / cols)
  return gridSpec(rows, cols, {
    hspace,
    wspace: 52,
    cells: Array.from({ length: n }, (_, i) => ({ id: `q${i}`, row: Math.floor(i / cols), col: i % cols })),
  })
}

/** The columns of small multiples for `n` panels: two from four panels on, else one. */
export function panelColumns(n: number): number {
  return n > 3 ? 2 : 1
}

/**
 * The grid of a pair plot of `k` quantities: the lower triangle with the diagonal (ids `p{row}_{col}`), or all `k` by
 * `k` cells with `full`, for views that use the upper triangle too (`pairView`, `posteriorPairsView`).
 *
 * @param k - The number of quantities.
 * @param full - Whether to include the upper triangle (default false).
 * @returns The grid.
 * @example
 * ```ts
 * import { pairGrid } from 'omai-render/views'
 * [pairGrid(3).cells.length, pairGrid(3, true).cells.length] // => [6, 9]
 * ```
 */
export function pairGrid(k: number, full = false): GridSpec {
  return gridSpec(k, k, {
    hspace: full ? 6 : 12,
    wspace: full ? 6 : 12,
    cells: Array.from({ length: k * k }, (_, i) => ({
      id: `p${Math.floor(i / k)}_${i % k}`,
      row: Math.floor(i / k),
      col: i % k,
    })).filter((c) => full || c.col <= c.row),
  })
}

/**
 * The axis labels of a panel of a pair plot: the quantity's name under the bottom row and beside the first column
 * (below its first row, whose panel is a density).
 *
 * @param names - The names of the quantities.
 * @param row - The panel's row.
 * @param col - The panel's column.
 * @returns The panel's axis options (empty inside the plot).
 * @example
 * ```ts
 * import { pairEdge } from 'omai-render/views'
 * [pairEdge(['a', 'b'], 1, 0), pairEdge(['a', 'b'], 0, 0)] // => [{ x: { label: 'a' }, y: { label: 'b' } }, {}]
 * ```
 */
export function pairEdge(
  names: readonly string[],
  row: number,
  col: number,
): { readonly x?: { readonly label: string }; readonly y?: { readonly label: string } } {
  return {
    ...(row === names.length - 1 ? { x: { label: names[col] ?? '' } } : {}),
    ...(col === 0 && row > 0 ? { y: { label: names[row] ?? '' } } : {}),
  }
}

/**
 * At most `cap` of the values, spread evenly over them, the first and the last included: a scatter of thousands is a blob.
 *
 * @param values - The values.
 * @param cap - The most to keep.
 * @returns The thinned values (all of them when there are no more than `cap`).
 * @example
 * ```ts
 * import { thinEvenly } from 'omai-render/views'
 * [thinEvenly([1, 2, 3, 4, 5, 6, 7], 3), thinEvenly([1, 2], 3)] // => [[1, 4, 7], [1, 2]]
 * ```
 */
export function thinEvenly(values: readonly number[], cap: number): number[] {
  return values.length <= cap
    ? [...values]
    : Array.from({ length: cap }, (_, i) => values[Math.floor((i * (values.length - 1)) / (cap - 1))] as number)
}
