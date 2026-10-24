import type { SeriesOption } from 'echarts'
import { WhiteboxError } from 'whitebox-engine/base'
import { contourLines } from 'whitebox-engine/numerics/geometry'
import type { Range } from '../figure/aspect.ts'
import { extentOf } from '../figure/limits.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { colourFor } from './_data.ts'
import { type Curve, lines } from './lines.ts'
import { type Layer, layer } from './plot.ts'

/** Styling shared by the function-plot layers. */
export interface FunctionStyle {
  readonly name?: string
  readonly colour?: string
  readonly width?: number
  readonly dash?: 'solid' | 'dashed' | 'dotted'
  readonly mode?: ColourMode
  readonly colourOffset?: number
}

function styled(curve: Curve, style: FunctionStyle): Curve {
  return {
    ...curve,
    ...(style.name === undefined ? {} : { name: style.name }),
    ...(style.colour === undefined ? {} : { colour: style.colour }),
    ...(style.width === undefined ? {} : { width: style.width }),
    ...(style.dash === undefined ? {} : { dash: style.dash }),
  }
}

function line(curves: readonly Curve[], style: FunctionStyle): Layer {
  return lines(curves, { mode: style.mode ?? 'light', colourOffset: style.colourOffset ?? 0 })
}

/** Options of {@link fn}. */
export interface FunctionOptions extends FunctionStyle {
  /** Sample points (default 400). */
  readonly n?: number
  /**
   * Break the curve at jumps larger than this multiple of the median step (asymptotes and discontinuities; default
   * 50, `Infinity` to never break).
   */
  readonly breakAt?: number
  /** Clip y values to this range (values outside become gaps; default none). */
  readonly clip?: Range
}

/**
 * A function of one variable sampled on a range, with gaps where it is not finite and at jumps (so `tan` and `1/x`
 * draw without vertical lines across their asymptotes).
 *
 * @param f - The function.
 * @param range - The x range.
 * @param options - Samples, jump threshold, y clipping and styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { fn } from 'whitebox-render/charts'
 * fn((x) => x * x, [-1, 1], { n: 3 }).series[0].data // => [[-1, 1], [0, 0], [1, 1]]
 * fn((x) => 1 / x, [-1, 1], { n: 5 }).series[0].data[2] // => [0, null]
 * ```
 */
export function fn(f: (x: number) => number, range: Range, options: FunctionOptions = {}): Layer {
  const n = Math.max(2, options.n ?? 400)
  const [a, b] = range
  const x = Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  const clip = options.clip
  const y = x.map((v) => {
    const w = f(v)
    return Number.isFinite(w) && (clip === undefined || (w >= clip[0] && w <= clip[1])) ? w : Number.NaN
  })
  const steps = y
    .slice(1)
    .map((v, i) => Math.abs(v - (y[i] ?? 0)))
    .filter(Number.isFinite)
    .sort((p, q) => p - q)
  const median = steps[Math.floor(steps.length / 2)] ?? 0
  const limit = median > 0 ? (options.breakAt ?? 50) * median : Number.POSITIVE_INFINITY
  // A jump that dwarfs the typical step (an asymptote, a discontinuity) becomes a gap.
  for (let i = 1; i < n; i++) {
    const p = y[i - 1] ?? Number.NaN
    const q = y[i] ?? Number.NaN
    if (Number.isFinite(p) && Number.isFinite(q) && Math.abs(q - p) > limit) y[i] = Number.NaN
  }
  return line([styled({ x, y }, options)], options)
}

/**
 * A parametric curve `(x(t), y(t))` sampled over a range of t.
 *
 * @param fx - x as a function of t.
 * @param fy - y as a function of t.
 * @param range - The t range.
 * @param options - Samples (default 400) and styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { parametric } from 'whitebox-render/charts'
 * parametric(Math.cos, Math.sin, [0, 2 * Math.PI], { n: 5 }).extent.x // => [-1, 1]
 * ```
 */
export function parametric(
  fx: (t: number) => number,
  fy: (t: number) => number,
  range: Range,
  options: FunctionStyle & { readonly n?: number } = {},
): Layer {
  const n = Math.max(2, options.n ?? 400)
  const t = Array.from({ length: n }, (_, i) => range[0] + ((range[1] - range[0]) * i) / (n - 1))
  return line([styled({ x: t.map(fx), y: t.map(fy) }, options)], options)
}

/**
 * A polar curve `r(θ)` drawn in cartesian coordinates.
 *
 * @param r - The radius as a function of the angle (radians).
 * @param range - The angle range (default one turn).
 * @param options - Samples (default 400) and styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { polar } from 'whitebox-render/charts'
 * polar(() => 2, [0, Math.PI], { n: 3 }).extent // => { x: [-2, 2], y: [0, 2] }
 * ```
 */
export function polar(
  r: (theta: number) => number,
  range: Range = [0, 2 * Math.PI],
  options: FunctionStyle & { readonly n?: number } = {},
): Layer {
  return parametric(
    (t) => r(t) * Math.cos(t),
    (t) => r(t) * Math.sin(t),
    range,
    options,
  )
}

/**
 * The implicit curve `f(x, y) = level` in a rectangle, by marching squares on an n by n grid.
 *
 * @param f - The function of two variables.
 * @param x - The x range.
 * @param y - The y range.
 * @param options - Grid size (default 120), level (default 0) and styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { implicit } from 'whitebox-render/charts'
 * const l = implicit((x, y) => x * x + y * y - 1, [-2, 2], [-2, 2], { n: 41 })
 * Math.round(l.extent.x[1] * 100) / 100 // => 1
 * ```
 */
export function implicit(
  f: (x: number, y: number) => number,
  x: Range,
  y: Range,
  options: FunctionStyle & { readonly n?: number; readonly level?: number } = {},
): Layer {
  const n = Math.max(3, options.n ?? 120)
  const xs = Array.from({ length: n }, (_, j) => x[0] + ((x[1] - x[0]) * j) / (n - 1))
  const ys = Array.from({ length: n }, (_, i) => y[0] + ((y[1] - y[0]) * i) / (n - 1))
  const z = ys.map((v) => xs.map((u) => f(u, v)))
  const paths = contourLines(z, options.level ?? 0, { x: xs, y: ys })
  const px: number[] = []
  const py: number[] = []
  for (const p of paths) {
    for (const q of p) {
      px.push(q[0])
      py.push(q[1])
    }
    px.push(Number.NaN)
    py.push(Number.NaN)
  }
  return line([styled({ x: px, y: py }, options)], options)
}

function slope(f: (x: number) => number, x0: number): number {
  // Five-point central difference with a step scaled to x0.
  const h = 1e-3 * Math.max(1, Math.abs(x0))
  return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)
}

/**
 * The tangent line of a function at a point, over a span around it, with the point marked.
 *
 * @param f - The function.
 * @param x0 - The point of tangency.
 * @param options - The derivative there (default a five-point finite difference), the span (default 2), styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { tangentLine } from 'whitebox-render/charts'
 * tangentLine((x) => x * x, 1, { derivative: 2, span: 2 }).series[0].data // => [[0, -1], [2, 3]]
 * ```
 */
export function tangentLine(
  f: (x: number) => number,
  x0: number,
  options: FunctionStyle & { readonly derivative?: number; readonly span?: number } = {},
): Layer {
  const m = options.derivative ?? slope(f, x0)
  const y0 = f(x0)
  const h = (options.span ?? 2) / 2
  return withPoints(
    line([styled({ x: [x0 - h, x0 + h], y: [y0 - m * h, y0 + m * h], dash: 'dashed' }, options)], options),
    [[x0, y0]],
    options,
  )
}

/**
 * The secant line of a function through two points, extended a little beyond them, with the points marked.
 *
 * @param f - The function.
 * @param a - The first x.
 * @param b - The second x.
 * @param options - Extension beyond the points as a fraction of their distance (default 0.25), styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { secantLine } from 'whitebox-render/charts'
 * secantLine((x) => x * x, 0, 2, { extend: 0 }).series[0].data // => [[0, 0], [2, 4]]
 * ```
 */
export function secantLine(
  f: (x: number) => number,
  a: number,
  b: number,
  options: FunctionStyle & { readonly extend?: number } = {},
): Layer {
  if (a === b) throw new WhiteboxError('value', 'secantLine: the two points must differ')
  const fa = f(a)
  const fb = f(b)
  const m = (fb - fa) / (b - a)
  const e = (options.extend ?? 0.25) * (b - a)
  return withPoints(
    line([styled({ x: [a - e, b + e], y: [fa - m * e, fb + m * e], dash: 'dashed' }, options)], options),
    [
      [a, fa],
      [b, fb],
    ],
    options,
  )
}

function withPoints(l: Layer, pts: readonly (readonly [number, number])[], style: FunctionStyle): Layer {
  const colour = colourFor(style.colour, style.mode ?? 'light', style.colourOffset ?? 0)
  const marks = {
    type: 'scatter',
    name: '__points',
    data: pts.map((p) => [p[0], p[1]]),
    symbolSize: 7,
    itemStyle: { color: colour },
    z: 4,
  } as SeriesOption
  return { ...l, series: [...l.series, marks] }
}

/** Riemann sum rules: left, right or midpoint rectangles, or trapezoids. */
export type RiemannRule = 'left' | 'right' | 'midpoint' | 'trapezoid'

interface PolyApi {
  coord(p: readonly number[]): number[]
}

/**
 * Riemann rectangles (or trapezoids) under a function on [a, b], with their signed total.
 *
 * @param f - The function.
 * @param range - `[a, b]`.
 * @param n - The number of pieces.
 * @param options - The rule (default midpoint), styling and opacity.
 * @returns The layer, and the sum of the pieces' signed areas.
 * @example
 * ```ts
 * import { riemann } from 'whitebox-render/charts'
 * riemann((x) => x, [0, 1], 2, { rule: 'left' }).sum // => 0.25
 * riemann((x) => x, [0, 1], 2, { rule: 'trapezoid' }).sum // => 0.5
 * ```
 */
export function riemann(
  f: (x: number) => number,
  range: Range,
  n: number,
  options: FunctionStyle & { readonly rule?: RiemannRule; readonly opacity?: number } = {},
): Layer & { readonly sum: number } {
  if (!(Number.isInteger(n) && n >= 1)) throw new WhiteboxError('value', 'riemann: n must be a positive integer')
  const rule = options.rule ?? 'midpoint'
  const [a, b] = range
  const w = (b - a) / n
  const pieces: [number, number][][] = []
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x0 = a + i * w
    const x1 = x0 + w
    if (rule === 'trapezoid') {
      const y0 = f(x0)
      const y1 = f(x1)
      sum += ((y0 + y1) / 2) * w
      pieces.push([
        [x0, 0],
        [x0, y0],
        [x1, y1],
        [x1, 0],
      ])
    } else {
      const h = f(rule === 'left' ? x0 : rule === 'right' ? x1 : (x0 + x1) / 2)
      sum += h * w
      pieces.push([
        [x0, 0],
        [x0, h],
        [x1, h],
        [x1, 0],
      ])
    }
  }
  const colour = colourFor(options.colour, options.mode ?? 'light', options.colourOffset ?? 0)
  const series = {
    type: 'custom',
    name: options.name ?? '__riemann',
    data: pieces.map((_, i) => [i]),
    renderItem: (params: { dataIndex: number }, api: PolyApi): Record<string, unknown> => ({
      type: 'polygon',
      shape: { points: (pieces[params.dataIndex] ?? []).map((p) => api.coord(p)) },
      style: { fill: colour, opacity: options.opacity ?? 0.3, stroke: colour, lineWidth: 1 },
    }),
    tooltip: { show: false },
    z: 1,
  } as unknown as SeriesOption
  const ys = pieces.flat().map((p) => p[1])
  return { ...layer([series], { x: [a, b], y: extentOf(ys) ?? [0, 1], baseline: 'y' }), sum }
}

/**
 * The Taylor polynomial of a function about a point from its derivatives there:
 * sum over k of f^(k)(x0) (x - x0)^k / k!.
 *
 * @param derivatives - f(x0), f'(x0), f''(x0), ... (the polynomial's order is one less than their number).
 * @param x0 - The expansion point.
 * @param range - The x range to draw.
 * @param options - Samples (default 200) and styling.
 * @returns The layer.
 * @example
 * ```ts
 * import { taylor } from 'whitebox-render/charts'
 * taylor([1, 1, 1], 0, [0, 1], { n: 2 }).series[0].data // => [[0, 1], [1, 2.5]]
 * ```
 */
export function taylor(
  derivatives: readonly number[],
  x0: number,
  range: Range,
  options: FunctionStyle & { readonly n?: number } = {},
): Layer {
  const poly = (x: number): number => {
    let s = 0
    let term = 1
    derivatives.forEach((d, k) => {
      if (k > 0) term *= (x - x0) / k
      s += d * term
    })
    return s
  }
  return fn(poly, range, {
    breakAt: Number.POSITIVE_INFINITY,
    ...options,
    ...(options.n === undefined ? {} : { n: options.n }),
  })
}

/**
 * Reference lines across the whole plot: vertical lines at x values and horizontal lines at y values (asymptotes,
 * thresholds, targets), with optional labels.
 *
 * @param lines - `x` and `y` values, each a number or `{ value, label }`.
 * @param options - Colour (default `muted`), dash (default dashed) and colour mode.
 * @returns The layer (it does not affect the axis ranges).
 * @example
 * ```ts
 * import { referenceLines } from 'whitebox-render/charts'
 * referenceLines({ x: [0], y: [{ value: 1, label: 'target' }] }).series[0].markLine.data.length // => 2
 * ```
 */
export function referenceLines(
  lines: {
    readonly x?: readonly (number | { readonly value: number; readonly label?: string })[]
    readonly y?: readonly (number | { readonly value: number; readonly label?: string })[]
  },
  options: { readonly colour?: string; readonly dash?: 'solid' | 'dashed' | 'dotted'; readonly mode?: ColourMode } = {},
): Layer {
  const colour = colourFor(options.colour ?? 'muted', options.mode ?? 'light')
  const item = (axis: 'xAxis' | 'yAxis', v: number | { readonly value: number; readonly label?: string }) => {
    const value = typeof v === 'number' ? v : v.value
    const label = typeof v === 'number' ? undefined : v.label
    return {
      [axis]: value,
      label: label === undefined ? { show: false } : { show: true, formatter: label, color: colour, fontSize: 10 },
    }
  }
  const series = {
    type: 'line',
    name: '__reference',
    data: [],
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: { color: colour, type: options.dash ?? 'dashed', width: 1 },
      data: [...(lines.x ?? []).map((v) => item('xAxis', v)), ...(lines.y ?? []).map((v) => item('yAxis', v))],
    },
  } as SeriesOption
  return layer([series], {})
}
