/**
 * Number formatting shared by axes, tooltips, readouts, tables and live equations.
 */

const MINUS = '−'
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

/** Options of {@link formatNumber}. */
export interface FormatOptions {
  /** Significant digits (default 4). */
  readonly digits?: number
  /** Use the typographic minus sign (default false: ASCII hyphen-minus). */
  readonly unicodeMinus?: boolean
  /** Magnitudes below this (but not zero) use exponent form (default 1e-3). */
  readonly smallExponent?: number
  /** Magnitudes at or above this use exponent form (default 1e5). */
  readonly largeExponent?: number
}

/**
 * Format a number for display: a few significant digits with trailing zeros dropped, and exponent form for very small
 * or large magnitudes, so axes read 0.5 rather than 0.49999999.
 *
 * @param value - The number.
 * @param options - Significant digits, minus sign and exponent thresholds.
 * @returns The text (`NaN`, `∞` and `−∞` for non-finite values).
 * @index function Format a number for axes and readouts: `formatNumber(Math.PI, { digits: 2 })`.
 * @example
 * ```ts
 * import { formatNumber } from 'mdml-render/theme'
 * formatNumber(0.49999999) // => '0.5'
 * formatNumber(123456) // => '1.23e5'
 * formatNumber(-0.00012) // => '-1.2e-4'
 * formatNumber(Math.PI, { digits: 2 }) // => '3.1'
 * ```
 */
export function formatNumber(value: number, options: FormatOptions = {}): string {
  const { digits = 4, unicodeMinus = false, smallExponent = 1e-3, largeExponent = 1e5 } = options
  const minus = unicodeMinus ? MINUS : '-'
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : `${MINUS}∞`
  const a = Math.abs(value)
  let text: string
  if (a !== 0 && (a < smallExponent || a >= largeExponent)) {
    const [mantissa = '0', exponent = '0'] = value.toExponential(Math.max(0, Math.min(digits, 3) - 1)).split('e')
    text = `${mantissa.replace(/\.?0+$/, '')}e${Number(exponent)}`
  } else {
    text = String(Number(value.toPrecision(Math.max(1, digits))))
    if (text === '-0') text = '0'
  }
  return text.replace(/-/g, minus)
}

/**
 * Format a tick on a log axis: `10⁻³` rather than 0.001 at powers of ten, {@link formatNumber} between them.
 *
 * @param value - A positive number.
 * @returns The label.
 * @example
 * ```ts
 * import { formatPower } from 'mdml-render/theme'
 * formatPower(0.001) // => '10⁻³'
 * formatPower(1) // => '1'
 * formatPower(30) // => '30'
 * ```
 */
export function formatPower(value: number): string {
  if (!(value > 0)) return formatNumber(value)
  const e = Math.round(Math.log10(value))
  if (!(Math.abs(value - 10 ** e) <= 1e-9 * 10 ** e)) return formatNumber(value)
  if (e === 0) return '1'
  if (e === 1) return '10'
  const digits = [...String(Math.abs(e))].map((d) => SUPERSCRIPT[Number(d)]).join('')
  return `10${e < 0 ? '⁻' : ''}${digits}`
}

/**
 * Format a fraction as a percentage.
 *
 * @param value - The fraction (0.25 is 25%).
 * @param digits - Decimal places (default 1).
 * @returns The text.
 * @example
 * ```ts
 * import { formatPercent } from 'mdml-render/theme'
 * formatPercent(0.256) // => '25.6%'
 * ```
 */
export function formatPercent(value: number, digits = 1): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : formatNumber(value)
}

/**
 * Format a duration in seconds: `850 ms`, `12.5 s`, `3 min 20 s`, `2 h 5 min`.
 *
 * @param seconds - The duration.
 * @returns The text.
 * @example
 * ```ts
 * import { formatDuration } from 'mdml-render/theme'
 * formatDuration(0.85) // => '850 ms'
 * formatDuration(200) // => '3 min 20 s'
 * ```
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return formatNumber(seconds)
  const s = Math.abs(seconds)
  const sign = seconds < 0 ? '-' : ''
  if (s < 1) return `${sign}${formatNumber(s * 1000, { digits: 3 })} ms`
  if (s < 60) return `${sign}${formatNumber(s, { digits: 3 })} s`
  if (s < 3600) {
    const m = Math.floor(s / 60)
    const r = Math.round(s - 60 * m)
    return `${sign}${m} min${r > 0 ? ` ${r} s` : ''}`
  }
  const h = Math.floor(s / 3600)
  const m = Math.round((s - 3600 * h) / 60)
  return `${sign}${h} h${m > 0 ? ` ${m} min` : ''}`
}

/**
 * Format a timestamp (milliseconds since the epoch, UTC) at a resolution suited to a span: dates for long spans, times
 * for short ones.
 *
 * @param ms - The time in milliseconds since 1970-01-01 UTC.
 * @param span - The span of the axis in milliseconds (default one day), which picks the resolution.
 * @returns The text.
 * @example
 * ```ts
 * import { formatTime } from 'mdml-render/theme'
 * formatTime(Date.UTC(2024, 2, 5, 14, 30)) // => '14:30'
 * formatTime(Date.UTC(2024, 2, 5), 90 * 86400000) // => '2024-03-05'
 * ```
 */
export function formatTime(ms: number, span = 86_400_000): string {
  const d = new Date(ms)
  const p = (n: number): string => String(n).padStart(2, '0')
  const date = `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`
  if (span >= 3 * 86_400_000) return date
  if (span >= 120_000) return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`
  return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`
}

/**
 * A nice number near `x` (1, 2, 5 or 10 times a power of ten), rounded or taken as the next one up.
 *
 * @param x - A positive number.
 * @param round - Round to the nearest nice number (default) or take the least nice number at least `x`.
 * @returns The nice number.
 * @example
 * ```ts
 * import { niceNumber } from 'mdml-render/theme'
 * niceNumber(0.37) // => 0.5
 * niceNumber(13, false) // => 20
 * ```
 */
export function niceNumber(x: number, round = true): number {
  if (!(x > 0) || !Number.isFinite(x)) return 1
  const e = Math.floor(Math.log10(x))
  const f = x / 10 ** e
  const nice = round ? (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) : f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10
  return Number((nice * 10 ** e).toPrecision(12))
}

/** Ticks on a linear axis: positions, their spacing and the range they cover. */
export interface Ticks {
  readonly ticks: readonly number[]
  readonly step: number
  readonly range: readonly [number, number]
}

/**
 * Nice ticks for a linear axis (Heckbert's labelling): about `count` ticks at a 1, 2 or 5 spacing, with the range
 * widened to whole ticks.
 *
 * @param lo - The low end of the data.
 * @param hi - The high end.
 * @param count - About how many ticks (default 5).
 * @returns The ticks.
 * @example
 * ```ts
 * import { niceTicks } from 'mdml-render/theme'
 * niceTicks(0.13, 0.92).ticks // => [0, 0.2, 0.4, 0.6, 0.8, 1]
 * niceTicks(-3, 7, 3).step // => 5
 * ```
 */
export function niceTicks(lo: number, hi: number, count = 5): Ticks {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { ticks: [], step: 1, range: [0, 1] }
  let [a, b] = lo <= hi ? [lo, hi] : [hi, lo]
  if (a === b) {
    const d = a === 0 ? 1 : Math.abs(a) * 0.1
    a -= d
    b += d
  }
  const step = niceNumber(niceNumber(b - a, false) / Math.max(1, count - 1), true)
  const start = Math.floor(a / step + 1e-9) * step
  const end = Math.ceil(b / step - 1e-9) * step
  const ticks: number[] = []
  for (let k = 0; start + k * step <= end + step * 1e-9; k++) ticks.push(Number((start + k * step).toPrecision(12)))
  return { ticks, step, range: [ticks[0] ?? a, ticks[ticks.length - 1] ?? b] }
}

/**
 * Ticks for a log axis: powers of ten across the range, thinned to about `count`, and the minor 2 to 9 multiples when
 * the range spans few decades.
 *
 * @param lo - The low end (positive).
 * @param hi - The high end.
 * @param count - About how many major ticks (default 6).
 * @returns The major ticks and the minor ticks.
 * @example
 * ```ts
 * import { logTicks } from 'mdml-render/theme'
 * logTicks(0.002, 300).major // => [0.001, 0.01, 0.1, 1, 10, 100, 1000]
 * logTicks(1, 10).minor.length // => 8
 * ```
 */
export function logTicks(
  lo: number,
  hi: number,
  count = 6,
): { readonly major: readonly number[]; readonly minor: readonly number[] } {
  if (!(lo > 0 && hi > 0)) return { major: [], minor: [] }
  const [a, b] = lo <= hi ? [lo, hi] : [hi, lo]
  const e0 = Math.floor(Math.log10(a) + 1e-9)
  const e1 = Math.ceil(Math.log10(b) - 1e-9)
  const decades = Math.max(1, e1 - e0)
  const stride = Math.max(1, Math.ceil(decades / Math.max(1, count)))
  const major: number[] = []
  for (let e = e0; e <= e1; e += stride) major.push(Number((10 ** e).toPrecision(12)))
  const minor: number[] = []
  if (stride === 1 && decades <= 4)
    for (let e = e0; e < e1; e++) for (let m = 2; m <= 9; m++) minor.push(Number((m * 10 ** e).toPrecision(12)))
  return { major, minor }
}

/**
 * A nice slider step for a range: 1, 2 or 5 times a power of ten giving about `positions` positions.
 *
 * @param lo - The low end.
 * @param hi - The high end.
 * @param positions - About how many positions (default 250).
 * @returns The step.
 * @example
 * ```ts
 * import { niceStep } from 'mdml-render/theme'
 * niceStep(0, 1) // => 0.005
 * ```
 */
export function niceStep(lo: number, hi: number, positions = 250): number {
  const raw = (hi - lo) / positions
  if (!(raw > 0) || !Number.isFinite(raw)) return 1
  const p = 10 ** Math.floor(Math.log10(raw))
  return Number(([1, 2, 5, 10].map((m) => m * p).find((c) => c >= raw * (1 - 1e-9)) ?? 10 * p).toPrecision(12))
}

/**
 * Decimal places needed to write multiples of a step exactly.
 *
 * @param step - The step.
 * @returns The number of decimals.
 * @example
 * ```ts
 * import { stepDecimals } from 'mdml-render/theme'
 * stepDecimals(0.05) // => 2
 * stepDecimals(1e-4) // => 4
 * stepDecimals(10) // => 0
 * ```
 */
export function stepDecimals(step: number): number {
  if (!(step > 0)) return 0
  const text = String(step)
  const exp = /e-(\d+)$/.exec(text)
  if (exp !== null) return Number(exp[1]) + (text.split('e')[0]?.split('.')[1]?.length ?? 0)
  return text.split('.')[1]?.length ?? 0
}
