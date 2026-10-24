/**
 * The rules of a typed number, as pure functions shared by `NumberField`, `SeedInput` and `SpaceControls`: parsing and
 * validating what is typed, stepping (buttons and arrow keys), the slider's mapping between value and position, the
 * ladder of 11 values between the bounds that the dropdown offers, and display.
 */

/** A number's type, bounds and scale. */
export interface NumberSpec {
  /** The smallest value (inclusive). */
  readonly min?: number
  /** The largest value (inclusive). */
  readonly max?: number
  /** Whole numbers only. */
  readonly integer?: boolean
  /** A log scale: stepping, sliding and the ladder move by factors (needs a positive range). */
  readonly log?: boolean
  /**
   * The step of the buttons and arrow keys on a linear scale (Shift moves ten). Defaults to 1 for integers and to a
   * round hundredth of the range otherwise. On a log scale the buttons step along 1, 2, 5, 10, ... (Shift: a decade).
   */
  readonly step?: number
}

/** A typed draft's value, or why it cannot be used. */
export type NumberCheck = { readonly value: number; readonly error?: undefined } | { readonly error: string }

const clean = (x: number): number => Number(x.toPrecision(12))

/** Whether both bounds are finite (and, on a log scale, the range positive): the number has a range to slide over. */
export function hasRange(spec: NumberSpec): boolean {
  const { min, max } = spec
  return (
    min !== undefined &&
    max !== undefined &&
    Number.isFinite(min) &&
    Number.isFinite(max) &&
    max > min &&
    (spec.log !== true || min > 0)
  )
}

/**
 * Parse a typed number: decimal or exponent notation with an optional sign (a typographic minus too).
 *
 * @param text - What was typed.
 * @returns The number, or `null` when the text is not one.
 */
export function parseNumber(text: string): number | null {
  const t = text.trim().replace(/^[−–]/, '-')
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return null
  const x = Number(t)
  return Number.isFinite(x) ? x : null
}

/**
 * Why a value is not allowed ("must be ≥ 0", "must be a whole number"), or `null` when it is.
 *
 * @param spec - The number's rules.
 * @param x - The value.
 * @returns A short message, or `null`.
 */
export function validateNumber(spec: NumberSpec, x: number): string | null {
  if (!Number.isFinite(x)) return 'not a number'
  if (spec.integer === true && !Number.isInteger(x)) return 'must be a whole number'
  if (spec.log === true && !(x > 0)) return 'must be > 0'
  if (spec.min !== undefined && x < spec.min) return `must be ≥ ${formatFieldValue(spec, spec.min)}`
  if (spec.max !== undefined && x > spec.max) return `must be ≤ ${formatFieldValue(spec, spec.max)}`
  return null
}

/**
 * Check a typed draft: parse it, then validate it (typing is validated, never silently clamped).
 *
 * @param spec - The number's rules.
 * @param text - The draft.
 * @returns The value, or the reason it is refused.
 */
export function checkNumber(spec: NumberSpec, text: string): NumberCheck {
  const x = parseNumber(text)
  if (x === null) return { error: 'not a number' }
  const error = validateNumber(spec, x)
  return error === null ? { value: x } : { error }
}

/**
 * Bring a value inside the rules: rounded for integers, then clamped to the bounds (what buttons, sliders and
 * programmatic changes do).
 *
 * @param spec - The number's rules.
 * @param x - The value.
 * @returns The nearest allowed value.
 */
export function clampNumber(spec: NumberSpec, x: number): number {
  const y = spec.integer === true ? Math.round(x) : x
  return Math.min(Math.max(y, spec.min ?? Number.NEGATIVE_INFINITY), spec.max ?? Number.POSITIVE_INFINITY)
}

/** The round number (1, 2 or 5 times a power of ten) nearest to `x` from below. */
function roundStep(x: number): number {
  const k = Math.floor(Math.log10(x))
  const m = x / 10 ** k
  return clean((m >= 5 ? 5 : m >= 2 ? 2 : 1) * 10 ** k)
}

/**
 * The linear step of the buttons and arrow keys: the stated step, 1 for integers, else a round hundredth of the range
 * (0.01 on [0, 1]), else 0.1.
 *
 * @param spec - The number's rules.
 * @returns The step.
 */
export function numberStep(spec: NumberSpec): number {
  if (spec.step !== undefined && spec.step > 0) return spec.step
  if (spec.integer === true) return 1
  if (hasRange(spec)) return roundStep(((spec.max as number) - (spec.min as number)) / 100)
  return 0.1
}

/** The 1, 2, 5 grid on a log scale around `x`, ascending. */
function logGrid(x: number): number[] {
  const k = Math.floor(Math.log10(x))
  const out: number[] = []
  for (let j = k - 1; j <= k + 1; j++) for (const m of [1, 2, 5]) out.push(clean(m * 10 ** j))
  return out
}

/**
 * The value one step from `x`: linear steps of {@link numberStep} (ten with `big`), or on a log scale the next value of
 * the 1, 2, 5 grid (a decade with `big`); then clamped. `null` when there is no move (at a bound).
 *
 * @param spec - The number's rules.
 * @param x - The current value.
 * @param dir - Up (`1`) or down (`-1`).
 * @param big - A big step (Shift).
 * @returns The next value, or `null`.
 */
export function stepNumber(spec: NumberSpec, x: number, dir: 1 | -1, big = false): number | null {
  let y: number
  if (spec.log === true && x > 0) {
    if (big) y = clean(dir > 0 ? x * 10 : x / 10)
    else {
      const grid = logGrid(x)
      const found = dir > 0 ? grid.find((g) => g > x * (1 + 1e-9)) : grid.reverse().find((g) => g < x * (1 - 1e-9))
      y = found ?? clean(dir > 0 ? x * 10 : x / 10)
    }
    if (spec.integer === true) {
      const r = Math.round(y)
      y = r === x ? x + dir : r
    }
  } else {
    y = clean(x + dir * numberStep(spec) * (big ? 10 : 1))
  }
  const next = clampNumber(spec, y)
  return next === x ? null : next
}

/**
 * Where a value sits along the slider, from 0 at `min` to 1 at `max` (clamped), on the number's scale.
 *
 * @param spec - The number's rules (with a range).
 * @param x - The value.
 * @returns The position in [0, 1] (0 without a range).
 */
export function toFraction(spec: NumberSpec, x: number): number {
  if (!hasRange(spec)) return 0
  const lo = spec.min as number
  const hi = spec.max as number
  const t =
    spec.log === true
      ? (Math.log(Math.max(x, Number.MIN_VALUE)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
      : (x - lo) / (hi - lo)
  return Number.isFinite(t) ? Math.min(Math.max(t, 0), 1) : 0
}

/**
 * The value at slider position `t`, rounded to something worth reading: the step grid on a linear scale (whole numbers
 * for integers), two significant figures on a log scale; the ends give the bounds exactly.
 *
 * @param spec - The number's rules (with a range).
 * @param t - The position in [0, 1].
 * @returns The value (`NaN` without a range).
 */
export function fromFraction(spec: NumberSpec, t: number): number {
  if (!hasRange(spec)) return Number.NaN
  const lo = spec.min as number
  const hi = spec.max as number
  const u = Math.min(Math.max(t, 0), 1)
  if (u === 0) return lo
  if (u === 1) return hi
  if (spec.log === true) {
    const x = Number(Math.exp(Math.log(lo) + u * (Math.log(hi) - Math.log(lo))).toPrecision(2))
    return clampNumber(spec, x)
  }
  const step = numberStep(spec)
  return clampNumber(spec, clean(lo + Math.round((u * (hi - lo)) / step) * step))
}

/**
 * The values the dropdown offers: `count` (default 11) evenly spaced between the bounds, geometrically on a log scale,
 * the ends exact and the rest rounded to readable values (whole numbers for integers, without repeats). Empty when the
 * number has no range.
 *
 * @param spec - The number's rules.
 * @param count - How many values (at least 2).
 * @returns The ladder, ascending.
 */
export function numberLadder(spec: NumberSpec, count = 11): number[] {
  if (!hasRange(spec) || count < 2) return []
  const lo = spec.min as number
  const hi = spec.max as number
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1)
    let x: number
    if (i === 0) x = lo
    else if (i === count - 1) x = hi
    else if (spec.log === true) x = Number(Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo))).toPrecision(2))
    else x = clean(lo + t * (hi - lo))
    if (spec.integer === true) x = Math.round(x)
    x = clampNumber(spec, x)
    if (out[out.length - 1] !== x) out.push(x)
  }
  return out
}

/**
 * A value as the field shows it: integers in full; otherwise up to four significant figures, in compact exponent form
 * (`1e-4`, `2.5e5`) when very small or large.
 *
 * @param spec - The number's rules.
 * @param x - The value.
 * @returns The text.
 */
export function formatFieldValue(spec: NumberSpec, x: number): string {
  if (!Number.isFinite(x)) return String(x)
  if (spec.integer === true) return String(Math.round(x))
  const a = Math.abs(x)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m = '', e = ''] = x.toExponential(3).split('e')
    return `${String(Number(m))}e${e.replace('+', '')}`
  }
  return String(Number(x.toPrecision(4)))
}
