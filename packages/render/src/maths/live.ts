import { formatNumber } from '../theme/format.ts'

/** A live value in an equation: highlighted, formatted to `digits` significant digits. */
export interface LiveValue {
  readonly value: number | string
  readonly digits?: number
  /** Stronger highlighting (the value being dragged, say). */
  readonly strong?: boolean
}

/** A slot of a live equation: a number, raw TeX, or a {@link LiveValue}. */
export type Slot = number | string | LiveValue

/** TeX with live values in its slots, from the {@link tex} tag. */
export interface EquationTemplate {
  readonly strings: readonly string[]
  readonly slots: readonly Slot[]
}

/**
 * The `tex` tag: TeX with `${...}` slots for live values, kept raw (backslashes need no escaping).
 *
 * @param strings - The template strings.
 * @param slots - The slot values.
 * @returns The template.
 * @example
 * ```ts
 * import { tex, toTex } from 'whitebox-render/maths'
 * const mu = 1.5
 * toTex(tex`\mu = ${mu}`) // => '\\mu = \\htmlClass{wb-slot}{1.5}'
 * ```
 */
export function tex(strings: TemplateStringsArray, ...slots: Slot[]): EquationTemplate {
  return { strings: strings.raw, slots }
}

/**
 * A live value with options, for a slot: `${live(p, { digits: 3, strong: true })}`.
 *
 * @param value - The value.
 * @param options - Digits and emphasis.
 * @returns The live value.
 * @example
 * ```ts
 * import { live } from 'whitebox-render/maths'
 * live(0.123456, { digits: 2 }).digits // => 2
 * ```
 */
export function live(value: number | string, options: Omit<LiveValue, 'value'> = {}): LiveValue {
  return { value, ...options }
}

/**
 * A number as TeX: significant digits, `\times 10^{k}` for exponents, `\infty` and `\text{NaN}`.
 *
 * @param value - The number.
 * @param digits - Significant digits (default the library's number format).
 * @returns The TeX.
 * @example
 * ```ts
 * import { numberTex } from 'whitebox-render/maths'
 * numberTex(0.00012) // => '1.2 \\times 10^{-4}'
 * numberTex(-Infinity) // => '-\\infty'
 * ```
 */
export function numberTex(value: number, digits?: number): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? '\\text{NaN}' : value > 0 ? '\\infty' : '-\\infty'
  const text = digits === undefined ? formatNumber(value) : formatNumber(value, { digits })
  const m = /^(-?[\d.]+)e(-?\d+)$/.exec(text)
  return m === null ? text : `${m[1]} \\times 10^{${m[2]}}`
}

/**
 * A live equation as one TeX string, each slot wrapped in `\htmlClass{wb-slot}` (or `wb-slot wb-slot-strong`) so a
 * stylesheet can highlight the values as controls move them.
 *
 * @param t - The template, or plain TeX.
 * @returns The TeX.
 * @example
 * ```ts
 * import { live, tex, toTex } from 'whitebox-render/maths'
 * toTex(tex`\sigma^2 = ${live(0.25, { strong: true })}`) // => '\\sigma^2 = \\htmlClass{wb-slot wb-slot-strong}{0.25}'
 * toTex('x') // => 'x'
 * ```
 */
export function toTex(t: EquationTemplate | string): string {
  if (typeof t === 'string') return t
  let out = t.strings[0] ?? ''
  t.slots.forEach((slot, i) => {
    const s: LiveValue = typeof slot === 'object' ? slot : { value: slot }
    const body = typeof s.value === 'number' ? numberTex(s.value, s.digits) : s.value
    out += `\\htmlClass{wb-slot${s.strong === true ? ' wb-slot-strong' : ''}}{${body}}${t.strings[i + 1] ?? ''}`
  })
  return out
}

/**
 * Substitute values for named parameters in TeX: each `{{name}}` becomes the value's TeX, highlighted as a slot. For
 * equations stored as data (in a catalogue, say) rather than written with the {@link tex} tag.
 *
 * @param template - TeX with `{{name}}` placeholders.
 * @param values - Values by name (a missing name stays as `name`).
 * @param options - Digits for numbers.
 * @returns The TeX.
 * @example
 * ```ts
 * import { bindTex } from 'whitebox-render/maths'
 * bindTex('y = {{a}} x + {{b}}', { a: 2, b: -1 }) // => 'y = \\htmlClass{wb-slot}{2} x + \\htmlClass{wb-slot}{-1}'
 * ```
 */
export function bindTex(
  template: string,
  values: Readonly<Record<string, number | string>>,
  options: { readonly digits?: number } = {},
): string {
  return template.replace(/\{\{\s*([A-Za-z_][\w]*)\s*\}\}/g, (_, name: string) => {
    const v = values[name]
    if (v === undefined) return name
    const body = typeof v === 'number' ? numberTex(v, options.digits) : v
    return `\\htmlClass{wb-slot}{${body}}`
  })
}
