/**
 * Options of choice controls: bare values or labelled options normalised to labelled ones, and cycling to the previous
 * or next enabled option with wrap-around.
 */

/** A value a choice can take. */
export type OptionValue = string | number | boolean

/** An option as given: a bare value, or a value with a label (and an optional note). */
export type OptionInput<T extends OptionValue> =
  | T
  | { readonly value: T; readonly label?: string; readonly description?: string; readonly disabled?: boolean }

/** An option with its label filled in. */
export interface ChoiceOption<T extends OptionValue> {
  readonly value: T
  readonly label: string
  readonly description?: string
  readonly disabled?: boolean
}

/**
 * Options with labels: a bare value is labelled by its text.
 *
 * @param options - The options as given.
 * @returns The options with labels.
 */
export function normaliseOptions<T extends OptionValue>(options: readonly OptionInput<T>[]): ChoiceOption<T>[] {
  return options.map((o) =>
    typeof o === 'object' ? { ...o, label: o.label ?? String(o.value) } : { value: o, label: String(o) },
  )
}

/**
 * The value `dir` places from `value` among the enabled options, wrapping around the ends (the previous and next
 * arrows of a select). A value not among the options moves to the first or last option.
 *
 * @param options - The options.
 * @param value - The current value.
 * @param dir - Next (`1`) or previous (`-1`).
 * @returns The value to move to, or `undefined` when no option is enabled.
 */
export function cycleOption<T extends OptionValue>(
  options: readonly ChoiceOption<T>[],
  value: T,
  dir: 1 | -1,
): T | undefined {
  const enabled = options.filter((o) => o.disabled !== true)
  if (enabled.length === 0) return undefined
  const at = enabled.findIndex((o) => o.value === value)
  if (at < 0) return (dir > 0 ? enabled[0] : enabled[enabled.length - 1])?.value
  return enabled[(at + dir + enabled.length) % enabled.length]?.value
}
