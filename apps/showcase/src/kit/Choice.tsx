import { type ReactNode, useId } from 'react'

/** One option of a {@link Choice}: the value it stands for, a short label and a one-line plain description. */
export interface ChoiceOption<T extends string | number> {
  readonly value: T
  readonly label: string
  readonly description: string
}

/**
 * A small, calm set of choices for the few stories whose reader decides how something is built before pressing play: a radio
 * group with a label, in which each choice has a short name and a one-line description under it. It is a real group of radio
 * inputs, so the arrow keys move between the choices, tab enters and leaves the group, and a screen reader hears the label,
 * the choice and its description. The choices sit in a row where they fit and stack at a narrow width. The value lives in the
 * page's own `useState`; this piece holds nothing.
 *
 * @param props - The label of the group, its options, the chosen value and what to do on a change.
 * @returns The group.
 */
export function Choice<T extends string | number>(props: {
  readonly label: string
  readonly options: readonly ChoiceOption<T>[]
  readonly value: T
  readonly onChange: (value: T) => void
}): ReactNode {
  const { label, options, value, onChange } = props
  const name = useId()
  return (
    <fieldset className="story-block m-0 min-w-0 border-0 p-0 font-sans" data-testid="choice">
      <legend className="mb-2 p-0 text-sm font-semibold text-fg">{label}</legend>
      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(9.5rem, 1fr))' }}>
        {options.map((option) => {
          const id = `${name}-${String(option.value)}`
          const chosen = option.value === value
          return (
            <label
              key={String(option.value)}
              htmlFor={id}
              className={`flex cursor-pointer flex-col gap-0.5 rounded-lg border px-3.5 py-2.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent ${
                chosen ? 'border-accent bg-surface' : 'border-line bg-bg hover:border-muted'
              }`}
            >
              <input
                id={id}
                type="radio"
                name={name}
                value={String(option.value)}
                checked={chosen}
                onChange={() => onChange(option.value)}
                aria-describedby={`${id}-description`}
                className="sr-only"
              />
              <span className={`text-[0.95rem] font-medium ${chosen ? 'text-fg' : 'text-muted'}`}>{option.label}</span>
              <span id={`${id}-description`} className="text-[0.8rem] leading-snug text-muted">
                {option.description}
              </span>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}
