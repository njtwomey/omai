import { type ReactNode, useId } from 'react'
import { cx, iconButtonClass, labelClass } from './classes.ts'
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon } from './icons.tsx'
import { cycleOption, normaliseOptions, type OptionInput, type OptionValue } from './options.ts'

/** Props of {@link Select}. */
export interface SelectProps<T extends OptionValue> {
  readonly label: ReactNode
  readonly value: T
  readonly options: readonly OptionInput<T>[]
  readonly onChange: (value: T) => void
  /** Show previous and next arrows that cycle through the options without opening the menu (default true). */
  readonly cycle?: boolean
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A choice from a dropdown menu, for long or wordy lists (datasets, models), with previous and next arrows either
 * side that step through the options in place, wrapping at the ends. Options are values or `{ value, label }`, of any
 * of string, number or boolean.
 *
 * @param props - The value, the options and the label.
 * @returns The select.
 */
export function Select<T extends OptionValue>(props: SelectProps<T>): ReactNode {
  const { label, value, onChange, cycle = true, disabled = false, className } = props
  const options = normaliseOptions(props.options)
  const id = useId()
  const at = options.findIndex((o) => o.value === value)
  const move = (dir: 1 | -1): void => {
    const next = cycleOption(options, value, dir)
    if (next !== undefined && next !== value) onChange(next)
  }
  return (
    <div className={cx('flex w-full min-w-36 max-w-xs flex-col gap-1.5', className)}>
      <label htmlFor={id} className={cx(labelClass, 'truncate')}>
        {label}
      </label>
      <div className="flex w-full">
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-l-md')}
            aria-label="Previous option"
            disabled={disabled || options.length < 2}
            onClick={() => move(-1)}
          >
            <ChevronLeftIcon />
          </button>
        ) : null}
        <div className={cx('relative min-w-0 flex-1', cycle && '-mx-px')}>
          <select
            id={id}
            disabled={disabled}
            value={at < 0 ? '' : String(at)}
            onChange={(e) => {
              const picked = options[Number(e.target.value)]
              if (picked !== undefined) onChange(picked.value)
            }}
            className={cx(
              'h-8 w-full min-w-0 appearance-none truncate border border-(--wb-line) bg-(--wb-raised) pr-7 pl-2.5 text-xs text-(--wb-fg) outline-none focus-visible:z-10 focus-visible:border-(--wb-ring) focus-visible:ring-2 focus-visible:ring-(--wb-ring)/40 disabled:opacity-50',
              cycle ? 'rounded-none' : 'rounded-md',
            )}
          >
            {at < 0 ? <option value="">(none)</option> : null}
            {options.map((o, i) => (
              <option key={String(o.value)} value={String(i)} disabled={o.disabled}>
                {o.label}
              </option>
            ))}
          </select>
          <ChevronDownIcon
            size={14}
            className="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2 text-(--wb-muted)"
          />
        </div>
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-r-md')}
            aria-label="Next option"
            disabled={disabled || options.length < 2}
            onClick={() => move(1)}
          >
            <ChevronRightIcon />
          </button>
        ) : null}
      </div>
    </div>
  )
}
