import type { ReactNode } from 'react'
import { cx, iconButtonClass, labelClass } from './classes.ts'
import { ChevronLeftIcon, ChevronRightIcon } from './icons.tsx'
import { cycleOption, normaliseOptions, type OptionInput, type OptionValue } from './options.ts'

/** Props of {@link Choice}. */
export interface ChoiceProps<T extends OptionValue> {
  readonly label: ReactNode
  readonly value: T
  readonly options: readonly OptionInput<T>[]
  readonly onChange: (value: T) => void
  /** Show previous and next arrows that cycle through the options (default false: every option is in view). */
  readonly cycle?: boolean
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A choice among two to five short options shown side by side (a segmented control): one press picks, and Left and
 * Right arrow keys move along. With `cycle`, previous and next arrows sit either side, as on a {@link Select}.
 *
 * @param props - The value, the options and the label.
 * @returns The segmented control.
 */
export function Choice<T extends OptionValue>(props: ChoiceProps<T>): ReactNode {
  const { label, value, onChange, cycle = false, disabled = false, className } = props
  const options = normaliseOptions(props.options)
  const move = (dir: 1 | -1): void => {
    const next = cycleOption(options, value, dir)
    if (next !== undefined && next !== value) onChange(next)
  }
  return (
    <div className={cx('flex w-full min-w-36 max-w-sm flex-col gap-1.5', className)}>
      <span className={cx(labelClass, 'truncate')}>{label}</span>
      <div className="flex w-full">
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-l-md')}
            aria-label="Previous option"
            disabled={disabled}
            onClick={() => move(-1)}
          >
            <ChevronLeftIcon />
          </button>
        ) : null}
        <div
          role="radiogroup"
          aria-label={typeof label === 'string' ? label : undefined}
          className={cx(
            'flex h-8 min-w-0 flex-1 gap-0.5 border border-(--wb-grid) bg-(--wb-surface) p-0.5',
            cycle ? '-mx-px' : 'rounded-md',
          )}
          onKeyDown={(e) => {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') move(1)
            else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') move(-1)
            else return
            e.preventDefault()
          }}
        >
          {options.map((o) => {
            const selected = o.value === value
            return (
              // biome-ignore lint/a11y/useSemanticElements: a segment drawn as a button; native radios cannot be drawn as segments
              <button
                key={String(o.value)}
                type="button"
                role="radio"
                aria-checked={selected}
                tabIndex={selected ? 0 : -1}
                title={o.description ?? o.label}
                disabled={disabled || o.disabled === true}
                onClick={() => onChange(o.value)}
                className={cx(
                  'min-w-0 flex-auto truncate rounded-[5px] px-2 text-xs whitespace-nowrap outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:opacity-40',
                  selected
                    ? 'bg-(--wb-bg) font-medium text-(--wb-fg) shadow-sm'
                    : 'text-(--wb-muted) hover:text-(--wb-fg)',
                )}
              >
                {o.label}
              </button>
            )
          })}
        </div>
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-r-md')}
            aria-label="Next option"
            disabled={disabled}
            onClick={() => move(1)}
          >
            <ChevronRightIcon />
          </button>
        ) : null}
      </div>
    </div>
  )
}
