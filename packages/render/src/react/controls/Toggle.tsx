/**
 * A boolean control: a switch with its label beside it (show the data, use a log scale).
 */

import { type ReactNode, useId } from 'react'
import { cx, labelClass } from './classes.ts'

/** Props of {@link Toggle}. */
export interface ToggleProps {
  readonly label: ReactNode
  readonly value: boolean
  readonly onChange: (value: boolean) => void
  /** A line under the label. */
  readonly description?: ReactNode
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A boolean as a switch with its label beside it (show the data, use a log scale).
 *
 * @param props - The value and the label.
 * @returns The switch.
 */
export function Toggle(props: ToggleProps): ReactNode {
  const { label, value, onChange, description, disabled = false, className } = props
  const id = useId()
  return (
    <div className={cx('flex min-w-36 max-w-xs items-center gap-2.5', className)}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        onClick={() => onChange(!value)}
        className={cx(
          'relative inline-flex h-[18px] w-8 shrink-0 items-center rounded-full border border-transparent outline-none transition-colors focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:opacity-40',
          value ? 'bg-(--wb-accent)' : 'bg-(--wb-grid)',
        )}
      >
        <span
          className={cx(
            'block size-3.5 rounded-full bg-(--wb-bg) shadow-sm transition-transform',
            value ? 'translate-x-[15px]' : 'translate-x-px',
          )}
        />
      </button>
      <div className="flex min-w-0 flex-col">
        <label htmlFor={id} className={cx(labelClass, 'cursor-pointer truncate text-(--wb-fg)')}>
          {label}
        </label>
        {description === undefined ? null : <span className="text-xs text-(--wb-muted)">{description}</span>}
      </div>
    </div>
  )
}
