/**
 * A boolean control: a switch with its label beside it (show the data, use a log scale).
 */

import { type ReactNode, useId } from 'react'
import { type ControlSize, type ControlWidth, controlSizes, controlWidth, cx, labelClassFor } from './classes.ts'

/** Props of {@link Toggle}. */
export interface ToggleProps {
  readonly label: ReactNode
  readonly value: boolean
  readonly onChange: (value: boolean) => void
  /** A line under the label. */
  readonly description?: ReactNode
  /** The size: `sm` (28px high), `md` (32px, the default) or `lg` (40px); sets height, text and padding. */
  readonly size?: ControlSize
  /**
   * The width: pixels, a CSS length, `'fill'` (the container's width) or `'auto'` (the content's). Not given: the
   * container's width between a minimum and a maximum.
   */
  readonly width?: ControlWidth
  /** Cut the label to one line with an ellipsis (default: it wraps). */
  readonly truncate?: boolean
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
  const { label, value, onChange, description, disabled = false, className, size = 'md', truncate = false } = props
  const frame = controlWidth(props.width, 'min-w-36 max-w-xs')
  const sized = controlSizes[size]
  const id = useId()
  return (
    <div
      className={cx('flex items-center gap-2.5', frame.className, className)}
      {...(frame.style === undefined ? {} : { style: frame.style })}
    >
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={value}
        disabled={disabled}
        onClick={() => onChange(!value)}
        className={cx(
          'relative inline-flex shrink-0 items-center rounded-full border border-transparent outline-none transition-colors motion-reduce:transition-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:opacity-40',
          sized.track,
          value ? 'bg-(--wb-accent)' : 'bg-(--wb-grid)',
        )}
      >
        <span
          className={cx(
            'block rounded-full bg-(--wb-bg) shadow-sm transition-transform motion-reduce:transition-none',
            sized.knob,
            value ? sized.knobOn : sized.knobOff,
          )}
        />
      </button>
      <div className="flex min-w-0 flex-col">
        <label htmlFor={id} className={cx(labelClassFor(size, truncate), 'cursor-pointer text-(--wb-fg)')}>
          {label}
        </label>
        {description === undefined ? null : <span className="text-xs text-(--wb-muted)">{description}</span>}
      </div>
    </div>
  )
}
