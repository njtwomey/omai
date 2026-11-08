/**
 * The settings gear of one option in a list of union options: a button that opens the option's own space as controls in
 * a popover. `ModelSelector` and `SlotVariants` draw one on each row; what differs between them (whether choosing the
 * gear also chooses the option, where the knobs are kept) stays with the list, which passes `onToggle` and `onChange`.
 */

import { asModeConfig, type Space, type SpaceValue } from 'omai-engine/space'
import { type ReactNode, useRef } from 'react'
import { cx } from './classes.ts'
import { SlidersIcon } from './icons.tsx'
import { Popover } from './Popover.tsx'
import { SpaceControls } from './SpaceControls.tsx'

/** Props of {@link OptionSettings}. */
export interface OptionSettingsProps {
  /** The option's name, in the gear's label and the popover's. */
  readonly name: string
  /** The option's own space (a union's `options[name]`). */
  readonly space: Space
  /** The option's knob values. */
  readonly value: Readonly<Record<string, SpaceValue>>
  /** Called with the knobs after an edit. */
  readonly onChange: (knobs: Readonly<Record<string, SpaceValue>>) => void
  /** Whether this option's popover is open. */
  readonly open: boolean
  /** The gear was pressed. */
  readonly onToggle: () => void
  /** The popover asked to close (Escape, a press outside). */
  readonly onClose: () => void
  /** Whether the knobs are out of reach (the whole list is disabled, or the option does not apply). */
  readonly disabled: boolean
}

/**
 * The gear and popover of one option: disabled when the option has no knobs, with a title that says so.
 *
 * @param props - The option, its knobs and the popover's state.
 * @returns The gear button and its popover.
 */
export function OptionSettings(props: OptionSettingsProps): ReactNode {
  const { name, space, value, onChange, open, onToggle, onClose, disabled } = props
  const anchor = useRef<HTMLButtonElement | null>(null)
  const count = Object.keys((space as { readonly spaces?: object }).spaces ?? {}).length
  return (
    <>
      <button
        type="button"
        ref={anchor}
        aria-label={`Settings of ${name}`}
        title={count === 0 ? `${name} has no settings` : `Settings of ${name}`}
        disabled={disabled || count === 0}
        onClick={onToggle}
        className={cx(
          'inline-flex size-8 shrink-0 items-center justify-center rounded-[5px] text-(--wb-muted) outline-none hover:bg-(--wb-bg) hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-30',
          open && 'bg-(--wb-bg) text-(--wb-accent)',
        )}
      >
        <SlidersIcon size={14} />
      </button>
      <Popover
        anchor={anchor}
        open={open}
        onClose={onClose}
        align="end"
        role="dialog"
        label={`Settings of ${name}`}
        className="w-64 p-3"
      >
        <div className="flex flex-col gap-2">
          <SpaceControls
            space={space}
            value={value}
            onChange={(next) => onChange(asModeConfig(next))}
            disabled={disabled}
          />
        </div>
      </Popover>
    </>
  )
}
