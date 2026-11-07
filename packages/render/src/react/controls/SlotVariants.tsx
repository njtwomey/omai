/**
 * The controls of a comparison over a slot: every option of a union (every solver, every loss) listed with the colour
 * its variant is drawn in, a short description, and a settings button that opens that option's own knobs in a popover.
 * The page holds the knobs of every option, so each variant of the comparison keeps its settings while the others are
 * edited. An option a rule puts out of reach is listed greyed with the reason, and is left out of the comparison.
 */

import { asModeConfig, defaults, type Space, type SpaceValue, type UnionSpace } from 'mdml-engine/space'
import { type ReactNode, useRef, useState } from 'react'
import { categoricalColour } from '../../theme/palettes.ts'
import { cx, labelClass } from './classes.ts'
import { SlidersIcon } from './icons.tsx'
import { Popover } from './Popover.tsx'
import { SpaceControls } from './SpaceControls.tsx'

/** The knobs held for each option of a slot: option name to the option's own knob values. */
export type SlotKnobs = Readonly<Record<string, Readonly<Record<string, SpaceValue>>>>

/** Props of {@link SlotVariants}. */
export interface SlotVariantsProps {
  /** The union whose options are compared. */
  readonly space: UnionSpace
  /** The knobs held for each option (an option missing from it is at its defaults). */
  readonly knobs: SlotKnobs
  readonly onChange: (knobs: SlotKnobs) => void
  readonly label?: ReactNode
  /** A short description under an option's name. */
  readonly descriptions?: Readonly<Record<string, string>>
  /** Whether an option applies (default all do); the others are listed greyed and are not compared. */
  readonly admissible?: (option: string) => boolean
  /** The note on an option that does not apply, shown as its title. */
  readonly why?: (option: string) => string | undefined
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * The colour of each option's variant, in the order the comparison runs them: the options that apply, one tab10 slot
 * each (the colour the overlaid curves give the same variant).
 *
 * @param names - The union's option names.
 * @param admissible - Whether an option applies (default all do).
 * @returns The colour by option; options that do not apply have none.
 */
export function variantColours(
  names: readonly string[],
  admissible: (option: string) => boolean = () => true,
): Readonly<Record<string, string>> {
  return Object.fromEntries(names.filter(admissible).map((n, i) => [n, categoricalColour(i)]))
}

/**
 * A list of the options of a union compared at once, each with its colour swatch and a settings button for its own
 * knobs. Unlike {@link ModelSelector} nothing is chosen: every option that applies is a variant.
 *
 * @param props - The union, the knobs held for its options and what to call with new ones.
 * @returns The list.
 * @index function React component: choose slot options to compare: `<SlotVariants space={union} knobs={k} onChange={setK} />`
 */
export function SlotVariants(props: SlotVariantsProps): ReactNode {
  const { space, knobs, onChange, disabled = false } = props
  const names = Object.keys(space.options)
  const colours = variantColours(names, props.admissible)
  const [open, setOpen] = useState<string | undefined>(undefined)
  const anchors = useRef<Record<string, HTMLButtonElement | null>>({})
  const knobsOf = (name: string): Readonly<Record<string, SpaceValue>> =>
    knobs[name] ?? asModeConfig(defaults(space.options[name] as Space))
  return (
    <div className={cx('flex w-full min-w-56 max-w-md flex-col gap-1.5', props.className)}>
      {props.label === undefined ? null : <span className={cx(labelClass, 'truncate')}>{props.label}</span>}
      <ul
        aria-label={typeof props.label === 'string' ? props.label : undefined}
        className="m-0 flex list-none flex-col gap-0.5 rounded-md border border-(--wb-grid) bg-(--wb-surface) p-0.5"
      >
        {names.map((name) => {
          const colour = colours[name]
          const allowed = colour !== undefined
          const title = allowed
            ? props.descriptions?.[name]
            : (props.why?.(name) ?? 'not available with the current settings')
          const count = Object.keys((space.options[name] as { readonly spaces?: object }).spaces ?? {}).length
          return (
            <li key={name} className="flex items-stretch gap-0.5" data-variant={name} data-allowed={allowed}>
              <div
                title={title}
                className={cx(
                  'flex min-h-8 min-w-0 flex-1 items-center gap-2 rounded-[5px] px-2 text-xs',
                  allowed ? 'text-(--wb-fg)' : 'text-(--wb-muted) opacity-50',
                )}
              >
                <span
                  aria-hidden="true"
                  className="size-2.5 shrink-0 rounded-full"
                  style={{
                    background: allowed ? colour : 'transparent',
                    border: `1.5px solid ${colour ?? 'var(--wb-muted)'}`,
                  }}
                />
                <span className="flex min-w-0 flex-col">
                  <span className="truncate font-medium">{name}</span>
                  {title === undefined ? null : (
                    <span className="truncate text-[10px] font-normal text-(--wb-muted)">{title}</span>
                  )}
                </span>
              </div>
              <button
                type="button"
                ref={(el) => {
                  anchors.current[name] = el
                }}
                aria-label={`Settings of ${name}`}
                title={count === 0 ? `${name} has no settings` : `Settings of ${name}`}
                disabled={disabled || !allowed || count === 0}
                onClick={() => setOpen(open === name ? undefined : name)}
                className={cx(
                  'inline-flex size-8 shrink-0 items-center justify-center rounded-[5px] text-(--wb-muted) outline-none hover:bg-(--wb-bg) hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-30',
                  open === name && 'bg-(--wb-bg) text-(--wb-accent)',
                )}
              >
                <SlidersIcon size={14} />
              </button>
              <Popover
                anchor={{ current: anchors.current[name] ?? null }}
                open={open === name}
                onClose={() => setOpen(undefined)}
                align="end"
                role="dialog"
                label={`Settings of ${name}`}
                className="w-64 p-3"
              >
                <div className="flex flex-col gap-2">
                  <SpaceControls
                    space={space.options[name] as Space}
                    value={knobsOf(name)}
                    onChange={(next) => onChange({ ...knobs, [name]: asModeConfig(next) })}
                    disabled={disabled}
                  />
                </div>
              </Popover>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
