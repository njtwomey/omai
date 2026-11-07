/**
 * The model selector: a list of the named options of a union space (the models of a task, the solvers of a slot),
 * the chosen one marked, each with a settings button that opens that option's own space as controls in a popover. The
 * compact form of a hierarchy: the list is the union's choice, the gear is the chosen option's knobs. Options a rule
 * puts out of reach are drawn disabled with the reason as their title.
 */

import { asModeConfig, defaults, type Space, type SpaceValue, type UnionSpace } from 'mdml-engine/space'
import { type ReactNode, useId, useRef, useState } from 'react'
import { cx, labelClass } from './classes.ts'
import { SlidersIcon } from './icons.tsx'
import { Popover } from './Popover.tsx'
import { SpaceControls } from './SpaceControls.tsx'
import { settleValue } from './space-controls.ts'

/** Props of {@link ModelSelector}. */
export interface ModelSelectorProps<V extends SpaceValue> {
  /** The union of the options, each with the space of its own knobs. */
  readonly space: UnionSpace
  /** The chosen option and its knobs (the union's value: the tag beside the knobs). */
  readonly value: V
  readonly onChange: (value: V) => void
  readonly label?: ReactNode
  /** A short description under an option's name. */
  readonly descriptions?: Readonly<Record<string, string>>
  /** Whether an option may be chosen (default all may); the others are drawn disabled. */
  readonly admissible?: (option: string) => boolean
  /** The note on an option that may not be chosen, shown as its title. */
  readonly why?: (option: string) => string | undefined
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A list of the options of a union space with a settings button on each: pressing a name chooses the option, pressing
 * its gear chooses it too and opens its knobs in a popover. The knobs set for an option are remembered while the
 * selector is mounted, so choosing another option and coming back finds them as left.
 *
 * @param props - The union space, its value and what to call with a new one.
 * @returns The selector.
 * @index function React component: choose a union option and its knobs: `<ModelSelector space={union} value={v} onChange={setV} />`
 */
export function ModelSelector<V extends SpaceValue>(props: ModelSelectorProps<V>): ReactNode {
  const { space, value, onChange, disabled = false } = props
  const tag = space.tag ?? 'kind'
  const names = Object.keys(space.options)
  const current = ((): string => {
    const named = asModeConfig(value)[tag]
    return typeof named === 'string' && named in space.options ? named : (space.default ?? (names[0] as string))
  })()
  // The knobs last set for each option, so that coming back to one finds its settings as they were.
  const [memory, setMemory] = useState<Readonly<Record<string, Readonly<Record<string, SpaceValue>>>>>({})
  const [open, setOpen] = useState<string | undefined>(undefined)
  const anchors = useRef<Record<string, HTMLButtonElement | null>>({})
  const labelId = useId()
  const knobsOf = (name: string): Readonly<Record<string, SpaceValue>> => {
    if (name === current) {
      const { [tag]: _tag, ...rest } = asModeConfig(value)
      return rest
    }
    return memory[name] ?? asModeConfig(defaults(space.options[name] as Space))
  }
  const choose = (name: string, knobs: Readonly<Record<string, SpaceValue>> = knobsOf(name)): void => {
    setMemory((m) => ({ ...m, [name]: knobs }))
    onChange(settleValue(space, { [tag]: name, ...knobs }) as V)
  }
  return (
    <div className={cx('flex w-full min-w-44 max-w-sm flex-col gap-1.5', props.className)}>
      {props.label === undefined ? null : (
        <span id={labelId} className={cx(labelClass, 'truncate')}>
          {props.label}
        </span>
      )}
      {/* Buttons that show which is chosen (`aria-pressed`): a radio group would owe its readers arrow keys and one tab stop. */}
      {/* biome-ignore lint/a11y/useSemanticElements: a styled box of rows; a fieldset draws differently */}
      <div
        role="group"
        aria-labelledby={props.label === undefined ? undefined : labelId}
        className="flex flex-col gap-0.5 rounded-md border border-(--wb-grid) bg-(--wb-surface) p-0.5"
      >
        {names.map((name) => {
          const selected = name === current
          const allowed = props.admissible?.(name) ?? true
          const reason = allowed
            ? props.descriptions?.[name]
            : (props.why?.(name) ?? 'not available with the current settings')
          const knobs = Object.keys((space.options[name] as { readonly spaces?: object }).spaces ?? {}).length
          return (
            <div key={name} className="flex items-stretch gap-0.5">
              <button
                type="button"
                aria-pressed={selected}
                title={reason}
                disabled={disabled || !allowed}
                onClick={() => choose(name)}
                className={cx(
                  'flex min-h-8 min-w-0 flex-1 flex-col justify-center rounded-[5px] px-2 text-left text-xs outline-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:opacity-40',
                  selected
                    ? 'bg-(--wb-bg) font-medium text-(--wb-fg) shadow-sm'
                    : 'text-(--wb-muted) hover:text-(--wb-fg)',
                )}
              >
                <span className="truncate">{name}</span>
                {props.descriptions?.[name] === undefined ? null : (
                  <span className="truncate text-[10px] font-normal text-(--wb-muted)">{props.descriptions[name]}</span>
                )}
              </button>
              <button
                type="button"
                ref={(el) => {
                  anchors.current[name] = el
                }}
                aria-label={`Settings of ${name}`}
                title={knobs === 0 ? `${name} has no settings` : `Settings of ${name}`}
                disabled={disabled || !allowed || knobs === 0}
                onClick={() => {
                  if (!selected) choose(name)
                  setOpen(open === name ? undefined : name)
                }}
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
                    onChange={(knobs) => choose(name, asModeConfig(knobs))}
                    disabled={disabled}
                  />
                </div>
              </Popover>
            </div>
          )
        })}
      </div>
    </div>
  )
}
