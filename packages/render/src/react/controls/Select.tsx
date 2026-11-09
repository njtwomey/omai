/**
 * A dropdown select for long or wordy option lists (datasets, models), with previous and next arrows either side that
 * step through the options in place, wrapping at the ends.
 */

import { type ReactNode, useId, useRef, useState } from 'react'
import {
  type ControlSize,
  type ControlWidth,
  controlSizes,
  controlWidth,
  cx,
  iconButtonClassFor,
  labelClassFor,
  menuItemClass,
} from './classes.ts'
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon } from './icons.tsx'
import { cycleOption, normaliseOptions, type OptionInput, type OptionValue } from './options.ts'
import { Popover } from './Popover.tsx'

/** Props of {@link Select}. */
export interface SelectProps<T extends OptionValue> {
  readonly label: ReactNode
  readonly value: T
  readonly options: readonly OptionInput<T>[]
  readonly onChange: (value: T) => void
  /** Show previous and next arrows that cycle through the options without opening the menu (default true). */
  readonly cycle?: boolean
  /**
   * How the options open: `native` is the browser's select (the default; best on touch screens), `menu` a popup list drawn
   * by the library inside a `Popover`, which stays in the page it is on (a slide, a scaled island) where a native list
   * would open outside it.
   */
  readonly appearance?: 'native' | 'menu'
  /** For `appearance="menu"`: draw the list into this element instead of the page (see `PopoverContainer`), as `container` on `Popover`. */
  readonly container?: HTMLElement | null
  /** The earlier name of `container`; `container` wins when both are given. */
  readonly menuContainer?: HTMLElement | null
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
 * A choice from a dropdown menu, for long or wordy lists (datasets, models), with previous and next arrows either
 * side that step through the options in place, wrapping at the ends. Options are values or `{ value, label }`, of any
 * of string, number or boolean.
 *
 * @param props - The value, the options and the label.
 * @returns The select.
 * @index function React component: a dropdown with cycling arrows: `<Select label="loss" value={v} options={names} onChange={setV} />`
 */
export function Select<T extends OptionValue>(props: SelectProps<T>): ReactNode {
  const { label, value, onChange, cycle = true, disabled = false, className, size = 'md', truncate = false } = props
  const frame = controlWidth(props.width, 'w-full min-w-36 max-w-xs')
  const sized = controlSizes[size]
  const options = normaliseOptions(props.options)
  const id = useId()
  const [open, setOpen] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const menu = props.appearance === 'menu'
  const at = options.findIndex((o) => o.value === value)
  const move = (dir: 1 | -1): void => {
    const next = cycleOption(options, value, dir)
    if (next !== undefined && next !== value) onChange(next)
  }
  return (
    <div
      className={cx('flex flex-col gap-1.5', frame.className, className)}
      {...(frame.style === undefined ? {} : { style: frame.style })}
    >
      <label htmlFor={id} className={labelClassFor(size, truncate)}>
        {label}
      </label>
      <div className="flex w-full">
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClassFor(size), 'rounded-l-md')}
            aria-label="Previous option"
            disabled={disabled || options.length < 2}
            onClick={() => move(-1)}
          >
            <ChevronLeftIcon />
          </button>
        ) : null}
        <div className={cx('relative min-w-0 flex-1', cycle && '-mx-px')}>
          {menu ? (
            <>
              <button
                ref={trigger}
                id={id}
                type="button"
                disabled={disabled}
                aria-haspopup="listbox"
                aria-expanded={open}
                onClick={() => setOpen((o) => !o)}
                className={cx(
                  sized.height,
                  sized.text,
                  'w-full min-w-0 truncate border border-(--wb-grid) bg-(--wb-bg) pr-7 pl-2.5 text-left text-(--wb-fg) outline-none focus-visible:z-10 focus-visible:border-(--wb-accent) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/40 disabled:opacity-50',
                  cycle ? 'rounded-none' : 'rounded-md',
                )}
              >
                {options[at]?.label ?? '(none)'}
              </button>
              <Popover
                anchor={trigger}
                open={open}
                onClose={() => setOpen(false)}
                role="listbox"
                label="options"
                matchAnchorWidth
                {...((props.container ?? props.menuContainer) === undefined
                  ? {}
                  : { container: props.container ?? props.menuContainer ?? null })}
              >
                {options.map((o, i) => (
                  <button
                    key={String(o.value)}
                    type="button"
                    role="option"
                    aria-selected={i === at}
                    data-selected={i === at}
                    disabled={o.disabled}
                    title={o.description}
                    className={menuItemClass}
                    onClick={() => {
                      onChange(o.value)
                      setOpen(false)
                    }}
                  >
                    {o.label}
                  </button>
                ))}
              </Popover>
            </>
          ) : (
            <select
              id={id}
              disabled={disabled}
              value={at < 0 ? '' : String(at)}
              onChange={(e) => {
                const picked = options[Number(e.target.value)]
                if (picked !== undefined) onChange(picked.value)
              }}
              className={cx(
                sized.height,
                sized.text,
                'w-full min-w-0 appearance-none truncate border border-(--wb-grid) bg-(--wb-bg) pr-7 pl-2.5 text-(--wb-fg) outline-none focus-visible:z-10 focus-visible:border-(--wb-accent) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/40 disabled:opacity-50',
                cycle ? 'rounded-none' : 'rounded-md',
              )}
            >
              {at < 0 ? <option value="">(none)</option> : null}
              {options.map((o, i) => (
                <option key={String(o.value)} value={String(i)} disabled={o.disabled} title={o.description}>
                  {o.label}
                </option>
              ))}
            </select>
          )}
          <ChevronDownIcon
            size={14}
            className="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2 text-(--wb-muted)"
          />
        </div>
        {cycle ? (
          <button
            type="button"
            className={cx(iconButtonClassFor(size), 'rounded-r-md')}
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
