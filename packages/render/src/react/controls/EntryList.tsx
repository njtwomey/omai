/**
 * A list to choose a registry entry from: a search box over the rows, and one row per entry showing its title and its
 * tags as chips, the chosen one marked. The dataset and environment selectors are this list over their registries;
 * `tagLabel` turns a camel-case tag into words for display.
 */

import { matchesText } from 'mdml-engine/base'
import { type ReactNode, useId, useState } from 'react'
import { cx, inputBoxClass, labelClass } from './classes.ts'

/** One row of an {@link EntryList}: an entry's name, title and tags, and what the search box also matches. */
export interface EntryRow {
  readonly name: string
  readonly title: string
  readonly tags: readonly string[]
  /** Searched, not shown. */
  readonly description?: string
  readonly disabled?: boolean
}

/** Props of {@link EntryList}. */
export interface EntryListProps {
  readonly label: ReactNode
  readonly rows: readonly EntryRow[]
  /** The chosen entry's name. */
  readonly value: string
  readonly onChange: (name: string) => void
  /** Show the search box (default true). */
  readonly search?: boolean
  readonly placeholder?: string
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A camel-case tag as words (`timeSeries` to `time series`, `classicControl` to `classic control`).
 *
 * @param tag - The tag.
 * @returns The words, lower case.
 */
export function tagLabel(tag: string): string {
  return tag.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase()
}

/**
 * Tags as a row of small chips.
 *
 * @param props - The tags.
 * @returns The chips.
 */
export function Tags(props: { readonly items: readonly string[]; readonly className?: string }): ReactNode {
  return (
    <span className={cx('flex flex-wrap gap-1', props.className)}>
      {props.items.map((t) => (
        <span
          key={t}
          className="rounded border border-(--wb-grid) px-1 text-[10px] leading-4 text-(--wb-muted) select-none"
        >
          {t}
        </span>
      ))}
    </span>
  )
}

/**
 * A list of entries to choose from, with a search box over their names, titles, tags and descriptions, each row
 * showing its title and tags; the chosen row is marked and announced as selected.
 *
 * @param props - The rows, the chosen name and what to call with another.
 * @returns The list.
 */
export function EntryList(props: EntryListProps): ReactNode {
  const { label, rows, value, onChange, search = true, placeholder = 'Search', disabled = false, className } = props
  const [text, setText] = useState('')
  const id = useId()
  const shown = rows.filter((r) => matchesText(text, [r.name, r.title, r.description, ...r.tags]))
  return (
    <div className={cx('flex w-full min-w-36 flex-col gap-1.5', className)}>
      <label htmlFor={search ? `${id}-search` : `${id}-list`} className={cx(labelClass, 'truncate')}>
        {label}
      </label>
      {search ? (
        <div className={cx(inputBoxClass, 'rounded-md')}>
          <input
            id={`${id}-search`}
            type="search"
            value={text}
            placeholder={placeholder}
            aria-label={`Search ${typeof label === 'string' ? label : 'entries'}`}
            disabled={disabled}
            onChange={(e) => setText(e.target.value)}
            className="h-full w-full min-w-0 bg-transparent px-2.5 text-xs text-(--wb-fg) outline-none placeholder:text-(--wb-muted)"
          />
        </div>
      ) : null}
      <div
        id={`${id}-list`}
        role="listbox"
        aria-label={typeof label === 'string' ? label : undefined}
        className="max-h-56 overflow-y-auto rounded-md border border-(--wb-grid) bg-(--wb-bg) py-1"
      >
        {shown.length === 0 ? <p className="px-2.5 py-1 text-xs text-(--wb-muted)">nothing matches</p> : null}
        {shown.map((r) => {
          const selected = r.name === value
          return (
            <button
              key={r.name}
              type="button"
              role="option"
              aria-selected={selected}
              disabled={disabled || r.disabled === true}
              data-selected={selected}
              onClick={() => onChange(r.name)}
              className="flex w-full flex-col items-start gap-0.5 px-2.5 py-1.5 text-left text-xs text-(--wb-fg) outline-none hover:bg-(--wb-surface) focus-visible:bg-(--wb-surface) disabled:opacity-40 data-[selected=true]:bg-(--wb-surface)"
            >
              <span className={cx('truncate', selected && 'font-semibold text-(--wb-accent)')}>{r.title}</span>
              <Tags items={r.tags} />
            </button>
          )
        })}
      </div>
    </div>
  )
}
