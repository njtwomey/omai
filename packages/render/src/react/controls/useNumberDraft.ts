/**
 * The editing behaviour of a typed number field: while focused the input holds a draft validated as it is typed, Enter
 * or blur commits a valid draft, and an invalid one is refused with its message.
 */

import { type FocusEvent, type KeyboardEvent, useState } from 'react'
import { checkNumber, formatFieldValue, type NumberSpec, stepNumber } from './number.ts'

/** What {@link useNumberDraft} returns: the input's props and the draft's error, if any. */
export interface NumberDraft {
  readonly error: string | null
  readonly props: {
    readonly value: string
    readonly inputMode: 'decimal' | 'numeric'
    readonly 'aria-invalid': true | undefined
    readonly onFocus: (e: FocusEvent<HTMLInputElement>) => void
    readonly onChange: (e: { readonly target: { readonly value: string } }) => void
    readonly onBlur: () => void
    readonly onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => void
  }
}

/**
 * The editing behaviour of a typed number: while focused the input holds a draft, validated as it is typed; Enter or
 * blur commits a valid draft (an invalid one is refused on Enter, with its message showing, and reverted on blur);
 * Escape reverts; ArrowUp and ArrowDown step (ten steps, or a decade, with Shift).
 *
 * @param options - The value, its rules, how to show it and what to call with a committed value.
 * @returns The input's props and the current error.
 */
export function useNumberDraft(options: {
  readonly value: number
  readonly spec: NumberSpec
  readonly onCommit: (value: number) => void
  readonly format?: (value: number) => string
  /** Reads a draft (default: a decimal number); with `draftText`, lets a value be typed and shown as words. */
  readonly parse?: (text: string) => number | null
  /** The text a draft starts as when the field gains focus (default the value as a plain number). */
  readonly draftText?: (value: number) => string
}): NumberDraft {
  const { value, spec, onCommit } = options
  const textOf = options.draftText ?? ((v: number): string => String(v))
  const format = options.format ?? ((v: number) => formatFieldValue(spec, v))
  const [draft, setDraft] = useState<string | null>(null)
  const checked = draft === null ? null : checkNumber(spec, draft, options.parse)
  const error = checked?.error ?? null
  const commit = (): void => {
    if (checked !== null && checked.error === undefined && checked.value !== value) onCommit(checked.value)
    setDraft(null)
  }
  return {
    error,
    props: {
      value: draft ?? format(value),
      inputMode: spec.integer === true && (spec.min ?? -1) >= 0 ? 'numeric' : 'decimal',
      'aria-invalid': error === null ? undefined : true,
      onFocus: (e) => {
        setDraft(textOf(value))
        e.currentTarget.select()
      },
      onChange: (e) => setDraft(e.target.value),
      onBlur: commit,
      onKeyDown: (e) => {
        if (e.key === 'Enter') {
          if (error === null) {
            const committed = checked !== null && checked.error === undefined ? checked.value : value
            commit()
            setDraft(textOf(committed))
          }
        } else if (e.key === 'Escape') {
          setDraft(textOf(value))
        } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault()
          const base = checked !== null && checked.error === undefined ? checked.value : value
          const next = stepNumber(spec, base, e.key === 'ArrowUp' ? 1 : -1, e.shiftKey)
          if (next !== null) {
            onCommit(next)
            setDraft(textOf(next))
          }
        }
      },
    },
  }
}
