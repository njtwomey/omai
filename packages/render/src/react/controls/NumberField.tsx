import { type PointerEvent, type ReactNode, useCallback, useId, useRef, useState } from 'react'
import { cx, iconButtonClass, inputBoxClass, labelClass, menuClass, menuItemClass } from './classes.ts'
import { useDismiss } from './dismiss.ts'
import { ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon } from './icons.tsx'
import {
  formatFieldValue,
  fromFraction,
  hasRange,
  type NumberSpec,
  numberLadder,
  stepNumber,
  toFraction,
} from './number.ts'
import { useNumberDraft } from './useNumberDraft.ts'

/** Props of {@link NumberField}. */
export interface NumberFieldProps extends NumberSpec {
  readonly label: ReactNode
  readonly value: number
  readonly onChange: (value: number) => void
  /** How the value shows when not being edited (default {@link formatFieldValue}). */
  readonly format?: (value: number) => string
  /** A unit shown beside the label. */
  readonly unit?: string
  /** The dropdown's values (default: 11 between the bounds); `false` hides it. */
  readonly ladder?: readonly number[] | false
  /** Whether to show the slider along the field's bottom edge (default: when both bounds are finite). */
  readonly slider?: boolean
  readonly disabled?: boolean
  readonly className?: string
}

/**
 * A typed number in one tight row, `‹ [ value ⌄ ] ›`: type a value, slide it along the rail on the field's bottom
 * edge, step it with the buttons or the arrow keys, or pick one of 11 values between its bounds from the dropdown.
 * Integer and log-scaled numbers step, slide and ladder on their own scale. Typing is validated rather than clamped: a
 * draft that breaks a rule turns the field red with a short message and is not applied.
 *
 * @param props - The value, its rules and its label.
 * @returns The field.
 */
export function NumberField(props: NumberFieldProps): ReactNode {
  const { label, value, onChange, unit, disabled = false, className } = props
  const spec: NumberSpec = {
    ...(props.min === undefined ? {} : { min: props.min }),
    ...(props.max === undefined ? {} : { max: props.max }),
    ...(props.integer === undefined ? {} : { integer: props.integer }),
    ...(props.log === undefined ? {} : { log: props.log }),
    ...(props.step === undefined ? {} : { step: props.step }),
  }
  const format = props.format ?? ((v: number) => formatFieldValue(spec, v))
  const ladder = props.ladder === false ? [] : (props.ladder ?? numberLadder(spec))
  const showSlider = props.slider ?? hasRange(spec)
  const id = useId()
  const errorId = useId()
  const commit = (v: number): void => {
    if (v !== value && Number.isFinite(v)) onChange(v)
  }
  const draft = useNumberDraft({ value, spec, onCommit: commit, format })
  const down = stepNumber(spec, value, -1)
  const up = stepNumber(spec, value, 1)
  const error = draft.error

  return (
    <div className={cx('flex w-full min-w-36 max-w-xs flex-col gap-1.5', className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className={cx(labelClass, 'min-w-0 truncate')}>
          {label}
        </label>
        {unit === undefined ? null : <span className="text-xs text-(--wb-muted)">{unit}</span>}
      </div>
      <div className="flex w-full">
        <button
          type="button"
          className={cx(iconButtonClass, 'rounded-l-md')}
          aria-label="Decrease"
          disabled={disabled || down === null}
          onClick={(e) => {
            const next = stepNumber(spec, value, -1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <ChevronLeftIcon />
        </button>
        <div
          className={cx(
            inputBoxClass,
            '-mx-px',
            disabled && 'opacity-50',
            error !== null && 'z-10 border-(--wb-danger) focus-within:border-(--wb-danger)',
          )}
        >
          <input
            id={id}
            disabled={disabled}
            aria-describedby={error === null ? undefined : errorId}
            className={cx(
              'h-full w-0 min-w-0 flex-1 bg-transparent pl-2 text-right font-mono text-xs tabular-nums outline-none',
              ladder.length > 0 ? 'pr-0.5' : 'pr-2',
            )}
            {...draft.props}
          />
          {ladder.length > 0 ? (
            <LadderMenu values={ladder} value={value} format={format} disabled={disabled} onPick={commit} />
          ) : null}
          {showSlider && hasRange(spec) ? (
            <Rail
              fraction={toFraction(spec, value)}
              disabled={disabled}
              label={typeof label === 'string' ? label : 'value'}
              valueText={format(value)}
              onFraction={(t) => commit(fromFraction(spec, t))}
              onStep={(dir, big) => {
                const next = stepNumber(spec, value, dir, big)
                if (next !== null) commit(next)
              }}
            />
          ) : null}
        </div>
        <button
          type="button"
          className={cx(iconButtonClass, 'rounded-r-md')}
          aria-label="Increase"
          disabled={disabled || up === null}
          onClick={(e) => {
            const next = stepNumber(spec, value, 1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <ChevronRightIcon />
        </button>
      </div>
      {error === null ? null : (
        <p id={errorId} role="alert" className="text-xs leading-none text-(--wb-danger)">
          {error}
        </p>
      )}
    </div>
  )
}

/** The dropdown of ladder values inside the field, the nearest to the current value marked. */
function LadderMenu(props: {
  readonly values: readonly number[]
  readonly value: number
  readonly format: (value: number) => string
  readonly disabled: boolean
  readonly onPick: (value: number) => void
}): ReactNode {
  const { values, value, format, disabled, onPick } = props
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setOpen(false), [])
  useDismiss(box, open, close)
  let nearest = 0
  values.forEach((v, i) => {
    if (Math.abs(v - value) < Math.abs((values[nearest] as number) - value)) nearest = i
  })
  return (
    <div ref={box} className="flex h-full">
      <button
        type="button"
        disabled={disabled}
        aria-label="Pick a value"
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex h-full shrink-0 items-center pr-1.5 pl-0.5 text-(--wb-muted) outline-none hover:text-(--wb-fg) focus-visible:text-(--wb-fg)"
        onClick={() => setOpen((o) => !o)}
      >
        <ChevronDownIcon size={14} />
      </button>
      {open ? (
        <div role="listbox" className={cx(menuClass, 'top-full right-0')}>
          {values.map((v, i) => (
            <button
              key={v}
              type="button"
              role="option"
              aria-selected={i === nearest}
              data-selected={i === nearest}
              className={cx(menuItemClass, 'justify-end font-mono tabular-nums')}
              onClick={() => {
                onPick(v)
                setOpen(false)
              }}
            >
              {format(v)}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** The slider on the field's bottom edge: press anywhere on it to jump there, drag to follow, arrow keys to step. */
function Rail(props: {
  readonly fraction: number
  readonly disabled: boolean
  readonly label: string
  readonly valueText: string
  readonly onFraction: (t: number) => void
  readonly onStep: (dir: 1 | -1, big: boolean) => void
}): ReactNode {
  const { fraction, disabled, label, valueText, onFraction, onStep } = props
  const rail = useRef<HTMLDivElement>(null)
  const at = (e: PointerEvent): number => {
    const r = rail.current?.getBoundingClientRect()
    return r === undefined || r.width === 0 ? 0 : (e.clientX - r.left) / r.width
  }
  return (
    <div
      ref={rail}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={`${label} slider`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fraction * 100)}
      aria-valuetext={valueText}
      aria-disabled={disabled}
      className={cx(
        'group absolute inset-x-0 -bottom-[5px] z-10 h-[10px] touch-none outline-none select-none',
        disabled ? 'pointer-events-none' : 'cursor-pointer',
      )}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        onFraction(at(e))
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) onFraction(at(e))
      }}
      onKeyDown={(e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onStep(1, e.shiftKey)
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onStep(-1, e.shiftKey)
        else if (e.key === 'Home') onFraction(0)
        else if (e.key === 'End') onFraction(1)
        else return
        e.preventDefault()
      }}
    >
      <div className="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2">
        <div className="h-full bg-(--wb-accent)" style={{ width: `${fraction * 100}%` }} />
      </div>
      <div
        className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-(--wb-accent) bg-(--wb-bg) transition-shadow group-hover:ring-3 group-hover:ring-(--wb-accent)/30 group-focus-visible:ring-3 group-focus-visible:ring-(--wb-accent)/50"
        style={{ left: `${fraction * 100}%` }}
      />
    </div>
  )
}
