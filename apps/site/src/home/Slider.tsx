import type { ReactNode } from 'react'

/** A labelled range input in the page's colours: the controls under a live figure. */
export function Slider(props: {
  readonly label: string
  readonly value: number
  readonly min: number
  readonly max: number
  readonly step: number
  readonly onChange: (value: number) => void
  readonly format?: (value: number) => string
}): ReactNode {
  const { label, value, min, max, step, onChange, format = (v) => String(v) } = props
  return (
    <label className="flex items-center gap-3 text-sm text-muted">
      <span className="w-24 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-0 flex-1 accent-[var(--wb-fg)]"
      />
      <span className="w-12 text-right font-mono text-xs text-fg tabular-nums">{format(value)}</span>
    </label>
  )
}
