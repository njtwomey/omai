// The controls every proof-of-concept page shares: eager against live, the seed and step budget, and the subscription
// (off, the method's defaults, everything, or chosen paths with one schedule).
import { Log, type Subscription } from 'mdml-engine/trace'
import { Choice, ControlPanel, NumberField } from 'mdml-render/react'
import type { ReactElement } from 'react'

export interface SessionSettings {
  readonly mode: 'eager' | 'live'
  readonly seed: number
  readonly steps: number
  readonly preset: 'off' | 'defaults' | 'all' | 'custom'
  readonly chosen: readonly string[]
  readonly every: number
  /** Append the generic view of the whole tape to the page (for debugging). */
  readonly tape: boolean
}

/** Settings that start from a mode and a subscription: a preset name, or chosen paths logged at every step. */
export function initialSettings(mode: 'eager' | 'live', logs: Subscription): SessionSettings {
  const base = { mode, seed: 0, steps: 300, every: 1, tape: false }
  return typeof logs === 'string'
    ? { ...base, preset: logs, chosen: [] }
    : { ...base, preset: 'custom', chosen: Object.keys(logs) }
}

export function logsOf(s: SessionSettings): Subscription {
  if (s.preset !== 'custom') return s.preset
  return Object.fromEntries(s.chosen.map((p) => [p, Log({ every: s.every })]))
}

export function SessionControls(props: {
  readonly paths: readonly string[]
  readonly settings: SessionSettings
  readonly onChange: (s: SessionSettings) => void
}): ReactElement {
  const { paths, settings: s, onChange } = props
  return (
    <div className="flex flex-col gap-2">
      <ControlPanel columns={4}>
        <Choice label="mode" value={s.mode} options={['eager', 'live']} onChange={(mode) => onChange({ ...s, mode })} />
        <NumberField
          label="seed"
          value={s.seed}
          onChange={(seed) => onChange({ ...s, seed })}
          min={0}
          max={99}
          integer
        />
        <NumberField
          label="steps"
          value={s.steps}
          onChange={(steps) => onChange({ ...s, steps })}
          min={10}
          max={100000}
          integer
          log
        />
        <Choice
          label="logs"
          value={s.preset}
          options={['off', 'defaults', 'all', 'custom']}
          onChange={(preset) => onChange({ ...s, preset })}
        />
      </ControlPanel>
      <label className="flex items-center gap-1 text-sm">
        <input type="checkbox" checked={s.tape} onChange={(e) => onChange({ ...s, tape: e.target.checked })} />
        show the whole tape
      </label>
      {s.preset === 'custom' ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          {paths.map((p) => (
            <label key={p} className="flex items-center gap-1 font-mono">
              <input
                type="checkbox"
                checked={s.chosen.includes(p)}
                onChange={(e) =>
                  onChange({ ...s, chosen: e.target.checked ? [...s.chosen, p] : s.chosen.filter((q) => q !== p) })
                }
              />
              {p}
            </label>
          ))}
          <span className="w-40">
            <NumberField
              label="every"
              value={s.every}
              onChange={(every) => onChange({ ...s, every })}
              min={1}
              max={1000}
              integer
              log
            />
          </span>
        </div>
      ) : null}
    </div>
  )
}
