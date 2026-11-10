import { Choice } from 'omai-render/react'
import { type ReactNode, useEffect, useState } from 'react'
import type { AnytimeRuns } from './useAnytimeRuns.ts'

/** The budgets on offer, in seconds. */
export const BUDGETS: readonly number[] = [3, 10, 30]

const phaseText = {
  queued: 'waiting',
  running: 'running',
  done: 'finished',
  budget: 'budget spent',
  stopped: 'stopped',
  failed: 'failed',
} as const

const button =
  'cursor-pointer rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'

/**
 * The controls of an anytime page: the time budget as a choice, stop and run again, a bar of the budget used, and a line
 * for each run (its phase, the steps of the method taken, the time). The page's runs are the `useAnytimeRuns` result.
 *
 * @param props - The runs, the budget in seconds and its setter, and the budgets on offer.
 * @returns The bar.
 */
export function RunBar<T>(props: {
  readonly runs: AnytimeRuns<T>
  readonly budget: number
  readonly onBudget: (seconds: number) => void
  readonly budgets?: readonly number[]
  /** The word for a step of the runs (default `steps`). */
  readonly unit?: string
}): ReactNode {
  const { runs, budget, onBudget, budgets = BUDGETS, unit = 'steps' } = props
  const [now, setNow] = useState(() => performance.now())
  const going = runs.running
  useEffect(() => {
    if (!going) return
    const timer = setInterval(() => setNow(performance.now()), 200)
    return () => clearInterval(timer)
  }, [going])
  const start = runs.clock?.start
  const elapsed = start === undefined ? 0 : ((runs.clock?.end ?? now) - start) / 1000
  const share = Math.min(1, elapsed / budget)
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-3">
      <div className="flex flex-wrap items-end gap-3">
        <Choice
          label="time budget"
          value={budget}
          options={budgets.map((b) => ({ value: b, label: `${b} s` }))}
          onChange={onBudget}
        />
        <div className="flex gap-2">
          <button type="button" className={button} disabled={!going} onClick={runs.stop}>
            Stop
          </button>
          <button type="button" className={button} disabled={going} onClick={runs.restart}>
            Run again
          </button>
        </div>
        <div className="flex min-w-40 flex-1 flex-col gap-1">
          <div className="flex justify-between text-xs tabular-nums text-muted">
            <span>{going ? 'running' : 'finished'}</span>
            <span>
              {elapsed.toFixed(1)} of {budget} s
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-bg">
            <div className="h-full rounded-full bg-(--wb-accent)" style={{ width: `${100 * (going ? share : 1)}%` }} />
          </div>
        </div>
      </div>
      <ul className="m-0 grid list-none gap-x-6 gap-y-1 p-0 text-xs text-muted sm:grid-cols-2 lg:grid-cols-3">
        {runs.runs.map((r) => (
          <li key={r.id} className="flex justify-between gap-3 tabular-nums">
            <span className="text-fg">{r.label}</span>
            <span title={r.error}>
              {phaseText[r.phase]}
              {r.t > 0 ? ` · ${r.t} ${unit}` : ''}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
