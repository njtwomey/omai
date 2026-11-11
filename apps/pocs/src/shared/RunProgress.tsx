import { type ReactNode, useEffect, useState } from 'react'
import { elapsedOf, going, type RunPhase, type RunState } from './run.ts'

/** The runs shorter than this show no bar (so a fast run does not flicker), in milliseconds. */
export const BAR_AFTER_MS = 400

const phaseWord: Record<RunPhase, string> = {
  idle: '',
  queued: 'queued',
  running: 'running',
  converged: 'converged',
  done: 'done',
  stopped: 'stopped',
  failed: 'failed',
}

const seconds = (ms: number): string => `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`

/**
 * What a run is doing, as a bar: a timer, the stage and its status ('running', 'sweep 41 of 300; change 3e-3, tolerance
 * 1e-4'), a progress bar of how far through it is (it jumps to full when the run converges or ends) and a Stop button that
 * ends the run where it stands and keeps what was computed. A run shorter than 0.4 s shows nothing while it goes; when
 * it has ended there is one line for a run that was stopped or failed (and for one that took 0.4 s or more and waits on a
 * button). For a run that waits for a button
 * (`auto: false`), `onStart` shows the button while it is idle.
 *
 * @param props - The run (the `useRun` result), its name, and the button that starts it.
 * @returns The bar, or nothing.
 */
export function RunProgress(props: {
  readonly run: RunState<unknown> & { readonly stop: () => void }
  /** What the run is called when it reports no stage ('NUTS reference', 'calibration'). */
  readonly label: string
  /** Starts the run: given, the idle state is a button of this name. */
  readonly onStart?: () => void
  readonly startText?: string
  /** What to say beside the start button. */
  readonly hint?: ReactNode
}): ReactNode {
  const { run, label, onStart, startText, hint } = props
  const [now, setNow] = useState(() => performance.now())
  const active = going(run.phase)
  useEffect(() => {
    if (!active) return
    setNow(performance.now())
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [active])
  const elapsed = elapsedOf(run, now)
  const p = run.progress
  const share = p === undefined || p.total <= 0 ? 0 : Math.min(1, p.done / p.total)
  const button =
    'cursor-pointer rounded-md border border-line bg-bg px-3 py-1 text-xs text-fg hover:bg-surface disabled:cursor-default disabled:opacity-50'
  if (run.phase === 'idle')
    return onStart === undefined ? null : (
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button type="button" className={button} onClick={onStart}>
          {startText ?? `Run the ${label}`}
        </button>
        {hint === undefined ? null : <span className="text-muted">{hint}</span>}
      </div>
    )
  if (!active) {
    const failed = run.phase === 'failed'
    // A run the page started itself says how it ended in its own status line; only a stop or a failure needs this one.
    if (onStart === undefined && !failed && run.phase !== 'stopped') return null
    if (elapsed < BAR_AFTER_MS && !failed && run.phase !== 'stopped') return null
    return (
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted" role="status">
        <span style={failed ? { color: 'var(--wb-danger)' } : undefined}>
          {label}: {phaseWord[run.phase]}
          {failed ? ` (${run.error ?? 'error'})` : ''} in {seconds(elapsed)}
          {run.phase === 'stopped' ? '; what was computed is kept' : ''}
        </span>
        {onStart === undefined ? null : (
          <button type="button" className={button} onClick={onStart}>
            Run again
          </button>
        )}
      </div>
    )
  }
  if (elapsed < BAR_AFTER_MS) return null
  return (
    <div
      className="flex flex-col gap-1.5 rounded-lg border border-line bg-surface p-3 text-xs"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-medium text-fg">
          {p?.stage ?? label}: {run.phase === 'queued' ? 'queued' : 'running'}
        </span>
        <span className="tabular-nums text-muted">{seconds(elapsed)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-bg" aria-hidden="true">
        <div className="h-full rounded-full bg-(--wb-accent)" style={{ width: `${100 * share}%` }} />
      </div>
      <div className="flex items-center justify-between gap-4">
        <span className="tabular-nums text-muted">
          {p?.detail ?? (run.phase === 'queued' ? 'waiting for the page' : '')}
        </span>
        <span className="flex items-center gap-3">
          <span className="tabular-nums text-muted">{Math.round(100 * share)}%</span>
          <button type="button" className={button} onClick={run.stop} title="Stop here and keep what has been computed">
            Stop
          </button>
        </span>
      </div>
    </div>
  )
}
