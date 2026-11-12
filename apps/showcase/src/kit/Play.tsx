import type { ReactNode } from 'react'
import type { StoryRun } from './useStoryRun.ts'

const BUTTON =
  'inline-flex items-center gap-2.5 rounded-full border border-line bg-bg px-5 py-2.5 font-sans text-[0.95rem] font-medium ' +
  'text-fg transition-colors hover:border-accent hover:bg-surface focus-visible:outline-2 focus-visible:outline-offset-2 ' +
  'focus-visible:outline-accent'

function PlayGlyph(): ReactNode {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" className="text-accent">
      <path d="M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5z" fill="currentColor" />
    </svg>
  )
}

function RepeatGlyph(): ReactNode {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5" />
    </svg>
  )
}

/**
 * The play button of a story: the only thing on the page that starts the computation. Idle, it is a button with a label and
 * what to expect ('about 3 seconds'); running, a calm indicator and the sentence the worker last said (the status is a live
 * region, so a screen reader hears it); done, how long it took and a 'Run again' that re-rolls the seed; failed, the message and
 * a retry. It carries no percentage: the worker says what it is doing, not how far it is.
 *
 * @param props - The run (from `useStoryRun`), the label of the button, and what the reader should expect of the wait.
 * @returns The control.
 */
export function Play<R>(props: {
  readonly run: StoryRun<R>
  /** What the button says in the idle state ('Fit the line'). */
  readonly label: string
  /** How long it takes, in the reader's terms ('about three seconds'). */
  readonly estimate: string
  /** What it does behind the page (default 'It runs in the background.'). */
  readonly note?: string
}): ReactNode {
  const { run, label, estimate, note = 'It runs in the background.' } = props
  const { phase } = run
  const seconds = run.elapsedMs === undefined ? '' : `${(run.elapsedMs / 1000).toFixed(run.elapsedMs < 9500 ? 1 : 0)} s`
  return (
    <div
      className="story-block flex flex-col justify-center rounded-xl border border-line bg-surface px-5 py-4 font-sans"
      // the block keeps one height in every phase, so nothing around it moves when the phase changes
      style={{ minHeight: '6.9rem' }}
      data-phase={phase}
      data-testid="play"
    >
      {phase === 'idle' ? (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <button type="button" onClick={run.start} className={BUTTON}>
            <PlayGlyph />
            {label}
          </button>
          <p className="m-0 text-sm text-muted">
            Takes {estimate}. {note}
          </p>
        </div>
      ) : null}
      {phase === 'running' ? (
        <div className="flex items-center gap-4" role="status" aria-live="polite">
          <span className="flex shrink-0 items-center gap-1.5" aria-hidden="true">
            <span className="story-dot" />
            <span className="story-dot" />
            <span className="story-dot" />
          </span>
          <p className="m-0 min-h-[1.5em] text-[0.95rem] text-fg">
            {run.status === '' ? 'Starting the worker' : run.status}
          </p>
        </div>
      ) : null}
      {phase === 'done' ? (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <button type="button" onClick={run.again} className={BUTTON}>
            <RepeatGlyph />
            Run again
          </button>
          <p className="m-0 text-sm text-muted">
            Done in {seconds}. Running again uses a new seed, so the numbers below change a little.
          </p>
        </div>
      ) : null}
      {phase === 'error' ? (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2" role="alert">
          <button type="button" onClick={run.start} className={BUTTON}>
            <RepeatGlyph />
            Try again
          </button>
          <p className="m-0 text-sm text-danger">It did not finish: {run.error}</p>
        </div>
      ) : null}
    </div>
  )
}
