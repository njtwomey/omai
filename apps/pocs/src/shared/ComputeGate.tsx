import { type ReactNode, useEffect, useRef, useState } from 'react'

/** What the gate needs of a live session: how far it has computed, whether it is still at it, and whether it has ended. */
export interface Computing {
  readonly head: number
  readonly computing: boolean
  readonly done: boolean
}

/** How long is left, in words: from the steps done and the time they took. */
function remaining(done: number, total: number, elapsedMs: number): string {
  if (done < 2 || elapsedMs < 600) return 'estimating the time left…'
  const left = ((total - done) * elapsedMs) / done / 1000
  if (left < 1) return 'under a second left'
  if (left < 90) return `about ${Math.round(left)} s left`
  return `about ${Math.round(left / 60)} min left`
}

/**
 * Holds a view back while a session computes: the view is not drawn at all (drawing it at every step is what makes
 * playing slow), and in its place, over the whole area, a progress bar with the step reached and an estimate of the
 * time left. When the run ends the view appears at the final step, to scrub through.
 *
 * @param props - The session's progress, the number of steps it will take, and the view.
 * @returns The view, or the progress in its place.
 */
export function ComputeGate(props: {
  readonly live: Computing
  readonly total: number
  readonly label?: string
  /** What the left of the second line says (default the step reached of the total). */
  readonly detail?: string
  /** Stop where it is and keep what has been computed: given, the gate shows a Cancel button. */
  readonly onCancel?: () => void
  readonly children: ReactNode
}): ReactNode {
  const { live, total, label = 'Training', detail, onCancel, children } = props
  const wrapper = useRef<HTMLDivElement>(null)
  const height = useRef(0)
  const started = useRef({ at: 0, head: 0 })
  const [now, setNow] = useState(0)
  const gated = live.computing && !live.done
  // The area keeps its size while the view is away, so the page does not jump.
  if (!gated && wrapper.current !== null) height.current = wrapper.current.getBoundingClientRect().height
  // The clock starts when the computing does; the head is read as it moves.
  // biome-ignore lint/correctness/useExhaustiveDependencies: restart the clock only when computing starts
  useEffect(() => {
    if (!gated) return
    started.current = { at: performance.now(), head: live.head }
    setNow(performance.now())
    const timer = setInterval(() => setNow(performance.now()), 250)
    return () => clearInterval(timer)
  }, [gated])
  if (!gated) return <div ref={wrapper}>{children}</div>
  const share = Math.min(1, live.head / Math.max(1, total))
  const text = remaining(live.head - started.current.head, total - started.current.head, now - started.current.at)
  return (
    <div
      ref={wrapper}
      className="flex items-center justify-center rounded-lg border border-line bg-surface p-8"
      style={{ minHeight: Math.max(240, height.current) }}
      role="status"
      aria-live="polite"
    >
      <div className="flex w-full max-w-md flex-col gap-3">
        <div className="flex items-baseline justify-between gap-4 text-sm">
          <span className="font-medium text-fg">{label}…</span>
          <span className="tabular-nums text-muted">{Math.round(100 * share)}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-bg">
          <div className="h-full rounded-full bg-(--wb-accent)" style={{ width: `${100 * share}%` }} />
        </div>
        <div className="flex justify-between text-xs text-muted">
          <span className="tabular-nums">{detail ?? `step ${live.head} of ${total}`}</span>
          <span>{text}</span>
        </div>
        {onCancel === undefined ? null : (
          <button
            type="button"
            onClick={onCancel}
            title="Stop here and keep what has been computed so far"
            className="mt-1 w-fit cursor-pointer self-end rounded-md border border-line bg-bg px-3 py-1.5 text-xs text-fg hover:bg-surface"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  )
}
