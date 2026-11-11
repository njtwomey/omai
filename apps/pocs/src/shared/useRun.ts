import { useCallback, useEffect, useRef, useState } from 'react'
import { going, idle, type RunHandle, type RunState, type RunTask, startRun } from './run.ts'

/** What `useRun` gives a page: the state of the run and its two buttons. */
export interface UseRun<R> extends RunState<R> {
  /** Whether the run is on its way (queued or running). */
  readonly going: boolean
  /** Starts the run (again, if it has ended): what `auto: false` waits for. */
  readonly start: () => void
  /** Ends the run where it stands, keeping what the task returned. */
  readonly stop: () => void
}

/** How often the page hears of progress, in milliseconds (a change of phase is heard at once). */
const EVERY_MS = 80

/**
 * Runs a task for a page and keeps its state: queued, running with the progress it reports, and at the end converged, done,
 * stopped (the value it had kept) or failed. A new `task` (give it a stable identity: a `useCallback` of the knobs, as for
 * `useDeferred`) cancels the one running and starts the new, unless `auto` is false, when the run waits for `start()`.
 * The task computes in slices that hand the thread back, or in a worker (`runWorker.ts`), so the page keeps painting and
 * the Stop button answers.
 *
 * @param task - The work, or `undefined` while the page is not ready for it.
 * @param options - `auto: false` for a run that starts on a button.
 * @returns The state, `start` and `stop`.
 */
export function useRun<R>(task: RunTask<R> | undefined, options: { readonly auto?: boolean } = {}): UseRun<R> {
  const auto = options.auto ?? true
  const [state, setState] = useState<RunState<R>>(() => idle<R>())
  const handle = useRef<RunHandle | undefined>(undefined)
  const begin = useCallback((): void => {
    handle.current?.cancel()
    handle.current = undefined
    if (task === undefined) {
      setState(idle<R>())
      return
    }
    let latest: RunState<R> | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    let phase = '' as string
    const flush = (): void => {
      timer = undefined
      if (latest !== undefined) setState(latest)
    }
    const h = startRun<R>(task, (s) => {
      latest = s
      const changed = s.phase !== phase
      phase = s.phase
      if (changed) {
        if (timer !== undefined) clearTimeout(timer)
        flush()
      } else if (timer === undefined) timer = setTimeout(flush, EVERY_MS)
    })
    handle.current = {
      stop: h.stop,
      cancel: () => {
        if (timer !== undefined) clearTimeout(timer)
        h.cancel()
      },
    }
  }, [task])
  useEffect(() => {
    if (auto) begin()
    else {
      handle.current?.cancel()
      handle.current = undefined
      setState(idle<R>())
    }
    return () => {
      handle.current?.cancel()
      handle.current = undefined
    }
  }, [auto, begin])
  const stop = useCallback((): void => handle.current?.stop(), [])
  return { ...state, going: going(state.phase), start: begin, stop }
}
