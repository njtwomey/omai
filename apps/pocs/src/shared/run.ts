/**
 * The runs of the pocs that take long enough to need a bar: one small contract between a page and the work it starts.
 * A task is an async function that is given a host, reports how far it is (`done` of `total`, with the stage it is in and
 * a line of detail), asks whether it has been stopped and hands the thread back to the page between slices of work; the
 * controller turns what it reports into a state (queued, running, converged, done, stopped, failed, the progress, the
 * time) that never goes backwards, and ends the run where it stands when it is stopped, keeping what the task returned.
 * Nothing here touches React: `useRun` is the hook, `runWorker.ts` the tasks that compute in a Web Worker, and
 * `RunProgress` the bar.
 */

/** How far a run is: `done` of `total` in the stage it is in, and what is worth saying about it. */
export interface RunProgress {
  readonly done: number
  readonly total: number
  /** What is running now ('EP', 'Gibbs reference', 'seeds'). A new stage starts its bar again. */
  readonly stage?: string
  /** A second line: the unit ('sweep 41 of 300'), or the change against the tolerance. */
  readonly detail?: string
  /** The run ended by meeting its own criterion (EP converged), so the bar goes to full. */
  readonly converged?: boolean
}

/** What a task, in the page's thread or a worker, can do: say how far it is, ask whether to stop, hand the thread back. */
export interface HostBase {
  /** Says how far the task is. The state never goes backwards within a stage. */
  readonly report: (progress: RunProgress) => void
  /** Whether the run has been stopped or replaced: the task returns what it has. */
  readonly stopped: () => boolean
  /** Resolves when the page has painted and answered what was waiting (a message to ourselves, not a timer). */
  readonly yieldToPage: () => Promise<void>
}

/** What a task is given. */
export interface RunHost<R = unknown> extends HostBase {
  /** Hands the page a result so far (the seeds done), which stays as the value if the run is stopped. */
  readonly partial: (value: R) => void
  /** Calls `handler` when the run is stopped or replaced (a worker is terminated there). */
  readonly onStop: (handler: () => void) => void
}

/** A piece of work: it returns its result, or what it has when it is stopped (undefined when it has nothing). */
export type RunTask<R> = (host: RunHost<R>) => Promise<R | undefined>

/** Where a run stands. */
export type RunPhase = 'idle' | 'queued' | 'running' | 'converged' | 'done' | 'stopped' | 'failed'

/** What the page knows of a run. */
export interface RunState<R> {
  readonly phase: RunPhase
  readonly progress: RunProgress | undefined
  /** `performance.now()` when the run was asked for, and when it ended (undefined while it goes on). */
  readonly startedAt: number
  readonly endedAt: number | undefined
  readonly value: R | undefined
  readonly error: string | undefined
}

/** A handle on a run. */
export interface RunHandle {
  /** Ends the run where it stands: the task is told, and returns what it has. */
  readonly stop: () => void
  /** Throws the run away: it is told to stop and nothing more is reported. */
  readonly cancel: () => void
}

/** The state of a run that is not going. */
export const idle = <R>(): RunState<R> => ({
  phase: 'idle',
  progress: undefined,
  startedAt: 0,
  endedAt: undefined,
  value: undefined,
  error: undefined,
})

/** Whether the run is on its way (asked for or computing). */
export const going = (phase: RunPhase): boolean => phase === 'queued' || phase === 'running'

/** Milliseconds a run has taken (or has been going, against `now`). */
export const elapsedOf = (state: RunState<unknown>, now: number): number =>
  Math.max(0, (state.endedAt ?? now) - state.startedAt)

/** A message to ourselves: the page paints and answers clicks before it arrives, and a background tab does not stretch the wait. */
export const yieldToPage = (): Promise<void> =>
  new Promise((resolve) => {
    const channel = new MessageChannel()
    channel.port1.onmessage = () => {
      channel.port1.close()
      resolve()
    }
    channel.port2.postMessage(null)
  })

/** What `startRun` is given besides the task. */
export interface StartOptions {
  /** How the task hands the thread back (default {@link yieldToPage}; a test gives `() => Promise.resolve()`). */
  readonly yieldToPage?: () => Promise<void>
  /** The clock in milliseconds (default `performance.now`). */
  readonly now?: () => number
}

/**
 * Starts a task and reports its state as it changes: queued at once, running when the page has had the thread once, then
 * each report of the task, and at the end converged or done (the bar full), stopped (what the task returned kept) or
 * failed (the error's message). The progress of a stage never goes backwards and never passes its total.
 *
 * @param task - The work.
 * @param onChange - Called with every new state.
 * @param options - The way to hand the thread back and the clock, for tests.
 * @returns The handle that stops or cancels the run.
 */
export function startRun<R>(
  task: RunTask<R>,
  onChange: (state: RunState<R>) => void,
  options: StartOptions = {},
): RunHandle {
  const now = options.now ?? (() => performance.now())
  const hand = options.yieldToPage ?? yieldToPage
  const startedAt = now()
  let alive = true
  let stopRequested = false
  let state: RunState<R> = { ...idle<R>(), phase: 'queued', startedAt }
  const handlers: (() => void)[] = []
  const emit = (patch: Partial<RunState<R>>): void => {
    state = { ...state, ...patch }
    if (alive) onChange(state)
  }
  const report = (p: RunProgress): void => {
    if (!alive || !going(state.phase)) return
    const before = state.progress
    const total = Math.max(0, p.total)
    const same = before !== undefined && before.stage === p.stage
    const done = Math.min(total, Math.max(0, same ? Math.max(before.done, p.done) : p.done))
    emit({ phase: 'running', progress: { ...p, done, total } })
  }
  const host: RunHost<R> = {
    report,
    partial: (value) => {
      if (alive && going(state.phase)) emit({ value })
    },
    stopped: () => stopRequested || !alive,
    yieldToPage: hand,
    onStop: (handler) => {
      if (stopRequested || !alive) handler()
      else handlers.push(handler)
    },
  }
  const end = (): void => {
    for (const h of handlers.splice(0)) h()
  }
  emit({})
  void (async () => {
    try {
      // Queued until the page has had the thread once: the row is drawn before the work starts.
      await hand()
      if (!alive) return
      if (stopRequested) {
        emit({ phase: 'stopped', endedAt: now() })
        return
      }
      emit({ phase: 'running' })
      const value = await task(host)
      const p = state.progress
      const finished = !stopRequested && alive
      emit({
        phase: stopRequested ? 'stopped' : p?.converged === true ? 'converged' : 'done',
        endedAt: now(),
        value: value ?? state.value,
        progress: finished && p !== undefined ? { ...p, done: p.total } : p,
      })
    } catch (e) {
      emit({ phase: 'failed', endedAt: now(), error: e instanceof Error ? e.message : String(e) })
    }
  })()
  return {
    stop: () => {
      if (!going(state.phase)) return
      stopRequested = true
      end()
    },
    cancel: () => {
      alive = false
      end()
    },
  }
}
