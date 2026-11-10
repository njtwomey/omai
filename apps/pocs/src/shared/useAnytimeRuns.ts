import { type AnytimeResult, runAnytimeAsync, type Snapshot } from 'omai-engine/probprog/inference'
import { type Config, type Method, type Session, session } from 'omai-engine/trace'
import { useCallback, useEffect, useRef, useState } from 'react'

/** One run of an inference method on a page: what to run, and what the page keeps of it at each snapshot. */
export interface AnytimeRunSpec<T> {
  readonly id: string
  readonly label: string
  // biome-ignore lint/suspicious/noExplicitAny: a list of methods is generic over their states and paths
  readonly method: Method<any, any, any>
  /** Knobs over the method's defaults. */
  readonly config?: Config
  readonly seed?: number
  /** What the session records on the tape (`'all'` for the message passing and sampling paths a record reads). */
  readonly logs?: 'all' | 'defaults'
  /** The page's reading of the live state at a snapshot (a posterior, a record, rows): taken now, kept as it is. */
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly derive: (s: Session<any, any, any>, snapshot: Snapshot<unknown>) => T
}

/** Where one run stands. */
export interface AnytimeRunView<T> {
  readonly id: string
  readonly label: string
  readonly phase: 'queued' | 'running' | 'done' | 'budget' | 'stopped' | 'failed'
  /** Steps of the method taken. */
  readonly t: number
  /** Milliseconds since the run started. */
  readonly elapsedMs: number
  /** What `derive` made at the latest snapshot (undefined until the first). */
  readonly value: T | undefined
  /** Every snapshot so far, with the step and the time it was taken at (for a record that grows). */
  readonly history: readonly { readonly t: number; readonly elapsedMs: number; readonly value: T }[]
  readonly error: string | undefined
}

/** What the runs of a page do and where they stand. */
export interface AnytimeRuns<T> {
  readonly runs: readonly AnytimeRunView<T>[]
  /** Whether any run is still going. */
  readonly running: boolean
  /** When the runs started on `performance.now`, and when the last ended (undefined while one is going). */
  readonly clock: { readonly start: number; readonly end: number | undefined } | undefined
  /** Stop every run where it is (what it has computed stays). */
  readonly stop: () => void
  /** Run them all again from the start. */
  readonly restart: () => void
}

const EARLY_MS = 250
const SNAPSHOT_MS = 1000
const SLICE_MS = 40

// A message to ourselves: the page paints and answers clicks before it arrives, and, unlike a timer, a tab in the
// background does not stretch the wait to a second.
const yieldToPage = (): Promise<void> =>
  new Promise((resolve) => {
    const channel = new MessageChannel()
    channel.port1.onmessage = () => {
      channel.port1.close()
      resolve()
    }
    channel.port2.postMessage(null)
  })

/**
 * Runs inference methods on the page without blocking it: every run is a session driven by the library's
 * `runAnytimeAsync` (slices that hand the thread back, so the page paints and answers clicks), all of them at once and
 * sharing the thread, within a time budget the user chooses. A snapshot of each run is read with its `derive` about once
 * a second (the first after a quarter of a second, so the page has something to show), and the last when the run ends,
 * the budget is spent or it is stopped. A change of `specs` or `budgetMs` starts over; the runs of the old ones stop.
 *
 * @param specs - What to run, or `undefined` while the page is not ready (a truth still being computed).
 * @param budgetMs - The time budget of the runs, in milliseconds of the clock they share.
 * @returns The runs' states, whether they are going, and stop and restart.
 */
export function useAnytimeRuns<T>(specs: readonly AnytimeRunSpec<T>[] | undefined, budgetMs: number): AnytimeRuns<T> {
  const [runs, setRuns] = useState<readonly AnytimeRunView<T>[]>([])
  const [clockInfo, setClockInfo] = useState<AnytimeRuns<T>['clock']>(undefined)
  const [round, setRound] = useState(0)
  const stopped = useRef(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new set of specs, budget or restart starts a new round
  useEffect(() => {
    if (specs === undefined) {
      setRuns([])
      return
    }
    let alive = true
    stopped.current = false
    const clock = (): number => performance.now()
    const t0 = clock()
    const view = (i: number, patch: Partial<AnytimeRunView<T>>): void => {
      if (!alive) return
      setRuns((old) => old.map((r, k) => (k === i ? { ...r, ...patch } : r)))
    }
    setRuns(
      specs.map((s) => ({
        id: s.id,
        label: s.label,
        phase: 'queued' as const,
        t: 0,
        elapsedMs: 0,
        value: undefined,
        history: [],
        error: undefined,
      })),
    )
    setClockInfo({ start: t0, end: undefined })
    const shouldStop = (): boolean => !alive || stopped.current
    const drive = async (spec: AnytimeRunSpec<T>, i: number): Promise<void> => {
      try {
        // Pending until the first slice: lets every run's row be drawn before the work starts.
        await yieldToPage()
        const s = session(spec.method, {
          ...(spec.config === undefined ? {} : { config: spec.config }),
          seed: spec.seed ?? 1,
          steps: 1_000_000_000,
          mode: 'live',
          ...(spec.logs === undefined ? {} : { logs: spec.logs }),
        })
        const history: { t: number; elapsedMs: number; value: T }[] = []
        const take = (x: Snapshot<unknown>): void => {
          const value = spec.derive(s, x)
          history.push({ t: x.t, elapsedMs: clock() - t0, value })
          view(i, { phase: 'running', t: x.t, elapsedMs: clock() - t0, value, history: [...history] })
        }
        // The first snapshot comes early, the rest every second.
        let result: AnytimeResult = await runAnytimeAsync(s, {
          budget: { ms: Math.min(EARLY_MS, budgetMs) },
          snapshot: { everyMs: SNAPSHOT_MS },
          clock,
          sliceMs: SLICE_MS,
          yieldToPage,
          shouldStop,
          onSnapshot: take,
        })
        const left = budgetMs - (clock() - t0)
        if (result.reason === 'budget' && left > 1)
          result = await runAnytimeAsync(s, {
            budget: { ms: left },
            snapshot: { everyMs: SNAPSHOT_MS },
            clock,
            sliceMs: SLICE_MS,
            yieldToPage,
            shouldStop,
            onSnapshot: take,
          })
        view(i, {
          phase: result.reason === 'ended' ? 'done' : result.reason === 'budget' ? 'budget' : 'stopped',
          t: result.t,
          elapsedMs: clock() - t0,
        })
      } catch (e) {
        view(i, { phase: 'failed', error: e instanceof Error ? e.message : String(e) })
      }
    }
    void Promise.all(specs.map(drive)).then(() => {
      if (alive) setClockInfo({ start: t0, end: clock() })
    })
    return () => {
      alive = false
    }
  }, [specs, budgetMs, round])
  const stop = useCallback(() => {
    stopped.current = true
  }, [])
  const restart = useCallback(() => setRound((r) => r + 1), [])
  return {
    runs,
    running: runs.some((r) => r.phase === 'queued' || r.phase === 'running'),
    clock: clockInfo,
    stop,
    restart,
  }
}
