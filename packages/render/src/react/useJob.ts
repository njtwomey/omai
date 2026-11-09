/**
 * `useJob`: a heavy one-shot computation in named stages, with progress, that starts only when it is active. Each stage
 * takes the previous stage's result; the hook yields to the page between stages (and between slices of a generator
 * stage) so a progress bar paints, discards the results of a job that its inputs have replaced, and keeps the stage
 * readable while it runs.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

/** What a stage's `run` is given besides the previous result. */
export interface JobContext {
  /** The stage's index. */
  readonly stage: number
  /** The stage's name. */
  readonly name: string
  /** Report how far this stage is (0 to 1), for a stage that awaits; a generator stage yields the fraction instead. */
  readonly progress: (fraction: number) => void
  /** Whether the job was replaced or unmounted: an awaiting stage can stop early. */
  readonly cancelled: () => boolean
}

/**
 * One stage of a {@link useJob} job. `run(prev, ctx)` gets the previous stage's result (`undefined` for the first) and
 * returns this stage's: a value, a promise of one, or a generator (`function*`) that yields its progress as a fraction
 * from 0 to 1 and returns the result, which the hook runs in slices of a few milliseconds per frame so a long stage
 * reports finer progress and the page paints between slices. A stage that neither awaits nor yields blocks the main
 * thread until it returns: only a worker keeps the page responsive then (see `workerProducer`).
 */
export interface JobStage {
  /** The name shown while the stage runs. */
  readonly name: string
  /** Share of the bar this stage takes relative to the others (default 1). */
  readonly weight?: number
  /** The work. */
  readonly run: (prev: never, ctx: JobContext) => unknown
}

/** Options of {@link useJob}. */
export interface JobOptions {
  /** The stages, in order. The list at the moment a job starts is the one it runs. */
  readonly stages: readonly JobStage[]
  /** Values the job depends on: a change of any (by `Object.is`) discards the running job and starts a new one. */
  readonly inputs?: readonly unknown[]
  /** Whether the job may run (default true). While false nothing runs and no timer exists. */
  readonly active?: boolean
  /** How long a generator stage runs before it gives the page a frame, in milliseconds (default 8). */
  readonly sliceMs?: number
}

/** Where a job is: not started, running, finished with a result, or failed. */
export type JobPhase = 'idle' | 'running' | 'done' | 'error'

/** What {@link useJob} returns. */
export interface Job<R> {
  readonly phase: JobPhase
  /** The index of the running stage (the last one when done, the failed one on error, -1 while idle). */
  readonly stage: number
  /** The name of that stage (empty while idle). */
  readonly stageName: string
  /** How many stages there are. */
  readonly count: number
  /** How far the whole job is, 0 to 1 (1 when done), counting stages by their weights. */
  readonly progress: number
  /** The last stage's result once `done`. */
  readonly result: R | undefined
  /** What a stage threw, once `error`. */
  readonly error: unknown
  /** Run the job again from the first stage. */
  readonly restart: () => void
}

interface State {
  readonly phase: JobPhase
  readonly stage: number
  readonly progress: number
  readonly result: unknown
  readonly error: unknown
}

const IDLE: State = { phase: 'idle', stage: -1, progress: 0, result: undefined, error: undefined }

/** The token of one job: set when it is replaced, with what to undo then (pending timers, a generator). */
interface Token {
  cancelled: boolean
  readonly undo: Set<() => void>
}

function cancel(token: Token): void {
  token.cancelled = true
  for (const undo of token.undo) undo()
  token.undo.clear()
}

// Resolve once the page has painted what was just set: the animation frame runs before the paint, the timer after.
// A timer alone covers a page whose frames are throttled (a hidden tab). Undone (and resolved) when the job is cancelled.
function afterPaint(token: Token): Promise<void> {
  return new Promise((resolve) => {
    let frame: number | undefined
    let after: ReturnType<typeof setTimeout> | undefined
    let fallback: ReturnType<typeof setTimeout> | undefined
    const finish = (): void => {
      token.undo.delete(finish)
      if (frame !== undefined) cancelAnimationFrame(frame)
      clearTimeout(after)
      clearTimeout(fallback)
      resolve()
    }
    token.undo.add(finish)
    if (typeof requestAnimationFrame === 'function')
      frame = requestAnimationFrame(() => (after = setTimeout(finish, 0)))
    fallback = setTimeout(finish, 100)
  })
}

const isGenerator = (v: unknown): v is Generator<number, unknown, undefined> =>
  typeof v === 'object' &&
  v !== null &&
  typeof (v as { next?: unknown }).next === 'function' &&
  typeof (v as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function'

const clamp = (f: unknown): number => (typeof f === 'number' && f > 0 ? Math.min(1, f) : 0)

async function execute(
  stages: readonly JobStage[],
  token: Token,
  sliceMs: number,
  set: (state: State) => void,
): Promise<void> {
  const weights = stages.map((s) => (s.weight !== undefined && s.weight > 0 ? s.weight : 1))
  const total = weights.reduce((a, b) => a + b, 0)
  const before = (i: number, fraction: number): number =>
    (weights.slice(0, i).reduce((a, b) => a + b, 0) + fraction * (weights[i] ?? 0)) / total
  let prev: unknown
  let at = 0
  try {
    for (const [i, stage] of stages.entries()) {
      at = i
      if (token.cancelled) return
      set({ ...IDLE, phase: 'running', stage: i, progress: before(i, 0) })
      // The stage's name and an empty bar are on the page before its work begins.
      await afterPaint(token)
      if (token.cancelled) return
      const ctx: JobContext = {
        stage: i,
        name: stage.name,
        progress: (f) => {
          if (!token.cancelled) set({ ...IDLE, phase: 'running', stage: i, progress: before(i, clamp(f)) })
        },
        cancelled: () => token.cancelled,
      }
      const out = stage.run(prev as never, ctx)
      if (isGenerator(out)) {
        const stop = (): void => void out.return(undefined)
        token.undo.add(stop)
        let t0 = performance.now() // allow: the slice of a frame given to a stage; not on the tape
        for (;;) {
          const step = out.next()
          if (step.done === true) {
            prev = step.value
            break
          }
          if (performance.now() - t0 < sliceMs) continue // allow: as above
          ctx.progress(step.value)
          await afterPaint(token)
          if (token.cancelled) return
          t0 = performance.now() // allow: as above
        }
        token.undo.delete(stop)
      } else {
        prev = await out
      }
      // A result that arrives after the job was replaced is dropped.
      if (token.cancelled) return
    }
    set({ phase: 'done', stage: Math.max(0, stages.length - 1), progress: 1, result: prev, error: undefined })
  } catch (error) {
    if (!token.cancelled) set({ ...IDLE, phase: 'error', stage: at, progress: before(at, 0), error })
  }
}

const sameInputs = (a: readonly unknown[], b: readonly unknown[]): boolean =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]))

/**
 * A heavy one-shot computation in named stages, with progress, that starts only when it is `active`. A stage's `run`
 * takes the previous stage's result and returns the next (a value, a promise, or a generator that yields its progress
 * from 0 to 1 and is run in slices of `sliceMs` per frame); the hook waits for the page to paint between stages, so the
 * stage's name and bar show before its work begins, and `stage` and `stageName` are always readable while it runs (a
 * `useLiveRun` shows its step-0 state while it prepares, which cannot say what it is doing).
 *
 * - Nothing starts while `active` is false, and no timer or frame request exists then. Deactivating a running job
 *   cancels it (it starts over when active again); a finished job keeps its result.
 * - A change of `inputs` (by `Object.is`) starts a new job and clears the result; whatever the old job delivers late is
 *   discarded. `restart()` runs the job again. Unmounting cancels.
 * - The result is one value held by the hook, set once when the last stage returns, never a state per step.
 * - A stage that neither awaits nor yields blocks the main thread until it returns, however many stages there are: the
 *   page paints between stages, not inside one. Only a worker keeps the page responsive through a single heavy
 *   stage (`workerProducer` makes one for a session); the hook does not make workers.
 *
 * @param options - The stages, the inputs the job depends on, `active` and the slice length.
 * @returns The phase, the running stage and its name, the progress, the result, the error and `restart`.
 * @index function React hook: a staged heavy computation with progress, started when active: `useJob({ stages, inputs, active })`
 * @example
 * ```tsx
 * import { useJob } from 'omai-render/react'
 * function Surface({ n, shown }: { n: number; shown: boolean }) {
 *   const job = useJob<number[]>({
 *     active: shown,
 *     inputs: [n],
 *     stages: [
 *       { name: 'sampling', run: () => Array.from({ length: n }, (_, i) => i) },
 *       { name: 'sorting', run: (xs: number[]) => [...xs].sort((a, b) => b - a) },
 *     ],
 *   })
 *   return job.phase === 'done' ? <p>{job.result?.length}</p> : <progress value={job.progress} aria-label={job.stageName} />
 * }
 * ```
 */
export function useJob<R = unknown>(options: JobOptions): Job<R> {
  const { inputs = [], active = true, sliceMs = 8 } = options
  const [state, setState] = useState<State>(IDLE)
  const [run, setRun] = useState(0)
  const latest = useRef(options)
  latest.current = options
  // What the finished job was made for, so a later activation or render does not run it again.
  const finished = useRef<{ readonly inputs: readonly unknown[]; readonly run: number } | undefined>(undefined)
  const restart = useCallback((): void => setRun((n) => n + 1), [])

  // biome-ignore lint/correctness/useExhaustiveDependencies: the inputs are the caller's dependency list, and the stages are read when the job starts
  useEffect(() => {
    const done = finished.current
    if (done !== undefined && done.run === run && sameInputs(done.inputs, inputs)) return
    finished.current = undefined
    if (!active) {
      setState((s) => (s === IDLE ? s : IDLE))
      return
    }
    const token: Token = { cancelled: false, undo: new Set() }
    const stages = latest.current.stages
    const set = (next: State): void => {
      if (token.cancelled) return
      if (next.phase === 'done') finished.current = { inputs, run }
      setState(next)
    }
    void execute(stages, token, sliceMs, set)
    return () => cancel(token)
  }, [active, run, sliceMs, ...inputs])

  const stages = options.stages
  return {
    phase: state.phase,
    stage: state.stage,
    stageName: stages[state.stage]?.name ?? '',
    count: stages.length,
    progress: state.progress,
    result: state.result as R | undefined,
    error: state.error,
    restart,
  }
}
