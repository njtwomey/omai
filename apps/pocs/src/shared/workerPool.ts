/**
 * The pool of the pocs that compute in Web Workers: the requests of a run (each names its `method`, the way of inferring
 * it runs) go to workers, one worker per request, at most `min(requests, cores - 1, 4)` at a time, each through the
 * producer's `workerStage` (which makes the worker, hands it the request, resolves with its result and terminates it when
 * the run is cancelled). The page keeps one status line per request (queued, running with its step and time, finishing,
 * done) and nothing else of a run in progress: it computes nothing and draws nothing until the results are in. Where there
 * are no Workers (a test) the requests run on the main thread one after the other through the poc's own `run`. The
 * worker's side is {@link serveJob}: a poc's `worker.ts` is `serveJob(self, run)`.
 */

import { type StageContext, type StageWorker, serveStage, workerStage } from 'omai-render/producer'
import { yieldToPage } from './run.ts'

/** What a worker says while it works. */
export interface JobStatus {
  readonly type: 'status'
  readonly phase: 'running' | 'finishing'
  readonly step: number
  readonly elapsedMs: number
}

/** What a request names, and what a result reports of itself. */
export interface PoolRequest {
  readonly method: string
}
export interface PoolResult {
  readonly steps: number
  readonly elapsedMs: number
}

/** What a poc's `run` is given by its host: a clock, a way to hand the thread back, whether to stop and where to report. */
export interface JobHost {
  readonly now: () => number
  readonly yieldToPage: () => Promise<void>
  readonly stopped: () => boolean
  readonly status: (status: JobStatus) => void
}

/** A poc's job: one request answered with one result. */
export type Job<Q, R> = (request: Q, host: JobHost) => Promise<R>

/** Where one method stands. */
export interface StatusLine {
  readonly id: string
  readonly label: string
  readonly phase: 'queued' | 'running' | 'finishing' | 'done' | 'stopped' | 'failed'
  readonly step: number
  readonly elapsedMs: number
  readonly error: string | undefined
}

/** What the page knows of a run. */
export interface PoolState<R> {
  readonly lines: readonly StatusLine[]
  readonly results: Readonly<Record<string, R>>
  readonly running: boolean
  /** The workers the run uses at once (0 where the methods run on the main thread). */
  readonly workers: number
  readonly startedAt: number
  readonly endedAt: number | undefined
}

/** What {@link runPool} needs from its host. */
export interface PoolOptions<Q, R> {
  /** Makes a worker of the poc's `worker.ts`; absent, the requests run on this thread through `run`. */
  readonly createWorker?: () => StageWorker
  /** The cores to leave alone (default: the page's own thread, one). */
  readonly cores?: number
  /** The poc's job, for where there are no Workers. */
  readonly run: Job<Q, R>
  readonly onChange: (state: PoolState<R>) => void
}

/** A handle on a run. */
export interface PoolHandle {
  /** End every run where it stands: the running methods return what they have, the queued ones do not start. */
  readonly stop: () => void
  /** Throw the run away: the workers are terminated and nothing more is reported. */
  readonly cancel: () => void
}

const isStatus = (x: unknown): x is JobStatus =>
  typeof x === 'object' && x !== null && (x as { readonly type?: unknown }).type === 'status'

/** The number of workers a run of `methods` methods uses: at most `min(methods, cores - 1, 4)`, at least one. */
export function poolSize(
  methods: number,
  cores = typeof navigator === 'undefined' ? 2 : navigator.hardwareConcurrency,
): number {
  return Math.max(1, Math.min(methods, cores - 1, 4))
}

/**
 * Runs the requests, each in its own worker, and reports the state as it changes.
 *
 * @param requests - One per method, in the order they start.
 * @param labels - The label of each method, by id.
 * @param options - The worker factory and the callback.
 * @returns The handle that stops or cancels the run.
 */
export function runPool<Q extends PoolRequest, R extends PoolResult>(
  requests: readonly Q[],
  labels: Readonly<Record<string, string>>,
  options: PoolOptions<Q, R>,
): PoolHandle {
  const startedAt = performance.now()
  let alive = true
  let stopping = false
  let lines: StatusLine[] = requests.map((r) => ({
    id: r.method,
    label: labels[r.method] ?? r.method,
    phase: 'queued',
    step: 0,
    elapsedMs: 0,
    error: undefined,
  }))
  const results: Record<string, R> = {}
  const size = options.createWorker === undefined ? 1 : poolSize(requests.length, options.cores)
  let endedAt: number | undefined
  const emit = (): void => {
    if (alive)
      options.onChange({
        lines: [...lines],
        results: { ...results },
        running: endedAt === undefined,
        workers: options.createWorker === undefined ? 0 : size,
        startedAt,
        endedAt,
      })
  }
  const patch = (id: string, change: Partial<StatusLine>): void => {
    lines = lines.map((l) => (l.id === id ? { ...l, ...change } : l))
    emit()
  }
  const live = new Map<string, StageWorker>()
  const cancels = new Set<() => void>()
  let stopInline = false
  const runOne = async (request: Q): Promise<void> => {
    const id = request.method
    if (stopping || !alive) {
      patch(id, { phase: 'stopped' })
      return
    }
    patch(id, { phase: 'running' })
    const onStatus = (s: JobStatus): void => patch(id, { phase: s.phase, step: s.step, elapsedMs: s.elapsedMs })
    try {
      let result: R
      if (options.createWorker === undefined) {
        result = await options.run(request, {
          now: () => performance.now(),
          yieldToPage,
          stopped: () => stopInline,
          status: onStatus,
        })
      } else {
        const make = options.createWorker
        const stage = workerStage<Q, R>({
          name: id,
          input: (r) => r,
          createWorker: () => {
            const real = make()
            live.set(id, real)
            return {
              postMessage: (m) => real.postMessage(m),
              terminate: () => real.terminate(),
              get onerror() {
                return real.onerror
              },
              set onerror(h) {
                real.onerror = h
              },
              get onmessage() {
                return real.onmessage
              },
              set onmessage(h) {
                real.onmessage = h === null ? null : (e) => (isStatus(e.data) ? onStatus(e.data) : h(e))
              },
            }
          },
        })
        const ctx: StageContext = {
          progress: () => undefined,
          onCancel: (undo) => {
            cancels.add(undo)
            return () => cancels.delete(undo)
          },
        }
        result = await stage.run(request, ctx)
      }
      results[id] = result
      patch(id, { phase: 'done', step: result.steps, elapsedMs: result.elapsedMs })
    } catch (e) {
      patch(id, { phase: 'failed', error: e instanceof Error ? e.message : String(e) })
    } finally {
      live.delete(id)
    }
  }
  // A worker takes the next request when it has finished one.
  let next = 0
  const lane = async (): Promise<void> => {
    while (alive) {
      const request = requests[next++]
      if (request === undefined) return
      await runOne(request)
    }
  }
  emit()
  void Promise.all(Array.from({ length: size }, lane)).then(() => {
    endedAt = performance.now()
    emit()
  })
  return {
    stop: () => {
      stopping = true
      stopInline = true
      for (const w of live.values()) w.postMessage({ type: 'stop' })
    },
    cancel: () => {
      alive = false
      for (const undo of [...cancels]) undo()
    },
  }
}

/**
 * The worker's side of the pool: answers each `run` request with the poc's job on the wire of the producer's `serveStage`,
 * relays the job's status messages and ends the job early on a `stop` message (it returns what it has). A poc's worker
 * entry is `serveJob(self, run)`.
 *
 * @param scope - The worker's global scope.
 * @param run - The poc's job.
 */
export function serveJob<Q, R>(
  scope: {
    postMessage(message: unknown, transfer?: Transferable[]): void
    onmessage: ((e: { readonly data: unknown }) => void) | null
  },
  run: Job<Q, R>,
): void {
  let stopRequested = false
  serveStage<Q, R>(scope, (request) => {
    stopRequested = false
    return run(request, {
      now: () => performance.now(),
      yieldToPage,
      stopped: () => stopRequested,
      status: (status) => scope.postMessage(status),
    })
  })
  const serve = scope.onmessage
  scope.onmessage = (e) => {
    const data = e.data as { readonly type?: unknown } | null
    if (data !== null && typeof data === 'object' && data.type === 'stop') stopRequested = true
    else serve?.(e)
  }
}
