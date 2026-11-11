/**
 * Tasks that compute in a Web Worker, for the work too heavy to share the page's thread even in slices (a sweep that
 * takes a second). The page side is `workerTask(make, request)`, a `RunTask` like any other: it starts the worker, relays
 * what the worker reports to the host, keeps the partial results the worker sends, and on Stop asks the worker to stop
 * (it answers with what it has at the next sweep) and terminates it if it does not within a moment. The worker's side is
 * `serveRun(self, work)`: its `work` is a task that is given the request and the same host, whose `report` goes back to the
 * page, whose `stopped` turns true on the page's Stop and whose `partial` sends a result so far.
 */

import type { RunHost, RunProgress, RunTask } from './run.ts'
import { yieldToPage } from './run.ts'

/** What a worker says to the page. */
export type WorkerMessage<R> =
  | { readonly type: 'progress'; readonly progress: RunProgress }
  | { readonly type: 'partial'; readonly value: R }
  | { readonly type: 'result'; readonly value: R | undefined }
  | { readonly type: 'error'; readonly message: string }

/** What the page says to a worker. */
export type PageMessage<Q> = { readonly type: 'run'; readonly request: Q } | { readonly type: 'stop' }

/** What a worker's task is given: the host, and a way to send a result so far. */
export interface WorkerHost<R> extends Omit<RunHost, 'onStop'> {
  readonly partial: (value: R) => void
}

/** A worker's task: the request answered with a result (or what it has when it is stopped). */
export type WorkerWork<Q, R> = (request: Q, host: WorkerHost<R>) => Promise<R | undefined>

/** The worker's global scope, as far as `serveRun` needs it. */
export interface WorkerScope {
  postMessage(message: unknown): void
  onmessage: ((e: MessageEvent) => void) | null
}

/**
 * The worker's side: answers a `run` message with the work's result (or its error), relays its reports, and turns a `stop`
 * message into `host.stopped()`. A worker's file is `serveRun(self, work)`.
 *
 * @param scope - The worker's global scope (`self`).
 * @param work - The task.
 */
export function serveRun<Q, R>(scope: WorkerScope, work: WorkerWork<Q, R>): void {
  let stopRequested = false
  scope.onmessage = (e) => {
    const message = e.data as PageMessage<Q>
    if (message.type === 'stop') {
      stopRequested = true
      return
    }
    stopRequested = false
    const send = (m: WorkerMessage<R>): void => scope.postMessage(m)
    work(message.request, {
      report: (progress) => send({ type: 'progress', progress }),
      partial: (value) => send({ type: 'partial', value }),
      stopped: () => stopRequested,
      yieldToPage,
    }).then(
      (value) => send({ type: 'result', value }),
      (error: unknown) => send({ type: 'error', message: error instanceof Error ? error.message : String(error) }),
    )
  }
}

/**
 * The page's side: a task that computes `request` in a worker made by `make` (`() => new Worker(new URL('./x.worker.ts',
 * import.meta.url), { type: 'module' })`). Its result is the worker's, or, when the run is stopped, what the worker
 * answered with (or the last partial result it sent).
 *
 * @param make - Makes the worker.
 * @param request - What the worker is asked, as plain data.
 * @param graceMs - How long a stopped worker has to answer before it is terminated (default 4000).
 * @returns The task.
 */
export function workerTask<Q, R>(make: () => Worker, request: Q, graceMs = 4000): RunTask<R> {
  return (host) =>
    new Promise<R | undefined>((resolve, reject) => {
      const worker = make()
      let last: R | undefined
      let settled = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const settle = (done: () => void): void => {
        if (settled) return
        settled = true
        if (timer !== undefined) clearTimeout(timer)
        worker.terminate()
        done()
      }
      worker.onmessage = (e: MessageEvent<WorkerMessage<R>>): void => {
        const m = e.data
        if (m.type === 'progress') host.report(m.progress)
        else if (m.type === 'partial') last = m.value
        else if (m.type === 'result') settle(() => resolve(m.value ?? last))
        else settle(() => reject(new Error(m.message)))
      }
      worker.onerror = (e): void => settle(() => reject(new Error(e.message)))
      host.onStop(() => {
        if (settled) return
        worker.postMessage({ type: 'stop' } satisfies PageMessage<Q>)
        timer = setTimeout(() => settle(() => resolve(last)), graceMs)
      })
      worker.postMessage({ type: 'run', request } satisfies PageMessage<Q>)
    })
}
