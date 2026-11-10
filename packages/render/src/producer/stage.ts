/**
 * A heavy step of a `useJob` job run in a Web Worker: `workerStage` is the page's side (a stage that hands the worker its
 * input, relays its progress, resolves with its result and terminates the worker when it is done or the job is replaced)
 * and `serveStage` is the worker's side (a worker entry is `serveStage(self, run)`). The wire is three messages: a `run`
 * request, and `progress`, `result` or `error` answers. Headless: no React (a stage is structurally a `JobStage`), so a
 * worker bundles only this.
 */

/** The part of a `Worker` a stage uses. */
export interface StageWorker {
  postMessage(message: unknown): void
  terminate(): void
  onmessage: ((event: { readonly data: unknown }) => void) | null
  onerror: ((event: { readonly message?: string }) => void) | null
}

/** The worker's side of the wire: the part of a dedicated worker's global scope {@link serveStage} uses. */
export interface StageScope {
  postMessage(message: unknown): void
  onmessage: ((event: { readonly data: unknown }) => void) | null
}

/** What a stage run reads from its job (a `JobContext` has both). */
export interface StageContext {
  readonly progress: (fraction: number) => void
  readonly onCancel: (undo: () => void) => () => void
}

/** Options of {@link workerStage}. */
export interface WorkerStageOptions<P> {
  /** The name shown while the stage runs. */
  readonly name: string
  /** Share of the job's bar this stage takes (default 1). */
  readonly weight?: number
  /** Make the worker: called when the stage runs, once per run (the worker is terminated when the run ends). */
  readonly createWorker: () => StageWorker
  /** What the worker is given, from the previous stage's result (default: the previous result itself). Must be cloneable. */
  readonly input?: (prev: P) => unknown
}

/** What a worker says back: how far it is, its result, or why it failed. */
type StageEvent =
  | { readonly type: 'progress'; readonly fraction: number }
  | { readonly type: 'result'; readonly value: unknown }
  | { readonly type: 'error'; readonly message: string }

function isStageEvent(x: unknown): x is StageEvent {
  if (typeof x !== 'object' || x === null) return false
  const t = (x as { readonly type?: unknown }).type
  return t === 'progress' || t === 'result' || t === 'error'
}

/**
 * A stage of a `useJob` job that runs in a Web Worker, so a single heavy step does not block the page. Each run makes
 * a worker, posts it `{ type: 'run', input }`, relays its `progress` messages to the job's bar, resolves with its
 * `result` and terminates the worker. When the job is replaced, deactivated or unmounted while the worker is still
 * working, the worker is terminated at once (not on its next message) and the stage's promise is dropped. A failure
 * (an `error` message, or the worker's own error event) rejects, and the job's phase is `error`.
 *
 * @param options - The name, the worker factory, the weight and what to send.
 * @returns A job stage: put it in `useJob`'s `stages`.
 * @index function A job stage computed in a Web Worker, terminated on cancel: `workerStage({ name, createWorker })`.
 * @example
 * ```ts
 * import { workerStage } from 'omai-render/producer'
 * const stage = workerStage<number>({
 *   name: 'solving',
 *   createWorker: () => new Worker(new URL('./solve.worker.ts', import.meta.url), { type: 'module' }),
 *   input: (n) => ({ n }),
 * })
 * stage.name // => 'solving'
 * ```
 */
export function workerStage<P = unknown, R = unknown>(
  options: WorkerStageOptions<P>,
): {
  readonly name: string
  readonly weight?: number
  readonly run: (prev: P, ctx: StageContext) => Promise<R>
} {
  const run = (prev: P, ctx: StageContext): Promise<R> =>
    new Promise<R>((resolve, reject) => {
      const worker = options.createWorker()
      let settled = false
      const forget = ctx.onCancel(() => end())
      // Stop the worker and stop listening: the end of every run, whatever way it ends.
      function end(): void {
        settled = true
        worker.onmessage = null
        worker.onerror = null
        worker.terminate()
      }
      const finish = (action: () => void): void => {
        if (settled) return
        end()
        forget()
        action()
      }
      worker.onmessage = (e) => {
        if (settled || !isStageEvent(e.data)) return
        const event = e.data
        if (event.type === 'progress') ctx.progress(event.fraction)
        else if (event.type === 'result') finish(() => resolve(event.value as R))
        else finish(() => reject(new Error(event.message)))
      }
      worker.onerror = (e) => finish(() => reject(new Error(e.message ?? 'the worker failed')))
      try {
        worker.postMessage({ type: 'run', input: options.input === undefined ? prev : options.input(prev) })
      } catch (error) {
        finish(() => reject(error instanceof Error ? error : new Error(String(error))))
      }
    })
  return { name: options.name, ...(options.weight === undefined ? {} : { weight: options.weight }), run }
}

/**
 * The worker's side of a {@link workerStage}: answer each `run` request by calling `run` with its input, relaying
 * what `run` reports as progress, and posting its result (or the message of what it threw). A worker entry is one
 * line: `serveStage(self, (input, progress) => solve(input, progress))`.
 *
 * @param scope - The worker's global scope (`self`).
 * @param run - The computation: the input, and a function to report progress from 0 to 1. It may return a promise.
 * @returns Nothing; the scope's `onmessage` is set.
 * @index function The worker entry of a workerStage: `serveStage(self, run)`.
 * @example
 * ```ts
 * import { serveStage } from 'omai-render/producer'
 * const sent: unknown[] = []
 * const scope = { postMessage: (m: unknown) => sent.push(m), onmessage: null as ((e: { data: unknown }) => void) | null }
 * serveStage(scope, (n: number) => n * 2)
 * scope.onmessage?.({ data: { type: 'run', input: 21 } })
 * ```
 */
export function serveStage<I = unknown, R = unknown>(
  scope: StageScope,
  run: (input: I, progress: (fraction: number) => void) => R | Promise<R>,
): void {
  scope.onmessage = (e) => {
    const data = e.data as { readonly type?: unknown; readonly input?: unknown } | null
    if (data === null || typeof data !== 'object' || data.type !== 'run') return
    const progress = (fraction: number): void => scope.postMessage({ type: 'progress', fraction })
    void (async (): Promise<void> => {
      try {
        scope.postMessage({ type: 'result', value: await run(data.input as I, progress) })
      } catch (error) {
        scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
      }
    })()
  }
}
