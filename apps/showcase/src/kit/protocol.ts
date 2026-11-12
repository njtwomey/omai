/**
 * The wire between a story's page and its worker, and the worker's side of it. The page asks `{ type: 'run', seed }`; the worker
 * answers with any number of `status` sentences (what it is doing now, for the play button to show) and then one `result` (plain
 * data, cloned across) or an `error`. A story's `worker.ts` is one line, `serveStory(self, run)`, so the worker bundles this
 * file and the story's `compute.ts` and nothing of React. A synchronous `run` can still say things as it goes: `postMessage`
 * leaves the worker at once, whatever the computation is doing.
 */

/** Says what the computation is doing now, in a sentence ('Warming up the sampler'). The last one said stays on screen. */
export type Say = (sentence: string) => void

/** What the page asks of a worker. */
export interface RunRequest {
  readonly type: 'run'
  readonly seed: number
}

/** What a worker says to the page. */
export type WorkerEvent<R> =
  | { readonly type: 'status'; readonly sentence: string }
  | { readonly type: 'result'; readonly value: R }
  | { readonly type: 'error'; readonly message: string }

/** A handler typed as a method, so a real worker scope's `onmessage` (typed on `MessageEvent`) fits. */
type Handler<E> = { bivarianceHack(event: E): void }['bivarianceHack']

/** The part of a worker's global scope the wire uses (`self`). */
export interface WorkerScope {
  postMessage(message: unknown): void
  onmessage: Handler<{ readonly data: unknown }> | null
}

const isRun = (x: unknown): x is RunRequest =>
  typeof x === 'object' && x !== null && (x as { type?: unknown }).type === 'run'

/**
 * The worker's side: answers a `run` request by calling `run(seed, say)`, relays what it says, and posts its result or the
 * message of what it threw. A story's `worker.ts` is `serveStory(self, run)`.
 *
 * @param scope - The worker's global scope (`self`).
 * @param run - The story's computation (its `compute.ts`).
 */
export function serveStory<R>(scope: WorkerScope, run: (seed: number, say: Say) => R | Promise<R>): void {
  scope.onmessage = (event) => {
    if (!isRun(event.data)) return
    const say: Say = (sentence) => scope.postMessage({ type: 'status', sentence } satisfies WorkerEvent<R>)
    void (async (): Promise<void> => {
      try {
        const value = await run((event.data as RunRequest).seed, say)
        scope.postMessage({ type: 'result', value } satisfies WorkerEvent<R>)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        scope.postMessage({ type: 'error', message } satisfies WorkerEvent<R>)
      }
    })()
  }
}
