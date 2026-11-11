import { useCallback, useEffect, useRef, useState } from 'react'

/** Where a reference computed in a worker stands. */
export type ReferenceState<R> =
  | { readonly status: 'idle' }
  | { readonly status: 'running'; readonly startedAt: number }
  | { readonly status: 'done'; readonly result: R; readonly ms: number }
  | { readonly status: 'failed'; readonly error: string }

/**
 * A reference (a long NUTS run) computed in a Web Worker, on demand: `start()` sends the request to a new worker, a change of
 * the request (a new seed, other knobs) stops the one running and forgets its result, and the page keeps responding
 * meanwhile. The worker's file is `self.onmessage = (e) => self.postMessage(compute(e.data))`.
 *
 * @param make - Makes the worker (`() => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' })`).
 * @param request - What the worker is asked, as plain data; it is the key of the result.
 * @returns The state, and `start` and `stop`.
 */
export function useReference<Q, R>(
  make: () => Worker,
  request: Q,
): { readonly state: ReferenceState<R>; readonly start: () => void; readonly stop: () => void } {
  const [state, setState] = useState<ReferenceState<R>>({ status: 'idle' })
  const worker = useRef<Worker | undefined>(undefined)
  const key = JSON.stringify(request)
  const stop = useCallback((): void => {
    worker.current?.terminate()
    worker.current = undefined
    setState({ status: 'idle' })
  }, [])
  // A new request forgets the old run.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the key is the request
  useEffect(() => stop, [key, stop])
  const start = useCallback((): void => {
    worker.current?.terminate()
    const t0 = performance.now()
    const w = make()
    worker.current = w
    setState({ status: 'running', startedAt: t0 })
    w.onmessage = (e: MessageEvent<R>): void => {
      setState({ status: 'done', result: e.data, ms: performance.now() - t0 })
      w.terminate()
      if (worker.current === w) worker.current = undefined
    }
    w.onerror = (e): void => setState({ status: 'failed', error: e.message })
    w.postMessage(JSON.parse(key))
  }, [make, key])
  return { state, start, stop }
}
