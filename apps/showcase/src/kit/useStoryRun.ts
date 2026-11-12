/**
 * `useStoryRun`: the hook behind a story's play button. Nothing runs when the page loads, and nothing is fetched ahead of
 * time: `start()` makes the story's worker (a module worker, so its bundle, with the engine in it, loads only now), asks it
 * to run with a seed, relays what it says, and holds the result. `again()` re-rolls the seed and runs once more; the old result
 * stays on screen until the new one arrives. The worker is terminated when the run ends, when another starts and when the page
 * unmounts.
 */

import { useCallback, useEffect, useReducer, useRef } from 'react'
import type { RunRequest, WorkerEvent } from './protocol.ts'
import { initialRun, nextSeed, type RunEvent, type RunState, stepRun } from './runState.ts'

/** What a story's page holds on a run. */
export interface StoryRun<R> extends RunState<R> {
  /** Starts the first run (a no-op while one is going). */
  readonly start: () => void
  /** Re-rolls the seed and runs again. */
  readonly again: () => void
}

/**
 * The run of a story.
 *
 * @param createWorker - Makes the story's worker: `() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })`
 *   (the literal has to be in the story's own file for the bundler to find the worker).
 * @param firstSeed - The seed of the first run (default 1).
 * @returns The state of the run and its two actions.
 */
export function useStoryRun<R>(createWorker: () => Worker, firstSeed = 1): StoryRun<R> {
  const [state, dispatch] = useReducer(
    (s: RunState<R>, e: RunEvent<R>) => stepRun(s, e),
    firstSeed,
    (seed) => initialRun<R>(seed),
  )
  const worker = useRef<Worker | undefined>(undefined)
  const going = useRef(false)
  const seed = useRef(firstSeed)
  const make = useRef(createWorker)
  make.current = createWorker

  const stop = useCallback((): void => {
    worker.current?.terminate()
    worker.current = undefined
  }, [])

  const run = useCallback(
    (next: number): void => {
      stop()
      seed.current = next
      going.current = true
      dispatch({ type: 'start', seed: next })
      const began = performance.now()
      let w: Worker
      try {
        w = make.current()
      } catch (error) {
        going.current = false
        dispatch({ type: 'error', message: error instanceof Error ? error.message : String(error) })
        return
      }
      worker.current = w
      const end = (event: RunEvent<R>): void => {
        if (worker.current !== w) return
        going.current = false
        stop()
        dispatch(event)
      }
      w.onmessage = (e: MessageEvent<WorkerEvent<R>>): void => {
        const m = e.data
        if (worker.current !== w) return
        if (m.type === 'status') dispatch({ type: 'status', sentence: m.sentence })
        else if (m.type === 'result') end({ type: 'result', value: m.value, elapsedMs: performance.now() - began })
        else end({ type: 'error', message: m.message })
      }
      w.onerror = (e: ErrorEvent): void => end({ type: 'error', message: e.message || 'the worker failed' })
      w.postMessage({ type: 'run', seed: next } satisfies RunRequest)
    },
    [stop],
  )

  const start = useCallback((): void => {
    if (!going.current) run(seed.current)
  }, [run])
  const again = useCallback((): void => {
    if (!going.current) run(nextSeed(seed.current))
  }, [run])

  useEffect(() => stop, [stop])
  return { ...state, start, again }
}
