/**
 * The React hook that runs a hub entry's method whenever its state changes, debounced: in a Web Worker when the entry
 * is in the catalogue and workers are available (a newer state stops a run still going), otherwise on the main thread,
 * reporting progress, the records, timing and where the run happened. Also the request and reply messages of the hub's
 * worker. Given the entry's module (`mdml-hub/lazy`), the worker loads that module alone; otherwise the catalogue is
 * imported when a run starts, never with the page.
 */

import type { Exposition } from 'mdml-engine/exposition'
import { useEffect, useRef, useState } from 'react'
import type { HubEntry, HubProgress } from '../entry/entry.ts'
import { type HubState, runDemo } from '../entry/run.ts'

/** A request to the hub's worker: run the catalogue entry `id` (found in `module`, if given) on `state`. */
export interface HubRunRequest {
  readonly token: number
  readonly id: string
  /** The entry's module in `mdml-hub/lazy`'s `entryModules`; omitted, the worker looks in the whole catalogue. */
  readonly module?: string
  readonly state: HubState
}

/** The worker's reply: progress, the records and timing, or the error message. */
export type HubRunReply =
  | { readonly token: number; readonly progress: HubProgress }
  | {
      readonly token: number
      readonly ok: true
      readonly exposition: readonly Exposition[]
      readonly rows: number
      readonly ms: number
    }
  | { readonly token: number; readonly ok: false; readonly error: string }

/** Where a run happened. */
export type RunPlace = 'worker' | 'main thread'

/** The state of an entry's latest run. */
export type HubRun =
  | { readonly status: 'running'; readonly previous?: HubRunDone; readonly progress?: HubProgress }
  | HubRunDone
  | { readonly status: 'error'; readonly error: string; readonly previous?: HubRunDone }

/** A finished run: its records, the rows of the dataset, how long it took and where it ran. */
export interface HubRunDone {
  readonly status: 'done'
  readonly exposition: readonly Exposition[]
  readonly rows: number
  readonly ms: number
  readonly place: RunPlace
  /** The state the run was made on. */
  readonly state: HubState
}

const now = (): number => performance.now()

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))

function spawn(): Worker | undefined {
  if (typeof Worker === 'undefined') return undefined
  try {
    return new Worker(new URL('./_worker.ts', import.meta.url), { type: 'module' })
  } catch {
    return undefined
  }
}

// Whether the worker can find this entry: given its module, the worker loads that; otherwise the entry must be the
// catalogue's own (imported here only when needed, so a page showing one entry does not load them all).
function inCatalogue(entry: HubEntry, module: string | undefined): Promise<boolean> {
  if (module !== undefined) return Promise.resolve(true)
  return import('../catalogue/index.ts').then(
    ({ catalogue }) => catalogue.byId.get(entry.id) === entry,
    () => false,
  )
}

/**
 * Run an entry's method on a state whenever the state changes, debounced: in a Web Worker when the entry is in the
 * catalogue and workers are available (a newer state stops a run still going), otherwise on the main thread. The last
 * finished run stays available while the next one runs.
 *
 * @param entry - The entry.
 * @param state - The dataset and parameters.
 * @param delay - The debounce in milliseconds (default 200; the first run starts at once).
 * @param module - The entry's module in `mdml-hub/lazy`'s `entryModules`, so the worker loads it alone (default: the
 *   worker looks the entry up in the whole catalogue).
 * @returns The latest run.
 */
export function useHubRun(entry: HubEntry, state: HubState, delay = 200, module?: string): HubRun {
  const [run, setRun] = useState<HubRun>({ status: 'running' })
  const worker = useRef<{ w: Worker; busy: boolean } | undefined>(undefined)
  const token = useRef(0)
  const first = useRef(true)
  const key = JSON.stringify(state)

  // biome-ignore lint/correctness/useExhaustiveDependencies: the state is compared by content (`key`)
  useEffect(() => {
    const t = ++token.current
    const done = (r: Omit<HubRunDone, 'status' | 'state'>): void => {
      if (t === token.current) setRun({ status: 'done', ...r, state })
    }
    const failed = (error: string): void => {
      if (t === token.current) setRun((p) => ({ status: 'error', error, ...previousOf(p) }))
    }
    const progress = (p: HubProgress): void => {
      if (t === token.current) setRun((r) => ({ status: 'running', progress: p, ...previousOf(r) }))
    }
    const onMain = (): void => {
      const start = now()
      runDemo(entry, state, progress).then(
        (r) => done({ ...r, ms: now() - start, place: 'main thread' }),
        (e: unknown) => failed(message(e)),
      )
    }
    const start = (): void => {
      setRun((p) => ({ status: 'running', ...previousOf(p) }))
      void inCatalogue(entry, module).then((found) => {
        if (t !== token.current) return
        if (found) inWorker()
        // Let the "running" state paint before a main-thread run blocks.
        else setTimeout(onMain, 0)
      })
    }
    const inWorker = (): void => {
      if (worker.current?.busy === true) {
        worker.current.w.terminate()
        worker.current = undefined
      }
      if (worker.current === undefined) {
        const w = spawn()
        if (w === undefined) {
          setTimeout(onMain, 0)
          return
        }
        worker.current = { w, busy: false }
      }
      const slot = worker.current
      slot.busy = true
      slot.w.onmessage = (e: MessageEvent<HubRunReply>) => {
        if (e.data.token !== token.current) return
        if ('progress' in e.data) {
          progress(e.data.progress)
          return
        }
        slot.busy = false
        if (e.data.ok) done({ exposition: e.data.exposition, rows: e.data.rows, ms: e.data.ms, place: 'worker' })
        else failed(e.data.error)
      }
      slot.w.onerror = (e: ErrorEvent) => {
        // The worker could not start (or crashed): drop it and run here instead.
        e.preventDefault()
        slot.w.terminate()
        if (worker.current === slot) worker.current = undefined
        if (t === token.current) onMain()
      }
      const request: HubRunRequest = { token: t, id: entry.id, state, ...(module === undefined ? {} : { module }) }
      slot.w.postMessage(request)
    }
    if (first.current) {
      first.current = false
      start()
      return
    }
    const timer = setTimeout(start, delay)
    return () => clearTimeout(timer)
  }, [entry, key, delay, module])

  useEffect(
    () => () => {
      worker.current?.w.terminate()
      worker.current = undefined
    },
    [],
  )
  return run
}

function previousOf(run: HubRun): { readonly previous?: HubRunDone } {
  if (run.status === 'done') return { previous: run }
  return run.previous === undefined ? {} : { previous: run.previous }
}
