/**
 * The settled signal: how a page that exports a figure (a screenshot, a PDF of a deck, a recording) knows that the
 * figure is whole. Every live figure carries `data-omai-state` on its root element: `preparing` while work is under way
 * towards the end of a run, `idle` when the run is incomplete and nothing is computing (it waits for Play), `ready`
 * when it is all there, `failed` when it never will be. A live session handle carries the end state as a promise,
 * `live.settled`, which resolves only when the whole tape exists. An exporter waits with `waitForSettled`, which
 * treats `idle` like `ready` and `failed` (the figure will not change by itself, so it captures what is shown) and
 * says which figures were idle. This file holds the vocabulary (the attribute, the states, the result of the promise),
 * the pure rule that maps a handle onto a state, the hook that makes the promise of a run, and `waitForSettled`, which
 * waits for every figure under an element. Importing it touches no browser global: the DOM is only read when
 * `waitForSettled` is called.
 */

import { useEffect, useState } from 'react'

/** The attribute every live figure carries on its root element: its value is an {@link OmaiState}. */
export const STATE_ATTRIBUTE = 'data-omai-state'

/**
 * Where a live figure is, as the value of `data-omai-state` on its root element.
 *
 * - `preparing`: work is under way towards the end: a tape being computed (a primed or eager session priming, a worker
 *   computing), a live run being played or stepping to its `maxSteps`, a chart whose first frame is not drawn yet. A
 *   run that is playing but whose tape is incomplete is `preparing` too (there is no `running`: the state says
 *   whether the figure is whole, not whether it moves).
 * - `idle`: the run is incomplete and nothing is computing now, and nothing will until someone acts: a live run that
 *   was never started (or whose play was stopped), a session that is not `active` or has not been played under the
 *   `'play'` policy, a session whose computing was halted. What is drawn is stable, so an exporter may capture it
 *   (`waitForSettled` settles on `idle`); `live.settled` stays pending, since the whole tape does not exist.
 * - `ready`: the figure is whole. For a session figure the whole tape exists and the end of the run is reachable (it
 *   stays `ready` while it plays); for a chart the first frame has been drawn (a later option does not make it
 *   `preparing` again); for a scene or a structure the first commit has happened.
 * - `failed`: it will not become ready: the run failed (a worker died, a step threw), or the session was disposed and
 *   nothing has used it since.
 *
 * @index structure Where a live figure is: 'preparing', 'idle', 'ready' or 'failed' (`data-omai-state`)
 */
export type OmaiState = 'preparing' | 'idle' | 'ready' | 'failed'

/**
 * What a `live.settled` promise resolves with, never rejecting, so an exporter needs no `try`: `{ ok: true }` when the
 * whole tape exists, `{ ok: false, error }` when the run failed (`error` is the failure's message) or when the run was
 * let go of before it ended (`error` is `'disposed'`: the session was disposed, remade by new options, or unmounted).
 *
 * @index structure What `live.settled` resolves with: `{ ok: true }` or `{ ok: false, error }`
 */
export type SettledResult = { readonly ok: true } | { readonly ok: false; readonly error: unknown }

/** The `error` of a {@link SettledResult} of a run that was let go of before it ended. */
export const DISPOSED = 'disposed'

/**
 * The state of a session handle: `failed` when it has an error, or was disposed before the whole tape existed;
 * otherwise `ready` once the whole tape exists, else `idle` when nothing is computing now, else `preparing`.
 *
 * @param handle - The handle's `done`, `error` and `disposed`, and `idle`: whether nothing is computing towards the end
 *   of the run now (absent counts as false: an incomplete tape is `preparing`).
 * @returns The state.
 * @example
 * settledStateOf({ done: false, error: undefined, disposed: false }) // => 'preparing'
 * settledStateOf({ done: false, error: undefined, disposed: false, idle: true }) // => 'idle'
 * settledStateOf({ done: true, error: undefined, disposed: false, idle: true }) // => 'ready'
 * settledStateOf({ done: true, error: undefined, disposed: false }) // => 'ready'
 * settledStateOf({ done: false, error: 'the worker failed', disposed: false }) // => 'failed'
 * settledStateOf({ done: false, error: undefined, disposed: true }) // => 'failed'
 */
export function settledStateOf(handle: {
  readonly done: boolean
  readonly error?: string | undefined
  readonly disposed: boolean
  readonly idle?: boolean | undefined
}): OmaiState {
  if (handle.error !== undefined) return 'failed'
  if (handle.done) return 'ready'
  if (handle.disposed) return 'failed'
  return handle.idle === true ? 'idle' : 'preparing'
}

interface Pending {
  readonly promise: Promise<SettledResult>
  readonly settle: (result: SettledResult) => void
  result?: SettledResult
}

function pending(): Pending {
  let settle: (result: SettledResult) => void = () => undefined
  const promise = new Promise<SettledResult>((resolve) => {
    settle = resolve
  })
  const made: Pending = {
    promise,
    settle: (result) => {
      if (made.result !== undefined) return
      made.result = result
      settle(result)
    },
  }
  return made
}

const runs = new WeakMap<object, Pending>()
// A run whose owner unmounted is settled as disposed after a microtask, not at once: a strict-mode double effect runs
// its cleanup and its setup in the same task, and the setup takes the run off this list.
const leaving = new WeakSet<object>()

/**
 * The `settled` promise of a run, made once per run and resolved when the run's state allows it. Called in the render
 * of a hook: resolving there is safe because it is idempotent, and it makes the promise already resolved on the first
 * render of a tape that is whole. A promise already settled whose outcome no longer matches the run (a tape that was whole and was disposed, a disposed run
 * used again) is replaced by a new one.
 *
 * @param run - The object that stands for the run (a producer, a run's state): one promise per object.
 * @param state - The run's state now (see {@link settledStateOf}): only `ready` and `failed` settle the promise.
 * @param error - The failure to resolve with when `state` is `failed`.
 * @returns The promise (see {@link SettledResult}).
 */
export function useRunSettled(run: object, state: OmaiState, error: unknown): Promise<SettledResult> {
  let p = runs.get(run)
  if (p === undefined || (p.result !== undefined && (p.result.ok ? state !== 'ready' : state !== 'failed'))) {
    p = pending()
    runs.set(run, p)
  }
  if (state === 'ready') p.settle({ ok: true })
  else if (state === 'failed') p.settle({ ok: false, error })
  // Leaving the run (unmounted, or replaced by a remake) settles a promise nothing else will settle.
  useEffect(() => {
    leaving.delete(run)
    return () => {
      leaving.add(run)
      queueMicrotask(() => {
        if (leaving.delete(run)) runs.get(run)?.settle({ ok: false, error: DISPOSED })
      })
    }
  }, [run])
  return p.promise
}

/**
 * The state of a figure that is whole once it has been committed: `preparing` in its first render, `ready` from the
 * first commit on (what `data-omai-state` of a scene or a diagram means).
 *
 * @returns The state, for the figure's root element.
 */
export function useCommitted(): OmaiState {
  const [state, setState] = useState<OmaiState>('preparing')
  useEffect(() => setState('ready'), [])
  return state
}

/**
 * What {@link waitForSettled} resolves with. `idle` lists the figures that read `idle` (their run is incomplete and
 * waits for Play: they were captured as shown); it is absent when there were none, so a page with nothing idle gets
 * exactly the shape it always had.
 */
export type WaitResult =
  | { readonly ok: true; readonly ready: number; readonly idle?: readonly unknown[] }
  | {
      readonly ok: false
      /** `failed` when a figure reads `failed`, `timeout` when some are still `preparing` at the deadline. */
      readonly reason: 'failed' | 'timeout'
      readonly failed: readonly unknown[]
      readonly pending: readonly unknown[]
      readonly idle?: readonly unknown[]
    }

/** Options of {@link waitForSettled}. */
export interface WaitOptions {
  /** The longest to wait, in milliseconds (default 30 000). At the deadline it resolves with `reason: 'timeout'`. */
  readonly timeoutMs?: number
}

interface Stated {
  getAttribute(name: string): string | null
}

function figures(root: ParentNode): Stated[] {
  const found = [...(root.querySelectorAll(`[${STATE_ATTRIBUTE}]`) as Iterable<Stated>)]
  // The root may itself be a figure (a `ParentNode` that is an element): it is not found by its own selector.
  const self = root as unknown as Partial<Stated>
  if (typeof self.getAttribute === 'function' && self.getAttribute(STATE_ATTRIBUTE) !== null) {
    found.unshift(self as Stated)
  }
  return found
}

/**
 * Wait until every live figure under `root` (every element with `data-omai-state`, `root` itself included) is `ready`,
 * `idle` or `failed`, so an exporter does not poll. An `idle` figure (an unstarted or paused run, nothing computing)
 * will not change by itself, so it counts as settled and is reported in `idle`. It watches the subtree with a `MutationObserver` for figures that appear
 * and states that change, and gives up at the deadline. It resolves at once when nothing under `root` carries the
 * attribute (call it after the tree has committed), and never rejects.
 *
 * It needs a DOM only when called (it reads `MutationObserver` from the global scope then): importing it is safe in Node.
 *
 * @param root - The element (or document) holding the figures.
 * @param options - The timeout.
 * @returns `{ ok: true, ready }` with the number of figures when none is `preparing` or `failed`, plus `idle` (the
 *   elements that read `idle`) when there were any; otherwise `{ ok: false, reason, failed, pending }`, with the
 *   elements in each state (and `idle` likewise).
 * @index function Wait until every `data-omai-state` figure under an element is ready, idle or failed: `await waitForSettled(root)`
 */
export function waitForSettled(root: ParentNode, options: WaitOptions = {}): Promise<WaitResult> {
  const timeoutMs = options.timeoutMs ?? 30_000
  return new Promise<WaitResult>((resolve) => {
    let watcher: MutationObserverLike | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    // Looks at the figures; settles (and stops watching) when none is preparing, or at the deadline.
    const look = (deadline: boolean): boolean => {
      const all = figures(root)
      const state = (f: Stated): string | null => f.getAttribute(STATE_ATTRIBUTE)
      const failed = all.filter((f) => state(f) === 'failed')
      const idle = all.filter((f) => state(f) === 'idle')
      const waiting = all.filter((f) => state(f) !== 'ready' && state(f) !== 'failed' && state(f) !== 'idle')
      const also = idle.length > 0 ? { idle } : {}
      if (waiting.length > 0 && !deadline) return false
      watcher?.disconnect()
      if (timer !== undefined) clearTimeout(timer)
      if (failed.length > 0) resolve({ ok: false, reason: 'failed', failed, pending: waiting, ...also })
      else if (waiting.length > 0) resolve({ ok: false, reason: 'timeout', failed, pending: waiting, ...also })
      else resolve({ ok: true, ready: all.length, ...also })
      return true
    }
    if (look(false)) return
    const Observer = (globalThis as { MutationObserver?: new (callback: () => void) => MutationObserverLike })
      .MutationObserver
    if (Observer !== undefined) {
      watcher = new Observer(() => look(false))
      watcher.observe(root as unknown as Node, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: [STATE_ATTRIBUTE],
      })
    }
    timer = setTimeout(() => look(true), timeoutMs)
  })
}

interface MutationObserverLike {
  observe(target: Node, options: Record<string, unknown>): void
  disconnect(): void
}
