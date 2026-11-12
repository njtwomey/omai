/**
 * Where a story's run stands, as a pure reducer over what happens to it (so `useStoryRun` is a thin shell and the states are
 * tested without a page or a worker). A run starts only on `start`: `idle` until then. Its result is held after it is done, and
 * a new run (`again`, with a new seed) keeps the previous result on screen until the new one arrives.
 */

/** Where a run is. */
export type RunPhase = 'idle' | 'running' | 'done' | 'error'

/** What the page knows of a run. */
export interface RunState<R> {
  readonly phase: RunPhase
  /** The latest result (kept while a later run is going, so the figures do not vanish). */
  readonly result: R | undefined
  /** The last sentence the worker said ('' before it says any). */
  readonly status: string
  /** The seed of the current (or last) run. */
  readonly seed: number
  /** How many runs have been started. */
  readonly runs: number
  /** Milliseconds the last finished run took. */
  readonly elapsedMs: number | undefined
  readonly error: string | undefined
}

/** What happens to a run. */
export type RunEvent<R> =
  | { readonly type: 'start'; readonly seed: number }
  | { readonly type: 'status'; readonly sentence: string }
  | { readonly type: 'result'; readonly value: R; readonly elapsedMs: number }
  | { readonly type: 'error'; readonly message: string }

/**
 * The state before anything has happened: idle, no result.
 *
 * @param seed - The seed of the first run.
 * @returns The initial state.
 */
export const initialRun = <R>(seed: number): RunState<R> => ({
  phase: 'idle',
  result: undefined,
  status: '',
  seed,
  runs: 0,
  elapsedMs: undefined,
  error: undefined,
})

/**
 * The next state. Late events (a status or result that arrives when the run is not going) are ignored.
 *
 * @param state - The state.
 * @param event - What happened.
 * @returns The new state.
 */
export function stepRun<R>(state: RunState<R>, event: RunEvent<R>): RunState<R> {
  switch (event.type) {
    case 'start':
      return { ...state, phase: 'running', status: '', seed: event.seed, runs: state.runs + 1, error: undefined }
    case 'status':
      return state.phase === 'running' ? { ...state, status: event.sentence } : state
    case 'result':
      return state.phase === 'running'
        ? { ...state, phase: 'done', result: event.value, elapsedMs: event.elapsedMs, status: '' }
        : state
    case 'error':
      return state.phase === 'running' ? { ...state, phase: 'error', error: event.message } : state
  }
}

/**
 * The seed of the run after one: re-rolled, and never the same as the last.
 *
 * @param seed - The seed of the last run.
 * @returns The next seed.
 */
export const nextSeed = (seed: number): number => (seed * 7919 + 104729) % 1_000_003
