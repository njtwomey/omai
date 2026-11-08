/**
 * React hooks over engine sessions, in two halves that know nothing of each other's pace: a producer that computes
 * the run ahead in the spare time of each animation frame (so the tape may stand at step 1000 while the page shows
 * step 100), and the tape itself, versioned as it grows, that the player and the panels read at their own rate. A
 * change of method or options is applied when the caller asks (live mode holds it until play), or after a short pause
 * (eager mode, and live mode without holding, so dragging a slider remakes the session once). A comparison runs
 * several variants of one method (the same seed, data, budget and subscription; only the knobs differ) under one
 * producer that keeps them in step, so one cursor can view all of them; a single session is a comparison of one. The
 * producer is the main-thread one by default (slices in the spare time of each frame); given a `producer` factory
 * (`workerProducer`) the same hook computes in a Web Worker and the tape streams back, and nothing else changes.
 *
 * When the run is computed and where the cursor starts is the start policy (`start`): `'play'` computes while playing
 * (live), `'first'` computes the whole run in the background and parks the cursor at step 0 until play (primed), `'end'`
 * computes the whole run in the background and shows its end (eager). Every computing is in time slices, so mounting a
 * session never blocks the page. The handle says where it is (`status`, `progress`) and lets go of everything it holds
 * (`dispose`).
 */

import { OmaiError } from 'omai-engine/base'
import type { Config, Method, Session, SessionOptions, SliceBudget } from 'omai-engine/trace'
import { useCallback, useEffect, useRef, useState } from 'react'
import { inlineProducer } from '../../producer/inline.ts'
import type { Variant } from '../../producer/make.ts'
import type { Producer, ProducerFactory } from '../../producer/producer.ts'
import { sharedHead } from '../../producer/step.ts'
import { DEFAULT_STEPS, type PlayStatus, progressOf, type StartPolicy, startOf, statusOf } from '../start.ts'

export type { PlayStatus, StartPolicy, Variant }

/** What a live session and a comparison of several share: the producer's state and controls, as the player drives them. */
export interface LiveHandle {
  /** Changes whenever the tape grows or the session is remade, so memoised views recompute. */
  readonly version: number
  /** The head of the tape: the step computed so far (of a comparison, the step every variant has reached). */
  readonly head: number
  readonly done: boolean
  /** Whether the producer is computing ahead in the background. */
  readonly computing: boolean
  /** Start computing ahead in the spare time of each animation frame, until the run ends. */
  compute(): void
  /** Stop computing ahead (the tape stays). */
  halt(): void
  /** Compute a slice now, on the spot (every variant of a comparison, kept in step). */
  advance(budget: SliceBudget): void
  /** Milliseconds the last slice took (the cost of one frame's work). */
  readonly lastMs: number
  /** The method or its options have changed since the session was made: held until `apply`, or about to be applied. */
  readonly stale: boolean
  /** A change is held until `apply` (live mode, holding). */
  readonly held: boolean
  /** Remake from the latest method and options now (also `session` from the next render). */
  apply(): unknown
  /**
   * Where the producer computes (`worker` when sessions are computed off the main thread), and, when a worker was asked
   * for and not used, why.
   */
  readonly producer?: { readonly kind: 'inline' | 'worker'; readonly fallback?: string }
  /** What went wrong in the producer, once something has (a worker that failed, a method it could not make). */
  readonly error?: string | undefined
  /** Tell the producer which step the consumer is viewing (a worker with a lookahead computes up to it and no more). */
  follow?(step: number): void
  /** Counts the probe answers that have arrived from a worker, so panels reading probes ask again. */
  readonly probed?: number
  /** The start policy in force: when the run is computed and where the cursor starts (see {@link StartPolicy}). */
  readonly start: StartPolicy
  /**
   * `preparing` while the tape is incomplete and nothing plays (under `'play'` the producer idles until play, so check
   * `computing` to tell a fill in progress from a wait), `ready` once the whole tape exists and nothing plays,
   * `running` while playing, including a play pressed before the tape is ready (it plays at the head, as live does).
   */
  readonly status: PlayStatus
  /**
   * The fraction of the run computed, 0 to 1. While the run goes on, the total is the options' `steps` (the engine's
   * default of 1000 without one); once it has ended, whatever step it ended at (a run that converged after 8 steps), it
   * is exactly 1, against its own length. So a run that ends early sits near 0 while it computes and jumps to 1 on the
   * step that ends it, and a hairline driven by it always completes.
   */
  readonly progress: number
  /** Whether play has been pressed (by the player or by `play()`) and not stopped. */
  readonly playing: boolean
  /** Ask for play: compute (if the tape is not whole) and let the player move the cursor. A no-op without a player. */
  play(): void
  /** Stop playing (the tape, and the priming of one, carry on). */
  pause(): void
  /**
   * Whether the session was let go of (`dispose()`) and nothing has used it since. Its `status` then reads
   * `'preparing'`, because the tape is at step 0 with nothing computed, which is not a restart: check this to tell a
   * disposed session from a fresh one. Cleared by `compute`, `play`, `advance` and `apply`. (It is a flag beside `status`
   * and not a member of {@link PlayStatus}, so a switch over `status` stays exhaustive.)
   */
  readonly disposed: boolean
  /**
   * Whether the session computes (`active` of the hook's options, default true). While false nothing computes: no
   * priming, the producer stopped (a worker pauses), `computing` reads false; the tape and the state are kept, and the
   * computing carries on when it is true again.
   */
  readonly active: boolean
  /**
   * Let go of everything: stop the producer, terminate a worker, drop the tape and the history buffers. The handle
   * stays valid, at step 0 with nothing computed and `disposed` true, and starts from nothing if it is used again
   * (`play`, `compute`). Unmounting does this by itself.
   */
  dispose(): void
}

/**
 * The members of a handle that never changes (a finished tape, a replay): ready, not playing, nothing to dispose.
 * Spread it into a handle made by hand.
 */
export const STILL_HANDLE: Pick<
  LiveHandle,
  'start' | 'status' | 'progress' | 'playing' | 'play' | 'pause' | 'dispose' | 'disposed' | 'active'
> = {
  start: 'end',
  status: 'ready',
  progress: 1,
  playing: false,
  disposed: false,
  active: true,
  play: () => undefined,
  pause: () => undefined,
  dispose: () => undefined,
}

/**
 * A session in a React page: the session, a version that changes as its tape grows, and the producer's controls.
 *
 * @index structure A live session: `live.session`, `live.head`, `live.compute()`, `live.apply()`
 */
export interface LiveSession<S, C extends Config> extends LiveHandle {
  readonly session: Session<S, C>
  /**
   * The configuration and seed the session on the tape was made with, which are the ones its figures are of. A change
   * to the method's options is applied after a pause (`DEBOUNCE_MS`, 250 ms) or held until `apply` (live mode), and
   * until then the page's options are newer than the run: draw what the run was made with from here, not from the
   * options. It is `session.config` and `session.seed`, read in one place, and changes when the session is remade.
   */
  readonly madeWith: { readonly config: C; readonly seed: number }
  /** Remake the session from the latest method and options now (it is also `session` from the next render). */
  apply(): Session<S, C>
}

/** A variant of a comparison as it runs: its label and its live session. */
export interface LiveVariant<S, C extends Config> {
  readonly label: string
  /** The variant's session; its controls act on the whole comparison, so the variants stay in step. */
  readonly live: LiveSession<S, C>
}

/**
 * Several variants of one method in a React page, under one producer: the same seed, data, budget and subscription
 * for all, so the knobs are the only difference. The handle's `head` is the step every variant has reached (see
 * `sharedHead`), `done` when all have ended, `computing` while any is.
 *
 * @index structure A comparison of variants under one producer: `live.variants`, `live.head`
 */
export interface LiveSessions<S, C extends Config> extends LiveHandle {
  readonly variants: readonly LiveVariant<S, C>[]
  /** Remake every variant from the latest method, options and variants now. */
  apply(): readonly Session<S, C>[]
}

/** How {@link useSession} treats a change of method or options. */
export interface SessionHookOptions {
  /** Under the `'play'` policy, hold a change until `apply` (the player's play) rather than apply it after a pause (default true). */
  readonly hold?: boolean
  /**
   * When the run is computed and where the cursor starts: `'play'` (live, the default), `'first'` (primed) or `'end'`
   * (eager). Omitted, an engine `mode: 'eager'` in the options means `'end'` and anything else `'play'`, so a page that
   * passes the reader's stored mode keeps working; given, it wins over the mode.
   */
  readonly start?: StartPolicy
  /** Called once per run when the whole tape exists (`status` is then `ready`, unless play was pressed). */
  readonly onReady?: () => void
  /**
   * Where the sessions are computed: a factory such as `workerProducer({ createWorker })` computes them in a Web Worker
   * and streams the tape back; omitted, they are computed on the main thread in the spare time of each frame. Keep the
   * factory stable (module level, or memoised): it is read when the sessions are made, and does not remake them.
   */
  readonly producer?: ProducerFactory
  /**
   * Whether the sessions compute (default true). While false nothing computes: no priming, the producer stopped (a
   * worker pauses), the player's clock off; the tape, the state and a change of options held are kept, and the work
   * carries on when it is true again. The declarative form of `halt()`, for a session on a page that is hidden (a slide
   * that is not showing), where `dispose()` would throw the tape away. `step` by hand still computes.
   */
  readonly active?: boolean
}

/** Milliseconds of computation per animation frame while computing ahead and playing: the rest of the frame is the page's. */
const FRAME_MS = 12
/** The same while priming and nothing plays (a whole run in the background): less, so the page stays responsive. */
const PRIME_MS = 8
/** How often the version moves while computing ahead, so views redraw at a steady rate rather than every frame. */
const BUMP_MS = 100
/** The pause after the last change before an unheld change is applied (one remake per slider drag). */
const DEBOUNCE_MS = 250

// A producer whose owner unmounted is disposed after a microtask, not at once: a strict-mode double effect runs its
// cleanup and its setup in the same task, and the setup takes the producer off this list, so it is not disposed.
const leaving = new WeakSet<object>()

// An engine `mode: 'eager'` computes when the session is made, on the spot; the hook computes it in slices instead (the
// `'end'` policy), so the session is made live.
function forProducer<S, C extends Config>(options: SessionOptions<S, C>): SessionOptions<S, C> {
  return options.mode === 'eager' ? { ...options, mode: 'live' } : options
}

function same(a: readonly unknown[], b: readonly unknown[]): boolean {
  return a.length === b.length && a.every((v, i) => Object.is(v, b[i]))
}

/**
 * Whether a handle is a comparison of several sessions.
 *
 * @param live - A session or a comparison.
 * @returns Whether it is a comparison.
 */
export function isComparison<S, C extends Config>(
  live: LiveSession<S, C> | LiveSessions<S, C>,
): live is LiveSessions<S, C> {
  return 'variants' in live
}

/**
 * The variants of a session or a comparison: a single session is a comparison of one, unlabelled.
 *
 * @param live - A session or a comparison.
 * @returns The variants.
 */
export function variantsOf<S, C extends Config>(
  live: LiveSession<S, C> | LiveSessions<S, C>,
): readonly LiveVariant<S, C>[] {
  return isComparison(live) ? live.variants : [{ label: '', live }]
}

/**
 * Several variants of one method, live, under one producer that keeps them in step: each frame, the variant furthest
 * behind takes the next slice (see `advanceInStep`), so the heads never drift more than a few steps apart and the
 * shared cursor, clamped to the slowest, hardly waits. Every variant has the same seed, data, budget and subscription
 * (the shared `options`); its own `config` is laid over the shared one.
 *
 * @param method - The method.
 * @param options - The options every variant shares. An engine `mode: 'eager'` is the `'end'` start policy (computed in
 *   slices in the background, not on the spot).
 * @param variants - The variants: a label and the knobs each changes. A change to them remakes the comparison, as a
 *   change to `deps` does.
 * @param deps - What a remake depends on (the caller names them, as for `useMemo`); not the mode.
 * @param hook - The start policy, whether the `'play'` policy holds changes until `apply`, the producer, `onReady`.
 * @returns The comparison.
 * @index function React hook: run variants of one method in step: `useSessions(method, options, variants, deps)`
 */
export function useSessions<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  variants: readonly Variant<C>[],
  deps: readonly unknown[],
  hook: SessionHookOptions = {},
): LiveSessions<S, C> {
  const allDeps = [...deps, JSON.stringify(variants)]
  const factory: ProducerFactory = hook.producer ?? inlineProducer
  const start = startOf(hook.start, options.mode)
  const prime = start !== 'play'
  const latest = useRef({ method, options, variants, deps: allDeps, factory })
  latest.current = { method, options, variants, deps: allDeps, factory }
  const [producer, setProducer] = useState<Producer<S, C>>(() => factory(method, forProducer(options), variants))
  const current = useRef(producer)
  const applied = useRef(allDeps)
  const [version, setVersion] = useState(0)
  const [wanted, setComputing] = useState(false)
  const [disposed, setDisposed] = useState(false)
  const active = hook.active ?? true
  // Computing is what is wanted and allowed: an inactive session keeps wanting, and carries on when it is active again.
  const computing = wanted && active
  const [playing, setPlaying] = useState(false)
  const playingNow = useRef(playing)
  playingNow.current = playing
  const disposedNow = useRef(disposed)
  disposedNow.current = disposed
  const activeNow = useRef(active)
  activeNow.current = active
  const hooks = useRef(hook)
  hooks.current = hook
  const [lastMs, setLastMs] = useState(0)
  const bump = useCallback(() => setVersion((v) => v + 1), [])
  const apply = useCallback((): readonly Session<S, C>[] => {
    const l = latest.current
    const next = l.factory(l.method, forProducer(l.options), l.variants)
    applied.current = l.deps
    current.current = next
    setProducer(next)
    setComputing(false)
    setPlaying(false)
    disposedNow.current = false
    setDisposed(false)
    bump()
    return next.sessions
  }, [bump])
  const advance = useCallback(
    (budget: SliceBudget): void => {
      const t0 = performance.now() // allow: the cost of a frame's work, shown in the status line; not on the tape
      current.current.advance(budget)
      setLastMs(performance.now() - t0) // allow: as above
      disposedNow.current = false
      setDisposed(false)
      bump()
    },
    [bump],
  )
  // A producer that is replaced or left is disposed (a worker terminated, its tape dropped), after a microtask so that a
  // strict-mode double effect, which cleans up and sets up again at once, keeps it.
  useEffect(() => {
    leaving.delete(producer)
    return () => {
      leaving.add(producer)
      queueMicrotask(() => {
        if (leaving.delete(producer)) producer.dispose()
      })
    }
  }, [producer])
  // The producer on the main thread: as many steps as fit in the frame's budget, every frame, until every run ends;
  // several variants take turns within the budget, the one furthest behind first. The version (and so the views) moves
  // at most every BUMP_MS.
  useEffect(() => {
    if (producer.background || !computing || typeof requestAnimationFrame !== 'function') return
    let lastBump: number | undefined
    let cost = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      // A frame queued before `dispose()` or before the session went inactive may still come: it computes nothing.
      if (disposedNow.current || !activeNow.current) return
      const sessions = current.current.sessions
      const t0 = performance.now() // allow: the frame budget and the cost of the frame's work; not on the tape
      current.current.advance({ ms: playingNow.current ? FRAME_MS : PRIME_MS, clock: () => performance.now() }) // allow: as above
      cost = performance.now() - t0 // allow: as above
      const ended = sessions.every((s) => s.status !== undefined)
      if (ended || lastBump === undefined || now - lastBump >= BUMP_MS) {
        lastBump = now
        setLastMs(cost)
        bump()
      }
      if (ended) {
        setComputing(false)
        return
      }
      handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [computing, producer, bump])
  // A producer in a worker computes by itself: told to start or stop with `computing`, it reports back as the tape
  // grows (the version moves at most every BUMP_MS), and the hook stops computing when every run has ended.
  useEffect(() => {
    if (!producer.background) return
    let last = Number.NEGATIVE_INFINITY
    let timer: ReturnType<typeof setTimeout> | undefined
    const fire = (): void => {
      timer = undefined
      last = performance.now() // allow: the rate views redraw at; not on the tape
      setLastMs(producer.lastMs)
      bump()
      if (producer.sessions.every((s) => s.status !== undefined)) setComputing(false)
    }
    const off = producer.subscribe(() => {
      const wait = BUMP_MS - (performance.now() - last) // allow: as above
      if (wait <= 0) fire()
      else timer ??= setTimeout(fire, wait)
    })
    return () => {
      off()
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [producer, bump])
  useEffect(() => {
    if (!producer.background) return
    if (computing) producer.start({ eager: prime })
    else producer.stop()
  }, [computing, producer, prime])
  // Priming (the `'first'` and `'end'` policies) computes the whole run as soon as the session is made, and when the
  // policy is switched on later (the player's eager toggle): in time slices on the main thread, or in the worker, which
  // is told to ignore the lookahead.
  // biome-ignore lint/correctness/useExhaustiveDependencies: a remake (a new producer) is primed again
  useEffect(() => {
    if (prime && active && !disposedNow.current && current.current.sessions.some((s) => s.status === undefined)) {
      setComputing(true)
    }
  }, [prime, active, producer])
  const stale = !same(allDeps, applied.current)
  const held = stale && start === 'play' && (hook.hold ?? true)
  // An unheld change is applied after a pause, and each further change restarts the pause.
  const changes = useRef({ deps: allDeps, n: 0 })
  if (!same(allDeps, changes.current.deps)) changes.current = { deps: allDeps, n: changes.current.n + 1 }
  const change = changes.current.n
  // biome-ignore lint/correctness/useExhaustiveDependencies: each change restarts the pause
  useEffect(() => {
    if (!stale || held) return
    const handle = setTimeout(apply, DEBOUNCE_MS)
    return () => clearTimeout(handle)
  }, [stale, held, change, apply])
  const compute = (): void => {
    disposedNow.current = false
    setDisposed(false)
    if (current.current.sessions.some((s) => s.status === undefined)) setComputing(true)
  }
  const halt = (): void => setComputing(false)
  const follow = (step: number): void => current.current.follow(step)
  const play = (): void => {
    setPlaying(true)
    compute()
  }
  const pause = (): void => setPlaying(false)
  const dispose = (): void => {
    current.current.dispose()
    disposedNow.current = true
    setDisposed(true)
    setComputing(false)
    setPlaying(false)
    bump()
  }
  const sessions = producer.sessions
  const head = sharedHead(sessions.map((s) => ({ head: s.t, done: s.status !== undefined })))
  const done = sessions.every((s) => s.status !== undefined)
  const progress = progressOf(head, done, options.steps ?? DEFAULT_STEPS)
  const status = statusOf(playing, done)
  const handle = {
    version,
    computing,
    compute,
    halt,
    advance,
    lastMs,
    stale,
    held,
    follow,
    producer: { kind: producer.kind, ...(producer.fallback === undefined ? {} : { fallback: producer.fallback }) },
    error: producer.error,
    probed: producer.probed,
    start,
    status,
    progress,
    playing,
    play,
    pause,
    disposed,
    active,
    dispose,
  }
  const list = sessions.map(
    (s, i): LiveVariant<S, C> => ({
      label: producer.labels[i] ?? '',
      live: {
        ...handle,
        session: s,
        madeWith: { config: s.config, seed: s.seed },
        head: s.t,
        done: s.status !== undefined,
        apply: () => apply()[i] ?? s,
      },
    }),
  )
  // `onReady` fires once per run, when the whole tape exists (a remake is a new run).
  const announced = useRef<object | undefined>(undefined)
  useEffect(() => {
    if (!done || announced.current === producer) return
    announced.current = producer
    hooks.current.onReady?.()
  }, [done, producer])
  return { ...handle, variants: list, head, done, apply }
}

/** The one variant of a session on its own. */
const SINGLE: readonly Variant<Config>[] = [{ label: '', config: {} }]

/**
 * A live session of a method.
 *
 * @param method - The method.
 * @param options - The session's options. An engine `mode: 'eager'` is the `'end'` start policy (computed in slices in
 *   the background, not on the spot).
 * @param deps - What a remake depends on (the caller names them, as for `useMemo`); not the mode.
 * @param hook - The start policy (`start: 'play' | 'first' | 'end'`), whether the `'play'` policy holds changes until
 *   `apply`, the producer, `onReady`.
 * @returns The live session, with its `status` and `progress`.
 * @index function React hook: run a method live: `useSession(method, { config, steps }, [config])`
 */
export function useSession<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  deps: readonly unknown[],
  hook: SessionHookOptions = {},
): LiveSession<S, C> {
  const [only] = useSessions(method, options, SINGLE as readonly Variant<C>[], deps, hook).variants
  if (only === undefined) throw new OmaiError('value', 'useSession: the session was not made')
  return only.live
}
