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
 */

import { MdmlError } from 'mdml-engine/base'
import type { Config, Method, Session, SessionOptions, SliceBudget } from 'mdml-engine/trace'
import { useCallback, useEffect, useRef, useState } from 'react'
import { inlineProducer } from '../../producer/inline.ts'
import type { Variant } from '../../producer/make.ts'
import type { Producer, ProducerFactory } from '../../producer/producer.ts'
import { sharedHead } from '../../producer/step.ts'

export type { Variant }

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
}

/** A session in a React page: the session, a version that changes as its tape grows, and the producer's controls. */
export interface LiveSession<S, C extends Config> extends LiveHandle {
  readonly session: Session<S, C>
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
 */
export interface LiveSessions<S, C extends Config> extends LiveHandle {
  readonly variants: readonly LiveVariant<S, C>[]
  /** Remake every variant from the latest method, options and variants now. */
  apply(): readonly Session<S, C>[]
}

/** How {@link useSession} treats a change of method or options. */
export interface SessionHookOptions {
  /** In live mode, hold a change until `apply` (the player's play) rather than apply it after a pause (default true). */
  readonly hold?: boolean
  /**
   * Where the sessions are computed: a factory such as `workerProducer({ createWorker })` computes them in a Web Worker
   * and streams the tape back; omitted, they are computed on the main thread in the spare time of each frame. Keep the
   * factory stable (module level, or memoised): it is read when the sessions are made, and does not remake them.
   */
  readonly producer?: ProducerFactory
}

/** Milliseconds of computation per animation frame while computing ahead: the rest of the frame is the page's. */
const FRAME_MS = 12
/** How often the version moves while computing ahead, so views redraw at a steady rate rather than every frame. */
const BUMP_MS = 100
/** The pause after the last change before an unheld change is applied (one remake per slider drag). */
const DEBOUNCE_MS = 250

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
 * @param options - The options every variant shares. Eager mode computes every variant whole when it is made, and
 *   finishes the current ones on the spot when switched on.
 * @param variants - The variants: a label and the knobs each changes. A change to them remakes the comparison, as a
 *   change to `deps` does.
 * @param deps - What a remake depends on (the caller names them, as for `useMemo`); not the mode.
 * @param hook - Whether live mode holds changes until `apply`.
 * @returns The comparison.
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
  const latest = useRef({ method, options, variants, deps: allDeps, factory })
  latest.current = { method, options, variants, deps: allDeps, factory }
  const [producer, setProducer] = useState<Producer<S, C>>(() => factory(method, options, variants))
  const current = useRef(producer)
  const applied = useRef(allDeps)
  const [version, setVersion] = useState(0)
  const [computing, setComputing] = useState(false)
  const [lastMs, setLastMs] = useState(0)
  const bump = useCallback(() => setVersion((v) => v + 1), [])
  const apply = useCallback((): readonly Session<S, C>[] => {
    const l = latest.current
    const next = l.factory(l.method, l.options, l.variants)
    applied.current = l.deps
    current.current = next
    setProducer(next)
    setComputing(false)
    bump()
    return next.sessions
  }, [bump])
  const advance = useCallback(
    (budget: SliceBudget): void => {
      const t0 = performance.now() // allow: the cost of a frame's work, shown in the status line; not on the tape
      current.current.advance(budget)
      setLastMs(performance.now() - t0) // allow: as above
      bump()
    },
    [bump],
  )
  // A producer that is replaced or left is disposed (a worker terminated, its tape dropped); one that has not started
  // holds nothing, so the second mount of a strict-mode double effect finds it as new.
  useEffect(() => () => producer.dispose(), [producer])
  // The producer on the main thread: as many steps as fit in the frame's budget, every frame, until every run ends;
  // several variants take turns within the budget, the one furthest behind first. The version (and so the views) moves
  // at most every BUMP_MS.
  useEffect(() => {
    if (producer.background || !computing || typeof requestAnimationFrame !== 'function') return
    let lastBump: number | undefined
    let cost = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      const sessions = current.current.sessions
      const t0 = performance.now() // allow: the frame budget and the cost of the frame's work; not on the tape
      current.current.advance({ ms: FRAME_MS, clock: () => performance.now() }) // allow: as above
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
  const mode = options.mode ?? 'live'
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
    if (computing) producer.start({ eager: mode === 'eager' })
    else producer.stop()
  }, [computing, producer, mode])
  // Eager switched on finishes the current runs on the spot (sessions made in eager mode are already whole); a worker is
  // told to run them to the end without waiting for the cursor.
  useEffect(() => {
    if (mode === 'eager' && current.current.sessions.some((s) => s.status === undefined)) {
      if (producer.background) {
        setComputing(true)
        return
      }
      setComputing(false)
      advance({ steps: Number.POSITIVE_INFINITY })
    }
  }, [mode, producer, advance])
  const stale = !same(allDeps, applied.current)
  const held = stale && mode === 'live' && (hook.hold ?? true)
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
    if (current.current.sessions.some((s) => s.status === undefined)) setComputing(true)
  }
  const halt = (): void => setComputing(false)
  const follow = (step: number): void => current.current.follow(step)
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
  }
  const list = producer.sessions.map(
    (s, i): LiveVariant<S, C> => ({
      label: producer.labels[i] ?? '',
      live: {
        ...handle,
        session: s,
        head: s.t,
        done: s.status !== undefined,
        apply: () => apply()[i] ?? s,
      },
    }),
  )
  return {
    ...handle,
    variants: list,
    head: sharedHead(list.map((v) => v.live)),
    done: list.every((v) => v.live.done),
    apply,
  }
}

/** The one variant of a session on its own. */
const SINGLE: readonly Variant<Config>[] = [{ label: '', config: {} }]

/**
 * A live session of a method.
 *
 * @param method - The method.
 * @param options - The session's options. Eager mode computes a session whole when it is made, and finishes the
 *   current one on the spot when switched on.
 * @param deps - What a remake depends on (the caller names them, as for `useMemo`); not the mode.
 * @param hook - Whether live mode holds changes until `apply`.
 * @returns The live session.
 */
export function useSession<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  deps: readonly unknown[],
  hook: SessionHookOptions = {},
): LiveSession<S, C> {
  const [only] = useSessions(method, options, SINGLE as readonly Variant<C>[], deps, hook).variants
  if (only === undefined) throw new MdmlError('value', 'useSession: the session was not made')
  return only.live
}
