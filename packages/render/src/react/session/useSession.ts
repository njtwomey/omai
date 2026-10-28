/**
 * A React hook over an engine session, in two halves that know nothing of each other's pace: a producer that computes
 * the run ahead in the spare time of each animation frame (so the tape may stand at step 1000 while the page shows
 * step 100), and the tape itself, versioned as it grows, that the player and the panels read at their own rate. A
 * change of method or options is applied when the caller asks (live mode holds it until play), or after a short pause
 * (eager mode, and live mode without holding, so dragging a slider remakes the session once).
 */

import {
  type Config,
  type Method,
  type Session,
  type SessionOptions,
  type SliceBudget,
  session,
} from 'mdml-engine/trace'
import { useCallback, useEffect, useRef, useState } from 'react'

/** A session in a React page: the session, a version that changes as its tape grows, and the producer's controls. */
export interface LiveSession<S, C extends Config> {
  readonly session: Session<S, C>
  /** Changes whenever the tape grows or the session is remade, so memoised views recompute. */
  readonly version: number
  /** The head of the tape: the step computed so far. */
  readonly head: number
  readonly done: boolean
  /** Whether the producer is computing ahead in the background. */
  readonly computing: boolean
  /** Start computing ahead in the spare time of each animation frame, until the run ends. */
  compute(): void
  /** Stop computing ahead (the tape stays). */
  halt(): void
  /** Compute a slice now, on the spot. */
  advance(budget: SliceBudget): void
  /** Milliseconds the last slice took (the cost of one frame's work). */
  readonly lastMs: number
  /** The method or its options have changed since the session was made: held until `apply`, or about to be applied. */
  readonly stale: boolean
  /** A change is held until `apply` (live mode, holding). */
  readonly held: boolean
  /** Remake the session from the latest method and options now (it is also `session` from the next render). */
  apply(): Session<S, C>
}

/** How {@link useSession} treats a change of method or options. */
export interface SessionHookOptions {
  /** In live mode, hold a change until `apply` (the player's play) rather than apply it after a pause (default true). */
  readonly hold?: boolean
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
  const latest = useRef({ method, options, deps })
  latest.current = { method, options, deps }
  const [s, setS] = useState<Session<S, C>>(() => session(method, options))
  const current = useRef(s)
  const applied = useRef(deps)
  const [version, setVersion] = useState(0)
  const [computing, setComputing] = useState(false)
  const [lastMs, setLastMs] = useState(0)
  const bump = useCallback(() => setVersion((v) => v + 1), [])
  const apply = useCallback((): Session<S, C> => {
    const l = latest.current
    const next = session(l.method, l.options)
    applied.current = l.deps
    current.current = next
    setS(next)
    setComputing(false)
    bump()
    return next
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
  // The producer: as many steps as fit in the frame's budget, every frame, until the run ends. The version (and so
  // the views) moves at most every BUMP_MS.
  // biome-ignore lint/correctness/useExhaustiveDependencies: restarted for each session
  useEffect(() => {
    if (!computing || typeof requestAnimationFrame !== 'function') return
    let lastBump: number | undefined
    let cost = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      const sess = current.current
      if (sess.status === undefined) {
        const t0 = performance.now() // allow: the frame budget and the cost of the frame's work; not on the tape
        sess.advance({ steps: Number.POSITIVE_INFINITY, ms: FRAME_MS, clock: () => performance.now() }) // allow: as above
        cost = performance.now() - t0 // allow: as above
      }
      const ended = sess.status !== undefined
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
  }, [computing, s, bump])
  const mode = options.mode ?? 'live'
  // Eager switched on finishes the current run on the spot (a session made in eager mode is already whole).
  // biome-ignore lint/correctness/useExhaustiveDependencies: checked again for each session
  useEffect(() => {
    if (mode === 'eager' && current.current.status === undefined) {
      setComputing(false)
      advance({ steps: Number.POSITIVE_INFINITY })
    }
  }, [mode, s, advance])
  const stale = !same(deps, applied.current)
  const held = stale && mode === 'live' && (hook.hold ?? true)
  // An unheld change is applied after a pause, and each further change restarts the pause.
  const changes = useRef({ deps, n: 0 })
  if (!same(deps, changes.current.deps)) changes.current = { deps, n: changes.current.n + 1 }
  const change = changes.current.n
  // biome-ignore lint/correctness/useExhaustiveDependencies: each change restarts the pause
  useEffect(() => {
    if (!stale || held) return
    const handle = setTimeout(apply, DEBOUNCE_MS)
    return () => clearTimeout(handle)
  }, [stale, held, change, apply])
  return {
    session: s,
    version,
    head: s.t,
    done: s.status !== undefined,
    computing,
    compute: () => {
      if (current.current.status === undefined) setComputing(true)
    },
    halt: () => setComputing(false),
    advance,
    lastMs,
    stale,
    held,
    apply,
  }
}
