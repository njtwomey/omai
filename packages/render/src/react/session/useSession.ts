/**
 * A React hook over an engine session: makes it when the method, configuration, seed, subscription or mode changes,
 * plays it live at a target number of steps per second (bounded by a time budget per animation frame), and bumps a
 * version so views re-read the tape.
 */

import {
  type Config,
  type Method,
  type Session,
  type SessionOptions,
  type SliceBudget,
  session,
} from 'mdml-engine/trace'
import { useCallback, useEffect, useMemo, useState } from 'react'

/** A session in a React page: the session, a version that changes as its tape grows, and playback. */
export interface LiveSession<S, C extends Config> {
  readonly session: Session<S, C>
  /** Changes whenever the tape grows, so memoised views recompute. */
  readonly version: number
  readonly playing: boolean
  readonly done: boolean
  /** Target steps per second while playing. */
  readonly fps: number
  setFps(fps: number): void
  play(): void
  pause(): void
  toggle(): void
  advance(budget: SliceBudget): void
  /** Milliseconds the last slice took (the cost of one frame's work). */
  readonly lastMs: number
}

/** Milliseconds of computation allowed per animation frame while playing. */
const FRAME_MS = 14

/**
 * A live session of a method.
 *
 * @param method - The method.
 * @param options - The session's options.
 * @param deps - What remaking the session depends on (the caller names them, as for `useMemo`).
 * @returns The live session.
 */
export function useSession<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  deps: readonly unknown[],
): LiveSession<S, C> {
  // biome-ignore lint/correctness/useExhaustiveDependencies: the caller names what the session depends on
  const s = useMemo(() => session(method, options), deps)
  const [version, setVersion] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [fps, setFps] = useState(60)
  const [lastMs, setLastMs] = useState(0)
  const bump = useCallback(() => setVersion((v) => v + 1), [])
  // A new session starts paused at whatever its mode computed up front.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset exactly when the session is remade
  useEffect(() => {
    setPlaying(false)
    bump()
  }, [s])
  const advance = useCallback(
    (budget: SliceBudget) => {
      const t0 = performance.now() // allow: the cost of a frame's work, shown in the status line; not on the tape
      s.advance(budget)
      setLastMs(performance.now() - t0) // allow: as above
      bump()
    },
    [s, bump],
  )
  useEffect(() => {
    if (!playing) return
    let last: number | undefined
    let carry = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      if (s.status !== undefined) {
        setPlaying(false)
        return
      }
      if (last !== undefined) {
        // Steps due at the target rate since the last frame (none until one is due), within the frame's time budget.
        const due = carry + ((now - last) * fps) / 1000
        const steps = Math.floor(due)
        carry = due - steps
        if (steps > 0) advance({ steps, ms: FRAME_MS, clock: () => performance.now() }) // allow: the frame budget
      }
      last = now
      handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [playing, s, fps, advance])
  return {
    session: s,
    version,
    playing,
    done: s.status !== undefined,
    fps,
    setFps,
    lastMs,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    advance,
  }
}
