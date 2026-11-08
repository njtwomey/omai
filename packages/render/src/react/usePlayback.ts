/**
 * Playback controllers for the `Player`: one interface over precomputed frames (`usePlayback`, `useTimelinePlayback`)
 * and a live algorithm stepped as it plays (`useLiveRun`, through `iterate`, so its states match an uninterrupted run
 * with the same seed), with play, pause, step, seek and loop, and the animation-frame ticker they share (`useTicker`).
 */

import type { Json } from 'omai-engine/base'
import type { Frame, Timeline } from 'omai-engine/scene'
import { type Algorithm, iterate, type RunStatus } from 'omai-engine/trace'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { prefersReducedMotion } from './_motion.ts'
import { advanceFrame, framesDue } from './playhead.ts'
import { RingBuffer } from './ring.ts'
import { type PlayStatus, progressOf, type StartPolicy, statusOf } from './start.ts'

/**
 * What a {@link Player} drives: the same controls over a precomputed sequence of frames ({@link usePlayback},
 * {@link useTimelinePlayback}) and over a live algorithm stepped as it plays ({@link useLiveRun}).
 *
 * @index structure The state and verbs of a playback: `c.frame`, `c.seek(n)`, `c.toggle()`
 */
export interface PlaybackController {
  /** The frame shown. */
  readonly frame: number
  /** The frames available to scrub over (for a live run, the steps computed so far). */
  readonly count: number
  readonly playing: boolean
  /** The target frames per second. */
  readonly fps: number
  /** Whether more frames may come (a live run that has not finished). */
  readonly open: boolean
  readonly play: () => void
  readonly pause: () => void
  readonly toggle: () => void
  readonly step: (dir: 1 | -1) => void
  readonly seek: (frame: number) => void
  readonly setFps: (fps: number) => void
  readonly reset: () => void
  /** Eager (the whole run computed at once and its end shown) against live, when the controller computes anything. */
  readonly eager?: boolean
  /** Switch eager on or off; absent when there is nothing to compute (precomputed frames), and so no button. */
  readonly setEager?: (on: boolean) => void
  /** Reset the page's controls to their defaults; absent when the page has none, and so no button. */
  readonly resetForm?: () => void
}

/**
 * Call `onFrames(n)` with the frames due while `playing`, from the display's animation frames (their timestamps are
 * the clock, so nothing here reads one). The clock does not wait for drawing: a long frame makes several frames due at
 * once (up to `cap`), so the caller can skip frames rather than fall behind.
 *
 * @param playing - Whether to tick.
 * @param fps - The target frames per second.
 * @param onFrames - Called with the frames due (the latest callback is used, so it may close over fresh state).
 * @param cap - The most frames due in one tick (default 4), so a tab back from the background does not race ahead; a
 *   caller that can skip frames cheaply passes more, so a slow frame does not slow the clock.
 */
export function useTicker(playing: boolean, fps: number, onFrames: (n: number) => void, cap = 4): void {
  const callback = useRef(onFrames)
  callback.current = onFrames
  useEffect(() => {
    if (!playing || typeof requestAnimationFrame !== 'function') return
    let last: number | undefined
    let carry = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      if (last !== undefined) {
        const due = framesDue(carry + (now - last), fps, cap)
        carry = due.carry
        if (due.frames > 0) callback.current(due.frames)
      }
      last = now
      handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [playing, fps, cap])
}

/** Options of {@link usePlayback}. */
export interface PlaybackOptions {
  /** The target frames per second (default 10). */
  readonly fps?: number
  /** Start over from the first frame at the end (default false: stop on the last frame). */
  readonly loop?: boolean
  /** Start playing at once (default false). A reader who prefers reduced motion gets a paused player whatever this says. */
  readonly autoplay?: boolean
  /**
   * Whether the clock runs (default true), with the meaning it has on `useLiveRun` and `useSession`. While false the
   * clock is off and nothing advances; the frame, the rate and `playing` are kept (a player that was playing is still
   * `playing`), and it carries on from where it stood when this is true again, unless the caller paused it meanwhile.
   * Use it for a player on a page that is hidden (a slide that is not showing). A step or a seek by hand still moves.
   */
  readonly active?: boolean
}

/**
 * Playback over `count` precomputed frames: play and pause at a target rate, step, seek, loop or stop at the end.
 * Playing from the last frame starts again from the first.
 *
 * The controller's state is the single source of truth: `pause()`, `seek()`, `step()` and `reset()` win over a tick
 * that was already queued before them (a tick reads the latest frame and the playing flag, not what the last render
 * saw), so `pause(); seek(0)` leaves the player paused at frame 0 whatever frame callback comes after.
 *
 * @param count - The number of frames.
 * @param options - Rate, looping, autoplay and `active`.
 * @returns The controller.
 * @index function React hook: drive n precomputed frames with play, step and seek: `usePlayback(n)`
 */
export function usePlayback(count: number, options: PlaybackOptions = {}): PlaybackController {
  const { loop = false, active = true } = options
  const [frame, setFrame] = useState(0)
  const [playing, setPlayingState] = useState(() => (options.autoplay ?? false) && !prefersReducedMotion())
  const [fps, setFps] = useState(options.fps ?? 10)
  const shown = Math.min(frame, Math.max(count - 1, 0))
  // What a callback queued before the latest render must read: the controller's state, kept in refs and set at the
  // moment of the call, not when the render that follows it commits.
  const frameAt = useRef(0)
  const playingNow = useRef(playing)
  const latest = useRef({ count, loop, active })
  latest.current = { count, loop, active }
  const setPlaying = useCallback((on: boolean): void => {
    playingNow.current = on
    setPlayingState(on)
  }, [])
  const moveTo = useCallback((to: number): void => {
    frameAt.current = to
    setFrame(to)
  }, [])
  const shownNow = useCallback((): number => Math.min(frameAt.current, Math.max(latest.current.count - 1, 0)), [])
  useTicker(playing && active && count > 1, fps, (n) => {
    const now = latest.current
    if (!playingNow.current || !now.active || now.count <= 1) return
    const next = advanceFrame(shownNow(), n, now.count, now.loop)
    moveTo(next.frame)
    if (next.ended) setPlaying(false)
  })
  const verbs = useMemo(() => {
    const play = (): void => {
      const { count: n, loop: wrap } = latest.current
      if (!wrap && shownNow() >= n - 1) moveTo(0)
      setPlaying(true)
    }
    const pause = (): void => setPlaying(false)
    return {
      play,
      pause,
      toggle: (): void => (playingNow.current ? pause() : play()),
      step: (dir: 1 | -1): void => {
        setPlaying(false)
        const { count: n, loop: wrap } = latest.current
        moveTo(advanceFrame(shownNow(), dir, n, wrap).frame)
      },
      seek: (f: number): void => moveTo(Math.min(Math.max(Math.round(f), 0), Math.max(latest.current.count - 1, 0))),
      reset: (): void => {
        setPlaying(false)
        moveTo(0)
      },
    }
  }, [moveTo, setPlaying, shownNow])
  return useMemo(
    () => ({ frame: shown, count, playing, fps, open: false, ...verbs, setFps }),
    [shown, count, playing, fps, verbs],
  )
}

/**
 * Playback of a precomputed `Timeline` at its own frame rate, with the frame to draw.
 *
 * @param timeline - The timeline.
 * @param options - Looping and autoplay (the rate defaults to the timeline's).
 * @returns The controller and the current frame.
 * @index function React hook: play a Timeline, giving the frame to draw: `useTimelinePlayback(timeline)`
 */
export function useTimelinePlayback(
  timeline: Timeline,
  options: PlaybackOptions = {},
): { readonly controller: PlaybackController; readonly current: Frame | undefined } {
  const controller = usePlayback(timeline.frames.length, { fps: timeline.fps, ...options })
  return { controller, current: timeline.frames[controller.frame] }
}

/** Milliseconds of computation per animation frame while priming a run in the background. */
const PRIME_MS = 8

/** A state of a run, as `iterate` yields it. */
type Item<S> = { readonly t: number; readonly state: S; readonly status?: RunStatus }

// The item as the history should hold it: its state through the algorithm's `snapshot`, for an algorithm that changes
// its state in place (every frame would otherwise show the latest one).
function kept<S>(algorithm: Algorithm<S>, item: Item<S>): Item<S> {
  return algorithm.snapshot === undefined ? item : { ...item, state: algorithm.snapshot(item.state) }
}

/** Options of {@link useLiveRun}. */
export interface LiveRunOptions extends PlaybackOptions {
  readonly maxSteps: number
  readonly seed?: number
  /** Algorithm steps per frame while playing (default 1); every state is kept, so stepping back moves one step. */
  readonly stepsPerFrame?: number
  /**
   * The most states kept for stepping back and scrubbing (default 2000), in a ring buffer: past it the oldest are
   * dropped, stepping back stops at the oldest kept (`oldest`), and the controller's frames count from there. A run
   * that is primed (`start` `'first'` or `'end'`) keeps its whole run, whatever this says.
   */
  readonly history?: number
  /**
   * When the run is computed and where the cursor starts: `'play'` (default) computes as it plays; `'first'` computes
   * the whole run in the background (in slices of each animation frame) and parks at step 0 until play; `'end'`
   * computes it the same way and shows its end when it is done. `autoplay` plays from the start regardless.
   */
  readonly start?: StartPolicy
  /** Called once when the whole run exists. */
  readonly onReady?: () => void
  /**
   * Whether the run computes (default true). While false nothing computes: no priming, and a playing run does not move
   * on (its clock is off); the states and the cursor are kept, and it carries on from where it stood when this is true
   * again. Use it for a run on a page that is hidden (a slide that is not showing), where {@link LiveRun.dispose}
   * would throw the states away. A step taken by hand (`controller.step(1)`) still computes.
   */
  readonly active?: boolean
}

/** What a run of {@link useLiveRun} was made with (see {@link LiveRun.madeWith}). */
export interface LiveRunMadeWith<S> {
  /** The algorithm of the run. */
  readonly algorithm: Algorithm<S>
  /** The algorithm's `config`, as the run was made with it. */
  readonly config: Json
  /** The seed of the run. */
  readonly seed: number
  /** The step budget (`maxSteps`). */
  readonly maxSteps: number
  /** The states the run keeps (`history`, or the whole run when it is primed), which is what its ring buffer was made for. */
  readonly capacity: number
}

/** What {@link useLiveRun} returns. */
export interface LiveRun<S> {
  readonly controller: PlaybackController
  /** The state shown (the latest computed, unless the reader has stepped back or scrubbed). */
  readonly state: S
  /** The algorithm's step number of the state shown. */
  readonly t: number
  /** The run's status once it has finished. */
  readonly status: string | undefined
  /** Where the run is (the run's own `status` is above): preparing, ready or running; see `PlayStatus`. */
  readonly phase: PlayStatus
  /**
   * The fraction of the run computed, 0 to 1. While the run goes on it is the step reached over `maxSteps`; once the
   * run has ended, whatever step it ended at (it converged after 8 of 1000), it is exactly 1, so an indicator of it
   * always completes. A run that ends early therefore jumps to 1 on the step that ends it.
   */
  readonly progress: number
  /** The algorithm's step number of the oldest state still kept (0 until the history buffer starts dropping). */
  readonly oldest: number
  /**
   * Whether the run was disposed and nothing has used it since: the states are gone, nothing computes, and `phase`
   * reads `'preparing'` only because the run is at step 0 (it is not a restart). Cleared by `controller.play()`,
   * a step forward, `controller.reset()` and a new algorithm or seed.
   */
  readonly disposed: boolean
  /**
   * What the run on the tape was made with: the algorithm and its `config`, the `seed`, the step budget and the number
   * of states kept, which are everything the hook keys a run by (a change to any of them remakes the
   * run, which starts again from its first state; `stepsPerFrame`, `fps` and the like do not). It is read in one
   * place, and switches in the render that remakes the run (also `controller.reset()` and a play after `dispose()`),
   * so a figure that draws what the run was made with reads it here and not from the options, which may be newer.
   * The hook holds nothing back and has no pause before a remake (a caller that debounces a slider does so before
   * the options reach the hook), so the two differ only for the render in between.
   *
   * It holds the options, not what depends on them: a figure that draws something computed from the data (the true
   * function, the dataset the algorithm was given) must take its own snapshot when the run is replaced, in the
   * producer that makes the algorithm or in a ref the caller assigns in the same place.
   *
   * @example
   * ```tsx
   * // The data and its truth are the caller's, so the caller snapshots them with the algorithm, in one memo:
   * const made = useMemo(() => {
   *   const data = makeData(noise, seed)
   *   return { algorithm: fitting(data), truth: data.truth } // `truth` stays with the algorithm it belongs to
   * }, [noise, seed])
   * const live = useLiveRun(made.algorithm, { maxSteps: 200, seed, start: 'first' })
   * // `live.madeWith.seed` and `live.madeWith.config` are those of the run on the tape; `made.truth` goes with it only
   * // because `made` and the algorithm were replaced together. Reading `truth` from the current options would show
   * // the new data under the old run.
   * drawTruth(made.truth)
   * ```
   */
  readonly madeWith: LiveRunMadeWith<S>
  /**
   * Stop computing and release the states and the generator; the run starts again from step 0 if played. Safe to call
   * at any time, including from an animation frame or a ticker callback that was already queued.
   */
  dispose(): void
}

/**
 * A live algorithm as a playback: each frame takes `stepsPerFrame` steps of the algorithm (through `iterate`, so the
 * states are exactly those of an uninterrupted run with the same seed) and keeps the states in a ring buffer, so the
 * reader can step back, scrub, and play forward again into new steps. A change of algorithm or seed starts a fresh
 * run. With `start: 'first'` the whole run is computed in the background while the cursor waits at step 0 for play.
 * Unmounting releases the run.
 *
 * @param algorithm - The algorithm (keep its identity stable across renders, e.g. with `useMemo`).
 * @param options - The step budget, seed, rate, steps per frame, history length, start policy and `active`.
 * @returns The controller, the state shown, its step, the run's status, and where the run is (`phase`, `progress`).
 * @index function React hook: step an Algorithm live while it plays: `useLiveRun(algorithm, { maxSteps: 200 })`
 */
export function useLiveRun<S>(algorithm: Algorithm<S>, options: LiveRunOptions): LiveRun<S> {
  const { maxSteps, seed = 0, stepsPerFrame = 1, history = 2000, start = 'play', active = true } = options
  const prime = start !== 'play'
  const capacity = prime ? Math.max(history, maxSteps + 1) : history
  const make = useCallback(() => {
    const gen = iterate(algorithm, { maxSteps, seed })
    const first = gen.next()
    const item = first.done === true ? undefined : first.value
    if (item === undefined) throw new Error(`${algorithm.name}: iterate yielded nothing`)
    const states = new RingBuffer<typeof item>(capacity)
    states.push(kept(algorithm, item))
    const madeWith: LiveRunMadeWith<S> = { algorithm, config: algorithm.config, seed, maxSteps, capacity }
    return { gen, states, madeWith }
  }, [algorithm, maxSteps, seed, capacity])
  const makeNow = useRef(make)
  makeNow.current = make
  const run = useRef<ReturnType<typeof make> | null>(null)
  const [version, setVersion] = useState(0)
  // The cursor is an absolute index (states ever produced), so it stays on its state as the ring drops older ones. The
  // state is what renders; `at` is the same value, set at the moment of the call, and is what a callback reads: a tick
  // queued before a `seek` must continue from the sought step, not from the one the last render saw.
  const [cursor, setCursor] = useState(0)
  const at = useRef(0)
  const [playing, setPlayingState] = useState(() => (options.autoplay ?? false) && !prefersReducedMotion())
  const playingNow = useRef(playing)
  const activeNow = useRef(active)
  activeNow.current = active
  const setPlaying = useCallback((on: boolean): void => {
    playingNow.current = on
    setPlayingState(on)
  }, [])
  const moveTo = useCallback((to: number): void => {
    at.current = to
    setCursor(to)
  }, [])
  const [fps, setFps] = useState(options.fps ?? 10)
  const [released, setReleased] = useState(false)
  // Set by `release` before anything renders, so a frame callback queued before the dispose sees it at once.
  const disposed = useRef(false)
  const [epoch, setEpoch] = useState(0)
  const madeFor = useRef<typeof make | null>(null)
  if (madeFor.current !== make) {
    run.current?.gen.return(undefined)
    run.current = make()
    madeFor.current = make
    at.current = 0
  }
  const r = run.current as ReturnType<typeof make>
  const last = r.states.last as NonNullable<typeof r.states.last>
  const finished = last.status !== undefined
  // A fresh run (new algorithm or seed) starts from its first state (`at` was reset with it; a seek made since the
  // render is kept).
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset exactly when the run is remade
  useEffect(() => {
    disposed.current = false
    setCursor(at.current)
    setReleased(false)
  }, [make])

  /** Compute up to `n` more states (the ring drops the oldest past its capacity); returns how many were added. */
  const extend = (n: number): number => {
    const live = run.current
    if (live === null || disposed.current) return 0
    let added = 0
    for (let i = 0; i < n; i++) {
      const tail = live.states.last
      if (tail === undefined || tail.status !== undefined) break
      const next = live.gen.next()
      if (next.done === true) break
      live.states.push(kept(algorithm, next.value))
      added++
    }
    return added
  }

  /** The cursor, clamped to the states kept now. */
  const clamped = (states: { readonly dropped: number; readonly length: number }): number =>
    Math.min(Math.max(at.current, states.dropped), states.dropped + states.length - 1)
  const dropped = r.states.dropped
  const count = r.states.length
  const abs = Math.min(Math.max(cursor, dropped), dropped + count - 1)
  const shown = abs - dropped
  /** Move forward `frames` frames from the latest position (whatever the playing flag says). */
  const advance = (frames: number): void => {
    // A frame queued before `dispose` finds no run: there is nothing to move through.
    if (run.current === null || disposed.current) return
    const live = run.current.states
    let to = clamped(live)
    for (let k = 0; k < frames; k++) {
      if (to < live.dropped + live.length - 1) to++
      else if (extend(stepsPerFrame) > 0) {
        setVersion((v) => v + 1)
        to = live.dropped + live.length - 1
      } else {
        setPlaying(false)
        break
      }
    }
    moveTo(to)
  }
  // A tick queued before a `pause`, `seek`, `reset` or `dispose` (or before the run went inactive) may still come
  // after it: it does nothing then, and when it moves it starts from where the cursor is now.
  const tick = (frames: number): void => {
    if (!playingNow.current || !activeNow.current) return
    advance(frames)
  }
  // The clock may fall behind when frames are slow, and then runs the algorithm several frames' steps at once, but no
  // more than half a second's worth.
  useTicker(playing && active, fps, tick, Math.max(4, Math.ceil(fps / 2)))

  // Priming: the whole run, in slices of each animation frame, from the moment the run is made.
  // biome-ignore lint/correctness/useExhaustiveDependencies: restarts for a new run (`make`) or a reset (`epoch`); `extend` reads the latest run
  useEffect(() => {
    if (!prime || !active || released || typeof requestAnimationFrame !== 'function') return
    let handle = requestAnimationFrame(function tick() {
      const live = run.current
      if (live === null || disposed.current) return
      const t0 = performance.now() // allow: the frame's budget for computing ahead; not on the tape
      while (live.states.last?.status === undefined && performance.now() - t0 < PRIME_MS) extend(1) // allow: as above
      setVersion((v) => v + 1)
      if (live.states.last?.status === undefined) handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [prime, active, released, make, epoch])

  // Eager: the cursor goes to the end once the whole run exists, if it has not been moved.
  const parked = useRef(true)
  // biome-ignore lint/correctness/useExhaustiveDependencies: on the edge to finished
  useEffect(() => {
    if (finished && start === 'end' && parked.current) moveTo(Number.POSITIVE_INFINITY)
  }, [finished])

  const notified = useRef<unknown>(null)
  const onReady = useRef(options.onReady)
  onReady.current = options.onReady
  useEffect(() => {
    if (!finished || notified.current === r) return
    notified.current = r
    onReady.current?.()
  }, [finished, r])

  const release = useCallback((): void => {
    const old = run.current
    old?.gen.return(undefined)
    old?.states.clear()
    run.current = null
    disposed.current = true
    madeFor.current = null
    setPlaying(false)
    setReleased(true)
    moveTo(0)
    setVersion((v) => v + 1)
  }, [setPlaying, moveTo])
  // Leaving releases the run, after a microtask so that a strict-mode double effect, which cleans up and sets up again at
  // once, keeps it.
  const leaving = useRef(false)
  useEffect(() => {
    leaving.current = false
    return () => {
      leaving.current = true
      queueMicrotask(() => {
        if (!leaving.current) return
        const old = run.current
        old?.gen.return(undefined)
        old?.states.clear()
      })
    }
  }, [])

  const item = r.states.at(shown) ?? last
  // The verbs read the refs and the run as it is now, never what this render saw, so a controller kept from an earlier
  // render, or a call made before the render of the one before it, acts on the latest state.
  const controller: PlaybackController = {
    frame: shown,
    count,
    playing,
    fps,
    open: !finished,
    play: () => {
      parked.current = false
      disposed.current = false
      setReleased(false)
      const live = run.current?.states
      if (live !== undefined && live.last?.status !== undefined && clamped(live) >= live.dropped + live.length - 1) {
        moveTo(live.dropped)
      }
      setPlaying(true)
    },
    pause: () => setPlaying(false),
    toggle: () => (playingNow.current ? controller.pause() : controller.play()),
    step: (dir) => {
      parked.current = false
      setPlaying(false)
      if (dir < 0) {
        const live = run.current?.states
        if (live !== undefined) moveTo(Math.max(clamped(live) - 1, live.dropped))
      } else {
        disposed.current = false
        setReleased(false)
        advance(1)
      }
    },
    seek: (f) => {
      parked.current = false
      const live = run.current?.states
      moveTo(live === undefined ? 0 : live.dropped + Math.min(Math.max(Math.round(f), 0), live.length - 1))
    },
    setFps,
    reset: () => {
      parked.current = start === 'end'
      disposed.current = false
      setReleased(false)
      setPlaying(false)
      run.current?.gen.return(undefined)
      run.current = makeNow.current()
      moveTo(0)
      setEpoch((e) => e + 1)
      setVersion((v) => v + 1)
    },
  }
  void version
  return {
    controller,
    state: item.state,
    t: item.t,
    status: last.status,
    phase: statusOf(playing, finished),
    progress: progressOf(dropped + count - 1, finished, maxSteps),
    oldest: r.states.at(0)?.t ?? 0,
    disposed: released,
    madeWith: r.madeWith,
    dispose: release,
  }
}
