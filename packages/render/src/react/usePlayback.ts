/**
 * Playback controllers for the `Player`: one interface over precomputed frames (`usePlayback`, `useTimelinePlayback`)
 * and a live algorithm stepped as it plays (`useLiveRun`, through `iterate`, so its states match an uninterrupted run
 * with the same seed), with play, pause, step, seek and loop, and the animation-frame ticker they share (`useTicker`).
 */

import type { Frame, Timeline } from 'omai-engine/scene'
import { type Algorithm, iterate } from 'omai-engine/trace'
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
}

/**
 * Playback over `count` precomputed frames: play and pause at a target rate, step, seek, loop or stop at the end.
 * Playing from the last frame starts again from the first.
 *
 * @param count - The number of frames.
 * @param options - Rate, looping and autoplay.
 * @returns The controller.
 * @index function React hook: drive n precomputed frames with play, step and seek: `usePlayback(n)`
 */
export function usePlayback(count: number, options: PlaybackOptions = {}): PlaybackController {
  const { loop = false } = options
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(() => (options.autoplay ?? false) && !prefersReducedMotion())
  const [fps, setFps] = useState(options.fps ?? 10)
  const shown = Math.min(frame, Math.max(count - 1, 0))
  useTicker(playing && count > 1, fps, (n) => {
    const next = advanceFrame(shown, n, count, loop)
    setFrame(next.frame)
    if (next.ended) setPlaying(false)
  })
  const play = useCallback(() => {
    if (!loop && shown >= count - 1) setFrame(0)
    setPlaying(true)
  }, [loop, shown, count])
  const pause = useCallback(() => setPlaying(false), [])
  return useMemo(
    () => ({
      frame: shown,
      count,
      playing,
      fps,
      open: false,
      play,
      pause,
      toggle: () => (playing ? pause() : play()),
      step: (dir: 1 | -1) => {
        setPlaying(false)
        setFrame(advanceFrame(shown, dir, count, loop).frame)
      },
      seek: (f: number) => setFrame(Math.min(Math.max(Math.round(f), 0), Math.max(count - 1, 0))),
      setFps,
      reset: () => {
        setPlaying(false)
        setFrame(0)
      },
    }),
    [shown, count, playing, fps, play, pause, loop],
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
  /** The fraction of the run computed, 0 to 1 (of `maxSteps`; 1 once the run has ended). */
  readonly progress: number
  /** The algorithm's step number of the oldest state still kept (0 until the history buffer starts dropping). */
  readonly oldest: number
  /** Stop computing and release the states and the generator; the run starts again from step 0 if played. */
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
 * @param options - The step budget, seed, rate, steps per frame, history length and start policy.
 * @returns The controller, the state shown, its step, the run's status, and where the run is (`phase`, `progress`).
 * @index function React hook: step an Algorithm live while it plays: `useLiveRun(algorithm, { maxSteps: 200 })`
 */
export function useLiveRun<S>(algorithm: Algorithm<S>, options: LiveRunOptions): LiveRun<S> {
  const { maxSteps, seed = 0, stepsPerFrame = 1, history = 2000, start = 'play' } = options
  const prime = start !== 'play'
  const capacity = prime ? Math.max(history, maxSteps + 1) : history
  const make = useCallback(() => {
    const gen = iterate(algorithm, { maxSteps, seed })
    const first = gen.next()
    const item = first.done === true ? undefined : first.value
    if (item === undefined) throw new Error(`${algorithm.name}: iterate yielded nothing`)
    const states = new RingBuffer<typeof item>(capacity)
    states.push(item)
    return { gen, states }
  }, [algorithm, maxSteps, seed, capacity])
  const run = useRef<ReturnType<typeof make> | null>(null)
  const [version, setVersion] = useState(0)
  // The cursor is an absolute index (states ever produced), so it stays on its state as the ring drops older ones.
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(() => (options.autoplay ?? false) && !prefersReducedMotion())
  const [fps, setFps] = useState(options.fps ?? 10)
  const [released, setReleased] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const madeFor = useRef<typeof make | null>(null)
  if (madeFor.current !== make) {
    run.current?.gen.return(undefined)
    run.current = make()
    madeFor.current = make
  }
  const r = run.current as ReturnType<typeof make>
  const last = r.states.last as NonNullable<typeof r.states.last>
  const finished = last.status !== undefined
  // A fresh run (new algorithm or seed) starts from its first state.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset exactly when the run is remade
  useEffect(() => {
    setCursor(0)
    setReleased(false)
  }, [make])

  /** Compute up to `n` more states (the ring drops the oldest past its capacity); returns how many were added. */
  const extend = (n: number): number => {
    const live = run.current as ReturnType<typeof make>
    let added = 0
    for (let i = 0; i < n; i++) {
      const tail = live.states.last
      if (tail === undefined || tail.status !== undefined) break
      const next = live.gen.next()
      if (next.done === true) break
      live.states.push(next.value)
      added++
    }
    return added
  }

  const dropped = r.states.dropped
  const count = r.states.length
  const abs = Math.min(Math.max(cursor, dropped), dropped + count - 1)
  const shown = abs - dropped
  const positioned = useRef(abs)
  positioned.current = abs
  const forward = (frames: number): void => {
    let at = positioned.current
    for (let k = 0; k < frames; k++) {
      const live = (run.current as ReturnType<typeof make>).states
      if (at < live.dropped + live.length - 1) at++
      else if (extend(stepsPerFrame) > 0) {
        setVersion((v) => v + 1)
        at = live.dropped + live.length - 1
      } else {
        setPlaying(false)
        break
      }
    }
    positioned.current = at
    setCursor(at)
  }
  // The clock may fall behind when frames are slow, and then runs the algorithm several frames' steps at once, but no
  // more than half a second's worth.
  useTicker(playing, fps, forward, Math.max(4, Math.ceil(fps / 2)))

  // Priming: the whole run, in slices of each animation frame, from the moment the run is made.
  // biome-ignore lint/correctness/useExhaustiveDependencies: restarts for a new run (`make`) or a reset (`epoch`); `extend` reads the latest run
  useEffect(() => {
    if (!prime || released || typeof requestAnimationFrame !== 'function') return
    let handle = requestAnimationFrame(function tick() {
      const live = run.current as ReturnType<typeof make>
      const t0 = performance.now() // allow: the frame's budget for computing ahead; not on the tape
      while (live.states.last?.status === undefined && performance.now() - t0 < PRIME_MS) extend(1) // allow: as above
      setVersion((v) => v + 1)
      if (live.states.last?.status === undefined) handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [prime, released, make, epoch])

  // Eager: the cursor goes to the end once the whole run exists, if it has not been moved.
  const parked = useRef(true)
  // biome-ignore lint/correctness/useExhaustiveDependencies: on the edge to finished
  useEffect(() => {
    if (finished && start === 'end' && parked.current) setCursor(Number.POSITIVE_INFINITY)
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
    madeFor.current = null
    setPlaying(false)
    setReleased(true)
    setCursor(0)
    setVersion((v) => v + 1)
  }, [])
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
  const controller: PlaybackController = {
    frame: shown,
    count,
    playing,
    fps,
    open: !finished,
    play: () => {
      parked.current = false
      setReleased(false)
      if (finished && shown >= count - 1) setCursor(dropped)
      setPlaying(true)
    },
    pause: () => setPlaying(false),
    toggle: () => (playing ? controller.pause() : controller.play()),
    step: (dir) => {
      parked.current = false
      setPlaying(false)
      if (dir < 0) setCursor(Math.max(abs - 1, dropped))
      else forward(1)
    },
    seek: (f) => {
      parked.current = false
      setCursor(dropped + Math.min(Math.max(Math.round(f), 0), count - 1))
    },
    setFps,
    reset: () => {
      parked.current = start === 'end'
      setPlaying(false)
      run.current?.gen.return(undefined)
      run.current = make()
      setCursor(0)
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
    dispose: release,
  }
}
