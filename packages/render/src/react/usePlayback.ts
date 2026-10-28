/**
 * Playback controllers for the `Player`: one interface over precomputed frames (`usePlayback`, `useTimelinePlayback`)
 * and a live algorithm stepped as it plays (`useLiveRun`, through `iterate`, so its states match an uninterrupted run
 * with the same seed), with play, pause, step, seek and loop, and the animation-frame ticker they share (`useTicker`).
 */

import type { Frame, Timeline } from 'mdml-engine/scene'
import { type Algorithm, iterate } from 'mdml-engine/trace'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { advanceFrame, framesDue } from './playhead.ts'

/**
 * What a {@link Player} drives: the same controls over a precomputed sequence of frames ({@link usePlayback},
 * {@link useTimelinePlayback}) and over a live algorithm stepped as it plays ({@link useLiveRun}).
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
}

/**
 * Call `onFrames(n)` with the frames due while `playing`, from the display's animation frames (their timestamps are
 * the clock, so nothing here reads one).
 *
 * @param playing - Whether to tick.
 * @param fps - The target frames per second.
 * @param onFrames - Called with the frames due (the latest callback is used, so it may close over fresh state).
 * @param cap - The most frames due in one tick (default 4), so a tab back from the background does not race ahead.
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
  /** Start playing at once (default false). */
  readonly autoplay?: boolean
}

/**
 * Playback over `count` precomputed frames: play and pause at a target rate, step, seek, loop or stop at the end.
 * Playing from the last frame starts again from the first.
 *
 * @param count - The number of frames.
 * @param options - Rate, looping and autoplay.
 * @returns The controller.
 */
export function usePlayback(count: number, options: PlaybackOptions = {}): PlaybackController {
  const { loop = false } = options
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(options.autoplay ?? false)
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
 */
export function useTimelinePlayback(
  timeline: Timeline,
  options: PlaybackOptions = {},
): { readonly controller: PlaybackController; readonly current: Frame | undefined } {
  const controller = usePlayback(timeline.frames.length, { fps: timeline.fps, ...options })
  return { controller, current: timeline.frames[controller.frame] }
}

/** Options of {@link useLiveRun}. */
export interface LiveRunOptions extends PlaybackOptions {
  readonly maxSteps: number
  readonly seed?: number
  /** Algorithm steps per frame while playing (default 1); every state is kept, so stepping back moves one step. */
  readonly stepsPerFrame?: number
  /** The most states kept for stepping back and scrubbing (default 2000; older ones are dropped). */
  readonly history?: number
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
}

/**
 * A live algorithm as a playback: each frame takes `stepsPerFrame` steps of the algorithm (through `iterate`, so the
 * states are exactly those of an uninterrupted run with the same seed) and keeps the states, so the reader can step
 * back, scrub, and play forward again into new steps. A change of algorithm or seed starts a fresh run.
 *
 * @param algorithm - The algorithm (keep its identity stable across renders, e.g. with `useMemo`).
 * @param options - The step budget, seed, rate, steps per frame and history length.
 * @returns The controller, the state shown, its step and the run's status.
 */
export function useLiveRun<S>(algorithm: Algorithm<S>, options: LiveRunOptions): LiveRun<S> {
  const { maxSteps, seed = 0, stepsPerFrame = 1, history = 2000 } = options
  const make = useCallback(() => {
    const gen = iterate(algorithm, { maxSteps, seed })
    const first = gen.next()
    const item = first.done === true ? undefined : first.value
    if (item === undefined) throw new Error(`${algorithm.name}: iterate yielded nothing`)
    return { gen, states: [item] }
  }, [algorithm, maxSteps, seed])
  const run = useRef<ReturnType<typeof make> | null>(null)
  const [version, setVersion] = useState(0)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(options.autoplay ?? false)
  const [fps, setFps] = useState(options.fps ?? 10)
  const madeFor = useRef<typeof make | null>(null)
  if (madeFor.current !== make) {
    run.current = make()
    madeFor.current = make
  }
  const r = run.current as ReturnType<typeof make>
  const last = r.states[r.states.length - 1] as (typeof r.states)[number]
  const finished = last.status !== undefined
  // A fresh run (new algorithm or seed) starts from its first state.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset exactly when the run is remade
  useEffect(() => setFrame(0), [make])

  /** Compute up to `n` more states, dropping the oldest past the history length; returns how many were dropped. */
  const extend = (n: number): number => {
    let added = 0
    for (let i = 0; i < n; i++) {
      const tail = r.states[r.states.length - 1]
      if (tail === undefined || tail.status !== undefined) break
      const next = r.gen.next()
      if (next.done === true) break
      r.states.push(next.value)
      added++
    }
    const excess = Math.max(r.states.length - history, 0)
    if (excess > 0) r.states.splice(0, excess)
    if (added > 0) setVersion((v) => v + 1)
    return excess
  }

  const count = r.states.length
  const shown = Math.min(frame, count - 1)
  const forward = (frames: number): void => {
    let at = shown
    for (let k = 0; k < frames; k++) {
      if (at < r.states.length - 1) at++
      else {
        at -= extend(stepsPerFrame)
        if (at < r.states.length - 1) at = r.states.length - 1
        else {
          setPlaying(false)
          break
        }
      }
    }
    setFrame(Math.min(at, r.states.length - 1))
  }
  useTicker(playing, fps, forward)
  const item = r.states[shown] ?? last
  const controller: PlaybackController = {
    frame: shown,
    count,
    playing,
    fps,
    open: !finished,
    play: () => {
      if (finished && shown >= count - 1) setFrame(0)
      setPlaying(true)
    },
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    step: (dir) => {
      setPlaying(false)
      if (dir < 0) setFrame(Math.max(shown - 1, 0))
      else forward(1)
    },
    seek: (f) => setFrame(Math.min(Math.max(Math.round(f), 0), count - 1)),
    setFps,
    reset: () => {
      setPlaying(false)
      run.current = make()
      setFrame(0)
      setVersion((v) => v + 1)
    },
  }
  void version
  return { controller, state: item.state, t: item.t, status: last.status }
}
