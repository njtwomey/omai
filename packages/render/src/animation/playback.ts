import type { Key } from 'whitebox-engine/random'
import type { Frame, Timeline } from 'whitebox-engine/scene'
import { type Algorithm, iterate } from 'whitebox-engine/trace'

/**
 * Frames to play, from either playback mode behind one interface: precomputed (a finished sequence or timeline) or
 * live (an algorithm stepped as frames are asked for). Players ask for a frame index; `ensure` computes up to it within
 * a budget, so a live run never blocks a frame for long.
 */
export interface PlaybackSource<F> {
  readonly kind: 'precomputed' | 'live'
  /** Frames available now. */
  available(): number
  /** Whether no more frames will come. */
  done(): boolean
  /** The frame at an index, if available. */
  frame(index: number): F | undefined
  /** Compute frames up to `index` (a live source steps at most its budget per call); returns `available()`. */
  ensure(index: number): number
}

/**
 * A precomputed source: a finished list of frames, or a timeline's frames.
 *
 * @param frames - The frames or a timeline.
 * @returns The source.
 * @example
 * ```ts
 * import { precomputedSource } from 'whitebox-render/animation'
 * const src = precomputedSource(['a', 'b'])
 * src.frame(1) // => 'b'
 * src.done() // => true
 * ```
 */
export function precomputedSource<F>(frames: readonly F[]): PlaybackSource<F>
export function precomputedSource(frames: Timeline): PlaybackSource<Frame>
export function precomputedSource<F>(frames: readonly F[] | Timeline): PlaybackSource<F | Frame> {
  const list: readonly (F | Frame)[] = Array.isArray(frames) ? frames : (frames as Timeline).frames
  return {
    kind: 'precomputed',
    available: () => list.length,
    done: () => true,
    frame: (i) => list[i],
    ensure: () => list.length,
  }
}

/** Options of {@link liveSource}. */
export interface LiveOptions {
  readonly maxSteps: number
  readonly seed?: number
  readonly key?: Key
  /** Most algorithm steps per `ensure` call (default 50): the frame budget. */
  readonly budget?: number
  /** Keep every `every`-th state as a frame (default 1). */
  readonly every?: number
}

/**
 * A live source: steps an algorithm (whitebox-engine's `iterate`) as frames are asked for, turning states into frames,
 * within a per-call step budget. Deterministic: the same seed gives the same frames however playback is paced.
 *
 * @param algorithm - The algorithm.
 * @param toFrame - A frame from a state and its step.
 * @param options - Step limit, seed or key, budget and thinning.
 * @returns The source.
 * @example
 * ```ts
 * import { liveSource } from 'whitebox-render/animation'
 * const count = { name: 'count', config: {}, init: () => 0, step: (s: number) => s + 1 }
 * const src = liveSource(count, (s) => s * 10, { maxSteps: 5, budget: 2 })
 * src.ensure(10) // => 2
 * src.ensure(10) // => 4
 * src.frame(3) // => 30
 * ```
 */
export function liveSource<S, F>(
  algorithm: Algorithm<S>,
  toFrame: (state: S, step: number) => F,
  options: LiveOptions,
): PlaybackSource<F> {
  const generator = iterate(algorithm, {
    maxSteps: options.maxSteps,
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const frames: F[] = []
  const budget = Math.max(1, options.budget ?? 50)
  const every = Math.max(1, options.every ?? 1)
  let finished = false
  return {
    kind: 'live',
    available: () => frames.length,
    done: () => finished,
    frame: (i) => frames[i],
    ensure: (index) => {
      let steps = 0
      while (!finished && frames.length <= index && steps < budget) {
        const next = generator.next()
        steps += 1
        if (next.done === true) {
          finished = true
          break
        }
        const { t, state, status } = next.value
        if (t % every === 0 || status !== undefined) frames.push(toFrame(state, t))
        if (status !== undefined) finished = true
      }
      return frames.length
    },
  }
}
