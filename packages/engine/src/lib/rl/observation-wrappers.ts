/**
 * Environment wrappers that change what the agent observes, as in gymnasium: running normalisation, frame stacking,
 * flattening structured observations, one-hot encoding of discrete states, appending the time step, and pixel
 * observations rendered from the environment's scenes by the software rasteriser.
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { rasterise } from '../scene/index.ts'
import { box, flatDim, flatten, type Space, type SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { type Env, type Reset, resetAfter } from './env.ts'
import { type RunningMeanStd, runningMeanStd, transformObservation, updateRunningMeanStd } from './wrappers.ts'

/** The state of {@link normalizeObservation}. */
export interface NormalizeObservationState<S> {
  readonly inner: S
  readonly stats: RunningMeanStd
}

/**
 * Normalise observations to zero mean and unit variance with running statistics (gymnasium's NormalizeObservation).
 * The statistics are updated with every observation, including those returned by resets, and carried across
 * episodes by `resetFrom`; `update: false` freezes them (for evaluation).
 *
 * @param env - An environment with vector observations.
 * @param options - The stability constant (default 1e-8), whether to update, and starting statistics.
 * @returns The normalising environment.
 * @concept environments
 * @example
 * ```ts
 * import { normalizeObservation } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * normalizeObservation(cartPole.create()).reset(key(0)).state.stats.count > 1 // => true
 * ```
 */
export function normalizeObservation<S, A>(
  env: Env<S, readonly number[], A>,
  options: { readonly epsilon?: number; readonly update?: boolean; readonly stats?: RunningMeanStd } = {},
): Env<NormalizeObservationState<S>, readonly number[], A> {
  const eps = options.epsilon ?? 1e-8
  const update = options.update ?? true
  if (!(eps > 0)) throw new MdmlError('value', 'normalizeObservation: epsilon must be positive')
  const size = flatDim(env.observationSpace)
  const norm = (o: readonly number[], st: RunningMeanStd): number[] =>
    o.map((v, i) => (v - (st.mean[i] ?? 0)) / Math.sqrt((st.var[i] ?? 1) + eps))
  const observe = (r: { readonly observation: readonly number[] }, st: RunningMeanStd): RunningMeanStd =>
    update ? updateRunningMeanStd(st, [r.observation]) : st
  const start = (
    r: Reset<S, readonly number[]>,
    st: RunningMeanStd,
  ): Reset<NormalizeObservationState<S>, readonly number[]> => {
    const stats = observe(r, st)
    return { ...r, observation: norm(r.observation, stats), state: { inner: r.state, stats } }
  }
  const inner = env.observe
  return {
    spec: env.spec,
    observationSpace: box(-Number.MAX_VALUE, Number.MAX_VALUE, [size]),
    actionSpace: env.actionSpace,
    reset: (k) => start(env.reset(k), options.stats ?? runningMeanStd(size)),
    resetFrom: (p, k) => start(resetAfter(env, p.inner, k), p.stats),
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const stats = observe(r, s.stats)
      return { ...r, observation: norm(r.observation, stats), state: { inner: r.state, stats } }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: NormalizeObservationState<S>) => norm(inner(s.inner), s.stats) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** The state of {@link frameStack}: the inner state and the last `n` observations (oldest first). */
export interface FrameStackState<S, O> {
  readonly inner: S
  readonly frames: readonly O[]
}

/**
 * Stack the last `n` observations (gymnasium's FrameStackObservation). After a reset the stack is padded with copies of
 * the first observation (`padding: 'reset'`, the default) or with a given padding observation.
 *
 * @param env - The environment.
 * @param n - The stack size.
 * @param options - The padding.
 * @returns The stacking environment; observations are arrays of `n` inner observations, oldest first.
 * @concept environments
 * @example
 * ```ts
 * import { frameStack } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * frameStack(cartPole.create(), 4).reset(key(0)).observation.length // => 4
 * ```
 */
export function frameStack<S, O, A>(
  env: Env<S, O, A>,
  n: number,
  options: { readonly padding?: 'reset' | { readonly value: O } } = {},
): Env<FrameStackState<S, O>, readonly O[], A> {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'frameStack: n must be a positive integer')
  const padding = options.padding ?? 'reset'
  const start = (r: Reset<S, O>): Reset<FrameStackState<S, O>, readonly O[]> => {
    const pad = padding === 'reset' ? r.observation : padding.value
    const frames = [...new Array<O>(n - 1).fill(pad), r.observation]
    return { ...r, observation: frames, state: { inner: r.state, frames } }
  }
  const resetFrom = env.resetFrom
  const space = env.observationSpace
  return {
    spec: env.spec,
    observationSpace: { kind: 'tuple', spaces: new Array<Space>(n).fill(space) },
    actionSpace: env.actionSpace,
    reset: (k) => start(env.reset(k)),
    ...(resetFrom === undefined
      ? {}
      : { resetFrom: (p: FrameStackState<S, O>, k: Key) => start(resetFrom(p.inner, k)) }),
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const frames = [...s.frames.slice(1), r.observation]
      return { ...r, observation: frames, state: { inner: r.state, frames } }
    },
    render: (s) => env.render(s.inner),
    observe: (s) => s.frames,
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/**
 * Flatten structured observations (tuples, dicts, boxes of any shape, discrete values) into number vectors with the
 * space's `flatten` (gymnasium's FlattenObservation).
 *
 * @param env - The environment.
 * @returns The flattening environment.
 * @concept environments
 * @example
 * ```ts
 * import { flattenObservation } from 'mdml-engine/rl'
 * import { blackjack } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * flattenObservation(blackjack.create()).reset(key(0)).observation.length // => 3
 * ```
 */
export function flattenObservation<S, O, A>(env: Env<S, O, A>): Env<S, readonly number[], A> {
  const space = env.observationSpace
  return transformObservation(
    env,
    (o: O) => flatten(space, o as unknown as SpaceValue),
    box(-Number.MAX_VALUE, Number.MAX_VALUE, [flatDim(space)]),
  )
}

/**
 * One-hot encode discrete observations (for neural agents on tabular environments such as FrozenLake).
 *
 * @param env - An environment with a discrete observation space.
 * @returns The encoding environment.
 * @throws MdmlError with code `value` when the observation space is not discrete.
 * @concept environments
 * @example
 * ```ts
 * import { oneHotObservation } from 'mdml-engine/rl'
 * import { frozenLake } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * oneHotObservation(frozenLake.create()).reset(key(0)).observation.slice(0, 3) // => [1, 0, 0]
 * ```
 */
export function oneHotObservation<S, A>(env: Env<S, number, A>): Env<S, readonly number[], A> {
  const space = env.observationSpace
  if (space.kind !== 'discrete') throw new MdmlError('value', 'oneHotObservation: needs a discrete observation space')
  const start = space.start ?? 0
  return transformObservation(
    env,
    (o: number) => {
      const v = new Array<number>(space.n).fill(0)
      v[o - start] = 1
      return v
    },
    box(0, 1, [space.n]),
  )
}

/** The state of {@link timeAwareObservation}. */
export interface TimeAwareState<S> {
  readonly inner: S
  readonly t: number
}

/**
 * Append the time step to vector observations (gymnasium's TimeAwareObservation), optionally divided by a horizon so it
 * lies in [0, 1]; this makes time-limited tasks Markov.
 *
 * @param env - An environment with vector observations.
 * @param options - `horizon`: divide the step count by it (default: the raw count).
 * @returns The time-aware environment.
 * @concept environments
 * @example
 * ```ts
 * import { timeAwareObservation } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = timeAwareObservation(cartPole.create())
 * env.step(env.reset(key(0)).state, 1, key(1)).observation[4] // => 1
 * ```
 */
export function timeAwareObservation<S, A>(
  env: Env<S, readonly number[], A>,
  options: { readonly horizon?: number } = {},
): Env<TimeAwareState<S>, readonly number[], A> {
  const scale = options.horizon ?? 1
  const withT = (o: readonly number[], t: number): number[] => [...o, t / scale]
  const resetFrom = env.resetFrom
  const start = (r: Reset<S, readonly number[]>): Reset<TimeAwareState<S>, readonly number[]> => ({
    ...r,
    observation: withT(r.observation, 0),
    state: { inner: r.state, t: 0 },
  })
  const inner = env.observe
  return {
    spec: env.spec,
    observationSpace: box(-Number.MAX_VALUE, Number.MAX_VALUE, [flatDim(env.observationSpace) + 1]),
    actionSpace: env.actionSpace,
    reset: (k) => start(env.reset(k)),
    ...(resetFrom === undefined ? {} : { resetFrom: (p: TimeAwareState<S>, k: Key) => start(resetFrom(p.inner, k)) }),
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const t = s.t + 1
      return { ...r, observation: withT(r.observation, t), state: { inner: r.state, t } }
    },
    render: (s) => env.render(s.inner),
    ...(inner === undefined ? {} : { observe: (s: TimeAwareState<S>) => withT(inner(s.inner), s.t) }),
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}

/** Options of {@link pixelObservation}. */
export interface PixelObservationOptions {
  /** Image width (default 84). */
  readonly width?: number
  /** Image height (default 84). */
  readonly height?: number
  /** Convert to one grey channel (gymnasium's GrayscaleObservation; luminance weights 0.299, 0.587, 0.114). */
  readonly grayscale?: boolean
  /** Samples per pixel along each axis (default 1). */
  readonly supersample?: number
}

/**
 * Pixel observations rendered from the environment's scenes by the software rasteriser (gymnasium's
 * AddRenderObservation with `render_only`, plus resizing and optional greyscale): an int32 tensor of shape
 * `[height, width, 3]` (or `[height, width, 1]` in greyscale) with values 0 to 255.
 *
 * @param env - The environment.
 * @param options - Size, greyscale and supersampling.
 * @returns The environment with pixel observations.
 * @concept environments
 * @example
 * ```ts
 * import { pixelObservation } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * pixelObservation(cartPole.create(), { width: 32, height: 24 }).reset(key(0)).observation.shape // => [24, 32, 3]
 * ```
 */
export function pixelObservation<S, O, A>(env: Env<S, O, A>, options: PixelObservationOptions = {}): Env<S, Tensor, A> {
  const width = options.width ?? 84
  const height = options.height ?? 84
  const grey = options.grayscale ?? false
  const channels = grey ? 1 : 3
  const pixels = (s: S): Tensor => {
    const img = rasterise(env.render(s), { width, height, samples: options.supersample ?? 1 })
    const out = new Int32Array(width * height * channels)
    for (let i = 0; i < width * height; i++) {
      const r = img.rgba[4 * i] ?? 0
      const g = img.rgba[4 * i + 1] ?? 0
      const b = img.rgba[4 * i + 2] ?? 0
      if (grey) out[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
      else {
        out[3 * i] = r
        out[3 * i + 1] = g
        out[3 * i + 2] = b
      }
    }
    return fromStorage(out, [height, width, channels], 'int32')
  }
  const resetFrom = env.resetFrom
  const lift = (r: Reset<S, O>): Reset<S, Tensor> => ({ ...r, observation: pixels(r.state) })
  return {
    spec: env.spec,
    observationSpace: box(0, 255, [height, width, channels]),
    actionSpace: env.actionSpace,
    reset: (k) => lift(env.reset(k)),
    ...(resetFrom === undefined ? {} : { resetFrom: (p: S, k: Key) => lift(resetFrom(p, k)) }),
    step: (s, action, k) => {
      const r = env.step(s, action, k)
      return { ...r, observation: pixels(r.state) }
    },
    render: (s) => env.render(s),
    observe: pixels,
    wrapped: env as Env<unknown, unknown, unknown>,
  }
}
