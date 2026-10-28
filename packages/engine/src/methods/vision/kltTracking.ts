/**
 * Kanade-Lucas-Tomasi point tracking watched, as a method of nothing but its configuration: a synthetic scene drifts
 * across the frame along a smooth loop with fresh sensor noise in every frame, and its visible corners are followed
 * frame to frame by the library's pyramidal Lucas-Kanade tracker. The points are on the tape with how many are still
 * tracked and their mean distance from where the motion really put them, and the frame with the points over it is a
 * probe.
 */

import type { DecisionSurface } from '../../lib/exposition/index.ts'
import { foldIn, type Key, normal } from '../../lib/random/index.ts'
import { image } from '../../lib/signal/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor,
} from '../../lib/trace/index.ts'
import { type KltState, kltAlgorithm } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid, scene } from './_scene.ts'

/** The frames of a drifting scene, with the drift of each and the corners to follow. */
interface Sequence {
  readonly size: number
  readonly frames: number[][][]
  /** The scene's offset (x, y) in each frame. */
  readonly offsets: readonly (readonly [number, number])[]
  readonly corners: readonly (readonly [number, number])[]
}

/** Where the scene sits in frame t: a figure of eight of the given amplitude, closed over the whole sequence. */
const drift = (t: number, count: number, amplitude: number): readonly [number, number] => {
  const phase = (2 * Math.PI * t) / count
  return [amplitude * Math.sin(phase), 0.5 * amplitude * Math.sin(2 * phase)]
}

function sequenceOf(key: Key, c: Config): Sequence {
  const size = c['size'] as number
  const count = c['frames'] as number
  const amplitude = c['amplitude'] as number
  const noise = c['noise'] as number
  const sc = scene(foldIn(key, 0), { size, noise: 0 })
  const clean = grid(sc.clean, size)
  const offsets = Array.from({ length: count }, (_, t) => drift(t, count, amplitude))
  const frames = offsets.map(([ox, oy], t) => {
    const moved = toFlat(image.shift(clean, [oy, ox], { order: 1, mode: 'nearest' }))
    const z = toFlat(normal(foldIn(key, t + 1), [size * size]))
    return grid(
      moved.map((v, i) => Math.min(1, Math.max(0, v + noise * (z[i] ?? 0)))),
      size,
    )
  })
  // Corners too near the border leave the frame as the scene drifts; they are followed until they do.
  return { size, frames, offsets, corners: sc.corners }
}

/** The mean distance of the tracked points from where the drift put their corners. */
function trackingError(seq: Sequence, s: KltState): number {
  const [ox, oy] = seq.offsets[s.frame] ?? [0, 0]
  let sum = 0
  let n = 0
  s.points.forEach(([px, py], i) => {
    if ((s.status[i] ?? 0) !== 1) return
    const [cx, cy] = seq.corners[i] ?? [0, 0]
    sum += Math.hypot(px - (cx + ox), py - (cy + oy))
    n++
  })
  return n === 0 ? Number.NaN : sum / n
}

/**
 * Corners of a drifting scene tracked frame to frame by pyramidal Lucas-Kanade.
 *
 * Configuration: `frames` (the length of the sequence; the run ends with it), `amplitude` (how far the scene drifts,
 * in pixels), `noise` (the sensor noise of every frame), `window` (the side of the window matched around each point)
 * and `size` (the image side). Paths: `track/points` (every point as (x, y), full), `track/alive` (points still
 * tracked), `track/error` (their mean distance from where the drift really put the corners, in pixels) and the probe
 * `track/picture` (the frame with the points over it, tracked or lost).
 */
export const kltTracking: Method<KltState, Config> = defineMethod<KltState, Config>({
  name: 'vision/klt-tracking',
  config: dict({
    frames: int(8, 120, { default: 48 }),
    amplitude: real(1, 12, { default: 6, label: 'drift amplitude (px)' }),
    noise: real(0, 0.2, { default: 0.03, label: 'sensor noise σ' }),
    window: int(5, 31, { default: 15, label: 'window side' }),
    size: int(32, 96, { default: 64, label: 'image side' }),
  }),
  make: (c, key) => {
    const seq = sequenceOf(key, c)
    const inner: Algorithm<KltState> = kltAlgorithm(seq.frames, seq.corners, { winSize: c['window'] as number })
    return {
      name: 'klt',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log(
          'track/points',
          next.points.map((p) => [...p]),
        )
        ctx.log(
          'track/alive',
          next.status.reduce((a, b) => a + b, 0),
        )
        ctx.log('track/error', trackingError(seq, next))
        return next
      },
      ...(inner.done === undefined ? {} : { done: inner.done }),
      serialise: (s) => ({ frame: s.frame, points: s.points.map((p) => [...p]), status: [...s.status] }),
      deserialise: (j) => j as unknown as KltState,
    }
  },
  contract: (c, key) => {
    const seq = sequenceOf(key, c)
    return {
      'track/points': tensor({
        role: 'state',
        shape: [seq.corners.length, 2],
        description: 'every point as (x, y); a lost point stays where it was lost',
      }),
      'track/alive': scalar({ description: 'points still tracked' }),
      'track/error': scalar({ description: 'mean distance of the tracked points from the true corners, in pixels' }),
      'track/picture': derived<KltState>({
        description: 'the frame with the points over it',
        compute: (s): DecisionSurface =>
          picture(seq.frames[s.frame]?.flat() ?? [], seq.size, seq.size, {
            points: s.points,
            classes: ['lost', 'tracked'],
            labels: [...s.status],
          }),
      }),
    }
  },
  defaults: { 'track/points': Log({ detail: 'full' }), 'track/alive': Log(), 'track/error': Log() },
})
