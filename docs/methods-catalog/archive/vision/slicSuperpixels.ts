/**
 * SLIC superpixels watched, as a method of nothing but its configuration: centres on a grid over a synthetic scene,
 * each step assigning every pixel to the nearest centre in colour and position and moving the centres to their
 * pixels' means, by the library's algorithm. The labels are on the tape with the largest centre move and the share of
 * pixels leaking across the scene's true edges, and the scene painted in its superpixels' mean grey with the
 * centres over it is a probe.
 */

import type { DecisionSurface } from '../../lib/exposition/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor,
} from '../../lib/trace/index.ts'
import { type SlicState, slicAlgorithm } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid, type Scene, scene } from './_scene.ts'

/** The share of pixels whose superpixel lies mostly in another region of the scene: what leaks across the true edges. */
function leak(sc: Scene, labels: ArrayLike<number>): number {
  const counts = new Map<number, Map<number, number>>()
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i] ?? -1
    if (l < 0) continue
    const r = sc.regions[i] ?? 0
    const m = counts.get(l) ?? new Map<number, number>()
    m.set(r, (m.get(r) ?? 0) + 1)
    counts.set(l, m)
  }
  const majority = new Map<number, number>()
  for (const [l, m] of counts) {
    let best = -1
    let most = -1
    for (const [r, n] of m)
      if (n > most) {
        best = r
        most = n
      }
    majority.set(l, best)
  }
  let wrong = 0
  let total = 0
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i] ?? -1
    if (l < 0) continue
    total++
    if (majority.get(l) !== (sc.regions[i] ?? 0)) wrong++
  }
  return total === 0 ? 0 : wrong / total
}

/** The scene painted in each superpixel's mean grey (pixels not yet assigned keep their own), with the centres. */
function paintedRecord(sc: Scene, s: SlicState): DecisionSurface {
  const { size } = sc
  const sums = new Map<number, [number, number]>()
  s.labels.forEach((l, i) => {
    if (l < 0) return
    const acc = sums.get(l) ?? [0, 0]
    acc[0] += sc.noisy[i] ?? 0
    acc[1] += 1
    sums.set(l, acc)
  })
  const painted = Array.from(s.labels, (l, i) => {
    const acc = sums.get(l)
    return acc === undefined ? (sc.noisy[i] ?? 0) : acc[0] / acc[1]
  })
  // A centre is [grey, row, column]: the scene is grey, so its features are one value.
  const centres = s.centres.map((c) => [c[2] ?? 0, c[1] ?? 0] as const)
  return picture(painted, size, size, {
    name: 'mean grey',
    points: centres,
    classes: ['centre'],
    labels: centres.map(() => 0),
  })
}

/**
 * The paths of {@link slicSuperpixels}: the superpixel of every pixel, the largest centre move and the leak across the true boundaries, and the derived `segments/picture`.
 *
 * @example
 * ```ts
 * import type { SlicSuperpixelsPath } from 'mdml-engine/methods/vision'
 * const path: SlicSuperpixelsPath = 'segments/moved'
 * path // => 'segments/moved'
 * ```
 */
export type SlicSuperpixelsPath = 'segments/labels' | 'segments/moved' | 'segments/leak' | 'segments/picture'

/**
 * SLIC superpixels of a synthetic scene, one assignment and update per step.
 *
 * Configuration: `segments` (about how many superpixels), `compactness` (the weight of position against grey: high
 * keeps the superpixels square, low lets them follow the edges), `noise` (the scene's pixel noise) and `size` (the
 * image side). Paths: `segments/labels` (the superpixel of every pixel, full), `segments/moved` (the largest centre
 * move of the step), `segments/leak` (the share of pixels whose superpixel lies mostly in another region: what
 * leaks across the true edges) and the probe `segments/picture` (the scene painted in its superpixels' mean grey, with the
 * centres).
 */
export const slicSuperpixels: Method<SlicState, Config, SlicSuperpixelsPath> = defineMethod<
  SlicState,
  Config,
  SlicSuperpixelsPath
>({
  name: 'vision/slic',
  config: dict({
    segments: int(4, 144, { default: 36, label: 'superpixels' }),
    compactness: real(1, 40, { default: 10, log: true }),
    noise: real(0, 0.3, { default: 0.05, label: 'pixel noise σ' }),
    size: int(16, 96, { default: 64, label: 'image side' }),
  }),
  make: (c, key) => {
    const size = c['size'] as number
    const sc = scene(key, { size, noise: c['noise'] as number })
    const inner: Algorithm<SlicState> = slicAlgorithm(grid(sc.noisy, size), {
      nSegments: c['segments'] as number,
      compactness: c['compactness'] as number,
    })
    return {
      name: 'slic',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log('segments/labels', grid(next.labels, size))
        ctx.log('segments/moved', next.moved)
        ctx.log('segments/leak', leak(sc, next.labels))
        return next
      },
      ...(inner.done === undefined ? {} : { done: inner.done }),
      serialise: (s) => ({ centres: s.centres, labels: Array.from(s.labels), moved: s.moved }),
      deserialise: (j) => {
        const o = j as { centres: number[][]; labels: number[]; moved: number }
        return { centres: o.centres, labels: Int32Array.from(o.labels), moved: o.moved }
      },
    }
  },
  contract: (c, key): Contract<SlicState, SlicSuperpixelsPath> => {
    const size = c['size'] as number
    const sc = scene(key, { size, noise: c['noise'] as number })
    return {
      'segments/labels': tensor({ role: 'state', shape: [size, size], description: 'the superpixel of every pixel' }),
      'segments/moved': scalar({ description: 'the largest centre move of the step, in pixels' }),
      'segments/leak': scalar({
        description: 'the share of pixels whose superpixel lies mostly in another region of the scene',
      }),
      'segments/picture': derived<SlicState>({
        description: 'the scene painted in its superpixels’ mean grey, with the centres over it',
        compute: (s): DecisionSurface => paintedRecord(sc, s),
      }),
    }
  },
  defaults: {
    'segments/labels': Log({ detail: 'full' }),
    'segments/moved': Log(),
    'segments/leak': Log(),
  },
})
