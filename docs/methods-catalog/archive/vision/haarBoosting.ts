/**
 * Viola-Jones boosting watched, as a method of nothing but its configuration: small windows of circles (positives)
 * and of squares and triangles (negatives), and the library's discrete AdaBoost adding one decision stump over a
 * Haar-like feature per step. The errors and the stump's vote are on the tape with every window's weight, and the
 * stump just chosen and the strong classifier's accumulated template over the window are probes.
 */

import { generate } from '../../lib/data/index.ts'
import { shapes } from '../../lib/datasets/index.ts'
import { type CurveSet, curveSet, type DecisionSurface } from '../../lib/exposition/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { toFlat, type Value } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
} from '../../lib/trace/index.ts'
import { type AdaBoostState, adaBoostAlgorithm, type HaarFeature, haarFeatures } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid } from './_scene.ts'

const SIDE = 16

/** The windows: positives first, then negatives, each `SIDE` by `SIDE`. */
interface Windows {
  readonly positives: number[][][]
  readonly negatives: number[][][]
  readonly pool: HaarFeature[]
}

function windowsOf(key: Key, c: Config): Windows {
  const n = c['windows'] as number
  const g = generate(shapes, { knobs: { n, size: SIDE, perImage: 1, noise: c['noise'] as number }, key })
  const pixels = toFlat(g.data.images)
  const labels = toFlat(g.data.labels as Value)
  const positives: number[][][] = []
  const negatives: number[][][] = []
  for (let i = 0; i < n; i++) {
    const win = grid(pixels.slice(i * SIDE * SIDE, (i + 1) * SIDE * SIDE), SIDE)
    // Class 0 is the circle.
    ;(labels[i] === 0 ? positives : negatives).push(win)
  }
  const stride = c['stride'] as number
  const pool = haarFeatures(SIDE, SIDE).filter((f) => f.rects.every((r) => r.every((v) => v % stride === 0)))
  return { positives, negatives, pool }
}

/** A feature's pattern over the window: +1 in the rectangles it adds, -1 in the ones it subtracts. */
function pattern(f: HaarFeature): number[] {
  const out = new Array<number>(SIDE * SIDE).fill(0)
  f.rects.forEach(([r, c, h, w], k) => {
    const sign = k % 2 === 0 ? -1 : 1
    for (let y = r; y < r + h; y++) for (let x = c; x < c + w; x++) out[y * SIDE + x] = sign
  })
  return out
}

/** The strong classifier's template: the stumps' patterns summed with their votes, signed by their polarity. */
function template(s: AdaBoostState): number[] {
  const out = new Array<number>(SIDE * SIDE).fill(0)
  for (const h of s.weak) {
    const p = pattern(h.feature)
    // Polarity 1 votes positive when the value is small: a bright region where the feature subtracts.
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) - h.polarity * h.alpha * (p[i] ?? 0)
  }
  return out
}

/** Every window's weight, positives and negatives apart. */
function weightsRecord(w: Windows, s: AdaBoostState): CurveSet {
  const m = w.positives.length
  const total = s.weights.reduce((a, b) => a + b, 0)
  const ws = Array.from(s.weights, (v) => v / total)
  return curveSet({
    series: [
      { name: 'circles (positive)', x: ws.slice(0, m).map((_, i) => i), y: ws.slice(0, m) },
      { name: 'squares and triangles (negative)', x: ws.slice(m).map((_, i) => m + i), y: ws.slice(m) },
    ],
    xLabel: 'window',
    yLabel: 'weight',
  })
}

/**
 * The paths of {@link haarBoosting}: the chosen stump’s error and vote, the strong classifier’s training error and size, and the derived row weights, stump and template.
 *
 * @example
 * ```ts
 * import type { HaarBoostingPath } from 'mdml-engine/methods/vision'
 * const path: HaarBoostingPath = 'boost/alpha'
 * path // => 'boost/alpha'
 * ```
 */
export type HaarBoostingPath =
  | 'boost/error'
  | 'boost/trainingError'
  | 'boost/alpha'
  | 'boost/stumps'
  | 'boost/weights'
  | 'boost/stump'
  | 'boost/template'

/**
 * Discrete AdaBoost over Haar-feature stumps, one stump per step: circles against squares and triangles.
 *
 * Configuration: `windows` (how many 16-pixel windows, about a third of them circles), `stride` (the grid the
 * features' rectangles sit on: 1 is the whole pool, 4 a coarse one), and `noise` (the windows' pixel noise). Paths:
 * `boost/error` (the chosen stump's weighted error), `boost/trainingError` (the strong classifier's error on the
 * windows), `boost/alpha` (the stump's vote), the probe `boost/weights` (every window's weight: the hard ones grow),
 * the probe `boost/stump` (the feature just chosen: bright where it adds, dark where it subtracts) and the probe
 * `boost/template` (the stumps' patterns summed with their votes: what the classifier has come to look for).
 */
export const haarBoosting: Method<AdaBoostState, Config, HaarBoostingPath> = defineMethod<
  AdaBoostState,
  Config,
  HaarBoostingPath
>({
  name: 'vision/haar-boosting',
  config: dict({
    windows: int(12, 240, { default: 60 }),
    stride: int(1, 4, { default: 2, label: 'feature grid stride' }),
    noise: real(0, 0.3, { default: 0.05, label: 'pixel noise σ' }),
  }),
  make: (c, key) => {
    const w = windowsOf(key, c)
    const inner: Algorithm<AdaBoostState> = adaBoostAlgorithm(w.positives, w.negatives, { features: w.pool })
    return {
      name: 'adaboost',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log('boost/error', next.error)
        ctx.log('boost/trainingError', next.trainingError)
        ctx.log('boost/alpha', next.weak.at(-1)?.alpha ?? 0)
        ctx.log('boost/stumps', next.weak.length)
        return next
      },
      ...(inner.done === undefined ? {} : { done: inner.done }),
      serialise: (s) => ({
        weights: Array.from(s.weights),
        weak: s.weak.map((h) => ({
          ...h,
          feature: { type: h.feature.type, rects: h.feature.rects.map((r) => [...r]) },
        })),
        error: s.error,
        trainingError: s.trainingError,
      }),
      deserialise: (j) => {
        const o = j as unknown as {
          weights: number[]
          weak: AdaBoostState['weak']
          error: number
          trainingError: number
        }
        return { weights: Float64Array.from(o.weights), weak: o.weak, error: o.error, trainingError: o.trainingError }
      },
    }
  },
  contract: (c, key): Contract<AdaBoostState, HaarBoostingPath> => {
    const w = windowsOf(key, c)
    return {
      'boost/error': scalar({ description: 'the weighted error of the stump just chosen' }),
      'boost/trainingError': scalar({ description: 'the strong classifier’s error on the windows' }),
      'boost/alpha': scalar({ description: 'the vote of the stump just chosen' }),
      'boost/stumps': scalar({ description: 'stumps so far' }),
      'boost/weights': derived<AdaBoostState>({
        description: 'every window’s weight: the ones the stumps get wrong grow',
        compute: (s): CurveSet => weightsRecord(w, s),
      }),
      'boost/stump': derived<AdaBoostState>({
        description: 'the feature just chosen: bright where it adds, dark where it subtracts',
        compute: (s): DecisionSurface => {
          const last = s.weak.at(-1)
          return picture(
            last === undefined ? new Array<number>(SIDE * SIDE).fill(0) : pattern(last.feature),
            SIDE,
            SIDE,
            {
              name: 'feature',
            },
          )
        },
      }),
      'boost/template': derived<AdaBoostState>({
        description: 'the stumps’ patterns summed with their votes: what the classifier looks for',
        compute: (s): DecisionSurface => picture(template(s), SIDE, SIDE, { name: 'vote' }),
      }),
    }
  },
  defaults: { 'boost/error': Log(), 'boost/trainingError': Log(), 'boost/alpha': Log(), 'boost/stumps': Log() },
})
