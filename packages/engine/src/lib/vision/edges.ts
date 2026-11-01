/**
 * Edges and lines in steps a session can scrub through: the Canny detector as its stages (smoothing, gradient,
 * non-maximum suppression, the double threshold, then hysteresis growing one ring of weak pixels per step) with every
 * stage's image as the state, and the Hough line transform as an accumulator that fills one edge pixel's votes at a
 * time, followed by peak extraction that returns the lines. Run to the end they equal the one-shot `canny` and
 * `houghLine` of `signal/image` exactly.
 */
import type { Json } from '../base/index.ts'
import { MdmlError } from '../base/index.ts'
import { type CannyOptions, cannyStages } from '../signal/image/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { greyOf } from './_image.ts'
import type { GreedyPath } from './greedy-contract.ts'

/** The stages of {@link cannyAlgorithm}, in order. */
export const CANNY_STAGES = ['input', 'smoothed', 'gradient', 'suppressed', 'thresholded', 'hysteresis'] as const

/** The state of {@link cannyAlgorithm}. */
export interface CannyState {
  /** The stage reached: one of {@link CANNY_STAGES}. */
  readonly stage: (typeof CANNY_STAGES)[number]
  /** The image of the stage: the input, the smoothed image, the gradient magnitude, the thinned magnitude, the 0/1/2 threshold classes, the edges so far. */
  readonly image: Tensor
  /** The hysteresis round (0 before it starts). */
  readonly round: number
  /** The edge pixels accepted so far (strong ones, then the weak ones joined to them). */
  readonly edgePixels: number
  /** The pixels the last hysteresis round added (the run is done when a round adds none). */
  readonly added: number
  readonly step: number
}

/** The paths of {@link cannyAlgorithm}. */
export type CannyPath = 'canny/stage' | 'canny/image' | 'canny/edgePixels'

/** What a Canny run logs. */
export const cannyContract: Contract<CannyState, CannyPath> = defineContract({
  'canny/stage': scalar({
    description: 'the index of the stage in input, smoothed, gradient, suppressed, thresholded, hysteresis',
  }),
  'canny/image': tensorSource({ role: 'state', shape: ['height', 'width'], description: 'the image of the stage' }),
  'canny/edgePixels': scalar({ description: 'the edge pixels accepted so far' }),
})

/**
 * The Canny detector as an algorithm: one step per stage (smoothing, gradient magnitude, non-maximum suppression, the
 * double threshold), then one step per hysteresis round, each joining the weak candidates 8-adjacent to the edges so
 * far, until nothing more joins. The last image is exactly `canny`'s edge map.
 *
 * @param x - Image `[H, W]`.
 * @param options - Smoothing, thresholds and border, as {@link CannyOptions}.
 * @returns The algorithm.
 * @concept edge-detection
 * @example
 * ```ts
 * import { cannyAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 12 }, () => Array.from({ length: 12 }, (_, j) => (j < 6 ? 0 : 1)))
 * runAlgorithm(cannyAlgorithm(x), { maxSteps: 50 }).state.stage // => 'hysteresis'
 * ```
 */
export function cannyAlgorithm(x: TensorLike, options: CannyOptions = {}): Algorithm<CannyState, CannyPath> {
  const input = greyOf(x, 'cannyAlgorithm')
  const { h, w } = input
  const stages = cannyStages(x, options)
  const thr = toFlat(stages.thresholded)
  const count = (m: ArrayLike<number>): number => Array.from(m).filter((v) => v > 0).length
  const tensorOf = (d: Float64Array): Tensor => fromStorage(d, [h, w], 'float64')
  const stageImage: Record<string, Tensor> = {
    input: tensorOf(Float64Array.from(input.data)),
    smoothed: stages.smoothed,
    gradient: stages.magnitude,
    suppressed: stages.suppressed,
    thresholded: stages.thresholded,
  }
  return {
    name: 'cannyStages',
    config: { sigma: options.sigma ?? 1, low: stages.low, high: stages.high } as Json,
    init: () => ({ stage: 'input', image: stageImage['input'] as Tensor, round: 0, edgePixels: 0, added: 0, step: 0 }),
    step: (s, ctx) => {
      let next: CannyState
      const at = CANNY_STAGES.indexOf(s.stage)
      if (at < 4) {
        const stage = CANNY_STAGES[at + 1] as CannyState['stage']
        next = { stage, image: stageImage[stage] as Tensor, round: 0, edgePixels: 0, added: 0, step: s.step + 1 }
      } else {
        const cur = toFlat(s.image)
        const grown = Float64Array.from(cur)
        if (s.round === 0) for (let i = 0; i < h * w; i++) grown[i] = thr[i] === 2 ? 1 : 0
        else
          for (let r = 0; r < h; r++)
            for (let c = 0; c < w; c++) {
              if (thr[r * w + c] !== 1 || cur[r * w + c] === 1) continue
              let near = false
              for (let dr = -1; dr <= 1 && !near; dr++)
                for (let dc = -1; dc <= 1 && !near; dc++) {
                  const rr = r + dr
                  const cc = c + dc
                  if (rr >= 0 && rr < h && cc >= 0 && cc < w && cur[rr * w + cc] === 1) near = true
                }
              if (near) grown[r * w + c] = 1
            }
        const edgePixels = count(grown)
        next = {
          stage: 'hysteresis',
          image: tensorOf(grown),
          round: s.round + 1,
          edgePixels,
          added: edgePixels - s.edgePixels,
          step: s.step + 1,
        }
      }
      ctx.log('canny/stage', CANNY_STAGES.indexOf(next.stage))
      ctx.log('canny/image', next.image)
      ctx.log('canny/edgePixels', next.edgePixels)
      return next
    },
    done: (s) => (s.stage === 'hysteresis' && s.round > 1 && s.added === 0 ? 'converged' : undefined),
    observe: (s) => ({ metrics: { edgePixels: s.edgePixels }, tensors: { image: { role: 'state', value: s.image } } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CannyState,
  }
}

/** The state of {@link houghLineAlgorithm}. */
export interface HoughState {
  /** The accumulator `[distances, angles]`, flat, row-major. */
  readonly accumulator: Float64Array
  /** The edge pixels that have voted. */
  readonly voted: number
  readonly step: number
}

/** The paths of {@link houghLineAlgorithm}. */
export type HoughPath = 'hough/accumulator' | 'hough/voted' | 'hough/maxVotes'

/** What a Hough run logs. */
export const houghContract: Contract<HoughState, HoughPath> = defineContract({
  'hough/accumulator': tensorSource({ role: 'state', shape: ['distances', 'angles'], description: 'the votes so far' }),
  'hough/voted': scalar({ description: 'the edge pixels that have voted' }),
  'hough/maxVotes': scalar({ description: 'the largest cell of the accumulator', direction: 'higher' }),
})

/** Options of {@link houghLineAlgorithm}. */
export interface HoughLineOptions {
  /** The angles in radians (default 180 from -pi/2 inclusive to pi/2 exclusive, as scikit-image). */
  readonly thetas?: readonly number[]
  /** Edge pixels voting per step (default 1). */
  readonly pixelsPerStep?: number
}

/** The geometry of a Hough accumulator. */
export interface HoughGeometry {
  readonly thetas: Float64Array
  readonly distances: Float64Array
  /** The accumulator's shape `[distances, angles]`. */
  readonly shape: readonly [number, number]
}

/**
 * The accumulator of the straight-line Hough transform filling one edge pixel's votes per step: each nonzero pixel
 * votes for every line `x cos(theta) + y sin(theta) = rho` through it (rho rounded half away from zero, as
 * scikit-image), in raster order. After the last pixel the accumulator equals `houghLine`'s.
 *
 * @param edges - Binary edge image `[H, W]`.
 * @param options - The angles and how many pixels vote per step.
 * @returns The algorithm and the geometry (angles, distances) of its accumulator.
 * @concept hough-transform
 * @example
 * ```ts
 * import { houghLineAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const edges = Array.from({ length: 5 }, () => [0, 0, 1, 0, 0])
 * const { algorithm } = houghLineAlgorithm(edges)
 * Math.max(...runAlgorithm(algorithm, { maxSteps: 20 }).state.accumulator) // => 5
 * ```
 */
export function houghLineAlgorithm(
  edges: TensorLike,
  options: HoughLineOptions = {},
): { readonly algorithm: Algorithm<HoughState, HoughPath>; readonly geometry: HoughGeometry } {
  const p = greyOf(edges, 'houghLineAlgorithm')
  const thetas =
    options.thetas === undefined
      ? Float64Array.from({ length: 180 }, (_, i) => -Math.PI / 2 + (Math.PI * i) / 180)
      : Float64Array.from(options.thetas)
  const nt = thetas.length
  const offset = Math.ceil(Math.sqrt(p.h * p.h + p.w * p.w))
  const nd = 2 * offset + 1
  const distances = Float64Array.from({ length: nd }, (_, i) => -offset + i)
  const cs = Float64Array.from(thetas, Math.cos)
  const sn = Float64Array.from(thetas, Math.sin)
  const pixels: number[] = []
  for (let i = 0; i < p.h * p.w; i++) if ((p.data[i] ?? 0) !== 0) pixels.push(i)
  const per = Math.max(1, options.pixelsPerStep ?? 1)
  const round = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v))
  const algorithm: Algorithm<HoughState, HoughPath> = {
    name: 'houghLine',
    config: { angles: nt, edgePixels: pixels.length, pixelsPerStep: per } as Json,
    init: () => ({ accumulator: new Float64Array(nd * nt), voted: 0, step: 0 }),
    step: (s, ctx) => {
      const acc = Float64Array.from(s.accumulator)
      const end = Math.min(pixels.length, s.voted + per)
      for (let k = s.voted; k < end; k++) {
        const i = pixels[k] as number
        const y = Math.floor(i / p.w)
        const x = i % p.w
        for (let j = 0; j < nt; j++) {
          const idx = round((cs[j] as number) * x + (sn[j] as number) * y) + offset
          acc[idx * nt + j] = (acc[idx * nt + j] ?? 0) + 1
        }
      }
      ctx.log('hough/voted', end)
      ctx.log('hough/maxVotes', Math.max(...acc))
      if (ctx.wants('hough/accumulator')) ctx.log('hough/accumulator', fromStorage(acc, [nd, nt], 'float64'))
      return { accumulator: acc, voted: end, step: s.step + 1 }
    },
    done: (s) => (s.voted >= pixels.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { voted: s.voted, maxVotes: Math.max(...s.accumulator) },
      tensors: { accumulator: { role: 'state', value: s.accumulator, shape: [nd, nt] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HoughState,
  }
  return { algorithm, geometry: { thetas, distances, shape: [nd, nt] } }
}

/** A line found in a Hough accumulator: `x cos(theta) + y sin(theta) = rho`. */
export interface HoughPeak {
  readonly rho: number
  readonly theta: number
  readonly votes: number
}

/** Options of {@link houghLinePeaks}. */
export interface HoughPeakOptions {
  /** The fewest votes of a peak (default half the largest cell). */
  readonly threshold?: number
  /** Cells either side of a peak (distance axis) that cannot be another peak (default 9). */
  readonly minDistance?: number
  /** Cells either side of a peak (angle axis) that cannot be another peak (default 10). */
  readonly minAngle?: number
  /** The most peaks to return (default all). */
  readonly numPeaks?: number
}

/** The state of {@link houghPeaksAlgorithm}: the accumulator being suppressed and the peaks taken. */
export interface HoughPeaksState {
  /** The accumulator `[distances, angles]`, flat; the cells around each peak are zeroed. */
  readonly accumulator: Float64Array
  readonly peaks: { rho: number; theta: number; votes: number }[]
  finished: boolean
}

/**
 * The lines of a Hough accumulator as a stepped algorithm, one peak per step: the largest cell at or above the threshold,
 * then a window around it (`minDistance` cells in distance, `minAngle` in angle) suppressed, so one line is one peak. It
 * logs the cell kept, its votes and the cells zeroed, and is done (`converged`) when no cell reaches the threshold,
 * `completed` at `numPeaks`. The state serialises.
 *
 * @param accumulator - The accumulator `[distances, angles]`.
 * @param geometry - Its angles and distances.
 * @param options - Threshold, suppression window and the number of peaks.
 * @returns The algorithm.
 * @throws MdmlError with code `shape` when the accumulator does not match the geometry.
 * @concept hough-transform
 * @example
 * ```ts
 * import { houghLineAlgorithm, houghPeaksAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { fromStorage } from 'mdml-engine/tensor'
 * const edges = Array.from({ length: 20 }, () => Array.from({ length: 20 }, (_, j) => (j === 7 ? 1 : 0)))
 * const { algorithm, geometry } = houghLineAlgorithm(edges)
 * const acc = fromStorage(runAlgorithm(algorithm, { maxSteps: 100 }).state.accumulator, [...geometry.shape], 'float64')
 * runAlgorithm(houghPeaksAlgorithm(acc, geometry), { maxSteps: 50 }).state.peaks[0]?.rho // => 7
 * ```
 */
export function houghPeaksAlgorithm(
  accumulator: Tensor,
  geometry: HoughGeometry,
  options: HoughPeakOptions = {},
): Algorithm<HoughPeaksState, GreedyPath> {
  const [nd, nt] = geometry.shape
  if (accumulator.shape[0] !== nd || accumulator.shape[1] !== nt)
    throw new MdmlError('shape', 'houghLinePeaks: the accumulator does not match the geometry')
  const acc0 = Float64Array.from(toFlat(accumulator))
  const maxVotes = acc0.reduce((a, b) => Math.max(a, b), 0)
  const threshold = options.threshold ?? 0.5 * maxVotes
  const dist = options.minDistance ?? 9
  const ang = options.minAngle ?? 10
  const limit = options.numPeaks ?? Number.POSITIVE_INFINITY
  return {
    name: 'vision/hough-peaks',
    config: { threshold, minDistance: dist, minAngle: ang, numPeaks: Number.isFinite(limit) ? limit : null },
    init: () => ({ accumulator: acc0, peaks: [], finished: false }),
    step: (s, ctx) => {
      const acc = s.accumulator
      let best = -1
      let arg = -1
      for (let i = 0; i < acc.length; i++)
        if ((acc[i] as number) > best) {
          best = acc[i] as number
          arg = i
        }
      if (arg < 0 || best < threshold || best <= 0) {
        s.finished = true
        return s
      }
      const di = Math.floor(arg / nt)
      const tj = arg % nt
      s.peaks.push({ rho: geometry.distances[di] as number, theta: geometry.thetas[tj] as number, votes: best })
      let zeroed = 0
      for (let a = Math.max(0, di - dist); a <= Math.min(nd - 1, di + dist); a++)
        for (let b = Math.max(0, tj - ang); b <= Math.min(nt - 1, tj + ang); b++) {
          if ((acc[a * nt + b] ?? 0) !== 0) zeroed++
          acc[a * nt + b] = 0
        }
      ctx.log('greedy/pick', arg)
      ctx.log('greedy/score', best)
      ctx.log('greedy/suppress', zeroed)
      return s
    },
    done: (s) => (s.finished ? 'converged' : s.peaks.length >= limit ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HoughPeaksState,
  }
}

/**
 * The lines of a Hough accumulator: repeatedly the largest cell at or above the threshold, then a window around it
 * (`minDistance` cells in distance, `minAngle` in angle) suppressed, so one line is one peak.
 * {@link houghPeaksAlgorithm} run to the end.
 *
 * @param accumulator - The accumulator `[distances, angles]`.
 * @param geometry - Its angles and distances.
 * @param options - Threshold, suppression window and the number of peaks.
 * @returns The peaks, strongest first.
 * @concept hough-transform
 * @example
 * ```ts
 * import { houghLineAlgorithm, houghLinePeaks } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { fromStorage } from 'mdml-engine/tensor'
 * const edges = Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => (j === 7 ? 1 : 0)))
 * const { algorithm, geometry } = houghLineAlgorithm(edges)
 * const acc = runAlgorithm(algorithm, { maxSteps: 100 }).state.accumulator
 * houghLinePeaks(fromStorage(acc, [...geometry.shape], 'float64'), geometry)[0]?.rho // => 7
 * ```
 */
export function houghLinePeaks(
  accumulator: Tensor,
  geometry: HoughGeometry,
  options: HoughPeakOptions = {},
): HoughPeak[] {
  return runAlgorithm(houghPeaksAlgorithm(accumulator, geometry, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state
    .peaks
}
