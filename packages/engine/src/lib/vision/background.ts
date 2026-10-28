/**
 * Background subtraction over video frames: the running average (cv2.accumulateWeighted) with a difference threshold,
 * and Stauffer and Grimson's adaptive mixture of Gaussians per pixel (the model behind OpenCV's MOG), both as traced
 * algorithms whose steps are frames, with the foreground masks they produce.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { greyOf, rasterOf, rasterTensor } from './_image.ts'

/**
 * The running average of images (cv2.accumulateWeighted): (1 - alpha) average + alpha image.
 *
 * @param x - The new image.
 * @param average - The running average (the same shape).
 * @param alpha - The weight of the new image.
 * @returns The updated average.
 * @concept background-subtraction
 * @example
 * ```ts
 * import { accumulateWeighted } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(accumulateWeighted([[1, 0]], [[0, 0]], 0.25)) // => [[0.25, 0]]
 * ```
 */
export function accumulateWeighted(x: TensorLike, average: TensorLike, alpha: number): Tensor {
  const a = rasterOf(x, 'accumulateWeighted')
  const b = rasterOf(average, 'accumulateWeighted')
  if (a.data.length !== b.data.length) throw new MdmlError('shape', 'accumulateWeighted: images differ in shape')
  return rasterTensor({ ...b, data: Float64Array.from(b.data, (v, i) => (1 - alpha) * v + alpha * (a.data[i] ?? 0)) })
}

/** The state of a background subtractor: the frames seen, the model's planes and the last foreground mask. */
export interface BackgroundState {
  readonly frame: number
  /** The model: the running mean (running average), or K weights, means and variances per pixel (mixture). */
  readonly model: Float64Array
  readonly foreground: Float64Array
}

/** Options of {@link runningAverageAlgorithm}. */
export interface RunningAverageOptions {
  /** The adaptation rate (default 0.05). */
  readonly alpha?: number
  /** The absolute difference above which a pixel is foreground (default 0.1). */
  readonly threshold?: number
}

const fraction = (m: Float64Array): number => m.reduce((a, b) => a + b, 0) / Math.max(1, m.length)

/**
 * Background subtraction by a running average, as an algorithm whose steps are frames: a pixel is foreground when it
 * differs from the background by more than the threshold, and the background follows each frame at rate alpha.
 *
 * @param frames - Grey frames `[H, W]`.
 * @param options - Rate and threshold.
 * @returns The algorithm; its state's `foreground` is the last frame's mask.
 * @concept background-subtraction
 * @example
 * ```ts
 * import { runningAverageAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const still = [[0.2, 0.2], [0.2, 0.2]]
 * const moved = [[0.2, 0.9], [0.2, 0.2]]
 * Array.from(runAlgorithm(runningAverageAlgorithm([still, still, moved]), { maxSteps: 3 }).state.foreground) // => [0, 1, 0, 0]
 * ```
 */
export function runningAverageAlgorithm(
  frames: readonly TensorLike[],
  options: RunningAverageOptions = {},
): Algorithm<BackgroundState> {
  const planes = frames.map((f) => greyOf(f, 'runningAverageAlgorithm').data)
  const alpha = options.alpha ?? 0.05
  const thr = options.threshold ?? 0.1
  return {
    name: 'running-average-background',
    config: { alpha, threshold: thr } as Json,
    schema: channels({ metrics: { foreground: { description: 'fraction of pixels in the foreground' } } }),
    init: () => ({ frame: 0, model: new Float64Array(0), foreground: new Float64Array(planes[0]?.length ?? 0) }),
    step: (s) => {
      const x = planes[s.frame] as Float64Array
      if (s.frame === 0) return { frame: 1, model: Float64Array.from(x), foreground: new Float64Array(x.length) }
      const foreground = Float64Array.from(x, (v, i) => (Math.abs(v - (s.model[i] ?? 0)) > thr ? 1 : 0))
      const model = Float64Array.from(s.model, (m, i) => (1 - alpha) * m + alpha * (x[i] ?? 0))
      return { frame: s.frame + 1, model, foreground }
    },
    done: (s) => (s.frame >= planes.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { foreground: fraction(s.foreground) } }),
  }
}

/** Options of {@link mixtureBackgroundAlgorithm}. */
export interface MixtureBackgroundOptions {
  /** Gaussians per pixel (default 3). */
  readonly components?: number
  /** The learning rate (default 0.01). */
  readonly alpha?: number
  /** The fraction of weight explained by the background components (default 0.7). */
  readonly backgroundRatio?: number
  /** A match is within this many standard deviations (default 2.5). */
  readonly matchSigmas?: number
  /** The standard deviation of a new component (default 0.1) and its weight (default 0.05). */
  readonly initialSigma?: number
  readonly initialWeight?: number
}

/**
 * Background subtraction by an adaptive mixture of Gaussians per pixel (Stauffer and Grimson 1999), as an algorithm
 * whose steps are frames: each pixel's components are matched within matchSigmas standard deviations, the matched
 * one's weight, mean and variance move towards the pixel at the learning rate, an unmatched pixel replaces the least
 * probable component, and the pixel is background when its match is among the highest-ranked (weight over standard
 * deviation) components whose weights reach the background ratio.
 *
 * @param frames - Grey frames `[H, W]`.
 * @param options - Components, rate, background ratio and initial spread.
 * @returns The algorithm; its state's `foreground` is the last frame's mask.
 * @concept background-subtraction
 * @example
 * ```ts
 * import { mixtureBackgroundAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const still = [[0.2, 0.2], [0.2, 0.2]]
 * const frames = [...new Array(10).fill(still), [[0.2, 0.9], [0.2, 0.2]]]
 * Array.from(runAlgorithm(mixtureBackgroundAlgorithm(frames), { maxSteps: 11 }).state.foreground) // => [0, 1, 0, 0]
 * ```
 */
export function mixtureBackgroundAlgorithm(
  frames: readonly TensorLike[],
  options: MixtureBackgroundOptions = {},
): Algorithm<BackgroundState> {
  const planes = frames.map((f) => greyOf(f, 'mixtureBackgroundAlgorithm').data)
  const K = options.components ?? 3
  const alpha = options.alpha ?? 0.01
  const T = options.backgroundRatio ?? 0.7
  const lambda = options.matchSigmas ?? 2.5
  const var0 = (options.initialSigma ?? 0.1) ** 2
  const w0 = options.initialWeight ?? 0.05
  const n = planes[0]?.length ?? 0
  return {
    name: 'mixture-background',
    config: { components: K, alpha, backgroundRatio: T } as Json,
    schema: channels({ metrics: { foreground: { description: 'fraction of pixels in the foreground' } } }),
    // Per pixel, K triples (weight, mean, variance).
    init: () => ({ frame: 0, model: new Float64Array(n * K * 3), foreground: new Float64Array(n) }),
    step: (s) => {
      const x = planes[s.frame] as Float64Array
      const model = Float64Array.from(s.model)
      const foreground = new Float64Array(n)
      for (let p = 0; p < n; p++) {
        const v = x[p] ?? 0
        const base = p * K * 3
        const W = (k: number): number => model[base + 3 * k] ?? 0
        const M = (k: number): number => model[base + 3 * k + 1] ?? 0
        const V = (k: number): number => model[base + 3 * k + 2] ?? 0
        // Rank components by weight over standard deviation.
        const order = Array.from({ length: K }, (_, k) => k).sort(
          (a, b) => W(b) / Math.sqrt(V(b) || 1) - W(a) / Math.sqrt(V(a) || 1),
        )
        let matched = -1
        for (const k of order)
          if (W(k) > 0 && Math.abs(v - M(k)) <= lambda * Math.sqrt(V(k))) {
            matched = k
            break
          }
        // The background components: the top-ranked whose weights reach T.
        let cum = 0
        let total = 0
        for (let k = 0; k < K; k++) total += W(k)
        const background = new Set<number>()
        for (const k of order) {
          if (W(k) <= 0) break
          background.add(k)
          cum += W(k) / (total || 1)
          if (cum > T) break
        }
        foreground[p] = matched >= 0 && background.has(matched) ? 0 : 1
        for (let k = 0; k < K; k++) model[base + 3 * k] = (1 - alpha) * W(k) + (k === matched ? alpha : 0)
        if (matched >= 0) {
          const rho = alpha / Math.max(W(matched), alpha)
          const m = M(matched)
          const mean = (1 - rho) * m + rho * v
          model[base + 3 * matched + 1] = mean
          model[base + 3 * matched + 2] = Math.max(1e-6, (1 - rho) * V(matched) + rho * (v - mean) ** 2)
        } else {
          // Replace the least probable component.
          const worst = order[order.length - 1] ?? 0
          model[base + 3 * worst] = s.frame === 0 ? 1 : w0
          model[base + 3 * worst + 1] = v
          model[base + 3 * worst + 2] = var0
        }
        let sum = 0
        for (let k = 0; k < K; k++) sum += model[base + 3 * k] ?? 0
        for (let k = 0; k < K; k++) model[base + 3 * k] = (model[base + 3 * k] ?? 0) / (sum || 1)
      }
      if (s.frame === 0) foreground.fill(0)
      return { frame: s.frame + 1, model, foreground }
    },
    done: (s) => (s.frame >= planes.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { foreground: fraction(s.foreground) } }),
  }
}

/**
 * The foreground masks of a frame sequence by a background subtractor, one per frame.
 *
 * @param frames - Grey frames `[H, W]`.
 * @param options - The method ('mixture', default, or 'running-average'), its options and tracing.
 * @returns One mask `[H, W]` per frame (1 is foreground).
 * @concept background-subtraction
 * @example
 * ```ts
 * import { subtractBackground } from 'mdml-engine/vision'
 * const still = [[0.2, 0.2], [0.2, 0.2]]
 * subtractBackground([still, still, [[0.9, 0.2], [0.2, 0.2]]], { method: 'running-average' }).length // => 3
 * ```
 */
export function subtractBackground(
  frames: readonly TensorLike[],
  options: (MixtureBackgroundOptions & RunningAverageOptions & { readonly method?: 'mixture' | 'running-average' }) &
    Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): Tensor[] {
  if (frames.length === 0) return []
  const r = greyOf(frames[0] as TensorLike, 'subtractBackground')
  const alg =
    options.method === 'running-average'
      ? runningAverageAlgorithm(frames, options)
      : mixtureBackgroundAlgorithm(frames, options)
  const masks: Tensor[] = []
  runAlgorithm(
    {
      ...alg,
      step: (s, ctx) => {
        const next = alg.step(s, ctx)
        masks.push(fromStorage(Float64Array.from(next.foreground), [r.h, r.w], 'float64'))
        return next
      },
    },
    {
      maxSteps: frames.length,
      ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
      ...(options.policy === undefined ? {} : { policy: options.policy }),
      ...(options.parent === undefined ? {} : { parent: options.parent }),
    },
  )
  return masks
}
