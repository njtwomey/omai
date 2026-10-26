/**
 * Progressive path tracing: render a scene with the graphics core's path tracer and record the tone-mapped image each
 * time the samples per pixel double, with the mean standard error of the pixels, so a view can play the noise
 * falling away as the image converges.
 */

import { MdmlError } from '../base/index.ts'
import { type ImageStep, type ImageSteps, imageSteps } from '../exposition/index.ts'
import type { Key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import type { RenderCamera } from './camera.ts'
import { type PathTraceOptions, type PathTracerState, pathTracerAlgorithm } from './pathtracing.ts'
import type { RayScene } from './raytracing.ts'
import { type FloatImage, imageToPixels } from './textures.ts'

/** Options of {@link progressiveRender}. */
export interface ProgressiveRenderOptions extends PathTraceOptions {
  /** The key every sample is drawn from. */
  readonly key: Key
  /** Exposure in stops applied before sRGB encoding (default 0). */
  readonly exposure?: number
  /** Called at each recorded step with the fraction done and the record so far. */
  readonly onProgress?: (fraction: number, record: ImageSteps) => void
}

/** The outcome of {@link progressiveRender}: the final linear image and the recorded steps. */
export interface ProgressiveRender {
  readonly image: FloatImage
  readonly samples: number
  readonly exposition: readonly [ImageSteps]
}

// The mean image of a state, and the mean standard error of its pixel means over every channel.
function summarise(state: PathTracerState, width: number, height: number): { image: FloatImage; error: number } {
  const m = Math.max(1, state.samples)
  const rgb = new Float64Array(state.sum.length)
  let se = 0
  for (let i = 0; i < rgb.length; i++) {
    const mu = (state.sum[i] ?? 0) / m
    rgb[i] = mu
    se += m > 1 ? Math.sqrt(Math.max(0, (state.sumSquares[i] ?? 0) / m - mu * mu) / (m - 1)) : 0
  }
  return { image: { width, height, rgb }, error: se / Math.max(1, rgb.length) }
}

/**
 * Path-trace a scene progressively, recording the image at 1, 2, 4, ... samples per pixel (and at the last sample):
 * each step holds the tone-mapped colour image and the mean standard error of the pixels, which falls like one over
 * the square root of the samples.
 *
 * @param scene - The scene.
 * @param camera - The camera.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param options - Samples, bounces and jitter, the key, the exposure and a progress callback.
 * @returns The final linear image, the samples taken and an `imageSteps` record of the render converging.
 */
export function progressiveRender(
  scene: RayScene,
  camera: RenderCamera,
  width: number,
  height: number,
  options: ProgressiveRenderOptions,
): ProgressiveRender {
  const { key, exposure, onProgress, ...trace } = options
  const samples = trace.samples ?? 16
  if (!(Number.isInteger(samples) && samples >= 1)) throw new MdmlError('value', 'progressiveRender: samples >= 1')
  const algorithm = pathTracerAlgorithm(scene, camera, width, height, trace)
  const marks: number[] = []
  for (let s = 1; s < samples; s *= 2) marks.push(s)
  marks.push(samples)
  const steps: ImageStep[] = []
  const recordOf = (so: readonly ImageStep[]): ImageSteps =>
    imageSteps({
      method: 'path tracing',
      height,
      width,
      channels: 3,
      groups: ['render'],
      steps: [...so],
      stepName: 'samples per pixel',
      objectiveName: 'mean standard error',
    })
  let state: PathTracerState | undefined
  let done = 0
  for (const mark of marks) {
    // Each slice continues the same run: step t always draws from foldIn(key, t), so the render is one keyed run.
    const run = runAlgorithm(algorithm, {
      key,
      maxSteps: mark - done,
      ...(state === undefined ? {} : { start: { state, step: done } }),
    })
    state = run.state
    done = mark
    const { image, error } = summarise(state, width, height)
    const pixels = imageToPixels(image, { exposure: exposure ?? 0 })
    const colour: number[] = []
    for (let i = 0; i < width * height; i++)
      for (let c = 0; c < 3; c++) colour.push((pixels.rgba[4 * i + c] ?? 0) / 255)
    steps.push({ iteration: mark, images: [[colour]], objective: error })
    onProgress?.(mark / samples, recordOf(steps))
  }
  const last = summarise(state as PathTracerState, width, height)
  return { image: last.image, samples, exposition: [recordOf(steps)] }
}
