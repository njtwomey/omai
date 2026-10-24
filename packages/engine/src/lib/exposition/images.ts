import { WhiteboxError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** The images at one recorded step: per group, its images as flat row-major grey levels (row 0 at the top). */
export interface ImageStep {
  /** The iteration (training step, epoch) the images were taken after; 0 is the initial model. */
  readonly iteration: number
  /** images[g][k]: image k of group g, `height` times `width` grey levels. */
  readonly images: readonly (readonly (readonly number[])[])[]
  /** The training objective at this step. */
  readonly objective?: number
}

/**
 * Small grey images at recorded steps of training, in named groups drawn as rows of a mosaic: a generative model's
 * data, its reconstructions and its samples from fixed noise, so a view can play them sharpening step by step.
 */
export interface ImageSteps {
  readonly kind: 'imageSteps'
  readonly method: string
  /** Each image's size in pixels. */
  readonly height: number
  readonly width: number
  /** The groups' names, in drawing order ('data', 'reconstruction', 'samples'). */
  readonly groups: readonly string[]
  readonly steps: readonly ImageStep[]
  /** The grey levels mapped to background and ink (default [0, 1]). */
  readonly range?: readonly [number, number]
  /** What a step is (default 'step'). */
  readonly stepName?: string
  /** The name of the steps' objective (default 'objective'). */
  readonly objectiveName?: string
}

/**
 * Image steps, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a step has the wrong number of groups or an image the wrong number of pixels;
 *   `value` when the size is not a positive integer.
 * @example
 * ```ts
 * import { imageSteps } from 'whitebox-engine/exposition'
 * const r = imageSteps({
 *   method: 'vae',
 *   height: 1,
 *   width: 2,
 *   groups: ['data', 'samples'],
 *   steps: [{ iteration: 0, images: [[[0, 1]], [[0.5, 0.5]]] }],
 * })
 * r.groups // => ['data', 'samples']
 * ```
 */
export function imageSteps(record: Omit<ImageSteps, 'kind'>): ImageSteps {
  const { height, width } = record
  if (!(Number.isInteger(height) && height > 0 && Number.isInteger(width) && width > 0))
    throw new WhiteboxError('value', `imageSteps: the size ${height} by ${width} is not positive whole pixels`)
  for (const [t, step] of record.steps.entries()) {
    sameLength('imageSteps', `steps[${t}].images`, step.images.length, record.groups.length)
    for (const [g, group] of step.images.entries())
      for (const [k, img] of group.entries())
        sameLength('imageSteps', `steps[${t}].images[${g}][${k}]`, img.length, height * width)
  }
  return { kind: 'imageSteps', ...record }
}
