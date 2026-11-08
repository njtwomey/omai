/**
 * The exposition record of samples over training (`sampleGrid`): what a generative model produces at recorded steps
 * from fixed noise (small images, or strings of text), so a view can show the grid of the viewed step with a filmstrip
 * of earlier steps beside it and the reader sees the samples sharpen. `sampleStepAt` and `filmstripSteps` choose what
 * to show at a step.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'

/** A sample: an image as flat row-major pixel values (grey, or interleaved red, green and blue), or a string. */
export type GridSample = readonly number[] | string

/** The samples at one recorded step. */
export interface SampleStep {
  /** The tape step the samples were drawn after. */
  readonly step: number
  readonly samples: readonly GridSample[]
  /** A short caption under each sample (its class, a score). */
  readonly labels?: readonly string[]
}

/** Samples from a generative model at recorded steps. */
export interface SampleGrid {
  readonly kind: 'sampleGrid'
  readonly modality: 'image' | 'text'
  /** The images' size in pixels (images only). */
  readonly height?: number
  readonly width?: number
  /** Values per pixel: 1 for grey (default), 3 for colour. */
  readonly channels?: 1 | 3
  /** The pixel values mapped to the ends of the ramp (default [0, 1]). */
  readonly range?: readonly [number, number]
  readonly steps: readonly SampleStep[]
  /** Samples per row of the grid (default 4). */
  readonly columns?: number
  /** What a step is (default `step`): `epoch`, `iteration`. */
  readonly stepName?: string
  readonly caption?: string
}

/**
 * A sample grid, with the samples checked to match the modality and the images the size.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `value` for an image size that is not whole positive pixels or steps out of order; `shape` when an
 *   image has the wrong number of values, a text sample is not a string, or the labels do not match the samples.
 * @example
 * ```ts
 * import { sampleGrid } from 'omai-engine/exposition'
 * const g = sampleGrid({ modality: 'text', steps: [{ step: 0, samples: ['xqz', 'ab'] }] })
 * g.steps.length // => 1
 * ```
 */
export function sampleGrid(record: Omit<SampleGrid, 'kind'>): SampleGrid {
  if (record.modality === 'image') {
    const { height, width } = record
    if (
      !(
        height !== undefined &&
        width !== undefined &&
        Number.isInteger(height) &&
        Number.isInteger(width) &&
        height > 0 &&
        width > 0
      )
    )
      throw new OmaiError('value', 'sampleGrid: images need a height and width in whole pixels')
  }
  const size = (record.height ?? 0) * (record.width ?? 0) * (record.channels ?? 1)
  let previous = Number.NEGATIVE_INFINITY
  for (const [t, s] of record.steps.entries()) {
    if (s.step < previous) throw new OmaiError('value', `sampleGrid: steps[${t}] is out of order`)
    previous = s.step
    for (const [i, x] of s.samples.entries()) {
      if (record.modality === 'text' && typeof x !== 'string')
        throw new OmaiError('shape', `sampleGrid: steps[${t}].samples[${i}] is not text`)
      if (record.modality === 'image' && (typeof x === 'string' || x.length !== size))
        throw new OmaiError('shape', `sampleGrid: steps[${t}].samples[${i}] is not ${size} pixel values`)
    }
    if (s.labels !== undefined && s.labels.length !== s.samples.length)
      throw new OmaiError(
        'shape',
        `sampleGrid: steps[${t}].labels has ${s.labels.length} entries, expected ${s.samples.length}`,
      )
  }
  return seal({ kind: 'sampleGrid', ...record })
}

/**
 * The samples to show at a tape step: those of the latest recorded step at or before it (the first when none is yet).
 *
 * @param record - The record.
 * @param step - The tape step.
 * @returns The recorded step, or `undefined` for a record without steps.
 * @example
 * ```ts
 * import { sampleGrid, sampleStepAt } from 'omai-engine/exposition'
 * const g = sampleGrid({ modality: 'text', steps: [{ step: 0, samples: ['a'] }, { step: 5, samples: ['b'] }] })
 * sampleStepAt(g, 4)?.samples // => ['a']
 * ```
 */
export function sampleStepAt(record: SampleGrid, step: number): SampleStep | undefined {
  let found: SampleStep | undefined
  for (const s of record.steps) if (s.step <= step) found = s
  return found ?? record.steps[0]
}

/**
 * The earlier recorded steps for a filmstrip: up to `count` of those before the viewed one, spread over the run on a
 * log scale of the step (the early steps, where the samples change most, are close together in it) and in order.
 *
 * @param record - The record.
 * @param step - The viewed tape step.
 * @param count - How many earlier steps (default 4).
 * @returns The earlier steps, earliest first, without the viewed one.
 * @example
 * ```ts
 * import { filmstripSteps, sampleGrid } from 'omai-engine/exposition'
 * const g = sampleGrid({ modality: 'text', steps: [0, 1, 2, 4, 8, 16, 32].map((step) => ({ step, samples: ['x'] })) })
 * filmstripSteps(g, 32, 3).map((s) => s.step) // => [0, 4, 16]
 * ```
 */
export function filmstripSteps(record: SampleGrid, step: number, count = 4): SampleStep[] {
  const current = sampleStepAt(record, step)
  const before = record.steps.filter((s) => s.step < (current?.step ?? 0))
  if (before.length <= count) return [...before]
  const chosen = new Set<number>()
  const lo = Math.log1p(before[0]?.step ?? 0)
  const hi = Math.log1p(before[before.length - 1]?.step ?? 0)
  for (let k = 0; k < count; k++) {
    const target = lo + (count === 1 ? 0 : (k / (count - 1)) * (hi - lo))
    let best = 0
    for (const [i, s] of before.entries())
      if (Math.abs(Math.log1p(s.step) - target) < Math.abs(Math.log1p(before[best]?.step ?? 0) - target)) best = i
    chosen.add(best)
  }
  return [...chosen].sort((a, b) => a - b).map((i) => before[i] as SampleStep)
}
