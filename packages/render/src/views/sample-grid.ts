/**
 * The view of a `SampleGrid` record at a step, as a scene: the samples of the viewed step in a grid (small images drawn
 * pixel by pixel, or text in cells), with an optional filmstrip beneath, a row per earlier step on a log scale of the
 * step, so the samples are seen to sharpen. The scene's size follows its content.
 */

import { filmstripSteps, type GridSample, type SampleGrid, type SampleStep, sampleStepAt } from 'omai-engine/exposition'
import { image, rect, type Scene, type SceneNode, scene, text, toPixels } from 'omai-engine/scene'

/** Options of {@link sampleGridScene}. */
export interface SampleGridOptions {
  /** Samples per row of the grid (default the record's, else 4). */
  readonly columns?: number
  /** Rows of earlier steps in the filmstrip (default 0: none). */
  readonly filmstrip?: number
  /** The scene's width in pixels (default 440). */
  readonly width?: number
}

const GAP = 6
const LABEL = 14

// An image sample as nested rows, for the scene's image node.
function rowsOf(record: SampleGrid, s: GridSample): number[][][] {
  const { height = 0, width = 0, channels = 1 } = record
  const v = s as readonly number[]
  return Array.from({ length: height }, (_, r) =>
    Array.from({ length: width }, (_, c) =>
      Array.from({ length: channels }, (_, k) => v[(r * width + c) * channels + k] ?? 0),
    ),
  )
}

// One sample as nodes in a cell: an image stretched to the cell, or text clipped to it.
function cell(
  record: SampleGrid,
  s: GridSample,
  x: number,
  y: number,
  w: number,
  h: number,
  caption?: string,
): SceneNode[] {
  if (record.modality === 'image')
    return [
      image(x, y, w, h, toPixels(rowsOf(record, s), { range: record.range ?? [0, 1] }), {
        smoothing: false,
        title: caption ?? 'sample',
      }),
    ]
  const chars = Math.max(4, Math.floor(w / 6.2))
  const words = String(s)
  const lines: string[] = []
  for (let i = 0; i < words.length && lines.length < Math.max(1, Math.floor(h / 13)); i += chars)
    lines.push(words.slice(i, i + chars))
  return [
    rect(x, y, w, h, { fill: 'surface', stroke: 'grid', strokeWidth: 1 }),
    ...lines.map((l, i) => text(x + 4, y + 13 + 13 * i, l, { size: 11, fill: 'fg' })),
  ]
}

/**
 * The scene of a sample grid at a tape step: the samples of the latest recorded step at or before it in a grid with
 * their captions, the step named above, and, if asked, the filmstrip beneath. Text cells show as many characters as fit.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded one).
 * @param options - Columns, filmstrip rows and width.
 * @returns The scene.
 * @example
 * ```ts
 * import { sampleGrid } from 'omai-engine/exposition'
 * import { sampleGridScene } from 'omai-render/views'
 * const g = sampleGrid({ modality: 'image', height: 1, width: 2, steps: [{ step: 0, samples: [[0, 1], [1, 0]] }] })
 * sampleGridScene(g, 0, { columns: 2 }).title // => '2 image samples after step 0'
 * ```
 */
export function sampleGridScene(record: SampleGrid, step?: number, options: SampleGridOptions = {}): Scene {
  const width = options.width ?? 440
  const columns = Math.max(1, options.columns ?? record.columns ?? 4)
  const stepName = record.stepName ?? 'step'
  const current: SampleStep | undefined = sampleStepAt(record, step ?? record.steps[record.steps.length - 1]?.step ?? 0)
  const nodes: SceneNode[] = []
  if (current === undefined)
    return scene(width, 40, [text(8, 24, 'no samples recorded yet', { size: 12, fill: 'muted' })])
  const samples = current.samples
  const rows = Math.ceil(samples.length / columns)
  const side = Math.floor((width - GAP * (columns + 1)) / columns)
  const cellH =
    record.modality === 'image'
      ? Math.floor((side * (record.height ?? 1)) / (record.width ?? 1))
      : Math.max(28, Math.round(side * 0.55))
  const captions = current.labels !== undefined
  let y = 22
  nodes.push(text(GAP, 14, `${stepName} ${current.step}`, { size: 12, fill: 'fg', weight: 'bold' }))
  for (const [i, s] of samples.entries()) {
    const x = GAP + (i % columns) * (side + GAP)
    const top = y + Math.floor(i / columns) * (cellH + GAP + (captions ? LABEL : 0))
    nodes.push(...cell(record, s, x, top, side, cellH, current.labels?.[i]))
    if (captions) nodes.push(text(x, top + cellH + 11, current.labels?.[i] ?? '', { size: 10, fill: 'muted' }))
  }
  y += rows * (cellH + GAP + (captions ? LABEL : 0))
  const earlier =
    options.filmstrip === undefined || options.filmstrip <= 0
      ? []
      : filmstripSteps(record, current.step, options.filmstrip)
  if (earlier.length > 0) {
    nodes.push(text(GAP, y + 10, `earlier ${stepName}s`, { size: 12, fill: 'fg', weight: 'bold' }))
    y += 16
    // A filmstrip row shows the first few samples of an earlier step, small, with the step on its left.
    const labelW = 52
    const small = Math.max(12, Math.min(34, Math.floor((width - labelW - GAP * 10) / 8)))
    const smallH =
      record.modality === 'image' ? Math.floor((small * (record.height ?? 1)) / (record.width ?? 1)) : small
    for (const e of earlier) {
      nodes.push(text(GAP, y + smallH / 2 + 4, `${e.step}`, { size: 11, fill: 'muted' }))
      if (record.modality === 'image')
        for (const [j, s] of e.samples.slice(0, 8).entries())
          nodes.push(...cell(record, s, labelW + j * (small + 3), y, small, smallH))
      else nodes.push(...cell(record, e.samples[0] ?? '', labelW, y, width - labelW - GAP, smallH))
      y += smallH + 4
    }
  }
  return scene(width, Math.ceil(y + GAP), nodes, {
    title: `${samples.length} ${record.modality} samples after ${stepName} ${current.step}${earlier.length > 0 ? `, with ${earlier.length} earlier ${stepName}s in a filmstrip` : ''}`,
  })
}
