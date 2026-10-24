// Private exposition helpers of the generative methods (methods/generative): generated samples beside the data at each
// recorded step (embedding steps coloured data against samples), a density on a grid at each step (a decision surface
// whose steps carry values), a regular grid pushed through a learned map (embedding steps with the grid's edges), and
// loss curves bucketed over steps.
import { linspaceNumbers } from '../../lib/base/index.ts'
import {
  type DecisionSurface,
  decisionSurface,
  type EmbeddingSteps,
  embeddingSteps,
  inputRows,
  type LearningCurve,
  learningCurve,
  type SurfaceStep,
} from '../../lib/exposition/index.ts'
import type { Tensor } from '../../lib/tensor/index.ts'
import { bucketed, type Prepared, smoothed, thin, toModel } from './_shared.ts'

/** The most data rows drawn beside the samples. */
export const DATA_LIMIT = 500

/** The box a picture of the data spans: its range widened by `padding` of the range on each side. */
export function boxOf(rows: readonly (readonly number[])[], padding: number): [number, number, number, number] {
  let x0 = Number.POSITIVE_INFINITY
  let x1 = Number.NEGATIVE_INFINITY
  let y0 = Number.POSITIVE_INFINITY
  let y1 = Number.NEGATIVE_INFINITY
  for (const r of rows) {
    x0 = Math.min(x0, r[0] ?? 0)
    x1 = Math.max(x1, r[0] ?? 0)
    y0 = Math.min(y0, r[1] ?? 0)
    y1 = Math.max(y1, r[1] ?? 0)
  }
  const sx = x1 > x0 ? x1 - x0 : 1
  const sy = y1 > y0 ? y1 - y0 : 1
  return [x0 - padding * sx, x1 + padding * sx, y0 - padding * sy, y1 + padding * sy]
}

/** A point clamped into a box (samples that stray far stay at the picture's edge, so the axes keep still). */
function clamp(p: readonly number[], box: readonly [number, number, number, number]): number[] {
  return [Math.min(Math.max(p[0] ?? 0, box[0]), box[1]), Math.min(Math.max(p[1] ?? 0, box[2]), box[3]), ...p.slice(2)]
}

/** Options of {@link samplesSteps}. */
export interface SamplesStepsOptions {
  /** What the samples are ('generated', 'reconstruction', 'denoised'). */
  readonly name?: string
  /** The objective at each frame. */
  readonly objective?: readonly number[]
  /** Draw no data rows (the points are the samples only). */
  readonly noData?: boolean
}

/**
 * Samples at each recorded step beside the data, as embedding steps: the data rows (at most {@link DATA_LIMIT}, the
 * same in every frame) then the samples, coloured 'data' against the samples' name; samples straying far outside the
 * data's range are clamped to its edge so the axes keep still. Only the first two coordinates are drawn.
 *
 * @param method - The record's method name.
 * @param p - The prepared data.
 * @param steps - The step of each frame.
 * @param samples - The samples at each frame, in the data's coordinates.
 * @param options - The samples' name and the objective.
 * @returns The record.
 */
export function samplesSteps(
  method: string,
  p: Prepared,
  steps: readonly number[],
  samples: readonly (readonly (readonly number[])[])[],
  options: SamplesStepsOptions = {},
): EmbeddingSteps {
  const data = options.noData === true ? [] : thin(p.rows, DATA_LIMIT).map((r) => [r[0] ?? 0, r[1] ?? 0])
  const box = boxOf(p.rows, 0.6)
  const m = samples[0]?.length ?? 0
  return embeddingSteps({
    method,
    positions: samples.map((s) => [...data, ...s.map((q) => clamp([q[0] ?? 0, q[1] ?? 0], box))]),
    iterations: [...steps],
    ...(options.objective === undefined ? {} : { objective: [...options.objective] }),
    colourBy: {
      kind: 'classes',
      labels: [...data.map(() => 0), ...Array.from({ length: m }, () => 1)],
      classes: ['data', options.name ?? 'generated'],
    },
  })
}

/** A regular grid over the data's first two coordinates: its axes and its points, in the model's coordinates. */
export interface Grid {
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** Points `[ny nx, d]`, row-major in y then x, standardised as the model sees them. */
  readonly points: Tensor
}

/** A grid of `resolution` by `resolution` points over the data's range widened by `padding` of it. */
export function gridOf(p: Prepared, resolution = 40, padding = 0.15): Grid {
  const [x0, x1, y0, y1] = boxOf(p.rows, padding)
  const x = linspaceNumbers(x0, x1, resolution)
  const y = linspaceNumbers(y0, y1, resolution)
  const rows: number[][] = []
  for (const b of y) for (const a of x) rows.push([a, b])
  return { x, y, points: toModel(p, rows) }
}

/** Values over a grid's points (row-major) as a surface grid. */
export function asGrid(grid: Grid, values: ArrayLike<number>): number[][] {
  return grid.y.map((_, i) => grid.x.map((_, j) => values[i * grid.x.length + j] ?? 0))
}

/**
 * A density on a grid at each recorded step, as a decision surface whose steps carry `values`, with the data rows and
 * the objective at each step.
 *
 * @param p - The prepared data (for the rows and feature names).
 * @param grid - The grid.
 * @param steps - The step of each frame.
 * @param densities - The density at the grid's points at each frame, in the data's coordinates.
 * @param objective - The objective's name and its value at each frame.
 * @returns The record.
 */
export function densitySteps(
  p: Prepared,
  grid: Grid,
  steps: readonly number[],
  densities: readonly ArrayLike<number>[],
  objective?: { readonly name: string; readonly values: readonly number[] },
): DecisionSurface {
  const frames: SurfaceStep[] = densities.map((d, t) => {
    const o = objective?.values[t]
    return {
      iteration: steps[t] ?? t,
      values: asGrid(grid, d),
      ...(o === undefined || !Number.isFinite(o) ? {} : { objective: o }),
    }
  })
  const last = frames.at(-1)
  const rows = thin(p.rows, 1000).map((r) => [r[0] ?? 0, r[1] ?? 0])
  return decisionSurface({
    features: [p.featureNames[0] ?? 'x0', p.featureNames[1] ?? 'x1'],
    x: [...grid.x],
    y: [...grid.y],
    ...(last?.values === undefined ? {} : { values: last.values }),
    rows: inputRows({ points: rows, features: [p.featureNames[0] ?? 'x0', p.featureNames[1] ?? 'x1'] }),
    steps: frames,
    stepName: 'step',
    ...(objective === undefined
      ? {}
      : {
          objective: {
            name: objective.name,
            iterations: [...steps],
            values: [...objective.values],
          },
        }),
  })
}

/** The edges of a regular grid of `nx` by `ny` nodes (row-major), each with weight 1. */
export function gridEdges(nx: number, ny: number): [number, number, number][] {
  const edges: [number, number, number][] = []
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) {
      const k = i * nx + j
      if (j + 1 < nx) edges.push([k, k + 1, 1])
      if (i + 1 < ny) edges.push([k, k + nx, 1])
    }
  return edges
}

/**
 * A loss trace as a learning curve over steps: the minibatch loss averaged over buckets of steps against its moving
 * average, or two traces against each other (a discriminator's and a generator's loss, a reconstruction term and a
 * KL term).
 *
 * @param first - The first trace and its label.
 * @param second - The second trace and its label (default the first, smoothed).
 * @returns The record.
 */
export function stepCurve(
  first: { readonly values: readonly number[]; readonly label: string },
  second?: { readonly values: readonly number[]; readonly label: string },
): LearningCurve {
  const a = bucketed(first.values)
  const b = second === undefined ? { means: smoothed(a.means) } : bucketed(second.values)
  return learningCurve({
    sizes: a.steps,
    train: a.means,
    validation: b.means.slice(0, a.means.length),
    resource: 'step',
    labels: { train: first.label, validation: second?.label ?? `${first.label} (moving average)` },
  })
}

/** The value of a trace at each frame's step (the mean of the steps since the previous frame; NaN at step 0). */
export function atFrames(values: readonly number[], steps: readonly number[]): number[] {
  return steps.map((s, t) => {
    const from = t === 0 ? 0 : (steps[t - 1] ?? 0)
    if (s === 0) return Number.NaN
    let sum = 0
    for (let i = from; i < s; i++) sum += values[i] ?? 0
    return sum / Math.max(1, s - from)
  })
}
