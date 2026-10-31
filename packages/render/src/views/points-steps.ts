/**
 * The view of a `PointsSteps` record at a step: the points so far (or the set at that step) as a scatter plot whose axes
 * hold still over the whole run, the latest step's points set apart; in three dimensions, as a scene to orbit with a
 * fixed box, or projected for a still. `pointsStepsLimits` is the extent of every step together.
 */

import type { EChartsOption } from 'echarts'
import { type PointsSteps, pointsAt } from 'mdml-engine/exposition'
import {
  axes3d,
  fitCamera,
  layer3d,
  camera as makeCamera,
  type Paint,
  point3d,
  type Scene,
  type Scene3D,
  scene,
  scene3d,
} from 'mdml-engine/scene'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { ariaOf, spanOf, type ViewOptions, viewFigure } from './_view.ts'

/**
 * The extent of each coordinate (the record's `range`, else the span over every step of the record), so the picture's axes do not move as points arrive.
 *
 * @param record - The record.
 * @returns One `[low, high]` per coordinate.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsLimits } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 1]] }, { step: 1, points: [[4, -1]] }] })
 * pointsStepsLimits(r) // => [[0, 4], [-1, 1]]
 * ```
 */
export function pointsStepsLimits(record: PointsSteps): [number, number][] {
  if (record.range !== undefined) return record.range.map((r) => [r[0], r[1]])
  return Array.from({ length: record.dims }, (_, d) =>
    spanOf(record.steps.flatMap((s) => s.points.map((p) => p[d] ?? 0))),
  )
}

/**
 * A two-dimensional points record at a step as a figure: all the points drawn so far, small and faint, the points the
 * latest step added in the warning colour, groups in tab10.
 *
 * @param record - The record (two-dimensional).
 * @param step - The tape step (default the last recorded one).
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsView } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]] }, { step: 1, points: [[1, 1]] }] })
 * pointsStepsView(r, 1).series.length // => 2
 * ```
 */
export function pointsStepsView(record: PointsSteps, step?: number, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25 })
  const mode = fig.mode
  const at = pointsAt(record, step ?? record.steps[record.steps.length - 1]?.step ?? 0)
  const names = record.features ?? ['x', 'y']
  const [xl, yl] = pointsStepsLimits(record)
  const old = at.points.length - at.fresh
  const fresh = old > 0 && record.accumulate !== false
  const layers: Layer[] = []
  const xy = (rows: readonly (readonly number[])[]): { x: number[]; y: number[] } => ({
    x: rows.map((p) => p[0] ?? 0),
    y: rows.map((p) => p[1] ?? 0),
  })
  const body = fresh ? at.points.slice(0, old) : at.points
  const bodyGroups = at.groups === undefined ? undefined : fresh ? at.groups.slice(0, old) : at.groups
  if (body.length > 0)
    layers.push(
      points(
        [
          {
            ...xy(body),
            name: 'points',
            size: body.length > 2000 ? 2 : 4,
            opacity: fresh ? 0.45 : 0.85,
            outline: false,
            ...(bodyGroups === undefined ? {} : { groups: bodyGroups }),
          },
        ],
        { mode },
      ),
    )
  if (fresh)
    layers.push(
      points([{ ...xy(at.points.slice(old)), name: 'latest', colour: 'warning', size: 7, outline: false }], { mode }),
    )
  return {
    ...cartesian(layers, {
      mode,
      figure: fig,
      x: { label: names[0] ?? 'x', limits: xl ?? [0, 1] },
      y: { label: names[1] ?? 'y', limits: yl ?? [0, 1] },
      legend: fresh,
      tooltip: false,
    }),
    animation: false,
    ...ariaOf(
      `Scatter plot of ${at.points.length} points in ${names.join(' and ')}${at.caption === undefined ? '' : `: ${at.caption}`}`,
    ),
  }
}

/**
 * A three-dimensional points record at a step as a scene to orbit: the points drawn so far, scaled so that every step's
 * extent fits the unit box (so the cloud does not rescale as it fills in), with axes, groups in tab10 and the latest
 * step's points larger.
 *
 * @param record - The record (three-dimensional).
 * @param step - The tape step (default the last recorded one).
 * @returns The 3-D scene.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsScene3d } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 3, steps: [{ step: 0, points: [[0, 0, 0], [1, 2, 3]] }] })
 * pointsStepsScene3d(r, 0).items.length // => 8
 * ```
 */
export function pointsStepsScene3d(record: PointsSteps, step?: number): Scene3D {
  const at = pointsAt(record, step ?? record.steps[record.steps.length - 1]?.step ?? 0)
  const limits = pointsStepsLimits(record)
  const unit = (v: number, d: number): number => {
    const [lo, hi] = limits[d] ?? [0, 1]
    return hi > lo ? (2 * (v - lo)) / (hi - lo) - 1 : 0
  }
  const old = at.points.length - at.fresh
  const items = at.points.map((p, i) =>
    point3d([unit(p[0] ?? 0, 0), unit(p[1] ?? 0, 1), unit(p[2] ?? 0, 2)], {
      fill:
        at.groups === undefined
          ? i >= old && record.accumulate !== false
            ? 'tab1'
            : 'tab0'
          : (categoricalColour((at.groups[i] ?? 0) % 10) as Paint),
      r: i >= old && record.accumulate !== false ? 2.6 : 1.8,
    }),
  )
  return scene3d([...items, ...axes3d(1.1)], {
    title: `${at.points.length} points in ${(record.features ?? ['x', 'y', 'z']).join(', ')}`,
  })
}

/**
 * A three-dimensional points record at a step projected to a flat scene from a fixed camera, for stills (slides,
 * export) where the orbit view is not available.
 *
 * @param record - The record (three-dimensional).
 * @param step - The tape step (default the last recorded one).
 * @param size - The scene's width and height (default 480 by 360).
 * @returns The scene.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsScene } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 3, steps: [{ step: 0, points: [[0, 0, 0], [1, 2, 3]] }] })
 * pointsStepsScene(r, 0).kind // => 'scene'
 * ```
 */
export function pointsStepsScene(
  record: PointsSteps,
  step?: number,
  size: readonly [number, number] = [480, 360],
): Scene {
  const s3 = pointsStepsScene3d(record, step)
  return scene(size[0], size[1], [
    layer3d(s3, fitCamera(makeCamera(), s3), { x: 0, y: 0, width: size[0], height: size[1] }),
  ])
}
