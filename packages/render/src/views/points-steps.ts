/**
 * The view of a `PointsSteps` record at a step: the points so far (or the set at that step) as a scatter plot whose axes
 * hold still over the whole run, the latest step's points set apart; in three dimensions, as a scene to orbit with a
 * fixed box, or projected for a still. `pointsStepsLimits` is the extent of every step together.
 */

import type { EChartsOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import { type PointsSteps, pointsAt } from 'mdml-engine/exposition'
import {
  axes3d,
  type Camera,
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
import { categoricalColour, colourScale, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { ariaOf, short, spanOf, type ViewOptions, viewFigure } from './_view.ts'

/** How the points are coloured and faded: the options of the 2-D figure and the 3-D scene. */
export interface PointsLook {
  /**
   * What colours the points: `groups` (a class each, tab10), `values` (a continuous channel along a ramp, with its
   * range written in the description) or `none`. Default: groups when the step has them, else values, else none.
   */
  readonly colour?: 'groups' | 'values' | 'none'
  /**
   * Fade a trail: a point that arrived this many recorded steps ago or more is drawn at a tenth of the opacity, those
   * in between in proportion (default no fading).
   */
  readonly fade?: number
}

/** The named viewpoints of a 3-D cloud. */
export type CameraPreset = 'iso' | 'front' | 'side' | 'top'

/** The names of the camera presets, in the order the view lists them. */
export const CAMERA_PRESETS: readonly CameraPreset[] = ['iso', 'front', 'side', 'top']

/**
 * The extent of the per-point values over the whole run (the record's `valueRange`, else the span of every step).
 *
 * @param record - The record.
 * @returns `[low, high]`.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsValueRange } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0], [1, 1]], values: [3, 7] }] })
 * pointsStepsValueRange(r) // => [3, 7]
 * ```
 */
export function pointsStepsValueRange(record: PointsSteps): [number, number] {
  if (record.valueRange !== undefined) return [record.valueRange[0], record.valueRange[1]]
  return spanOf(record.steps.flatMap((s) => s.values ?? []))
}

/**
 * The opacity of each point of a trail: one for a point that arrived at the latest step, falling in a line to a tenth
 * for a point `fade` steps older or more; all ones without fading.
 *
 * @param born - The step each point arrived at.
 * @param fade - The age in steps at which a point has faded (default none).
 * @returns One opacity per point.
 * @example
 * ```ts
 * import { trailOpacity } from 'mdml-render/views'
 * trailOpacity([0, 5, 10], 10) // => [0.1, 0.55, 1]
 * ```
 */
export function trailOpacity(born: readonly number[], fade?: number): number[] {
  if (fade === undefined || !(fade > 0)) return born.map(() => 1)
  const latest = maxOf(born)
  return born.map((b) => 0.1 + 0.9 * Math.min(1, Math.max(0, 1 - (latest - b) / fade)))
}

/**
 * The viewpoint of a named preset: `iso` is matplotlib's default, `front` looks along the y axis at the x-z plane,
 * `side` along the x axis at the y-z plane, and `top` down the z axis at the x-y plane (orthographic for the three
 * flat ones, so no depth distorts the projection).
 *
 * @param name - The preset.
 * @returns The camera's angles and projection.
 * @example
 * ```ts
 * import { cameraPreset } from 'mdml-render/views'
 * cameraPreset('top') // => { projection: 'orthographic', azimuth: -90, elevation: 90 }
 * ```
 */
export function cameraPreset(name: CameraPreset): Pick<Camera, 'projection' | 'azimuth' | 'elevation'> {
  switch (name) {
    case 'iso':
      return { projection: 'perspective', azimuth: -60, elevation: 30 }
    case 'front':
      return { projection: 'orthographic', azimuth: -90, elevation: 0 }
    case 'side':
      return { projection: 'orthographic', azimuth: 0, elevation: 0 }
    case 'top':
      return { projection: 'orthographic', azimuth: -90, elevation: 90 }
  }
}

// What colours the points of a step, resolved from the look and what the step has.
function colouring(at: ReturnType<typeof pointsAt>, look: PointsLook | undefined): 'groups' | 'values' | 'none' {
  const want = look?.colour
  if (want === 'groups' && at.groups !== undefined) return 'groups'
  if (want === 'values' && at.values !== undefined) return 'values'
  if (want !== undefined) return 'none'
  return at.groups !== undefined ? 'groups' : at.values !== undefined ? 'values' : 'none'
}

// The colour of each point along the ramp over the run's value range, in ten bands so that a chart needs ten series.
function valueColours(record: PointsSteps, values: readonly number[], mode: ColourMode, bands = 10): string[] {
  const [lo, hi] = pointsStepsValueRange(record)
  const scale = colourScale(sequentialRamp('viridis', mode), [lo, hi])
  const span = hi > lo ? hi - lo : 1
  return values.map((v) => {
    const band = Math.min(bands - 1, Math.max(0, Math.floor(((v - lo) / span) * bands)))
    return scale(lo + ((band + 0.5) / bands) * span)
  })
}

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
 * @param options - Mode and figure, and the look: what colours the points and whether a trail fades.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsView } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]] }, { step: 1, points: [[1, 1]] }] })
 * pointsStepsView(r, 1).series.length // => 2
 * ```
 */
export function pointsStepsView(
  record: PointsSteps,
  step?: number,
  options: ViewOptions & PointsLook = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25 })
  const mode = fig.mode
  const at = pointsAt(record, step ?? record.steps[record.steps.length - 1]?.step ?? 0)
  const names = record.features ?? ['x', 'y']
  const [xl, yl] = pointsStepsLimits(record)
  const old = at.points.length - at.fresh
  const how = colouring(at, options)
  const trail = options.fade !== undefined && options.fade > 0
  // The newest points are set apart in the warning colour unless a value channel or a fading trail already says what is new.
  const fresh = old > 0 && record.accumulate !== false && how !== 'values' && !trail
  const layers: Layer[] = []
  const opacity = trailOpacity(at.born, options.fade)
  const colours = how === 'values' && at.values !== undefined ? valueColours(record, at.values, mode) : undefined
  const body = fresh ? old : at.points.length
  // Points are drawn in bands of one colour and one opacity: a series each, as the chart takes one style per series.
  const bands = new Map<string, number[]>()
  for (let i = 0; i < body; i++) {
    const level = trail ? Math.round((((opacity[i] ?? 1) - 0.1) / 0.9) * 4) : 0
    const colour = colours === undefined ? '' : (colours[i] ?? '')
    const key = `${level}|${colour}`
    bands.set(key, [...(bands.get(key) ?? []), i])
  }
  for (const [key, index] of bands) {
    const [level = '0', colour = ''] = key.split('|')
    const rows = index.map((i) => at.points[i] ?? [])
    const base = fresh ? 0.45 : 0.85
    layers.push(
      points(
        [
          {
            x: rows.map((p) => p[0] ?? 0),
            y: rows.map((p) => p[1] ?? 0),
            name: 'points',
            size: at.points.length > 2000 ? 2 : 4,
            opacity: trail ? 0.1 + 0.9 * (Number(level) / 4) : base,
            outline: false,
            ...(colour === '' ? {} : { colour }),
            ...(how === 'groups' && at.groups !== undefined ? { groups: index.map((i) => at.groups?.[i] ?? 0) } : {}),
          },
        ],
        { mode },
      ),
    )
  }
  if (fresh)
    layers.push(
      points(
        [
          {
            x: at.points.slice(old).map((p) => p[0] ?? 0),
            y: at.points.slice(old).map((p) => p[1] ?? 0),
            name: 'latest',
            colour: 'warning',
            size: 7,
            outline: false,
          },
        ],
        { mode },
      ),
    )
  const [vlo, vhi] = pointsStepsValueRange(record)
  const valueNote =
    how === 'values' ? `, coloured by ${record.valueName ?? 'value'} from ${short(vlo)} to ${short(vhi)}` : ''
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
      `Scatter plot of ${at.points.length} points in ${names.join(' and ')}${valueNote}${trail ? ', older points fading' : ''}${at.caption === undefined ? '' : `: ${at.caption}`}`,
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
 * @param options - The look (what colours the points, whether a trail fades) and the colour mode.
 * @returns The 3-D scene.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * import { pointsStepsScene3d } from 'mdml-render/views'
 * const r = pointsSteps({ dims: 3, steps: [{ step: 0, points: [[0, 0, 0], [1, 2, 3]] }] })
 * pointsStepsScene3d(r, 0).items.length // => 8
 * ```
 */
export function pointsStepsScene3d(
  record: PointsSteps,
  step?: number,
  options: PointsLook & { readonly mode?: ColourMode } = {},
): Scene3D {
  const at = pointsAt(record, step ?? record.steps[record.steps.length - 1]?.step ?? 0)
  const limits = pointsStepsLimits(record)
  const unit = (v: number, d: number): number => {
    const [lo, hi] = limits[d] ?? [0, 1]
    return hi > lo ? (2 * (v - lo)) / (hi - lo) - 1 : 0
  }
  const old = at.points.length - at.fresh
  const how = colouring(at, options)
  const colours =
    how === 'values' && at.values !== undefined
      ? valueColours(record, at.values, options.mode ?? 'light', 24)
      : undefined
  const opacity = trailOpacity(at.born, options.fade)
  const trail = options.fade !== undefined && options.fade > 0
  const items = at.points.map((p, i) => {
    const latest = i >= old && record.accumulate !== false
    const fill: Paint =
      how === 'groups' && at.groups !== undefined
        ? (categoricalColour((at.groups[i] ?? 0) % 10) as Paint)
        : colours !== undefined
          ? (colours[i] as Paint)
          : latest && !trail
            ? 'tab1'
            : 'tab0'
    return point3d([unit(p[0] ?? 0, 0), unit(p[1] ?? 0, 1), unit(p[2] ?? 0, 2)], {
      fill,
      r: latest ? 2.6 : 1.8,
      ...(trail ? { opacity: opacity[i] ?? 1 } : {}),
    })
  })
  const [vlo, vhi] = pointsStepsValueRange(record)
  const note =
    colours === undefined ? '' : `, coloured by ${record.valueName ?? 'value'} from ${short(vlo)} to ${short(vhi)}`
  return scene3d([...items, ...axes3d(1.1)], {
    title: `${at.points.length} points in ${(record.features ?? ['x', 'y', 'z']).join(', ')}${note}`,
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
