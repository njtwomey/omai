/**
 * Player frames for a margin classifier (a support vector machine) over its solver's steps: the training rows over the
 * decision surface with the margins dashed and the support vectors ringed by the size of their dual coefficients, the
 * functional margins, and the dual objective as SMO (sequential minimal optimisation) updates one pair at a time.
 */

import type { EChartsOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import { type DecisionSurface, type MarginRecord, surfaceAt } from 'mdml-engine/exposition'
import { histogram } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { outlined, short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'
import { classIndices, classPointSets, decisionSurfaceLayers, type Rows } from './surfaces.ts'

/** Options of {@link marginFrames}. */
export interface MarginViewOptions extends ViewOptions {
  /** Training rows to draw instead of the record's (their first two columns, or `columns`), and their labels. */
  readonly points?: Rows
  readonly labels?: readonly (number | string)[]
  readonly columns?: readonly [number, number]
  /** A decision surface to draw under the rows instead of the record's. */
  readonly surface?: DecisionSurface
  /** The most frames the solver's steps are played in, without recorded surfaces (default 120). */
  readonly maxFrames?: number
}

/**
 * A margin classifier (an SVM) as frames over its solver's steps: the training rows over the decision surface with the
 * margins f = ±1 dashed and the support vectors ringed (by the size of their dual coefficients at the end), the pair
 * SMO updated at the step joined by a line; the functional margins y·f(x) as a histogram with 0 (the boundary) and 1
 * (the margin) marked; and the dual objective with the step marked. When the surface records its steps through
 * training, each frame is one of them: the boundary, margins and support vectors as they were then. Without solver
 * steps it is one frame.
 *
 * @param record - The record.
 * @param options - Mode, figure, training rows, labels, columns, the decision surface and the frame budget.
 * @returns The frames.
 */
export function marginFrames(record: MarginRecord, options: MarginViewOptions = {}): ViewFrames {
  const surface = options.surface ?? record.surface
  const own = options.points === undefined
  const rows = options.points ?? record.rows?.points
  const columns = own ? ([0, 1] as const) : (options.columns ?? [0, 1])
  const classes = surface?.classes ?? record.rows?.classes
  const labels =
    options.labels !== undefined ? classIndices(options.labels, classes) : own ? record.rows?.labels : undefined
  const fig = viewFigure(options, { aspect: rows === undefined ? 1.6 : 1.75 })
  const mode = fig.mode
  const dual = record.dualObjective
  const solverSteps = record.steps ?? dual?.map((_, i) => i) ?? []
  const recorded = surface?.steps !== undefined && surface.steps.length > 1 ? surface.steps : undefined
  const steps = dual?.length ?? 0
  const count = recorded?.length ?? Math.max(1, Math.min(steps, options.maxFrames ?? 120))
  // The entry of the solver's trace a frame shows.
  const traceIndex = (frame: number): number => {
    const f = Math.min(Math.max(frame, 0), count - 1)
    if (recorded !== undefined) {
      const it = recorded[f]?.iteration ?? 0
      let best = 0
      for (let i = 1; i < solverSteps.length; i++)
        if (Math.abs((solverSteps[i] ?? 0) - it) < Math.abs((solverSteps[best] ?? 0) - it)) best = i
      return best
    }
    return steps <= 1 ? 0 : Math.round((f * (steps - 1)) / Math.max(1, count - 1))
  }
  const alpha = record.support.map((_, i) => Math.abs(record.dualCoefficients[i] ?? 0))
  const top = Math.max(1e-12, maxOf(alpha))
  // y·f(x) per row at a frame: the recorded step's decision values when there are, the final margins otherwise.
  const marginsAt = (frame: number): readonly number[] => {
    const values = recorded?.[Math.min(Math.max(frame, 0), count - 1)]?.rowValues
    if (values === undefined || labels === undefined) return record.margins
    return values.map((v, i) => ((labels[i] ?? 0) === 1 ? v : -v))
  }
  const option = (frame: number): EChartsOption => {
    const f = Math.min(Math.max(frame, 0), count - 1)
    const s = traceIndex(f)
    const step = recorded?.[f]
    const panels: Record<string, Panel> = {
      margins: {
        layers: [
          histogram(marginsAt(f), { bins: 30, name: '__margins', colour: 'tab0', mode }),
          referenceLines(
            {
              x: [
                { value: 0, label: 'boundary' },
                { value: 1, label: 'margin' },
              ],
            },
            { mode },
          ),
        ],
        options: { x: { label: 'functional margin y·f(x)' }, y: { label: 'rows' } },
      },
    }
    if (dual !== undefined && steps > 1)
      panels['dual'] = {
        layers: [
          lines([{ x: solverSteps, y: dual, name: 'dual objective', colour: 'fgSecondary', width: 1.4 }], { mode }),
          points([{ x: [solverSteps[s] ?? s], y: [dual[s] ?? 0], name: '__now', colour: 'accent', size: 9 }], {
            mode,
          }),
        ],
        options: { x: { label: 'solver step' }, y: { label: 'dual objective' } },
      }
    if (rows !== undefined) {
      const layers: Layer[] = []
      const shown = surface === undefined ? undefined : surfaceAt(surface, f)
      if (shown !== undefined) {
        // The surface's regions only: the rows are drawn here, with the support vectors.
        const { rows: _rows, ...regions } = shown
        layers.push(...decisionSurfaceLayers(regions, { mode }).layers)
      }
      layers.push(points(classPointSets(rows, columns, labels, classes, { size: 6 }), { mode }))
      const [a, b] = columns
      // The support vectors at the step (rings of one size), or at the end (by their coefficients).
      const support = step?.highlight ?? record.support
      layers.push(
        outlined(
          points(
            [
              {
                x: support.map((i) => rows[i]?.[a] ?? 0),
                y: support.map((i) => rows[i]?.[b] ?? 0),
                ...(step?.highlight === undefined
                  ? { sizes: alpha.map((v) => v / top), sizeRange: [10, 20] as const }
                  : { size: 13 }),
                name: 'support vectors',
                colour: 'none',
                outline: false,
              },
            ],
            { mode },
          ),
          mode,
          1.6,
        ),
      )
      const pair = record.workingSets?.[s]
      if (pair !== undefined && steps > 1)
        layers.push(
          lines(
            [
              {
                x: pair.map((i) => rows[i]?.[a] ?? 0),
                y: pair.map((i) => rows[i]?.[b] ?? 0),
                name: '__pair',
                colour: 'accent',
                width: 2.5,
                markers: true,
              },
            ],
            { mode },
          ),
        )
      const names = surface?.features ?? record.rows?.features
      panels['rows'] = {
        layers,
        options: {
          title: `${support.length} support vectors`,
          x: {
            label: names?.[0] ?? `x${a}`,
            ...(surface === undefined ? {} : { limits: [surface.x[0] ?? 0, surface.x[surface.x.length - 1] ?? 1] }),
          },
          y: {
            label: names?.[1] ?? `x${b}`,
            ...(surface === undefined ? {} : { limits: [surface.y[0] ?? 0, surface.y[surface.y.length - 1] ?? 1] }),
          },
        },
      }
    }
    const right = ['margins', 'dual'].filter((id) => panels[id] !== undefined)
    const grid =
      rows !== undefined
        ? gridSpec(right.length, 2, {
            widthRatios: [1.3, 1],
            wspace: 64,
            hspace: 52,
            cells: [
              { id: 'rows', row: 0, col: 0, rowSpan: right.length },
              ...right.map((id, r) => ({ id, row: r, col: 1 })),
            ],
          })
        : gridSpec(right.length, 1, { hspace: 52, cells: right.map((id, r) => ({ id, row: r, col: 0 })) })
    return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
  }
  return {
    count,
    label: (f) => {
      if (steps <= 1) return `${record.support.length} support vectors`
      const s = traceIndex(f)
      const it = recorded?.[Math.min(Math.max(f, 0), count - 1)]?.iteration ?? solverSteps[s] ?? s
      const last = recorded?.[recorded.length - 1]?.iteration ?? solverSteps[solverSteps.length - 1] ?? steps - 1
      return `step ${it}/${last} · ${short(dual?.[s] ?? 0)}`
    },
    option,
  }
}
