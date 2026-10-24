import type { EChartsOption } from 'echarts'
import type { DecisionSurface, MarginRecord } from 'whitebox-engine/exposition'
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
  /** The training rows (their first two columns, or `columns`) and labels. */
  readonly points?: Rows
  readonly labels?: readonly (number | string)[]
  readonly columns?: readonly [number, number]
  /** The model's decision surface, drawn under the rows. */
  readonly surface?: DecisionSurface
  /** The most frames the solver's steps are played in (default 120). */
  readonly maxFrames?: number
}

/**
 * A margin classifier (an SVM) as frames over its solver's steps: the training rows with the support vectors ringed (by
 * the size of their dual coefficients) over the decision surface, the pair SMO updated at the current step joined by a
 * line; the functional margins y·f(x) as a histogram with 0 (the boundary) and 1 (the margin) marked; and the dual
 * objective with the current step. Without solver steps it is one frame.
 *
 * @param record - The record.
 * @param options - Mode, figure, training rows, labels, columns, the decision surface and the frame budget.
 * @returns The frames.
 */
export function marginFrames(record: MarginRecord, options: MarginViewOptions = {}): ViewFrames {
  const rows = options.points
  const fig = viewFigure(options, { aspect: rows === undefined ? 1.6 : 1.75 })
  const mode = fig.mode
  const steps = record.dualObjective?.length ?? 0
  const count = Math.max(1, Math.min(steps, options.maxFrames ?? 120))
  const stepOf = (frame: number): number =>
    steps <= 1 ? 0 : Math.round((Math.min(Math.max(frame, 0), count - 1) * (steps - 1)) / Math.max(1, count - 1))
  const columns = options.columns ?? [0, 1]
  const labels = options.labels === undefined ? undefined : classIndices(options.labels, options.surface?.classes)
  const alpha = record.support.map((_, i) => Math.abs(record.dualCoefficients[i] ?? 0))
  const top = Math.max(1e-12, ...alpha)
  const margins: Panel = {
    layers: [
      histogram(record.margins, { bins: 30, name: '__margins', colour: 'tab0', mode }),
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
  }
  const option = (frame: number): EChartsOption => {
    const s = stepOf(frame)
    const panels: Record<string, Panel> = { margins }
    if (record.dualObjective !== undefined && steps > 1)
      panels['dual'] = {
        layers: [
          lines([{ y: record.dualObjective, name: 'dual objective', colour: 'fgSecondary', width: 1.4 }], { mode }),
          points([{ x: [s], y: [record.dualObjective[s] ?? 0], name: '__now', colour: 'accent', size: 9 }], { mode }),
        ],
        options: { x: { label: 'solver step' }, y: { label: 'dual objective' } },
      }
    if (rows !== undefined) {
      const layers: Layer[] = []
      if (options.surface !== undefined) layers.push(...decisionSurfaceLayers(options.surface, { mode }).layers)
      layers.push(points(classPointSets(rows, columns, labels, options.surface?.classes, { size: 6 }), { mode }))
      const [a, b] = columns
      layers.push(
        outlined(
          points(
            [
              {
                x: record.support.map((i) => rows[i]?.[a] ?? 0),
                y: record.support.map((i) => rows[i]?.[b] ?? 0),
                sizes: alpha.map((v) => v / top),
                sizeRange: [10, 20],
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
      panels['rows'] = {
        layers,
        options: {
          title: `${record.support.length} support vectors`,
          x: { label: options.surface?.features[0] ?? `x${a}` },
          y: { label: options.surface?.features[1] ?? `x${b}` },
          ...(options.surface === undefined
            ? {}
            : {
                x: {
                  label: options.surface.features[0],
                  limits: [options.surface.x[0] ?? 0, options.surface.x[options.surface.x.length - 1] ?? 1],
                },
                y: {
                  label: options.surface.features[1],
                  limits: [options.surface.y[0] ?? 0, options.surface.y[options.surface.y.length - 1] ?? 1],
                },
              }),
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
    label: (f) =>
      steps > 1
        ? `step ${stepOf(f) + 1}/${steps} · ${short(record.dualObjective?.[stepOf(f)] ?? 0)}`
        : `${record.support.length} support vectors`,
    option,
  }
}
