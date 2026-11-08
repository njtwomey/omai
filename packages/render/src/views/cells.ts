/**
 * Player frames for value grids and confusion matrices: grids of values (feature maps, filters, hidden states) as
 * blocky small multiples on one shared colour range, and a classifier's confusion matrix with counts or row shares and
 * the accuracy, each played through training when the record has steps.
 */

import type { EChartsOption } from 'echarts'
import type { ConfusionMatrix, ValueGrids } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { heatmap, matrix } from '../charts/matrices.ts'
import { cartesian } from '../charts/plot.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber, formatPercent } from '../theme/format.ts'
import { short, spanOf, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link valueGridFrames}. */
export type ValueGridViewOptions = ViewOptions

type Grid = readonly (readonly number[])[]

/**
 * The grids across and down of a value-grid record.
 *
 * @param record - The value-grid record.
 * @returns The number of columns and rows of grids.
 */
export function cellsLayout(record: ValueGrids): { readonly columns: number; readonly rows: number } {
  const g = Math.max(1, record.grids.length)
  const columns = Math.min(g, record.columns ?? Math.ceil(Math.sqrt(g)))
  return { columns, rows: Math.ceil(g / columns) }
}

// The colour range of grids: shared over every grid and step, or one per grid over its steps; symmetric when diverging.
function rangesOf(record: ValueGrids): [number, number][] {
  const each = record.grids.map((g, i) => [g.values, ...(record.steps ?? []).map((s) => s.grids[i] ?? [])])
  const span = (grids: readonly Grid[]): [number, number] => {
    const [lo, hi] = spanOf(grids.flatMap((g) => g.flat()))
    if (record.diverging === true) {
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1
      return [-m, m]
    }
    return hi > lo ? [lo, hi] : [lo - 0.5, hi + 0.5]
  }
  if (record.scale === 'each') return each.map(span)
  const shared = span(each.flat())
  return record.grids.map(() => shared)
}

/**
 * Value grids as frames for a player: every grid as blocky cells (one value one cell, row 0 at the top; never
 * smoothed), in small multiples titled by name, on a shared colour range with one colour bar (or each grid on its own
 * range), diverging about zero when the record asks; with steps, the grids at each recorded step of training, the
 * colour range held over all of them so they visibly form.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The frames.
 * @example
 * ```ts
 * import { valueGrids } from 'omai-engine/exposition'
 * import { valueGridFrames } from 'omai-render/views'
 * const r = valueGrids({ grids: [{ name: 'a', values: [[0, 1], [1, 0]] }, { name: 'b', values: [[1, 1], [0, 0]] }] })
 * valueGridFrames(r).count // => 1
 * ```
 */
export function valueGridFrames(record: ValueGrids, options: ValueGridViewOptions = {}): ViewFrames {
  const { columns, rows } = cellsLayout(record)
  const first = record.grids[0]?.values ?? []
  const h = Math.max(1, first.length)
  const w = Math.max(1, first[0]?.length ?? 1)
  const single = record.grids.length === 1
  const aspect = Math.min(2.4, Math.max(0.9, (columns * w) / (rows * h) + (single ? 0.3 : 0.15)))
  const fig = viewFigure(options, { aspect })
  const mode = fig.mode
  const ranges = rangesOf(record)
  const steps = record.steps ?? []
  const count = Math.max(1, steps.length)
  const shared = record.scale !== 'each'
  const axes = record.axes
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const step = steps[t]
    const panels: Record<string, Panel> = {}
    for (const [i, g] of record.grids.entries()) {
      const values = step?.grids[i] ?? g.values
      const n = values.length
      // Heat maps put row 0 at the bottom: reverse, and label the rows top-down.
      const bar = single || (shared ? i === record.grids.length - 1 : false)
      const layer = heatmap(
        [...values].reverse().map((r) => [...r]),
        {
          smooth: 'cells',
          range: ranges[i] ?? [0, 1],
          ...(record.diverging === true ? { centre: 0 } : {}),
          colourBar: bar,
          ...(record.valueName === undefined ? {} : { label: record.valueName }),
          mode,
        },
      )
      const labelled = single && axes !== undefined
      panels[`g${i}`] = {
        layers: [layer],
        options: {
          ...(single ? {} : { title: g.name }),
          x: labelled ? { label: axes[1] } : { show: false, splitLine: false },
          y: labelled
            ? {
                label: axes[0],
                format: (v: number) => (Number.isInteger(v) ? String(n - 1 - v) : ''),
              }
            : { show: false, splitLine: false },
          ...(labelled ? {} : { aspect: 'equal' as const }),
        },
      }
    }
    const grid = gridSpec(rows, columns, {
      wspace: single ? 0 : 10,
      hspace: single ? 0 : 26,
      cells: record.grids.map((_, i) => ({ id: `g${i}`, row: Math.floor(i / columns), col: i % columns })),
    })
    return subplots(grid, panels, { figure: fig, legend: false, tooltip: false })
  }
  const stepName = record.stepName ?? 'step'
  return {
    count,
    label: (f) => {
      const s = steps[Math.min(Math.max(f, 0), count - 1)]
      if (s === undefined)
        return record.grids.length === 1 ? (record.grids[0]?.name ?? '') : `${record.grids.length} grids`
      return `${stepName} ${s.iteration}${s.objective === undefined ? '' : ` · ${short(s.objective)}`}`
    },
    option,
  }
}

/** Options of {@link confusionFrames}. */
export interface ConfusionViewOptions extends ViewOptions {
  /** Show each true class's row as fractions of its rows (recall along the diagonal) instead of counts. */
  readonly normalise?: boolean
}

/**
 * The share of rows on the diagonal of confusion counts.
 *
 * @param counts - Confusion counts, true class by row.
 * @returns The share of the total that lies on the diagonal.
 */
export function accuracyOf(counts: Grid): number {
  let right = 0
  let all = 0
  for (const [i, row] of counts.entries())
    for (const [j, v] of row.entries()) {
      all += v
      if (i === j) right += v
    }
  return all > 0 ? right / all : Number.NaN
}

/**
 * A confusion matrix as frames for a player: true classes down, predicted across, every cell's count (or row share)
 * written in it, the accuracy in the title; with steps, the matrix at each recorded step of training on one colour
 * range, so the diagonal fills in.
 *
 * @param record - The record.
 * @param options - Mode, figure and row normalisation.
 * @returns The frames.
 * @example
 * ```ts
 * import { confusionMatrix } from 'omai-engine/exposition'
 * import { confusionFrames } from 'omai-render/views'
 * confusionFrames(confusionMatrix({ classes: ['a', 'b'], counts: [[3, 1], [0, 4]] })).label(0) // => 'accuracy 87.5%'
 * ```
 */
export function confusionFrames(record: ConfusionMatrix, options: ConfusionViewOptions = {}): ViewFrames {
  const k = record.classes.length
  const fig = viewFigure(options, { aspect: k > 6 ? 1.15 : 1.25 })
  const mode = fig.mode
  const steps = record.steps ?? []
  const count = Math.max(1, steps.length)
  const normalise = options.normalise === true
  const countsAt = (f: number): Grid => steps[Math.min(Math.max(f, 0), count - 1)]?.counts ?? record.counts
  const shown = (counts: Grid): number[][] =>
    normalise
      ? counts.map((row) => {
          const s = row.reduce((a, b) => a + b, 0)
          return row.map((v) => (s > 0 ? v / s : 0))
        })
      : counts.map((r) => [...r])
  const [, top] = spanOf([record.counts, ...steps.map((s) => s.counts)].flatMap((c) => shown(c).flat()))
  const option = (frame: number): EChartsOption => {
    const counts = countsAt(frame)
    return cartesian(
      [
        matrix(shown(counts), {
          rowLabels: record.classes,
          colLabels: record.classes,
          ramp: 'blues',
          range: [0, normalise ? 1 : Math.max(1, top)],
          annotate: k <= 20,
          format: normalise ? (v) => formatNumber(v, { digits: 2 }) : (v) => String(Math.round(v)),
          colourBar: false,
          mode,
        }),
      ],
      {
        figure: fig,
        mode,
        title: `accuracy ${formatPercent(accuracyOf(counts), 1)}${record.split === undefined ? '' : ` on the ${record.split} rows`}`,
        x: { label: 'predicted' },
        y: { label: 'true' },
        legend: false,
      },
    )
  }
  const stepName = record.stepName ?? 'step'
  return {
    count,
    label: (f) => {
      const s = steps[Math.min(Math.max(f, 0), count - 1)]
      const acc = `accuracy ${formatPercent(accuracyOf(countsAt(f)), 1)}`
      return s === undefined ? acc : `${stepName} ${s.iteration} · ${acc}`
    },
    option,
  }
}
