/**
 * Views of tabular records: a coefficient table as a forest plot and as table data, a likelihood table of each
 * feature's contribution to each class's score (naive Bayes, discriminant analysis), a query's nearest neighbours with
 * their distances, and the trials of a hyperparameter search with the best so far.
 */

import type { EChartsOption } from 'echarts'
import { maxOf, minOf } from 'mdml-engine/base'
import type { CoefficientTable, LikelihoodTable, NeighbourRecord, SearchTrials } from 'mdml-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { intervalChart } from '../charts/evaluation.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { matrix } from '../charts/matrices.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber } from '../theme/format.ts'
import type { TableData } from '../trace/tables.ts'
import { outlined, type ViewOptions, viewFigure, wantsLog } from './_view.ts'
import type { Rows } from './surfaces.ts'

/** The interval of each coefficient: the record's, else the estimate ± 1.96 standard errors, else none. */
export function coefficientIntervals(table: CoefficientTable): (readonly [number, number] | undefined)[] {
  return table.names.map((_, i) => {
    const given = table.intervals?.[i]
    if (given !== undefined) return given
    const se = table.standardErrors?.[i]
    const e = table.estimates[i] ?? 0
    return se === undefined ? undefined : ([e - 1.96 * se, e + 1.96 * se] as const)
  })
}

// p-values below double precision's reach print as a bound, not as zero.
function pText(p: number): string {
  return p < 1e-16 ? '<1e-16' : formatNumber(p, { digits: 2, smallExponent: 1e-4 })
}

/** Whether a coefficient's name is the intercept's. */
export function isIntercept(name: string): boolean {
  return /^(const|intercept|\(intercept\)|bias|b0)$/i.test(name.trim())
}

/**
 * A coefficient table as table data: name, estimate, standard error, interval and p-value (the columns the record
 * has), numbers to four significant figures.
 *
 * @param table - The record.
 * @returns The table.
 * @example
 * ```ts
 * import { coefficientTable } from 'mdml-engine/exposition'
 * import { coefficientRows } from 'mdml-render/views'
 * coefficientRows(coefficientTable({ names: ['x'], estimates: [2], standardErrors: [0.1] })).columns // => ['coefficient', 'estimate', 'std. error', 'interval']
 * ```
 */
export function coefficientRows(table: CoefficientTable): TableData {
  const f = (v: number): string => formatNumber(v, { digits: 4 })
  const intervals = coefficientIntervals(table)
  const level = table.level === undefined ? '' : ` (${Math.round(table.level * 100)}%)`
  const columns = [
    'coefficient',
    'estimate',
    ...(table.standardErrors === undefined ? [] : ['std. error']),
    ...(intervals.some((i) => i !== undefined) ? [`interval${level}`] : []),
    ...(table.pValues === undefined ? [] : ['p-value']),
  ]
  const rows = table.names.map((name, i) => {
    const iv = intervals[i]
    return [
      name,
      f(table.estimates[i] ?? 0),
      ...(table.standardErrors === undefined ? [] : [f(table.standardErrors[i] ?? 0)]),
      ...(intervals.some((x) => x !== undefined) ? [iv === undefined ? '' : `[${f(iv[0])}, ${f(iv[1])}]`] : []),
      ...(table.pValues === undefined ? [] : [pText(table.pValues[i] ?? 0)]),
    ]
  })
  return { columns, rows }
}

/**
 * A coefficient table as a forest plot: each estimate a point with its interval as a bar, zero dashed (the intercept
 * left out unless asked for).
 *
 * @param table - The record.
 * @param options - Mode, figure and whether to plot the intercept.
 * @returns The ECharts option.
 */
export function coefficientView(
  table: CoefficientTable,
  options: ViewOptions & { readonly intercept?: boolean } = {},
): EChartsOption {
  // The intercept usually sits on another scale from the slopes: it stays in the table, out of the plot.
  const keep = table.names.map((n, i) => (options.intercept === true || !isIntercept(n) ? i : -1)).filter((i) => i >= 0)
  const traj = table.trajectories
  if (traj !== undefined && traj.at.length > 1) {
    // The estimates as they evolved (an online learner's coefficients over the samples it has seen).
    const fig = viewFigure(options, { aspect: 1.6, padding: { right: 72 } })
    return cartesian(
      [
        referenceLines({ y: [0] }, { mode: fig.mode, colour: 'grid', dash: 'solid' }),
        lines(
          keep.map((j) => ({
            x: traj.at,
            y: traj.values.map((row) => row[j] ?? Number.NaN),
            name: table.names[j] ?? `x${j}`,
            width: 1.6,
          })),
          { mode: fig.mode },
        ),
      ],
      {
        figure: fig,
        mode: fig.mode,
        x: { label: traj.resource, scale: wantsLog(traj.at) ? 'log' : 'linear' },
        y: { label: 'estimate' },
        tooltip: 'item',
      },
    )
  }
  const fig = viewFigure(options, { height: Math.max(160, 64 + 26 * keep.length), width: 640 })
  const intervals = coefficientIntervals(table)
  return intervalChart(
    keep.map((i) => {
      const iv = intervals[i]
      return {
        name: table.names[i] ?? '',
        value: { value: table.estimates[i] ?? 0, ...(iv === undefined ? {} : { interval: iv }) },
      }
    }),
    {
      figure: { ...fig, padding: { ...fig.padding, left: 110 } },
      mode: fig.mode,
      reference: 0,
      x: { label: 'estimate' },
    },
  )
}

/**
 * How each feature contributed to each class's score (naive Bayes, discriminant analysis): the per-class, per-feature
 * log-likelihood contributions with the log prior as a matrix, and the posterior over the classes (a softmax of the
 * rows' totals), the winner highlighted.
 *
 * @param table - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 */
export function likelihoodView(table: LikelihoodTable, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.9 })
  const mode = fig.mode
  const cells = table.classes.map((_, c) => [table.logPriors[c] ?? 0, ...(table.contributions[c] ?? [])])
  const totals = cells.map((row) => row.reduce((s, v) => s + v, 0))
  const best = totals.indexOf(maxOf(totals))
  // The posterior over the classes: a softmax of the totals.
  const top = maxOf(totals)
  const weights = totals.map((v) => Math.exp(v - top))
  const sum = weights.reduce((a, b) => a + b, 0)
  const posterior = weights.map((w) => w / sum)
  // Colours span the twelve nats below the largest term; anything lower is as dark as it gets.
  const hi = maxOf(cells.flat())
  const bare = (f: string): string => {
    const b = f.replace(/\s*\([^)]*\)\s*$/, '')
    return b.length > 12 ? `${b.slice(0, 11)}…` : b
  }
  const panels: Record<string, Panel> = {
    table: {
      layers: [
        matrix(cells, {
          rowLabels: [...table.classes],
          colLabels: ['log prior', ...table.features.map(bare)],
          ramp: 'viridis',
          range: [hi - 12, hi],
          colourBar: false,
          annotate: cells.length * (table.features.length + 1) <= 60,
          mode,
        }),
      ],
      options: {
        x: { label: 'log-likelihood contribution' },
        ...(table.scope === undefined
          ? {}
          : {
              title:
                table.scope.kind === 'average'
                  ? `average over each class's rows (${table.scope.rows} rows)`
                  : `one query${table.scope.row === undefined ? '' : `, row ${table.scope.row}`}`,
            }),
      },
    },
    totals: {
      layers: [
        bars(
          table.classes.map((c, k) => `${c} ${formatNumber(posterior[k] ?? 0, { digits: 2 })}`),
          [
            { name: '__other', values: posterior.map((v, c) => (c === best ? 0 : v)), colour: 'muted' },
            { name: '__best', values: posterior.map((v, c) => (c === best ? v : 0)), colour: 'accent' },
          ],
          { mode, stacked: true, horizontal: true },
        ),
      ],
      options: {
        // For class averages the softmax compares the classes' typical rows; it predicts nothing.
        title:
          table.scope?.kind === 'average'
            ? `largest: ${table.classes[best] ?? best}`
            : `predicts ${table.classes[best] ?? best}`,
        x: { label: 'posterior probability', limits: [0, 1] },
      },
    },
  }
  const grid = gridSpec(1, 2, {
    widthRatios: [3, 1],
    wspace: 96,
    cells: [
      { id: 'table', row: 0, col: 0 },
      { id: 'totals', row: 0, col: 1 },
    ],
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

/** Options of {@link neighbourView}. */
export interface NeighbourViewOptions extends ViewOptions {
  /** The training rows the neighbours index (their first two columns, or `columns`). */
  readonly points?: Rows
  readonly columns?: readonly [number, number]
  /** The query rows, when they are not the training rows themselves (default: query q is training row q). */
  readonly queries?: Rows
  /** The query shown (default 0). */
  readonly query?: number
  readonly labels?: readonly (number | string)[]
}

/**
 * The neighbours of one query: given the rows, the query joined to each of its neighbours with the radius of the
 * farthest drawn (in the metric's own units when it is Euclidean), beside the neighbours' distances in order.
 *
 * @param record - The record.
 * @param options - Mode, figure, rows, queries, the query shown and labels.
 * @returns The ECharts option.
 */
export function neighbourView(record: NeighbourRecord, options: NeighbourViewOptions = {}): EChartsOption {
  const own = options.points === undefined ? record.rows : undefined
  const rows = options.points ?? own?.points
  const labelsGiven = options.labels ?? own?.labels
  const queriesGiven = options.queries ?? (own === undefined ? undefined : record.queries)
  const fig = viewFigure(options, { aspect: rows === undefined ? 1.6 : 1.75 })
  const mode = fig.mode
  const q = Math.min(Math.max(options.query ?? 0, 0), Math.max(record.indices.length - 1, 0))
  const idx = record.indices[q] ?? []
  const dist = record.distances[q] ?? []
  // Each neighbour's bar in its label's colour, so the vote can be read off the chart.
  const labelOf = (i: number): number => {
    const l = labelsGiven?.[i]
    return l === undefined ? 0 : typeof l === 'number' ? l : [...new Set(labelsGiven)].indexOf(l)
  }
  const classes = labelsGiven === undefined ? [0] : [...new Set(idx.map(labelOf))].sort((p, r) => p - r)
  const panels: Record<string, Panel> = {
    distances: {
      layers: [
        bars(
          idx.map((_, k) => `${k + 1}`),
          classes.map((c) => ({
            name: `__class${c}`,
            colour: labelsGiven === undefined ? 'tab0' : `tab${c % 10}`,
            values: dist.map((d, k) => (labelOf(idx[k] ?? 0) === c || labelsGiven === undefined ? d : 0)),
          })),
          { mode, stacked: true },
        ),
      ],
      options: {
        title: `query ${q} · ${idx.length} neighbours`,
        x: { label: 'neighbour rank' },
        y: { label: `${record.metric} distance` },
      },
    },
  }
  if (rows !== undefined) {
    const [a, b] = own === undefined ? (options.columns ?? [0, 1]) : [0, 1]
    const query = queriesGiven?.[q] ?? rows[q] ?? []
    const qx = query[a] ?? 0
    const qy = query[b] ?? 0
    const labels = labelsGiven
    const groups = labels?.map((l) => (typeof l === 'number' ? l : [...new Set(labels)].indexOf(l)))
    const layers: Layer[] = [
      points(
        [
          {
            x: rows.map((r) => r[a] ?? 0),
            y: rows.map((r) => r[b] ?? 0),
            ...(groups === undefined ? { colour: 'muted' } : { groups }),
            name: '__rows',
            size: 5,
            opacity: 0.45,
          },
        ],
        { mode },
      ),
      lines(
        [
          {
            x: idx.flatMap((i) => [qx, rows[i]?.[a] ?? 0, Number.NaN]),
            y: idx.flatMap((i) => [qy, rows[i]?.[b] ?? 0, Number.NaN]),
            name: '__spokes',
            colour: 'accent',
            width: 1.2,
          },
        ],
        { mode },
      ),
    ]
    const r = dist[dist.length - 1]
    if (r !== undefined && record.metric === 'euclidean') {
      const circle = Array.from({ length: 97 }, (_, k) => (2 * Math.PI * k) / 96)
      layers.push(
        lines(
          [
            {
              x: circle.map((t) => qx + r * Math.cos(t)),
              y: circle.map((t) => qy + r * Math.sin(t)),
              name: '__radius',
              colour: 'accent',
              dash: 'dashed',
              width: 1,
            },
          ],
          { mode },
        ),
      )
    }
    layers.push(
      outlined(
        points(
          [
            {
              x: idx.map((i) => rows[i]?.[a] ?? 0),
              y: idx.map((i) => rows[i]?.[b] ?? 0),
              name: 'neighbours',
              colour: 'none',
              size: 11,
              outline: false,
            },
          ],
          {
            mode,
          },
        ),
        mode,
        1.4,
      ),
      outlined(
        points([{ x: [qx], y: [qy], name: 'query', colour: 'accent', size: 13, symbol: 'diamond' }], { mode }),
        mode,
        1.5,
      ),
    )
    panels['rows'] = {
      layers,
      options: {
        x: { label: own?.features[0] ?? `x${a}` },
        y: { label: own?.features[1] ?? `x${b}` },
        aspect: 'equal',
      },
    }
  }
  const ids = rows === undefined ? ['distances'] : ['rows', 'distances']
  const grid = gridSpec(1, ids.length, {
    widthRatios: ids.length === 2 ? [1.9, 1] : [1],
    wspace: 64,
    cells: ids.map((id, c) => ({ id, row: 0, col: c })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

/** Options of {@link searchTrialsView}. */
export interface SearchViewOptions extends ViewOptions {
  /** The parameters drawn against the value (default the record's first two numeric ones). */
  readonly parameters?: readonly string[]
  /** The objective's name (default 'objective'). */
  readonly label?: string
}

/**
 * Every trial of a search (grid, random, halving, Bayesian): the value of each trial in order with the best so far as a
 * step line (pruned trials muted, failed ones crossed, the spread of per-fold values as whiskers), beside the value against each of up
 * to two parameters.
 *
 * @param record - The record.
 * @param options - Mode, figure, the parameters drawn and the objective's name.
 * @returns The ECharts option.
 */
export function searchTrialsView(record: SearchTrials, options: SearchViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2 })
  const mode = fig.mode
  const label = options.label ?? record.scoreName ?? 'objective'
  const trials = record.trials
  const better = (a: number, b: number): boolean => (record.higherIsBetter ? a > b : a < b)
  let incumbent = Number.NaN
  const bestSoFar = trials.map((t) => {
    if (t.state === 'complete' && (Number.isNaN(incumbent) || better(t.value, incumbent))) incumbent = t.value
    return incumbent
  })
  const of = (state: 'complete' | 'pruned' | 'failed'): number[] =>
    trials.flatMap((t, i) => (t.state === state ? [i] : []))
  const history: Layer[] = []
  // Per-fold (or per-step) values as a whisker from the lowest to the highest at each trial.
  const xs: number[] = []
  const ys: number[] = []
  trials.forEach((t, i) => {
    if (t.intermediate === undefined || t.intermediate.length < 2) return
    xs.push(i, i, Number.NaN)
    ys.push(minOf(t.intermediate), maxOf(t.intermediate), Number.NaN)
  })
  if (xs.length > 0)
    history.push(lines([{ x: xs, y: ys, name: '__spread', colour: 'muted', width: 1.2, opacity: 0.7 }], { mode }))
  history.push(
    lines(
      [{ x: trials.map((_, i) => i), y: bestSoFar, name: 'best so far', colour: 'accent', step: 'end', width: 1.8 }],
      { mode },
    ),
    points(
      [
        {
          x: of('complete'),
          y: of('complete').map((i) => trials[i]?.value ?? 0),
          name: 'complete',
          colour: 'tab0',
          size: 6,
        },
        {
          x: of('pruned'),
          y: of('pruned').map((i) => trials[i]?.value ?? 0),
          name: 'pruned',
          colour: 'muted',
          size: 5,
        },
        {
          x: of('failed'),
          y: of('failed').map(() => incumbent),
          name: 'failed',
          colour: 'danger',
          size: 7,
          symbol: 'path://M0,0L10,10M10,0L0,10',
        },
      ].filter((s) => s.x.length > 0),
      { mode },
    ),
  )
  const panels: Record<string, Panel> = {
    history: {
      layers: history,
      options: { title: `${record.method}: ${trials.length} trials`, x: { label: 'trial' }, y: { label } },
    },
  }
  const numeric = record.parameters.filter((p) => trials.some((t) => typeof t.params[p] === 'number'))
  const shown = (options.parameters ?? numeric).slice(0, 2)
  for (const p of shown) {
    const done = trials.filter((t) => typeof t.params[p] === 'number')
    const xs = done.map((t) => t.params[p] as number)
    const best = record.best === undefined ? undefined : trials[record.best]
    panels[`p:${p}`] = {
      layers: [
        points(
          [
            {
              x: xs,
              y: done.map((t) => t.value),
              name: '__trials',
              colour: 'tab0',
              size: 6,
              opacity: 0.7,
            },
          ],
          { mode },
        ),
        ...(best !== undefined && typeof best.params[p] === 'number'
          ? [
              outlined(
                points(
                  [
                    {
                      x: [best.params[p] as number],
                      y: [best.value],
                      name: 'best',
                      colour: 'accent',
                      size: 12,
                      symbol: 'diamond',
                    },
                  ],
                  {
                    mode,
                  },
                ),
                mode,
                1.5,
              ),
            ]
          : []),
      ],
      options: { x: { label: p, scale: wantsLog(xs) ? 'log' : 'linear' }, y: { label } },
    }
  }
  const ids = ['history', ...shown.map((p) => `p:${p}`)]
  const grid = gridSpec(1, ids.length, {
    widthRatios: ids.map((id) => (id === 'history' ? 1.6 : 1)),
    wspace: 64,
    cells: ids.map((id, c) => ({ id, row: 0, col: c, ...(c > 0 ? { sharey: 'history' } : {}) })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}
