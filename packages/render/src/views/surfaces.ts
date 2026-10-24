import type { EChartsOption } from 'echarts'
import type { DecisionSurface, HyperparameterSurface, PosteriorGrid } from 'whitebox-engine/exposition'
import { contours, decisionRegions } from '../charts/fields.ts'
import { band, lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { type AxisSpec, cartesian, type Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import type { DataAspect } from '../figure/aspect.ts'
import { formatNumber } from '../theme/format.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { outlined, type ViewOptions, viewFigure, viewMode } from './_view.ts'

/** Rows of input data: `rows[i]` is row i's features. */
export type Rows = readonly (readonly number[])[]

/** Options of {@link decisionSurfaceView}. */
export interface SurfaceViewOptions extends ViewOptions {
  /** The training rows to draw over the surface (their first two columns, or `columns`). */
  readonly points?: Rows
  /** A label per row: a class index or a class name from the record's `classes`. */
  readonly labels?: readonly (number | string)[]
  /** Which columns of `points` are the surface's two features (default `[0, 1]`). */
  readonly columns?: readonly [number, number]
  /** Probability contours: levels, or `false` for none (default 0.25, 0.5, 0.75 for two classes; none otherwise). */
  readonly contours?: readonly number[] | false
  /** The data aspect (default `auto`; `equal` when the features share units). */
  readonly aspect?: DataAspect
}

/** The class index of each label: numbers as given, names looked up in the classes (unknown names give -1). */
export function classIndices(labels: readonly (number | string)[], classes: readonly string[] | undefined): number[] {
  return labels.map((l) => (typeof l === 'number' ? l : (classes?.indexOf(l) ?? -1)))
}

/**
 * Rows coloured by class as point sets, one per class in class order (so class k takes tab10 slot k, as its region
 * does), with unlabelled rows muted.
 *
 * @param rows - The rows.
 * @param columns - The two columns drawn.
 * @param labels - A class index per row (or none).
 * @param names - Class names.
 * @returns The point sets.
 */
export function classPointSets(
  rows: Rows,
  columns: readonly [number, number],
  labels: readonly number[] | undefined,
  names: readonly string[] | undefined,
  style: Partial<PointSet> = {},
): PointSet[] {
  const [a, b] = columns
  const x = rows.map((r) => r[a] ?? 0)
  const y = rows.map((r) => r[b] ?? 0)
  if (labels === undefined) return [{ x, y, name: 'data', colour: 'fg', size: 6, ...style }]
  const k = Math.max(-1, ...labels) + 1
  const sets: PointSet[] = []
  for (let c = -1; c < k; c++) {
    const idx = labels.flatMap((l, i) => (l === c ? [i] : []))
    if (idx.length === 0) continue
    sets.push({
      x: idx.map((i) => x[i] ?? 0),
      y: idx.map((i) => y[i] ?? 0),
      name: c < 0 ? 'unlabelled' : (names?.[c] ?? `class ${c}`),
      colour: c < 0 ? 'muted' : `tab${c % 10}`,
      size: 7,
      opacity: 1,
      symbol: ['circle', 'rect', 'triangle', 'diamond'][Math.max(c, 0) % 4] ?? 'circle',
      ...style,
    })
  }
  return sets
}

function oneHot(labels: readonly (readonly number[])[], k: number): number[][][] {
  return Array.from({ length: k }, (_, c) => labels.map((row) => row.map((v) => (v === c ? 1 : 0))))
}

/**
 * The layers of a decision surface: smooth class regions (faded where the model is unsure) with their boundaries and
 * probability contours, or a smooth field with contours for regressors and densities, or, for one feature, the class
 * probabilities (or values) as curves; then the training rows coloured by class.
 *
 * @param surface - The record.
 * @param options - Mode, training rows and labels, contours.
 * @returns The layers and the axes.
 */
export function decisionSurfaceLayers(
  surface: DecisionSurface,
  options: SurfaceViewOptions = {},
): { readonly layers: Layer[]; readonly x: AxisSpec; readonly y: AxisSpec } {
  const mode: ColourMode = viewMode(options)
  const { x, y, classes } = surface
  const k = surface.probabilities?.length ?? (surface.labels === undefined ? 0 : Math.max(...surface.labels.flat()) + 1)
  const layers: Layer[] = []
  const labels = options.labels === undefined ? undefined : classIndices(options.labels, classes)
  const xAxis: AxisSpec = { label: surface.features[0] }
  if (y.length < 2) {
    // One feature: the probabilities (or values) as curves over it.
    const curves =
      surface.probabilities !== undefined
        ? surface.probabilities.map((p, c) => ({
            x,
            y: p[0] ?? [],
            name: classes?.[c] ?? `class ${c}`,
            colour: `tab${c % 10}`,
          }))
        : [{ x, y: surface.values?.[0] ?? surface.labels?.[0] ?? [], name: 'prediction', colour: 'tab0' }]
    layers.push(lines(curves, { mode }))
    if (options.points !== undefined) {
      const col = options.columns?.[0] ?? 0
      const rows = options.points.map((r) => [r[col] ?? 0, 0])
      layers.push(points(classPointSets(rows, [0, 1], labels, classes, { size: 6, opacity: 0.5 }), { mode }))
    }
    return {
      layers,
      x: xAxis,
      y: { label: surface.probabilities !== undefined ? 'probability' : 'prediction' },
    }
  }
  const scores = surface.values
  if (scores !== undefined && surface.probabilities === undefined && surface.classes?.length === 2) {
    // A binary decision function (an SVM's f(x)): the regions are its sign, drawn smoothly from the scores, with the
    // margins f = ±1 dashed.
    layers.push(
      decisionRegions([scores.map((r) => r.map((v) => -v)), scores], { x, y, mode, confidence: false, opacity: 0.5 }),
    )
    const levels = options.contours ?? [-1, 1]
    if (levels !== false)
      layers.push(
        contours(scores, { x, y, levels, colour: 'fg', dash: 'dashed', width: 1, opacity: 0.7, labels: false, mode }),
      )
  } else if (surface.probabilities !== undefined || surface.labels !== undefined) {
    const fields = surface.probabilities ?? oneHot(surface.labels ?? [], k)
    layers.push(decisionRegions(fields, { x, y, mode, confidence: surface.probabilities !== undefined, opacity: 0.5 }))
    const levels = options.contours ?? (surface.probabilities !== undefined && k === 2 ? [0.25, 0.5, 0.75] : false)
    if (levels !== false && surface.probabilities?.[1] !== undefined)
      layers.push(
        contours(surface.probabilities[1], {
          x,
          y,
          levels: levels.filter((l) => l !== 0.5),
          colour: 'fg',
          dash: 'dashed',
          width: 0.8,
          opacity: 0.6,
          mode,
        }),
      )
  } else if (surface.values !== undefined) {
    layers.push(heatmap(surface.values, { x, y, mode, colourBar: true, label: 'prediction' }))
    layers.push(contours(surface.values, { x, y, levels: 6, colour: 'fg', width: 0.7, opacity: 0.55, mode }))
  }
  if (options.points !== undefined)
    layers.push(
      outlined(points(classPointSets(options.points, options.columns ?? [0, 1], labels, classes), { mode }), mode),
    )
  return { layers, x: xAxis, y: { label: surface.features[1] } }
}

/**
 * A decision surface as one figure: the model's regions or field over two features, with the training rows.
 *
 * @param surface - The record.
 * @param options - Mode, figure, training rows and labels, contours and aspect.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { decisionSurface } from 'whitebox-engine/exposition'
 * import { decisionSurfaceView } from 'whitebox-render/views'
 * const s = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0, 1], probabilities: [[[0.9, 0.4], [0.6, 0.1]], [[0.1, 0.6], [0.4, 0.9]]] })
 * decisionSurfaceView(s).series.length > 0 // => true
 * ```
 */
export function decisionSurfaceView(surface: DecisionSurface, options: SurfaceViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25 })
  const { layers, x, y } = decisionSurfaceLayers(surface, { ...options, mode: fig.mode })
  const xr: [number, number] = [surface.x[0] ?? 0, surface.x[surface.x.length - 1] ?? 1]
  const yr: [number, number] = [surface.y[0] ?? 0, surface.y[surface.y.length - 1] ?? 1]
  const two = surface.y.length >= 2
  return cartesian(layers, {
    figure: fig,
    mode: fig.mode,
    x: two ? { ...x, limits: xr } : x,
    y: two ? { ...y, limits: yr } : y,
    aspect: options.aspect ?? 'auto',
    tooltip: 'item',
  })
}

/** Options of {@link posteriorView}. */
export interface PosteriorViewOptions extends ViewOptions {
  /** The input's name (default `x`). */
  readonly inputLabel?: string
  readonly outputLabel?: string
  /** Show the record's posterior samples (default true). */
  readonly samples?: boolean
}

/**
 * A predictive distribution on a grid (a GP, Bayesian regression, isotonic fits): for one input, the mean with bands of
 * one and two standard deviations, posterior samples and the observations; for two inputs, the mean as coloured
 * points with the observations.
 *
 * @param grid - The record.
 * @param options - Mode, figure, axis labels and whether to draw samples.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { posteriorGrid } from 'whitebox-engine/exposition'
 * import { posteriorView } from 'whitebox-render/views'
 * const g = posteriorGrid({ inputs: [[0], [1], [2]], mean: [0, 1, 0], variance: [1, 0.1, 1], observed: { inputs: [[1]], targets: [1] } })
 * posteriorView(g).series.some((s) => s.name === 'observed') // => true
 * ```
 */
export function posteriorView(grid: PosteriorGrid, options: PosteriorViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const d = grid.inputs[0]?.length ?? 1
  const layers: Layer[] = []
  if (d === 1) {
    const x = grid.inputs.map((r) => r[0] ?? 0)
    const sd = grid.variance.map((v) => Math.sqrt(Math.max(v, 0)))
    for (const k of [2, 1])
      layers.push(
        band(
          [
            {
              x,
              lower: grid.mean.map((m, i) => m - k * (sd[i] ?? 0)),
              upper: grid.mean.map((m, i) => m + k * (sd[i] ?? 0)),
              colour: 'tab0',
              opacity: k === 2 ? 0.14 : 0.24,
              name: `__band${k}`,
            },
          ],
          { mode },
        ),
      )
    if ((options.samples ?? true) && grid.samples !== undefined)
      layers.push(
        lines(
          grid.samples.map((s, i) => ({
            x,
            y: s,
            name: `__sample${i}`,
            colour: `tab${(i + 1) % 10}`,
            width: 1,
            opacity: 0.7,
          })),
          { mode },
        ),
      )
    layers.push(lines([{ x, y: grid.mean, name: 'mean', colour: 'tab0', width: 2.2 }], { mode }))
    if (grid.observed !== undefined)
      layers.push(
        outlined(
          points(
            [
              {
                x: grid.observed.inputs.map((r) => r[0] ?? 0),
                y: grid.observed.targets,
                name: 'observed',
                colour: 'fg',
                size: 7,
              },
            ],
            { mode },
          ),
          mode,
        ),
      )
    return cartesian(layers, {
      figure: fig,
      mode,
      x: { label: options.inputLabel ?? 'x' },
      y: { label: options.outputLabel ?? 'y' },
      legend: false,
      tooltip: 'item',
    })
  }
  layers.push(
    points(
      [
        {
          x: grid.inputs.map((r) => r[0] ?? 0),
          y: grid.inputs.map((r) => r[1] ?? 0),
          values: [...grid.mean],
          name: '__mean',
          size: 9,
          symbol: 'rect',
          opacity: 0.9,
          outline: false,
        },
      ],
      { mode },
    ),
  )
  if (grid.observed !== undefined)
    layers.push(
      outlined(
        points(
          [
            {
              x: grid.observed.inputs.map((r) => r[0] ?? 0),
              y: grid.observed.inputs.map((r) => r[1] ?? 0),
              name: 'observed',
              colour: 'fg',
              size: 6,
            },
          ],
          { mode },
        ),
        mode,
      ),
    )
  return cartesian(layers, { figure: fig, mode, x: { label: 'x0' }, y: { label: 'x1' }, legend: false })
}

/** Options of {@link hyperparameterSurfaceView}. */
export interface HyperSurfaceViewOptions extends ViewOptions {
  /** The objective's name for the colour bar (default 'objective'). */
  readonly label?: string
  /** A point to mark (a control's value), in the parameters' own units. */
  readonly selected?: readonly [number, number]
}

/**
 * An objective over two hyperparameters (a log marginal likelihood, a validation score): a smooth field with contours
 * (the colour range is the upper part of the values, so a steep fall-off does not wash the optimum out), log axes where
 * the record asks, and the optimiser's path to its end.
 *
 * @param surface - The record.
 * @param options - Mode, figure, the objective's name and a selected point.
 * @returns The ECharts option.
 */
export function hyperparameterSurfaceView(
  surface: HyperparameterSurface,
  options: HyperSurfaceViewOptions = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.35 })
  const mode = fig.mode
  const [lx, ly] = surface.logScale ?? [false, false]
  // Log axes are drawn as log10 coordinates with powers of ten as ticks, so the field stays one smooth image.
  const tx = (v: number): number => (lx ? Math.log10(Math.max(v, 1e-300)) : v)
  const ty = (v: number): number => (ly ? Math.log10(Math.max(v, 1e-300)) : v)
  const xs = surface.x.map(tx)
  const ys = surface.y.map(ty)
  const finite = surface.values
    .flat()
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  const top = finite[finite.length - 1] ?? 1
  const low = finite[Math.floor(0.25 * (finite.length - 1))] ?? 0
  const clipped = surface.values.map((row) => row.map((v) => (Number.isFinite(v) ? Math.max(v, low) : Number.NaN)))
  const layers: Layer[] = [
    heatmap(clipped, { x: xs, y: ys, mode, range: [low, top] }),
    contours(clipped, { x: xs, y: ys, levels: 8, colour: 'fg', width: 0.7, opacity: 0.5, labels: false, mode }),
  ]
  const path = surface.path
  if (path !== undefined && path.length > 0) {
    layers.push(
      lines(
        [
          {
            x: path.map((p) => tx(p[0])),
            y: path.map((p) => ty(p[1])),
            name: 'optimiser',
            colour: 'fg',
            width: 1.6,
            markers: true,
          },
        ],
        {
          mode,
        },
      ),
    )
    const end = path[path.length - 1] ?? [0, 0]
    layers.push(
      outlined(
        points([{ x: [tx(end[0])], y: [ty(end[1])], name: 'optimum', colour: 'accent', size: 12, symbol: 'diamond' }], {
          mode,
        }),
        mode,
        1.5,
      ),
    )
  }
  if (options.selected !== undefined)
    layers.push(
      outlined(
        points(
          [{ x: [tx(options.selected[0])], y: [ty(options.selected[1])], name: 'selected', colour: 'tab3', size: 11 }],
          { mode },
        ),
        mode,
        1.5,
      ),
    )
  const power = (v: number): string => formatNumber(10 ** v, { digits: 2 })
  return cartesian(layers, {
    figure: fig,
    mode,
    title: options.label ?? 'objective',
    x: { label: surface.parameters[0], limits: [xs[0] ?? 0, xs[xs.length - 1] ?? 1], ...(lx ? { format: power } : {}) },
    y: { label: surface.parameters[1], limits: [ys[0] ?? 0, ys[ys.length - 1] ?? 1], ...(ly ? { format: power } : {}) },
    legend: false,
    tooltip: 'item',
  })
}
