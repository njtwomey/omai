import type { EChartsOption } from 'echarts'
import type { DecisionSurface } from 'whitebox-engine/exposition'
import { contours, decisionRegions } from '../charts/fields.ts'
import { lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { type AxisSpec, cartesian, type Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import type { DataAspect } from '../figure/aspect.ts'
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
  if (surface.probabilities !== undefined || surface.labels !== undefined) {
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
