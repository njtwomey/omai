import type { EChartsOption } from 'echarts'
import type { ComponentsRecord } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { lines } from '../charts/lines.ts'
import { matrix } from '../charts/matrices.ts'
import type { Layer } from '../charts/plot.ts'
import { cartesian } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatPercent } from '../theme/format.ts'
import { type ViewOptions, viewFigure } from './_view.ts'
import { classIndices, classPointSets } from './surfaces.ts'

/** Options of {@link componentsView}. */
export interface ComponentsViewOptions extends ViewOptions {
  /** A label per row, to colour the scores (classes, groups). */
  readonly labels?: readonly (number | string)[]
  /** The two components of the biplot (default the first two). */
  readonly pair?: readonly [number, number]
  /** The components shown in the loadings and the scree (default up to 10). */
  readonly maxComponents?: number
  /** The most features drawn as loadings and biplot arrows, the largest first (default 12). */
  readonly maxFeatures?: number
}

/** The features worth drawing: the `max` with the largest loadings over the components shown, in feature order. */
function topFeatures(record: ComponentsRecord, components: readonly number[], max: number): number[] {
  const norm = record.features.map((_, f) => Math.hypot(...components.map((k) => record.components[k]?.[f] ?? 0)))
  if (norm.length <= max) return norm.map((_, f) => f)
  return norm
    .map((v, f) => [v, f] as const)
    .sort((p, q) => q[0] - p[0])
    .slice(0, max)
    .map(([, f]) => f)
    .sort((p, q) => p - q)
}

/**
 * The scree of a decomposition: each component's share of the variance as a bar and the cumulative share as a line
 * (or, without ratios, the variances; or the reconstruction error per rank).
 *
 * @param record - The record.
 * @param options - Mode and the number of components shown.
 * @returns The panel, or `undefined` when the record has nothing to plot.
 */
export function screePanel(record: ComponentsRecord, options: ComponentsViewOptions = {}): Panel | undefined {
  const mode = options.mode ?? 'light'
  const k = Math.min(record.components.length, options.maxComponents ?? 10)
  const names = Array.from({ length: k }, (_, i) => `${i + 1}`)
  const ratio = record.explainedVarianceRatio?.slice(0, k)
  if (ratio !== undefined) {
    let acc = 0
    const cumulative = ratio.map((r) => (acc += r))
    const layers: Layer[] = [
      bars(names, [{ name: 'share', values: ratio, colour: 'tab0' }], { mode }),
      lines([{ x: names.map((_, i) => i), y: cumulative, name: 'cumulative', colour: 'fgSecondary', markers: true }], {
        mode,
      }),
    ]
    return {
      layers,
      options: { x: { label: 'component' }, y: { label: 'variance explained', format: 'percent', limits: [0, 1] } },
    }
  }
  const variance = record.explainedVariance?.slice(0, k)
  if (variance !== undefined)
    return {
      layers: [bars(names, [{ name: 'variance', values: variance, colour: 'tab0' }], { mode })],
      options: { x: { label: 'component' }, y: { label: 'variance' } },
    }
  const error = record.reconstructionError?.slice(0, k)
  if (error !== undefined)
    return {
      layers: [
        lines([{ x: error.map((_, i) => i + 1), y: error, name: 'error', colour: 'tab0', markers: true }], { mode }),
      ],
      options: { x: { label: 'rank' }, y: { label: 'reconstruction error' } },
    }
  return undefined
}

/** A feature name cut to fit an axis (units in brackets dropped first). */
function shortName(name: string, max = 14): string {
  const bare = name.replace(/\s*\([^)]*\)\s*$/, '')
  return bare.length > max ? `${bare.slice(0, max - 1)}…` : bare
}

/**
 * The loadings of a decomposition as a matrix: features down, components across, coloured on a diverging ramp centred
 * on zero, each cell written when there is room.
 *
 * @param record - The record.
 * @param options - Mode and the number of components shown.
 * @returns The panel.
 */
export function loadingsPanel(record: ComponentsRecord, options: ComponentsViewOptions = {}): Panel {
  const mode = options.mode ?? 'light'
  const k = Math.min(record.components.length, options.maxComponents ?? 10)
  const comps = record.components.slice(0, k)
  const keep = topFeatures(
    record,
    comps.map((_, i) => i),
    options.maxFeatures ?? 12,
  )
  const byFeature = keep.map((f) => comps.map((c) => c[f] ?? 0))
  const top = Math.max(1e-12, ...comps.flat().map(Math.abs))
  return {
    layers: [
      matrix(byFeature, {
        rowLabels: keep.map((f) => shortName(record.features[f] ?? '')),
        colLabels: comps.map((_, i) => `${i + 1}`),
        centre: 0,
        range: [-top, top],
        colourBar: false,
        annotate: k <= 4 && keep.length <= 12,
        mode,
      }),
    ],
    options: {
      x: { label: 'component' },
      ...(keep.length < record.features.length ? { title: `top ${keep.length} of ${record.features.length}` } : {}),
    },
  }
}

/**
 * A biplot: the rows in the plane of two components (coloured by label) with each feature's loading as an arrow from
 * the origin, scaled to the scores' spread.
 *
 * @param record - A record with scores.
 * @param options - Mode, labels and the pair of components.
 * @returns The panel, or `undefined` without scores.
 */
export function biplotPanel(record: ComponentsRecord, options: ComponentsViewOptions = {}): Panel | undefined {
  const scores = record.scores
  if (scores === undefined || record.components.length < 1) return undefined
  const mode = options.mode ?? 'light'
  const [a, b] = options.pair ?? [0, Math.min(1, record.components.length - 1)]
  const own = options.labels === undefined ? record.colourBy : undefined
  const labels =
    options.labels !== undefined
      ? classIndices(options.labels, undefined)
      : own?.kind === 'classes'
        ? own.labels
        : undefined
  const values = own?.kind === 'values' ? own.values : undefined
  const names = own?.kind === 'classes' ? own.classes : undefined
  const plane = scores.map((s) => [s[a] ?? 0, s[b] ?? 0])
  // Kernel methods weight training rows, not features: no arrows. Otherwise the longest few.
  const shown = record.space === 'samples' ? [] : topFeatures(record, [a, b], options.maxFeatures ?? 12)
  const spread = Math.max(1e-12, ...plane.flat().map(Math.abs))
  const ca = record.components[a] ?? []
  const cb = record.components[b] ?? []
  const longest = Math.max(1e-12, ...shown.map((f) => Math.hypot(ca[f] ?? 0, cb[f] ?? 0)))
  const scale = (0.85 * spread) / longest
  const lengths = shown.map((f) => Math.hypot(ca[f] ?? 0, cb[f] ?? 0))
  const cutoff = [...lengths].sort((p, q) => q - p)[Math.min(7, lengths.length - 1)] ?? 0
  const arrows = lines(
    shown.map((f) => ({
      x: [0, (ca[f] ?? 0) * scale],
      y: [0, (cb[f] ?? 0) * scale],
      name: `__loading${f}`,
      colour: 'fg',
      width: 1.4,
    })),
    { mode },
  )
  const ends = points(
    [
      {
        x: shown.map((f) => (ca[f] ?? 0) * scale),
        y: shown.map((f) => (cb[f] ?? 0) * scale),
        // Only the longest arrows are named, so names do not pile up at the origin.
        labels: shown.map((f, i) => ((lengths[i] ?? 0) >= cutoff ? shortName(record.features[f] ?? '', 13) : '')),
        name: '__features',
        colour: 'fg',
        size: 5,
        symbol: 'triangle',
      },
    ],
    { mode },
  )
  const ratio = record.explainedVarianceRatio
  const axisName = (i: number): string =>
    `component ${i + 1}${ratio?.[i] === undefined ? '' : ` (${formatPercent(ratio[i] ?? 0, 0)})`}`
  return {
    layers: [
      values === undefined
        ? points(classPointSets(plane, [0, 1], labels, names, { size: 6, opacity: 0.7 }), { mode })
        : points(
            [
              {
                x: plane.map((p) => p[0] ?? 0),
                y: plane.map((p) => p[1] ?? 0),
                values: [...values],
                name: '__rows',
                size: 6,
              },
            ],
            { mode },
          ),
      ...(shown.length === 0 ? [] : [arrows, ends]),
    ],
    options: { x: { label: axisName(a) }, y: { label: axisName(b) }, aspect: 'equal' },
  }
}

/**
 * A linear decomposition (PCA, factor analysis, ICA, NMF, PLS) as one figure: the biplot of the scores (when the record
 * has them), the scree with the cumulative share of variance, and the loadings.
 *
 * @param record - The record.
 * @param options - Mode, figure, labels, the biplot's pair and the number of components shown.
 * @returns The ECharts option.
 */
export function componentsView(record: ComponentsRecord, options: ComponentsViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: record.scores === undefined ? 1.6 : 1.75 })
  const o = { ...options, mode: fig.mode }
  // Kernel methods' loadings weight training rows: nothing to read off them.
  const panels: Record<string, Panel> = record.space === 'samples' ? {} : { loadings: loadingsPanel(record, o) }
  const scree = screePanel(record, o)
  if (scree !== undefined) panels['scree'] = scree
  const bi = biplotPanel(record, o)
  if (bi !== undefined) panels['biplot'] = bi
  const right = ['scree', 'loadings'].filter((id) => panels[id] !== undefined)
  if (right.length === 0) {
    const only = bi ?? { layers: [], options: {} }
    return cartesian(only.layers, { ...only.options, figure: fig, mode: fig.mode, legend: false, tooltip: 'item' })
  }
  const grid =
    bi !== undefined
      ? gridSpec(right.length, 2, {
          widthRatios: [1.25, 1],
          heightRatios: right.length === 2 ? [1, 1.7] : [1],
          wspace: 112,
          hspace: 48,
          cells: [
            { id: 'biplot', row: 0, col: 0, rowSpan: right.length },
            ...right.map((id, r) => ({ id, row: r, col: 1 })),
          ],
        })
      : gridSpec(1, right.length, {
          wspace: 72,
          cells: right.map((id, c) => ({ id, row: 0, col: c })),
        })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}
