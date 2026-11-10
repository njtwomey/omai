/**
 * The views of a `PosteriorComparison` record, the charts that show a benchmark campaign: for each quantity the oracle's
 * density laid under each method's (a kernel density of the method's quantiles, or the density of its normal), the
 * ECDFs over one another, and the difference of each method's ECDF from the oracle's CDF, where the Kolmogorov
 * distance is the largest gap and a method too narrow, too wide or shifted has its own shape. The table of the
 * record's agreement measures is `comparisonTable`, drawn by the marked-table view.
 */

import type { EChartsOption } from 'echarts'
import { OmaiError } from 'omai-engine/base'
import { normal } from 'omai-engine/dists'
import type { ComparedMethod, ComparedQuantity, PosteriorComparison } from 'omai-engine/exposition'
import { toFlat } from 'omai-engine/tensor'
import { type Panel, subplots } from '../charts/compose.ts'
import { density, ecdfCurve, kde } from '../charts/distributions.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { ariaOf, panelColumns, panelGrid, type ViewOptions, viewFigure } from './_view.ts'

/** The charts of a comparison. */
export type ComparisonChart = 'density' | 'ecdf' | 'difference'

/** Options of {@link posteriorComparisonView}. */
export interface ComparisonViewOptions extends ViewOptions {
  /** `density`, `ecdf`, or `difference` of each ECDF from the oracle's CDF (default `density`). */
  readonly chart?: ComparisonChart
  /** The quantities drawn, by name (default the first `max`). */
  readonly quantities?: readonly string[]
  /** The most quantities drawn when none are named (default 6). */
  readonly max?: number
}

/** The colour of a method: the oracle is the foreground, the methods take the tab colours in order. */
const colourOf = (record: PosteriorComparison, method: string): string =>
  `tab${Math.max(0, record.methods.indexOf(method)) % 10}`

/** The density of an oracle on its grid: the slope of its CDF (central differences, one-sided at the ends). */
function slope(x: readonly number[], F: readonly number[]): number[] {
  return x.map((_, i) => {
    const a = Math.max(0, i - 1)
    const b = Math.min(x.length - 1, i + 1)
    return ((F[b] ?? 0) - (F[a] ?? 0)) / Math.max(1e-300, (x[b] ?? 0) - (x[a] ?? 0))
  })
}

/** The span a quantity's charts cover: the oracle's grid, else the methods' quantiles. */
function spanOf(q: ComparedQuantity): [number, number] {
  if (q.truth !== undefined) return [q.truth.x[0] as number, q.truth.x.at(-1) as number]
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const m of q.methods) {
    const edge = m.normal === undefined ? (m.quantiles ?? []) : [m.mean - 4 * m.sd, m.mean + 4 * m.sd]
    for (const v of edge) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return [lo, hi]
}

function densityLayer(m: ComparedMethod, range: [number, number], colour: string, mode: ColourMode): Layer {
  return m.normal === undefined
    ? kde(m.quantiles ?? [], { mode, name: m.name, colour, range, n: 160 })
    : density(normal(m.normal.mean, Math.max(m.normal.sd, 1e-12)), { mode, name: m.name, colour, range, n: 160 })
}

/** A method's ECDF at the points: the share of its quantiles at or below each, or its normal's CDF. */
function ecdfAt(m: ComparedMethod, x: readonly number[]): number[] {
  if (m.normal !== undefined) return toFlat(normal(m.normal.mean, Math.max(m.normal.sd, 1e-12)).cdf(x))
  const q = m.quantiles ?? []
  let at = 0
  return x.map((v) => {
    while (at < q.length && (q[at] as number) <= v) at++
    return at / Math.max(1, q.length)
  })
}

/**
 * Posterior comparison charts: for each quantity the oracle in the foreground colour (a filled density, a CDF) and each
 * method in a colour of its own over it. As `density`, a method that is too narrow is a taller bell, too wide a
 * flatter one, biased a shifted one; as `ecdf` the curves over one another; as `difference` each method's ECDF minus the
 * oracle's CDF, where the largest gap from zero is the method's Kolmogorov distance and the shape reads the error (a
 * dip then a rise is too narrow, one sign throughout biased).
 *
 * @param record - The comparison.
 * @param options - The chart, quantities, mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` when no quantity can be drawn: none named is in the record, or the difference is asked of quantities without an oracle.
 * @index function Methods' posteriors over the oracle's as densities, ECDFs or ECDF differences: `posteriorComparisonView(posteriorComparison(methods, { truth }))`.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { posteriorComparisonView } from 'omai-render/views'
 * posteriorComparisonView(expositionSamples.posteriorComparison(), { chart: 'difference' }).grid.length // => 1
 * ```
 */
export function posteriorComparisonView(
  record: PosteriorComparison,
  options: ComparisonViewOptions = {},
): EChartsOption {
  const chart = options.chart ?? 'density'
  const named =
    options.quantities === undefined
      ? record.quantities.slice(0, options.max ?? 6)
      : options.quantities.flatMap((n) => record.quantities.filter((q) => q.name === n))
  // The difference from the oracle needs one: a quantity without it has no panel.
  const qs = named.filter((q) => chart !== 'difference' || q.truth !== undefined)
  if (qs.length === 0)
    throw new OmaiError('value', 'posteriorComparisonView: no quantity to draw (the difference needs an oracle)')
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.3, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const span = spanOf(q)
    const layers: Layer[] = []
    if (chart === 'density') {
      if (q.truth !== undefined)
        layers.push(
          lines(
            [
              {
                x: q.truth.x,
                y: slope(q.truth.x, q.truth.cdf),
                name: record.oracle,
                colour: 'fg',
                width: 2.2,
                area: 0.12,
              },
            ],
            { mode },
          ),
        )
      for (const m of q.methods) layers.push(densityLayer(m, span, colourOf(record, m.name), mode))
    } else if (chart === 'ecdf') {
      if (q.truth !== undefined)
        layers.push(
          lines([{ x: q.truth.x, y: [...q.truth.cdf], name: record.oracle, colour: 'fg', width: 2.2 }], { mode }),
        )
      for (const m of q.methods)
        layers.push(
          m.normal === undefined
            ? ecdfCurve(m.quantiles ?? [], { mode, name: m.name, colour: colourOf(record, m.name) })
            : lines(
                [
                  {
                    x: spanGrid(span),
                    y: ecdfAt(m, spanGrid(span)),
                    name: m.name,
                    colour: colourOf(record, m.name),
                    width: 1.6,
                  },
                ],
                { mode },
              ),
        )
    } else {
      const t = q.truth as NonNullable<typeof q.truth>
      layers.push(referenceLines({ y: [0] }, { mode, colour: 'muted', dash: 'dashed' }))
      for (const m of q.methods)
        layers.push(
          lines(
            [
              {
                x: t.x,
                y: ecdfAt(m, t.x).map((v, j) => v - (t.cdf[j] ?? 0)),
                name: m.name,
                colour: colourOf(record, m.name),
                width: 1.6,
              },
            ],
            { mode },
          ),
        )
    }
    panels[`q${i}`] = {
      layers,
      options: {
        title: q.name,
        ...(i >= qs.length - cols ? { x: { label: q.name } } : {}),
        ...(chart === 'difference' ? { y: { label: 'ECDF minus oracle CDF' } } : {}),
        ...(chart === 'ecdf' ? { y: { limits: [0, 1] } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: true, tooltip: 'item' }),
    ...ariaOf(
      `${chart === 'difference' ? 'ECDF differences' : chart === 'ecdf' ? 'ECDFs' : 'Densities'} of ${record.methods.join(', ')} against ${record.oracle} for ${qs.map((q) => q.name).join(', ')}`,
    ),
  }
}

/** An even grid over a span, for a curve with no oracle grid to follow. */
function spanGrid([lo, hi]: [number, number]): number[] {
  return Array.from({ length: 160 }, (_, i) => lo + ((hi - lo) * i) / 159)
}
