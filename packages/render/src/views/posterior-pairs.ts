/**
 * The view of a `PosteriorPairs` record, the corner plot of several inference methods over one another: a lower
 * triangle of scatter plots (small translucent points, a colour per method, the axes of a coordinate shared by every
 * panel it is on), on the diagonal each method's marginal density, and in the upper triangle each method's correlation
 * or the ellipse that holds most of its Gaussian approximation. The reference value (the truth) is marked in every
 * panel, divergent transitions in red. The view shows the run up to a cursor (a step): the draws that had arrived, and
 * the densities, correlations and ellipses of the record's frame at it, so playing the cursor along the steps draws the
 * methods converging on the posterior. The pair grid, its edge labels and the even thinning are shared with `pairView`.
 */

import type { EChartsOption } from 'echarts'
import { OmaiError } from 'omai-engine/base'
import { drawsUpTo, type PairsMethod, type PosteriorPairs, pairIndex, pairsFrameAt } from 'omai-engine/exposition'
import type { Paint } from 'omai-engine/scene'
import { type Panel, subplotRects, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { chartFrame, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import type { Rect } from '../figure/figure.ts'
import { resolvePaint, tokens } from '../theme/tokens.ts'
import { ariaOf, pairEdge, pairGrid, short, type ViewOptions, viewFigure } from './_view.ts'

/** What the upper triangle of the pair plot shows: each method's correlation as a number, or its 50% and 90% ellipses. */
export type PairsUpper = 'correlation' | 'ellipses'

/** Options of {@link posteriorPairsView}. */
export interface PairsViewOptions extends ViewOptions {
  /** The step the run is shown at: the draws up to it and the statistics of the frame at it (default the end). */
  readonly cursor?: number
  /** The methods shown, by name (default all); a hidden method keeps its colour. */
  readonly methods?: readonly string[]
  /** The coordinates shown, in the order of the plot (default all of the record's, in its order). */
  readonly coordinates?: readonly string[]
  /** The upper triangle (default `correlation`). */
  readonly upper?: PairsUpper
}

/**
 * The tab colour of the method at an index: the tab10 palette in order without `tab3`, red, which the divergent
 * transitions keep to themselves.
 *
 * @param index - The method's index in the record.
 * @returns The token, `tab0`, `tab1`, `tab2`, `tab4` ...
 * @example
 * ```ts
 * import { pairsMethodColour } from 'omai-render/views'
 * [0, 1, 2, 3, 4].map(pairsMethodColour) // => ['tab0', 'tab1', 'tab2', 'tab4', 'tab5']
 * ```
 */
export function pairsMethodColour(index: number): string {
  const order = [0, 1, 2, 4, 5, 6, 7, 8, 9]
  return `tab${order[index % order.length]}`
}

/** The red of the divergent transitions (tab10's red is not given to a method). */
export const DIVERGENT_COLOUR = 'tab3'

/**
 * The default size of a pair plot of `k` coordinates: square, wider with more panels so each stays readable, with room
 * above for the legend.
 *
 * @param k - The coordinates shown.
 * @returns The figure options.
 * @example
 * ```ts
 * import { pairsFigureOptions } from 'omai-render/views'
 * [pairsFigureOptions(3).width, pairsFigureOptions(6).width] // => [520, 720]
 * ```
 */
export function pairsFigureOptions(k: number): {
  aspect: number
  width: number
  padding: { top: number; right: number }
} {
  return { aspect: 1, width: k <= 3 ? 520 : k <= 5 ? 640 : 720, padding: { top: 30, right: 16 } }
}

/** The points of the ellipse that holds `mass` of a Gaussian with these means, sds and correlation. */
function ellipse(
  mx: number,
  my: number,
  sx: number,
  sy: number,
  r: number,
  mass: number,
): { x: number[]; y: number[] } {
  const radius = Math.sqrt(-2 * Math.log(1 - mass))
  const n = 48
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n
    const u = radius * Math.cos(t)
    const v = radius * Math.sin(t)
    x.push(mx + sx * u)
    y.push(my + sy * (r * u + Math.sqrt(Math.max(0, 1 - r * r)) * v))
  }
  return { x, y }
}

/**
 * The corner plot of several methods' posterior draws at a step of the run. The lower triangle scatters the draws that
 * had arrived by then, a colour per method; the diagonal draws each method's density of the draws so far on a grid
 * shared by every method; the upper triangle writes each method's correlation (or draws its Gaussian ellipses). The axes
 * of a coordinate are the record's, the same at every step, so a cursor played along the run moves the methods and
 * not the frame. The reference value is a cross-hair in each panel and a diamond where the pair's two coordinates have
 * one; the legend names the methods, the divergent transitions and the reference.
 *
 * @param record - The record.
 * @param options - The cursor, the methods and coordinates shown, the upper triangle, mode and figure.
 * @returns The ECharts option, one grid per panel.
 * @throws OmaiError `value` for fewer than two coordinates shown, or none of the methods named.
 * @index function Corner plot of methods' posterior draws at a step: `posteriorPairsView(r)`.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { posteriorPairsView } from 'omai-render/views'
 * const r = expositionSamples.posteriorPairs()
 * [posteriorPairsView(r).grid.length, posteriorPairsView(r, { coordinates: ['a', 'b'] }).grid.length] // => [4, 4]
 * ```
 */
export function posteriorPairsView(record: PosteriorPairs, options: PairsViewOptions = {}): EChartsOption {
  const names = record.coordinates.map((c) => c.name)
  const chosen =
    options.coordinates === undefined
      ? names.map((_, i) => i)
      : options.coordinates.flatMap((n) => (names.includes(n) ? [names.indexOf(n)] : []))
  if (chosen.length < 2) throw new OmaiError('value', 'posteriorPairsView: needs at least two coordinates')
  const shown = record.methods.flatMap((m, i) =>
    options.methods === undefined || options.methods.includes(m.name) ? [{ m, i }] : [],
  )
  if (shown.length === 0) throw new OmaiError('value', 'posteriorPairsView: none of the methods is in the record')
  const k = chosen.length
  const K = record.coordinates.length
  const cursor = options.cursor ?? record.stepRange[1]
  const frame = pairsFrameAt(record, cursor)
  const upper = options.upper ?? 'correlation'
  const fig = viewFigure(options, pairsFigureOptions(k))
  const mode = fig.mode
  const t = tokens(mode)
  const grid = pairGrid(k, true)
  const labels = chosen.map((j) => names[j] as string)
  const coordinate = (r: number) => record.coordinates[chosen[r] as number]
  const colour = (i: number): string => pairsMethodColour(i)
  // The draws that had arrived at the cursor: the points are ascending in step.
  const seen = (m: PairsMethod): number => drawsUpTo(m.points.steps, cursor)
  const seenDivergent = (m: PairsMethod): number => drawsUpTo(m.divergent.steps, cursor)
  const column = (m: PairsMethod, c: number, n: number): number[] =>
    (m.points.values[c] as readonly number[]).slice(0, n)

  // The height of the densities, the same at every step: the tallest of the frames with enough draws to be a density.
  const ceiling = (r: number): number => {
    const j = chosen[r] as number
    const G = (coordinate(r)?.grid.length ?? 1) as number
    let top = 0
    for (const { m } of shown)
      m.frames.density.forEach((d, f) => {
        if ((m.frames.count[f] as number) < 30 && f < m.frames.density.length - 1) return
        for (let g = 0; g < G; g++) top = Math.max(top, d[j * G + g] ?? 0)
      })
    return top > 0 ? 1.12 * top : 1
  }

  const panels: Record<string, Panel> = {}
  const text: Record<string, unknown>[] = []
  const rects = subplotRects(grid, {}, fig)
  for (let r = 0; r < k; r++)
    for (let c = 0; c < k; c++) {
      const id = `p${r}_${c}`
      const cx = coordinate(c)
      const cy = coordinate(r)
      if (cx === undefined || cy === undefined) continue
      const axes = {
        x: { limits: cx.range as [number, number], show: r === k - 1, splitLine: false },
        y: { limits: cy.range as [number, number], show: c === 0 && r > 0, splitLine: false },
      }
      const edge = pairEdge(labels, r, c)
      if (r === c) {
        const G = cx.grid.length
        const layers: Layer[] = []
        if (frame >= 0)
          for (const { m, i } of shown)
            if ((m.frames.count[frame] as number) >= 2)
              layers.push(
                lines(
                  [
                    {
                      x: cx.grid,
                      y: (m.frames.density[frame] as readonly number[]).slice(
                        (chosen[c] as number) * G,
                        ((chosen[c] as number) + 1) * G,
                      ),
                      name: m.name,
                      colour: colour(i),
                      width: 1.6,
                      area: 0.14,
                    },
                  ],
                  { mode },
                ),
              )
        if (cx.truth !== undefined)
          layers.push(referenceLines({ x: [cx.truth] }, { mode, colour: 'fg', dash: 'dashed' }))
        panels[id] = {
          layers,
          options: { x: { ...axes.x, ...edge.x }, y: { limits: [0, ceiling(c)], show: false, splitLine: false } },
        }
      } else if (r > c) {
        // Lower triangle: x is the column's coordinate, y the row's.
        const layers: Layer[] = []
        for (const { m, i } of shown) {
          const n = seen(m)
          if (n === 0) continue
          layers.push(
            points(
              [
                {
                  x: column(m, chosen[c] as number, n),
                  y: column(m, chosen[r] as number, n),
                  name: m.name,
                  colour: colour(i),
                  size: 3.6,
                  opacity: 0.4,
                  outline: false,
                },
              ],
              { mode },
            ),
          )
        }
        for (const { m } of shown) {
          const n = seenDivergent(m)
          if (n === 0) continue
          layers.push(
            points(
              [
                {
                  x: (m.divergent.values[chosen[c] as number] as readonly number[]).slice(0, n),
                  y: (m.divergent.values[chosen[r] as number] as readonly number[]).slice(0, n),
                  name: 'divergent',
                  colour: DIVERGENT_COLOUR,
                  size: 6,
                  symbol: 'cross',
                },
              ],
              { mode },
            ),
          )
        }
        if (cx.truth !== undefined || cy.truth !== undefined)
          layers.push(
            referenceLines(
              {
                ...(cx.truth === undefined ? {} : { x: [cx.truth] }),
                ...(cy.truth === undefined ? {} : { y: [cy.truth] }),
              },
              { mode, colour: 'fg', dash: 'dashed' },
            ),
          )
        if (cx.truth !== undefined && cy.truth !== undefined)
          layers.push(
            points(
              [{ x: [cx.truth], y: [cy.truth], name: record.referenceLabel, colour: 'fg', symbol: 'diamond', size: 9 }],
              { mode },
            ),
          )
        panels[id] = { layers, options: { x: { ...axes.x, ...edge.x }, y: { ...axes.y, ...edge.y } } }
      } else {
        // Upper triangle: x is the column's coordinate, y the row's (the lower panel turned over).
        const layers: Layer[] = []
        const a = chosen[r] as number
        const b = chosen[c] as number
        const rho = (m: PairsMethod): number =>
          (m.frames.correlation[frame] as readonly number[])[pairIndex(K, Math.min(a, b), Math.max(a, b))] ?? 0
        const ready = (m: PairsMethod): boolean => frame >= 0 && (m.frames.count[frame] as number) >= 2
        if (upper === 'ellipses') {
          for (const { m, i } of shown) {
            if (!ready(m)) continue
            const mean = m.frames.mean[frame] as readonly number[]
            const sd = m.frames.sd[frame] as readonly number[]
            for (const mass of [0.5, 0.9]) {
              const e = ellipse(mean[b] as number, mean[a] as number, sd[b] as number, sd[a] as number, rho(m), mass)
              layers.push(
                lines(
                  [
                    {
                      x: e.x,
                      y: e.y,
                      name: mass === 0.9 ? m.name : `__${m.name} 50`,
                      colour: colour(i),
                      width: mass === 0.9 ? 1.5 : 1,
                      dash: mass === 0.9 ? 'solid' : 'dashed',
                    },
                  ],
                  { mode },
                ),
              )
            }
          }
          if (cx.truth !== undefined && cy.truth !== undefined)
            layers.push(
              points(
                [{ x: [cx.truth], y: [cy.truth], name: '__reference', colour: 'fg', symbol: 'diamond', size: 7 }],
                { mode },
              ),
            )
        } else {
          const rect = rects[id] as Rect
          const size = Math.max(9, Math.min(fig.fontSize, (rect.height - 8) / (shown.length + 0.6)))
          const wide = rect.width >= 70
          shown.forEach(({ m, i }, row) => {
            if (!ready(m)) return
            const value = rho(m)
            text.push({
              type: 'text',
              silent: true,
              z: 10,
              left: rect.x + rect.width / 2,
              top: rect.y + rect.height / 2 + (row - (shown.length - 1) / 2) * size * 1.35,
              style: {
                text: `${wide ? 'r = ' : ''}${value.toFixed(2).replace('-', '−')}`,
                fill: resolvePaint(colour(i) as Paint, mode),
                font: `${600} ${size}px ${t.font}`,
                align: 'center',
                verticalAlign: 'middle',
              },
            })
          })
        }
        panels[id] = { layers, options: { x: { ...axes.x, show: false }, y: { ...axes.y, show: false } } }
      }
    }

  const option = subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' }) as Record<string, unknown>
  const series = option['series'] as { name?: string }[]
  const has = (n: string): boolean => series.some((s) => s.name === n)
  const legendNames = [...shown.map(({ m }) => m.name), 'divergent', record.referenceLabel].filter(has)
  const frameLegend = chartFrame({ ...fig, mode }, { legendNames })['legend'] as object
  return {
    ...option,
    legend: { ...frameLegend, show: true, top: 4, data: legendNames, selectedMode: false },
    graphic: [...((option['graphic'] as unknown[] | undefined) ?? []), ...text],
    ...ariaOf(
      `Pair plot of ${labels.join(', ')} for ${shown.map(({ m }) => m.name).join(', ')} at step ${short(cursor)}${record.methods.some((m) => m.numDivergent > 0) ? ', divergent transitions marked' : ''}`,
    ),
  } as EChartsOption
}
