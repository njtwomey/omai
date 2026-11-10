/**
 * The views of a `ChainDiagnostics` record, the charts of an MCMC run: trace plots (every chain's draws over the
 * iterations, the divergent transitions marked), rank plots (each chain's histogram of the pooled ranks of its draws,
 * flat when the chains mix, with the expected count drawn) and the R-hat and ESS table (`chainTable` of the
 * record, drawn by the marked-table view).
 */

import type { EChartsOption } from 'echarts'
import type { ChainDiagnostics, ChainQuantity } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { points } from '../charts/points.ts'
import { panelColumns, panelGrid, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link traceView} and {@link rankView}. */
export interface ChainViewOptions extends ViewOptions {
  /** The quantities drawn, by name (default the worst few by R-hat, else the first, at most `max`). */
  readonly quantities?: readonly string[]
  /** The most quantities drawn when none are named (default 6). */
  readonly max?: number
}

/**
 * The quantities a chain chart draws: those named, else the ones with the largest R-hat first (the ones that need
 * looking at), the rest in order, at most `max`.
 *
 * @param record - The diagnostics.
 * @param options - Names and the cap.
 * @returns The quantities.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { chosenQuantities } from 'omai-render/views'
 * const r = chainDiagnostics({ a: [[1, 2], [2, 1]], b: [[1, 2], [2, 1]] }, { summary: [{ site: 'b', name: 'b', rhat: 1.2 }] })
 * chosenQuantities(r, { max: 1 }).map((q) => q.name) // => ['b']
 * ```
 */
export function chosenQuantities(record: ChainDiagnostics, options: ChainViewOptions = {}): ChainQuantity[] {
  if (options.quantities !== undefined)
    return options.quantities.flatMap((n) => record.quantities.filter((q) => q.name === n))
  return [...record.quantities]
    .map((q, i) => ({ q, i }))
    .sort((a, b) => (b.q.rhat ?? 0) - (a.q.rhat ?? 0) || a.i - b.i)
    .slice(0, options.max ?? 6)
    .map((x) => x.q)
}

/**
 * Trace plots: for each quantity the draws of every chain over the iterations, one colour per chain, with the divergent
 * transitions marked on their chain. Chains that mix look like overlapping noise around one level.
 *
 * @param record - The diagnostics.
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @index function Trace plots of MCMC chains per quantity with divergences marked: `traceView(chainDiagnostics(draws))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { traceView } from 'omai-render/views'
 * traceView(chainDiagnostics({ mu: [[0, 1, 0, 1], [1, 0, 1, 0]] })).grid.length // => 1
 * ```
 */
export function traceView(record: ChainDiagnostics, options: ChainViewOptions = {}): EChartsOption {
  const qs = chosenQuantities(record, options)
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const hits = record.divergences.filter((d) => d.draw < (q.chains[d.chain]?.length ?? 0))
    panels[`q${i}`] = {
      layers: [
        lines(
          q.chains.map((c, k) => ({ y: [...c], name: `chain ${k}`, colour: `tab${k}`, width: 1, opacity: 0.85 })),
          { mode },
        ),
        ...(hits.length === 0
          ? []
          : [
              points(
                [
                  {
                    x: hits.map((d) => d.draw),
                    y: hits.map((d) => q.chains[d.chain]?.[d.draw] ?? 0),
                    name: 'divergent',
                    colour: 'tab3',
                    size: 9,
                  },
                ],
                { mode },
              ),
            ]),
      ],
      options: {
        title: q.rhat === undefined ? q.name : `${q.name}   R̂ ${q.rhat.toFixed(3)}`,
        ...(i >= qs.length - cols ? { x: { label: 'draw' } } : {}),
      },
    }
  }
  return subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' })
}

/**
 * Rank plots: for each quantity the histogram of each chain's ranks among all the draws pooled, as steps, with the count
 * a perfectly mixed chain would have in each bin drawn dashed. A chain that sits high or low is a ramp, one that
 * wanders a hump; chains of the same distribution are flat and close to the line (Vehtari et al. 2021).
 *
 * @param record - The diagnostics.
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @index function Rank plots of MCMC chains per quantity: `rankView(chainDiagnostics(draws))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { rankView } from 'omai-render/views'
 * rankView(chainDiagnostics({ mu: [[0, 1, 0, 1], [1, 0, 1, 0]] }, { bins: 2 })).grid.length // => 1
 * ```
 */
export function rankView(record: ChainDiagnostics, options: ChainViewOptions = {}): EChartsOption {
  const qs = chosenQuantities(record, options)
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const bins = q.ranks.bins
    const draws = q.chains[0]?.length ?? 0
    // The draws counted are all of the chain's, the trace may be thinned: the expected count is the chain's draws over the bins.
    const expected = (q.ranks.counts[0]?.reduce((a, b) => a + b, 0) ?? draws) / bins
    panels[`q${i}`] = {
      layers: [
        lines(
          q.ranks.counts.map((c, k) => ({
            // A plateau per bin: the count held from the bin's start to its end.
            x: c.flatMap((_, b) => [b, b + 1]),
            y: c.flatMap((v) => [v, v]),
            name: `chain ${k}`,
            colour: `tab${k}`,
            width: 1.4,
          })),
          { mode },
        ),
        referenceLines({ y: [expected] }, { mode, colour: 'muted' }),
      ],
      options: {
        title: q.name,
        y: { limits: [0, Math.max(expected * 2, ...q.ranks.counts.flat())] },
        ...(i >= qs.length - cols ? { x: { label: 'rank bin' } } : {}),
      },
    }
  }
  return subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' })
}
