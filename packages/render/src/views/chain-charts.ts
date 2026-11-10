/**
 * The other charts of a `ChainDiagnostics` record, beside the trace and rank plots of `chain-diagnostics.ts`: the ECDF of
 * each chain's ranks minus the uniform CDF inside the simultaneous band, the autocorrelation of each chain, the
 * effective sample size by quantile and as the draws accumulate, the energy distributions with each chain's E-BFMI, and
 * the pair plot of a few quantities with the divergent transitions marked. Each reads what the record holds and says
 * what it lacks (`hasChart`), so a page offers the charts its record can draw.
 */

import type { EChartsOption } from 'echarts'
import { OmaiError } from 'omai-engine/base'
import { type ChainDiagnostics, MAX_DIVERGENT_VALUES } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { kde } from '../charts/distributions.ts'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { ariaOf, panelColumns, panelGrid, short, viewFigure } from './_view.ts'
import { type ChainViewOptions, chosenQuantities } from './chain-diagnostics.ts'

/** The charts of a chain record beyond the trace and rank plots, by name. */
export type ChainChart = 'ecdf' | 'autocorrelation' | 'essQuantile' | 'essEvolution' | 'energy' | 'pairs'

/**
 * Whether the record holds what a chart draws: the ECDF difference, the autocorrelation, the ESS curves and the
 * energy are optional parts of the record (`chainDiagnostics` options), and a pair plot needs two quantities.
 *
 * @param record - The diagnostics.
 * @param chart - The chart.
 * @returns Whether the chart can be drawn.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { hasChart } from 'omai-render/views'
 * const r = chainDiagnostics({ mu: [[1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]] })
 * [hasChart(r, 'ecdf'), hasChart(r, 'energy'), hasChart(r, 'pairs')] // => [true, false, false]
 * ```
 */
export function hasChart(record: ChainDiagnostics, chart: ChainChart): boolean {
  switch (chart) {
    case 'ecdf':
      return record.ecdf !== undefined && record.quantities.some((q) => q.ecdfDiff !== undefined)
    case 'autocorrelation':
      return record.quantities.some((q) => q.acf !== undefined)
    case 'essQuantile':
      return record.quantities.some((q) => (q.essCurves?.quantile.probs.length ?? 0) > 1)
    case 'essEvolution':
      return record.quantities.some((q) => (q.essCurves?.evolution.draws.length ?? 0) > 1)
    case 'energy':
      return record.energy !== undefined
    case 'pairs':
      return record.quantities.length >= 2
  }
}

const missing = (chart: string, what: string): OmaiError =>
  new OmaiError(
    'value',
    `${chart}: the record has no ${what} (make it with chainDiagnostics and the option that adds it)`,
  )

/**
 * ECDF-difference plots: for each quantity the ECDF of each chain's pooled ranks minus the uniform CDF, against the
 * uniform CDF, inside the shaded simultaneous band that holds every chain for chains of one distribution with the
 * record's probability. Easier to read than a rank histogram for few draws: a chain that sits high is a curve below
 * zero, one that wanders an S, one that is too narrow or wide a hump or a dip; a curve leaving the band is the signal.
 *
 * @param record - The diagnostics, made with the ECDF (the default of `chainDiagnostics`).
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` when the record has no ECDF.
 * @index function ECDF-difference plots of MCMC chains with the simultaneous band: `ecdfView(chainDiagnostics(draws))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { ecdfView } from 'omai-render/views'
 * ecdfView(chainDiagnostics({ mu: [[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]] }, { ecdfPoints: 3 })).grid.length // => 1
 * ```
 */
export function ecdfView(record: ChainDiagnostics, options: ChainViewOptions = {}): EChartsOption {
  const e = record.ecdf
  if (e === undefined) throw missing('ecdfView', 'ECDF')
  const qs = chosenQuantities(record, options).filter((q) => q.ecdfDiff !== undefined)
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const x = [0, ...e.z, 1]
  const pad = (v: readonly number[]): number[] => [0, ...v, 0]
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const diff = q.ecdfDiff ?? []
    const reach = Math.max(...e.upper, ...e.lower.map(Math.abs), ...diff.flat().map(Math.abs))
    panels[`q${i}`] = {
      layers: [
        band([{ x, lower: pad(e.lower), upper: pad(e.upper), name: '__band', colour: 'muted', opacity: 0.25 }], {
          mode,
        }),
        referenceLines({ y: [0] }, { mode, colour: 'muted', dash: 'dashed' }),
        lines(
          diff.map((d, k) => ({ x, y: pad(d), name: `chain ${k}`, colour: `tab${k}`, width: 1.4 })),
          { mode },
        ),
      ],
      options: {
        title: q.name,
        y: { limits: [-reach * 1.2, reach * 1.2] },
        ...(i >= qs.length - cols ? { x: { label: 'fractional rank' } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' }),
    ...ariaOf(
      `ECDF difference of the chains of ${qs.map((q) => q.name).join(', ')} inside the ${short(e.level * 100)} percent band`,
    ),
  }
}

/**
 * Autocorrelation plots: for each quantity the autocorrelation of every chain at lags up to the record's `acf.lags`,
 * one thin line per chain and the mean over the chains thick, with zero dashed. Draws that decay to zero within a few
 * lags are nearly independent; a slow decay is a small effective sample size.
 *
 * @param record - The diagnostics, made with the autocorrelation (the default of `chainDiagnostics`).
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` when the record has no autocorrelation.
 * @index function Autocorrelation plots of MCMC chains per quantity: `autocorrelationView(chainDiagnostics(draws))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { autocorrelationView } from 'omai-render/views'
 * autocorrelationView(chainDiagnostics({ mu: [[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]] })).grid.length // => 1
 * ```
 */
export function autocorrelationView(record: ChainDiagnostics, options: ChainViewOptions = {}): EChartsOption {
  const qs = chosenQuantities(record, options).filter((q) => q.acf !== undefined)
  if (qs.length === 0) throw missing('autocorrelationView', 'autocorrelation')
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const acf = q.acf
    if (acf === undefined) continue
    const lag = Array.from({ length: acf.lags + 1 }, (_, k) => k)
    const low = Math.min(0, ...acf.chains.flat())
    panels[`q${i}`] = {
      layers: [
        referenceLines({ y: [0] }, { mode, colour: 'muted', dash: 'dashed' }),
        lines(
          acf.chains.map((c, k) => ({ x: lag, y: c, name: `chain ${k}`, colour: `tab${k}`, width: 1, opacity: 0.6 })),
          { mode },
        ),
        lines([{ x: lag, y: acf.mean, name: 'mean', colour: 'fg', width: 2 }], { mode }),
      ],
      options: {
        title: q.name,
        y: { limits: [Math.max(-1, low - 0.1), 1.05] },
        ...(i >= qs.length - cols ? { x: { label: 'lag' } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' }),
    ...ariaOf(`Autocorrelation of the chains of ${qs.map((q) => q.name).join(', ')}`),
  }
}

/** The effective sample size the usual advice asks for: at least 100 per chain, about 400 with four chains. */
const ESS_TARGET = 400

/** Which ESS curve {@link essView} draws. */
export interface EssViewOptions extends ChainViewOptions {
  /** `quantile`: the ESS of each quantile's indicator; `evolution`: bulk and tail ESS as the draws accumulate (default `quantile`). */
  readonly curve?: 'quantile' | 'evolution'
}

/**
 * Effective sample size plots: by quantile, for each quantity the ESS of the indicator of each quantile (low in the
 * tails when the chains explore them badly), or as the draws accumulate, the bulk and tail ESS against the number of
 * draws per chain (a healthy run grows in line with the draws; a flat curve has stopped learning). The line of 400
 * marks the usual minimum.
 *
 * @param record - The diagnostics, made with `essCurves` (`chainEssCurves` of `omai-engine/sampling`).
 * @param options - The curve, quantities, mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` when the record has no ESS curves.
 * @index function Effective sample size by quantile or as the draws accumulate: `essView(chainDiagnostics(draws, { essCurves }))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { essView } from 'omai-render/views'
 * const curves = { quantile: { probs: [0.25, 0.5, 0.75], ess: [300, 450, 320] }, evolution: { draws: [10, 20], bulk: [8, 18], tail: [5, 9] } }
 * essView(chainDiagnostics({ mu: [[0, 1, 0, 1], [1, 0, 1, 0]] }, { essCurves: { mu: curves } })).grid.length // => 1
 * ```
 */
export function essView(record: ChainDiagnostics, options: EssViewOptions = {}): EChartsOption {
  const evolution = options.curve === 'evolution'
  const qs = chosenQuantities(record, options).filter((q) => q.essCurves !== undefined)
  if (qs.length === 0) throw missing('essView', 'ESS curves')
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const c = q.essCurves
    if (c === undefined) continue
    const top = Math.max(ESS_TARGET, ...(evolution ? [...c.evolution.bulk, ...c.evolution.tail] : c.quantile.ess))
    panels[`q${i}`] = {
      layers: [
        referenceLines({ y: [ESS_TARGET] }, { mode, colour: 'muted', dash: 'dashed' }),
        evolution
          ? lines(
              [
                { x: c.evolution.draws, y: c.evolution.bulk, name: 'bulk', colour: 'tab0', markers: true, width: 1.6 },
                { x: c.evolution.draws, y: c.evolution.tail, name: 'tail', colour: 'tab1', markers: true, width: 1.6 },
              ],
              { mode },
            )
          : lines(
              [{ x: c.quantile.probs, y: c.quantile.ess, name: 'ESS', colour: 'tab0', markers: true, width: 1.6 }],
              {
                mode,
              },
            ),
      ],
      options: {
        title: q.name,
        y: { limits: [0, top * 1.1] },
        ...(i >= qs.length - cols ? { x: { label: evolution ? 'draws per chain' : 'quantile' } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: evolution, tooltip: 'item' }),
    ...ariaOf(
      `Effective sample size ${evolution ? 'as the draws accumulate' : 'by quantile'} of ${qs.map((q) => q.name).join(', ')}`,
    ),
  }
}

/**
 * The energy plot: the marginal energy distribution (each chain's energy minus its mean) and the distribution of the
 * change of energy between draws, as filled steps on common bins, with each chain's E-BFMI in the title. The two should
 * overlap; when the change is much narrower than the marginal, or an E-BFMI is under 0.3, the sampler cannot move
 * between energy levels and the model needs reparametrising.
 *
 * @param record - The diagnostics, made with the `energy` option.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` when the record has no energy.
 * @index function Energy and energy-transition distributions with E-BFMI per chain: `energyView(chainDiagnostics(draws, { energy }))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { energyView } from 'omai-render/views'
 * const energy = [[1, 3, 2, 4, 2, 3], [2, 1, 3, 2, 4, 3]]
 * energyView(chainDiagnostics({ mu: [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]] }, { energy })).grid.length // => 1
 * ```
 */
export function energyView(record: ChainDiagnostics, options: ChainViewOptions = {}): EChartsOption {
  const e = record.energy
  if (e === undefined) throw missing('energyView', 'energy')
  const fig = viewFigure(options, { aspect: 1.9 })
  const mode = fig.mode
  const x = e.edges.slice(0, -1).flatMap((lo, b) => [lo, e.edges[b + 1] as number])
  const flat = (v: readonly number[]): number[] => v.flatMap((d) => [d, d])
  const worst = Math.min(...e.bfmi)
  const panel: Panel = {
    layers: [
      lines([{ x, y: flat(e.marginal), name: 'energy', colour: 'tab0', width: 1.4, area: 0.25 }], { mode }),
      lines([{ x, y: flat(e.transition), name: 'energy change', colour: 'tab1', width: 1.4, area: 0.25 }], { mode }),
    ],
    options: {
      title: `E-BFMI ${e.bfmi.map((v) => short(v)).join(', ')}${worst < 0.3 ? '   low: reparametrise' : ''}`,
      x: { label: 'energy (centred) and its change' },
      y: { label: 'density' },
    },
  }
  return {
    ...subplots(
      gridSpec(1, 1, { cells: [{ id: 'q0', row: 0, col: 0 }] }),
      { q0: panel },
      { figure: fig, legend: true, tooltip: 'item' },
    ),
    ...ariaOf('Distribution of the energy and of its change between draws, with the E-BFMI of each chain'),
  }
}

/** Options of {@link pairView}. */
export interface PairViewOptions extends ChainViewOptions {
  /** The most points drawn per chain (default 150, thinned evenly: a scatter of thousands is a blob). */
  readonly maxPoints?: number
}

/**
 * The pair plot of a few quantities: a lower triangle of scatter plots of the draws of every chain (a colour each)
 * with the divergent transitions on top in red, and on the diagonal each quantity's density. Divergences that
 * gather at one end of a pair (the narrow neck of a funnel) show where the sampler failed. Up to four quantities
 * (the ones named, else those with the worst R-hat).
 *
 * @param record - The diagnostics (at least two quantities).
 * @param options - Quantities (default the four worst by R-hat), points per chain, mode and figure.
 * @returns The ECharts option.
 * @throws OmaiError `value` for fewer than two quantities.
 * @index function Pair plot of a few quantities with the divergent transitions marked: `pairView(chainDiagnostics(draws, { divergent }))`.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * import { pairView } from 'omai-render/views'
 * const a = [[0, 1, 0, 1, 2], [1, 0, 1, 0, 2]]
 * pairView(chainDiagnostics({ a, b: a }, { divergent: [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]] })).grid.length // => 3
 * ```
 */
export function pairView(record: ChainDiagnostics, options: PairViewOptions = {}): EChartsOption {
  const qs = chosenQuantities(record, { ...options, max: options.max ?? 4 })
  if (qs.length < 2) throw new OmaiError('value', 'pairView: needs at least two quantities')
  const k = qs.length
  const fig = viewFigure(options, { aspect: 1 })
  const mode = fig.mode
  const cap = options.maxPoints ?? 150
  const thin = (v: readonly number[]): number[] =>
    v.length <= cap
      ? [...v]
      : Array.from({ length: cap }, (_, i) => v[Math.floor((i * (v.length - 1)) / (cap - 1))] as number)
  const grid = gridSpec(k, k, {
    hspace: 12,
    wspace: 12,
    cells: qs.flatMap((_, r) => qs.slice(0, r + 1).map((__, c) => ({ id: `p${r}_${c}`, row: r, col: c }))),
  })
  const panels: Record<string, Panel> = {}
  const shown = Math.min(record.divergences.length, MAX_DIVERGENT_VALUES)
  for (const [r, qy] of qs.entries())
    for (const [c, qx] of qs.slice(0, r + 1).entries()) {
      const edge = {
        ...(r === k - 1 ? { x: { label: qx.name } } : {}),
        ...(c === 0 && r > 0 ? { y: { label: qy.name } } : {}),
      }
      if (r === c) {
        panels[`p${r}_${c}`] = {
          layers: [kde(qx.chains.flat(), { mode, colour: 'tab0', fill: 0.2 })],
          options: r === k - 1 ? { x: { label: qx.name } } : {},
        }
        continue
      }
      const layers = qx.chains.map((xs, chain) =>
        points(
          [
            {
              x: thin(xs),
              y: thin(qy.chains[chain] ?? []),
              name: `chain ${chain}`,
              colour: `tab${chain}`,
              size: 4,
              opacity: 0.45,
            },
          ],
          { mode },
        ),
      )
      const hx = qx.atDivergences
      const hy = qy.atDivergences
      if (hx !== undefined && hy !== undefined && hx.length === shown && hy.length === shown && shown > 0)
        layers.push(points([{ x: hx, y: hy, name: 'divergent', colour: 'tab3', size: 7 }], { mode }))
      panels[`p${r}_${c}`] = { layers, options: edge }
    }
  return {
    ...subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' }),
    ...ariaOf(
      `Pair plot of ${qs.map((q) => q.name).join(', ')}${record.divergences.length > 0 ? ' with the divergent transitions marked' : ''}`,
    ),
  }
}
