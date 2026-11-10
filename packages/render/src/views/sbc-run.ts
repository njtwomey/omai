/**
 * The views of a `SbcRun` record, the charts of simulation-based calibration: the histogram of the ranks of the true
 * value among the posterior draws with the count a uniform rank would give each bin and the binomial band around it,
 * and the ECDF of the ranks minus the uniform CDF with the simultaneous band (Säilynoja et al. 2022), per quantity. A
 * calibrated method is flat inside the band; too confident is a U or a curve that dips then rises out of the band, too
 * wide a hump, biased a slope. The verdict table is `sbcTable` of the record, drawn by the marked-table view.
 */

import type { EChartsOption } from 'echarts'
import type { SbcRun, SbcRunQuantity } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { ariaOf, panelColumns, panelGrid, short, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link sbcRankView} and {@link sbcEcdfView}. */
export interface SbcViewOptions extends ViewOptions {
  /** The quantities drawn, by name (default the failing ones first, then in order, at most `max`). */
  readonly quantities?: readonly string[]
  /** The most quantities drawn when none are named (default 6). */
  readonly max?: number
}

/**
 * The quantities a calibration chart draws: those named, else the failing ones first (the ones that need looking at),
 * then the rest in order, at most `max`.
 *
 * @param record - The calibration run.
 * @param options - Names and the cap.
 * @returns The quantities.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { chosenCalibrated } from 'omai-render/views'
 * chosenCalibrated(expositionSamples.sbcRun()).map((q) => q.name) // => ['mu']
 * ```
 */
export function chosenCalibrated(record: SbcRun, options: SbcViewOptions = {}): SbcRunQuantity[] {
  if (options.quantities !== undefined)
    return options.quantities.flatMap((n) => record.quantities.filter((q) => q.name === n))
  return [...record.quantities]
    .map((q, i) => ({ q, i }))
    .sort((a, b) => Number(a.q.passes) - Number(b.q.passes) || a.i - b.i)
    .slice(0, options.max ?? 6)
    .map((x) => x.q)
}

const titleOf = (q: SbcRunQuantity): string => (q.passes ? q.name : `${q.name}   ${q.diagnosis.split(':')[0]}`)

/**
 * Rank histograms: for each quantity the counts of the ranks of the true value as steps filled to the axis (red when
 * the quantity fails), the shaded binomial band around the count a uniform rank would give each bin, and that count
 * dashed. A bar outside the band is a bin that has too many or too few ranks; the shape (U, hump, slope) is the reading.
 *
 * @param record - The calibration run.
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @index function Rank histograms with the binomial band of a simulation-based calibration run: `sbcRankView(sbcRun(sbc(...)))`.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { sbcRankView } from 'omai-render/views'
 * sbcRankView(expositionSamples.sbcRun()).grid.length // => 1
 * ```
 */
export function sbcRankView(record: SbcRun, options: SbcViewOptions = {}): EChartsOption {
  const qs = chosenCalibrated(record, options)
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const cells = record.numRanks + 1
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const bins = q.histogram.length
    const at = (b: number): number => (b * cells) / bins
    // A plateau per bin: the count held from the bin's start to its end.
    const x = q.histogram.flatMap((_, b) => [at(b), at(b + 1)])
    const flat = (v: readonly number[]): number[] => v.flatMap((c) => [c, c])
    const top = Math.max(...q.histogram, ...q.band.upper)
    panels[`q${i}`] = {
      layers: [
        band(
          [{ x, lower: flat(q.band.lower), upper: flat(q.band.upper), name: '__band', colour: 'muted', opacity: 0.25 }],
          {
            mode,
          },
        ),
        lines([{ x, y: flat(q.expected), name: '__expected', colour: 'muted', dash: 'dashed', width: 1 }], { mode }),
        lines([{ x, y: flat(q.histogram), name: q.name, colour: q.passes ? 'tab0' : 'tab3', width: 1.6, area: 0.18 }], {
          mode,
        }),
      ],
      options: {
        title: titleOf(q),
        y: { limits: [0, top * 1.15] },
        ...(i >= qs.length - cols ? { x: { label: 'rank of the true value' } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' }),
    ...ariaOf(
      `Rank histograms of ${qs.map((q) => q.name).join(', ')}: flat inside the band when the method is calibrated`,
    ),
  }
}

/**
 * ECDF-difference plots: for each quantity the ECDF of the ranks minus the uniform CDF against the uniform CDF, inside
 * the shaded simultaneous band that holds the whole curve for a calibrated method with the record's probability. The
 * curve is red when the quantity fails, and its title says how many points lie outside the band. The reading by shape:
 * a dip then a rise (a smile) is too confident, a rise then a dip too wide, one sign throughout biased.
 *
 * @param record - The calibration run.
 * @param options - Quantities, mode and figure.
 * @returns The ECharts option.
 * @index function ECDF-difference plots with the simultaneous band of a simulation-based calibration run: `sbcEcdfView(sbcRun(sbc(...)))`.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { sbcEcdfView } from 'omai-render/views'
 * sbcEcdfView(expositionSamples.sbcRun()).grid.length // => 1
 * ```
 */
export function sbcEcdfView(record: SbcRun, options: SbcViewOptions = {}): EChartsOption {
  const qs = chosenCalibrated(record, options)
  const cols = panelColumns(qs.length)
  const fig = viewFigure(options, { aspect: cols === 2 ? 1.7 : Math.max(1.2, 3.2 / Math.max(1, qs.length)) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  for (const [i, q] of qs.entries()) {
    const e = q.ecdfDifference
    // The curve and the band run from the corner (0, 0) to the corner (1, 0): both ECDFs are pinned there.
    const x = [0, ...e.z, 1]
    const pad = (v: readonly number[]): number[] => [0, ...v, 0]
    const reach = Math.max(...e.upper, ...e.diff.map(Math.abs), ...e.lower.map(Math.abs))
    panels[`q${i}`] = {
      layers: [
        band([{ x, lower: pad(e.lower), upper: pad(e.upper), name: '__band', colour: 'muted', opacity: 0.25 }], {
          mode,
        }),
        referenceLines({ y: [0] }, { mode, colour: 'muted', dash: 'dashed' }),
        lines([{ x, y: pad(e.diff), name: q.name, colour: q.passes ? 'tab0' : 'tab3', width: 1.6 }], { mode }),
      ],
      options: {
        title: e.outside === 0 ? q.name : `${q.name}   ${e.outside} outside the band`,
        y: { limits: [-reach * 1.2, reach * 1.2] },
        ...(i >= qs.length - cols ? { x: { label: 'uniform CDF' } } : {}),
      },
    }
  }
  return {
    ...subplots(panelGrid(qs.length, cols), panels, { figure: fig, legend: false, tooltip: 'item' }),
    ...ariaOf(
      `ECDF difference of the ranks of ${qs.map((q) => q.name).join(', ')} with the ${short(record.level * 100)} percent simultaneous band`,
    ),
  }
}
