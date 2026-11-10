/**
 * The exposition record of a simulation-based calibration run (`sbcRun`): for each quantity, the histogram of the ranks
 * of the true value among the posterior draws with the count each bin should have and the exact binomial band around it,
 * the ECDF of the ranks minus the uniform CDF with the simultaneous band of Säilynoja et al. (2022), the p-values of the
 * uniformity tests and the verdict with its reading (too confident, too wide, biased). It is built from the result of
 * the validation kernel's `sbc` by its shape alone (`SbcResultLike`), so a view draws a calibration campaign without
 * the record depending on the kernel.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'
import { binomialInterval, ecdfBand } from './ecdf-band.ts'
import { type CellMark, type MarkedTable, markedTable } from './tables.ts'

/** What one quantity's ranks say, in the shape the validation kernel's `sbc` returns (`SbcQuantity`). */
export interface SbcQuantityLike {
  readonly name: string
  /** The rank of the truth among the draws, one per replication, in 0..numRanks. */
  readonly ranks: readonly number[]
  readonly numRanks: number
  /** Counts of the ranks in equal bins of the values 0..numRanks. */
  readonly histogram: readonly number[]
  readonly chiSquare: { readonly statistic: number; readonly pValue: number }
  readonly ecdf: { readonly statistic: number; readonly pValue: number }
  readonly pValue: number
  readonly diagnosis: string
  readonly passes: boolean
}

/** The result of the validation kernel's `sbc`, by shape. */
export interface SbcResultLike {
  readonly replications: number
  readonly numRanks: number
  readonly alpha: number
  readonly quantities: readonly SbcQuantityLike[]
  readonly passes: boolean
}

/** The calibration of one quantity: the rank histogram with its band, the ECDF difference with its band, and the tests. */
export interface SbcRunQuantity {
  readonly name: string
  /** Counts of the ranks per bin. */
  readonly histogram: readonly number[]
  /** The count a uniform rank would give each bin on average, and the pointwise binomial band around it. */
  readonly expected: readonly number[]
  readonly band: { readonly lower: readonly number[]; readonly upper: readonly number[] }
  /**
   * The ECDF of the ranks minus the uniform CDF at the grid `z` (proportions), the simultaneous band around zero, and
   * how many grid points lie outside it.
   */
  readonly ecdfDifference: {
    readonly z: readonly number[]
    readonly diff: readonly number[]
    readonly lower: readonly number[]
    readonly upper: readonly number[]
    readonly outside: number
  }
  readonly chiSquare: { readonly statistic: number; readonly pValue: number }
  readonly ecdfTest: { readonly statistic: number; readonly pValue: number }
  readonly pValue: number
  /** `uniform`, or the reading of a failing histogram. */
  readonly diagnosis: string
  readonly passes: boolean
}

/** A calibration run: the quantities, the number of replications and ranks, the level of the bands and the verdict. */
export interface SbcRun {
  readonly kind: 'sbcRun'
  readonly replications: number
  /** The number of draws each rank is taken among: ranks are in 0..numRanks. */
  readonly numRanks: number
  /** The level of the tests together, as the kernel was run with. */
  readonly alpha: number
  /** The probability the bands hold: the simultaneous ECDF band and the pointwise histogram bands. */
  readonly level: number
  readonly passes: boolean
  readonly quantities: readonly SbcRunQuantity[]
}

/** Options of {@link sbcRun}. */
export interface SbcRunOptions {
  /** The probability the bands hold (default `1 - alpha / quantities`, the level the kernel's tests use per quantity). */
  readonly level?: number
  /** The most grid points of the ECDF difference (default 100; the ranks are read at every point when there are fewer). */
  readonly points?: number
}

/** The grid of cells read for a rank ECDF: `i / (L + 1)` for `i` in 1..L, thinned evenly to at most `points`. */
function rankGrid(numRanks: number, points: number): number[] {
  if (numRanks <= points) return Array.from({ length: numRanks }, (_, i) => i + 1)
  const out = new Set<number>()
  for (let j = 1; j <= points; j++)
    out.add(Math.min(numRanks, Math.max(1, Math.round((j * (numRanks + 1)) / (points + 1)))))
  return [...out].sort((a, b) => a - b)
}

/**
 * The record of a calibration run, from the validation kernel's `sbc` result: for each quantity the ranks' histogram
 * with the exact binomial band, the ECDF difference with the exact simultaneous band, the tests and the verdict.
 *
 * @param result - The result of `sbc` (any object of that shape).
 * @param options - The level of the bands and the resolution of the ECDF difference.
 * @returns The record.
 * @throws OmaiError `shape` for a rank outside 0..numRanks, a histogram that does not cover the ranks, or no replications;
 *   `value` for a level outside (0, 1).
 * @index function Rank histogram and ECDF-difference data with exact bands from a simulation-based calibration result.
 * @example
 * ```ts
 * import { sbcRun } from 'omai-engine/exposition'
 * // Ranks of 10 replications among 3 draws (ranks 0..3), evenly spread.
 * const ranks = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
 * const q = { name: 'mu', ranks, numRanks: 3, histogram: [2, 3, 3, 2], chiSquare: { statistic: 0.4, pValue: 0.94 }, ecdf: { statistic: 0.05, pValue: 0.9 }, pValue: 1, diagnosis: 'uniform', passes: true }
 * const r = sbcRun({ replications: 10, numRanks: 3, alpha: 0.01, quantities: [q], passes: true })
 * [r.quantities[0]?.expected, r.quantities[0]?.ecdfDifference.z] // => [[2.5, 2.5, 2.5, 2.5], [0.25, 0.5, 0.75]]
 * ```
 */
export function sbcRun(result: SbcResultLike, options: SbcRunOptions = {}): SbcRun {
  const { replications, numRanks } = result
  if (!(replications >= 1)) throw new OmaiError('shape', 'sbcRun: no replications')
  const level = options.level ?? 1 - result.alpha / Math.max(1, result.quantities.length)
  if (!(level > 0 && level < 1)) throw new OmaiError('value', 'sbcRun: the level must be inside (0, 1)')
  const grid = rankGrid(numRanks, options.points ?? 100)
  const z = grid.map((i) => i / (numRanks + 1))
  // The band depends on the number of replications and the grid alone: it is the same for every quantity.
  const band = ecdfBand(replications, z, level)
  const quantities = result.quantities.map((q): SbcRunQuantity => {
    if (q.ranks.length !== replications)
      throw new OmaiError('shape', `sbcRun: ${q.name} has ${q.ranks.length} ranks, expected ${replications}`)
    const counts = new Array<number>(numRanks + 1).fill(0)
    for (const r of q.ranks) {
      if (!(Number.isInteger(r) && r >= 0 && r <= numRanks))
        throw new OmaiError('shape', `sbcRun: ${q.name} has a rank ${r} outside 0..${numRanks}`)
      counts[r] = (counts[r] ?? 0) + 1
    }
    const bins = q.histogram.length
    if (bins < 1 || q.histogram.reduce((s, v) => s + v, 0) !== replications)
      throw new OmaiError('shape', `sbcRun: the histogram of ${q.name} does not count every rank`)
    // The share of the values 0..L in each bin, by the kernel's rule for placing a rank in a bin.
    const width = (numRanks + 1) / bins
    const share = new Array<number>(bins).fill(0)
    for (let r = 0; r <= numRanks; r++) {
      const b = Math.min(bins - 1, Math.floor(r / width))
      share[b] = (share[b] ?? 0) + 1 / (numRanks + 1)
    }
    const intervals = share.map((p) => binomialInterval(replications, p, level))
    let acc = 0
    const cumulative = counts.map((c) => (acc += c))
    const diff = grid.map((i, j) => (cumulative[i - 1] ?? 0) / replications - (z[j] ?? 0))
    const lower = (band.lower as number[]).map((v, j) => v - (z[j] ?? 0))
    const upper = (band.upper as number[]).map((v, j) => v - (z[j] ?? 0))
    const outside = diff.filter((d, j) => d < (lower[j] ?? 0) - 1e-12 || d > (upper[j] ?? 0) + 1e-12).length
    return {
      name: q.name,
      histogram: [...q.histogram],
      expected: share.map((p) => p * replications),
      band: { lower: intervals.map((v) => v[0]), upper: intervals.map((v) => v[1]) },
      ecdfDifference: { z, diff, lower, upper, outside },
      chiSquare: { ...q.chiSquare },
      ecdfTest: { ...q.ecdf },
      pValue: q.pValue,
      diagnosis: q.diagnosis,
      passes: q.passes,
    }
  })
  return seal({ kind: 'sbcRun', replications, numRanks, alpha: result.alpha, level, passes: result.passes, quantities })
}

/**
 * The calibration of a run as a table: one row per quantity with the two tests' p-values, the smaller doubled, the
 * number of ECDF points outside the band and the verdict; a failing quantity is marked bad with its reading, a passing
 * one ok.
 *
 * @param record - The calibration run.
 * @returns The table.
 * @example
 * ```ts
 * import { sbcRun, sbcTable } from 'omai-engine/exposition'
 * const q = { name: 'mu', ranks: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], numRanks: 1, histogram: [9, 1], chiSquare: { statistic: 6.4, pValue: 0.011 }, ecdf: { statistic: 0.4, pValue: 0.001 }, pValue: 0.002, diagnosis: 'truth ranks low: the posterior is biased high', passes: false }
 * const t = sbcTable(sbcRun({ replications: 10, numRanks: 1, alpha: 0.01, quantities: [q], passes: false }))
 * [t.marks?.[0]?.status, t.cells[0]?.[5]] // => ['bad', 'truth ranks low: the posterior is biased high']
 * ```
 */
export function sbcTable(record: SbcRun): MarkedTable {
  const marks: CellMark[] = record.quantities.map((q, row) => ({
    row,
    col: 5,
    status: q.passes ? 'ok' : 'bad',
    ...(q.passes ? {} : { note: q.diagnosis }),
  }))
  return markedTable({
    columns: ['quantity', 'chi-square p', 'ECDF p', 'p-value', 'ECDF points outside', 'verdict'],
    cells: record.quantities.map((q) => [
      q.name,
      q.chiSquare.pValue,
      q.ecdfTest.pValue,
      q.pValue,
      q.ecdfDifference.outside,
      q.passes ? 'uniform' : q.diagnosis,
    ]),
    marks,
    digits: 3,
    caption: `${record.replications} replications, ranks among ${record.numRanks} draws: ${record.passes ? 'calibrated' : 'NOT calibrated'} at level ${record.alpha}`,
  })
}
