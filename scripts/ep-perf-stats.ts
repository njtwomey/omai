/**
 * The pure parts of the EP performance harness (`scripts/ep-perf.ts`): the median and spread of repeated timings, the calibration
 * figure that normalises a baseline taken on another or a busier machine, the comparison of a fresh measurement with the baseline,
 * and the small formatting helpers of the report. Nothing here imports the engine or reads a clock, so it is tested on its own
 * (`scripts/ep-perf.test.ts`); the clock lives in `ep-perf.ts`.
 */

/** The summary of repeated measurements of one quantity (all in one unit, microseconds or seconds). */
export interface Stat {
  /** The median, the figure the baseline keeps. */
  readonly median: number
  readonly min: number
  readonly max: number
  /** The number of measurements. */
  readonly n: number
  /** The spread, `(max - min) / median`: how much of the median the runs moved by (0 for one run). */
  readonly spread: number
}

/**
 * The median of numbers (the mean of the middle two for an even count).
 *
 * @param xs - At least one number.
 * @returns The median.
 */
export function median(xs: readonly number[]): number {
  if (xs.length === 0) throw new Error('median of nothing')
  const s = [...xs].sort((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 === 1 ? (s[mid] as number) : ((s[mid - 1] as number) + (s[mid] as number)) / 2
}

/**
 * The median, range and relative spread of repeated measurements.
 *
 * @param xs - The measurements.
 * @returns The summary.
 */
export function summarise(xs: readonly number[]): Stat {
  const m = median(xs)
  const min = Math.min(...xs)
  const max = Math.max(...xs)
  return { median: m, min, max, n: xs.length, spread: m > 0 ? (max - min) / m : 0 }
}

/**
 * The calibration figure of a run: the mean of the medians of the calibration workload timed at the start and at the end (so a load
 * that came or went during the run shows in the two and is averaged). Milliseconds of the fixed workload: lower is a faster or an
 * idler machine.
 *
 * @param start - The timings of the workload before the measurements.
 * @param end - The timings after.
 * @returns The figure and the drift between the two ends (`end / start`, 1 when the machine was steady).
 */
export function calibrationOf(
  start: readonly number[],
  end: readonly number[],
): { readonly ms: number; readonly drift: number } {
  const a = median(start)
  const b = median(end)
  return { ms: (a + b) / 2, drift: b / a }
}

/**
 * The factor by which a time measured now is to be scaled to be compared with the baseline's: a machine that runs the calibration
 * workload 1.3 times slower is expected to run everything 1.3 times slower, so a time of 1.3 times the baseline is no regression.
 *
 * @param nowMs - The calibration figure of this run.
 * @param baseMs - The calibration figure of the baseline.
 * @returns `nowMs / baseMs`.
 */
export const machineFactor = (nowMs: number, baseMs: number): number => nowMs / baseMs

/** One comparison of a fresh time with the baseline's. */
export interface Comparison {
  readonly key: string
  readonly base: number
  readonly now: number
  /** `now / base / factor`: above 1, slower than the baseline after normalising for the machine. */
  readonly ratio: number
  readonly slower: boolean
}

/**
 * Compare fresh timings with the baseline's, normalised by the calibration, and flag the ones slower than `ratio` times the baseline.
 *
 * @param base - The baseline's median by key.
 * @param now - The fresh medians by key (keys the baseline lacks are left out; keys only the baseline has are left out too).
 * @param factor - `machineFactor` of this run against the baseline's.
 * @param ratio - How much slower than the baseline (after the normalisation) fails: 1.5 by default.
 * @returns The comparisons, the slowest ratio first.
 */
export function compareTimes(
  base: ReadonlyMap<string, number>,
  now: ReadonlyMap<string, number>,
  factor: number,
  ratio = 1.5,
): Comparison[] {
  const out: Comparison[] = []
  for (const [key, t] of now) {
    const b = base.get(key)
    if (b === undefined || !(b > 0)) continue
    const r = t / b / factor
    out.push({ key, base: b, now: t, ratio: r, slower: r > ratio })
  }
  return out.sort((x, y) => y.ratio - x.ratio)
}

/** Whether a key is chosen by `--only` (a prefix of the key, or of what follows the first slash). */
export const chosenKey = (only: readonly string[] | undefined, key: string): boolean =>
  only === undefined ||
  only.some((o) => key === o || key.startsWith(o) || key.slice(key.indexOf('/') + 1).startsWith(o))

/**
 * A duration in the unit that reads best.
 *
 * @param us - Microseconds.
 * @returns For example `3.2 us`, `41 ms`, `2.31 s`.
 */
export function formatMicros(us: number): string {
  if (!Number.isFinite(us)) return String(us)
  if (us < 10) return `${us.toFixed(2)} us`
  if (us < 1000) return `${us.toFixed(1)} us`
  if (us < 1e4) return `${(us / 1000).toFixed(2)} ms`
  if (us < 1e6) return `${(us / 1000).toFixed(1)} ms`
  return `${(us / 1e6).toFixed(2)} s`
}

/** A fraction as a percentage with one decimal. */
export const percent = (x: number): string => `${(100 * x).toFixed(1)}%`

/**
 * A markdown table.
 *
 * @param head - The column titles.
 * @param rows - The rows (cells as text).
 * @returns The table text, with the numeric-looking columns left as they come.
 */
export function table(head: readonly string[], rows: readonly (readonly string[])[]): string {
  const line = (cells: readonly string[]): string => `| ${cells.join(' | ')} |`
  return [line(head), line(head.map(() => '---')), ...rows.map(line)].join('\n')
}
