/**
 * What the prose of 'How expectation propagation works' says: numbers read off the result (the measurements, the one update taken
 * apart, the sweeps, the stand-ins, the final answer against the exact one), and the few comparisons the text makes (does the
 * answer match the exact one, how many humps does the exact posterior have), each worked out from the data so the text stays
 * true whatever the seed gave. Pure functions, light enough for the page's bundle.
 */

import type { Bell, Message, Result } from './compute.ts'

/** A number to a fixed number of decimals (a minus sign, not a hyphen). */
export const fixed = (v: number, digits = 2): string => v.toFixed(digits).replace('-', '−')

/** A number in scientific form for very small values, such as `3 \u00d7 10\u207b\u2077`. */
export const sci = (v: number): string => {
  const [m, e] = v.toExponential(0).split('e')
  const sup: Record<string, string> = {
    '-': '\u207b',
    '0': '\u2070',
    '1': '\u00b9',
    '2': '\u00b2',
    '3': '\u00b3',
    '4': '\u2074',
    '5': '\u2075',
    '6': '\u2076',
    '7': '\u2077',
    '8': '\u2078',
    '9': '\u2079',
  }
  return `${m} \u00d7 10${[...(e as string).replace('+', '')].map((c) => sup[c] ?? c).join('')}`
}

/** A bell as `mean ± sd`, or a dash for a message that is not a bell. */
export const bellText = (b: Bell | null): string => (b === null ? '—' : `${fixed(b.mean)} ± ${fixed(b.sd)}`)

/** The mean of a message, or undefined when its precision is not positive. */
export const meanOf = (m: Message): number | undefined => (m.precision > 0 ? m.shift / m.precision : undefined)

/** A message as `mean (precision)` for a table: with a precision of zero it says nothing. */
export function messageText(m: Message): string {
  if (m.precision === 0 && m.shift === 0) return 'flat'
  const mean = m.precision === 0 ? undefined : m.shift / m.precision
  const digits = Math.abs(m.precision) < 0.1 ? 3 : 2
  return `${mean === undefined ? '—' : fixed(mean)} (${fixed(m.precision, digits)})`
}

/** The numbers about the exact posterior. */
export interface ExactReading {
  /** How many separate humps the exact posterior has (local maxima above a twentieth of its peak). */
  readonly humps: number
  /** The distance between EP's mean and the exact mean, in exact standard deviations. */
  readonly meanGap: number
  /** EP's standard deviation over the exact one. */
  readonly spreadRatio: number
  /** Whether EP's answer is within a tenth of an exact standard deviation in the mean and ten per cent in the spread. */
  readonly close: boolean
}

/**
 * Compare the EP answer with the exact posterior.
 *
 * @param result - The result.
 * @returns The comparison.
 */
export function readExact(result: Result): ExactReading {
  const d = result.exact.density
  const peak = Math.max(...d)
  const humps = d.filter(
    (v, i) => i > 0 && i < d.length - 1 && v > (d[i - 1] as number) && v >= (d[i + 1] as number) && v > 0.05 * peak,
  ).length
  const meanGap = Math.abs(result.ep.mean - result.exact.mean) / result.exact.sd
  const spreadRatio = result.ep.sd / result.exact.sd
  return { humps: Math.max(1, humps), meanGap, spreadRatio, close: meanGap < 0.1 && Math.abs(spreadRatio - 1) < 0.1 }
}

/** The numbers about the run's sweeps. */
export interface SweepReading {
  /** The belief after the first sweep and after the last. */
  readonly first: Bell | null
  readonly last: Bell | null
  /** The largest change of a belief over the last sweep, in standard deviations. */
  readonly lastChange: number
  /** How many stand-ins have a precision that is not positive at the end (they are not bells). */
  readonly notBells: number
  /** Whether the answer ended with a proper belief. */
  readonly proper: boolean
}

/**
 * The numbers of the sweeps.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readSweeps(result: Result): SweepReading {
  const at = (i: number): Bell | null => {
    const mean = result.beliefs.mean[i]
    const sd = result.beliefs.sd[i]
    return mean === undefined || sd === undefined || !Number.isFinite(mean) || !Number.isFinite(sd)
      ? null
      : { mean, sd }
  }
  const lastSet = result.standIns.at(-1)
  return {
    first: at(0),
    last: at(result.beliefs.sweep.length - 1),
    lastChange: result.beliefs.change.at(-1) ?? Number.NaN,
    notBells: lastSet === undefined ? 0 : lastSet.messages.filter((m, i) => i > 0 && !(m.precision > 0)).length,
    proper: Number.isFinite(result.ep.mean) && Number.isFinite(result.ep.sd),
  }
}

/** The positions (1 to rows) of the measurements the generator made clutter. */
export const clutterPositions = (result: Result): number[] => result.clutter.flatMap((c, i) => (c ? [i + 1] : []))

/** `1, 2 and 5`: a list in words. */
export function listOf(items: readonly (string | number)[]): string {
  if (items.length <= 1) return items.join('')
  return `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`
}

/**
 * The measurements (1 to rows) whose stand-in ends with a precision far below the others': near zero or negative, so they say
 * almost nothing about the location. "Far below" is a quarter of the median precision of the measurements' stand-ins. Nothing
 * is said of a run that did not settle.
 *
 * @param result - The result.
 * @returns The positions of the weak ones, in order.
 */
export function weakOnes(result: Result): number[] {
  const last = result.standIns.at(-1)
  if (last === undefined || !result.settled) return []
  const measured = last.messages.slice(1).map((m) => m.precision)
  const sorted = measured.filter((p) => p > 0).sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  if (median === undefined) return []
  return measured.flatMap((p, i) => (p < 0.25 * median ? [i + 1] : []))
}
