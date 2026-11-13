/**
 * What the prose of 'Counts with too many zeros' says: how many counts are zero, which of the zeros the model gave to the
 * switch and how many really were structural, what the gate said to the switch of the lowest and the highest zero, how the belief
 * about the chance narrowed, how the intercept compares with and without the switch, and how EP compares with the library's
 * exact posterior. Pure functions of the result, light enough for the page's bundle.
 */

import type { Moments, Result } from './compute.ts'
import { TRUTH } from './data.ts'

/** The chance above which a zero is called structural when the page needs one answer for each zero. */
export const CALL = 0.5

/** The two chances that separate the colours of the zeros in the figure: under the first, between, over the second. */
export const BANDS = [0.4, 0.7] as const

/** The logistic function: log-odds to a chance. */
export const logistic = (logOdds: number): number => 1 / (1 + Math.exp(-logOdds))

/** A zero with what the model said about it. */
export interface Zero {
  /** The index of the count. */
  readonly index: number
  readonly x: number
  /** The chance of being structural. */
  readonly chance: number
  /** The log-odds of the gate's message to its switch and of the message from the chance's factor. */
  readonly gate: number
  readonly prior: number
  /** Whether the zero really was structural. */
  readonly structural: boolean
}

/** What the page reads off one result about the zeros. */
export interface ZeroReading {
  /** Every zero, in the order of x. */
  readonly zeros: readonly Zero[]
  /** The counts above zero. */
  readonly positives: number
  /** How many zeros really were structural. */
  readonly structural: number
  /** The chances of the zeros added up: the number of zeros the model expects to be structural. */
  readonly expected: number
  /** The zeros with a chance over {@link CALL}, and how many of them really were structural. */
  readonly called: number
  readonly calledRight: number
  /** The structural zeros the model did not call. */
  readonly missed: number
  /** The zeros in each band of {@link BANDS}: low, middle, high, and how many of each really were structural. */
  readonly bands: readonly [number, number, number]
  readonly bandsStructural: readonly [number, number, number]
  /** The lowest chance any zero was given. */
  readonly lowest: number
}

/**
 * The zeros of a result with what the model said about each.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readZeros(result: Result): ZeroReading {
  const zeros: Zero[] = result.y
    .map((y, index) => ({
      index,
      y,
      x: result.x[index] as number,
      chance: result.chance[index] as number,
      gate: result.gate[index] as number,
      prior: result.prior[index] as number,
      structural: result.structural[index] === 1,
    }))
    .filter((r) => r.y === 0)
    .sort((a, b) => a.x - b.x)
    .map(({ y: _y, ...zero }) => zero)
  const called = zeros.filter((z) => z.chance > CALL)
  const inBand = (low: number, high: number): readonly Zero[] => zeros.filter((z) => z.chance >= low && z.chance < high)
  const [low, middle, high] = [inBand(0, BANDS[0]), inBand(BANDS[0], BANDS[1]), inBand(BANDS[1], 1.01)]
  const real = (band: readonly Zero[]): number => band.filter((z) => z.structural).length
  return {
    zeros,
    positives: result.y.length - zeros.length,
    structural: zeros.filter((z) => z.structural).length,
    expected: zeros.reduce((s, z) => s + z.chance, 0),
    called: called.length,
    calledRight: called.filter((z) => z.structural).length,
    missed: zeros.filter((z) => z.structural && z.chance <= CALL).length,
    bands: [low.length, middle.length, high.length],
    bandsStructural: [real(low), real(middle), real(high)],
    lowest: Math.min(...zeros.map((z) => z.chance)),
  }
}

/** The distance of an estimate from the truth, in its own spreads. */
export const spreadsFrom = (belief: Moments, truth: number): number => Math.abs(belief.mean - truth) / belief.sd

/** What the page reads off one result about the beliefs. */
export interface BeliefReading {
  /** The spread of the belief about the chance after the first and after the last number of counts. */
  readonly firstSd: number
  readonly lastSd: number
  /** After how many of the numbers of counts the nine-tenths range held the true chance. */
  readonly inside: number
  readonly sizes: number
  /** The intercept's distance from the truth, in spreads, with the switch and without it. */
  readonly withSwitch: number
  readonly withoutSwitch: number
  /** The slope's distance from the truth, with the switch and without it. */
  readonly slopeWith: number
  readonly slopeWithout: number
  /** The ratio of EP's spread to the exact one for the chance, the intercept and the slope: smallest and largest. */
  readonly sdRatio: readonly [number, number]
  /** The largest difference between EP's mean and the exact mean, in exact spreads. */
  readonly meanGap: number
  /** The distance of the exact mean of the chance, the intercept and the slope from the truth, in exact spreads: the largest. */
  readonly exactFromTruth: number
}

/**
 * The numbers of the beliefs the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readBeliefs(result: Result): BeliefReading {
  const first = result.beliefs[0]
  const last = result.beliefs.at(-1)
  if (first === undefined || last === undefined) throw new Error('the result has no beliefs')
  const { exact, withSwitch: w, withoutSwitch: wo } = result
  const ep: readonly [Moments, Moments, Moments] = [last, w.b0, w.b1]
  const reference: readonly [Moments, Moments, Moments] = [exact.pi, exact.b0, exact.b1]
  const truth = [TRUTH.inflation, TRUTH.intercept, TRUTH.slope] as const
  const ratios = ep.map((m, i) => m.sd / (reference[i] as Moments).sd)
  return {
    firstSd: first.sd,
    lastSd: last.sd,
    inside: result.beliefs.filter((b) => b.lower <= TRUTH.inflation && TRUTH.inflation <= b.upper).length,
    sizes: result.beliefs.length,
    withSwitch: spreadsFrom(w.b0, TRUTH.intercept),
    withoutSwitch: spreadsFrom(wo.b0, TRUTH.intercept),
    slopeWith: spreadsFrom(w.b1, TRUTH.slope),
    slopeWithout: spreadsFrom(wo.b1, TRUTH.slope),
    sdRatio: [Math.min(...ratios), Math.max(...ratios)],
    meanGap: Math.max(
      ...ep.map((m, i) => Math.abs(m.mean - (reference[i] as Moments).mean) / (reference[i] as Moments).sd),
    ),
    exactFromTruth: Math.max(...reference.map((m, i) => spreadsFrom(m, truth[i] as number))),
  }
}

/** The rate the model's coefficients give at a covariate value. */
export const rateAt = (b: { readonly b0: Moments; readonly b1: Moments }, x: number): number =>
  Math.exp(b.b0.mean + b.b1.mean * x)

/** The rate the data were made from, at a covariate value. */
export const trueRateAt = (x: number): number => Math.exp(TRUTH.intercept + TRUTH.slope * x)

/**
 * Odds in words.
 *
 * @param logOdds - The log-odds.
 * @returns The odds as 'a to 1', or '1 to a' when they are below even.
 * @example
 * ```ts
 * oddsText(Math.log(48)) // => '48 to 1'
 * oddsText(Math.log(0.5)) // => '1 to 2.0'
 * ```
 */
export function oddsText(logOdds: number): string {
  const odds = Math.exp(logOdds)
  if (odds >= 10) return `${Math.round(odds)} to 1`
  return odds >= 1 ? `${odds.toFixed(1)} to 1` : `1 to ${(1 / odds).toFixed(1)}`
}
