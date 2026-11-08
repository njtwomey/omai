/**
 * Classical compensators designed in the frequency domain: the lead (phase added around a chosen frequency, from the
 * ratio of its pole to its zero), the lag (low-frequency gain raised for steady-state accuracy at the price of a little
 * phase), their series, and the textbook design rules that place them from a plant's margins: where a lead's centre must
 * sit so the new crossover has the phase to spare, and where a lag's corner must sit so its lag is gone by the
 * crossover.
 */

import { OmaiError } from '../../base/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { evalfr, type Lti, series, stabilityMargins, type TransferFunction, tf } from '../../systems/index.ts'
import { item } from '../../tensor/index.ts'

/** The magnitude and phase (degrees) of a single-input single-output system at a frequency. */
function responseAt(sys: Lti, w: number): { gain: number; phase: number } {
  const z = evalfr(sys, { re: 0, im: w })
  const re = item(z.re)
  const im = item(z.im)
  return { gain: Math.hypot(re, im), phase: (Math.atan2(im, re) * 180) / Math.PI }
}

/** Options of {@link leadCompensator}. */
export interface LeadOptions {
  /** The phase to add at the centre, in degrees (0 to 90, exclusive). */
  readonly phase: number
  /** The centre frequency, rad/s: where the added phase is greatest. */
  readonly centre: number
  /** Where the gain is one: at DC (the default: the zero and pole sit round the centre geometrically) or at the centre. */
  readonly unitGain?: 'dc' | 'centre'
}

/** A designed compensator with the numbers that define it. */
export interface Compensator {
  /** The transfer function C(s). */
  readonly system: TransferFunction
  /** The zero (rad/s, as a positive corner frequency). */
  readonly zero: number
  /** The pole (rad/s). */
  readonly pole: number
  /** The ratio of the larger corner to the smaller: the lead's alpha, the lag's beta. */
  readonly ratio: number
  /** The frequency at which a lead adds the most phase, sqrt(zero pole). */
  readonly centre: number
  /** The largest phase the compensator adds (lead) or removes (lag), in degrees. */
  readonly maxPhase: number
}

/**
 * A lead compensator C(s) = (1 + s/z) / (1 + s/p) with p > z, which adds phase between its corners, most at the
 * geometric mean sqrt(z p). To add phase phi at the centre, p / z = alpha = (1 + sin phi) / (1 - sin phi); the corners
 * are z = centre / sqrt(alpha) and p = centre sqrt(alpha). With DC gain 1 its gain at the centre is sqrt(alpha) (the high
 * frequency gain is alpha); `unitGain: 'centre'` scales it to gain 1 there so that it does not move the crossover.
 *
 * @param options - The phase to add, the centre and the gain normalisation.
 * @returns The compensator with its corners.
 * @throws OmaiError with code `value` for a phase outside (0, 90) or a non-positive centre.
 * @concept lead-lag-compensation
 * @example
 * ```ts
 * import { leadCompensator } from 'omai-engine/dynamics/control'
 * const c = leadCompensator({ phase: 45, centre: 10 })
 * [Math.round(c.ratio * 1e6) / 1e6, Math.round(c.zero * 1e4) / 1e4, Math.round(c.pole * 1e4) / 1e4] // => [5.828427, 4.1421, 24.1421]
 * ```
 */
export function leadCompensator(options: LeadOptions): Compensator {
  const { phase, centre } = options
  if (!(phase > 0 && phase < 90)) throw new OmaiError('value', 'leadCompensator: the phase is between 0 and 90 degrees')
  if (!(centre > 0)) throw new OmaiError('value', 'leadCompensator: the centre frequency must be positive')
  const s = Math.sin((phase * Math.PI) / 180)
  const alpha = (1 + s) / (1 - s)
  const zero = centre / Math.sqrt(alpha)
  const pole = centre * Math.sqrt(alpha)
  const k = options.unitGain === 'centre' ? 1 / Math.sqrt(alpha) : 1
  return {
    system: tf([k / zero, k], [1 / pole, 1]),
    zero,
    pole,
    ratio: alpha,
    centre,
    maxPhase: phase,
  }
}

/** Options of {@link lagCompensator}. */
export interface LagOptions {
  /** The zero, rad/s: the corner below which the gain is boosted. Put it a decade or so below the crossover. */
  readonly zero: number
  /** The factor by which the low-frequency gain exceeds the high-frequency gain (beta > 1). */
  readonly ratio: number
  /** Which gain is one: the high-frequency one (default: the crossover is left alone, the DC gain is multiplied by `ratio`) or the DC one. */
  readonly unitGain?: 'high' | 'dc'
}

/**
 * A lag compensator C(s) = beta (1 + s/z) / (1 + beta s/z), pole p = z / beta below the zero z: it multiplies the
 * low-frequency gain (and so the error constants: the velocity constant of a type-1 loop) by beta while leaving the gain
 * above z at one, at the price of phase lag between p and z (at most asin((beta - 1) / (beta + 1)) at sqrt(z p)).
 * `unitGain: 'dc'` scales it to attenuate instead, DC gain 1 and high-frequency gain 1 / beta.
 *
 * @param options - The zero, the ratio and the gain normalisation.
 * @returns The compensator with its corners.
 * @throws OmaiError with code `value` for a ratio of at most 1 or a non-positive zero.
 * @concept lead-lag-compensation
 * @example
 * ```ts
 * import { lagCompensator } from 'omai-engine/dynamics/control'
 * const c = lagCompensator({ zero: 0.5, ratio: 10 })
 * [c.pole, Math.round(c.maxPhase * 1e4) / 1e4] // => [0.05, 54.9032]
 * ```
 */
export function lagCompensator(options: LagOptions): Compensator {
  const { zero, ratio } = options
  if (!(zero > 0)) throw new OmaiError('value', 'lagCompensator: the zero must be positive')
  if (!(ratio > 1)) throw new OmaiError('value', 'lagCompensator: the ratio must exceed 1')
  const pole = zero / ratio
  const k = options.unitGain === 'dc' ? 1 : ratio
  return {
    system: tf([k / zero, k], [1 / pole, 1]),
    zero,
    pole,
    ratio,
    centre: Math.sqrt(zero * pole),
    maxPhase: (Math.asin((ratio - 1) / (ratio + 1)) * 180) / Math.PI,
  }
}

/**
 * The series of compensators, C = C1 C2 ...: a lead-lag, typically.
 *
 * @param compensators - The compensators, applied in order.
 * @returns The transfer function of their product.
 * @concept lead-lag-compensation
 * @example
 * ```ts
 * import { lagCompensator, leadCompensator, leadLag } from 'omai-engine/dynamics/control'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(leadLag(leadCompensator({ phase: 30, centre: 5 }), lagCompensator({ zero: 0.5, ratio: 4 })).den).length // => 3
 * ```
 */
export function leadLag(...compensators: readonly Compensator[]): TransferFunction {
  if (compensators.length === 0) throw new OmaiError('value', 'leadLag: give at least one compensator')
  return series(...compensators.map((c) => c.system))
}

/** A lead design: the compensator, why it is where it is, and what it did to the loop. */
export interface LeadDesign extends Compensator {
  /** The phase margin (degrees) and gain crossover of the plant alone. */
  readonly before: { readonly pm: number; readonly wgc: number }
  /** The phase margin and crossover of the compensated loop. */
  readonly after: { readonly pm: number; readonly wgc: number }
  /** The phase the lead was asked to add (the shortfall plus the allowance, degrees). */
  readonly requested: number
  /** The compensated loop C(s) P(s). */
  readonly loop: Lti
}

/** Options of {@link leadDesign}. */
export interface LeadDesignOptions {
  /** The phase margin wanted, degrees. */
  readonly phaseMargin: number
  /** Extra phase added to allow for the crossover moving right, where the plant has less phase (default 5 degrees). */
  readonly allowance?: number
}

/**
 * The textbook lead design for a phase margin. The lead must add the shortfall of the margin (plus an allowance for the
 * crossover moving right, where the plant has less phase), so alpha follows from the phase; and its centre must sit
 * where the plant's gain is 1 / sqrt(alpha), because the lead (unit DC gain) will raise the gain there by sqrt(alpha) and
 * make that the new crossover, where the lead's phase is greatest. The centre is found by root-finding on the plant's
 * gain. The loop is C(s) P(s) with P the plant (including any gain already chosen for the error constant).
 *
 * @param plant - The open-loop plant (single input, single output) with a gain crossover.
 * @param options - The margin wanted and the allowance.
 * @returns The lead with its centre, the margins before and after, and the loop.
 * @throws OmaiError with code `value` when the plant has no gain crossover, the margin is already met, or no frequency has the needed gain.
 * @concept lead-lag-compensation
 * @example
 * ```ts
 * import { leadDesign } from 'omai-engine/dynamics/control'
 * import { tf } from 'omai-engine/systems'
 * // 4 / (s (s + 2)) has a phase margin of 51.8 degrees; a lead for 65 (the crossover moves right, so the margin lands near it).
 * const d = leadDesign(tf([4], [1, 2, 0]), { phaseMargin: 65 })
 * [Math.round(d.before.pm), d.after.pm > 62 && d.after.pm < 67] // => [52, true]
 * ```
 */
export function leadDesign(plant: Lti, options: LeadDesignOptions): LeadDesign {
  const m = stabilityMargins(plant)
  if (!Number.isFinite(m.pm) || !(m.wgc > 0))
    throw new OmaiError('value', 'leadDesign: the plant has no gain crossover')
  const requested = options.phaseMargin - m.pm + (options.allowance ?? 5)
  if (!(requested > 0)) throw new OmaiError('value', 'leadDesign: the plant already has the phase margin wanted')
  if (requested >= 89) throw new OmaiError('value', 'leadDesign: one lead cannot add that much phase; use two')
  const s = Math.sin((requested * Math.PI) / 180)
  const alpha = (1 + s) / (1 - s)
  // ln|P(j w)| = -ln(alpha) / 2, searched in log frequency around the old crossover.
  const target = -Math.log(alpha) / 2
  const f = (lw: number): number => Math.log(responseAt(plant, 10 ** lw).gain) - target
  const centre = bracketRoot(f, Math.log10(m.wgc))
  const lead = leadCompensator({ phase: requested, centre })
  const loop = series(lead.system, plant)
  const after = stabilityMargins(loop)
  return {
    ...lead,
    before: { pm: m.pm, wgc: m.wgc },
    after: { pm: after.pm, wgc: after.wgc },
    requested,
    loop,
  }
}

/** A root of f(log10 w) near a starting exponent, bracketing outwards by decades before calling Brent. */
function bracketRoot(f: (lw: number) => number, lw0: number): number {
  const f0 = f(lw0)
  if (f0 === 0) return 10 ** lw0
  for (let span = 0.25; span <= 6; span *= 2) {
    for (const [a, b] of [
      [lw0, lw0 + span],
      [lw0 - span, lw0],
    ] as const) {
      if (f(a) * f(b) <= 0) return 10 ** brentq(f, a, b, { xtol: 1e-12 }).root
    }
  }
  throw new OmaiError('value', 'leadDesign: the plant never has the gain the lead needs')
}

/** A lag design: the compensator and what it did to the loop. */
export interface LagDesign extends Compensator {
  /** The phase margin (degrees) and gain crossover of the plant alone. */
  readonly before: { readonly pm: number; readonly wgc: number }
  /** The phase margin and crossover of the compensated loop. */
  readonly after: { readonly pm: number; readonly wgc: number }
  /** The compensator's phase (degrees, negative) at the plant's crossover. */
  readonly phaseAtCrossover: number
  /** The compensated loop. */
  readonly loop: Lti
}

/** Options of {@link lagDesign}. */
export interface LagDesignOptions {
  /** The factor by which to raise the low-frequency gain (the error constant), beta > 1. */
  readonly ratio: number
  /** How many decades below the gain crossover to put the zero (default 1: the lag has gone by the crossover). */
  readonly decades?: number
}

/**
 * The textbook lag design for steady-state accuracy: the zero a decade (or `decades`) below the plant's gain crossover so
 * that the lag's phase is nearly gone there, and the pole `ratio` times lower, which multiplies the DC gain (and the
 * error constants) by `ratio` without changing the loop gain above the zero. The phase it costs at the crossover is
 * reported, and the margin after.
 *
 * @param plant - The open-loop plant with a gain crossover.
 * @param options - The ratio and the placement.
 * @returns The lag with its corners, its phase at the crossover, and the margins before and after.
 * @throws OmaiError with code `value` when the plant has no gain crossover.
 * @concept lead-lag-compensation
 * @example
 * ```ts
 * import { lagDesign } from 'omai-engine/dynamics/control'
 * import { tf } from 'omai-engine/systems'
 * const d = lagDesign(tf([2], [1, 3, 2, 0]), { ratio: 10 })
 * [d.phaseAtCrossover > -10, Math.abs(d.after.pm - d.before.pm) < 10] // => [true, true]
 * ```
 */
export function lagDesign(plant: Lti, options: LagDesignOptions): LagDesign {
  const m = stabilityMargins(plant)
  if (!Number.isFinite(m.pm) || !(m.wgc > 0)) throw new OmaiError('value', 'lagDesign: the plant has no gain crossover')
  const lag = lagCompensator({ zero: m.wgc / 10 ** (options.decades ?? 1), ratio: options.ratio })
  const loop = series(lag.system, plant)
  const after = stabilityMargins(loop)
  return {
    ...lag,
    before: { pm: m.pm, wgc: m.wgc },
    after: { pm: after.pm, wgc: after.wgc },
    phaseAtCrossover: responseAt(lag.system, m.wgc).phase,
    loop,
  }
}
