/**
 * What the story fixes before anything runs: what is believed about the room's temperature before it is measured, and how large
 * the error of each of the two instruments is. Nothing here computes; the page draws the prior from it before play, and the
 * computation builds its model from it.
 */

/** The belief about the temperature (degrees Celsius) before any reading: a bell curve with this mean and spread. */
export const PRIOR: { readonly mean: number; readonly sd: number } = { mean: 20, sd: 4 }

/** The spread of the error of the digital thermometer, in degrees. */
export const DIGITAL_SD = 0.5

/** The spread of the error of the dial thermometer, in degrees. */
export const DIAL_SD = 2

/** The names the page uses for the three factors, in the order they send their messages. */
export const SOURCES: readonly ('prior' | 'digital' | 'dial')[] = ['prior', 'digital', 'dial']
