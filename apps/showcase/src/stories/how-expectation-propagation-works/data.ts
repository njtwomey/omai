/**
 * The fixed facts of 'How expectation propagation works': the model is one unknown location with a wide prior and `ROWS`
 * measurements, each of which is, with probability `CLUTTER_CHANCE`, a stray value ("clutter") instead of a measurement of
 * the location. Nothing here depends on the seed; the seed only decides the values. Light enough for the page's bundle.
 */

/** How many measurements the example has. */
export const ROWS = 6

/** The chance that a measurement is clutter. */
export const CLUTTER_CHANCE = 0.4

/** The standard deviation of the clutter, and of the prior on the location. */
export const CLUTTER_SD = Math.sqrt(10)
export const PRIOR_SD = 10

/** The tolerance of the run: it stops when no belief moves by more than this, in standard deviations. */
export const TOLERANCE = 1e-6

/** The most sweeps the run is allowed. */
export const MAX_SWEEPS = 60

/** The sweep whose update the page takes apart (the second: every factor already has a stand-in of its own by then). */
export const SHOWN_SWEEP = 2
