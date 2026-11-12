/**
 * What the coin story fixes before anything runs: how many flips there are, after how many of them the page stops to look at
 * the posterior, and what counts as 'close enough to fair'. Nothing here computes; the page uses it to draw an empty frame
 * before play, and the computation uses it to know where to look.
 */

/** The flips the coin is tossed in all. */
export const FLIPS = 100

/** The numbers of flips after which the belief is read off: none at all, then more and more. */
export const STAGES: readonly number[] = [0, 1, 3, 10, 30, 100]

/** The bias counts as fair when it lies between these two (a coin that lands heads 45 to 55 times in 100, in the long run). */
export const TOLERANCE: readonly [number, number] = [0.45, 0.55]
