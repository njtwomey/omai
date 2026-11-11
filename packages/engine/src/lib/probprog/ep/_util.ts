/**
 * The small helpers the atoms of this folder share: a plain sum and a value computed once. Not exported from the
 * folder; a helper that more than one file needs lives here and not in a copy.
 */

/** The sum of a list of numbers, left to right. */
export const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/** A value computed the first time it is asked for (a reference grid that a calibration over many seeds never reads). */
export function once<T>(make: () => T): () => T {
  let value: T | undefined
  return () => (value ??= make())
}
