/**
 * The small helpers the atoms of this folder share: a value computed once. Not exported from the
 * folder; a helper that more than one file needs lives here and not in a copy.
 */

/** A value computed the first time it is asked for (a reference grid that a calibration over many seeds never reads). */
export function once<T>(make: () => T): () => T {
  let value: T | undefined
  return () => (value ??= make())
}
