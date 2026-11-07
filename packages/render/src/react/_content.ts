// Internal: a value held by its content, for the hooks that take an options or arguments object a caller rebuilds on every render.

import { useMemo } from 'react'

/**
 * The value, held steady while its JSON text is unchanged: the first object with some content is returned again for
 * every later object with the same content, so a `useMemo` that depends on it recomputes when the content changes and
 * not on every render that rebuilds an equal object.
 *
 * @param value - A JSON value (arguments, knobs, options), or `undefined`.
 * @returns The value of the first render with this content.
 */
export function useByContent<T>(value: T): T {
  const key = JSON.stringify(value)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the point is to hold the value by its text, not its identity
  return useMemo(() => value, [key])
}
