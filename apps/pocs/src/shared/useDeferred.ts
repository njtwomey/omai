import { useEffect, useState } from 'react'

/**
 * Computes something heavy after the page has been painted once, so the page shows what it is waiting for instead of
 * a blank while the thread is busy: undefined until it is there, and again whenever `compute` changes (so give it a
 * stable identity: a module-level function or a `useCallback`).
 *
 * @param compute - The work (it blocks the thread while it runs).
 * @returns The result, or undefined while it is being made.
 */
export function useDeferred<T>(compute: () => T): T | undefined {
  const [value, setValue] = useState<T | undefined>(undefined)
  useEffect(() => {
    setValue(undefined)
    const timer = setTimeout(() => setValue(compute()), 60)
    return () => clearTimeout(timer)
  }, [compute])
  return value
}
