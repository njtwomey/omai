/**
 * `useParams`: the state of a figure's own knobs (sliders, toggles, choices) as one record with a patching `set`, a
 * `reset` to the initial values and a flag that says whether anything moved.
 */

import { useCallback, useMemo, useRef, useState } from 'react'

/** One value of a parameter record: a number, a string, a boolean, or an array of them. */
export type ParamValue = number | string | boolean | readonly (number | string | boolean)[]

/** A record of parameters. */
export type ParamRecord = { readonly [name: string]: ParamValue }

/** What {@link useParams} returns. */
export interface Params<T extends ParamRecord> {
  /** The current values. The same object until something changes. */
  readonly values: T
  /** The values the hook started with (fixed for the life of the component). */
  readonly initial: T
  /** Merge a patch into the values; a value equal to the current one changes nothing. Stable across renders. */
  readonly set: (patch: Partial<T>) => void
  /** Put the named parameters (default all) back to their initial values. Stable across renders. */
  readonly reset: (keys?: readonly (keyof T)[]) => void
  /** True when every value equals its initial value. */
  readonly isDefault: boolean
}

// Equal values: numbers by `===` (so 0 and -0 agree) and also NaN with NaN, anything else by `Object.is`, arrays
// element by element.
function same(a: ParamValue | undefined, b: ParamValue | undefined): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    const [x, y] = [a as readonly unknown[], b as readonly unknown[]]
    return x.length === y.length && x.every((v, i) => v === y[i] || Object.is(v, y[i]))
  }
  return a === b || Object.is(a, b)
}

// The record with the patch merged in, or the record itself when no value in the patch differs.
function merged<T extends ParamRecord>(current: T, patch: Partial<T>): T {
  let next: Record<string, ParamValue> | undefined
  for (const [name, value] of Object.entries(patch) as [string, ParamValue | undefined][]) {
    if (value === undefined || same(current[name], value)) continue
    next ??= { ...current }
    next[name] = value
  }
  return (next ?? current) as T
}

/**
 * The state of a figure's own knobs: the position of a slider, a toggle, a choice, as one record of numbers, strings,
 * booleans and arrays of them. `set` merges a patch, `reset` puts back the initial values (all, or the named ones),
 * and `isDefault` says whether anything has moved, to show a "reset" button only when it is wanted. `set` and `reset`
 * are the same functions on every render, and `values` is the same object until a value really changes (a patch of
 * equal values, arrays included, leaves it alone), so they are safe in effect and memo dependencies.
 *
 * For a figure's own parameters: a noise level, a tilt, a kernel width the page draws itself. Controls generated from
 * an engine `Space` (`SpaceControls`) are for an estimator's hyperparameters and hold their own state; use those when
 * the knobs belong to a method. The initial values are read once, on the first render.
 *
 * @param initial - The starting values.
 * @returns The values, the initial values, `set`, `reset` and `isDefault`.
 * @index function React hook: a figure's knobs as one record with set, reset, isDefault: `useParams({ noise: 0.1 })`
 * @example
 * ```tsx
 * import { useParams } from 'omai-render/react'
 * function Knobs() {
 *   const { values, set, reset, isDefault } = useParams({ noise: 0.1, tilt: 30 })
 *   return (
 *     <>
 *       <input type="range" value={values.noise} onChange={(e) => set({ noise: Number(e.target.value) })} />
 *       {!isDefault && <button onClick={() => reset()}>reset</button>}
 *     </>
 *   )
 * }
 * ```
 */
export function useParams<T extends ParamRecord>(initial: T): Params<T> {
  const first = useRef(initial)
  const [values, setValues] = useState<T>(initial)
  const set = useCallback((patch: Partial<T>): void => setValues((current) => merged(current, patch)), [])
  const reset = useCallback((keys?: readonly (keyof T)[]): void => {
    const start = first.current
    setValues((current) => {
      const names = keys ?? (Object.keys(start) as (keyof T)[])
      return merged(current, Object.fromEntries(names.map((k) => [k, start[k]])) as Partial<T>)
    })
  }, [])
  const isDefault = useMemo(
    () => (Object.keys(first.current) as (keyof T)[]).every((k) => same(values[k], first.current[k])),
    [values],
  )
  return useMemo(() => ({ values, initial: first.current, set, reset, isDefault }), [values, set, reset, isDefault])
}
