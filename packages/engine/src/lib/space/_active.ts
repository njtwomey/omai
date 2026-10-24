// Conditional dimensions of dict spaces, shared by sampling, defaults, encoding, grids and neighbourhoods. Not
// re-exported: the public form is `activeDims` in operations.ts.
import type { DictSpace, Space, SpaceValue } from './types.ts'

/** Whether a child space's condition holds for the values chosen so far. */
export function conditionHolds(s: Space, chosen: Readonly<Record<string, SpaceValue>>): boolean {
  if (s.when === undefined) return true
  if (!(s.when.key in chosen)) return false
  return s.when.in.includes(chosen[s.when.key] as string | number | boolean)
}

/**
 * Fill a dict value key by key: unconditional keys first, then conditional keys whose condition holds on the keys
 * chosen so far, repeated until nothing changes (so conditions may depend on conditional keys). `choose` gives a key's
 * value. The result lists keys in the space's order.
 */
export function resolveDict(
  space: DictSpace,
  choose: (key: string, s: Space, index: number) => SpaceValue,
): Record<string, SpaceValue> {
  const entries = Object.entries(space.spaces)
  const chosen: Record<string, SpaceValue> = {}
  let changed = true
  while (changed) {
    changed = false
    for (const [i, [key, s]] of entries.entries()) {
      if (key in chosen || !conditionHolds(s, chosen)) continue
      chosen[key] = choose(key, s, i)
      changed = true
    }
  }
  const out: Record<string, SpaceValue> = {}
  for (const [key] of entries) if (key in chosen) out[key] = chosen[key] as SpaceValue
  return out
}

/**
 * The keys of a dict space that are active for a value: their conditions hold on the value's own active keys. A key
 * can be active and still missing from the value (which then does not belong to the space).
 */
export function activeKeysOf(space: DictSpace, value: Readonly<Record<string, SpaceValue>>): string[] {
  return Object.keys(resolveDict(space, (key) => (key in value ? (value[key] as SpaceValue) : Number.NaN)))
}
