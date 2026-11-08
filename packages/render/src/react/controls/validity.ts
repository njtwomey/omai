/**
 * Validity greying from the option tables: the properties a table records for each option (`smooth`, `hasProx`,
 * `likelihood`, `convex`) say which options do not apply to the choices already made, and a rule built here turns
 * them into the `admissible` and `why` the controls take, so an option that cannot be chosen is drawn disabled with
 * the reason as its title, not hidden.
 */

import type { Json } from 'omai-engine/base'
import { defaults, type OptionTable, type Space, type SpaceValue } from 'omai-engine/space'
import type { ModeTable, SlotValues } from 'omai-engine/trace'
import type { Admissibility, ControlPath } from './space-controls.ts'

/** One slot of a configuration whose options come from a table, with what rules its options out (types erased). */
export interface TableSlot {
  /** The key of the configuration that names the option (the table's tag). */
  readonly tag: string
  /** The options' names, in the table's order. */
  readonly names: readonly string[]
  /** What stands in the way of an option, or `undefined` when nothing does. */
  readonly reason: (option: string) => string | undefined
}

/**
 * A slot from an option table and a rule over the options' properties. The rule may read the page's other choices
 * through its own closure (a solver that needs a proximal operator, given the penalty chosen).
 *
 * @param table - A table of options or modes (`classificationLosses`, `penalties`, `smoothSolvers`).
 * @param reason - Given an option's properties and name, why it does not apply, or `undefined` when it does.
 * @returns The slot.
 */
export function tableSlot<P>(
  table: Pick<OptionTable<unknown, P>, 'tag' | 'names' | 'where'>,
  reason: (properties: P, name: string) => string | undefined,
): TableSlot {
  return {
    tag: table.tag,
    names: table.names,
    reason: (option) => {
      let found: string | undefined
      table.where((properties, name) => {
        if (name === option) found = reason(properties, name)
        return false
      })
      return found
    },
  }
}

/**
 * A slot from a table of modes and the other slots' values: a mode that does not admit them (its `needs`, the mode
 * table's `admissible`) is ruled out, with the reason read off its `needs` ("needs penalty: none or l2").
 *
 * @param table - A table of modes (`smoothSolvers`, `glmSolvers`).
 * @param others - The other slots' current values, by slot (a primitive, or a union value read by its `kind`).
 * @returns The slot.
 */
export function modeSlot<Problem, P extends Json>(table: ModeTable<Problem, P>, others: SlotValues): TableSlot {
  const admitted = new Set(table.admissible(others))
  return {
    tag: table.tag,
    names: table.names,
    reason: (option) => {
      if (admitted.has(option) || !table.names.includes(option)) return undefined
      const needs = Object.entries(table.modeOf({ [table.tag]: option } as Record<string, SpaceValue>).needs ?? {})
      return `needs ${needs.map(([slot, allowed]) => `${slot}: ${allowed.join(' or ')}`).join(', ')}`
    },
  }
}

/**
 * The admissibility rule of the controls of a space whose slots are option tables: the choice at `[key, tag]` (the
 * tag of the table's union, under the slot's top-level key) is ruled by that slot's reason. Controls elsewhere are
 * left as they are. Pass it to `SpaceControls` as `admissible` and `why`.
 *
 * @param slots - The slots by the top-level key of the configuration.
 * @returns The rule.
 */
export function tableRules(slots: Readonly<Record<string, TableSlot>>): Admissibility {
  const slotOf = (path: ControlPath): TableSlot | undefined => {
    const key = path[0]
    const slot = typeof key === 'string' ? slots[key] : undefined
    return slot !== undefined && path.length === 2 && path[1] === slot.tag ? slot : undefined
  }
  return {
    admissible: (path, option) => slotOf(path)?.reason(String(option)) === undefined,
    why: (path, option) => slotOf(path)?.reason(String(option)),
  }
}

/**
 * The names of a slot's options that nothing rules out, in the table's order (for a comparison that runs only the
 * options that apply).
 *
 * @param slot - The slot.
 * @returns The names.
 */
export function admissibleNames(slot: TableSlot): string[] {
  return slot.names.filter((n) => slot.reason(n) === undefined)
}

/**
 * A value moved off the options its slots rule out: where the option a slot's value names does not apply, the slot
 * takes the first option that does, with that option's knobs at their defaults (the rest of the value is kept). Call
 * it when a choice that other slots depend on changes, so the greyed option is never the one in use.
 *
 * @param space - The configuration's space (a dict whose slots are the tables' unions).
 * @param value - The current value.
 * @param slots - The slots by the top-level key of the configuration.
 * @returns The value, unchanged when every slot's option applies (or none does).
 */
export function settleAdmissible(
  space: Space,
  value: SpaceValue,
  slots: Readonly<Record<string, TableSlot>>,
): SpaceValue {
  if (space.kind !== 'dict' || typeof value !== 'object' || value === null || Array.isArray(value)) return value
  const out: Record<string, SpaceValue> = { ...(value as Readonly<Record<string, SpaceValue>>) }
  let moved = false
  for (const [key, slot] of Object.entries(slots)) {
    const slotSpace = space.spaces[key]
    const current = out[key]
    if (slotSpace?.kind !== 'union' || typeof current !== 'object' || current === null || Array.isArray(current))
      continue
    const named = (current as Readonly<Record<string, SpaceValue>>)[slot.tag]
    if (typeof named !== 'string' || slot.reason(named) === undefined) continue
    const next = admissibleNames(slot)[0]
    if (next === undefined) continue
    out[key] = { ...(defaults(slotSpace.options[next] as Space) as object), [slot.tag]: next } as SpaceValue
    moved = true
  }
  return moved ? out : value
}
