/**
 * An option table: the shape every option set shares when its options are values rather than algorithms (a loss, a
 * penalty, a schedule, a kernel). Each option has its own knobs (a dict space), properties the validity rules and the
 * selectors read, and a `make` that binds the knobs into the value; the table is one `union` space over the knobs
 * and the dispatch from a configuration to its option. The modes of `trace/modes.ts` are the same idea for algorithms.
 *
 * @index table optionTable and modeTable: slots filled from named options.
 */

import { MdmlError } from '../base/index.ts'
import { defaults } from './operations.ts'
import type { DictSpace, SpaceValue, UnionSpace } from './types.ts'
import { union } from './types.ts'

/** A mode's or option's knobs as a value: the tag that names it, beside its own knobs. */
export type ModeConfig = Readonly<Record<string, SpaceValue>>

/** The options of a table of modes or options. */
export interface ModeTableOptions {
  /** The key of the configuration that names the option (default `kind`). */
  readonly tag?: string
  /** The option chosen by default (default the first). */
  readonly default?: string
  readonly label?: string
  readonly description?: string
}

/** One option of a table: its knobs, its properties, and the value its knobs make. */
export interface Option<T, Props> {
  readonly description?: string
  readonly space: DictSpace
  readonly properties: Props
  make(knobs: ModeConfig): T
}

/** An option with its name, as the table hands it back. */
export type NamedOption<T, Props> = Option<T, Props> & { readonly name: string }

/** A table of options for one slot: the union space over their knobs and the dispatch from a configuration. */
export interface OptionTable<T, Props> {
  /** The key of a configuration that names the option. */
  readonly tag: string
  /** The options' names, in declaration order. */
  readonly names: readonly string[]
  /** The slot as a space, for controls, grids and tuners. */
  readonly space: UnionSpace
  /** The option a configuration names (the default when it names none). */
  optionOf(config: SpaceValue | undefined): NamedOption<T, Props>
  /** The value the configuration's option makes from its knobs. */
  make(config: SpaceValue | undefined): T
  /** The properties of the configuration's option. */
  properties(config: SpaceValue | undefined): Props
  /** The names of the options whose properties satisfy a predicate (what a validity rule or a selector asks). */
  where(predicate: (properties: Props, name: string) => boolean): string[]
}

/**
 * A table of options for one slot.
 *
 * @param table - The options by name.
 * @param options - The tag, the default option and metadata.
 * @returns The table.
 * @throws MdmlError with code `value` for an empty table or a default that is not an option (as `union` does).
 */
export function optionTable<T, Props>(
  table: Readonly<Record<string, Option<T, Props>>>,
  options: ModeTableOptions = {},
): OptionTable<T, Props> {
  const tag = options.tag ?? 'kind'
  const names = Object.keys(table)
  if (names.length === 0) throw new MdmlError('value', 'optionTable: need at least one option')
  const space = union(Object.fromEntries(names.map((n) => [n, table[n]?.space as DictSpace])), {
    tag,
    ...(options.default === undefined ? {} : { default: options.default }),
    ...(options.label === undefined ? {} : { label: options.label }),
    ...(options.description === undefined ? {} : { description: options.description }),
  })
  const asRecord = (config: SpaceValue | undefined): ModeConfig =>
    typeof config === 'object' && config !== null && !Array.isArray(config)
      ? (config as Readonly<Record<string, SpaceValue>>)
      : {}
  const optionOf = (config: SpaceValue | undefined): NamedOption<T, Props> => {
    const name = asRecord(config)[tag]
    const chosen = typeof name === 'string' && name in table ? name : (options.default ?? (names[0] as string))
    return { ...(table[chosen] as Option<T, Props>), name: chosen }
  }
  return {
    tag,
    names,
    space,
    optionOf,
    make: (config) => optionOf(config).make(asRecord(config)),
    properties: (config) => optionOf(config).properties,
    where: (predicate) => names.filter((n) => predicate((table[n] as Option<T, Props>).properties, n)),
  }
}

/**
 * A configuration of an option table with the knobs it leaves out at their defaults (a union value must be whole): the
 * chosen option's defaults, overridden by what the configuration gives, with the tag naming the option.
 *
 * @param table - The option table.
 * @param config - A configuration, possibly partial (only a name, or only some knobs), or none for the default option.
 * @returns The whole configuration.
 * @example
 * ```ts
 * import { dict, int, optionTable, completeOption } from 'mdml-engine/space'
 * const k = dict({ k: int(0, 9, { default: 3 }) })
 * const table = optionTable({ a: { space: k, properties: null, make: () => 0 } })
 * completeOption(table, { kind: 'a' }) // => { k: 3, kind: 'a' }
 * ```
 */
export function completeOption<T, P>(table: OptionTable<T, P>, config: SpaceValue | undefined): SpaceValue {
  const given = (typeof config === 'object' && config !== null && !Array.isArray(config) ? config : {}) as ModeConfig
  const chosen = table.optionOf(given)
  return { ...(defaults(chosen.space) as ModeConfig), ...given, [table.tag]: chosen.name }
}
