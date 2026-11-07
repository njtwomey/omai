/**
 * An option table: the shape every option set shares when its options are values rather than algorithms (a loss, a
 * penalty, a schedule, a kernel). Each option has its own knobs (a dict space), properties the validity rules and the
 * selectors read, and a `make` that binds the knobs into the value; the table is one `union` space over the knobs
 * and the dispatch from a configuration to its option. It is the one builder of such tables: the modes of `trace/modes.ts`
 * (options whose value builds an algorithm) are made with it.
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

/**
 * A value of a dict or union space as a record of knobs: the value itself when it is a record, an empty record for
 * anything else (a missing value, a number, a list).
 *
 * @param value - A value of a space.
 * @returns The record.
 * @example
 * ```ts
 * import { asModeConfig } from 'mdml-engine/space'
 * asModeConfig({ kind: 'sgd', lr: 0.1 }) // => { kind: 'sgd', lr: 0.1 }
 * asModeConfig(3) // => {}
 * ```
 */
export function asModeConfig(value: SpaceValue | undefined): ModeConfig {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as ModeConfig) : {}
}

/** The knobs of an option read by name; the type argument is what the option's knobs are (numbers by default). */
export type Knobs<V extends SpaceValue = number> = (name: string) => V

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
 * @example
 * ```ts
 * import { dict, optionTable, real } from 'mdml-engine/space'
 * const kernels = optionTable(
 *   {
 *     small: { space: dict({}), properties: { smooth: true }, make: () => 'small' },
 *     wide: { space: dict({ scale: real(0, 1) }), properties: { smooth: false }, make: (k) => `wide ${k.scale}` },
 *   },
 *   { default: 'wide' },
 * )
 * kernels.names // => ['small', 'wide']
 * kernels.make({ kind: 'wide', scale: 0.5 }) // => 'wide 0.5'
 * kernels.where((properties) => properties.smooth) // => ['small']
 * ```
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
  const optionOf = (config: SpaceValue | undefined): NamedOption<T, Props> => {
    const name = asModeConfig(config)[tag]
    const chosen =
      typeof name === 'string' && Object.hasOwn(table, name) ? name : (options.default ?? (names[0] as string))
    return { ...(table[chosen] as Option<T, Props>), name: chosen }
  }
  return {
    tag,
    names,
    space,
    optionOf,
    make: (config) => optionOf(config).make(asModeConfig(config)),
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
  const given = asModeConfig(config)
  const chosen = table.optionOf(given)
  return { ...(defaults(chosen.space) as ModeConfig), ...given, [table.tag]: chosen.name }
}

/**
 * The knobs of a configuration as a whole record: the option's defaults, overridden by what the configuration gives.
 *
 * @param space - The option's knob space.
 * @param config - The configuration, possibly partial.
 * @returns The record of every knob of the space.
 * @example
 * ```ts
 * import { dict, fillKnobs, int } from 'mdml-engine/space'
 * fillKnobs(dict({ k: int(0, 9, { default: 3 }), m: int(0, 9, { default: 4 }) }), { m: 7 }) // => { k: 3, m: 7 }
 * ```
 */
export function fillKnobs(space: DictSpace, config: ModeConfig): ModeConfig {
  return { ...(defaults(space) as ModeConfig), ...config }
}

/**
 * The knobs of a configuration read by name, the option's defaults filling those the configuration leaves out.
 *
 * @param space - The option's knob space.
 * @param config - The configuration, possibly partial.
 * @returns A function from a knob's name to its value.
 * @example
 * ```ts
 * import { dict, int, readKnobs } from 'mdml-engine/space'
 * const k = readKnobs(dict({ k: int(0, 9, { default: 3 }), m: int(0, 9, { default: 4 }) }), { m: 7 })
 * [k('k'), k('m')] // => [3, 7]
 * ```
 */
export function readKnobs<V extends SpaceValue = number>(space: DictSpace, config: ModeConfig): Knobs<V> {
  const full = fillKnobs(space, config)
  return (name) => full[name] as V
}

/**
 * An option whose value is built from its knobs: `make` reads the configuration against the knob space (defaults
 * filling the gaps) and hands the knobs to `build`. The one builder behind the option tables of the algorithms' slots.
 *
 * @param description - What the option is, for a control.
 * @param properties - The properties the validity rules and selectors read.
 * @param space - The option's knob space.
 * @param build - The value the knobs make.
 * @returns The option.
 * @example
 * ```ts
 * import { dict, int, knobOption } from 'mdml-engine/space'
 * const option = knobOption('two', null, dict({ k: int(0, 9, { default: 3 }) }), (k) => 2 * k('k'))
 * [option.make({}), option.make({ k: 5 })] // => [6, 10]
 * ```
 */
export function knobOption<T, Props, V extends SpaceValue = number>(
  description: string,
  properties: Props,
  space: DictSpace,
  build: (k: Knobs<V>) => T,
): Option<T, Props> {
  return { description, properties, space, make: (config) => build(readKnobs<V>(space, config)) }
}

/**
 * An option with no properties (a table that is only selected from, not filtered): `knobOption` with `{}` for them.
 *
 * @param description - What the option is, for a control.
 * @param space - The option's knob space.
 * @param build - The value the knobs make.
 * @returns The option.
 * @example
 * ```ts
 * import { dict, int, plainOption } from 'mdml-engine/space'
 * plainOption('one', dict({ k: int(0, 9, { default: 3 }) }), (k) => k('k') + 1).make({}) // => 4
 * ```
 */
export function plainOption<T, V extends SpaceValue = number>(
  description: string,
  space: DictSpace,
  build: (k: Knobs<V>) => T,
): Option<T, Record<string, never>> {
  return knobOption(description, {}, space, build)
}
