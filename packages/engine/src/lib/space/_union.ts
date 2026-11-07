// The parts of a tagged union's value, shared by sampling, membership, defaults, encoding, grids and neighbourhoods.
// Not re-exported: the public forms are the operations over any space.
import type { DictSpace, SpaceValue, UnionSpace } from './types.ts'

/** The key under which a union's value names its option. */
export function tagOf(space: UnionSpace): string {
  return space.tag ?? 'kind'
}

/** The option names of a union, in declaration order. */
export function optionNames(space: UnionSpace): string[] {
  return Object.keys(space.options)
}

/** The option a union's value names, when it names one; else the default option. */
export function optionOf(space: UnionSpace, value: SpaceValue | undefined): string {
  const names = optionNames(space)
  const fallback = space.default ?? (names[0] as string)
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return fallback
  const name = (value as Readonly<Record<string, SpaceValue>>)[tagOf(space)]
  return typeof name === 'string' && Object.hasOwn(space.options, name) ? name : fallback
}

/** A union value split into the option it names and the option's own values (everything but the tag). */
export function splitUnion(
  space: UnionSpace,
  value: SpaceValue | undefined,
): { readonly name: string; readonly option: DictSpace; readonly rest: Readonly<Record<string, SpaceValue>> } {
  const name = optionOf(space, value)
  const record =
    typeof value === 'object' && value !== null && !Array.isArray(value)
      ? (value as Readonly<Record<string, SpaceValue>>)
      : {}
  const { [tagOf(space)]: _tag, ...rest } = record
  return { name, option: space.options[name] as DictSpace, rest }
}

/** A union value from an option's name and its values: the tag first, then the option's keys. */
export function joinUnion(space: UnionSpace, name: string, rest: Readonly<Record<string, SpaceValue>>): SpaceValue {
  return { [tagOf(space)]: name, ...rest }
}
