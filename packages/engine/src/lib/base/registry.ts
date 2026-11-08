/**
 * Registries of named entries, the shape the dataset and environment registries share: a registry is plain data (its
 * entries in registration order, names unique), `register` returns a new registry with more entries, `lookup` and
 * `find` get one by name, and `matchesText` is the search-box test the registries' queries use. Nothing registers
 * itself on import: the library builds its registries from its catalogues as values, and an app extends one with
 * `register` and keeps the result.
 *
 * @index structure Registries of named entries, the shape dataset and environment registries share.
 */

import { OmaiError } from './errors.ts'

/** What every registry entry has: a name, unique within its registry, to look it up by. */
export interface Named {
  readonly name: string
}

/**
 * A registry: named entries in registration order, with unique names. A registry is a value; extending one with
 * {@link register} gives a new one and leaves the old untouched.
 *
 * @concept registries
 */
export interface Registry<E extends Named> {
  readonly entries: readonly E[]
}

/**
 * A registry of entries, in the order given.
 *
 * @param entries - The entries.
 * @returns The registry.
 * @throws OmaiError with code `value` when two entries share a name.
 * @concept registries
 * @example
 * ```ts
 * import { registry } from 'omai-engine/base'
 * registry([{ name: 'a' }, { name: 'b' }]).entries.length // => 2
 * ```
 */
export function registry<E extends Named>(entries: readonly E[]): Registry<E> {
  const seen = new Set<string>()
  for (const e of entries) {
    if (seen.has(e.name)) throw new OmaiError('value', `registry: the name '${e.name}' is registered twice`)
    seen.add(e.name)
  }
  return { entries: [...entries] }
}

/**
 * A new registry with more entries: a new name is appended, and an entry whose name is already registered replaces
 * the old one in place (so an app can re-title or re-tag a library entry). The given registry is left as it was.
 *
 * @param base - The registry to extend.
 * @param entries - The entries to add.
 * @returns The extended registry.
 * @throws OmaiError with code `value` when two of the added entries share a name.
 * @concept registries
 * @example
 * ```ts
 * import { register, registry } from 'omai-engine/base'
 * const base = registry([{ name: 'a', v: 1 }])
 * const more = register(base, [{ name: 'b', v: 2 }, { name: 'a', v: 3 }])
 * more.entries.map((e) => `${e.name}${e.v}`) // => ['a3', 'b2']
 * base.entries.length // => 1
 * ```
 */
export function register<E extends Named>(base: Registry<E>, entries: readonly E[]): Registry<E> {
  const added = new Set<string>()
  const byName = new Map(base.entries.map((e) => [e.name, e] as const))
  for (const e of entries) {
    if (added.has(e.name)) throw new OmaiError('value', `register: the name '${e.name}' is added twice`)
    added.add(e.name)
    byName.set(e.name, e)
  }
  return { entries: [...byName.values()] }
}

/**
 * The entry of a name, if registered.
 *
 * @param reg - The registry.
 * @param name - The entry's name.
 * @returns The entry, or `undefined`.
 * @concept registries
 * @example
 * ```ts
 * import { find, registry } from 'omai-engine/base'
 * find(registry([{ name: 'a' }]), 'b') // => undefined
 * ```
 */
export function find<E extends Named>(reg: Registry<E>, name: string): E | undefined {
  return reg.entries.find((e) => e.name === name)
}

/**
 * The entry of a name.
 *
 * @param reg - The registry.
 * @param name - The entry's name.
 * @returns The entry.
 * @throws OmaiError with code `value` when nothing of that name is registered.
 * @concept registries
 * @example
 * ```ts
 * import { lookup, registry } from 'omai-engine/base'
 * lookup(registry([{ name: 'a', v: 1 }]), 'a').v // => 1
 * lookup(registry([{ name: 'a', v: 1 }]), 'b') // => throws value
 * ```
 */
export function lookup<E extends Named>(reg: Registry<E>, name: string): E {
  const found = find(reg, name)
  if (found === undefined) throw new OmaiError('value', `lookup: nothing named '${name}' is registered`)
  return found
}

/**
 * The search-box test: whether every word of a query appears, ignoring case, somewhere in an entry's fields. An empty
 * or missing query matches everything.
 *
 * @param text - The query, words separated by spaces.
 * @param fields - The entry's searchable text (missing fields are skipped).
 * @returns Whether the entry matches.
 * @concept registries
 * @example
 * ```ts
 * import { matchesText } from 'omai-engine/base'
 * matchesText('two MOONS', ['moons', 'Two moons', undefined]) // => true
 * matchesText('moons real', ['moons', 'Two moons']) // => false
 * matchesText('', ['anything']) // => true
 * ```
 */
export function matchesText(text: string | undefined, fields: readonly (string | undefined)[]): boolean {
  const words = (text ?? '')
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0)
  if (words.length === 0) return true
  const haystack = fields
    .filter((f): f is string => typeof f === 'string')
    .join('\n')
    .toLowerCase()
  return words.every((w) => haystack.includes(w))
}
