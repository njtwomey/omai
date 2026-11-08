/**
 * The method registry both sides of a producer resolve names against: a page hands a worker the name of a method (and
 * a JSON configuration), the worker looks the method up here, and a tape file names its method the same way when it is
 * loaded back. Methods are plain modules, so the registry is plain data an app builds from the ones it offers; there is
 * no import-time registration.
 */

import { OmaiError } from 'omai-engine/base'
import type { Method } from 'omai-engine/trace'

/** A method of any state, configuration and path type, as a registry holds it. */
// biome-ignore lint/suspicious/noExplicitAny: a registry is generic over methods
export type AnyMethod = Method<any, any, any>

/** Methods by name. */
export interface MethodRegistry {
  /** The method called `name`, or `undefined`. */
  get(name: string): AnyMethod | undefined
  /** The method called `name`. @throws OmaiError with code `value` listing the names that exist. */
  need(name: string): AnyMethod
  /** The names, in the order given. */
  names(): readonly string[]
}

/**
 * A registry of methods, looked up by their `name`.
 *
 * @param methods - The methods on offer.
 * @returns The registry.
 * @throws OmaiError with code `value` when two methods share a name.
 * @index function Methods by name for a worker and tape files: `methodRegistry([methodA, methodB])`.
 */
export function methodRegistry(methods: readonly AnyMethod[]): MethodRegistry {
  const byName = new Map<string, AnyMethod>()
  for (const m of methods) {
    if (byName.has(m.name)) throw new OmaiError('value', `methodRegistry: two methods are called '${m.name}'`)
    byName.set(m.name, m)
  }
  const get = (name: string): AnyMethod | undefined => byName.get(name)
  return {
    get,
    need: (name) => {
      const m = get(name)
      if (m === undefined)
        throw new OmaiError('value', `no method called '${name}' (known: ${[...byName.keys()].join(', ') || 'none'})`)
      return m
    },
    names: () => [...byName.keys()],
  }
}
