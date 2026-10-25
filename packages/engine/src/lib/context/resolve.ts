/**
 * How high-level functions pick their randomness and tracing: the `ContextOptions` they accept (`key`, `ctx`, tracer,
 * policy) and the resolvers that choose an explicit key first, then a key named from the given context, then one from
 * the default context.
 */

import type { Key } from '../random/index.ts'
import type { TracePolicy, Tracer } from '../trace/index.ts'
import type { Context } from './context.ts'
import { defaultContext } from './default.ts'

/** The options every high-level function accepts for randomness and tracing. */
export interface ContextOptions {
  readonly key?: Key
  readonly ctx?: Context
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/**
 * The key a function should use: the explicit `key`, else `ctx.key(name)`, else the default context's key for the name.
 * This is the one rule for randomness in every high-level API.
 *
 * @param options - The caller's options.
 * @param name - The function's name (its key namespace).
 * @returns The key.
 * @concept context
 * @example
 * ```ts
 * import { resolveKey, createContext } from 'mdml-engine/context'
 * const ctx = createContext({ seed: 3 })
 * resolveKey({ ctx }, 'kmeans').words[0] === ctx.key('kmeans').words[0] // => true
 * ```
 */
export function resolveKey(options: ContextOptions, name: string): Key {
  return options.key ?? (options.ctx ?? defaultContext()).key(name)
}

/**
 * The tracer a function should use: the explicit one, else the context's.
 *
 * @param options - The caller's options.
 * @returns The tracer.
 * @example
 * ```ts
 * import { resolveTracer } from 'mdml-engine/context'
 * resolveTracer({}).enabled // => false
 * ```
 */
export function resolveTracer(options: ContextOptions): Tracer {
  return options.tracer ?? (options.ctx ?? defaultContext()).tracer
}

/**
 * The trace policy a function should use: the explicit one, else the context's.
 *
 * @param options - The caller's options.
 * @returns The policy.
 * @example
 * ```ts
 * import { resolvePolicy } from 'mdml-engine/context'
 * resolvePolicy({}).detail // => 'summary'
 * ```
 */
export function resolvePolicy(options: ContextOptions): TracePolicy {
  return options.policy ?? (options.ctx ?? defaultContext()).policy
}
