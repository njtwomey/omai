import { type Context, createContext } from './context.ts'

// The one documented piece of module state in whitebox-js: the context that convenience layers (class wrappers,
// scripts) use when none is given. Core functions never read it; they take keys or contexts explicitly.
let current: Context = createContext()

/**
 * The default context (seed 0 unless replaced with {@link setDefaultContext}).
 *
 * @returns The default context.
 * @concept context
 * @example
 * ```ts
 * import { defaultContext } from 'whitebox-engine/context'
 * defaultContext().kind // => 'context'
 * ```
 */
export function defaultContext(): Context {
  return current
}

/**
 * Replace the default context, typically once at application start ("set the global seed"). Functions that receive an
 * explicit key or context are unaffected.
 *
 * @param ctx - The new default.
 * @returns The previous default (to restore it).
 * @concept context
 * @example
 * ```ts
 * import { createContext, defaultContext, setDefaultContext } from 'whitebox-engine/context'
 * const previous = setDefaultContext(createContext({ seed: 7 }))
 * defaultContext().seed // => 7
 * setDefaultContext(previous).seed // => 7
 * ```
 */
export function setDefaultContext(ctx: Context): Context {
  const previous = current
  current = ctx
  return previous
}
