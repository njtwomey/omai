/**
 * The process-wide default context (seed 0 unless replaced), used when a function is given neither a key nor a context;
 * `setDefaultContext` replaces it, typically once at application start to set the global seed.
 */

import { type Context, createContext } from './context.ts'

// The one documented piece of module state in omai: the context that convenience layers (class wrappers,
// scripts) use when none is given. Core functions never read it; they take keys or contexts explicitly.
//
// It is process-wide, so it lives on `globalThis` under `Symbol.for('omai-engine/defaultContext')` and not in a module
// variable: a second loaded copy of omai-engine (two bundler chunks, a dependency optimiser's duplicate) then shares
// the one default, and the seed an app sets at start is the seed every copy uses. The holder is made on first use, so
// importing this file reads nothing. (A copy of another version finds the holder as this one left it.)
const SLOT = Symbol.for('omai-engine/defaultContext')
interface Holder {
  current: Context
}
function holder(): Holder {
  const slots = globalThis as unknown as Record<symbol, Holder | undefined>
  const made = slots[SLOT] ?? { current: createContext() }
  slots[SLOT] = made
  return made
}

/**
 * The default context (seed 0 unless replaced with {@link setDefaultContext}), shared by every loaded copy of omai-engine.
 *
 * @returns The default context.
 * @concept context
 * @example
 * ```ts
 * import { defaultContext } from 'omai-engine/context'
 * defaultContext().kind // => 'context'
 * ```
 */
export function defaultContext(): Context {
  return holder().current
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
 * import { createContext, defaultContext, setDefaultContext } from 'omai-engine/context'
 * const previous = setDefaultContext(createContext({ seed: 7 }))
 * defaultContext().seed // => 7
 * setDefaultContext(previous).seed // => 7
 * ```
 */
export function setDefaultContext(ctx: Context): Context {
  const slot = holder()
  const previous = slot.current
  slot.current = ctx
  return previous
}
