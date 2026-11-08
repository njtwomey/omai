/**
 * The main-thread producer: the sessions are real ones, computed by the caller in slices (the hook gives each animation
 * frame a few milliseconds). Nothing here runs by itself; it is the producer that has always been, behind the
 * interface a worker producer shares. Disposing it drops the sessions and their tape; used again afterwards it makes
 * them anew, from step 0.
 */

import type { Json } from 'omai-engine/base'
import type { Config, Derived, Method, SessionOptions } from 'omai-engine/trace'
import { type Made, makeSessions, type Variant } from './make.ts'
import type { Producer } from './producer.ts'
import { advanceInStep } from './step.ts'

/**
 * A producer computing on the main thread.
 *
 * @param method - The method.
 * @param options - The options every variant shares (eager mode computes the sessions whole when they are made).
 * @param variants - The variants.
 * @param fallback - Why this stands in for a worker, when it does.
 * @returns The producer.
 * @index function The main-thread producer that computes in slices: `inlineProducer(method, options, variants)`.
 */
export function inlineProducer<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  variants: readonly Variant<C>[],
  fallback?: string,
): Producer<S, C> {
  let held: Made<S, C> | undefined = makeSessions(method, options, variants)
  const made = (): Made<S, C> => {
    held ??= makeSessions(method, options, variants)
    return held
  }
  const labels = variants.map((v) => v.label)
  return {
    kind: 'inline',
    ...(fallback === undefined ? {} : { fallback }),
    get sessions() {
      return made().sessions
    },
    labels,
    background: false,
    advance: (budget) => advanceInStep(made().sessions, budget),
    start: () => undefined,
    stop: () => undefined,
    follow: () => undefined,
    probeAsync: (variant, t, path, args?: Json): Promise<Derived> => {
      try {
        const s = made().sessions[variant]
        if (s === undefined) throw new Error(`no variant ${variant}`)
        return Promise.resolve(s.probe(t, path, args))
      } catch (e) {
        return Promise.reject(e)
      }
    },
    error: undefined,
    lastMs: 0,
    probed: 0,
    subscribe: () => () => undefined,
    dispose: () => {
      held = undefined
    },
  }
}
