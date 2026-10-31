/**
 * Making the sessions of a comparison: one session per variant, with the shared options and each variant's knobs laid
 * over the shared configuration. The one place it is written, used by the main-thread producer and by the worker, so a
 * session is the same wherever it is computed.
 */

import { type Config, type Method, type Session, type SessionOptions, session } from 'mdml-engine/trace'

/** One variant of a comparison: a label and the knob values that set it apart. */
export interface Variant<C extends Config> {
  readonly label: string
  /** Knob values over the shared options' `config`: what this variant changes, and nothing else. */
  readonly config: Partial<C>
}

/** The sessions of a comparison as made: one per variant, with its label. */
export interface Made<S, C extends Config> {
  readonly labels: readonly string[]
  readonly sessions: readonly Session<S, C>[]
}

/**
 * Make one session per variant of a method.
 *
 * @param method - The method.
 * @param options - The options every variant shares.
 * @param variants - The variants.
 * @returns The sessions with their labels.
 */
export function makeSessions<S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  variants: readonly Variant<C>[],
): Made<S, C> {
  return {
    labels: variants.map((v) => v.label),
    sessions: variants.map((v) =>
      session(method, { ...options, config: { ...options.config, ...v.config } as Partial<C> }),
    ),
  }
}
