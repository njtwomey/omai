import { isTensor, type Value } from '../tensor/index.ts'
import type { TensorRole } from './events.ts'

/** A collector for intermediate values that model code exposes by name. */
export interface TapContext {
  readonly enabled: boolean
  readonly entries: ReadonlyMap<string, { readonly role: TensorRole; readonly value: Value }>
  tap(name: string, value: Value, role?: TensorRole): Value
}

/**
 * Create a tap context. Model code calls `taps.tap('layer1', h)` at points of interest; when the context is enabled the
 * concrete values are collected (for activations in traces, probes, attention maps), otherwise tapping costs nothing.
 * The context is passed explicitly, never global.
 *
 * @param enabled - Whether to collect (default true).
 * @returns The context.
 * @concept tracing
 * @example
 * ```ts
 * import { tapContext } from 'whitebox-engine/trace'
 * import { tensor } from 'whitebox-engine/tensor'
 * const taps = tapContext()
 * taps.tap('h', tensor([1, 2]))
 * [...taps.entries.keys()] // => ['h']
 * ```
 */
export function tapContext(enabled = true): TapContext {
  const entries = new Map<string, { role: TensorRole; value: Value }>()
  return {
    enabled,
    entries,
    tap: (name, value, role = 'activation') => {
      if (enabled && isTensor(value)) entries.set(name, { role, value })
      return value
    },
  }
}
