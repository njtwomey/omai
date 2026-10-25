/**
 * Taps: named intermediate values (activations, attention weights) that model code exposes to a collector. `tapsFor`
 * binds the collector to a run so values are recorded only when the policy selects activations at that step, and cost
 * nothing otherwise.
 */

import { isTensor, type Value } from '../tensor/index.ts'
import type { TensorRole } from './events.ts'
import type { RunHandle } from './tracer.ts'

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
 * import { tapContext } from 'mdml-engine/trace'
 * import { tensor } from 'mdml-engine/tensor'
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

/**
 * A tap context bound to a run: enabled only when the run records activations at this step (the policy selects the
 * `activation` role and the step is on the schedule), and each tapped value is recorded as a tensor event at once
 * (with the role given to `tap`, `activation` by default) as well as collected. So model code taps freely and the
 * policy alone decides what reaches the trace.
 *
 * @param run - The run to record into.
 * @param step - The current step.
 * @returns The context.
 * @concept tracing
 * @example
 * ```ts
 * import { policies, recorder, startRun, tapsFor } from 'mdml-engine/trace'
 * import { tensor } from 'mdml-engine/tensor'
 * const rec = recorder()
 * const run = startRun(rec, { name: 'net', seed: 0, config: {}, policy: policies.debug })
 * tapsFor(run, 0).tap('h', tensor([1, 2]))
 * rec.events.filter((e) => e.type === 'tensor').length // => 1
 * tapsFor(startRun(rec, { name: 'quiet', seed: 0, config: {} }), 0).enabled // => false
 * ```
 */
export function tapsFor(run: RunHandle, step: number): TapContext {
  const enabled = run.enabled && run.policy.roles.includes('activation') && run.due(step)
  const inner = tapContext(enabled)
  return {
    enabled,
    entries: inner.entries,
    tap: (name, value, role = 'activation') => {
      if (enabled && isTensor(value)) run.tensor(step, name, role, value)
      return inner.tap(name, value, role)
    },
  }
}
