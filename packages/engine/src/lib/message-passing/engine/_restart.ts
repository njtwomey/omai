/**
 * The restarts of a run that stalls: the damping of the k-th restart, whether a restart could do anything, whether a state
 * is one that has stalled, and the state after a restart. Internal to message passing.
 */

import type { CompiledGraph } from '../core/graph.ts'
import { residualUnresolved } from './_cavities.ts'
import type { EngineSettings, MpState } from './state.ts'

/** Whether no message is NaN or +Infinity. */
export const allFinite = (messages: readonly (readonly number[])[]): boolean =>
  !messages.some((m) => m.some((v) => Number.isNaN(v) || v === Number.POSITIVE_INFINITY))

/**
 * The damping of the `k`-th automatic restart of a run damped by `damping`: at least a half, then a step of 0.3 more for each
 * further restart, at most 0.9.
 */
function restartDamping(damping: number, k: number): number {
  return Math.min(0.9, Math.max(damping, 0.5) + 0.3 * (k - 1))
}

/**
 * Whether a restart from the start messages could do anything: only when it is damped more than the run it replaces. A run
 * already damped by a half or more (or by more than 0.9) that restarts would repeat itself to the last digit, so it is not
 * restarted: whatever stalled it would stall the restart, and its verdict is the one a restart would have reached.
 */
export function restartDiffers(damping: number, k: number): boolean {
  return restartDamping(damping, k) > damping
}

/**
 * The largest of the values and `floor` (NaN if any is NaN, as `Math.max` is), by a loop: `Math.max(...values)` with one value per
 * edge overflows the call stack at about 120 000 of them.
 */
export function largestOf(values: ArrayLike<number>, floor?: number): number {
  let top = floor ?? Number.NEGATIVE_INFINITY
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number
    if (Number.isNaN(v)) return Number.NaN
    if (v > top) top = v
  }
  return top
}

/** Whether a state that has stopped moving is one that has run out of updates it can compute, and may be restarted. */
export function isStalled(g: CompiledGraph, settings: EngineSettings, state: MpState): boolean {
  const residual = settings.schedule === 'residual'
  if (state.restarts >= (settings.restarts ?? 0)) return false
  if (!restartDiffers(settings.damping, state.restarts + 1)) return false
  if (!allFinite(state.messages)) return true
  if (residual)
    return (
      state.updates > 0 &&
      largestOf(state.residuals, 0) < settings.tol &&
      residualUnresolved(g, settings.projection, state.candidates) > 0
    )
  return state.sweeps >= 1 && state.cursor === 0 && state.change < settings.tol && state.unresolved > 0
}

/** The state after a restart: every message back at its start, the damping raised, the sweeps, updates and repair counts carried on. */
export function restartOf(settings: EngineSettings, init: () => MpState, state: MpState): MpState {
  const k = state.restarts + 1
  const damping = restartDamping(settings.damping, k)
  return {
    ...init(),
    sweeps: state.sweeps,
    updates: state.updates,
    improper: state.improper,
    restarts: k,
    damping,
    restartAt: state.sweeps,
  }
}
