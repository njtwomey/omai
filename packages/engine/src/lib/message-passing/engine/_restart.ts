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

/** The dampings the restarts step through: a half, then 0.8, then 0.9, the most a restart damps. */
const RESTART_SCHEDULE = [0.5, 0.8, 0.9] as const

/**
 * The damping of the `k`-th automatic restart of a run damped by `damping`: the `k`-th of the schedule (a half, 0.8, 0.9) among the
 * dampings stronger than the run's own, the last one for any further restart. A run damped by less than a half goes through all
 * three; one damped by a half or more skips ahead (a run damped by 0.5 restarts at 0.8, then 0.9), so a restart is always damped
 * more than the run it replaces; one damped by 0.9 or more has none stronger and gets its own damping back (no restart differs).
 */
function restartDamping(damping: number, k: number): number {
  const stronger = RESTART_SCHEDULE.filter((v) => v > damping)
  return stronger[Math.min(k, stronger.length) - 1] ?? damping
}

/**
 * Whether the `k`-th restart from the start messages could do anything: only when it is damped more than the attempt it replaces (the
 * run itself for the first, the previous restart after). A restart damped as much as that attempt would repeat it to the last digit, so
 * it is not made: whatever stalled it would stall the restart, and its verdict is the one a restart would have reached. A run damped by
 * 0.9 or more has none; one damped by 0.5 has two (0.8 and 0.9); one damped by less than a half has three.
 */
export function restartDiffers(damping: number, k: number): boolean {
  return restartDamping(damping, k) > (k > 1 ? restartDamping(damping, k - 1) : damping)
}

/**
 * The largest of the values and `floor` (NaN if any is NaN, as `Math.max` is), by a loop: `Math.max(...values)` with one value per
 * edge overflows the call stack at about 120 000 of them.
 */
export function largestOf(values: ArrayLike<number>, floor = Number.NEGATIVE_INFINITY): number {
  let top = floor
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
