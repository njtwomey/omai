/**
 * When a session computes and when it plays: the three start policies, the status a host reads to show how far a
 * session is, and the fraction of the run computed. Pure, shared by `useSession`, `useLiveRun` and the player.
 */

import type { Mode } from 'mdml-engine/trace'

/**
 * When a session computes and where the cursor stands at the start.
 *
 * - `'play'` (live): nothing is computed until play; the producer then computes ahead of the cursor while it plays.
 *   Use it when exploring, or when the run may be long or endless.
 * - `'first'` (primed): the whole run is computed in the background from the moment the session is made, and the
 *   cursor is parked at step 0, paused, until play. Use it for slides and demos where the animation should be ready
 *   the moment play is pressed.
 * - `'end'` (eager): the whole run is computed in the background and the cursor jumps to its end when it is done. Use
 *   it for a short run where the finished result is wanted at once.
 *
 * @index structure When a session computes and where its cursor starts: 'play' (live), 'first' (primed), 'end' (eager)
 */
export type StartPolicy = 'play' | 'first' | 'end'

/**
 * Where a session is for a host that shows a quiet indicator: `preparing` while the tape is incomplete and nothing is
 * playing, `ready` once the whole tape exists and nothing is playing, `running` while it plays (including a play
 * pressed before the tape was ready, which plays at the head as live does).
 */
export type PlayStatus = 'preparing' | 'ready' | 'running'

/** The run's budget in steps when the options give none (the engine's own default). */
export const DEFAULT_STEPS = 1000

/**
 * The effective start policy: the one asked for, else eager mode means `'end'` and anything else `'play'`.
 *
 * @param start - The policy asked for, if any.
 * @param mode - The session's engine mode (`'eager'` maps to `'end'`).
 * @returns The policy.
 * @example
 * startOf(undefined, 'eager') // => 'end'
 * startOf('first', 'eager') // => 'first'
 * startOf(undefined, undefined) // => 'play'
 */
export function startOf(start: StartPolicy | undefined, mode: Mode | undefined): StartPolicy {
  return start ?? (mode === 'eager' ? 'end' : 'play')
}

/**
 * The status of a session.
 *
 * @param playing - Whether play has been pressed and not stopped.
 * @param done - Whether the whole tape exists (every run has ended).
 * @returns `running` while playing, else `ready` when the tape is whole, else `preparing`.
 * @example
 * statusOf(false, false) // => 'preparing'
 * statusOf(false, true) // => 'ready'
 * statusOf(true, false) // => 'running'
 */
export function statusOf(playing: boolean, done: boolean): PlayStatus {
  return playing ? 'running' : done ? 'ready' : 'preparing'
}

/**
 * The fraction of a run computed. The total is the run's budget in steps, so a run that stops early (it converged)
 * reads less than 1 until it ends, and exactly 1 once it has.
 *
 * @param head - The step reached (of a comparison, the step every variant has reached).
 * @param done - Whether the run has ended.
 * @param total - The budget in steps (default 1000, the engine's own).
 * @returns A number from 0 to 1.
 * @example
 * progressOf(50, false, 200) // => 0.25
 * progressOf(80, true, 200) // => 1
 */
export function progressOf(head: number, done: boolean, total: number = DEFAULT_STEPS): number {
  if (done) return 1
  return total > 0 ? Math.min(Math.max(head / total, 0), 1) : 0
}
