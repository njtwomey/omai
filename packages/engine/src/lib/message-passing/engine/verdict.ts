/**
 * How a run stands: the verdict of a state (converged, stopped as stalled, failed as degenerate, diverged, completed, or
 * still going) from the beliefs, the unresolved updates and the restarts, and why when it did not converge.
 */

import type { RunStatus } from '../../trace/index.ts'
import type { CompiledGraph } from '../core/graph.ts'
import { residualUnresolved } from './_cavities.ts'
import { allFinite, largestOf, restartDiffers } from './_restart.ts'
import { beliefOf } from './cavity.ts'
import type { EngineSettings, MpState } from './state.ts'

/** How a run stands: the status it has reached (undefined while it can go on) and, for one that did not converge, why. */
export interface MpVerdict {
  readonly status: RunStatus | undefined
  /** What went wrong (`diverged`, `stalled`, `degenerate`) or ran out (`completed`); undefined for a run that is converged or still going. */
  readonly reason: string | undefined
}

/**
 * The verdict of a state: `converged` when the beliefs have stopped moving (below `tol` over a sweep, or no residual above
 * it), every update that can be computed reproduces its message, and no belief is without a density. A run whose beliefs
 * stopped moving while some updates could not be computed (an improper cavity: the old message is kept, so the change is
 * zero for want of an update, not at a fixed point) is `stopped` with the reason `stalled` (which a damped restart,
 * `EngineSettings.restarts`, tries to recover from); one that has been restarted and is stuck on the same kind of update
 * is `converged` with the count in the reason (the stall is not an artifact of its first trajectory: the usual skip of an
 * update with an improper cavity, which leaves that message as it was). A fixed point with a belief that has no density is
 * `failed` with the reason `degenerate`; messages that are not finite are `diverged`; a run that reaches the sweep cap (counted from its last restart) is `completed`.
 *
 * @param g - The compiled graph.
 * @param state - A state of a run.
 * @param s - The slots the verdict depends on: the tolerance, the sweep cap, the schedule and the projection.
 * @returns The status and the reason.
 */
export function mpVerdict(
  g: CompiledGraph,
  state: MpState,
  s: Pick<EngineSettings, 'tol' | 'maxSweeps' | 'schedule' | 'projection'> &
    Partial<Pick<EngineSettings, 'damping' | 'restarts'>>,
): MpVerdict {
  if (!allFinite(state.messages)) return { status: 'diverged', reason: 'a message is not finite' }
  const still =
    s.schedule === 'residual'
      ? state.updates > 0 && (state.residuals.length === 0 || largestOf(state.residuals) < s.tol)
      : state.sweeps >= 1 && state.cursor === 0 && state.change < s.tol
  if (still) {
    const unresolved =
      s.schedule === 'residual' ? residualUnresolved(g, s.projection, state.candidates) : state.unresolved
    const many = `${unresolved} message update${unresolved === 1 ? '' : 's'}`
    // A run asked to restart but already damped as much as a restart would be is not restarted (see `restartDiffers`): it is judged as the restart would be.
    const replays = (s.restarts ?? 0) > 0 && !restartDiffers(s.damping ?? 0, 1)
    if (unresolved > 0 && state.restarts === 0 && !replays)
      return {
        status: 'stopped',
        reason: `stalled: ${many} could not be computed in the last sweep (an improper cavity, or an update the repair policy dropped), so those messages were not recomputed and the beliefs stopped moving without reaching a fixed point`,
      }
    const improper = g.variables.flatMap((v, i) =>
      v.edges.length > 0 && v.observed === undefined && !v.family.isProper(beliefOf(g, state.messages, i))
        ? [v.name]
        : [],
    )
    if (improper.length > 0)
      return {
        status: 'failed',
        reason: `degenerate: the beliefs of ${improper.slice(0, 4).join(', ')}${improper.length > 4 ? ` and ${improper.length - 4} more` : ''} have no density at the point where the messages stopped moving`,
      }
    return {
      status: 'converged',
      reason:
        unresolved === 0
          ? undefined
          : state.restarts === 0
            ? `converged, but ${many} could not be computed (an improper cavity: the skip of such an update leaves its message as it was); a restart would repeat the run, which is already damped by ${s.damping ?? 0}, so none was made`
            : `converged after ${state.restarts} damped restart${state.restarts === 1 ? '' : 's'}, but ${many} could not be computed even then (an improper cavity: the skip of such an update leaves its message as it was)`,
    }
  }
  return state.sweeps - state.restartAt >= s.maxSweeps
    ? { status: 'completed', reason: 'the sweep budget ran out' }
    : { status: undefined, reason: undefined }
}
