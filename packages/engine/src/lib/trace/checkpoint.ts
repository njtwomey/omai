import { WhiteboxError } from '../base/index.ts'
import { type Algorithm, type RunOptions, type RunResult, runAlgorithm } from './algorithm.ts'
import type { CheckpointEvent, Run } from './events.ts'

/**
 * Continue a run from a checkpoint: the algorithm restarts from the stored state and step, with the same step keys, so
 * the continuation is exactly what the original run would have done.
 *
 * @param algorithm - The algorithm (it must implement `deserialise`).
 * @param checkpoint - A checkpoint event from a recorded run.
 * @param options - The original seed or key, and how many more steps to run.
 * @returns The continued run's result.
 * @throws WhiteboxError with code `value` when the algorithm cannot deserialise states.
 * @example
 * ```ts
 * import { resume, type Algorithm } from 'whitebox-engine/trace'
 * const count: Algorithm<number> = {
 *   name: 'count', config: {}, init: () => 0, step: (s) => s + 1,
 *   serialise: (s) => s, deserialise: (j) => j as number,
 * }
 * resume(count, { type: 'checkpoint', runId: 'r', step: 5, state: 5 }, { seed: 0, maxSteps: 2 }).state // => 7
 * ```
 */
export function resume<S>(
  algorithm: Algorithm<S>,
  checkpoint: CheckpointEvent,
  options: Omit<RunOptions, 'start'>,
): RunResult<S> {
  if (algorithm.deserialise === undefined) {
    throw new WhiteboxError('value', `resume: ${algorithm.name} does not deserialise states`)
  }
  return runAlgorithm(algorithm, {
    ...options,
    start: { state: checkpoint.state, step: checkpoint.step, serialised: true },
  })
}

/**
 * Reconstruct the state of a recorded run at any step: start from the nearest earlier checkpoint and re-run the
 * remaining steps with their original keys.
 *
 * @param algorithm - The algorithm that produced the run.
 * @param run - The recorded run (with checkpoints).
 * @param step - The step to reconstruct.
 * @param options - The original seed or key.
 * @returns The state at `step`.
 * @throws WhiteboxError with code `value` when no checkpoint at or before `step` exists.
 * @example
 * ```ts
 * import { seek, type Algorithm } from 'whitebox-engine/trace'
 * const count: Algorithm<number> = {
 *   name: 'count', config: {}, init: () => 0, step: (s) => s + 1,
 *   serialise: (s) => s, deserialise: (j) => j as number,
 * }
 * const run = { schema: 'whitebox.run/1' as const, events: [{ type: 'checkpoint' as const, runId: 'r', step: 4, state: 4 }] }
 * seek(count, run, 6, { seed: 0 }) // => 6
 * ```
 */
export function seek<S>(algorithm: Algorithm<S>, run: Run, step: number, options: Pick<RunOptions, 'seed' | 'key'>): S {
  let best: CheckpointEvent | undefined
  for (const e of run.events)
    if (e.type === 'checkpoint' && e.step <= step && (best === undefined || e.step > best.step)) best = e
  if (best === undefined) throw new WhiteboxError('value', `seek: no checkpoint at or before step ${step}`)
  return resume(algorithm, best, { ...options, maxSteps: step - best.step }).state
}
