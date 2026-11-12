import { type ModelRun, thinModelRun } from 'omai-engine/exposition'

/**
 * The most steps a replay record keeps. A step of a small model is a few hundred bytes of text (a message in words for each edge
 * and the beliefs of the nodes), so forty steps of a dozen messages are some tens of kilobytes to send from the worker.
 */
export const MAX_REPLAY_STEPS = 40

/**
 * A run's record cut to what a `MessageReplay` shows, in the worker that made it: at most {@link MAX_REPLAY_STEPS} steps, the
 * first and last always and between them the steps in which the beliefs moved most. A run with fewer steps is returned as it is.
 * This file has no React in it, so the worker bundles only this and the engine.
 *
 * @param record - The record of the run (`modelRunFromTape` over its tape).
 * @returns The record to put in the result.
 */
export function replayRecord(record: ModelRun): ModelRun {
  return thinModelRun(record, MAX_REPLAY_STEPS)
}
