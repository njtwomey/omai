import { type ModelRun, modelRunFromTape, type TapeLike, thinModelRun } from 'omai-engine/exposition'
import { type ModelLike, toStructure } from 'omai-engine/probprog/model'
import type { Structure } from 'omai-engine/scene'

/**
 * The most steps a replay record keeps. A step of a small model is a few hundred bytes of text (a message in words for each edge
 * and the beliefs of the nodes), so forty steps of a dozen messages are some tens of kilobytes to send from the worker.
 */
export const MAX_REPLAY_STEPS = 40

/** The most message badges a replay keeps for each step: more would crowd the lines of a small graph. */
export const MAX_REPLAY_MESSAGES = 6

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

/**
 * The record of a run for a `MessageReplay`: the tape read as a run on the given graph, with at most
 * {@link MAX_REPLAY_MESSAGES} messages a step, and cut by {@link replayRecord}.
 *
 * @param structure - The graph the run was on (a factor graph).
 * @param tape - The tape of the run (`session.run()`).
 * @returns The record to put in the result.
 */
export function replayOf(structure: Structure, tape: TapeLike): ModelRun {
  return replayRecord(modelRunFromTape(structure, { run: tape }, { maxMessages: MAX_REPLAY_MESSAGES }))
}

/**
 * The record of a run on a bound model for a `MessageReplay`: {@link replayOf} on the model's factor graph.
 *
 * @param bound - The model the run was on (a bound model, or a model).
 * @param tape - The tape of the run (`session.run()`).
 * @returns The record to put in the result.
 */
export function replayOfModel(bound: ModelLike, tape: TapeLike): ModelRun {
  return replayOf(toStructure(bound, { view: 'factor' }), tape)
}
