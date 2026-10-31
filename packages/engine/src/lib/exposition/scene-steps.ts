/**
 * The exposition record of a scene per step (`sceneSteps`): a `Scene` (agents on a plane, a plant and its horizon, a
 * point cloud being drawn) recorded at steps of a run, so a view shows the latest one at or before the viewed step. It
 * makes what an environment's `frame` events carry a first-class record that any method can return.
 */

import { MdmlError } from '../base/index.ts'
import type { Scene } from '../scene/index.ts'

/** One recorded scene: the tape step it was taken at, the scene and an optional caption. */
export interface SceneStep {
  /** The tape step (the viewed step picks the latest record at or before it). */
  readonly step: number
  readonly scene: Scene
  readonly caption?: string
}

/** Scenes at recorded steps. */
export interface SceneSteps {
  readonly kind: 'sceneSteps'
  readonly steps: readonly SceneStep[]
  /** What the step counts (`generation`, `frame`), for the caption. */
  readonly stepName?: string
}

/**
 * Scenes at steps, with the steps checked to be in order.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` when the steps are not in order.
 * @example
 * ```ts
 * import { sceneSteps } from 'mdml-engine/exposition'
 * import { circle, scene } from 'mdml-engine/scene'
 * const s = (r: number) => scene(40, 40, [circle(20, 20, r)])
 * sceneSteps({ steps: [{ step: 0, scene: s(5) }, { step: 4, scene: s(9) }] }).steps.length // => 2
 * ```
 */
export function sceneSteps(record: Omit<SceneSteps, 'kind'>): SceneSteps {
  let previous = Number.NEGATIVE_INFINITY
  for (const [t, s] of record.steps.entries()) {
    if (s.step < previous) throw new MdmlError('value', `sceneSteps: steps[${t}] is out of order`)
    previous = s.step
  }
  return { kind: 'sceneSteps', ...record }
}

/**
 * The scene at a step: the latest recorded one at or before it (the first when none is yet).
 *
 * @param record - The record.
 * @param step - The tape step.
 * @returns The recorded step, or `undefined` when the record has none.
 * @example
 * ```ts
 * import { sceneStepAt, sceneSteps } from 'mdml-engine/exposition'
 * import { scene } from 'mdml-engine/scene'
 * const r = sceneSteps({ steps: [{ step: 0, scene: scene(1, 1, []) }, { step: 5, scene: scene(2, 2, []) }] })
 * sceneStepAt(r, 7)?.scene.width // => 2
 * ```
 */
export function sceneStepAt(record: SceneSteps, step: number): SceneStep | undefined {
  return record.steps.findLast((s) => s.step <= step) ?? record.steps[0]
}
