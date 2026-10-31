/**
 * The view of a `SceneSteps` record at a step: the latest recorded scene at or before it, with its caption written
 * along the bottom edge. The scenes are drawn as they were recorded, so the headless view is a choice among them.
 */

import { type SceneSteps, sceneStepAt } from 'mdml-engine/exposition'
import { type Scene, scene, text } from 'mdml-engine/scene'

/**
 * The scene to show at a step, with the step's caption added.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded one).
 * @returns The scene (an empty one of 1 by 1 when the record has none).
 * @example
 * ```ts
 * import { sceneSteps } from 'mdml-engine/exposition'
 * import { scene } from 'mdml-engine/scene'
 * import { sceneStepsScene } from 'mdml-render/views'
 * const r = sceneSteps({ steps: [{ step: 0, scene: scene(80, 40, []) }, { step: 3, scene: scene(80, 40, []), caption: 'later' }] })
 * sceneStepsScene(r, 4).children.length // => 1
 * ```
 */
export function sceneStepsScene(record: SceneSteps, step?: number): Scene {
  const at = step === undefined ? record.steps[record.steps.length - 1] : sceneStepAt(record, step)
  if (at === undefined) return scene(1, 1, [])
  if (at.caption === undefined) return at.scene
  return {
    ...at.scene,
    children: [...at.scene.children, text(8, at.scene.height - 8, at.caption, { size: 11, fill: 'muted' })],
  }
}
