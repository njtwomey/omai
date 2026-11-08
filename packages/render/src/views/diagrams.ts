/**
 * The view of a `DiagramRecord`: its structure laid out once for all steps, so nodes never jump, with one step's
 * overlay of active nodes, edges and value badges drawn over it.
 */

import type { DiagramRecord } from 'omai-engine/exposition'
import type { Scene } from 'omai-engine/scene'
import { diagram } from '../structures/diagram.ts'

/**
 * A diagram record as a scene: the structure laid out once for all its steps (so nodes never jump), with one step's
 * overlay (active nodes and edges, value badges) over it; the last step by default, none when the record has none.
 *
 * @param record - The record.
 * @param step - The step whose overlay to draw (default the last).
 * @returns The scene.
 * @example
 * ```ts
 * import { diagramRecord } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { diagramScene } from 'omai-render/views'
 * const s = structure('net', [{ id: 'x', label: 'x', role: 'input' }, { id: 'y', label: 'y', role: 'output' }], [{ from: 'x', to: 'y' }])
 * diagramScene(diagramRecord({ structure: s })).kind // => 'scene'
 * ```
 */
export function diagramScene(record: DiagramRecord, step?: number): Scene {
  const steps = record.steps ?? []
  const at = steps[Math.min(Math.max(step ?? steps.length - 1, 0), steps.length - 1)]
  const frames = steps.map((s) => s.overlay)
  return diagram(record.structure, {
    ...(at === undefined ? {} : { overlay: at.overlay }),
    ...(frames.length === 0 ? {} : { frames }),
  })
}
