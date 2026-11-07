/**
 * The exposition record that carries a structure as a diagram with a run: a network's architecture, a computation graph
 * or a graphical model, as a scene graph, with optional steps that highlight active nodes and edges and add badges.
 */

import { MdmlError } from '../base/index.ts'
import type { Structure, StructureOverlay } from '../scene/index.ts'
import { seal } from './_check.ts'

/** One state laid over a diagram: the active nodes and edges and the badges, with what happened in a line. */
export interface DiagramStep {
  readonly overlay: StructureOverlay
  readonly caption?: string
}

/**
 * A structure as a record, so a diagram travels with a run: a network's architecture (`describe(net)`), a
 * computation graph, a graphical model. Optionally states laid over it in turn (a forward pass layer by layer and
 * the deltas back, messages passed), each with a caption, so a view can play them over the diagram.
 */
export interface DiagramRecord {
  readonly kind: 'diagram'
  readonly structure: Structure
  readonly steps?: readonly DiagramStep[]
  /** What a step is (default 'step'): 'pass', 'message', 'layer'. */
  readonly stepName?: string
}

/**
 * A diagram record, with its overlays checked against the structure.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` when an overlay names a node the structure does not have.
 * @example
 * ```ts
 * import { diagramRecord } from 'mdml-engine/exposition'
 * import { structure } from 'mdml-engine/scene'
 * const s = structure('net', [{ id: 'x', label: 'x', role: 'input' }, { id: 'y', label: 'y', role: 'output' }], [{ from: 'x', to: 'y' }])
 * diagramRecord({ structure: s, steps: [{ overlay: { active: ['x'] }, caption: 'forward' }] }).steps?.length // => 1
 * ```
 */
export function diagramRecord(record: Omit<DiagramRecord, 'kind'>): DiagramRecord {
  const ids = new Set(record.structure.nodes.map((n) => n.id))
  for (const [t, step] of (record.steps ?? []).entries()) {
    const named = [...(step.overlay.active ?? []), ...Object.keys(step.overlay.badges ?? {})]
    const unknown = named.find((id) => !ids.has(id))
    if (unknown !== undefined)
      throw new MdmlError('value', `diagram: steps[${t}] names the node ${unknown}, which the structure lacks`)
  }
  return seal({ kind: 'diagram', ...record })
}
