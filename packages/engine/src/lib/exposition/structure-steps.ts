/**
 * The exposition record of a graph whose appearance follows a run (`structureSteps`): a fixed `Structure` (the nodes
 * and edges) with, at each recorded step, the attributes of its nodes and edges (a value as colour, a size, a label,
 * a position) so a view can redraw the same graph as the run goes on: a belief propagating, a causal graph being
 * found, a search tree growing, an embedding laid out.
 */

import { MdmlError } from '../base/index.ts'
import type { Structure } from '../scene/index.ts'

/** How a node looks at a step. A value maps to the ramp, a category to a categorical colour. */
export interface NodeLook {
  /** A number mapped along the record's ramp over its `range`. */
  readonly value?: number
  /** A class (0, 1, ...) drawn in a categorical colour; wins over `value`. */
  readonly category?: number
  /** A relative size (default 1). */
  readonly size?: number
  /** Replaces the structure's label. */
  readonly label?: string
  /** A position in the plane, in any units (the view fits them); nodes without one keep their layout. */
  readonly position?: readonly [number, number]
}

/** How an edge looks at a step. */
export interface EdgeLook {
  readonly from: string
  readonly to: string
  /** A number mapped along the ramp. */
  readonly value?: number
  /** A relative thickness (default 1). */
  readonly width?: number
  readonly label?: string
  readonly dashed?: boolean
  /** Whether the edge is hidden at this step (an edge not yet found). */
  readonly hidden?: boolean
}

/** One recorded state of the graph: the step it was taken at and the looks of the nodes and edges it changes. */
export interface StructureStep {
  /** The tape step (the viewed step picks the latest record at or before it). */
  readonly step: number
  /** Looks by node id; nodes not named look as they did in the structure. */
  readonly nodes?: Readonly<Record<string, NodeLook>>
  readonly edges?: readonly EdgeLook[]
  readonly caption?: string
}

/** A structure with the looks of its nodes and edges at recorded steps. */
export interface StructureSteps {
  readonly kind: 'structureSteps'
  readonly structure: Structure
  readonly steps: readonly StructureStep[]
  /** The values mapped to the ramp (default the span of all node and edge values). */
  readonly range?: readonly [number, number]
  /** What a node's value is (`belief`, `activation`), for the legend. */
  readonly valueName?: string
}

/**
 * A structure with per-step looks, with its ids checked against the structure.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `value` when a step names a node or an edge the structure lacks, or the steps are not in order.
 * @example
 * ```ts
 * import { structureSteps } from 'mdml-engine/exposition'
 * import { structure } from 'mdml-engine/scene'
 * const s = structure('graph', [{ id: 'a', label: 'a', role: 'latent' }, { id: 'b', label: 'b', role: 'latent' }], [{ from: 'a', to: 'b' }])
 * structureSteps({ structure: s, steps: [{ step: 0, nodes: { a: { value: 1 } } }] }).steps.length // => 1
 * ```
 */
export function structureSteps(record: Omit<StructureSteps, 'kind'>): StructureSteps {
  const ids = new Set(record.structure.nodes.map((n) => n.id))
  const edges = new Set(record.structure.edges.map((e) => `${e.from}\u0000${e.to}`))
  let previous = Number.NEGATIVE_INFINITY
  for (const [t, step] of record.steps.entries()) {
    if (step.step < previous) throw new MdmlError('value', `structureSteps: steps[${t}] is out of order`)
    previous = step.step
    const unknown = Object.keys(step.nodes ?? {}).find((id) => !ids.has(id))
    if (unknown !== undefined)
      throw new MdmlError('value', `structureSteps: steps[${t}] names the node ${unknown}, which the structure lacks`)
    const missing = (step.edges ?? []).find((e) => !edges.has(`${e.from}\u0000${e.to}`))
    if (missing !== undefined)
      throw new MdmlError(
        'value',
        `structureSteps: steps[${t}] names the edge ${missing.from} to ${missing.to}, which the structure lacks`,
      )
  }
  return { kind: 'structureSteps', ...record }
}

/**
 * The state of a graph at a step: the latest recorded one at or before it (the first when none is yet).
 *
 * @param record - The record.
 * @param step - The tape step.
 * @returns The step's looks, or `undefined` when the record has no steps.
 * @example
 * ```ts
 * import { structureSteps, structureStepAt } from 'mdml-engine/exposition'
 * import { structure } from 'mdml-engine/scene'
 * const s = structure('graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * const r = structureSteps({ structure: s, steps: [{ step: 0 }, { step: 10, caption: 'later' }] })
 * structureStepAt(r, 12)?.caption // => 'later'
 * ```
 */
export function structureStepAt(record: StructureSteps, step: number): StructureStep | undefined {
  return record.steps.findLast((s) => s.step <= step) ?? record.steps[0]
}
