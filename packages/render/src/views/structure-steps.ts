/**
 * The view of a `StructureSteps` record at a step: the structure's graph laid out once (so nodes keep their places as
 * the run goes on, unless the step gives them positions), with each node's colour, size, label and position and each
 * edge's colour, width, label and dash read from the latest recorded step at or before the viewed one.
 */

import {
  type EdgeLook,
  type NodeLook,
  type StructureStep,
  type StructureSteps,
  structureStepAt,
} from 'mdml-engine/exposition'
import { circle, line, type Paint, type Scene, type SceneNode, scene, text } from 'mdml-engine/scene'
import { circularLayout, forceLayout } from '../layout/force.ts'
import { boxBoundary } from '../layout/graph.ts'
import { formatNumber } from '../theme/format.ts'
import { categoricalColour, colourScale, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { spanOf } from './_view.ts'

/** Options of {@link structureStepsScene}. */
export interface StructureStepsSceneOptions {
  readonly mode?: ColourMode
  /** The scene's size (default 440 by 320). */
  readonly width?: number
  readonly height?: number
  /** The layout of nodes the steps give no position (default `force`, `circular` for a ring). */
  readonly layout?: 'force' | 'circular'
  readonly seed?: number
}

const NODE_R = 14

/**
 * The value range a record's colours are mapped over: the one it states, else the span of every node and edge value in
 * every step (so colours mean the same thing at every step).
 *
 * @param record - The record.
 * @returns The range.
 * @example
 * ```ts
 * import { structureSteps } from 'mdml-engine/exposition'
 * import { structure } from 'mdml-engine/scene'
 * import { structureValueRange } from 'mdml-render/views'
 * const s = structure('g', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * structureValueRange(structureSteps({ structure: s, steps: [{ step: 0, nodes: { a: { value: 2 } } }, { step: 1, nodes: { a: { value: 5 } } }] })) // => [2, 5]
 * ```
 */
export function structureValueRange(record: StructureSteps): [number, number] {
  if (record.range !== undefined) return [record.range[0], record.range[1]]
  const values: number[] = []
  for (const s of record.steps) {
    for (const n of Object.values(s.nodes ?? {})) if (n.value !== undefined) values.push(n.value)
    for (const e of s.edges ?? []) if (e.value !== undefined) values.push(e.value)
  }
  return spanOf(values)
}

// The step's position of each node, fitted to a box: every step's positions share one frame, so a moving node moves.
function positionFrame(record: StructureSteps): { x: [number, number]; y: [number, number] } | undefined {
  const xs: number[] = []
  const ys: number[] = []
  for (const s of record.steps)
    for (const n of Object.values(s.nodes ?? {}))
      if (n.position !== undefined) {
        xs.push(n.position[0])
        ys.push(n.position[1])
      }
  return xs.length === 0 ? undefined : { x: spanOf(xs), y: spanOf(ys) }
}

/**
 * A structure at a step as a scene: nodes as discs coloured by value or class and sized, labelled, with the edges
 * between them thick or thin, dashed, labelled or hidden, as the latest recorded step at or before `step` says.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded one).
 * @param options - Mode, size, layout and seed.
 * @returns The scene.
 * @example
 * ```ts
 * import { structureSteps } from 'mdml-engine/exposition'
 * import { structure } from 'mdml-engine/scene'
 * import { structureStepsScene } from 'mdml-render/views'
 * const s = structure('g', ['a', 'b'].map((id) => ({ id, label: id, role: 'latent' as const })), [{ from: 'a', to: 'b' }])
 * structureStepsScene(structureSteps({ structure: s, steps: [{ step: 0, nodes: { a: { value: 1 } } }] })).kind // => 'scene'
 * ```
 */
export function structureStepsScene(
  record: StructureSteps,
  step?: number,
  options: StructureStepsSceneOptions = {},
): Scene {
  const mode = options.mode ?? 'light'
  const width = options.width ?? 440
  const height = options.height ?? 320
  const at: StructureStep | undefined =
    step === undefined ? record.steps[record.steps.length - 1] : structureStepAt(record, step)
  const margin = NODE_R * 2.2
  const sized = record.structure.nodes.map((n) => ({ id: n.id, width: 2 * NODE_R, height: 2 * NODE_R }))
  const graph = { nodes: sized, edges: record.structure.edges.map((e) => ({ from: e.from, to: e.to })) }
  const base =
    options.layout === 'circular'
      ? circularLayout(graph, { margin: 0 })
      : forceLayout(graph, { seed: options.seed ?? 0, margin: 0 })
  const sx = (width - 2 * margin) / Math.max(1, base.width)
  const sy = (height - 2 * margin) / Math.max(1, base.height)
  const frame = positionFrame(record)
  const place = (id: string): [number, number] => {
    const given = at?.nodes?.[id]?.position
    if (given !== undefined && frame !== undefined) {
      const fx = (given[0] - frame.x[0]) / (frame.x[1] - frame.x[0] || 1)
      const fy = (given[1] - frame.y[0]) / (frame.y[1] - frame.y[0] || 1)
      return [margin + fx * (width - 2 * margin), height - margin - fy * (height - 2 * margin)]
    }
    const b = base.nodes[id]
    return [margin + (b?.x ?? 0) * sx, margin + (b?.y ?? 0) * sy]
  }
  const [lo, hi] = structureValueRange(record)
  const ramp = colourScale(sequentialRamp('viridis', mode), [lo, hi > lo ? hi : lo + 1])
  const paintOf = (look: { value?: number; category?: number } | undefined): Paint => {
    if (look?.category !== undefined) return categoricalColour(look.category) as Paint
    if (look?.value !== undefined) return ramp(look.value) as Paint
    return 'surface'
  }
  const looks = at?.nodes ?? {}
  const edgeLooks = new Map<string, EdgeLook>((at?.edges ?? []).map((e) => [`${e.from}\u0000${e.to}`, e]))
  const children: SceneNode[] = []
  const radius = (id: string): number => NODE_R * Math.min(2.5, Math.max(0.3, looks[id]?.size ?? 1))
  // Edges under the nodes, stopping at the discs' edges.
  for (const e of record.structure.edges) {
    const look = edgeLooks.get(`${e.from}\u0000${e.to}`)
    if (look?.hidden === true) continue
    const a = place(e.from)
    const b = place(e.to)
    const from = boxBoundary({ x: a[0], y: a[1], width: 2 * radius(e.from), height: 2 * radius(e.from) }, b)
    const to = boxBoundary({ x: b[0], y: b[1], width: 2 * radius(e.to), height: 2 * radius(e.to) }, a)
    const stroke = look?.value === undefined ? 'muted' : (ramp(look.value) as Paint)
    children.push(
      line(from[0], from[1], to[0], to[1], {
        stroke,
        strokeWidth: 1.4 * Math.min(6, Math.max(0.3, look?.width ?? 1)),
        ...(look?.dashed === true ? { dash: [5, 4] } : {}),
        ...(e.directed === false ? {} : { arrow: true }),
      }),
    )
    if (look?.label !== undefined)
      children.push(
        text((from[0] + to[0]) / 2, (from[1] + to[1]) / 2 - 4, look.label, { size: 10, anchor: 'middle', fill: 'fg' }),
      )
  }
  for (const n of record.structure.nodes) {
    const [x, y] = place(n.id)
    const look: NodeLook | undefined = looks[n.id]
    const r = radius(n.id)
    children.push(circle(x, y, r, { fill: paintOf(look), stroke: 'fg', strokeWidth: 1.2, title: n.detail ?? n.label }))
    children.push(text(x, y + r + 11, look?.label ?? n.label, { size: 11, anchor: 'middle', fill: 'fg' }))
  }
  if (at?.caption !== undefined) children.push(text(8, 14, at.caption, { size: 11, fill: 'muted' }))
  if (record.valueName !== undefined && hi >= lo)
    children.push(
      text(
        8,
        height - 6,
        `${record.valueName}: ${formatNumber(lo, { digits: 3 })} to ${formatNumber(hi, { digits: 3 })}`,
        {
          size: 10,
          fill: 'muted',
        },
      ),
    )
  return scene(width, height, children, {
    title: `Graph of ${record.structure.nodes.length} nodes and ${record.structure.edges.length} edges${at?.caption === undefined ? '' : `: ${at.caption}`}`,
  })
}
