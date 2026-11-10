/**
 * The view of a `ModelRun` record: the model's plate diagram (or factor graph) at a step of the run, with what is
 * believed of each node written in it (`μ 0.56 · σ 0.83`, a small interval or bar glyph under the node scaled over the
 * whole run, observed nodes as such), the edges that carried a message at this step active and labelled with the message
 * in words, and beside it the convergence series (change, evidence or bound) with a cursor at the step and the count of
 * improper messages. The diagram is the library's one plate-diagram renderer (`diagram`): this view builds the overlay,
 * the frames that keep the layout still, and the glyphs, and a run in progress needs nothing more than a record of its
 * latest events.
 */

import type { EChartsOption } from 'echarts'
import { type ModelRun, type ModelRunSeries, modelRunAt, type NodeState } from 'omai-engine/exposition'
import {
  circle,
  line,
  rect,
  type Scene,
  type SceneNode,
  type StructureNode,
  type StructureOverlay,
} from 'omai-engine/scene'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { gridSpec } from '../figure/grid.ts'
import type { LayoutBox } from '../layout/graph.ts'
import { type DiagramOptions, diagram } from '../structures/diagram.ts'
import { type ViewOptions, viewFigure, wantsLog } from './_view.ts'

/** Options of {@link modelRunScene}: the diagram's layout options and the mode. */
export interface ModelRunSceneOptions extends Omit<DiagramOptions, 'overlay' | 'frames' | 'decorate'> {
  /** Draw the small interval and bar glyphs under the nodes (default true). */
  readonly glyphs?: boolean
}

/** The overlay badges of a belief: its numbers where it has them, else what it is. */
function badgesOf(state: NodeState): Record<string, string | number> | undefined {
  if (state.improper === true) return { improper: 'yes' }
  if (state.observed !== undefined) {
    const v = state.observed
    return typeof v === 'number' || typeof v === 'string' ? { obs: v } : undefined
  }
  const out: Record<string, string | number> = {}
  if (state.mean !== undefined && state.elements === undefined && state.probs?.length !== 2) {
    out['μ'] = state.mean
    if (state.sd !== undefined) out['σ'] = state.sd
  } else if (state.probs !== undefined && state.probs.length === 2) out['p'] = state.probs[1] as number
  else if (state.probs !== undefined && state.probs.length > 2) {
    const best = state.probs.indexOf(Math.max(...state.probs))
    out['state'] = best
  } else if (state.mean !== undefined) out['μ̄'] = state.mean
  if (state.rhat !== undefined) out['R̂'] = Number(state.rhat.toFixed(3))
  return Object.keys(out).length === 0 ? undefined : out
}

/**
 * The overlay of a run at a step: the edges that carried a message at it, active and labelled with the message in words
 * (a message between an edge's ends in either direction is on that edge as the structure draws it), their end nodes
 * active, and a badge on every node that has a belief by then. Nothing but the ids of the structure is read.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded).
 * @returns The overlay (empty for a record with no steps).
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunOverlay } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }, { id: 'b', label: 'b', role: 'latent' }], [{ from: 'a', to: 'b' }])
 * const r = modelRun({ structure: s, status: 'running', series: [], steps: [{ step: 1, nodes: { a: { mean: 1, sd: 2, text: '1 ± 2' } }, messages: [{ from: 'b', to: 'a', text: 'N(μ = 1, σ² = 4)' }] }] })
 * modelRunOverlay(r).edges // => [{ from: 'a', to: 'b', label: 'N(μ = 1, σ² = 4)' }]
 * ```
 */
export function modelRunOverlay(record: ModelRun, step?: number): StructureOverlay {
  const at = modelRunAt(record, step)
  if (at === undefined) return {}
  const directed = new Map<string, { from: string; to: string }>()
  for (const e of record.structure.edges) {
    directed.set(`${e.from}\u0000${e.to}`, e)
    if (!directed.has(`${e.to}\u0000${e.from}`)) directed.set(`${e.to}\u0000${e.from}`, e)
  }
  const edges: { from: string; to: string; label: string }[] = []
  const active = new Set<string>()
  for (const m of at.messages) {
    active.add(m.from)
    active.add(m.to)
    const e = directed.get(`${m.from}\u0000${m.to}`)
    if (e === undefined || m.from === m.to) continue
    const key = `${e.from}\u0000${e.to}`
    const prior = edges.find((x) => `${x.from}\u0000${x.to}` === key)
    if (prior === undefined) edges.push({ from: e.from, to: e.to, label: m.text })
    else if (!prior.label.includes(m.text))
      edges[edges.indexOf(prior)] = { ...prior, label: `${prior.label}\n${m.text}` }
  }
  const badges: Record<string, Record<string, string | number>> = {}
  for (const [id, state] of Object.entries(at.nodes)) {
    const b = badgesOf(state)
    if (b !== undefined) badges[id] = b
  }
  return { active: [...active], edges, badges }
}

/**
 * The overlay of every recorded step, in order: pass them as the diagram's `frames` so its nodes are sized for the widest
 * badge any step gives them and the layout never moves as the run goes on.
 *
 * @param record - The record.
 * @returns One overlay per step.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunFrames } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * modelRunFrames(modelRun({ structure: s, status: 'running', series: [], steps: [{ step: 1, nodes: {} }, { step: 2, nodes: {} }] })).length // => 2
 * ```
 */
export function modelRunFrames(record: ModelRun): StructureOverlay[] {
  return record.steps.map((s) => modelRunOverlay(record, s.step))
}

/** The span a node's belief covers over the whole run (two sd either side of its mean), so a glyph keeps its scale. */
function spans(record: ModelRun): Map<string, [number, number]> {
  const out = new Map<string, [number, number]>()
  for (const s of record.steps)
    for (const [id, n] of Object.entries(s.nodes)) {
      if (n.mean === undefined || n.probs !== undefined || n.elements !== undefined) continue
      const w = 2 * (n.sd ?? 0)
      const [lo, hi] = out.get(id) ?? [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]
      out.set(id, [Math.min(lo, n.mean - w), Math.max(hi, n.mean + w)])
    }
  return out
}

/**
 * The glyphs of a run at a step, as the diagram's `decorate` takes them: under each node with a belief a short axis with
 * the interval `mean ± sd` as a bar and the mean as a dot (on the span the node's belief covers over the run, so
 * the bar visibly narrows and moves as the run converges), and for a discrete belief small bars of the probabilities.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded).
 * @returns The function to hand to `diagram`.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunDecorate } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * const r = modelRun({ structure: s, status: 'running', series: [], steps: [{ step: 1, nodes: { a: { mean: 0, sd: 1, text: '0 ± 1' } } }] })
 * modelRunDecorate(r)({ id: 'a', label: 'a', role: 'latent' }, { x: 50, y: 20, width: 40, height: 40 })?.length // => 3
 * ```
 */
export function modelRunDecorate(record: ModelRun, step?: number): NonNullable<DiagramOptions['decorate']> {
  const at = modelRunAt(record, step)
  const span = spans(record)
  return (node: StructureNode, box: LayoutBox): readonly SceneNode[] | undefined => {
    const n = at?.nodes[node.id]
    if (n === undefined || box.width < 24 || n.observed !== undefined || n.improper === true) return undefined
    const w = Math.min(56, Math.max(32, box.width))
    const x0 = box.x - w / 2
    const y = box.y + box.height / 2 + 8
    if (n.probs !== undefined) {
      const k = n.probs.length
      const bw = w / k
      return n.probs.map((p, i) =>
        rect(x0 + i * bw + 0.5, y + 5 - 11 * p, Math.max(1, bw - 1), Math.max(0.5, 11 * p), {
          fill: 'tab0',
          stroke: 'none',
        }),
      )
    }
    const range = span.get(node.id)
    if (n.mean === undefined || range === undefined) return undefined
    const [lo, hi] = range
    const scale = (v: number): number => x0 + (hi > lo ? ((v - lo) / (hi - lo)) * w : w / 2)
    const sd = n.sd ?? 0
    return [
      line(x0, y, x0 + w, y, { stroke: 'grid', strokeWidth: 1 }),
      line(scale(n.mean - sd), y, scale(n.mean + sd), y, { stroke: 'tab0', strokeWidth: 3.5 }),
      circle(scale(n.mean), y, 2.6, { fill: 'fg', stroke: 'bg', strokeWidth: 0.8 }),
    ]
  }
}

/**
 * The model run at a step as a scene: the diagram of its structure with the overlay and the glyphs of the step, laid out
 * for every step so nothing moves between them.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded).
 * @param options - The diagram's layout options, and whether to draw the glyphs.
 * @returns The scene.
 * @index function The plate diagram of a model run at a step: beliefs in the nodes, messages on the edges.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunScene } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * modelRunScene(modelRun({ structure: s, status: 'running', series: [], steps: [{ step: 1, nodes: { a: { mean: 0, sd: 1, text: '0 ± 1' } } }] })).kind // => 'scene'
 * ```
 */
export function modelRunScene(record: ModelRun, step?: number, options: ModelRunSceneOptions = {}): Scene {
  const { glyphs = true, ...layout } = options
  return diagram(record.structure, {
    ...layout,
    overlay: modelRunOverlay(record, step),
    frames: modelRunFrames(record),
    ...(glyphs ? { decorate: modelRunDecorate(record, step) } : {}),
  })
}

/** The latest value of a series at or before a step. */
function valueAt(series: ModelRunSeries | undefined, step: number): number | undefined {
  if (series === undefined) return undefined
  let found: number | undefined
  for (const [i, s] of series.steps.entries()) {
    if (s > step) break
    found = series.values[i]
  }
  return found
}

/**
 * The line under the picture: the step, whether the run is still going or how it ended, the improper updates counted so
 * far (and how many were skipped), and the latest change and evidence or bound.
 *
 * @param record - The record.
 * @param step - The tape step (default the last recorded).
 * @returns The line.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunStatus } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * const r = modelRun({ structure: s, status: 'running', steps: [{ step: 4, nodes: {} }], series: [{ name: 'improper', label: 'improper updates', steps: [4], values: [2] }] })
 * modelRunStatus(r) // => 'step 4 · running · improper updates 2'
 * ```
 */
export function modelRunStatus(record: ModelRun, step?: number): string {
  const at = modelRunAt(record, step)
  const t = at?.step ?? 0
  const parts = [`step ${t}`, t >= (record.steps.at(-1)?.step ?? 0) ? record.status : 'running']
  for (const s of record.series) {
    const v = valueAt(s, t)
    if (v === undefined) continue
    if (s.name === 'improper' || s.name === 'skipped') {
      if (s.name === 'improper' || v > 0) parts.push(`${s.name === 'skipped' ? 'skipped' : s.label} ${v}`)
    } else if (s.name === 'change') parts.push(`change ${Number(v.toPrecision(3))}`)
    else if (s.name === 'evidence' || s.name === 'elbo') parts.push(`${s.name} ${Number(v.toPrecision(5))}`)
  }
  return parts.join(' · ')
}

/**
 * The convergence series of a run as a figure, one panel each (the change over steps on a log axis when it spans
 * decades, the evidence estimate or bound) with a cursor at the viewed step. The counters (improper, skipped, damped,
 * reset) are in {@link modelRunStatus}, not drawn.
 *
 * @param record - The record.
 * @param step - The step the cursor marks (default the last recorded).
 * @param options - Mode and figure.
 * @returns The ECharts option, or `undefined` when the run has nothing to chart.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { modelRunConvergence } from 'omai-render/views'
 * const s = structure('model-graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
 * const r = modelRun({ structure: s, status: 'running', steps: [], series: [{ name: 'change', label: 'largest change', steps: [1, 2, 3], values: [1, 0.1, 0.01] }] })
 * modelRunConvergence(r)?.grid.length // => 1
 * ```
 */
export function modelRunConvergence(
  record: ModelRun,
  step?: number,
  options: ViewOptions = {},
): EChartsOption | undefined {
  const charted = record.series
    .filter(
      (s) =>
        s.steps.length > 0 &&
        ['change', 'evidence', 'elbo', 'logEvidence', 'objective', 'divergences'].includes(s.name),
    )
    .slice(0, 3)
  if (charted.length === 0) return undefined
  const fig = viewFigure(options, { aspect: Math.max(1.4, 4.2 / (charted.length + 0.6)) })
  const mode = fig.mode
  const at = step ?? record.steps.at(-1)?.step
  const panels: Record<string, Panel> = {}
  for (const [i, s] of charted.entries()) {
    const log = s.name === 'change' && wantsLog(s.values.filter((v) => v > 0))
    panels[`s${i}`] = {
      layers: [
        lines([{ x: [...s.steps], y: [...s.values], name: s.label, colour: `tab${i}`, width: 1.8 }], { mode }),
        ...(at === undefined ? [] : [referenceLines({ x: [at] }, { mode, colour: 'accent', dash: 'solid' })]),
      ],
      options: {
        title: s.label,
        ...(i === charted.length - 1 ? { x: { label: 'step' } } : {}),
        ...(log ? { y: { scale: 'log' as const } } : {}),
      },
    }
  }
  const grid = gridSpec(charted.length, 1, {
    hspace: 34,
    cells: charted.map((_, i) => ({ id: `s${i}`, row: i, col: 0, ...(i > 0 ? { sharex: 's0' } : {}) })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}
