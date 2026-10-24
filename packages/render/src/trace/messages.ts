import type { SeriesOption } from 'echarts'
import { type Frame, type Structure, type StructureOverlay, type Timeline, timeline } from 'whitebox-engine/scene'
import { messages, type Run } from 'whitebox-engine/trace'
import { type Layer, layer } from '../charts/plot.ts'
import { categoricalColour } from '../theme/palettes.ts'

/**
 * A message-passing timeline: one row per edge (`from → to`, or the message kind when there are no ends), a mark at
 * each step a message passed along it, coloured by kind.
 *
 * @param run - The trace.
 * @param options - A message kind to keep (default all).
 * @returns The layer, rows as categories.
 * @example
 * ```ts
 * import { messageTimeline } from 'whitebox-render/trace'
 * const m = (step: number, from: string, to: string) => ({ type: 'message' as const, runId: 'r', step, kind: 'belief', from, to, payload: null })
 * messageTimeline({ schema: 'whitebox.run/1', events: [m(0, 'a', 'b'), m(1, 'b', 'c')] }).yCategories // => ['a → b', 'b → c']
 * ```
 */
export function messageTimeline(run: Run, options: { readonly kind?: string } = {}): Layer {
  const ms = messages(run, options.kind)
  const rows: string[] = []
  const kinds: string[] = []
  const rowOf = (m: (typeof ms)[number]): string =>
    m.from !== undefined && m.to !== undefined ? `${m.from} → ${m.to}` : (m.from ?? m.to ?? m.kind)
  for (const m of ms) {
    const r = rowOf(m)
    if (!rows.includes(r)) rows.push(r)
    if (!kinds.includes(m.kind)) kinds.push(m.kind)
  }
  const series: SeriesOption[] = kinds.map(
    (k, i) =>
      ({
        type: 'scatter',
        name: k,
        symbol: 'rect',
        symbolSize: [4, 12],
        data: ms.filter((m) => m.kind === k).map((m) => [m.step, rows.indexOf(rowOf(m))]),
        itemStyle: { color: categoricalColour(i) },
      }) as SeriesOption,
  )
  const steps = ms.map((m) => m.step)
  return layer(series, steps.length > 0 ? { x: [Math.min(...steps), Math.max(...steps)] } : {}, { yCategories: rows })
}

/**
 * Animation frames of a structure from its message events: at each step the edges carrying messages are active, and
 * their end nodes highlighted, with a numeric payload (or `payload.value`) as a badge on the receiving node.
 *
 * @param structure - The structure the messages pass over (node ids match the messages' `from` and `to`).
 * @param run - The trace.
 * @param options - Message kind, frames per second and caption.
 * @returns The timeline, one frame per step with messages.
 * @example
 * ```ts
 * import { messageFrames } from 'whitebox-render/trace'
 * import { structure } from 'whitebox-engine/scene'
 * const s = structure('g', [{ id: 'a', label: 'a', role: 'latent' }, { id: 'b', label: 'b', role: 'latent' }], [{ from: 'a', to: 'b' }])
 * const run = { schema: 'whitebox.run/1' as const, events: [{ type: 'message' as const, runId: 'r', step: 2, kind: 'm', from: 'a', to: 'b', payload: 0.5 }] }
 * messageFrames(s, run).frames[0].overlay.badges // => { b: { message: 0.5 } }
 * ```
 */
export function messageFrames(
  structure: Structure,
  run: Run,
  options: { readonly kind?: string; readonly fps?: number } = {},
): Timeline & { readonly frames: readonly (Frame & { readonly kind: 'structure' })[] } {
  const bySteps = new Map<number, ReturnType<typeof messages>>()
  for (const m of messages(run, options.kind)) bySteps.set(m.step, [...(bySteps.get(m.step) ?? []), m])
  const frames = [...bySteps.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, ms]) => {
      const active = new Set<string>()
      const edges: { from: string; to: string }[] = []
      const badges: Record<string, Record<string, string | number>> = {}
      for (const m of ms) {
        if (m.from !== undefined) active.add(m.from)
        if (m.to !== undefined) active.add(m.to)
        if (m.from !== undefined && m.to !== undefined) edges.push({ from: m.from, to: m.to })
        const p = m.payload
        const field =
          p !== null && typeof p === 'object' && !Array.isArray(p)
            ? (p as Readonly<Record<string, unknown>>)['value']
            : undefined
        const value = typeof p === 'number' ? p : typeof field === 'number' ? field : undefined
        if (value !== undefined && m.to !== undefined) badges[m.to] = { ...badges[m.to], message: value }
      }
      const overlay: StructureOverlay = { active: [...active], edges, badges }
      return { kind: 'structure' as const, step, structure, overlay, caption: `step ${step}` }
    })
  return timeline(frames, { fps: options.fps ?? 4 }) as Timeline & {
    readonly frames: readonly (Frame & { readonly kind: 'structure' })[]
  }
}
