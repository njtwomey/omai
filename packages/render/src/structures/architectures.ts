import {
  arrow,
  rect,
  type Scene,
  type SceneNode,
  type Structure,
  type StructureNode,
  scene,
  text,
} from 'whitebox-engine/scene'
import { type DiagramOptions, diagram } from './diagram.ts'

/** The views of a network's structure. */
export type ArchitectureView = 'diagram' | 'compact' | 'shapes' | 'params'

/** Options of {@link architecture}. */
export interface ArchitectureOptions extends Omit<DiagramOptions, 'layout'> {
  /** The drawing's width for the shape flow and the treemap (default 560). */
  readonly width?: number
}

/** The numbers of a shape badge (`[8, 16]` to `[8, 16]`), or `undefined` when there is none. */
function shapeOf(n: StructureNode): number[] | undefined {
  const s = n.badges?.['shape']
  if (typeof s !== 'string') return undefined
  const nums = s.match(/-?\d+/g)?.map(Number)
  return nums === undefined || nums.length === 0 ? undefined : nums
}

const paramsOf = (n: StructureNode): number => {
  const p = n.badges?.['params']
  return typeof p === 'number' ? p : 0
}

/** The nodes in the order data flows (each after the nodes feeding it; ties in the structure's order). */
export function flowOrder(s: Structure): StructureNode[] {
  const preds = new Map<string, string[]>()
  for (const e of s.edges) if (e.kind !== 'skip') preds.set(e.to, [...(preds.get(e.to) ?? []), e.from])
  const done = new Set<string>()
  const out: StructureNode[] = []
  const byId = new Map(s.nodes.map((n) => [n.id, n]))
  const visit = (id: string, seen: Set<string>): void => {
    if (done.has(id) || seen.has(id)) return
    seen.add(id)
    for (const p of preds.get(id) ?? []) visit(p, seen)
    done.add(id)
    const n = byId.get(id)
    if (n !== undefined) out.push(n)
  }
  for (const n of s.nodes) visit(n.id, new Set())
  return out
}

const compactNumber = (v: number): string =>
  v >= 1e6 ? `${(v / 1e6).toPrecision(3)}M` : v >= 1e3 ? `${(v / 1e3).toPrecision(3)}k` : String(v)

/** The tensor-shape flow: each layer a bar as wide as (the log of) its output's size per example, in flow order. */
function shapeFlow(s: Structure, width: number): Scene {
  // A composite block's listed parts sit beside the flow, not in it.
  const nodes = flowOrder(s).filter((n) => n.role !== 'internal')
  const sizes = nodes.map((n) => {
    const shape = shapeOf(n)
    if (shape === undefined) return 1
    const perExample = shape.length > 1 ? shape.slice(1) : shape
    return perExample.reduce((a, b) => a * Math.max(b, 1), 1)
  })
  const most = Math.max(2, ...sizes)
  const row = 30
  const gap = 14
  const labelWidth = 150
  const barSpace = width - labelWidth - 24
  const children: SceneNode[] = []
  nodes.forEach((n, i) => {
    const y = 12 + i * (row + gap)
    const w = Math.max(6, (Math.log2(1 + (sizes[i] ?? 1)) / Math.log2(1 + most)) * barSpace)
    const x = labelWidth + (barSpace - w) / 2
    const fill = n.role === 'input' ? 'tab0' : n.role === 'output' ? 'tab1' : paramsOf(n) > 0 ? 'tab2' : 'tab7'
    children.push(
      rect(x, y, w, row, {
        id: n.id,
        fill,
        opacity: 0.85,
        radius: 4,
        title: `${n.label} ${n.badges?.['shape'] ?? ''}`,
      }),
      text(labelWidth - 10, y + row / 2, n.label, { anchor: 'end', baseline: 'middle', size: 12 }),
      text(x + w / 2, y + row / 2, String(n.badges?.['shape'] ?? ''), {
        anchor: 'middle',
        baseline: 'middle',
        size: 11,
        fill: 'bg',
      }),
    )
    if (i + 1 < nodes.length) {
      const cx = labelWidth + barSpace / 2
      children.push(arrow(cx, y + row + 1, cx, y + row + gap - 1, { stroke: 'muted' }))
    }
  })
  return scene(width, 24 + nodes.length * (row + gap) - gap, children, { title: 'tensor shapes' })
}

interface Item {
  readonly node: StructureNode
  readonly value: number
}

/** Squarified treemap (Bruls, Huizing and van Wijk): rectangles of the given areas, close to square, filling a box. */
function squarify(
  items: readonly Item[],
  x: number,
  y: number,
  w: number,
  h: number,
): [Item, number, number, number, number][] {
  const total = items.reduce((a, b) => a + b.value, 0)
  if (items.length === 0 || total <= 0 || w <= 0 || h <= 0) return []
  const scale = (w * h) / total
  const out: [Item, number, number, number, number][] = []
  let rest = [...items].sort((a, b) => b.value - a.value)
  let [bx, by, bw, bh] = [x, y, w, h]
  const worst = (row: readonly Item[], side: number): number => {
    const areas = row.map((r) => r.value * scale)
    const sum = areas.reduce((a, b) => a + b, 0)
    const max = Math.max(...areas)
    const min = Math.min(...areas)
    return Math.max((side * side * max) / (sum * sum), (sum * sum) / (side * side * min))
  }
  while (rest.length > 0) {
    const side = Math.min(bw, bh)
    let row: Item[] = [rest[0] as Item]
    let i = 1
    while (i < rest.length && worst([...row, rest[i] as Item], side) <= worst(row, side)) {
      row = [...row, rest[i] as Item]
      i++
    }
    rest = rest.slice(i)
    const area = row.reduce((a, r) => a + r.value * scale, 0)
    if (bw >= bh) {
      const colW = area / bh
      let cy = by
      for (const r of row) {
        const rh = (r.value * scale) / colW
        out.push([r, bx, cy, colW, rh])
        cy += rh
      }
      bx += colW
      bw -= colW
    } else {
      const rowH = area / bw
      let cx = bx
      for (const r of row) {
        const rw = (r.value * scale) / rowH
        out.push([r, cx, by, rw, rowH])
        cx += rw
      }
      by += rowH
      bh -= rowH
    }
  }
  return out
}

/** The parameter-count treemap: one rectangle per layer with parameters, its area its count. */
function paramTreemap(s: Structure, width: number): Scene {
  // When a composite block's parts are listed, count the parts rather than the block, so nothing counts twice.
  const expanded = s.nodes.some((n) => n.role === 'internal')
  const items = s.nodes
    .filter((n) => paramsOf(n) > 0 && !(expanded && n.data?.['parts'] !== undefined))
    .map((node) => ({ node, value: paramsOf(node) }))
  const total = items.reduce((a, b) => a + b.value, 0)
  const height = Math.round(width * 0.6)
  const cells = squarify(items, 0, 22, width, height - 22)
  const children: SceneNode[] = [
    text(0, 12, `${compactNumber(total)} parameters`, { size: 12, fill: 'muted', baseline: 'middle' }),
  ]
  cells.forEach(([item, x, y, w, h], i) => {
    children.push(
      rect(x + 1, y + 1, Math.max(0, w - 2), Math.max(0, h - 2), {
        id: item.node.id,
        fill: `tab${i % 10}` as 'tab0',
        opacity: 0.8,
        radius: 3,
        title: `${item.node.label}: ${item.value} parameters`,
      }),
    )
    if (w > 54 && h > 30) {
      children.push(
        text(x + 6, y + 15, item.node.label, { size: 11, fill: 'bg', weight: 'bold' }),
        text(x + 6, y + 29, `${compactNumber(item.value)} · ${((100 * item.value) / total).toFixed(1)}%`, {
          size: 10,
          fill: 'bg',
        }),
      )
    }
  })
  return scene(width, height, children, { title: 'parameters' })
}

/**
 * Views of a network's structure (as `describe` in `whitebox-engine/nn` gives it): the full `diagram` (blocks and
 * repeats as groups, skip connections, shapes on the edges), a `compact` stack (one box per layer, no groups or edge
 * labels), the tensor-shape flow (`shapes`: each layer a bar sized by its output per example), and a treemap of where
 * the parameters are (`params`).
 *
 * @param s - The network's structure.
 * @param view - Which view.
 * @param options - Diagram options and the width of the shape flow and treemap.
 * @returns The scene.
 * @example
 * ```ts
 * import { describe, mlp } from 'whitebox-engine/nn'
 * import { architecture } from 'whitebox-render/structures'
 * architecture(describe(mlp([2, 4, 1]), [8, 2]), 'params').title // => 'parameters'
 * ```
 */
export function architecture(s: Structure, view: ArchitectureView, options: ArchitectureOptions = {}): Scene {
  const width = options.width ?? 560
  switch (view) {
    case 'diagram':
      return diagram(s, { direction: 'down', ...options, layout: 'layered' })
    case 'compact':
      return diagram(
        {
          ...s,
          groups: [],
          nodes: s.nodes.map(({ group: _, detail: __, ...n }) => n),
          edges: s.edges.map(({ label: _, ...e }) => e),
        },
        { direction: 'down', ...options, layout: 'layered', rankSep: 28 },
      )
    case 'shapes':
      return shapeFlow(s, width)
    case 'params':
      return paramTreemap(s, width)
  }
}
