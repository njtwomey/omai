/**
 * Structure diagrams: `diagram` lays out any engine `Structure` (layered by default, a tidy tree for trees, or
 * circular, force-directed, bipartite or grid) and draws it as a scene, with node shapes, colours and labels by role,
 * routed edges with arrowheads and labels, self-loops, groups such as plates and repeated blocks, and node sizes held
 * steady across an animation's frames.
 */

import { maxOf, minOf } from 'mdml-engine/base'
import {
  circle,
  group as groupNode,
  path,
  rect,
  type Scene,
  type SceneNode,
  type Structure,
  type StructureEdge,
  type StructureNode,
  type StructureOverlay,
  scene,
  tex,
  text,
} from 'mdml-engine/scene'
import { circularLayout, forceLayout } from '../layout/force.ts'
import { type Box, boxBoundary, type EdgeRoute, type GraphLayout, type LayoutGraph } from '../layout/graph.ts'
import { bipartiteLayout } from '../layout/grids.ts'
import { layered } from '../layout/layered.ts'
import { tidyTree } from '../layout/trees.ts'
import { type NodeLook, type StructureStyle, styleFor } from './styles.ts'

/** Options of {@link diagram}. */
export interface DiagramOptions {
  /**
   * The layout (default the style's: layered for most kinds, a tidy tree for trees, force for networks); `bipartite`
   * puts factors in one row and the variables in another, as factor graphs are drawn.
   */
  readonly layout?: 'layered' | 'tree' | 'circular' | 'force' | 'bipartite'
  /** Ranks or depths down the page or across it (default the structure's own preference, else down). */
  readonly direction?: 'down' | 'right'
  /** The visual language (default the one for the structure's type). */
  readonly style?: StructureStyle
  /** State laid over the diagram: active nodes and edges, value badges (one frame of an animation). */
  readonly overlay?: StructureOverlay
  /**
   * Every frame of an animation the overlay comes from: nodes are sized for the largest badges any frame gives them,
   * so the layout stays put while badges come and go (default: sized for the overlay alone).
   */
  readonly frames?: readonly StructureOverlay[]
  /** The seed of a force layout (default 0). */
  readonly seed?: number
  /** Space between ranks (default 56) and between neighbours (default 28). */
  readonly rankSep?: number
  readonly nodeSep?: number
}

const CHAR = 0.6

function lines(
  n: StructureNode,
  badges: Readonly<Record<string, string | number>> | undefined,
  look: NodeLook,
): string[] {
  const out = [n.label]
  // Round shapes keep to their label (and badges); their detail shows on hover.
  const compact = look.shape === 'circle' || look.shape === 'dot' || look.shape === 'square'
  if (n.detail !== undefined && !compact) out.push(n.detail)
  if (badges !== undefined && Object.keys(badges).length > 0)
    out.push(
      Object.entries(badges)
        .map(([k, v]) => `${k} ${typeof v === 'number' ? Number(v.toPrecision(4)) : v}`)
        .join(' · '),
    )
  return out
}

const isTex = (s: string): boolean => s.length > 2 && s.startsWith('$') && s.endsWith('$')
const textWidth = (s: string, size: number): number =>
  isTex(s) ? (s.length - 2) * size * 0.45 : s.length * size * CHAR

function nodeSize(look: NodeLook, content: readonly string[], size: number): { width: number; height: number } {
  const wide = maxOf(content.map((s, i) => textWidth(s, i === 0 ? size : size - 2)))
  const tall = size + 6 + (content.length - 1) * (size + 2)
  switch (look.shape) {
    case 'dot':
      return { width: 12, height: 12 }
    case 'square':
      return { width: 14, height: 14 }
    case 'circle': {
      const d = Math.max(36, wide + 16, tall + 10)
      return { width: d, height: d }
    }
    case 'ellipse':
      return { width: Math.max(56, wide * 1.25 + 20), height: Math.max(32, tall + 12) }
    case 'diamond':
      return { width: Math.max(56, wide * 1.6 + 16), height: Math.max(40, tall * 1.6) }
    default:
      return { width: Math.max(40, wide + 20), height: Math.max(28, tall + 8) }
  }
}

function shapeNodes(look: NodeLook, b: Box, active: boolean): SceneNode[] {
  const stroke = active ? 'accent' : look.stroke
  const style = {
    fill: look.fill,
    stroke,
    strokeWidth: active ? 2.5 : (look.strokeWidth ?? 1.2),
    ...(look.dash === undefined ? {} : { dash: look.dash }),
  }
  const x0 = b.x - b.width / 2
  const y0 = b.y - b.height / 2
  switch (look.shape) {
    case 'circle':
    case 'dot': {
      const r = Math.min(b.width, b.height) / 2
      const outer = circle(b.x, b.y, r, style)
      return look.double === true ? [outer, circle(b.x, b.y, r - 4, { fill: 'none', stroke, strokeWidth: 1 })] : [outer]
    }
    case 'ellipse':
      return [{ kind: 'ellipse', cx: b.x, cy: b.y, rx: b.width / 2, ry: b.height / 2, ...style }]
    case 'diamond':
      return [
        path(
          [
            [b.x, y0],
            [x0 + b.width, b.y],
            [b.x, y0 + b.height],
            [x0, b.y],
          ],
          { closed: true, ...style },
        ),
      ]
    case 'square':
    case 'rect':
      return [rect(x0, y0, b.width, b.height, style)]
    case 'round':
      return [rect(x0, y0, b.width, b.height, { radius: 7, ...style })]
  }
}

function labelNodes(look: NodeLook, content: readonly string[], b: Box, size: number): SceneNode[] {
  // Factors and dots carry their label outside, below.
  const outside = look.shape === 'square' || look.shape === 'dot'
  const top = outside ? b.y + b.height / 2 + size : b.y - ((content.length - 1) * (size + 2)) / 2
  return content.map((s, i) => {
    const y = top + i * (size + 2)
    const fill = i === 0 && !outside ? look.text : 'muted'
    const sz = i === 0 ? size : size - 2
    return isTex(s)
      ? tex(b.x, y, s.slice(1, -1), { size: sz + 2, fill, anchor: 'middle' })
      : text(b.x, y, s, { size: sz, fill, anchor: 'middle', baseline: 'middle' })
  })
}

/** The point where a ray from a node's centre at `angle` leaves its outline. */
function outlineAt(look: NodeLook, b: Box, angle: number): [number, number] {
  const [ux, uy] = [Math.cos(angle), Math.sin(angle)]
  if (look.shape === 'circle' || look.shape === 'dot') {
    const r = Math.min(b.width, b.height) / 2
    return [b.x + r * ux, b.y + r * uy]
  }
  if (look.shape === 'ellipse') {
    const [a, c] = [b.width / 2, b.height / 2]
    const t = 1 / Math.hypot(ux / a, uy / c)
    return [b.x + t * ux, b.y + t * uy]
  }
  return boxBoundary(b, [b.x + ux * 1e4, b.y + uy * 1e4])
}

/**
 * A self-loop on a node: a teardrop leaving and re-entering its outline on the side freest of other edges (above,
 * when free), with the point for its label just beyond the loop's far end.
 */
function selfLoop(
  look: NodeLook,
  b: Box,
  layout: GraphLayout,
  id: string,
): { points: [number, number][]; label: [number, number]; direction: [number, number] } {
  // Directions in which other edges leave or enter the node.
  const taken: number[] = []
  for (const r of layout.edges) {
    const end = r.from === id ? r.points[0] : r.to === id ? r.points[r.points.length - 1] : undefined
    if (end !== undefined) taken.push(Math.atan2(end[1] - b.y, end[0] - b.x))
  }
  const gap = (a: number): number =>
    Math.min(Math.PI, minOf(taken.map((t) => Math.abs(Math.atan2(Math.sin(a - t), Math.cos(a - t))))))
  // Up first, then the sides, then down; the first with the widest berth wins.
  const candidates = [-90, -45, -135, 0, 180, 45, 135, 90].map((d) => (d * Math.PI) / 180)
  let angle = candidates[0] as number
  for (const c of candidates) if (gap(c) > gap(angle) + 1e-6) angle = c
  const reach = Math.max(22, Math.min(b.width, b.height) * 0.7)
  const spread = 0.42
  const [x0, y0] = outlineAt(look, b, angle - spread)
  const [x1, y1] = outlineAt(look, b, angle + spread)
  const centreOut = (t: number, d: number): [number, number] => {
    const [ox, oy] = outlineAt(look, b, t)
    const r = Math.hypot(ox - b.x, oy - b.y) + d
    return [b.x + r * Math.cos(t), b.y + r * Math.sin(t)]
  }
  const points: [number, number][] = [
    [x0, y0],
    centreOut(angle - spread * 1.25, reach * 0.65),
    centreOut(angle, reach),
    centreOut(angle + spread * 1.25, reach * 0.65),
    [x1, y1],
  ]
  const direction: [number, number] = [Math.cos(angle), Math.sin(angle)]
  return { points, label: centreOut(angle, reach + 9), direction }
}

/** Every group's members, with the members of its descendant groups. */
function groupMembers(s: Structure): Map<string, Set<string>> {
  const parent = new Map(s.groups.map((g) => [g.id, g.parent]))
  const out = new Map<string, Set<string>>(s.groups.map((g) => [g.id, new Set<string>()]))
  for (const n of s.nodes) {
    let g = n.group
    const seen = new Set<string>()
    while (g !== undefined && !seen.has(g)) {
      seen.add(g)
      out.get(g)?.add(n.id)
      g = parent.get(g)
    }
  }
  return out
}

function depthOf(s: Structure, id: string): number {
  const parent = new Map(s.groups.map((g) => [g.id, g.parent]))
  let d = 0
  let g = parent.get(id)
  while (g !== undefined && d < 32) {
    d++
    g = parent.get(g)
  }
  return d
}

/**
 * Lay out a structure and draw it as a scene: the layout (layered by default, a tidy tree for trees), node shapes,
 * colours and labels by role in the structure's visual language (`styleFor`), edges routed with arrow heads and labels,
 * self-loops as teardrops on the node's freest side, groups as plates (with their count), repeated blocks and clusters
 * around their members (nested by `parent`), and an optional overlay of active nodes and edges and value badges (with
 * `frames`, nodes are sized for every frame's badges, so an animation never moves the layout). Labels written `$...$`
 * are typeset as TeX.
 *
 * @param s - The structure.
 * @param options - Layout, direction, style, overlay (and the frames it comes from), seed and spacing.
 * @returns The scene, with node ids as scene ids (edges are `edge:<index>`, groups `group:<id>`).
 * @index function Draw a Structure as a scene: `diagram(structure, { layout: 'layered' })`.
 * @example
 * ```ts
 * import { structure } from 'mdml-engine/scene'
 * import { diagram } from 'mdml-render/structures'
 * const s = structure('bayes-net', [{ id: 'z', label: 'z', role: 'latent' }, { id: 'x', label: 'x', role: 'observed' }], [{ from: 'z', to: 'x' }])
 * diagram(s).kind // => 'scene'
 * ```
 */
export function diagram(s: Structure, options: DiagramOptions = {}): Scene {
  const style = options.style ?? styleFor(s.type)
  const size = style.fontSize
  const overlay = options.overlay
  const active = new Set(overlay?.active ?? [])
  const activeEdges = new Set((overlay?.edges ?? []).map((e) => `${e.from}\u0000${e.to}`))
  const looks = new Map(s.nodes.map((n) => [n.id, style.node(n.role)]))
  const content = new Map(
    s.nodes.map((n) => [n.id, lines(n, overlay?.badges?.[n.id] ?? n.badges, looks.get(n.id) as NodeLook)]),
  )
  // A node's box fits its content in this frame and, given the animation's frames, in every one of them.
  const sizeOf = (n: StructureNode): { width: number; height: number } => {
    const look = looks.get(n.id) as NodeLook
    let box = nodeSize(look, content.get(n.id) as string[], size)
    for (const f of options.frames ?? []) {
      const other = nodeSize(look, lines(n, f.badges?.[n.id] ?? n.badges, look), size)
      box = { width: Math.max(box.width, other.width), height: Math.max(box.height, other.height) }
    }
    return box
  }
  // Self-loops are drawn here, not by the layout: the layout routes the other edges.
  const routed = s.edges.filter((e) => e.from !== e.to)
  const graph: LayoutGraph = {
    nodes: s.nodes.map((n) => ({
      id: n.id,
      ...sizeOf(n),
      ...(n.group === undefined ? {} : { group: n.group }),
    })),
    edges: routed.map((e) => ({ from: e.from, to: e.to })),
    groups: s.groups.map((g) => ({ id: g.id, ...(g.parent === undefined ? {} : { parent: g.parent }) })),
  }
  const direction = options.direction ?? s.direction ?? 'down'
  const which = options.layout ?? style.layout
  const layout: GraphLayout =
    which === 'tree'
      ? tidyTree(graph, { direction })
      : which === 'circular'
        ? circularLayout(graph)
        : which === 'force'
          ? forceLayout(graph, { seed: options.seed ?? 0 })
          : which === 'bipartite'
            ? bipartiteLayout(
                graph,
                s.nodes.filter((n) => n.role !== 'factor').map((n) => n.id),
                { orientation: direction === 'right' ? 'columns' : 'rows', gap: options.rankSep ?? 110 },
              )
            : layered(graph, {
                direction,
                ...(options.rankSep === undefined ? {} : { rankSep: options.rankSep }),
                ...(options.nodeSep === undefined ? {} : { nodeSep: options.nodeSep }),
              })

  // Groups: deepest first, each around its members and its child groups, with room for its label.
  const members = groupMembers(s)
  const boxes = new Map<string, [number, number, number, number]>()
  const ordered = [...s.groups].sort((a, b) => depthOf(s, b.id) - depthOf(s, a.id))
  for (const g of ordered) {
    let [x0, y0, x1, y1] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, -Infinity, -Infinity]
    for (const id of members.get(g.id) ?? []) {
      const b = layout.nodes[id]
      if (b === undefined) continue
      x0 = Math.min(x0, b.x - b.width / 2)
      y0 = Math.min(y0, b.y - b.height / 2)
      x1 = Math.max(x1, b.x + b.width / 2)
      y1 = Math.max(y1, b.y + b.height / 2)
    }
    for (const child of s.groups.filter((c) => c.parent === g.id)) {
      const c = boxes.get(child.id)
      if (c === undefined) continue
      x0 = Math.min(x0, c[0])
      y0 = Math.min(y0, c[1])
      x1 = Math.max(x1, c[2])
      y1 = Math.max(y1, c[3])
    }
    if (Number.isFinite(x0)) boxes.set(g.id, [x0 - 12, y0 - 24, x1 + 12, y1 + (g.count === undefined ? 12 : 22)])
  }
  const groupScene: SceneNode[] = [...s.groups]
    .sort((a, b) => depthOf(s, a.id) - depthOf(s, b.id))
    .flatMap((g) => {
      const b = boxes.get(g.id)
      if (b === undefined) return []
      const look = style.group(g)
      const [x0, y0, x1, y1] = b
      const out: SceneNode[] = [
        rect(x0, y0, x1 - x0, y1 - y0, {
          id: `group:${g.id}`,
          fill: look.fill,
          stroke: look.stroke,
          radius: look.radius,
          ...(look.dash === undefined ? {} : { dash: look.dash }),
        }),
        text(x0 + 8, y0 + 14, g.label, { size: size - 1, fill: 'muted' }),
      ]
      if (g.count !== undefined)
        out.push(
          text(x1 - 8, y1 - 7, g.kind === 'repeat' ? `× ${g.count}` : String(g.count), {
            size: size - 1,
            fill: 'fg',
            anchor: 'end',
          }),
        )
      return out
    })

  // An edge with a twin in the other direction bends to its left, so the pair (and their labels) separate.
  const pairs = new Set(s.edges.map((e) => `${e.from}\u0000${e.to}`))
  const bend = (points: readonly (readonly [number, number])[]): (readonly [number, number])[] => {
    const a = points[0]
    const b = points[points.length - 1]
    if (a === undefined || b === undefined || points.length !== 2) return [...points]
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const len = Math.hypot(dx, dy) || 1
    const off = 10
    return [a, [(a[0] + b[0]) / 2 + (dy / len) * off, (a[1] + b[1]) / 2 - (dx / len) * off], b]
  }
  // Each route belongs to the next routed edge with its ends (layouts skip edges whose ends are missing).
  const routeOf = new Map<StructureEdge, EdgeRoute>()
  let next = 0
  for (const r of layout.edges) {
    while (next < routed.length && !(routed[next]?.from === r.from && routed[next]?.to === r.to)) next++
    const e = routed[next]
    if (e === undefined) break
    routeOf.set(e, r)
    next++
  }
  const loopExtent: [number, number][] = []
  const edgeScene: SceneNode[] = s.edges.flatMap((e, i) => {
    const look = style.edge(e)
    const on = activeEdges.has(`${e.from}\u0000${e.to}`)
    if (e.from === e.to) {
      const b = layout.nodes[e.from]
      if (b === undefined) return []
      const loop = selfLoop(looks.get(e.from) as NodeLook, b, layout, e.from)
      for (const p of loop.points) loopExtent.push([p[0], p[1]])
      const out: SceneNode[] = [
        path(loop.points, {
          id: `edge:${i}`,
          stroke: on ? 'accent' : look.stroke,
          strokeWidth: on ? 2.5 : (look.strokeWidth ?? 1.2),
          arrow: look.arrow,
          smooth: true,
          ...(look.dash === undefined ? {} : { dash: look.dash }),
        }),
      ]
      if (e.label !== undefined) {
        const [lx, ly] = loop.label
        const w = textWidth(e.label, size - 2)
        const anchor = Math.abs(loop.direction[0]) < 0.4 ? 'middle' : loop.direction[0] > 0 ? 'start' : 'end'
        loopExtent.push([lx - w, ly - size], [lx + w, ly + size])
        out.push(text(lx, ly, e.label, { size: size - 2, fill: 'muted', anchor, baseline: 'middle' }))
      }
      return out
    }
    const original = routeOf.get(e)
    if (original === undefined) return []
    const route = pairs.has(`${e.to}\u0000${e.from}`) ? { ...original, points: bend(original.points) } : original
    const out: SceneNode[] = [
      path(route.points, {
        id: `edge:${i}`,
        stroke: on ? 'accent' : look.stroke,
        strokeWidth: on ? 2.5 : (look.strokeWidth ?? 1.2),
        arrow: look.arrow,
        smooth: route.points.length > 2,
        ...(look.dash === undefined ? {} : { dash: look.dash }),
      }),
    ]
    if (e.label !== undefined) {
      const mid = route.points[Math.floor((route.points.length - 1) / 2)]
      const next = route.points[Math.floor((route.points.length - 1) / 2) + 1] ?? mid
      if (mid !== undefined && next !== undefined)
        out.push(
          text((mid[0] + next[0]) / 2 + 4, (mid[1] + next[1]) / 2 - 4, e.label, { size: size - 2, fill: 'muted' }),
        )
    }
    return out
  })

  const nodeScene: SceneNode[] = s.nodes.flatMap((n) => {
    const b = layout.nodes[n.id]
    if (b === undefined) return []
    const look = looks.get(n.id) as NodeLook
    return [
      groupNode([...shapeNodes(look, b, active.has(n.id)), ...labelNodes(look, content.get(n.id) ?? [], b, size)], {
        id: n.id,
        title: [n.label, n.detail].filter((x) => x !== undefined).join(': '),
      }),
    ]
  })

  const notes: SceneNode[] = (s.annotations ?? []).flatMap((a) => {
    const target = a.target === undefined ? undefined : (layout.nodes[a.target] ?? undefined)
    const fill = a.kind === 'warning' ? 'tab3' : a.kind === 'highlight' ? 'accent' : 'muted'
    if (target === undefined) return [text(8, 14, a.text, { size: size - 1, fill })]
    return [text(target.x + target.width / 2 + 6, target.y - target.height / 2, a.text, { size: size - 2, fill })]
  })

  // Groups and notes can reach past the layout's own box: grow the scene to hold them.
  let [minX, minY, maxX, maxY] = [0, 0, layout.width, layout.height]
  for (const [x0, y0, x1, y1] of boxes.values()) {
    minX = Math.min(minX, x0 - 8)
    minY = Math.min(minY, y0 - 8)
    maxX = Math.max(maxX, x1 + 8)
    maxY = Math.max(maxY, y1 + 8)
  }
  for (const [x, y] of loopExtent) {
    minX = Math.min(minX, x - 6)
    minY = Math.min(minY, y - 6)
    maxX = Math.max(maxX, x + 6)
    maxY = Math.max(maxY, y + 6)
  }
  const children = [...groupScene, ...edgeScene, ...nodeScene, ...notes]
  return scene(
    maxX - minX,
    maxY - minY,
    minX === 0 && minY === 0 ? children : [groupNode(children, { translate: [-minX, -minY] })],
    { ...(s.title === undefined ? {} : { title: s.title }) },
  )
}
