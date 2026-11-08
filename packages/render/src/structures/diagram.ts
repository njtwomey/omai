/**
 * Structure diagrams: `diagram` lays out any engine `Structure` (layered by default, a tidy tree for trees, or
 * circular, force-directed, bipartite or grid) and draws it as a scene, with node shapes, colours and labels by role,
 * routed edges with arrowheads and labels, self-loops, groups such as plates and repeated blocks, and node sizes held
 * steady across an animation's frames.
 */

import { maxOf, minOf } from 'omai-engine/base'
import {
  circle,
  group as groupNode,
  type Paint,
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
} from 'omai-engine/scene'
import { circularLayout, forceLayout } from '../layout/force.ts'
import {
  boxBoundary,
  type EdgeRoute,
  type GraphLayout,
  type LayoutBox,
  type LayoutGraph,
  manualLayout,
} from '../layout/graph.ts'
import { bipartiteLayout } from '../layout/grids.ts'
import { layered } from '../layout/layered.ts'
import { tidyTree } from '../layout/trees.ts'
import { type EdgeLook, type NodeLook, type StructureStyle, styleFor } from './styles.ts'

/** Options of {@link diagram}. */
export interface DiagramOptions {
  /**
   * The layout (default the style's: layered for most kinds, a tidy tree for trees, force for networks); `bipartite`
   * puts factors in one row and the variables in another, as factor graphs are drawn; `manual` puts every node at its
   * `position` (the default when every node has one) and throws `OmaiError` `value` for a node without.
   */
  readonly layout?: 'layered' | 'tree' | 'circular' | 'force' | 'bipartite' | 'manual'
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
  /**
   * Space between ranks (default 56) and between neighbours (default 28). Layered honours both; tree both (as the
   * space between depths and between siblings); bipartite both (the gap between the two rows, and the space between
   * nodes along a row); circular `nodeSep` (the gap between neighbours on the circle); force `rankSep` (the ideal edge
   * length); manual neither.
   */
  readonly rankSep?: number
  readonly nodeSep?: number
}

const CHAR = 0.6

/** One line of a node's text: the label's lines are `main` (the label's size and colour), the rest small and muted. */
interface Line {
  readonly text: string
  readonly main: boolean
}

function lines(
  n: StructureNode,
  badges: Readonly<Record<string, string | number>> | undefined,
  look: NodeLook,
): Line[] {
  const out: Line[] = n.label.split('\n').map((text) => ({ text, main: true }))
  // Round shapes keep to their label (and badges); their detail shows on hover.
  const compact = look.shape === 'circle' || look.shape === 'dot' || look.shape === 'square'
  if (n.detail !== undefined && !compact) out.push({ text: n.detail, main: false })
  if (badges !== undefined && Object.keys(badges).length > 0)
    out.push({
      text: Object.entries(badges)
        .map(([k, v]) => `${k} ${typeof v === 'number' ? Number(v.toPrecision(4)) : v}`)
        .join(' · '),
      main: false,
    })
  return out
}

const isTex = (s: string): boolean => s.length > 2 && s.startsWith('$') && s.endsWith('$')
const textWidth = (s: string, size: number): number =>
  isTex(s) ? (s.length - 2) * size * 0.45 : s.length * size * CHAR

/** The room the boxes behind the first take in a `stack`: two more, each offset by this much up and to the right. */
const STACK = 4

function nodeSize(look: NodeLook, content: readonly Line[], size: number): { width: number; height: number } {
  const wide = maxOf(content.map((l) => textWidth(l.text, l.main ? size : size - 2)))
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
    case 'none':
      return { width: wide + 8, height: tall }
    case 'stack':
      return { width: Math.max(40, wide + 20) + 2 * STACK, height: Math.max(28, tall + 8) + 2 * STACK }
    case 'rect':
    case 'round':
      return { width: Math.max(40, wide + 20), height: Math.max(28, tall + 8) }
  }
}

/** The front box of a node: the whole box, or for a `stack` the box without the room its back boxes take. */
function frontOf(look: NodeLook, b: LayoutBox): LayoutBox {
  return look.shape === 'stack'
    ? { x: b.x - STACK, y: b.y + STACK, width: b.width - 2 * STACK, height: b.height - 2 * STACK }
    : b
}

function shapeNodes(look: NodeLook, b: LayoutBox, active: boolean): SceneNode[] {
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
    case 'none':
      // Text only: an invisible box the size of the text, so the node can still be hovered and picked.
      return [
        rect(x0, y0, b.width, b.height, { fill: 'bg', stroke: active ? 'accent' : 'none', opacity: active ? 1 : 0 }),
      ]
    case 'stack': {
      // Back boxes first, each peeking out up and to the right of the one in front.
      const f = frontOf(look, b)
      return [2, 1, 0].map((k) =>
        rect(f.x - f.width / 2 + k * STACK, f.y - f.height / 2 - k * STACK, f.width, f.height, { radius: 3, ...style }),
      )
    }
  }
}

/** Whether a look carries its label outside the shape, below it (factors and dots). */
const isOutside = (look: NodeLook): boolean => look.shape === 'square' || look.shape === 'dot'

/** The vertical centre of each line of a node's text. */
function lineYs(look: NodeLook, count: number, b: LayoutBox, size: number): number[] {
  const c = frontOf(look, b)
  const top = isOutside(look) ? b.y + b.height / 2 + size : c.y - ((count - 1) * (size + 2)) / 2
  return Array.from({ length: count }, (_, i) => top + i * (size + 2))
}

function labelNodes(look: NodeLook, content: readonly Line[], b: LayoutBox, size: number): SceneNode[] {
  const outside = isOutside(look)
  const ys = lineYs(look, content.length, b, size)
  const x = frontOf(look, b).x
  return content.map((line, i) => {
    const y = ys[i] as number
    const fill = line.main && !outside ? look.text : 'muted'
    const sz = line.main ? size : size - 2
    return isTex(line.text)
      ? tex(x, y, line.text.slice(1, -1), { size: sz + 2, fill, anchor: 'middle' })
      : text(x, y, line.text, { size: sz, fill, anchor: 'middle', baseline: 'middle' })
  })
}

/** The point where a ray from a node's centre at `angle` leaves its outline. */
function outlineAt(look: NodeLook, b: LayoutBox, angle: number): [number, number] {
  const [ux, uy] = [Math.cos(angle), Math.sin(angle)]
  switch (look.shape) {
    case 'circle':
    case 'dot': {
      const r = Math.min(b.width, b.height) / 2
      return [b.x + r * ux, b.y + r * uy]
    }
    case 'ellipse': {
      const [a, c] = [b.width / 2, b.height / 2]
      const t = 1 / Math.hypot(ux / a, uy / c)
      return [b.x + t * ux, b.y + t * uy]
    }
    case 'diamond':
    case 'square':
    case 'rect':
    case 'round':
    case 'none':
    case 'stack':
      return boxBoundary(b, [b.x + ux * 1e4, b.y + uy * 1e4])
  }
}

/**
 * A self-loop on a node: a teardrop leaving and re-entering its outline on the side freest of other edges (above,
 * when free), with the point for its label just beyond the loop's far end.
 */
function selfLoop(
  look: NodeLook,
  b: LayoutBox,
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

/** A box as `[x0, y0, x1, y1]`. */
type Rect = readonly [number, number, number, number]

const overlaps = (a: Rect, b: Rect): boolean => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]

/** The point a fraction `t` (0 to 1) of the way along a polyline, by length. */
function pointAlong(points: readonly (readonly [number, number])[], t: number): [number, number] {
  const first = points[0]
  if (first === undefined) return [0, 0]
  const lengths = points
    .slice(1)
    .map((p, i) =>
      Math.hypot(
        p[0] - (points[i] as readonly [number, number])[0],
        p[1] - (points[i] as readonly [number, number])[1],
      ),
    )
  const total = lengths.reduce((a, b) => a + b, 0)
  let left = Math.min(1, Math.max(0, t)) * total
  for (const [i, len] of lengths.entries()) {
    const a = points[i] as readonly [number, number]
    const b = points[i + 1] as readonly [number, number]
    if (left <= len || i === lengths.length - 1) {
      const f = len === 0 ? 0 : Math.min(1, left / len)
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
    }
    left -= len
  }
  return [first[0], first[1]]
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
 * are typeset as TeX, and a label with `\n` takes several lines.
 *
 * A node's `look` (shape, fill, stroke, text colour, outline width and dash; the shapes include `none`, text alone, and
 * `stack`, boxes piled behind the first) and an edge's `look` override the style's. With `layout: 'manual'` (or when
 * every node has a `position`) nodes go where they are put. An edge with `bidirectional` is drawn once with an arrow
 * head at each end (an edge the other way round between the same nodes is absorbed into it and leaves the layout
 * graph, so ranks are unaffected). An edge label sits at the middle of the edge, or at `labelAt` along it, in a layer
 * above the nodes on a plate, and is moved off the nodes it would cover.
 *
 * @param s - The structure.
 * @param options - Layout, direction, style, overlay (and the frames it comes from), seed and spacing.
 * @returns The scene, with node ids as scene ids (edges are `edge:<index>`, groups `group:<id>`).
 * @index function Draw a Structure as a scene: `diagram(structure, { layout: 'layered' })`.
 * @example
 * ```ts
 * import { structure } from 'omai-engine/scene'
 * import { diagram } from 'omai-render/structures'
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
  const looks = new Map(s.nodes.map((n): [string, NodeLook] => [n.id, { ...style.node(n.role), ...n.look }]))
  const content = new Map(
    s.nodes.map((n) => [n.id, lines(n, overlay?.badges?.[n.id] ?? n.badges, looks.get(n.id) as NodeLook)]),
  )
  // A node's box fits its content in this frame and, given the animation's frames, in every one of them.
  const sizeOf = (n: StructureNode): { width: number; height: number } => {
    const look = looks.get(n.id) as NodeLook
    let box = nodeSize(look, content.get(n.id) as Line[], size)
    for (const f of options.frames ?? []) {
      const other = nodeSize(look, lines(n, f.badges?.[n.id] ?? n.badges, look), size)
      box = { width: Math.max(box.width, other.width), height: Math.max(box.height, other.height) }
    }
    return box
  }
  // Self-loops are drawn here, not by the layout: the layout routes the other edges.
  // A bidirectional edge absorbs the edge the other way round between its ends: one line, arrow heads at both ends.
  const absorbed = new Set<StructureEdge>()
  for (const e of s.edges) {
    if (e.bidirectional !== true || absorbed.has(e)) continue
    for (const t of s.edges) if (t !== e && t.from === e.to && t.to === e.from && !absorbed.has(t)) absorbed.add(t)
  }
  const routed = s.edges.filter((e) => e.from !== e.to && !absorbed.has(e))
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
  const placedByHand = s.nodes.length > 0 && s.nodes.every((n) => n.position !== undefined)
  const which = options.layout ?? (placedByHand ? 'manual' : style.layout)
  const widest = Math.max(10, maxOf(graph.nodes.map((v) => Math.max(v.width, v.height))))
  const layout: GraphLayout =
    which === 'manual'
      ? manualLayout(
          graph,
          Object.fromEntries(s.nodes.flatMap((n) => (n.position === undefined ? [] : [[n.id, n.position]]))),
        )
      : which === 'tree'
        ? tidyTree(graph, {
            direction,
            ...(options.rankSep === undefined ? {} : { levelSep: options.rankSep }),
            ...(options.nodeSep === undefined ? {} : { siblingSep: options.nodeSep }),
          })
        : which === 'circular'
          ? circularLayout(graph, options.nodeSep === undefined ? {} : { gap: options.nodeSep })
          : which === 'force'
            ? forceLayout(graph, {
                seed: options.seed ?? 0,
                ...(options.rankSep === undefined ? {} : { length: options.rankSep }),
              })
            : which === 'bipartite'
              ? bipartiteLayout(
                  graph,
                  s.nodes.filter((n) => n.role !== 'factor').map((n) => n.id),
                  {
                    orientation: direction === 'right' ? 'columns' : 'rows',
                    gap: options.rankSep ?? 110,
                    ...(options.nodeSep === undefined ? {} : { spacing: widest + options.nodeSep }),
                  },
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
    // The box is at least as wide as its label (which starts 8 px in).
    if (Number.isFinite(x0))
      boxes.set(g.id, [
        x0 - 12,
        y0 - 24,
        Math.max(x1 + 12, x0 - 12 + textWidth(g.label, size - 1) + 16),
        y1 + (g.count === undefined ? 12 : 22),
      ])
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

  // An edge with a twin in the other direction bends to its left, so the pair (and their labels) separate: a straight
  // edge into a bow, a routed one (through the ranks it crosses) with its inner points moved aside.
  const pairs = new Set(s.edges.filter((e) => !absorbed.has(e)).map((e) => `${e.from}\u0000${e.to}`))
  const bend = (points: readonly (readonly [number, number])[]): (readonly [number, number])[] => {
    const a = points[0]
    const b = points[points.length - 1]
    if (a === undefined || b === undefined || points.length < 2) return [...points]
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const len = Math.hypot(dx, dy) || 1
    const off = 10
    if (points.length === 2) return [a, [(a[0] + b[0]) / 2 + (dy / len) * off, (a[1] + b[1]) / 2 - (dx / len) * off], b]
    return points.map((p, i): [number, number] =>
      i === 0 || i === points.length - 1 ? [p[0], p[1]] : [p[0] + (dy / len) * off, p[1] - (dx / len) * off],
    )
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
  const edgeLook = (e: StructureEdge): EdgeLook => ({ ...style.edge(e), ...e.look })
  /** An arrow head at the start of an edge: a short reversed arrow lying on its first stretch. */
  const tailArrow = (
    points: readonly (readonly [number, number])[],
    id: string,
    stroke: Paint,
    strokeWidth: number,
  ): SceneNode[] => {
    const a = points[0]
    const b = points[1]
    if (a === undefined || b === undefined) return []
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    if (len < 1e-9) return []
    const reach = Math.min(len, 8 + 2 * strokeWidth)
    const from: [number, number] = [a[0] + ((b[0] - a[0]) / len) * reach, a[1] + ((b[1] - a[1]) / len) * reach]
    return [path([from, [a[0], a[1]]], { id, stroke, strokeWidth, arrow: true })]
  }
  // Labels of straight and routed edges are drawn after the nodes, so they are collected here.
  const pending: { readonly edge: StructureEdge; readonly points: readonly (readonly [number, number])[] }[] = []
  const edgeScene: SceneNode[] = s.edges.flatMap((e, i) => {
    if (absorbed.has(e)) return []
    const look = edgeLook(e)
    const on = activeEdges.has(`${e.from}\u0000${e.to}`)
    const stroke = on ? 'accent' : look.stroke
    const strokeWidth = on ? 2.5 : (look.strokeWidth ?? 1.2)
    const arrow = look.arrow || e.bidirectional === true
    if (e.from === e.to) {
      const b = layout.nodes[e.from]
      if (b === undefined) return []
      const loop = selfLoop(looks.get(e.from) as NodeLook, b, layout, e.from)
      for (const p of loop.points) loopExtent.push([p[0], p[1]])
      const out: SceneNode[] = [
        path(loop.points, {
          id: `edge:${i}`,
          stroke,
          strokeWidth,
          arrow,
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
        stroke,
        strokeWidth,
        arrow,
        smooth: route.points.length > 2,
        ...(look.dash === undefined ? {} : { dash: look.dash }),
      }),
      ...(e.bidirectional === true ? tailArrow(route.points, `edge:${i}:tail`, stroke, strokeWidth) : []),
    ]
    if (e.label !== undefined) pending.push({ edge: e, points: route.points })
    return out
  })

  const nodeScene: SceneNode[] = s.nodes.flatMap((n) => {
    const b = layout.nodes[n.id]
    if (b === undefined) return []
    const look = looks.get(n.id) as NodeLook
    return [
      groupNode([...shapeNodes(look, b, active.has(n.id)), ...labelNodes(look, content.get(n.id) ?? [], b, size)], {
        id: n.id,
        title: [n.label.replaceAll('\n', ' '), n.detail].filter((x) => x !== undefined).join(': '),
      }),
    ]
  })

  // Edge labels, above the nodes and on a plate that hides what is under them; the middle of the edge is the default
  // place, moved aside along the edge or to its other side when that would cover a node or an earlier label.
  const fs = size - 2
  const nodeBoxes = [...Object.values(layout.nodes)].map(
    (b): Rect => [b.x - b.width / 2, b.y - b.height / 2, b.x + b.width / 2, b.y + b.height / 2],
  )
  const labelBoxes: Rect[] = []
  const labelScene: SceneNode[] = pending.flatMap(({ edge, points }) => {
    const label = edge.label as string
    const w = textWidth(label, fs)
    // The text's box for an anchor, as drawn: centred (an explicit `labelAt`) or starting at the anchor.
    const boxAt = (x: number, y: number, centred: boolean): Rect => [
      (centred ? x - w / 2 : x) - 3,
      y - fs * 0.5 - 2,
      (centred ? x + w / 2 : x + w) + 3,
      y + fs * 0.5 + 2,
    ]
    let x: number
    let y: number
    let centred = false
    if (edge.labelAt !== undefined) {
      ;[x, y] = pointAlong(points, edge.labelAt)
      centred = true
    } else {
      const k = Math.floor((points.length - 1) / 2)
      const mid = points[k]
      const nxt = points[k + 1] ?? mid
      if (mid === undefined || nxt === undefined) return []
      ;[x, y] = [(mid[0] + nxt[0]) / 2 + 4, (mid[1] + nxt[1]) / 2 - 4]
      const clear = (r: Rect): boolean => ![...nodeBoxes, ...labelBoxes].some((q) => overlaps(r, q))
      if (!clear(boxAt(x, y, false))) {
        search: for (const t of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) {
          const [px, py] = pointAlong(points, t)
          for (const [dx, dy] of [
            [4, -4],
            [-w - 4, -4],
            [4, fs + 6],
            [-w - 4, fs + 6],
            [-w / 2, -26],
            [-w / 2, 34],
            [-w / 2, -44],
            [-w / 2, 52],
          ] as const) {
            if (clear(boxAt(px + dx, py + dy, false))) {
              ;[x, y] = [px + dx, py + dy]
              break search
            }
          }
        }
      }
    }
    const box = boxAt(x, y, centred)
    labelBoxes.push(box)
    loopExtent.push([box[0], box[1]], [box[2], box[3]])
    return [
      rect(box[0], box[1], box[2] - box[0], box[3] - box[1], { fill: 'bg', stroke: 'none', opacity: 0.9, radius: 3 }),
      text(x, y, label, {
        size: fs,
        fill: 'muted',
        baseline: 'middle',
        ...(centred ? { anchor: 'middle' as const } : {}),
      }),
    ]
  })

  const notes: SceneNode[] = (s.annotations ?? []).flatMap((a) => {
    const target = a.target === undefined ? undefined : (layout.nodes[a.target] ?? undefined)
    const fill = a.kind === 'warning' ? 'tab3' : a.kind === 'highlight' ? 'accent' : 'muted'
    if (target === undefined) {
      loopExtent.push([8 + textWidth(a.text, size - 1), 14 + size])
      return [text(8, 14, a.text, { size: size - 1, fill })]
    }
    const [nx, ny] = [target.x + target.width / 2 + 6, target.y - target.height / 2]
    loopExtent.push([nx + textWidth(a.text, size - 2), ny + 4])
    return [text(nx, ny, a.text, { size: size - 2, fill })]
  })
  // Labels written below a shape (factors and dots) reach past its box.
  for (const n of s.nodes) {
    const b = layout.nodes[n.id]
    const look = looks.get(n.id) as NodeLook
    if (b === undefined || !isOutside(look)) continue
    const rows = content.get(n.id) ?? []
    const ys = lineYs(look, rows.length, b, size)
    rows.forEach((l, i) => {
      const half = textWidth(l.text, l.main ? size : size - 2) / 2
      loopExtent.push([b.x - half, (ys[i] as number) - size / 2], [b.x + half, (ys[i] as number) + size / 2])
    })
  }

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
  const children = [...groupScene, ...edgeScene, ...nodeScene, ...labelScene, ...notes]
  return scene(
    maxX - minX,
    maxY - minY,
    minX === 0 && minY === 0 ? children : [groupNode(children, { translate: [-minX, -minY] })],
    { ...(s.title === undefined ? {} : { title: s.title }) },
  )
}
