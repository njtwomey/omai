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
  type StructureGroup,
  type StructureNode,
  type StructureNote,
  type StructureNotePosition,
  type StructureOverlay,
  scene,
  tex,
  text,
} from 'omai-engine/scene'
import { circularLayout, forceLayout } from '../layout/force.ts'
import {
  boxBoundary,
  type EdgeRoute,
  type EdgeRouting,
  type GraphLayout,
  type LayoutBox,
  type LayoutGraph,
  manualLayout,
  withOrthogonalRoutes,
} from '../layout/graph.ts'
import { bipartiteLayout } from '../layout/grids.ts'
import { layered } from '../layout/layered.ts'
import { tidyTree } from '../layout/trees.ts'
import { texExtent } from '../maths/tex.ts'
import { type EdgeLook, type GroupLook, type NodeLook, type StructureStyle, styleFor } from './styles.ts'

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
  /**
   * How edges are drawn: `straight` (the default: lines, and polylines through the ranks a layered edge crosses) or
   * `orthogonal`, with horizontal and vertical segments and right-angle bends. An orthogonal edge leaves its source
   * on the side facing the target and enters the target perpendicular to its side, with its arrow head along the last
   * segment: straight when the boxes overlap on an axis, else an L or a Z (bend at the middle of the gap) that clears
   * the other nodes, else a detour; edges that leave or enter at one port merge, and parallel runs get tracks of their
   * own. A layered layout routes through its rank channels (long edges down the column of each rank they skip). Meant
   * for flowcharts and pipelines (layered and manual layouts); straight stays right for trees, networks and fan-in
   * layers of a network. Edge `via` waypoints are honoured under the manual layout, by both routes. An edge's
   * `fromSide` and `toSide` (and `fromAt`, `toAt` along them) fix the sides it leaves and enters by, under every layout
   * but the layered one, whose edges run along the ranks. Edges that leave or enter at one port merge into one line
   * only when they look the same (colour, width, dash, kind); edges that look different get ports side by side.
   * Nodes that are not obstacles (`StructureNode.obstacle`) are crossed rather than avoided.
   */
  readonly route?: EdgeRouting
  /**
   * Whether edges that look the same share a port where they leave or enter a node (default true: they merge into one
   * line). `false` gives every edge a port of its own along the side. Orthogonal routes only.
   */
  readonly merge?: boolean
  /**
   * Orthogonal routes: keep the runs of different edges apart (default true). On, a run never lies along another
   * edge's: edges that leave or enter at one port merge only when they look alike and cross as many ranks (a long edge
   * beside a bundle of short ones gets a port and a track of its own, so the bundle is not mistaken for its origin),
   * and runs of different edges that lie on one line, or end within a track gap of each other, are put on tracks of
   * their own. `false` gives the routes of the earlier router: runs of edges that share a port are merged wherever
   * they overlap (pass `clearance: 6` and `trackGap: 8` too for exactly those routes).
   */
  readonly separate?: boolean
  /**
   * Orthogonal routes: how far an edge keeps from the nodes it passes or goes around, in pixels (default 18): the
   * distance of a detour over a row of nodes (a back edge, an edge that skips nodes) from their sides, and of a track
   * in a rank channel from the nodes beside it. The scene grows to hold what the routes need.
   */
  readonly clearance?: number
  /** Orthogonal routes: the gap between parallel tracks, and between ports side by side on one side (default 10). */
  readonly trackGap?: number
  /** The radius of the rounded corners of orthogonal edges, in pixels (default 0: sharp corners). */
  readonly cornerRadius?: number
  /**
   * The fill of the plate behind an edge label: `page` (the default: the page background, a little see-through, which
   * hides the line under the label), `none` (no plate: the label sits beside its edge, off the nodes and the other
   * edges, and the line is not hidden), or any paint (a card colour, when the diagram is drawn on a card).
   */
  readonly labelPlate?: 'page' | 'none' | Paint
  /**
   * Under the `manual` layout, the point of the node positions' own units that becomes the top left of the drawing
   * (default: the top left of everything drawn). A fixed origin gives the same position the same place in the scene
   * whichever nodes are shown, so the states of an animation line up; the scene then runs from the origin to the far
   * edge of what is drawn. Positions are node centres, so choose an origin left of and above every node's box; anything
   * beyond it still fits (the scene is widened to hold it) but then moves with the content.
   */
  readonly origin?: readonly [x: number, y: number]
}

const CHAR = 0.6
/** How far an orthogonal edge keeps from the nodes it passes, and the gap between its parallel tracks. */
const DEFAULT_CLEARANCE = 18
const DEFAULT_TRACK_GAP = 10

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
  if (n.detail !== undefined && !compact) for (const text of n.detail.split('\n')) out.push({ text, main: false })
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
/**
 * The width of a line of text drawn at `size`: the source length times a glyph width, a little more when bold; a line
 * written `$...$` is measured as typeset (it is drawn two pixels larger, and KaTeX's own layout gives its width).
 */
const textWidth = (s: string, size: number, bold = false): number =>
  isTex(s) ? texExtent(s.slice(1, -1), size + 2).width : s.length * size * CHAR * (bold ? 1.05 : 1)

/** The room the boxes behind the first take in a `stack`: two more, each offset by this much up and to the right. */
const STACK = 4

function nodeSize(look: NodeLook, content: readonly Line[], size: number): { width: number; height: number } {
  const bold = look.fontWeight === 'bold'
  const wide = maxOf(content.map((l) => textWidth(l.text, l.main ? size : size - 2, l.main && bold)))
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
    case 'pill':
      return { width: Math.max(48, wide + 28), height: Math.max(28, tall + 8) }
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
    ...(look.fillOpacity === undefined ? {} : { fillOpacity: look.fillOpacity }),
    ...(look.strokeOpacity === undefined ? {} : { strokeOpacity: look.strokeOpacity }),
    ...(look.opacity === undefined ? {} : { opacity: look.opacity }),
  }
  const x0 = b.x - b.width / 2
  const y0 = b.y - b.height / 2
  switch (look.shape) {
    case 'circle':
    case 'dot': {
      const r = Math.min(b.width, b.height) / 2
      const outer = circle(b.x, b.y, r, style)
      return look.double === true
        ? [
            outer,
            circle(b.x, b.y, r - 4, {
              fill: 'none',
              stroke,
              strokeWidth: 1,
              ...(look.strokeOpacity === undefined ? {} : { strokeOpacity: look.strokeOpacity }),
              ...(look.opacity === undefined ? {} : { opacity: look.opacity }),
            }),
          ]
        : [outer]
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
    case 'pill':
      return [rect(x0, y0, b.width, b.height, { radius: b.height / 2, ...style })]
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

/** Where a block of text lines sits: its anchor, the centre of each line, and the box it covers. */
interface Spot {
  readonly x: number
  readonly ys: readonly number[]
  readonly anchor: 'start' | 'middle' | 'end'
  readonly box: Rect
}

/** The box a block of lines covers: `width` wide from the anchor, a line's height above the first and below the last. */
function spotBox(x: number, ys: readonly number[], width: number, anchor: Spot['anchor'], size: number): Rect {
  const x0 = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x
  return [x0, (ys[0] as number) - size / 2, x0 + width, (ys[ys.length - 1] as number) + size / 2]
}

/**
 * The places a block of text may take around a node, in the order they are tried: below it (centred, then to the right
 * and to the left of the middle, where an edge leaving the middle does not reach), above it the same way, then beside it
 * (right, then left). `below` is the centre of the first line when it sits below the node.
 */
function spotsAround(b: LayoutBox, rows: readonly string[], sizes: readonly number[], below: number): Spot[] {
  const size = sizes[0] as number
  const step = size + 2
  const n = rows.length
  const width = maxOf(rows.map((r, i) => textWidth(r, sizes[i] as number)))
  const sideX = b.width / 2 + 5
  const mk = (x: number, first: number, anchor: Spot['anchor']): Spot => {
    const ys = Array.from({ length: n }, (_, i) => first + i * step)
    return { x, ys, anchor, box: spotBox(x, ys, width, anchor, size) }
  }
  const above = b.y - b.height / 2 - size - (n - 1) * step
  const centred = b.y - ((n - 1) * step) / 2
  return [
    mk(b.x, below, 'middle'),
    mk(b.x + 6, below, 'start'),
    mk(b.x - 6, below, 'end'),
    mk(b.x, above, 'middle'),
    mk(b.x + 6, above, 'start'),
    mk(b.x - 6, above, 'end'),
    mk(b.x + sideX, centred, 'start'),
    mk(b.x - sideX, centred, 'end'),
  ]
}

/**
 * The vertical centre of each row of a block of lines, relative to the middle of the block, and how far the first row
 * reaches above its centre and the last below. Plain rows are `size + 2` apart; a typeset row taller than a plain one
 * (a fraction, a script, a norm) pushes its neighbours away so the lines do not touch.
 */
function rowCentres(
  rows: readonly string[],
  size: number,
): { centres: number[]; tall: number; above: number; below: number } {
  const heightOf = (row: string): number =>
    isTex(row) ? Math.max(size, texExtent(row.slice(1, -1), size + 2).height + 3) : size
  const centres = [0]
  for (let i = 1; i < rows.length; i++)
    centres.push((centres[i - 1] as number) + (heightOf(rows[i - 1] as string) + heightOf(rows[i] as string)) / 2 + 2)
  const span = (centres[centres.length - 1] as number) - (centres[0] as number)
  const above = heightOf(rows[0] as string) / 2
  const below = heightOf(rows[rows.length - 1] as string) / 2
  return { centres: centres.map((c) => c - span / 2), tall: span + above + below, above, below }
}

/** A node's note as it is drawn: where, whether tied to the node and how, and its text style. */
interface NoteSpec {
  readonly text: string
  readonly position: StructureNotePosition | undefined
  readonly tie: false | 'solid' | 'dashed'
  readonly colour: Paint
  readonly size: number | undefined
  readonly bold: boolean
}

function noteSpecOf(note: string | StructureNote): NoteSpec {
  if (typeof note === 'string')
    return { text: note, position: undefined, tie: false, colour: 'muted', size: undefined, bold: false }
  return {
    text: note.text,
    position: note.position,
    tie: note.tie === 'dashed' ? 'dashed' : note.tie === true ? 'solid' : false,
    colour: note.colour ?? 'muted',
    size: note.size,
    bold: note.weight === 'bold',
  }
}

/**
 * Where a note fixed to one side of its node sits: centred above or below it, or beside it, with room for a tie line
 * when it has one (the gap between the outline and the text is half a line, and 12 px with a tie).
 */
function noteSpot(
  b: LayoutBox,
  rows: readonly string[],
  size: number,
  position: StructureNotePosition,
  tied: boolean,
  bold: boolean,
): Spot {
  const gap = tied ? 12 : size / 2
  const width = maxOf(rows.map((r) => textWidth(r, size, bold)))
  const { centres, above, below } = rowCentres(rows, size)
  const [first, last] = [centres[0] as number, centres[centres.length - 1] as number]
  // The rows of a block whose middle is at `y`, and the box they cover.
  const at = (x: number, y: number, anchor: Spot['anchor']): Spot => {
    const ys = centres.map((c) => y + c)
    const x0 = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x
    return { x, ys, anchor, box: [x0, y + first - above, x0 + width, y + last + below] }
  }
  switch (position) {
    case 'above':
      return at(b.x, b.y - b.height / 2 - gap - below - last, 'middle')
    case 'below':
      return at(b.x, b.y + b.height / 2 + gap + above - first, 'middle')
    case 'left':
      return at(b.x - b.width / 2 - gap, b.y, 'end')
    case 'right':
      return at(b.x + b.width / 2 + gap, b.y, 'start')
  }
}

/** The tie line from a node's outline to the box of its note: along the axis the note sits on, with a pixel to spare. */
function tieBetween(b: LayoutBox, box: Rect): [[number, number], [number, number]] {
  const [top, bottom] = [b.y - b.height / 2, b.y + b.height / 2]
  if (box[3] <= top + 0.5)
    return [
      [b.x, top],
      [b.x, box[3] + 1],
    ]
  if (box[1] >= bottom - 0.5)
    return [
      [b.x, bottom],
      [b.x, box[1] - 1],
    ]
  if (box[0] >= b.x + b.width / 2 - 0.5)
    return [
      [b.x + b.width / 2, b.y],
      [box[0] - 1, b.y],
    ]
  return [
    [b.x - b.width / 2, b.y],
    [box[2] + 1, b.y],
  ]
}

function labelNodes(look: NodeLook, content: readonly Line[], b: LayoutBox, size: number, spot?: Spot): SceneNode[] {
  const outside = isOutside(look)
  const ys = spot?.ys ?? lineYs(look, content.length, b, size)
  const x = spot?.x ?? frontOf(look, b).x
  const anchor = spot?.anchor ?? 'middle'
  return content.map((line, i) => {
    const y = ys[i] as number
    const fill = line.main && !outside ? look.text : 'muted'
    const sz = line.main ? size : size - 2
    return isTex(line.text)
      ? tex(x, y, line.text.slice(1, -1), { size: sz + 2, fill, anchor })
      : text(x, y, line.text, {
          size: sz,
          fill,
          anchor,
          baseline: 'middle',
          ...(line.main && look.fontWeight !== undefined ? { weight: look.fontWeight } : {}),
        })
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
    case 'pill': {
      // A stadium: flat top and bottom, a half circle at each end.
      const r = b.height / 2
      const flat = Math.max(0, b.width / 2 - r)
      const ty = uy === 0 ? Number.POSITIVE_INFINITY : r / Math.abs(uy)
      if (Math.abs(ux) * ty <= flat) return [b.x + ux * ty, b.y + uy * ty]
      // The end circle, centred on (±flat, 0): the ray's far intersection with it.
      const cx = Math.sign(ux) * flat
      const dot = ux * cx
      const t = dot + Math.sqrt(dot * dot - (cx * cx - r * r))
      return [b.x + ux * t, b.y + uy * t]
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
type Rect = readonly [x0: number, y0: number, x1: number, y1: number]

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

/** The index of the segment of a polyline a fraction `t` (0 to 1) of the way along it, by length, lies on. */
function segmentAt(points: readonly (readonly [number, number])[], t: number): number {
  const lengths = points
    .slice(1)
    .map((p, i) =>
      Math.hypot(
        p[0] - (points[i] as readonly [number, number])[0],
        p[1] - (points[i] as readonly [number, number])[1],
      ),
    )
  let left = Math.min(1, Math.max(0, t)) * lengths.reduce((a, b) => a + b, 0)
  for (const [i, len] of lengths.entries()) {
    if (left <= len) return i
    left -= len
  }
  return Math.max(0, lengths.length - 1)
}

/** The plate that hides what is under a block of text: a box in the background colour, a little see-through. */
const padded = (r: Rect): Rect => [r[0] - 2, r[1] - 2, r[2] + 2, r[3] + 2]

const plateOf = (box: Rect, fill: Paint = 'bg', opacity = 0.9, id?: string): SceneNode =>
  rect(box[0], box[1], box[2] - box[0], box[3] - box[1], {
    fill,
    stroke: 'none',
    opacity,
    radius: 3,
    ...(id === undefined ? {} : { id }),
  })

/** Whether the segment from `p` to `q` crosses the inside of a box (Liang and Barsky's clipping). */
function segmentHits(p: readonly [number, number], q: readonly [number, number], r: Rect): boolean {
  let [t0, t1] = [0, 1]
  const [dx, dy] = [q[0] - p[0], q[1] - p[1]]
  for (const [d, lo] of [
    [-dx, p[0] - r[0]],
    [dx, r[2] - p[0]],
    [-dy, p[1] - r[1]],
    [dy, r[3] - p[1]],
  ] as const) {
    if (d === 0) {
      if (lo <= 0) return false
    } else {
      const t = lo / d
      if (d < 0) t0 = Math.max(t0, t)
      else t1 = Math.min(t1, t)
      if (t0 >= t1) return false
    }
  }
  return true
}

/** Whether any segment of a polyline crosses the inside of a box. */
const polylineHits = (points: readonly (readonly [number, number])[], r: Rect): boolean =>
  points.slice(1).some((q, i) => segmentHits(points[i] as readonly [number, number], q, r))

/**
 * Where a line along an axis, entering a node at `p` on its box, meets the node's outline: the box's own side for
 * boxes, further in for the round shapes (a circle, an ellipse, a diamond) whose outline leaves the box at the middle
 * of each side only.
 */
function enterOutline(
  look: NodeLook,
  b: LayoutBox,
  p: readonly [number, number],
  inward: readonly [number, number],
): [number, number] {
  const horizontal = Math.abs(inward[0]) > Math.abs(inward[1])
  const [hw, hh] = [b.width / 2, b.height / 2]
  // The half extent of the outline along the line's axis, at the line's offset from the node's centre.
  const offset = horizontal ? p[1] - b.y : p[0] - b.x
  const [along, across] = horizontal ? [hw, hh] : [hh, hw]
  let half = along
  if (look.shape === 'circle' || look.shape === 'dot') {
    const r = Math.min(hw, hh)
    half = Math.abs(offset) < r ? Math.sqrt(r * r - offset * offset) : along
  } else if (look.shape === 'ellipse') {
    half = Math.abs(offset) < across ? along * Math.sqrt(1 - (offset / across) ** 2) : along
  } else if (look.shape === 'diamond') {
    half = Math.abs(offset) < across ? along * (1 - Math.abs(offset) / across) : along
  } else if (look.shape === 'pill') {
    // The ends are half circles of radius `hh`: a line along the axis meets the end circle past the flat middle.
    const flat = Math.max(0, hw - hh)
    if (horizontal) half = Math.abs(offset) < hh ? flat + Math.sqrt(hh * hh - offset * offset) : along
    else
      half =
        Math.abs(offset) <= flat ? hh : Math.abs(offset) < hw ? Math.sqrt(hh * hh - (Math.abs(offset) - flat) ** 2) : hh
  }
  return horizontal ? [b.x - Math.sign(inward[0]) * half, p[1]] : [p[0], b.y - Math.sign(inward[1]) * half]
}

/** An orthogonal route with its two ends brought onto the outlines of the shapes they join. */
function clipToOutlines(
  points: readonly (readonly [number, number])[],
  fromLook: NodeLook,
  from: LayoutBox | undefined,
  toLook: NodeLook,
  to: LayoutBox | undefined,
): (readonly [number, number])[] {
  const out = points.map((p): [number, number] => [p[0], p[1]])
  const [a, b, y, z] = [out[0], out[1], out[out.length - 2], out[out.length - 1]]
  if (
    from !== undefined &&
    a !== undefined &&
    b !== undefined &&
    fromLook.shape !== 'rect' &&
    fromLook.shape !== 'round'
  )
    out[0] = enterOutline(fromLook, from, a, [a[0] - b[0], a[1] - b[1]])
  if (to !== undefined && y !== undefined && z !== undefined && toLook.shape !== 'rect' && toLook.shape !== 'round')
    out[out.length - 1] = enterOutline(toLook, to, z, [z[0] - y[0], z[1] - y[1]])
  return out
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
 *
 * Orthogonal routes keep their distance from the nodes they go around (`clearance`, 18 px by default; the scene grows
 * to hold the detour) and keep unrelated edges apart (`separate`, on by default, with `trackGap` between tracks):
 *
 * ```ts
 * import { structure } from 'omai-engine/scene'
 * import { diagram } from 'omai-render/structures'
 * const row = structure('flow', ['a', 'b', 'c'].map((id, i) => ({ id, label: id, role: 'block', position: [i * 100, 0] })), [
 *   { from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'a', to: 'c' },
 * ])
 * diagram(row, { route: 'orthogonal', clearance: 40 }).height > diagram(row, { route: 'orthogonal' }).height // => true
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
    const nsize = look.fontSize ?? size
    let box = nodeSize(look, content.get(n.id) as Line[], nsize)
    for (const f of options.frames ?? []) {
      const other = nodeSize(look, lines(n, f.badges?.[n.id] ?? n.badges, look), nsize)
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
  const direction = options.direction ?? s.direction ?? 'down'
  const placedByHand = s.nodes.length > 0 && s.nodes.every((n) => n.position !== undefined)
  const which = options.layout ?? (placedByHand ? 'manual' : style.layout)
  const orthogonal = options.route === 'orthogonal'
  const edgeLook = (e: StructureEdge): EdgeLook => ({ ...style.edge(e), ...e.look })
  // Nodes edges may cross: said so, or invisible (no outline and no text), which only holds a place.
  const obstacle = (n: StructureNode): boolean =>
    n.obstacle ??
    !(
      (looks.get(n.id) as NodeLook).shape === 'none' &&
      (content.get(n.id) ?? []).every((l) => l.text.trim() === '') &&
      n.note === undefined
    )
  // Edges merge at a port only when they are drawn the same.
  const mergeKey = (e: StructureEdge): string => {
    const look = edgeLook(e)
    return JSON.stringify([look.stroke, look.strokeWidth ?? 1.2, look.dash ?? null, e.kind ?? null])
  }
  // An orthogonal edge may leave or enter a round shape only near the middle of a side, where its outline is.
  const reachOf = (look: NodeLook, box: { readonly width: number; readonly height: number }): number | undefined => {
    switch (look.shape) {
      case 'circle':
      case 'ellipse':
      case 'diamond':
      case 'pill':
        return 0.35 * (Math.min(box.width, box.height) / 2)
      case 'dot':
      case 'square':
        return 0
      default:
        return undefined
    }
  }
  const graph: LayoutGraph = {
    nodes: s.nodes.map((n) => {
      const box = sizeOf(n)
      const reach = orthogonal ? reachOf(looks.get(n.id) as NodeLook, box) : undefined
      return {
        id: n.id,
        ...box,
        ...(n.group === undefined ? {} : { group: n.group }),
        ...(reach === undefined ? {} : { reach }),
        ...(orthogonal && !obstacle(n) ? { obstacle: false } : {}),
      }
    }),
    edges: routed.map((e) => ({
      from: e.from,
      to: e.to,
      ...(which === 'manual' && e.via !== undefined && e.via.length > 0 ? { via: e.via } : {}),
      ...(orthogonal
        ? {
            mergeKey: mergeKey(e),
            ...(e.fromSide === undefined ? {} : { fromSide: e.fromSide }),
            ...(e.toSide === undefined ? {} : { toSide: e.toSide }),
            ...(e.fromAt === undefined ? {} : { fromAt: e.fromAt }),
            ...(e.toAt === undefined ? {} : { toAt: e.toAt }),
          }
        : {}),
    })),
    groups: s.groups.map((g) => ({ id: g.id, ...(g.parent === undefined ? {} : { parent: g.parent }) })),
  }
  // The router's own knobs, given only for orthogonal routes (straight diagrams are left exactly as they were).
  const router = {
    clearance: options.clearance ?? DEFAULT_CLEARANCE,
    trackGap: options.trackGap ?? DEFAULT_TRACK_GAP,
    separate: options.separate ?? true,
  }
  const routerOptions = orthogonal ? router : {}
  const widest = Math.max(10, maxOf(graph.nodes.map((v) => Math.max(v.width, v.height))))
  const routing: EdgeRouting = orthogonal ? 'orthogonal' : 'straight'
  const placed: GraphLayout =
    which === 'manual'
      ? manualLayout(
          graph,
          Object.fromEntries(s.nodes.flatMap((n) => (n.position === undefined ? [] : [[n.id, n.position]]))),
          {
            route: routing,
            flow: direction,
            ...(options.origin === undefined ? {} : { origin: options.origin }),
            ...(options.merge === undefined ? {} : { merge: options.merge }),
            ...routerOptions,
          },
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
                  route: routing,
                  ...(options.merge === undefined ? {} : { merge: options.merge }),
                  ...routerOptions,
                  ...(options.rankSep === undefined ? {} : { rankSep: options.rankSep }),
                  ...(options.nodeSep === undefined ? {} : { nodeSep: options.nodeSep }),
                })
  // The layouts that draw straight lines take right angles from the router, which keeps their node boxes.
  const layout: GraphLayout =
    orthogonal && which !== 'manual' && which !== 'layered'
      ? withOrthogonalRoutes(placed, graph, {
          flow: direction,
          ...(options.merge === undefined ? {} : { merge: options.merge }),
          clearance: router.clearance,
          spacing: router.trackGap,
          separate: router.separate,
        })
      : placed

  // Groups: deepest first, each around its members and its child groups, with room for its label. The label is one
  // under the type size, and the room for it 24 px; a `labelSize` takes the room with it (its size plus 13).
  const groupLookOf = (g: StructureGroup): GroupLook => ({ ...style.group(g), ...g.look })
  const labelSizeOf = (look: GroupLook): number => look.labelSize ?? size - 1
  const labelRoom = (look: GroupLook): number => (look.labelSize === undefined ? 24 : look.labelSize + 13)
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
      // A note beside a member is inside the group's frame: under it unless it says another side.
      const noted = s.nodes.find((m) => m.id === id)?.note
      if (noted !== undefined) {
        const spec = noteSpecOf(noted)
        const rows = spec.text.split('\n')
        const ns = spec.size ?? size - 2
        if (spec.position === undefined) {
          const half = maxOf(rows.map((r) => textWidth(r, ns, spec.bold))) / 2
          x0 = Math.min(x0, b.x - half)
          x1 = Math.max(x1, b.x + half)
          y1 = Math.max(y1, b.y + b.height / 2 + 4 + rows.length * (ns + 2) + (spec.tie === false ? 0 : 8))
        } else {
          const box = noteSpot(b, rows, ns, spec.position, spec.tie !== false, spec.bold).box
          x0 = Math.min(x0, box[0])
          y0 = Math.min(y0, box[1])
          x1 = Math.max(x1, box[2])
          y1 = Math.max(y1, box[3])
        }
      }
    }
    for (const child of s.groups.filter((c) => c.parent === g.id)) {
      const c = boxes.get(child.id)
      if (c === undefined) continue
      x0 = Math.min(x0, c[0])
      y0 = Math.min(y0, c[1])
      x1 = Math.max(x1, c[2])
      y1 = Math.max(y1, c[3])
    }
    // The box is at least as wide as its label (which starts 8 px in), and has room for it at the top or the bottom.
    if (Number.isFinite(x0)) {
      const look = groupLookOf(g)
      const room = labelRoom(look)
      const below = look.labelPosition === 'bottom-left'
      boxes.set(g.id, [
        x0 - 12,
        y0 - (below ? 12 : room),
        Math.max(x1 + 12, x0 - 12 + textWidth(g.label, labelSizeOf(look), look.labelWeight === 'bold') + 16),
        y1 + (below ? room : g.count === undefined ? 12 : 22),
      ])
    }
  }
  const groupScene: SceneNode[] = [...s.groups]
    .sort((a, b) => depthOf(s, a.id) - depthOf(s, b.id))
    .flatMap((g) => {
      const b = boxes.get(g.id)
      if (b === undefined) return []
      const look = groupLookOf(g)
      const [x0, y0, x1, y1] = b
      const lsz = labelSizeOf(look)
      const position = look.labelPosition ?? 'top-left'
      const labelAt: [number, number] =
        position === 'bottom-left'
          ? [x0 + 8, y1 - 7]
          : [position === 'top-centre' ? (x0 + x1) / 2 : x0 + 8, y0 + labelRoom(look) - 10]
      const out: SceneNode[] = [
        rect(x0, y0, x1 - x0, y1 - y0, {
          id: `group:${g.id}`,
          fill: look.fill,
          stroke: look.stroke,
          radius: look.radius,
          ...(look.strokeWidth === undefined ? {} : { strokeWidth: look.strokeWidth }),
          ...(look.dash === undefined ? {} : { dash: look.dash }),
          ...(look.fillOpacity === undefined ? {} : { fillOpacity: look.fillOpacity }),
          ...(look.strokeOpacity === undefined ? {} : { strokeOpacity: look.strokeOpacity }),
          ...(look.opacity === undefined ? {} : { opacity: look.opacity }),
        }),
        text(labelAt[0], labelAt[1], g.label, {
          size: lsz,
          fill: look.text ?? 'muted',
          ...(position === 'top-centre' ? { anchor: 'middle' as const } : {}),
          ...(look.labelWeight === undefined ? {} : { weight: look.labelWeight }),
        }),
      ]
      if (g.count !== undefined)
        out.push(
          text(x1 - 8, y1 - 7, g.kind === 'repeat' ? `× ${g.count}` : String(g.count), {
            size: lsz,
            fill: look.text ?? 'fg',
            anchor: 'end',
          }),
        )
      return out
    })

  // An edge with a twin in the other direction bends to its left, so the pair (and their labels) separate: a straight
  // edge into a bow, a routed one (through the ranks it crosses) with its inner points moved aside, an orthogonal one
  // kept apart by the router.
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
  /** An arrow head at the start of an edge: a short reversed arrow lying on its first stretch. */
  const tailArrow = (
    points: readonly (readonly [number, number])[],
    id: string,
    stroke: Paint,
    strokeWidth: number,
    opacity?: number,
  ): SceneNode[] => {
    const a = points[0]
    const b = points[1]
    if (a === undefined || b === undefined) return []
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    if (len < 1e-9) return []
    const reach = Math.min(len, 8 + 2 * strokeWidth)
    const from: [number, number] = [a[0] + ((b[0] - a[0]) / len) * reach, a[1] + ((b[1] - a[1]) / len) * reach]
    return [
      path([from, [a[0], a[1]]], {
        id,
        stroke,
        strokeWidth,
        arrow: true,
        ...(opacity === undefined ? {} : { opacity }),
      }),
    ]
  }
  // What is drawn of each edge, for the labels and notes that must keep off the lines.
  const drawn: { readonly edge: StructureEdge; readonly points: readonly (readonly [number, number])[] }[] = []
  // Labels of straight and routed edges are drawn after the nodes, so they are collected here.
  const pending: {
    readonly edge: StructureEdge
    readonly index: number
    readonly points: readonly (readonly [number, number])[]
    readonly orthogonal: boolean
  }[] = []
  // Edge labels, notes and the details of a node are two points under the type size.
  const fs = size - 2
  const labelSize = fs
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
      drawn.push({ edge: e, points: loop.points })
      const out: SceneNode[] = [
        path(loop.points, {
          id: `edge:${i}`,
          stroke,
          strokeWidth,
          arrow,
          smooth: true,
          ...(look.dash === undefined ? {} : { dash: look.dash }),
          ...(look.opacity === undefined ? {} : { opacity: look.opacity }),
        }),
      ]
      if (e.label !== undefined) {
        const [lx, ly] = loop.label
        const rows = e.label.split('\n')
        const fs = look.labelSize ?? labelSize
        const step = fs + 2
        const w = maxOf(rows.map((r) => textWidth(r, fs, look.labelWeight === 'bold')))
        const half = ((rows.length - 1) * step) / 2
        const anchor = Math.abs(loop.direction[0]) < 0.4 ? 'middle' : loop.direction[0] > 0 ? 'start' : 'end'
        loopExtent.push([lx - w, ly - size - half], [lx + w, ly + size + half])
        rows.forEach((row, k) => {
          const y = ly - half + k * step
          out.push(
            isTex(row)
              ? tex(lx, y, row.slice(1, -1), { size: fs + 2, fill: look.labelColour ?? 'muted', anchor })
              : text(lx, y, row, {
                  size: fs,
                  fill: look.labelColour ?? 'muted',
                  anchor,
                  baseline: 'middle',
                  ...(look.labelWeight === undefined ? {} : { weight: look.labelWeight }),
                }),
          )
        })
      }
      return out
    }
    const original = routeOf.get(e)
    if (original === undefined) return []
    const orthogonalRoute = original.route === 'orthogonal'
    const twin = pairs.has(`${e.to}\u0000${e.from}`)
    let points: readonly (readonly [number, number])[] =
      twin && !orthogonalRoute ? bend(original.points) : original.points
    if (orthogonalRoute) {
      points = clipToOutlines(
        points,
        looks.get(e.from) as NodeLook,
        layout.nodes[e.from],
        looks.get(e.to) as NodeLook,
        layout.nodes[e.to],
      )
      for (const p of points) loopExtent.push([p[0], p[1]])
    }
    // Waypoints make corners, not curves.
    const curved = !orthogonalRoute && !(which === 'manual' && (e.via?.length ?? 0) > 0)
    drawn.push({ edge: e, points })
    const out: SceneNode[] = [
      path(points, {
        id: `edge:${i}`,
        stroke,
        strokeWidth,
        arrow,
        smooth: curved && points.length > 2,
        ...(orthogonalRoute && (options.cornerRadius ?? 0) > 0 ? { cornerRadius: options.cornerRadius as number } : {}),
        ...(look.dash === undefined ? {} : { dash: look.dash }),
        ...(look.opacity === undefined ? {} : { opacity: look.opacity }),
      }),
      ...(e.bidirectional === true ? tailArrow(points, `edge:${i}:tail`, stroke, strokeWidth, look.opacity) : []),
    ]
    if (e.label !== undefined) pending.push({ edge: e, index: i, points, orthogonal: orthogonalRoute })
    return out
  })

  // Text beside nodes (the labels of factors and dots, and notes) takes the first place that no edge, node or other
  // text covers: under the node by default, then above, then to its right and its left.
  const solid = new Set(s.nodes.filter(obstacle).map((n) => n.id))
  const nodeBoxes = Object.entries(layout.nodes)
    .filter(([id]) => solid.has(id))
    .map(([, b]): Rect => [b.x - b.width / 2, b.y - b.height / 2, b.x + b.width / 2, b.y + b.height / 2])
  const beside: Rect[] = []
  const covered = (r: Rect): boolean =>
    [...nodeBoxes, ...beside].some((q) => overlaps(r, q)) || drawn.some((d) => polylineHits(d.points, r))
  const nodeScene: SceneNode[] = s.nodes.flatMap((n) => {
    const b = layout.nodes[n.id]
    if (b === undefined) return []
    const look = looks.get(n.id) as NodeLook
    const nsize = look.fontSize ?? size
    const rows = content.get(n.id) ?? []
    let spot: Spot | undefined
    const plates: SceneNode[] = []
    // Where the text beside the node ends, for its note to keep clear of.
    let reach: Rect = [b.x - b.width / 2, b.y - b.height / 2, b.x + b.width / 2, b.y + b.height / 2]
    const extend = (r: Rect): void => {
      reach = [Math.min(reach[0], r[0]), Math.min(reach[1], r[1]), Math.max(reach[2], r[2]), Math.max(reach[3], r[3])]
    }
    if (isOutside(look) && rows.length > 0) {
      const sizes = rows.map((l) => (l.main ? nsize : nsize - 2))
      const spots = spotsAround(
        b,
        rows.map((l) => l.text),
        sizes,
        b.y + b.height / 2 + nsize,
      )
      const free = spots.find((sp) => !covered(sp.box))
      spot = free ?? (spots[0] as Spot)
      if (free === undefined) plates.push(plateOf(padded(spot.box)))
      beside.push(spot.box)
      extend(spot.box)
      // Labels written beside a shape reach past its box.
      const at = spot
      rows.forEach((l, i) => {
        const width = textWidth(l.text, sizes[i] as number)
        const x0 = at.anchor === 'middle' ? at.x - width / 2 : at.anchor === 'end' ? at.x - width : at.x
        const y = at.ys[i] as number
        loopExtent.push([x0, y - nsize / 2], [x0 + width, y + nsize / 2])
      })
    }
    const noteNodes: SceneNode[] = []
    if (n.note !== undefined) {
      const spec = noteSpecOf(n.note)
      const noteRows = spec.text.split('\n')
      const ns = spec.size ?? fs
      // A tie needs room between the node and its note.
      const extra = spec.tie === false ? 0 : 8
      const home: LayoutBox = {
        x: b.x,
        y: (reach[1] + reach[3]) / 2,
        width: 2 * Math.max(b.x - reach[0], reach[2] - b.x) + 2 * extra,
        height: reach[3] - reach[1] + 2 * extra,
      }
      let at: Spot
      if (spec.position !== undefined) {
        // A side the note asked for: there, and on a plate only when something else is already there.
        at = noteSpot(home, noteRows, ns, spec.position, spec.tie !== false, spec.bold)
        if (covered(at.box)) plates.push(plateOf(padded(at.box)))
      } else {
        const spots = spotsAround(
          home,
          noteRows,
          noteRows.map(() => ns),
          home.y + home.height / 2 + ns,
        )
        const free = spots.find((sp) => !covered(sp.box))
        at = free ?? (spots[0] as Spot)
        if (free === undefined) plates.push(plateOf(padded(at.box)))
      }
      beside.push(at.box)
      nodeBoxes.push(at.box)
      loopExtent.push([at.box[0], at.box[1]], [at.box[2], at.box[3]])
      if (spec.tie !== false) {
        const [from, to] = tieBetween(b, at.box)
        noteNodes.push(
          path([from, to], {
            stroke: spec.colour,
            strokeWidth: 1,
            ...(spec.tie === 'dashed' ? { dash: [3, 3] } : {}),
          }),
        )
      }
      noteRows.forEach((row, i) => {
        const y = at.ys[i] as number
        noteNodes.push(
          isTex(row)
            ? tex(at.x, y, row.slice(1, -1), { size: ns + 2, fill: spec.colour, anchor: at.anchor })
            : text(at.x, y, row, {
                size: ns,
                fill: spec.colour,
                anchor: at.anchor,
                baseline: 'middle',
                ...(spec.bold ? { weight: 'bold' as const } : {}),
              }),
        )
      })
    }
    return [
      groupNode(
        [...shapeNodes(look, b, active.has(n.id)), ...plates, ...labelNodes(look, rows, b, nsize, spot), ...noteNodes],
        {
          id: n.id,
          title: [n.label.replaceAll('\n', ' '), n.detail?.replaceAll('\n', ' ')]
            .filter((x) => x !== undefined)
            .join(': '),
        },
      ),
    ]
  })

  // Edge labels, above the nodes and on a plate that hides what is under them; the middle of the edge is the default
  // place (the middle of the longest segment, for an orthogonal edge), moved aside along the edge or to its other side
  // when that would cover a node or an earlier label. A label may have several lines, each plain or TeX.
  const labelBoxes: Rect[] = []
  const plate = options.labelPlate ?? 'page'
  // Without a plate the line under a label would show through it: the label then keeps off its own edge too.
  const plated = plate !== 'none'
  const labelScene: SceneNode[] = pending.flatMap(({ edge, index, points, orthogonal: right }) => {
    const look = edgeLook(edge)
    const fs = look.labelSize ?? labelSize
    const rows = (edge.label as string).split('\n')
    const w = maxOf(rows.map((r) => textWidth(r, fs, look.labelWeight === 'bold')))
    const { centres, tall } = rowCentres(rows, fs)
    // The text's box for an anchor, as drawn: centred (an explicit `labelAt`) or starting at the anchor.
    const boxAt = (x: number, y: number, centred: boolean): Rect => [
      (centred ? x - w / 2 : x) - 3,
      y - tall / 2 - 2,
      (centred ? x + w / 2 : x + w) + 3,
      y + tall / 2 + 2,
    ]
    let x: number
    let y: number
    let centred = false
    if (edge.labelAt !== undefined) {
      ;[x, y] = pointAlong(points, edge.labelAt)
      centred = true
      if (!plated) {
        // Beside the line at that point, not on it: above or below a flat stretch, to the side of an upright one.
        const k = Math.max(0, Math.min(points.length - 2, segmentAt(points, edge.labelAt)))
        const [p, q] = [points[k] as readonly [number, number], points[k + 1] as readonly [number, number]]
        const flat = Math.abs(p[1] - q[1]) < Math.abs(p[0] - q[0])
        const near: [number, number][] = flat
          ? [
              [x, y - (tall / 2 + 7)],
              [x, y + (tall / 2 + 7)],
            ]
          : [
              [x + w / 2 + 7, y],
              [x - w / 2 - 7, y],
            ]
        const free = near.find(
          ([cx, cy]) => ![...nodeBoxes, ...labelBoxes].some((o) => overlaps(boxAt(cx, cy, true), o)),
        )
        ;[x, y] = free ?? (near[0] as [number, number])
      }
    } else if (right) {
      // On the middle of a long segment, on its plate; beside the segment when that is covered; never on a node.
      const segs = points.slice(1).map((q, k) => ({
        p: points[k] as readonly [number, number],
        q,
        k,
        len: Math.hypot(
          q[0] - (points[k] as readonly [number, number])[0],
          q[1] - (points[k] as readonly [number, number])[1],
        ),
      }))
      const mid = (segs.length - 1) / 2
      segs.sort((a, b) => b.len - a.len || Math.abs(a.k - mid) - Math.abs(b.k - mid))
      const on: { x: number; y: number; centred: boolean }[] = []
      const off: { x: number; y: number; centred: boolean }[] = []
      for (const sg of segs) {
        const horizontal = Math.abs(sg.p[1] - sg.q[1]) < 1e-6
        for (const t of [0.5, 0.3, 0.7]) {
          const [cx, cy] = [sg.p[0] + (sg.q[0] - sg.p[0]) * t, sg.p[1] + (sg.q[1] - sg.p[1]) * t]
          if (horizontal ? sg.len >= w + 12 : sg.len >= tall + 10) on.push({ x: cx, y: cy, centred: true })
          if (horizontal)
            off.push({ x: cx, y: cy - (tall / 2 + 7), centred: true }, { x: cx, y: cy + (tall / 2 + 7), centred: true })
          else off.push({ x: cx + 7, y: cy, centred: false }, { x: cx - 7 - w, y: cy, centred: false })
        }
      }
      const all = [...on, ...off]
      const others = drawn.filter((d) => d.edge !== edge)
      const own = drawn.filter((d) => d.edge === edge)
      const free = (r: Rect, lines: boolean): boolean =>
        ![...nodeBoxes, ...labelBoxes].some((q) => overlaps(r, q)) &&
        (!lines || !others.some((d) => polylineHits(d.points, r))) &&
        (plated || !own.some((d) => polylineHits(d.points, r)))
      const pick =
        all.find((c) => free(boxAt(c.x, c.y, c.centred), true)) ??
        all.find((c) => free(boxAt(c.x, c.y, c.centred), false))
      const first = segs[0]
      if (pick === undefined && first === undefined) return []
      const fallback =
        first === undefined
          ? undefined
          : { x: (first.p[0] + first.q[0]) / 2, y: (first.p[1] + first.q[1]) / 2, centred: true }
      const at = (pick ?? fallback) as { x: number; y: number; centred: boolean }
      ;[x, y, centred] = [at.x, at.y, at.centred]
    } else {
      const k = Math.floor((points.length - 1) / 2)
      const mid = points[k]
      const nxt = points[k + 1] ?? mid
      if (mid === undefined || nxt === undefined) return []
      ;[x, y] = [(mid[0] + nxt[0]) / 2 + 4, (mid[1] + nxt[1]) / 2 - 4]
      const own = drawn.filter((d) => d.edge === edge)
      const clear = (r: Rect): boolean =>
        ![...nodeBoxes, ...labelBoxes].some((q) => overlaps(r, q)) &&
        (plated || !own.some((d) => polylineHits(d.points, r)))
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
    // The label is one group, `edge:<i>:label`, with its plate `edge:<i>:plate` (if any) under the text lines.
    return [
      groupNode(
        [
          ...(plated
            ? [
                plate === 'page'
                  ? plateOf(box, 'bg', 0.9, `edge:${index}:plate`)
                  : plateOf(box, plate, 1, `edge:${index}:plate`),
              ]
            : []),
          ...rows.map((row, i) =>
            isTex(row)
              ? tex(x, y + (centres[i] as number), row.slice(1, -1), {
                  size: fs + 2,
                  fill: look.labelColour ?? 'muted',
                  anchor: centred ? 'middle' : 'start',
                })
              : text(x, y + (centres[i] as number), row, {
                  size: fs,
                  fill: look.labelColour ?? 'muted',
                  baseline: 'middle',
                  ...(centred ? { anchor: 'middle' as const } : {}),
                  ...(look.labelWeight === undefined ? {} : { weight: look.labelWeight }),
                }),
          ),
        ],
        { id: `edge:${index}:label` },
      ),
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

  // Groups and notes can reach past the layout's own box: grow the scene to hold them.
  let [minX, minY, maxX, maxY] = [0, 0, layout.width, layout.height]
  for (const [x0, y0, x1, y1] of boxes.values()) {
    minX = Math.min(minX, x0 - 8)
    minY = Math.min(minY, y0 - 8)
    maxX = Math.max(maxX, x1 + 8)
    maxY = Math.max(maxY, y1 + 8)
  }
  // A fixed origin lets nodes lie left of or above it: the scene is widened to hold them.
  if (options.origin !== undefined)
    for (const b of Object.values(layout.nodes)) {
      minX = Math.min(minX, b.x - b.width / 2 - 16)
      minY = Math.min(minY, b.y - b.height / 2 - 16)
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
