/**
 * The common shapes of graph layouts: a graph to lay out (nodes with sizes, edges, optional groups) and the positions a
 * layout gives back (node boxes, edge routes, group boxes). Every layout in this folder takes and returns these, so a
 * diagram can swap layouts freely.
 */

import { OmaiError } from 'omai-engine/base'
import { type OrthogonalOptions, orthogonalRoutes } from './orthogonal.ts'

/** A node to place: its id and box size in pixels, and the group it belongs to, if any. */
export interface LayoutNode {
  readonly id: string
  readonly width: number
  readonly height: number
  readonly group?: string
  /**
   * How far from the middle of a side an orthogonal edge may leave or enter the node, in pixels (default: half the side
   * less 8, so an edge keeps off the corners). Round shapes, whose outline leaves the box away from its side centres,
   * ask for less.
   */
  readonly reach?: number
}

/**
 * An edge to route. `via` lists waypoints the edge passes through, in the layout's own units: the `manual` layout
 * honours them (a straight route is a polyline through them, an orthogonal one runs right-angled legs between them);
 * the layouts that place nodes themselves ignore them.
 */
export interface LayoutEdge {
  readonly from: string
  readonly to: string
  readonly via?: readonly (readonly [number, number])[]
}

/**
 * How an edge's polyline is drawn: `straight` (the default) joins its points as they come, `orthogonal` runs only
 * horizontal and vertical segments with right-angle bends.
 */
export type EdgeRouting = 'straight' | 'orthogonal'

/**
 * A graph to lay out, with the nesting of its groups (a group inside another names it as `parent`).
 *
 * @index structure The graph every layout takes: nodes with sizes, edges and optional groups.
 */
export interface LayoutGraph {
  readonly nodes: readonly LayoutNode[]
  readonly edges: readonly LayoutEdge[]
  readonly groups?: readonly { readonly id: string; readonly parent?: string }[]
}

/** A placed box: its centre and size. */
export interface LayoutBox {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/**
 * A routed edge: the points of its polyline, from the boundary of `from` to the boundary of `to`. A route of an
 * orthogonal layout says so in `route` (every segment is then horizontal or vertical, and the last one enters `to`
 * perpendicular to its side); a straight route leaves the field out.
 */
export interface EdgeRoute {
  readonly from: string
  readonly to: string
  readonly points: readonly (readonly [number, number])[]
  readonly route?: EdgeRouting
}

/**
 * A layout: node boxes by id, edge routes in the order of the graph's edges, group boxes, and the overall size.
 *
 * @index structure What every layout returns: node boxes, edge routes and group boxes.
 */
export interface GraphLayout {
  readonly nodes: Readonly<Record<string, LayoutBox>>
  readonly edges: readonly EdgeRoute[]
  readonly groups: Readonly<Record<string, LayoutBox>>
  readonly width: number
  readonly height: number
}

/**
 * The point where the segment from a box's centre towards `towards` leaves the box.
 *
 * @param box - The box.
 * @param towards - A point outside it.
 * @returns The boundary point.
 */
export function boxBoundary(box: LayoutBox, towards: readonly [number, number]): [number, number] {
  const dx = towards[0] - box.x
  const dy = towards[1] - box.y
  if (dx === 0 && dy === 0) return [box.x, box.y]
  const sx = dx === 0 ? Number.POSITIVE_INFINITY : box.width / 2 / Math.abs(dx)
  const sy = dy === 0 ? Number.POSITIVE_INFINITY : box.height / 2 / Math.abs(dy)
  const s = Math.min(sx, sy, 1)
  return [box.x + dx * s, box.y + dy * s]
}

/**
 * Straight edge routes between placed nodes, clipped to the node boxes. An edge with waypoints (`via`) is the polyline
 * through them, clipped at its two ends.
 *
 * @param nodes - The node boxes.
 * @param edges - The edges.
 * @returns One route per edge whose ends are both placed.
 */
export function straightRoutes(nodes: Readonly<Record<string, LayoutBox>>, edges: readonly LayoutEdge[]): EdgeRoute[] {
  return edges.flatMap((e) => {
    const a = nodes[e.from]
    const b = nodes[e.to]
    if (a === undefined || b === undefined) return []
    const via = e.via ?? []
    const first = via[0] ?? ([b.x, b.y] as const)
    const last = via[via.length - 1] ?? ([a.x, a.y] as const)
    return [{ from: e.from, to: e.to, points: [boxBoundary(a, first), ...via, boxBoundary(b, last)] }]
  })
}

/**
 * The boxes around each group's members, with padding (and room above for a label).
 *
 * @param graph - The graph (its nodes name their groups).
 * @param nodes - The placed node boxes.
 * @param padding - Space around the members (default 12), and twice that above for the label.
 * @returns Group boxes by group id.
 */
export function groupBoxes(
  graph: LayoutGraph,
  nodes: Readonly<Record<string, LayoutBox>>,
  padding = 12,
): Record<string, LayoutBox> {
  const extent = new Map<string, [number, number, number, number]>()
  for (const n of graph.nodes) {
    const b = nodes[n.id]
    if (n.group === undefined || b === undefined) continue
    const e = extent.get(n.group) ?? [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, -Infinity, -Infinity]
    extent.set(n.group, [
      Math.min(e[0], b.x - b.width / 2),
      Math.min(e[1], b.y - b.height / 2),
      Math.max(e[2], b.x + b.width / 2),
      Math.max(e[3], b.y + b.height / 2),
    ])
  }
  const out: Record<string, LayoutBox> = {}
  for (const [g, [x0, y0, x1, y1]] of extent) {
    const top = y0 - padding * 2
    const bottom = y1 + padding
    out[g] = { x: (x0 + x1) / 2, y: (top + bottom) / 2, width: x1 - x0 + 2 * padding, height: bottom - top }
  }
  return out
}

/**
 * Shift a layout so everything (nodes, edges and groups) sits inside `[margin, width - margin]` and the same for y,
 * and report the size.
 *
 * @param nodes - Node boxes.
 * @param edges - Edge routes.
 * @param groups - Group boxes.
 * @param margin - Space around the drawing (default 16).
 * @returns The normalised layout.
 */
export function normaliseLayout(
  nodes: Readonly<Record<string, LayoutBox>>,
  edges: readonly EdgeRoute[],
  groups: Readonly<Record<string, LayoutBox>>,
  margin = 16,
): GraphLayout {
  let [x0, y0, x1, y1] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, -Infinity, -Infinity]
  const take = (b: LayoutBox): void => {
    x0 = Math.min(x0, b.x - b.width / 2)
    y0 = Math.min(y0, b.y - b.height / 2)
    x1 = Math.max(x1, b.x + b.width / 2)
    y1 = Math.max(y1, b.y + b.height / 2)
  }
  for (const b of Object.values(nodes)) take(b)
  for (const b of Object.values(groups)) take(b)
  for (const e of edges) for (const [x, y] of e.points) take({ x, y, width: 0, height: 0 })
  if (!Number.isFinite(x0)) return { nodes: {}, edges: [], groups: {}, width: 2 * margin, height: 2 * margin }
  const dx = margin - x0
  const dy = margin - y0
  const move = (b: LayoutBox): LayoutBox => ({ ...b, x: b.x + dx, y: b.y + dy })
  const map = (r: Readonly<Record<string, LayoutBox>>): Record<string, LayoutBox> =>
    Object.fromEntries(Object.entries(r).map(([k, b]) => [k, move(b)]))
  return {
    nodes: map(nodes),
    edges: edges.map((e) => ({ ...e, points: e.points.map(([x, y]): [number, number] => [x + dx, y + dy]) })),
    groups: map(groups),
    width: x1 - x0 + 2 * margin,
    height: y1 - y0 + 2 * margin,
  }
}

/**
 * The ports of a graph for the orthogonal router: how far off the middle of a side each node lets an edge leave.
 *
 * @param graph - The graph.
 * @returns The `reach` of every node that gives one, by id.
 */
export function portReach(graph: LayoutGraph): Record<string, number> {
  return Object.fromEntries(graph.nodes.flatMap((n) => (n.reach === undefined ? [] : [[n.id, n.reach]])))
}

/**
 * The same layout with right-angled edges: the nodes stay where they are and every edge of the graph is routed again
 * by the orthogonal router (see `orthogonalRoutes`), then the whole is shifted to sit inside the margin. It turns any
 * layout of straight edges (tree, circular, force, grid) into an orthogonal one.
 *
 * @param layout - The layout whose node boxes are kept.
 * @param graph - The graph (its edges are routed; its nodes may carry a `reach`).
 * @param options - The router's options (flow, clearance, spacing).
 * @param margin - Space around the drawing (default 16).
 * @returns The layout with orthogonal edge routes.
 * @example
 * ```ts
 * import { manualLayout, withOrthogonalRoutes } from 'omai-render/layout'
 * const g = { nodes: ['a', 'b'].map((id) => ({ id, width: 20, height: 10 })), edges: [{ from: 'a', to: 'b' }] }
 * const l = withOrthogonalRoutes(manualLayout(g, { a: [0, 0], b: [100, 60] }), g)
 * l.edges[0]?.route // => 'orthogonal'
 * ```
 */
export function withOrthogonalRoutes(
  layout: GraphLayout,
  graph: LayoutGraph,
  options: OrthogonalOptions = {},
  margin = 16,
): GraphLayout {
  const edges = orthogonalRoutes(layout.nodes, graph.edges, { reach: portReach(graph), ...options })
  return normaliseLayout(layout.nodes, edges, layout.groups, margin)
}

/** Options of {@link manualLayout}. */
export interface ManualLayoutOptions {
  /** Space around the drawing (default 16). */
  readonly margin?: number
  /** Edges as straight lines (the default) or with right angles. */
  readonly route?: EdgeRouting
  /** The direction the flow runs, which an orthogonal route prefers to enter along (default `down`). */
  readonly flow?: 'down' | 'right'
}

/**
 * A layout from positions given by hand: each node's box is centred on its position (with the size the graph gives it),
 * edges are straight lines clipped to the boxes (or, with `route: 'orthogonal'`, right-angled routes that clear the
 * other boxes), and the whole is shifted to sit inside the margin. An edge with waypoints (`via`) passes through them.
 *
 * @param graph - The graph (nodes with sizes, and edges).
 * @param positions - The centre of every node by id, in the layout's own units.
 * @param margin - Space around the drawing (default 16), or the options.
 * @returns The layout (group boxes are left to the caller, as for the other layouts' groups of a diagram).
 * @throws OmaiError `value` when a node has no position or a position is not finite.
 * @example
 * ```ts
 * import { manualLayout } from 'omai-render/layout'
 * const g = { nodes: ['a', 'b'].map((id) => ({ id, width: 20, height: 10 })), edges: [{ from: 'a', to: 'b' }] }
 * const l = manualLayout(g, { a: [0, 0], b: [100, 0] })
 * l.nodes['b'].x - l.nodes['a'].x // => 100
 * ```
 */
export function manualLayout(
  graph: LayoutGraph,
  positions: Readonly<Record<string, readonly [number, number]>>,
  margin: number | ManualLayoutOptions = 16,
): GraphLayout {
  const options: ManualLayoutOptions = typeof margin === 'number' ? { margin } : margin
  const placed: Record<string, LayoutBox> = {}
  for (const n of graph.nodes) {
    const at = positions[n.id]
    if (at === undefined) throw new OmaiError('value', `manualLayout: the node '${n.id}' has no position`)
    if (!Number.isFinite(at[0]) || !Number.isFinite(at[1]))
      throw new OmaiError('value', `manualLayout: the position of '${n.id}' is not finite`)
    placed[n.id] = { x: at[0], y: at[1], width: n.width, height: n.height }
  }
  const routes =
    options.route === 'orthogonal'
      ? orthogonalRoutes(placed, graph.edges, {
          reach: portReach(graph),
          ...(options.flow === undefined ? {} : { flow: options.flow }),
        })
      : straightRoutes(placed, graph.edges)
  return normaliseLayout(placed, routes, {}, options.margin ?? 16)
}
