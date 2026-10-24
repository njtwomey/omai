/**
 * The common shapes of graph layouts: a graph to lay out (nodes with sizes, edges, optional groups) and the positions a
 * layout gives back (node boxes, edge routes, group boxes). Every layout in this folder takes and returns these, so a
 * diagram can swap layouts freely.
 */

/** A node to place: its id and box size in pixels, and the group it belongs to, if any. */
export interface LayoutNode {
  readonly id: string
  readonly width: number
  readonly height: number
  readonly group?: string
}

/** An edge to route. */
export interface LayoutEdge {
  readonly from: string
  readonly to: string
}

/** A graph to lay out. */
export interface LayoutGraph {
  readonly nodes: readonly LayoutNode[]
  readonly edges: readonly LayoutEdge[]
}

/** A placed box: its centre and size. */
export interface Box {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/** A routed edge: the points of its polyline, from the boundary of `from` to the boundary of `to`. */
export interface EdgeRoute {
  readonly from: string
  readonly to: string
  readonly points: readonly (readonly [number, number])[]
}

/** A layout: node boxes by id, edge routes in the order of the graph's edges, group boxes, and the overall size. */
export interface GraphLayout {
  readonly nodes: Readonly<Record<string, Box>>
  readonly edges: readonly EdgeRoute[]
  readonly groups: Readonly<Record<string, Box>>
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
export function boxBoundary(box: Box, towards: readonly [number, number]): [number, number] {
  const dx = towards[0] - box.x
  const dy = towards[1] - box.y
  if (dx === 0 && dy === 0) return [box.x, box.y]
  const sx = dx === 0 ? Number.POSITIVE_INFINITY : box.width / 2 / Math.abs(dx)
  const sy = dy === 0 ? Number.POSITIVE_INFINITY : box.height / 2 / Math.abs(dy)
  const s = Math.min(sx, sy, 1)
  return [box.x + dx * s, box.y + dy * s]
}

/**
 * Straight edge routes between placed nodes, clipped to the node boxes.
 *
 * @param nodes - The node boxes.
 * @param edges - The edges.
 * @returns One route per edge whose ends are both placed.
 */
export function straightRoutes(nodes: Readonly<Record<string, Box>>, edges: readonly LayoutEdge[]): EdgeRoute[] {
  return edges.flatMap((e) => {
    const a = nodes[e.from]
    const b = nodes[e.to]
    if (a === undefined || b === undefined) return []
    return [{ from: e.from, to: e.to, points: [boxBoundary(a, [b.x, b.y]), boxBoundary(b, [a.x, a.y])] }]
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
  nodes: Readonly<Record<string, Box>>,
  padding = 12,
): Record<string, Box> {
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
  const out: Record<string, Box> = {}
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
  nodes: Readonly<Record<string, Box>>,
  edges: readonly EdgeRoute[],
  groups: Readonly<Record<string, Box>>,
  margin = 16,
): GraphLayout {
  let [x0, y0, x1, y1] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, -Infinity, -Infinity]
  const take = (b: Box): void => {
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
  const move = (b: Box): Box => ({ ...b, x: b.x + dx, y: b.y + dy })
  const map = (r: Readonly<Record<string, Box>>): Record<string, Box> =>
    Object.fromEntries(Object.entries(r).map(([k, b]) => [k, move(b)]))
  return {
    nodes: map(nodes),
    edges: edges.map((e) => ({ ...e, points: e.points.map(([x, y]): [number, number] => [x + dx, y + dy]) })),
    groups: map(groups),
    width: x1 - x0 + 2 * margin,
    height: y1 - y0 + 2 * margin,
  }
}
