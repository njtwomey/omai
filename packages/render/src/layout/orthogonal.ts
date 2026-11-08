/**
 * Orthogonal edge routing: right-angled connectors between placed node boxes, with no rendering and no layout
 * dependency. A connector leaves its source on the side facing the target, runs along horizontal and vertical segments
 * and enters the target perpendicular to its side. `orthogonalRoutes` routes edges between boxes (straight when the
 * boxes overlap on an axis, else an L or a Z with its bend at the middle of the gap, else a detour around the boxes in
 * the way), `separateRoutes` spreads parallel segments of different edges onto their own tracks while edges of one kind
 * (the same `mergeKey`) that leave or enter at the same port stay merged; edges of different kinds get ports side by
 * side, and edges that leave where another enters run side by side, as twin edges do. Edges may name their sides and
 * places along them, and boxes may be passable. The layered layout builds on the same pieces to route edges through its
 * rank channels.
 */

import type { EdgeRoute, LayoutBox, LayoutEdge } from './graph.ts'

/** A side of a box. */
export type PortSide = 'top' | 'right' | 'bottom' | 'left'

/**
 * An edge to route orthogonally: a {@link LayoutEdge}, whose `fromSide`, `toSide`, `fromAt`, `toAt` and `mergeKey` the
 * router honours (the sides it must leave and enter by, where along them, and which edges may share a port).
 */
export type OrthogonalEdge = LayoutEdge

/** Options of {@link orthogonalRoutes} and {@link separateRoutes}. */
export interface OrthogonalOptions {
  /**
   * The direction the flow runs. When an L-shaped route could leave by either of two sides, it enters the target along
   * the flow (from above for `down`, from the left for `right`); likewise for Z-shaped routes (default `down`).
   */
  readonly flow?: 'down' | 'right'
  /** How far a segment keeps from a box that is not an end of its edge, in pixels (default 6). */
  readonly clearance?: number
  /** The gap between parallel segments of different edges, in pixels (default 8). */
  readonly spacing?: number
  /** How far from the middle of a side an edge may leave or enter each node, by id (default half the side less 8). */
  readonly reach?: Readonly<Record<string, number>>
  /**
   * The ids of boxes that are not obstacles: routes may pass through them (an invisible placeholder, a caption), and
   * they stay the ends of their own edges. Default none.
   */
  readonly passable?: readonly string[]
  /**
   * Whether edges with the same `mergeKey` (by default all of them) share one port where they leave or enter a node
   * (default true). With `false` every edge gets a port of its own wherever several meet at a side.
   */
  readonly merge?: boolean
}

type Pt = [x: number, y: number]

/** A box (or a waypoint, with no size) as the router sees it: its extent and how far from centre its ports may lie. */
interface Item {
  readonly x: number
  readonly y: number
  readonly hw: number
  readonly hh: number
  readonly rx: number
  readonly ry: number
}

interface Candidate {
  readonly points: Pt[]
  readonly exit: PortSide
  readonly entry: PortSide
}

const EPS = 1e-6
/** The shortest stretch a bend may leave at either end of a segment. */
const MIN_STUB = 3
/** The cost of a bend in a detour, in pixels of length. */
const BEND = 20
/** The cost of a detour leaving or entering by a side that does not face the other end. */
const WRONG_SIDE = 24
/** How close to a corner an edge may leave a side. */
const CORNER = 8
/** The extra cost of a detour that leaves or enters by a side another kind of edge uses. */
const OCCUPIED = 40

/**
 * How far from the middle of a side an edge may leave or enter when the node states no reach: all of the side but 8
 * pixels at each corner (a third of the half side, on a small box).
 *
 * @param half - Half the length of the side, in pixels.
 * @returns The usable half-length.
 * @example
 * ```ts
 * import { sideReach } from 'omai-render/layout'
 * sideReach(30) // => 22
 * sideReach(10) // => 7
 * ```
 */
export function sideReach(half: number): number {
  return Math.max(0, half - Math.min(CORNER, half * 0.3))
}

const DIRS: Readonly<Record<PortSide, readonly [number, number]>> = {
  top: [0, -1],
  right: [1, 0],
  bottom: [0, 1],
  left: [-1, 0],
}
const SIDES: readonly PortSide[] = ['top', 'right', 'bottom', 'left']

function itemOf(box: LayoutBox, reach: number | undefined): Item {
  const hw = box.width / 2
  const hh = box.height / 2
  return {
    x: box.x,
    y: box.y,
    hw,
    hh,
    rx: reach === undefined ? sideReach(hw) : Math.min(reach, hw),
    ry: reach === undefined ? sideReach(hh) : Math.min(reach, hh),
  }
}

const pointItem = (p: readonly [number, number]): Item => ({ x: p[0], y: p[1], hw: 0, hh: 0, rx: 0, ry: 0 })

/** Whether an axis-aligned segment passes through the inside of a box grown by `c`. */
function hits(p: readonly [number, number], q: readonly [number, number], o: Item, c: number): boolean {
  const x0 = o.x - o.hw - c + EPS
  const x1 = o.x + o.hw + c - EPS
  const y0 = o.y - o.hh - c + EPS
  const y1 = o.y + o.hh + c - EPS
  if (x0 >= x1 || y0 >= y1) return false
  const [lx, hx] = [Math.min(p[0], q[0]), Math.max(p[0], q[0])]
  const [ly, hy] = [Math.min(p[1], q[1]), Math.max(p[1], q[1])]
  if (Math.abs(p[0] - q[0]) < EPS) return p[0] > x0 && p[0] < x1 && hy > y0 && ly < y1
  if (Math.abs(p[1] - q[1]) < EPS) return p[1] > y0 && p[1] < y1 && hx > x0 && lx < x1
  return hx > x0 && lx < x1 && hy > y0 && ly < y1
}

function clear(points: readonly Pt[], obstacles: readonly Item[], c: number): boolean {
  for (let i = 0; i + 1 < points.length; i++)
    for (const o of obstacles) if (hits(points[i] as Pt, points[i + 1] as Pt, o, c)) return false
  return true
}

/** The unit direction of the last segment of a route. */
function lastDirection(points: readonly Pt[]): [number, number] | undefined {
  for (let i = points.length - 1; i > 0; i--) {
    const [a, b] = [points[i - 1] as Pt, points[i] as Pt]
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    if (len > EPS) return [(b[0] - a[0]) / len, (b[1] - a[1]) / len]
  }
  return undefined
}

/**
 * Drop repeated points and points in the middle of a straight run (including the turn-backs of a route that doubles
 * on itself).
 */
function tidy(points: readonly (readonly [number, number])[]): Pt[] {
  const out: Pt[] = []
  for (const p of points) {
    const last = out[out.length - 1]
    if (last === undefined || Math.abs(last[0] - p[0]) > EPS || Math.abs(last[1] - p[1]) > EPS) out.push([p[0], p[1]])
  }
  let changed = true
  while (changed && out.length > 2) {
    changed = false
    for (let i = 1; i + 1 < out.length; i++) {
      const [a, b, c] = [out[i - 1] as Pt, out[i] as Pt, out[i + 1] as Pt]
      const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
      if (Math.abs(cross) < EPS) {
        out.splice(i, 1)
        changed = true
        break
      }
    }
  }
  return out
}

/**
 * The line a straight edge takes through the overlap `[lo, hi]` when one end is where several edges meet: through that
 * end's middle (the port the others share), if it lies inside the overlap.
 */
function held(a: number, b: number, lo: number, hi: number, prefer: 'a' | 'b' | undefined): number[] {
  const at = prefer === 'a' ? a : prefer === 'b' ? b : undefined
  return at !== undefined && at >= lo - EPS && at <= hi + EPS ? [at] : []
}

/** The candidate routes between two items, in the order they are preferred (fewest bends first). */
function candidates(
  a: Item,
  b: Item,
  flow: 'down' | 'right',
  incoming: readonly [number, number] | undefined,
  prefer: 'a' | 'b' | undefined,
): Candidate[] {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const sx = dx < 0 ? -1 : 1
  const sy = dy < 0 ? -1 : 1
  const exitV = a.y + sy * a.hh
  const entryV = b.y - sy * b.hh
  const exitH = a.x + sx * a.hw
  const entryH = b.x - sx * b.hw
  const yGap = Math.abs(dy) > a.hh + b.hh + EPS
  const xGap = Math.abs(dx) > a.hw + b.hw + EPS
  const [exitSideV, entrySideV]: [PortSide, PortSide] = sy > 0 ? ['bottom', 'top'] : ['top', 'bottom']
  const [exitSideH, entrySideH]: [PortSide, PortSide] = sx > 0 ? ['right', 'left'] : ['left', 'right']
  const out: Candidate[] = []
  const lerp = (u: number, v: number, t: number): number => u + (v - u) * t

  // Straight: the boxes overlap on an axis (as far as the stretches ports may use), so one segment joins them.
  const [xlo, xhi] = [Math.max(a.x - a.rx, b.x - b.rx), Math.min(a.x + a.rx, b.x + b.rx)]
  if (xlo <= xhi + EPS && yGap) {
    const mid = (xlo + xhi) / 2
    for (const x of [...held(a.x, b.x, xlo, xhi, prefer), mid, (xlo + mid) / 2, (mid + xhi) / 2, xlo, xhi])
      out.push({
        points: [
          [x, exitV],
          [x, entryV],
        ],
        exit: exitSideV,
        entry: entrySideV,
      })
  }
  const [ylo, yhi] = [Math.max(a.y - a.ry, b.y - b.ry), Math.min(a.y + a.ry, b.y + b.ry)]
  if (ylo <= yhi + EPS && xGap) {
    const mid = (ylo + yhi) / 2
    for (const y of [...held(a.y, b.y, ylo, yhi, prefer), mid, (ylo + mid) / 2, (mid + yhi) / 2, ylo, yhi])
      out.push({
        points: [
          [exitH, y],
          [entryH, y],
        ],
        exit: exitSideH,
        entry: entrySideH,
      })
  }

  // L: leave by one side, turn once, enter by the other axis. `hv` leaves sideways and enters from above or below.
  const hv: Candidate | undefined =
    Math.abs(dx) > a.hw + EPS && Math.abs(dy) > b.hh + EPS
      ? {
          points: [
            [exitH, a.y],
            [b.x, a.y],
            [b.x, entryV],
          ],
          exit: exitSideH,
          entry: entrySideV,
        }
      : undefined
  const vh: Candidate | undefined =
    Math.abs(dy) > a.hh + EPS && Math.abs(dx) > b.hw + EPS
      ? {
          points: [
            [a.x, exitV],
            [a.x, b.y],
            [entryH, b.y],
          ],
          exit: exitSideV,
          entry: entrySideH,
        }
      : undefined
  // Enter along the flow; after a waypoint, turn there.
  let ls = flow === 'down' ? [hv, vh] : [vh, hv]
  if (incoming !== undefined) {
    const inHorizontal = Math.abs(incoming[0]) > 0.5
    const key = (c: Candidate | undefined): number => {
      if (c === undefined) return 2
      const horizontal = Math.abs((c.points[1] as Pt)[1] - (c.points[0] as Pt)[1]) < EPS
      return horizontal !== inHorizontal ? 0 : 1
    }
    ls = [...ls].sort((p, q) => key(p) - key(q))
  }
  for (const c of ls) if (c !== undefined) out.push(c)

  // Z: leave and enter by facing sides, with the middle segment across the middle of the gap.
  const ts = [0.5, 0.35, 0.65, 0.2, 0.8]
  const zv: Candidate[] =
    yGap && Math.abs(dx) > EPS
      ? ts.map((t) => {
          const y = lerp(exitV, entryV, t)
          return {
            points: [
              [a.x, exitV],
              [a.x, y],
              [b.x, y],
              [b.x, entryV],
            ],
            exit: exitSideV,
            entry: entrySideV,
          }
        })
      : []
  const zh: Candidate[] =
    xGap && Math.abs(dy) > EPS
      ? ts.map((t) => {
          const x = lerp(exitH, entryH, t)
          return {
            points: [
              [exitH, a.y],
              [x, a.y],
              [x, b.y],
              [entryH, b.y],
            ],
            exit: exitSideH,
            entry: entrySideH,
          }
        })
      : []
  out.push(...(flow === 'down' ? [...zv, ...zh] : [...zh, ...zv]))
  if (incoming === undefined) return out
  // Never turn back on the way in.
  return out.filter((c) => {
    const first = c.points[1] as Pt
    const start = c.points[0] as Pt
    const len = Math.hypot(first[0] - start[0], first[1] - start[1])
    if (len < EPS) return true
    return ((first[0] - start[0]) * incoming[0] + (first[1] - start[1]) * incoming[1]) / len > -0.5
  })
}

/** The port of a side: the middle of it, or the fraction `at` (0 to 1) of the way along the stretch ports may use. */
function portOf(o: Item, side: PortSide, at?: number): Pt {
  const [ux, uy] = DIRS[side]
  const p: Pt = [o.x + ux * o.hw, o.y + uy * o.hh]
  if (at !== undefined) {
    if (ux === 0) p[0] = o.x + (2 * at - 1) * o.rx
    else p[1] = o.y + (2 * at - 1) * o.ry
  }
  return p
}

/** The side of a box a point on its outline lies on, if any. */
function sideOf(o: Item, p: readonly [number, number]): PortSide | undefined {
  if (o.hw === 0 && o.hh === 0) return undefined
  const [dx, dy] = [p[0] - o.x, p[1] - o.y]
  let best: PortSide | undefined
  let gap = 1
  for (const [side, d] of [
    ['left', Math.abs(dx + o.hw)],
    ['right', Math.abs(dx - o.hw)],
    ['top', Math.abs(dy + o.hh)],
    ['bottom', Math.abs(dy - o.hh)],
  ] as const)
    if (d < gap) {
      gap = d
      best = side
    }
  return best
}

/**
 * A candidate with the end at `which` moved to the fraction `at` of the way along its side (the first or the last
 * stretch runs along the side's normal, so moving it moves the port); `undefined` when a straight route cannot hold that
 * position at its other end.
 */
function pinned(c: Candidate, which: 'exit' | 'entry', at: number, a: Item, b: Item): Candidate | undefined {
  const side = c[which]
  const k = side === 'top' || side === 'bottom' ? 0 : 1
  const o = which === 'exit' ? a : b
  const to = (k === 0 ? o.x : o.y) + (2 * at - 1) * (k === 0 ? o.rx : o.ry)
  const points = c.points.map((p): Pt => [p[0], p[1]])
  const n = points.length
  const [i0, i1] = which === 'exit' ? [0, 1] : [n - 1, n - 2]
  ;(points[i0] as Pt)[k] = to
  ;(points[i1] as Pt)[k] = to
  if (n === 2) {
    // A straight route: the other end lies on the same line, so it must be able to hold it.
    const other = which === 'exit' ? b : a
    if (Math.abs(to - (k === 0 ? other.x : other.y)) > (k === 0 ? other.rx : other.ry) + EPS) return undefined
  }
  return { points, exit: c.exit, entry: c.entry }
}

/** A binary min-heap of `[cost, order, state]`, ordered by cost then insertion order (so ties are deterministic). */
class Heap {
  private readonly items: [number, number, number][] = []
  private order = 0

  get size(): number {
    return this.items.length
  }

  push(cost: number, state: number): void {
    const items = this.items
    items.push([cost, this.order++, state])
    let i = items.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this.less(i, p)) {
        const t = items[i] as [number, number, number]
        items[i] = items[p] as [number, number, number]
        items[p] = t
        i = p
      } else break
    }
  }

  pop(): [number, number, number] {
    const items = this.items
    const top = items[0] as [number, number, number]
    const last = items.pop() as [number, number, number]
    if (items.length > 0) {
      items[0] = last
      let i = 0
      for (;;) {
        const [l, r] = [2 * i + 1, 2 * i + 2]
        let m = i
        if (l < items.length && this.less(l, m)) m = l
        if (r < items.length && this.less(r, m)) m = r
        if (m === i) break
        const t = items[i] as [number, number, number]
        items[i] = items[m] as [number, number, number]
        items[m] = t
        i = m
      }
    }
    return top
  }

  private less(i: number, j: number): boolean {
    const [p, q] = [this.items[i] as [number, number, number], this.items[j] as [number, number, number]]
    return p[0] < q[0] || (p[0] === q[0] && p[1] < q[1])
  }
}

const HEADING: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
]

/**
 * The cheapest right-angled path from a side of `a` to a side of `b` around the obstacles: Dijkstra over the grid of
 * lines through the obstacles' grown edges, with a cost for every bend and for leaving by a side that does not face
 * the other end.
 */
function detour(
  a: Item,
  b: Item,
  obstacles: readonly Item[],
  c: number,
  exits: readonly PortSide[],
  entries: readonly PortSide[],
  extra: {
    /** The cost of leaving `a` by a side, and of entering `b` by one (a side other edges use costs more). */
    readonly exitCost?: (side: PortSide) => number
    readonly entryCost?: (side: PortSide) => number
    /** Where along the side the ports lie, as a fraction (default the middle). */
    readonly fromAt?: number | undefined
    readonly toAt?: number | undefined
  } = {},
): Pt[] | undefined {
  const stub = c + 1
  const stubOf = (o: Item, side: PortSide, pin: number | undefined): { port: Pt; at: Pt } => {
    const raw = portOf(o, side, pin)
    // The grid's lines are rounded to a millionth: a port off the middle of its side must lie on one.
    const port: Pt = pin === undefined ? raw : [Math.round(raw[0] * 1e6) / 1e6, Math.round(raw[1] * 1e6) / 1e6]
    const [ux, uy] = DIRS[side]
    return { port, at: [port[0] + ux * stub, port[1] + uy * stub] }
  }
  const blockers = [...obstacles, a, b]
  const xsSet = new Set<number>()
  const ysSet = new Set<number>()
  const addAxis = (set: Set<number>, v: number): void => void set.add(Math.round(v * 1e6) / 1e6)
  for (const o of blockers) {
    addAxis(xsSet, o.x - o.hw - c)
    addAxis(xsSet, o.x + o.hw + c)
    addAxis(ysSet, o.y - o.hh - c)
    addAxis(ysSet, o.y + o.hh + c)
  }
  const starts = exits.map((s) => ({ side: s, ...stubOf(a, s, extra.fromAt) }))
  const goals = entries.map((s) => ({ side: s, ...stubOf(b, s, extra.toAt) }))
  for (const s of [...starts, ...goals]) {
    addAxis(xsSet, s.at[0])
    addAxis(ysSet, s.at[1])
  }
  addAxis(xsSet, (a.x + b.x) / 2)
  addAxis(ysSet, (a.y + b.y) / 2)
  const xs = [...xsSet].sort((p, q) => p - q)
  const ys = [...ysSet].sort((p, q) => p - q)
  const [nx, ny] = [xs.length, ys.length]
  const ix = (x: number): number => xs.indexOf(Math.round(x * 1e6) / 1e6)
  const iy = (y: number): number => ys.indexOf(Math.round(y * 1e6) / 1e6)
  const state = (i: number, j: number, h: number): number => (i * ny + j) * 4 + h
  const memo = new Map<number, boolean>()
  const free = (i: number, j: number, h: number): boolean => {
    // The segment from grid point (i, j) one step along heading h.
    const [i2, j2] = [
      i + (HEADING[h] as readonly [number, number])[0],
      j + (HEADING[h] as readonly [number, number])[1],
    ]
    if (i2 < 0 || i2 >= nx || j2 < 0 || j2 >= ny) return false
    const [lo, hi] = [state(Math.min(i, i2), Math.min(j, j2), 0), h % 2]
    const key = lo * 2 + hi
    const known = memo.get(key)
    if (known !== undefined) return known
    const p: Pt = [xs[i] as number, ys[j] as number]
    const q: Pt = [xs[i2] as number, ys[j2] as number]
    const ok = !blockers.some((o) => hits(p, q, o, c))
    memo.set(key, ok)
    return ok
  }
  const dist = new Float64Array(nx * ny * 4).fill(Number.POSITIVE_INFINITY)
  const from = new Int32Array(nx * ny * 4).fill(-1)
  const heap = new Heap()
  // A side that points away from the other end costs a detour of its own.
  const facing = (o: Item, side: PortSide, other: Item): number => {
    const [ux, uy] = DIRS[side]
    return ux * (other.x - o.x) + uy * (other.y - o.y) <= EPS ? WRONG_SIDE : 0
  }
  for (const s of starts) {
    const h = HEADING.findIndex((d) => d[0] === DIRS[s.side][0] && d[1] === DIRS[s.side][1])
    const [i, j] = [ix(s.at[0]), iy(s.at[1])]
    const st = state(i, j, h)
    const cost = stub + facing(a, s.side, b) + (extra.exitCost?.(s.side) ?? 0)
    if (cost < (dist[st] as number)) {
      dist[st] = cost
      heap.push(cost, st)
    }
  }
  while (heap.size > 0) {
    const [cost, , st] = heap.pop()
    if (cost > (dist[st] as number)) continue
    const h = st % 4
    const cell = (st - h) / 4
    const [i, j] = [Math.floor(cell / ny), cell % ny]
    for (const nh of [h, (h + 1) % 4, (h + 3) % 4]) {
      if (!free(i, j, nh)) continue
      const [i2, j2] = [
        i + (HEADING[nh] as readonly [number, number])[0],
        j + (HEADING[nh] as readonly [number, number])[1],
      ]
      const len = Math.abs((xs[i2] as number) - (xs[i] as number)) + Math.abs((ys[j2] as number) - (ys[j] as number))
      const next = cost + len + (nh === h ? 0 : BEND)
      const ns = state(i2, j2, nh)
      if (next < (dist[ns] as number) - 1e-9) {
        dist[ns] = next
        from[ns] = st
        heap.push(next, ns)
      }
    }
  }
  let best = Number.POSITIVE_INFINITY
  let bestState = -1
  let bestGoal: (typeof goals)[number] | undefined
  for (const g of goals) {
    const [i, j] = [ix(g.at[0]), iy(g.at[1])]
    const [ux, uy] = DIRS[g.side]
    for (let h = 0; h < 4; h++) {
      const hd = HEADING[h] as readonly [number, number]
      // Arriving by moving out of the box would turn back on the last stretch.
      if (hd[0] === ux && hd[1] === uy) continue
      const st = state(i, j, h)
      const total = (dist[st] as number) + stub + facing(b, g.side, a) + (extra.entryCost?.(g.side) ?? 0)
      if (total < best - 1e-9) {
        best = total
        bestState = st
        bestGoal = g
      }
    }
  }
  if (bestGoal === undefined || !Number.isFinite(best)) return undefined
  const grid: Pt[] = []
  for (let st = bestState; st >= 0; st = from[st] as number) {
    const cell = (st - (st % 4)) / 4
    grid.push([xs[Math.floor(cell / ny)] as number, ys[cell % ny] as number])
  }
  grid.reverse()
  const start = starts.find((s) => {
    const first = grid[0] as Pt
    return Math.abs(s.at[0] - first[0]) < 1e-6 && Math.abs(s.at[1] - first[1]) < 1e-6
  })
  return tidy([...(start === undefined ? [] : [start.port]), ...grid, bestGoal.port])
}

/** A straight line between two boxes that overlap each other, clipped to their outlines (a last resort). */
function overlapping(a: Item, b: Item): Pt[] {
  const [dx, dy] = [b.x - a.x, b.y - a.y]
  const clip = (o: Item): number =>
    Math.min(
      dx === 0 ? Number.POSITIVE_INFINITY : o.hw / Math.abs(dx),
      dy === 0 ? Number.POSITIVE_INFINITY : o.hh / Math.abs(dy),
      1,
    )
  const [ta, tb] = [clip(a), clip(b)]
  return [
    [a.x + dx * ta, a.y + dy * ta],
    [b.x - dx * tb, b.y - dy * tb],
  ]
}

/**
 * One leg of an edge: from `a` to `b`, which are boxes or waypoints. `undefined` when only a detour would do and none is
 * allowed.
 */
function routeLeg(
  a: Item,
  b: Item,
  obstacles: readonly Item[],
  options: {
    readonly flow: 'down' | 'right'
    readonly clearance: number
    readonly incoming?: readonly [number, number] | undefined
    /** The end where edges meet (it has several): a straight edge runs through that end's middle. */
    readonly prefer?: 'a' | 'b' | undefined
    readonly fromSide?: PortSide | undefined
    readonly toSide?: PortSide | undefined
    /** Where along the side the edge leaves and enters, as a fraction of the stretch ports may use. */
    readonly fromAt?: number | undefined
    readonly toAt?: number | undefined
    /** Whether a route would overdraw an edge routed already (merging at a shared port does not count). */
    readonly overdraws?: ((points: readonly Pt[]) => boolean) | undefined
    /** Whether a detour around the boxes may be taken (else `undefined` when no L or Z is clear). */
    readonly detour: boolean
    readonly exitCost?: ((side: PortSide) => number) | undefined
    readonly entryCost?: ((side: PortSide) => number) | undefined
  },
): Pt[] | undefined {
  const { flow, clearance: c, incoming, fromSide, toSide, prefer, fromAt, toAt } = options
  let all = candidates(a, b, flow, incoming, prefer).filter(
    (k) => (fromSide === undefined || k.exit === fromSide) && (toSide === undefined || k.entry === toSide),
  )
  if (fromAt !== undefined || toAt !== undefined)
    all = all.flatMap((k) => {
      const moved = fromAt === undefined ? k : pinned(k, 'exit', fromAt, a, b)
      const both = moved === undefined || toAt === undefined ? moved : pinned(moved, 'entry', toAt, a, b)
      return both === undefined ? [] : [both]
    })
  const fits = all.filter((k) => clear(k.points, obstacles, c))
  const first = fits.find((k) => options.overdraws?.(k.points) !== true) ?? fits[0]
  if (first !== undefined) return first.points
  if (!options.detour) return undefined
  const extra = {
    ...(options.exitCost === undefined ? {} : { exitCost: options.exitCost }),
    ...(options.entryCost === undefined ? {} : { entryCost: options.entryCost }),
    fromAt,
    toAt,
  }
  const found = (window: readonly Item[]): Pt[] | undefined =>
    detour(a, b, window, c, fromSide === undefined ? SIDES : [fromSide], toSide === undefined ? SIDES : [toSide], extra)
  // Look among the boxes near the two ends first (the grid is small), then among them all.
  const [x0, x1] = [Math.min(a.x - a.hw, b.x - b.hw) - 80, Math.max(a.x + a.hw, b.x + b.hw) + 80]
  const [y0, y1] = [Math.min(a.y - a.hh, b.y - b.hh) - 80, Math.max(a.y + a.hh, b.y + b.hh) + 80]
  const near = obstacles.filter((o) => o.x + o.hw > x0 && o.x - o.hw < x1 && o.y + o.hh > y0 && o.y - o.hh < y1)
  let routed = near.length < obstacles.length ? found(near) : undefined
  if (routed === undefined || !clear(routed, obstacles, c)) routed = found(obstacles.slice(0, 400)) ?? routed
  if (routed !== undefined) return routed
  return (all[0] ?? candidates(a, b, flow, undefined, undefined)[0])?.points ?? overlapping(a, b)
}

/** Whether two polylines run along each other (collinear segments that overlap for more than a pixel). */
function runsAlong(p: readonly (readonly [number, number])[], q: readonly (readonly [number, number])[]): boolean {
  for (let i = 0; i + 1 < p.length; i++)
    for (let j = 0; j + 1 < q.length; j++) {
      const [a, b, c, d] = [p[i], p[i + 1], q[j], q[j + 1]] as [Pt, Pt, Pt, Pt]
      for (const k of [0, 1] as const) {
        const m: 0 | 1 = k === 0 ? 1 : 0
        const flat = Math.abs(a[m] - b[m]) < EPS && Math.abs(c[m] - d[m]) < EPS && Math.abs(a[m] - c[m]) < 0.5
        if (
          flat &&
          Math.min(Math.max(a[k], b[k]), Math.max(c[k], d[k])) - Math.max(Math.min(a[k], b[k]), Math.min(c[k], d[k])) >
            1
        )
          return true
      }
    }
  return false
}

/**
 * Route edges between placed boxes with right angles. Each edge leaves its source on the side facing the target and
 * enters the target perpendicular to its side:
 * - when the boxes overlap on an axis (as far as the stretches of their sides that ports may use), one straight
 *   segment joins them, in the middle of the overlap;
 * - otherwise an L (one bend) or a Z (two bends, the middle segment across the middle of the gap), whichever clears
 *   every other box and has the fewest bends (an L that enters along the flow comes first);
 * - when none is clear, a detour around the boxes in the way, with the fewest bends that fit.
 *
 * An edge with waypoints (`via`) is routed leg by leg through them (a waypoint is a corner: the route turns there).
 * Afterwards parallel segments of different edges are spread onto tracks (`separateRoutes`): edges that leave or enter
 * at the same port stay merged where they overlap. Deterministic: the same boxes and edges always give the same routes.
 *
 * @param nodes - The node boxes by id.
 * @param edges - The edges (with optional waypoints and forced sides); an edge whose ends are not both placed, or that
 *   joins a node to itself, has no route.
 * @param options - Flow, clearance, spacing and port reach.
 * @returns One route per routable edge, in order, each marked `route: 'orthogonal'`.
 * @example
 * ```ts
 * import { orthogonalRoutes } from 'omai-render/layout'
 * const nodes = { a: { x: 0, y: 0, width: 40, height: 20 }, b: { x: 100, y: 60, width: 40, height: 20 } }
 * const [r] = orthogonalRoutes(nodes, [{ from: 'a', to: 'b' }])
 * r?.points.length // => 3
 * ```
 */
export function orthogonalRoutes(
  nodes: Readonly<Record<string, LayoutBox>>,
  edges: readonly OrthogonalEdge[],
  options: OrthogonalOptions = {},
): EdgeRoute[] {
  const { flow = 'down', clearance = 6, spacing = 8, reach = {}, passable = [], merge = true } = options
  const items = new Map(Object.entries(nodes).map(([id, box]) => [id, itemOf(box, reach[id])]))
  const free = new Set(passable)
  const blocking = [...items.entries()].filter(([id]) => !free.has(id)).map(([, o]) => o)
  const degree = new Map<string, { out: number; in: number }>()
  const live = edges.filter((e) => e.from !== e.to && items.has(e.from) && items.has(e.to))
  for (const e of edges) {
    if (e.from === e.to) continue
    const [u, v] = [degree.get(e.from) ?? { out: 0, in: 0 }, degree.get(e.to) ?? { out: 0, in: 0 }]
    u.out++
    v.in++
    degree.set(e.from, u)
    degree.set(e.to, v)
  }
  // Edges merge at a port only when they look the same (their `mergeKey`); without merging, none do.
  const keys = live.map((e, i) => (merge ? (e.mergeKey ?? '') : `#${i}`))
  const done: { readonly a: Item; readonly b: Item; readonly points: Pt[]; readonly key: string }[] = []
  // Edges that meet at a port do not count (they merge, or are given ports side by side afterwards).
  const overdraws = (points: readonly Pt[]): boolean =>
    done.some((r) => {
      const [p0, p1] = [points[0] as Pt, points[points.length - 1] as Pt]
      const [q0, q1] = [r.points[0] as Pt, r.points[r.points.length - 1] as Pt]
      const merges = (u: Pt, v: Pt): boolean => Math.abs(u[0] - v[0]) < 0.5 && Math.abs(u[1] - v[1]) < 0.5
      return !merges(p0, q0) && !merges(p1, q1) && runsAlong(points, r.points)
    })
  // A side another kind of edge already uses costs a detour extra: a back edge leaves a free side instead.
  const used =
    (o: Item, key: string) =>
    (side: PortSide): number =>
      done.some(
        (r) =>
          r.key !== key &&
          ((r.a === o && sideOf(o, r.points[0] as Pt) === side) ||
            (r.b === o && sideOf(o, r.points[r.points.length - 1] as Pt) === side)),
      )
        ? OCCUPIED
        : 0
  const route = (e: OrthogonalEdge, key: string, detours: boolean): Pt[] | undefined => {
    const a = items.get(e.from) as Item
    const b = items.get(e.to) as Item
    const anchors = [a, ...(e.via ?? []).map(pointItem), b]
    let points: Pt[] = []
    let incoming: [number, number] | undefined
    for (let k = 0; k + 1 < anchors.length; k++) {
      const [u, v] = [anchors[k] as Item, anchors[k + 1] as Item]
      const first = k === 0
      const last = k + 2 === anchors.length
      const leg = routeLeg(
        u,
        v,
        blocking.filter((o) => o !== u && o !== v),
        {
          flow,
          clearance,
          incoming,
          fromSide: first ? e.fromSide : undefined,
          toSide: last ? e.toSide : undefined,
          fromAt: first ? e.fromAt : undefined,
          toAt: last ? e.toAt : undefined,
          overdraws,
          detour: detours,
          exitCost: first ? used(u, key) : undefined,
          entryCost: last ? used(v, key) : undefined,
          prefer:
            anchors.length === 2
              ? (degree.get(e.to)?.in ?? 0) > 1
                ? 'b'
                : (degree.get(e.from)?.out ?? 0) > 1
                  ? 'a'
                  : undefined
              : undefined,
        },
      )
      if (leg === undefined) return undefined
      points = points.length === 0 ? leg : [...points, ...leg]
      incoming = lastDirection(leg) ?? incoming
    }
    return tidy(points)
  }
  // Edges an L or a Z serves first, then the ones that need a detour, which see all of those and keep off them.
  const routed: (Pt[] | undefined)[] = live.map(() => undefined)
  for (const detours of [false, true])
    live.forEach((e, i) => {
      if (routed[i] !== undefined) return
      const points = route(e, keys[i] as string, detours)
      if (points === undefined) return
      routed[i] = points
      done.push({ a: items.get(e.from) as Item, b: items.get(e.to) as Item, points, key: keys[i] as string })
    })
  const routes: EdgeRoute[] = live.map((e, i) => ({
    from: e.from,
    to: e.to,
    points: routed[i] as Pt[],
    route: 'orthogonal',
  }))
  return separateRoutes(routes, {
    spacing,
    clearance,
    nodes,
    reach,
    passable,
    ends: live.map((e, i) => ({
      key: keys[i] as string,
      pinFrom: e.fromAt !== undefined,
      pinTo: e.toAt !== undefined,
    })),
  })
}

interface Segment {
  readonly route: number
  /** The index of its first point in the route. */
  readonly index: number
  readonly horizontal: boolean
  readonly lo: number
  readonly hi: number
  /** The coordinate across it (y of a horizontal segment, x of a vertical one). */
  readonly at: number
  /** The first or the last segment of its route: it sits on a port and does not move. */
  readonly fixed: boolean
}

/**
 * Where an edge leaves a node at a port at which another enters it, the two would overdraw each other along their end
 * segments: each moves a little to its left as it runs, so they run side by side.
 */
function splitPorts(routes: readonly EdgeRoute[], pts: Pt[][], by: number): void {
  const key = (id: string, p: Pt): string => `${id}@${Math.round(p[0])},${Math.round(p[1])}`
  const ends = new Map<string, { leaving: number; entering: number }>()
  const touch = (k: string, leaving: boolean): void => {
    const e = ends.get(k) ?? { leaving: 0, entering: 0 }
    if (leaving) e.leaving++
    else e.entering++
    ends.set(k, e)
  }
  const at = (i: number): { first: string; last: string } | undefined => {
    const [p, r] = [pts[i] as Pt[], routes[i] as EdgeRoute]
    if (p.length < 2) return undefined
    return { first: key(r.from, p[0] as Pt), last: key(r.to, p[p.length - 1] as Pt) }
  }
  pts.forEach((_, i) => {
    const k = at(i)
    if (k === undefined) return
    touch(k.first, true)
    touch(k.last, false)
  })
  const mixed = (k: string): boolean => {
    const e = ends.get(k)
    return e !== undefined && e.leaving > 0 && e.entering > 0
  }
  const shift = (a: Pt, b: Pt, c: Pt | undefined, moved: Pt[], sign: number): void => {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    if (len < EPS) return
    const [nx, ny] = [((b[1] - a[1]) / len) * by * sign, (-(b[0] - a[0]) / len) * by * sign]
    // The next segment, across this one, keeps its direction and a stub's length.
    if (c !== undefined) {
      const along = nx !== 0 ? 0 : 1
      const [was, now] = [b[along] - c[along], b[along] + (along === 0 ? nx : ny) - c[along]]
      if (Math.abs(was) >= EPS && (was * now <= 0 || Math.abs(now) < MIN_STUB)) return
    }
    for (const q of moved) {
      q[0] += nx
      q[1] += ny
    }
  }
  pts.forEach((p, i) => {
    const k = at(i)
    if (k === undefined) return
    const n = p.length
    if (n === 2) {
      if (mixed(k.first) || mixed(k.last)) shift(p[0] as Pt, p[1] as Pt, undefined, [p[0] as Pt, p[1] as Pt], 1)
      return
    }
    if (mixed(k.first)) shift(p[0] as Pt, p[1] as Pt, p[2] as Pt, [p[0] as Pt, p[1] as Pt], 1)
    if (mixed(k.last)) shift(p[n - 1] as Pt, p[n - 2] as Pt, p[n - 3] as Pt, [p[n - 1] as Pt, p[n - 2] as Pt], -1)
  })
}

/** One end of a route on a side of its node: where along the side, and how the route runs off it. */
interface PortEnd {
  readonly route: number
  readonly first: boolean
  readonly node: string
  readonly side: PortSide
  /** The coordinate along the side. */
  readonly at: number
  /** Which way the route turns along the side after its first stretch (-1, 0 for straight, 1). */
  readonly run: number
  /** How far from the side that turn is. */
  readonly depth: number
  readonly pinned: boolean
}

/** The side a stretch from a port to its first corner leaves by, if it is horizontal or vertical. */
function normalSide(port: Pt, corner: Pt): PortSide | undefined {
  const [dx, dy] = [corner[0] - port[0], corner[1] - port[1]]
  if (Math.abs(dy) < EPS && Math.abs(dx) >= EPS) return dx > 0 ? 'right' : 'left'
  if (Math.abs(dx) < EPS && Math.abs(dy) >= EPS) return dy > 0 ? 'bottom' : 'top'
  return undefined
}

/**
 * Edges of different kinds that leave or enter a node at one port would overdraw each other (the later one hides the
 * earlier): give each kind a port of its own, side by side along the side, in the order that lets their first turns
 * nest without crossing (edges turning one way take the lesser positions, the farther turns the outer ones).
 */
function spreadPorts(
  routes: readonly EdgeRoute[],
  pts: Pt[][],
  ctx: {
    readonly items: ReadonlyMap<string, Item>
    readonly boxes: readonly { readonly id: string; readonly item: Item }[]
    readonly keyOf: (route: number) => string
    readonly ends: readonly RouteEnds[]
    readonly spacing: number
    readonly clearance: number
  },
): void {
  const { items, keyOf, ends, spacing, clearance } = ctx
  const groups = new Map<string, PortEnd[]>()
  routes.forEach((r, route) => {
    const p = pts[route] as Pt[]
    const n = p.length
    if (n < 2) return
    for (const first of [true, false]) {
      const node = first ? r.from : r.to
      const item = items.get(node)
      if (item === undefined) continue
      const [port, corner, next] = first ? [p[0] as Pt, p[1] as Pt, p[2]] : [p[n - 1] as Pt, p[n - 2] as Pt, p[n - 3]]
      const side = normalSide(port, corner)
      if (side === undefined || sideOf(item, port) !== side) continue
      const k: 0 | 1 = side === 'top' || side === 'bottom' ? 0 : 1
      const run = next === undefined ? 0 : Math.abs(next[k] - corner[k]) < EPS ? 0 : Math.sign(next[k] - corner[k])
      const end: PortEnd = {
        route,
        first,
        node,
        side,
        at: port[k],
        run,
        depth: Math.abs(corner[k === 0 ? 1 : 0] - port[k === 0 ? 1 : 0]),
        pinned: (first ? ends[route]?.pinFrom : ends[route]?.pinTo) === true,
      }
      const id = `${node}|${side}`
      const list = groups.get(id)
      if (list === undefined) groups.set(id, [end])
      else list.push(end)
    }
  })
  const moveEnd = (e: PortEnd, to: number): void => {
    const p = pts[e.route] as Pt[]
    const n = p.length
    const k = e.side === 'top' || e.side === 'bottom' ? 0 : 1
    const [i0, i1] = e.first ? [0, 1] : [n - 1, n - 2]
    if (Math.abs((p[i0] as Pt)[k] - to) < EPS) return
    const route = routes[e.route] as EdgeRoute
    if (n === 2) {
      // A straight edge: the other end lies on the same line, so it must be able to hold it.
      const other = items.get(e.first ? route.to : route.from)
      if (other === undefined) return
      const reach = k === 0 ? other.rx : other.ry
      if (Math.abs(to - (k === 0 ? other.x : other.y)) > reach + EPS) return
    } else {
      // The next stretch, across this one, keeps its direction and a stub's length.
      const next = p[e.first ? 2 : n - 3] as Pt
      const [was, now] = [(p[i1] as Pt)[k] - next[k], to - next[k]]
      if (Math.abs(was) >= EPS && (was * now <= 0 || Math.abs(now) < MIN_STUB)) return
    }
    // What moves: the port and the corner after it (a straight edge's two ends), checked against the other boxes.
    const probe = (n === 2 ? [p[0], p[1]] : [p[i0], p[i1], p[e.first ? 2 : n - 3]]).map(
      (q): Pt => [(q as Pt)[0], (q as Pt)[1]],
    )
    ;(probe[0] as Pt)[k] = to
    ;(probe[1] as Pt)[k] = to
    const own = new Set([route.from, route.to])
    const others = ctx.boxes.filter((o) => !own.has(o.id)).map((o) => o.item)
    if (!clear(probe, others, clearance)) return
    ;(p[i0] as Pt)[k] = to
    ;(p[i1] as Pt)[k] = to
  }
  for (const [id, list] of groups) {
    const item = items.get(id.slice(0, id.lastIndexOf('|')))
    if (item === undefined) continue
    const side = list[0]?.side as PortSide
    const k = side === 'top' || side === 'bottom' ? 0 : 1
    const [centre, half] = k === 0 ? [item.x, item.rx] : [item.y, item.ry]
    list.sort((u, v) => u.at - v.at || u.route - v.route)
    let cluster: PortEnd[] = []
    const flush = (): void => {
      const classes = new Map<string, PortEnd[]>()
      for (const e of cluster) {
        const key = keyOf(e.route)
        const c = classes.get(key)
        if (c === undefined) classes.set(key, [e])
        else c.push(e)
      }
      cluster = []
      const m = classes.size
      if (m < 2 || half <= 0) return
      // Turning one way, the nearer turns take the inner positions; a straight edge goes between.
      const order = [...classes.values()].sort((u, v) => {
        const [a, b] = [u[0] as PortEnd, v[0] as PortEnd]
        if (a.run !== b.run) return a.run - b.run
        return (a.run >= 0 ? -1 : 1) * (a.depth - b.depth) || a.route - b.route
      })
      const step = Math.min(spacing * 1.5, (2 * half) / (m - 1))
      const anchored = order.findIndex((c) => c.some((e) => e.pinned))
      order.forEach((c, j) => {
        if (c.some((e) => e.pinned)) return
        const offset = (j - (anchored >= 0 ? anchored : (m - 1) / 2)) * step
        for (const e of c) moveEnd(e, Math.min(centre + half, Math.max(centre - half, e.at + offset)))
      })
    }
    for (const e of list) {
      const last = cluster[cluster.length - 1]
      if (last !== undefined && e.at - last.at > 0.5) flush()
      cluster.push(e)
    }
    flush()
  }
}

/** What `separateRoutes` knows of an edge beyond its route. */
export interface RouteEnds {
  /** The edges that may merge at a port are those with the same key (default all of them). */
  readonly key?: string
  /** The edge leaves at a position of its own choosing (`fromAt`): its first port does not move. */
  readonly pinFrom?: boolean
  /** The edge enters at a position of its own choosing (`toAt`): its last port does not move. */
  readonly pinTo?: boolean
}

/**
 * Spread the parallel segments of different routes so they do not overdraw. Segments of different edges that lie on
 * the same line and overlap (a horizontal run through a rank channel, say) are put on tracks `spacing` apart, centred
 * on the line; segments of edges that leave or enter at the same port are merged, so they stay on one line where they
 * overlap, but only when the edges are the same kind (their `key`): edges of different kinds that meet at one port are
 * first given ports side by side along the side, in the order that keeps them from crossing. The first and last
 * segment of a route sit on ports and stay put; the others move only while they keep clear of the other boxes and of
 * their own neighbours' ends.
 *
 * @param routes - Orthogonal routes (segments that are neither horizontal nor vertical are left alone).
 * @param options - The spacing between tracks (default 8), the clearance from boxes (default 6), the boxes to keep
 *   clear of (the ends of each route are exempt; `passable` names those that are not obstacles), the ports' `reach`
 *   by node id, and `ends`, one entry per route (its merge key and whether its ports are pinned).
 * @returns The routes with their ports and middle segments moved, each marked `route: 'orthogonal'`.
 * @example
 * ```ts
 * import { separateRoutes } from 'omai-render/layout'
 * const r = (from: string, x0: number, x1: number): { from: string; to: string; points: [number, number][] } => ({
 *   from, to: `${from}'`, points: [[x0, 0], [x0, 20], [x1, 20], [x1, 40]],
 * })
 * const [p, q] = separateRoutes([r('a', 0, 50), r('b', 10, 60)])
 * p?.points[1]?.[1] === q?.points[1]?.[1] // => false
 * ```
 */
export function separateRoutes(
  routes: readonly EdgeRoute[],
  options: {
    readonly spacing?: number
    readonly clearance?: number
    readonly nodes?: Readonly<Record<string, LayoutBox>>
    readonly reach?: Readonly<Record<string, number>>
    readonly passable?: readonly string[]
    readonly ends?: readonly RouteEnds[]
  } = {},
): EdgeRoute[] {
  const { spacing = 8, clearance = 6, nodes = {}, reach = {}, passable = [], ends = [] } = options
  const pts: Pt[][] = routes.map((r) => tidy(r.points))
  const free = new Set(passable)
  const items = new Map(Object.entries(nodes).map(([id, b]) => [id, itemOf(b, reach[id])]))
  const boxes = Object.entries(nodes)
    .filter(([id]) => !free.has(id))
    .map(([id, b]) => ({ id, item: itemOf(b, reach[id]) }))
  const keyOf = (route: number): string => ends[route]?.key ?? ''
  spreadPorts(routes, pts, { items, boxes, keyOf, ends, spacing, clearance })
  splitPorts(routes, pts, spacing / 2)
  const segments: Segment[] = []
  pts.forEach((p, route) => {
    for (let i = 0; i + 1 < p.length; i++) {
      const [a, b] = [p[i] as Pt, p[i + 1] as Pt]
      const horizontal = Math.abs(a[1] - b[1]) < EPS
      if (!horizontal && Math.abs(a[0] - b[0]) >= EPS) continue
      segments.push({
        route,
        index: i,
        horizontal,
        lo: horizontal ? Math.min(a[0], b[0]) : Math.min(a[1], b[1]),
        hi: horizontal ? Math.max(a[0], b[0]) : Math.max(a[1], b[1]),
        at: horizontal ? a[1] : a[0],
        fixed: i === 0 || i === p.length - 2,
      })
    }
  })
  const same = (p: Pt, q: Pt): boolean => Math.abs(p[0] - q[0]) < 0.5 && Math.abs(p[1] - q[1]) < 0.5
  // Edges that leave at one port, or enter at one, are one line where they overlap.
  const merged = (s: Segment, t: Segment): boolean => {
    const [p, q] = [pts[s.route] as Pt[], pts[t.route] as Pt[]]
    return (
      keyOf(s.route) === keyOf(t.route) &&
      (same(p[0] as Pt, q[0] as Pt) || same(p[p.length - 1] as Pt, q[q.length - 1] as Pt))
    )
  }
  const conflict = (s: Segment, t: Segment): boolean =>
    Math.min(s.hi, t.hi) - Math.max(s.lo, t.lo) > 0.5 && !merged(s, t)

  const moveTo = (s: Segment, to: number): boolean => {
    const p = pts[s.route] as Pt[]
    const k = s.horizontal ? 1 : 0
    const [a, b] = [p[s.index] as Pt, p[s.index + 1] as Pt]
    const [before, after] = [p[s.index - 1] as Pt | undefined, p[s.index + 2] as Pt | undefined]
    if (before === undefined || after === undefined) return false
    // The segments either side keep their direction and a stub's length.
    for (const [other, end] of [
      [before, a],
      [after, b],
    ] as const) {
      const [was, now] = [end[k] - other[k], to - other[k]]
      if (Math.abs(was) >= EPS && (was * now <= 0 || Math.abs(now) < MIN_STUB)) return false
    }
    const moved: Pt[] = [before, s.horizontal ? [a[0], to] : [to, a[1]], s.horizontal ? [b[0], to] : [to, b[1]], after]
    const own = new Set([routes[s.route]?.from, routes[s.route]?.to])
    const others = boxes.filter((o) => !own.has(o.id)).map((o) => o.item)
    if (!clear(moved, others, clearance)) return false
    a[k] = to
    b[k] = to
    return true
  }

  for (const horizontal of [true, false]) {
    const list = segments.filter((s) => s.horizontal === horizontal).sort((p, q) => p.at - q.at)
    let cluster: Segment[] = []
    const flush = (): void => {
      const movable = cluster.filter((s) => !s.fixed)
      if (cluster.length > 1 && movable.length > 0) {
        const ordered = [
          ...cluster.filter((s) => s.fixed),
          ...[...movable].sort((p, q) => p.lo - q.lo || p.hi - q.hi || p.route - q.route),
        ]
        const tracks: Segment[][] = []
        const hasFixed = cluster.some((s) => s.fixed)
        for (const s of ordered) {
          // Fixed segments stay on the first track; the others take the first track free of what they would overdraw.
          let k = 0
          if (!s.fixed) while (tracks[k]?.some((t) => conflict(s, t)) === true) k++
          const track = tracks[k] ?? []
          track.push(s)
          tracks[k] = track
        }
        if (tracks.length > 1) {
          const base = cluster.reduce((m, s) => m + s.at, 0) / cluster.length
          tracks.forEach((track, k) => {
            const offset = hasFixed
              ? (k % 2 === 1 ? 1 : -1) * Math.ceil(k / 2) * spacing
              : (k - (tracks.length - 1) / 2) * spacing
            const target = (hasFixed ? (cluster.find((s) => s.fixed) as Segment).at : base) + offset
            for (const s of track) {
              if (s.fixed) continue
              for (const f of [1, 0.6, 0.3]) {
                const to = s.at + (target - s.at) * f
                if (moveTo(s, to)) break
              }
            }
          })
        }
      }
      cluster = []
    }
    for (const s of list) {
      const last = cluster[cluster.length - 1]
      if (last !== undefined && s.at - last.at > 0.75) flush()
      cluster.push(s)
    }
    flush()
  }
  return routes.map((r, i) => ({ ...r, points: tidy(pts[i] as Pt[]), route: 'orthogonal' }))
}
