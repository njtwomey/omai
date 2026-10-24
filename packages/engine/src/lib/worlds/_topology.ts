// Coordinates, directions and shapes of the three topologies (square grids, pointy-top hexagonal grids in odd-r offset
// coordinates, and graphs). Private to worlds: world.ts exposes these through the World functions.
import type { Topology } from './world.ts'

/** Row and column offsets of the square directions N, E, S, W, NE, SE, SW, NW. */
export const SQUARE_OFFSETS: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 1],
  [1, 1],
  [1, -1],
  [-1, -1],
]

/** Row and column offsets of the hexagonal directions E, SE, SW, W, NW, NE on even rows (odd-r layout). */
const HEX_EVEN: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
]

/** The same on odd rows, which sit half a cell to the right. */
const HEX_ODD: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
  [-1, 1],
]

/** The circumradius of a hexagon one unit wide (centre to corner). */
export const HEX_RADIUS: number = 1 / Math.sqrt(3)

/** Number of cells. */
export function cellCount(t: Topology): number {
  return t.kind === 'graph' ? t.n : t.rows * t.cols
}

/** Number of sides a cell has (wall bits): 4, 6, or the node's degree. */
export function sideCount(t: Topology, cell: number): number {
  if (t.kind === 'square') return 4
  if (t.kind === 'hex') return 6
  return t.neighbours[cell]?.length ?? 0
}

/** Number of move directions: 8 on square grids (4 sides then 4 diagonals), 6 on hexagonal grids, the degree on graphs. */
export function directionCount(t: Topology, cell: number): number {
  return t.kind === 'square' ? 8 : sideCount(t, cell)
}

/** The row and column offsets of a direction from a cell of a square or hexagonal grid. */
export function offset(t: Topology, cell: number, direction: number): readonly [number, number] {
  if (t.kind === 'square') return SQUARE_OFFSETS[direction] ?? [0, 0]
  if (t.kind === 'hex') {
    const r = Math.floor(cell / t.cols)
    return ((r & 1) === 1 ? HEX_ODD : HEX_EVEN)[direction] ?? [0, 0]
  }
  return [0, 0]
}

/** The cell one step from `cell` in `direction`, or -1 off the map. */
export function step(t: Topology, cell: number, direction: number): number {
  if (t.kind === 'graph') return t.neighbours[cell]?.[direction] ?? -1
  if (direction < 0 || direction >= (t.kind === 'square' ? 8 : 6)) return -1
  const r = Math.floor(cell / t.cols)
  const c = cell % t.cols
  const [dr, dc] = offset(t, cell, direction)
  const r2 = r + dr
  const c2 = c + dc
  if (r2 < 0 || r2 >= t.rows || c2 < 0 || c2 >= t.cols) return -1
  return r2 * t.cols + c2
}

/** The side of `to` that faces `from` (they must be adjacent across a side), or -1. */
export function oppositeSide(t: Topology, from: number, side: number, to: number): number {
  if (t.kind === 'square') return side < 4 ? (side + 2) % 4 : -1
  if (t.kind === 'hex') return (side + 3) % 6
  return t.neighbours[to]?.indexOf(from) ?? -1
}

/** The side of `a` across which `b` lies, or -1 when they are not side neighbours. */
export function sideBetween(t: Topology, a: number, b: number): number {
  const sides = sideCount(t, a)
  for (let s = 0; s < sides; s++) if (step(t, a, s) === b) return s
  return -1
}

/** The centre of a cell in world units (square cells are unit squares; hexagons are one unit wide). */
export function centre(t: Topology, cell: number): [number, number] {
  if (t.kind === 'graph') {
    const p = t.positions?.[cell]
    if (p !== undefined) return [p[0], p[1]]
    const angle = (2 * Math.PI * cell) / Math.max(1, t.n) - Math.PI / 2
    return [1 + Math.cos(angle), 1 + Math.sin(angle)]
  }
  const r = Math.floor(cell / t.cols)
  const c = cell % t.cols
  if (t.kind === 'square') return [c + 0.5, r + 0.5]
  return [c + 0.5 + 0.5 * (r & 1), HEX_RADIUS * (1 + 1.5 * r)]
}

/** The corners of a cell, in order, so side `k` runs from corner `k` to corner `k + 1`. */
export function corners(t: Topology, cell: number): [number, number][] {
  if (t.kind === 'graph') return []
  if (t.kind === 'square') {
    const r = Math.floor(cell / t.cols)
    const c = cell % t.cols
    return [
      [c, r],
      [c + 1, r],
      [c + 1, r + 1],
      [c, r + 1],
    ]
  }
  const [x, y] = centre(t, cell)
  const out: [number, number][] = []
  for (let k = 0; k < 6; k++) {
    const a = ((-30 + 60 * k) * Math.PI) / 180
    out.push([x + HEX_RADIUS * Math.cos(a), y + HEX_RADIUS * Math.sin(a)])
  }
  return out
}

/** The width and height of the map in world units. */
export function extent(t: Topology): [number, number] {
  if (t.kind === 'square') return [t.cols, t.rows]
  if (t.kind === 'hex') return [t.cols + (t.rows > 1 ? 0.5 : 0), HEX_RADIUS * (2 + 1.5 * (t.rows - 1))]
  let w = 0
  let h = 0
  for (let v = 0; v < t.n; v++) {
    const [x, y] = centre(t, v)
    w = Math.max(w, x + 1)
    h = Math.max(h, y + 1)
  }
  return [w, h]
}

/** The cell containing a point, or -1 off the map (graphs have no cells in the plane). */
export function cellAt(t: Topology, x: number, y: number): number {
  if (t.kind === 'graph') return -1
  if (t.kind === 'square') {
    const r = Math.floor(y)
    const c = Math.floor(x)
    return r < 0 || r >= t.rows || c < 0 || c >= t.cols ? -1 : r * t.cols + c
  }
  // Pixel to fractional axial coordinates, cube rounding, then axial to odd-r offset.
  const px = x - 0.5
  const py = y - HEX_RADIUS
  const qf = ((Math.sqrt(3) / 3) * px - py / 3) / HEX_RADIUS
  const rf = ((2 / 3) * py) / HEX_RADIUS
  const sf = -qf - rf
  let q = Math.round(qf)
  let r = Math.round(rf)
  const s = Math.round(sf)
  const dq = Math.abs(q - qf)
  const dr = Math.abs(r - rf)
  const ds = Math.abs(s - sf)
  if (dq > dr && dq > ds) q = -r - s
  else if (dr > ds) r = -q - s
  const col = q + (r - (r & 1)) / 2
  return r < 0 || r >= t.rows || col < 0 || col >= t.cols ? -1 : r * t.cols + col
}

/** Axial coordinates `[q, r]` of an odd-r hexagonal cell. */
export function axial(cols: number, cell: number): [number, number] {
  const r = Math.floor(cell / cols)
  const c = cell % cols
  return [c - (r - (r & 1)) / 2, r]
}
