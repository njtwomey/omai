import { WhiteboxError } from '../base/index.ts'
import { closestPointOnSegment, raySegmentHit, segmentsTouch, sweptCircleHit } from '../numerics/geometry/index.ts'
import { alongSegment, buildIndex, inBox, type SegmentIndex, segmentOf } from './_geometry.ts'
import { step } from './_topology.ts'
import { cellAtPoint, cellCentre, cellCorners, hasWall, passable, terrainOf, type World, worldSize } from './world.ts'

/** A continuous agent's pose: a position in world units and a heading in radians (0 east, growing clockwise on screen). */
export interface Pose {
  readonly x: number
  readonly y: number
  readonly heading: number
}

/**
 * The continuous geometry of a square or hexagonal world: its solid segments (walls, the sides of impassable cells
 * and the map edge, which block motion) and opaque segments (walls and the sides of opaque cells, which block sight),
 * each bucketed for fast queries.
 *
 * @concept collision-detection
 */
export interface WorldGeometry {
  readonly world: World
  readonly solid: SegmentIndex
  readonly opaque: SegmentIndex
}

function segmentsOf(w: World, blocks: (cell: number) => boolean): number[] {
  const t = w.topology
  if (t.kind === 'graph') throw new WhiteboxError('value', 'worlds: graph worlds have no continuous geometry')
  const out: number[] = []
  for (let c = 0; c < w.n; c++) {
    if (blocks(c)) continue
    const cs = cellCorners(w, c)
    for (let side = 0; side < cs.length; side++) {
      const nb = step(t, c, side)
      const edge = nb < 0 || blocks(nb)
      if (!edge && !(hasWall(w, c, side) && c < nb)) continue
      const a = cs[side] ?? ([0, 0] as const)
      const b = cs[(side + 1) % cs.length] ?? ([0, 0] as const)
      out.push(a[0], a[1], b[0], b[1])
    }
  }
  return out
}

/**
 * The wall segments of a world: `'solid'` ones block motion (walls between cells, sides of impassable cells facing
 * passable ones, and the map edge); `'opaque'` ones block sight (walls, and sides of opaque cells facing clear ones).
 *
 * @param w - A square or hexagonal world.
 * @param kind - Which segments (default `'solid'`).
 * @returns The segments as point pairs.
 * @throws WhiteboxError with code `value` for graph worlds.
 * @concept collision-detection
 * @example
 * ```ts
 * import { fromAscii, wallSegments } from 'whitebox-engine/worlds'
 * wallSegments(fromAscii(['.'])) // => [[[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]]
 * wallSegments(fromAscii(['..#'])).length // => 6
 * ```
 */
export function wallSegments(w: World, kind: 'solid' | 'opaque' = 'solid'): [[number, number], [number, number]][] {
  const flat = kind === 'solid' ? segmentsOf(w, (c) => !passable(w, c)) : segmentsOf(w, (c) => terrainOf(w, c).opaque)
  const out: [[number, number], [number, number]][] = []
  for (let i = 0; i < flat.length; i += 4) {
    out.push([
      [flat[i] ?? 0, flat[i + 1] ?? 0],
      [flat[i + 2] ?? 0, flat[i + 3] ?? 0],
    ])
  }
  return out
}

/**
 * Precompute the continuous geometry of a world for motion, sight and distance queries.
 *
 * @param w - A square or hexagonal world.
 * @returns The geometry.
 * @throws WhiteboxError with code `value` for graph worlds.
 * @concept collision-detection
 * @example
 * ```ts
 * import { fromAscii, worldGeometry } from 'whitebox-engine/worlds'
 * worldGeometry(fromAscii(['..', '.#'])).solid.count // => 8
 * ```
 */
export function worldGeometry(w: World): WorldGeometry {
  return {
    world: w,
    solid: buildIndex(segmentsOf(w, (c) => !passable(w, c))),
    opaque: buildIndex(segmentsOf(w, (c) => terrainOf(w, c).opaque)),
  }
}

/** The result of moving a circle. */
export interface Motion {
  /** Where the centre ended. */
  readonly position: [number, number]
  /** Whether a wall stopped or deflected the motion. */
  readonly collided: boolean
  /** The normal of the last wall touched (pointing away from it), if any. */
  readonly normal: [number, number] | null
  /** The distance the centre travelled. */
  readonly travelled: number
  /** The centre's path: the start, each point where it touched a wall, and the end. */
  readonly path: readonly (readonly [number, number])[]
}

const SKIN = 1e-9

/**
 * Move a circle (a disc agent) through a world by a displacement, stopping at the first wall it would touch (swept
 * collision against every solid segment: their lines and endpoints) and, with `slide`, sliding along it for the rest
 * of the move (up to `maxBounces` contacts). The centre never comes within `radius` of a wall, so it never passes
 * through one; a radius of 0 moves a point. Keep the radius below half a cell for corridors one cell wide.
 *
 * @param g - The world geometry.
 * @param from - The centre `[x, y]`.
 * @param displacement - The move `[dx, dy]`.
 * @param radius - The radius (at least 0).
 * @param options - Slide along walls (default true) and the contact cap (default 4).
 * @returns The motion.
 * @throws WhiteboxError with code `value` for a negative radius.
 * @concept collision-detection
 * @example
 * ```ts
 * import { fromAscii, moveCircle, worldGeometry } from 'whitebox-engine/worlds'
 * const g = worldGeometry(fromAscii(['..#']))
 * const m = moveCircle(g, [0.5, 0.5], [3, 0], 0.25)
 * [Math.round(m.position[0] * 1e6) / 1e6, m.collided] // => [1.75, true]
 * Math.abs(moveCircle(g, [0.5, 0.5], [1, 0.5], 0.25).position[1] - 0.75) < 1e-6 // => true
 * ```
 */
export function moveCircle(
  g: WorldGeometry,
  from: readonly [number, number],
  displacement: readonly [number, number],
  radius: number,
  options: { readonly slide?: boolean; readonly maxBounces?: number } = {},
): Motion {
  if (!(radius >= 0)) throw new WhiteboxError('value', 'moveCircle: the radius must be at least 0')
  let p: [number, number] = [from[0], from[1]]
  let rem: [number, number] = [displacement[0], displacement[1]]
  let collided = false
  let normal: [number, number] | null = null
  let travelled = 0
  const route: [number, number][] = [p]
  const bounces = options.maxBounces ?? 4
  for (let bounce = 0; bounce <= bounces; bounce++) {
    const len = Math.hypot(rem[0], rem[1])
    if (len < 1e-15) break
    const q: [number, number] = [p[0] + rem[0], p[1] + rem[1]]
    const pad = radius + SKIN
    let best = Number.POSITIVE_INFINITY
    let n: [number, number] = [0, 0]
    for (const id of inBox(
      g.solid,
      Math.min(p[0], q[0]) - pad,
      Math.min(p[1], q[1]) - pad,
      Math.max(p[0], q[0]) + pad,
      Math.max(p[1], q[1]) + pad,
    )) {
      const [a, b] = segmentOf(g.solid, id)
      const hit = radius > 0 ? sweptCircleHit(p, rem, radius, a, b) : raySegmentHit(p, rem, a, b)
      if (hit.t < best && hit.t <= 1) {
        best = hit.t
        n = hit.normal
      }
    }
    if (best === Number.POSITIVE_INFINITY) {
      p = q
      travelled += len
      route.push(p)
      break
    }
    collided = true
    normal = n
    const t = Math.max(0, best - SKIN / len)
    p = [p[0] + t * rem[0], p[1] + t * rem[1]]
    travelled += t * len
    route.push(p)
    if (options.slide === false) break
    const left: [number, number] = [(1 - t) * rem[0], (1 - t) * rem[1]]
    const dn = left[0] * n[0] + left[1] * n[1]
    rem = dn < 0 ? [left[0] - dn * n[0], left[1] - dn * n[1]] : left
  }
  return { position: p, collided, normal, travelled, path: route }
}

/**
 * Move a disc agent: turn by `turn` radians, then move `forward` along the new heading and `strafe` to its right,
 * colliding and sliding as {@link moveCircle}.
 *
 * @param g - The world geometry.
 * @param pose - The pose.
 * @param control - Forward and sideways distances and the turn.
 * @param radius - The agent's radius (default 0.2).
 * @returns The new pose and whether a wall was touched.
 * @concept collision-detection
 * @example
 * ```ts
 * import { fromAscii, stepPose, worldGeometry } from 'whitebox-engine/worlds'
 * const g = worldGeometry(fromAscii(['....']))
 * const r = stepPose(g, { x: 0.5, y: 0.5, heading: 0 }, { forward: 1 })
 * [r.pose.x, r.collided] // => [1.5, false]
 * ```
 */
export function stepPose(
  g: WorldGeometry,
  pose: Pose,
  control: { readonly forward?: number; readonly strafe?: number; readonly turn?: number },
  radius = 0.2,
): { readonly pose: Pose; readonly collided: boolean } {
  const heading = pose.heading + (control.turn ?? 0)
  const f = control.forward ?? 0
  const s = control.strafe ?? 0
  const c = Math.cos(heading)
  const si = Math.sin(heading)
  const m = moveCircle(g, [pose.x, pose.y], [f * c - s * si, f * si + s * c], radius)
  return { pose: { x: m.position[0], y: m.position[1], heading }, collided: m.collided }
}

/** Where a ray hits a wall. */
export interface RayHit {
  readonly distance: number
  readonly point: [number, number]
  /** The wall's normal, facing the ray. */
  readonly normal: [number, number]
  /** The cell on the far side of the wall (-1 at the map edge). */
  readonly cell: number
}

/**
 * Cast a ray from a point and return the first wall it hits within a distance.
 *
 * @param g - The world geometry.
 * @param origin - The start `[x, y]`.
 * @param angle - The direction in radians (0 east, clockwise on screen).
 * @param options - The maximum distance (default the world's diagonal) and which walls (`'opaque'` for sight, the
 *   default, or `'solid'` for motion).
 * @returns The hit, or null when nothing is hit in range.
 * @example
 * ```ts
 * import { castRay, fromAscii, worldGeometry } from 'whitebox-engine/worlds'
 * const hit = castRay(worldGeometry(fromAscii(['...#'])), [0.5, 0.5], 0)
 * [hit?.distance, hit?.normal, hit?.cell] // => [2.5, [-1, 0], 3]
 * ```
 */
export function castRay(
  g: WorldGeometry,
  origin: readonly [number, number],
  angle: number,
  options: { readonly maxDistance?: number; readonly kind?: 'solid' | 'opaque' } = {},
): RayHit | null {
  const [w, h] = worldSize(g.world)
  const max = options.maxDistance ?? Math.hypot(w, h) + 1
  const ix = options.kind === 'solid' ? g.solid : g.opaque
  const d: [number, number] = [Math.cos(angle), Math.sin(angle)]
  const end: [number, number] = [origin[0] + max * d[0], origin[1] + max * d[1]]
  let best = Number.POSITIVE_INFINITY
  let normal: [number, number] = [0, 0]
  for (const id of alongSegment(ix, origin, end)) {
    const [a, b] = segmentOf(ix, id)
    const hit = raySegmentHit(origin, d, a, b)
    if (hit.t < best) {
      best = hit.t
      normal = hit.normal
    }
  }
  if (!(best <= max)) return null
  const point: [number, number] = [origin[0] + best * d[0], origin[1] + best * d[1]]
  const cell = cellAtPoint(g.world, point[0] - 1e-7 * normal[0], point[1] - 1e-7 * normal[1])
  return { distance: best, point, normal, cell }
}

/**
 * Sweep a point along a segment (a point agent's move, a bullet, a probe): the first solid wall it meets between the
 * two ends, as a fraction `t` of the way and the point and normal of contact.
 *
 * @param g - The world geometry.
 * @param from - The start.
 * @param to - The end.
 * @param options - Which walls (`'solid'`, the default, or `'opaque'`).
 * @returns The first contact, or null when the segment is clear.
 * @concept collision-detection
 * @example
 * ```ts
 * import { fromAscii, sweepSegment, worldGeometry } from 'whitebox-engine/worlds'
 * const g = worldGeometry(fromAscii(['..#']))
 * sweepSegment(g, [0.5, 0.5], [2.5, 0.5]) // => { t: 0.75, point: [2, 0.5], normal: [-1, 0] }
 * sweepSegment(g, [0.5, 0.5], [1.5, 0.5]) // => null
 * ```
 */
export function sweepSegment(
  g: WorldGeometry,
  from: readonly [number, number],
  to: readonly [number, number],
  options: { readonly kind?: 'solid' | 'opaque' } = {},
): { readonly t: number; readonly point: [number, number]; readonly normal: [number, number] } | null {
  const ix = options.kind === 'opaque' ? g.opaque : g.solid
  const d: [number, number] = [to[0] - from[0], to[1] - from[1]]
  let best = Number.POSITIVE_INFINITY
  let normal: [number, number] = [0, 0]
  for (const id of alongSegment(ix, from, to)) {
    const [a, b] = segmentOf(ix, id)
    const hit = raySegmentHit(from, d, a, b)
    if (hit.t <= 1 && hit.t < best) {
      best = hit.t
      normal = hit.normal
    }
  }
  if (best === Number.POSITIVE_INFINITY) return null
  return { t: best, point: [from[0] + best * d[0], from[1] + best * d[1]], normal }
}

/**
 * Whether two points see each other: no opaque segment touches the segment between them (touching a wall's end
 * counts as blocked, so sight never slips through a corner). The test is exactly symmetric.
 *
 * @param g - The world geometry.
 * @param a - A point.
 * @param b - Another point.
 * @returns Whether the line of sight is clear.
 * @example
 * ```ts
 * import { fromAscii, lineOfSight, worldGeometry } from 'whitebox-engine/worlds'
 * const g = worldGeometry(fromAscii(['...', '.#.', '...']))
 * [lineOfSight(g, [0.5, 0.5], [2.5, 0.5]), lineOfSight(g, [0.5, 0.5], [2.5, 2.5])] // => [true, false]
 * ```
 */
export function lineOfSight(g: WorldGeometry, a: readonly [number, number], b: readonly [number, number]): boolean {
  const [p, q] = a[0] < b[0] || (a[0] === b[0] && a[1] <= b[1]) ? [a, b] : [b, a]
  for (const id of alongSegment(g.opaque, p, q)) {
    const [c, d] = segmentOf(g.opaque, id)
    if (segmentsTouch(p, q, c, d)) return false
  }
  return true
}

/**
 * Whether two cells see each other, centre to centre.
 *
 * @param g - The world geometry.
 * @param a - A cell.
 * @param b - Another cell.
 * @returns Whether the line of sight is clear.
 * @example
 * ```ts
 * import { cellLineOfSight, fromAscii, worldGeometry } from 'whitebox-engine/worlds'
 * cellLineOfSight(worldGeometry(fromAscii(['.#.'])), 0, 2) // => false
 * ```
 */
export function cellLineOfSight(g: WorldGeometry, a: number, b: number): boolean {
  return lineOfSight(g, cellCentre(g.world, a), cellCentre(g.world, b))
}

/** What an agent sees. */
export interface FieldOfView {
  /** The visible cells (centres inside the cone, in range and in sight), in order. */
  readonly cells: readonly number[]
  /** The visible region as a polygon: the eye, then the ray hits across the cone. */
  readonly polygon: readonly (readonly [number, number])[]
}

/**
 * The field of view of an agent: a cone about its heading, cut off by range and by opaque walls. The polygon is a
 * fan of rays (for drawing view cones); the cells are those whose centres lie inside the cone and in sight.
 *
 * @param g - The world geometry.
 * @param pose - The eye.
 * @param options - The cone's full angle (default `pi / 2`), the range (default 8) and the rays in the fan (default 32).
 * @returns The visible cells and polygon.
 * @example
 * ```ts
 * import { fieldOfView, fromAscii, worldGeometry } from 'whitebox-engine/worlds'
 * const g = worldGeometry(fromAscii(['....', '.#..']))
 * fieldOfView(g, { x: 0.5, y: 0.5, heading: 0 }, { angle: Math.PI }).cells // => [0, 1, 2, 3, 4]
 * ```
 */
export function fieldOfView(
  g: WorldGeometry,
  pose: Pose,
  options: { readonly angle?: number; readonly range?: number; readonly rays?: number } = {},
): FieldOfView {
  const angle = options.angle ?? Math.PI / 2
  const range = options.range ?? 8
  const rays = Math.max(1, Math.floor(options.rays ?? 32))
  const eye: [number, number] = [pose.x, pose.y]
  const polygon: [number, number][] = [eye]
  for (let i = 0; i <= rays; i++) {
    const a = pose.heading - angle / 2 + (angle * i) / rays
    const hit = castRay(g, eye, a, { maxDistance: range })
    const r = hit?.distance ?? range
    polygon.push([pose.x + r * Math.cos(a), pose.y + r * Math.sin(a)])
  }
  const here = cellAtPoint(g.world, pose.x, pose.y)
  const cells: number[] = []
  for (let c = 0; c < g.world.n; c++) {
    if (c === here) {
      cells.push(c)
      continue
    }
    const [x, y] = cellCentre(g.world, c)
    const dx = x - pose.x
    const dy = y - pose.y
    if (Math.hypot(dx, dy) > range) continue
    const off = Math.atan2(dy, dx) - pose.heading
    const wrapped = Math.abs(Math.atan2(Math.sin(off), Math.cos(off)))
    if (wrapped <= angle / 2 + 1e-12 && lineOfSight(g, eye, [x, y])) cells.push(c)
  }
  return { cells, polygon }
}

/**
 * The exact distance from a point to the nearest solid segment (walls, impassable cells and the map edge).
 *
 * @param g - The world geometry.
 * @param point - The point.
 * @returns The distance (Infinity when the world has no solid segment).
 * @example
 * ```ts
 * import { distanceToWalls, fromAscii, worldGeometry } from 'whitebox-engine/worlds'
 * distanceToWalls(worldGeometry(fromAscii(['...', '...', '...'])), [1.5, 1]) // => 1
 * ```
 */
export function distanceToWalls(g: WorldGeometry, point: readonly [number, number]): number {
  const ix = g.solid
  if (ix.count === 0) return Number.POSITIVE_INFINITY
  let best = Number.POSITIVE_INFINITY
  const reach = Math.max(ix.nx, ix.ny)
  for (let k = 0; k <= reach; k++) {
    for (const id of inBox(ix, point[0] - k, point[1] - k, point[0] + k, point[1] + k)) {
      const [a, b] = segmentOf(ix, id)
      best = Math.min(best, closestPointOnSegment(point, a, b).distance)
    }
    if (best <= k) break
  }
  return best
}

/** 1-D squared Euclidean distance transform of Felzenszwalb and Huttenlocher (lower envelope of parabolas). */
function edt1d(f: Float64Array): Float64Array {
  const n = f.length
  const d = new Float64Array(n)
  const v = new Int32Array(n)
  const z = new Float64Array(n + 1)
  let k = -1
  for (let q = 0; q < n; q++) {
    const fq = f[q] ?? 0
    if (fq === Number.POSITIVE_INFINITY) continue
    let s = Number.NEGATIVE_INFINITY
    while (k >= 0) {
      const p = v[k] ?? 0
      s = (fq + q * q - ((f[p] ?? 0) + p * p)) / (2 * (q - p))
      if (s > (z[k] ?? 0)) break
      k--
    }
    k++
    v[k] = q
    z[k] = k === 0 ? Number.NEGATIVE_INFINITY : s
    z[k + 1] = Number.POSITIVE_INFINITY
  }
  if (k < 0) return d.fill(Number.POSITIVE_INFINITY)
  let j = 0
  for (let q = 0; q < n; q++) {
    while ((z[j + 1] ?? 0) < q) j++
    const p = v[j] ?? 0
    d[q] = (q - p) * (q - p) + (f[p] ?? 0)
  }
  return d
}

/**
 * Distances from walls as a field over cells. On square grids, `'euclidean'` (the exact Euclidean distance
 * transform, scipy's `distance_transform_edt`), `'taxicab'` and `'chessboard'` (scipy's `distance_transform_cdt`)
 * give each cell's distance, centre to centre, to the nearest impassable cell (0 on impassable cells); with `border`,
 * the outside of the map counts as impassable. `'exact'` (any square or hexagonal world) gives the distance from
 * each cell centre to the nearest solid segment, so thin walls and the map edge count.
 *
 * @param w - The world.
 * @param options - The metric (default `'euclidean'`) and whether the border is a wall (default false).
 * @returns The distance of each cell (Infinity when nothing is impassable).
 * @throws WhiteboxError with code `value` for a grid metric on a non-square world.
 * @example
 * ```ts
 * import { fromAscii, wallDistanceField } from 'whitebox-engine/worlds'
 * Array.from(wallDistanceField(fromAscii(['#...']))) // => [0, 1, 2, 3]
 * Array.from(wallDistanceField(fromAscii(['#..', '...']), { metric: 'chessboard' })) // => [0, 1, 2, 1, 1, 2]
 * Array.from(wallDistanceField(fromAscii(['..']), { metric: 'exact' })) // => [0.5, 0.5]
 * ```
 */
export function wallDistanceField(
  w: World,
  options: { readonly metric?: 'euclidean' | 'taxicab' | 'chessboard' | 'exact'; readonly border?: boolean } = {},
): Float64Array {
  const metric = options.metric ?? 'euclidean'
  if (metric === 'exact') {
    const g = worldGeometry(w)
    return Float64Array.from({ length: w.n }, (_, c) => (passable(w, c) ? distanceToWalls(g, cellCentre(w, c)) : 0))
  }
  const t = w.topology
  if (t.kind !== 'square') throw new WhiteboxError('value', `wallDistanceField: '${metric}' needs a square grid`)
  const pad = options.border === true ? 1 : 0
  const rows = t.rows + 2 * pad
  const cols = t.cols + 2 * pad
  const blocked = (r: number, c: number): boolean => {
    const rr = r - pad
    const cc = c - pad
    return rr < 0 || rr >= t.rows || cc < 0 || cc >= t.cols || !passable(w, rr * t.cols + cc)
  }
  const field = new Float64Array(rows * cols)
  if (metric === 'euclidean') {
    for (let i = 0; i < rows * cols; i++)
      field[i] = blocked(Math.floor(i / cols), i % cols) ? 0 : Number.POSITIVE_INFINITY
    for (let c = 0; c < cols; c++) {
      const col = edt1d(Float64Array.from({ length: rows }, (_, r) => field[r * cols + c] ?? 0))
      for (let r = 0; r < rows; r++) field[r * cols + c] = col[r] ?? 0
    }
    for (let r = 0; r < rows; r++) {
      const row = edt1d(field.slice(r * cols, (r + 1) * cols))
      for (let c = 0; c < cols; c++) field[r * cols + c] = Math.sqrt(row[c] ?? 0)
    }
  } else {
    field.fill(Number.POSITIVE_INFINITY)
    const queue: number[] = []
    for (let i = 0; i < rows * cols; i++) {
      if (blocked(Math.floor(i / cols), i % cols)) {
        field[i] = 0
        queue.push(i)
      }
    }
    const k = metric === 'taxicab' ? 4 : 8
    const offsets = [
      [-1, 0],
      [0, 1],
      [1, 0],
      [0, -1],
      [-1, 1],
      [1, 1],
      [1, -1],
      [-1, -1],
    ].slice(0, k)
    for (let q = 0; q < queue.length; q++) {
      const i = queue[q] ?? 0
      const r = Math.floor(i / cols)
      const c = i % cols
      for (const [dr = 0, dc = 0] of offsets) {
        const r2 = r + dr
        const c2 = c + dc
        if (r2 < 0 || r2 >= rows || c2 < 0 || c2 >= cols) continue
        const j = r2 * cols + c2
        if (field[j] !== Number.POSITIVE_INFINITY) continue
        field[j] = (field[i] ?? 0) + 1
        queue.push(j)
      }
    }
  }
  if (pad === 0) return field
  return Float64Array.from(
    { length: w.n },
    (_, i) => field[(Math.floor(i / t.cols) + 1) * cols + (i % t.cols) + 1] ?? 0,
  )
}
