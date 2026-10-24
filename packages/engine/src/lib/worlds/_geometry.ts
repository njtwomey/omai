// Segment geometry for continuous worlds: a bucket index over wall segments, robust touching tests, ray and
// swept-circle contact times, and point-segment distances. Private to worlds (continuous.ts is the public face).
import { orient } from '../numerics/geometry/index.ts'

/** Wall segments `[x1, y1, x2, y2]` packed four numbers each, bucketed by the unit squares their boxes overlap. */
export interface SegmentIndex {
  readonly segments: Float64Array
  readonly count: number
  /** Bucket grid origin and size: bucket `(i, j)` covers `[x0 + i, x0 + i + 1) x [y0 + j, y0 + j + 1)`. */
  readonly x0: number
  readonly y0: number
  readonly nx: number
  readonly ny: number
  /** Compressed rows: the segments of bucket `b` are `ids[start[b] .. start[b + 1]]`. */
  readonly start: Int32Array
  readonly ids: Int32Array
}

const EPS = 1e-9

/** Bucket the segments (each into every unit square its slightly enlarged bounding box touches). */
export function buildIndex(flat: readonly number[]): SegmentIndex {
  const count = flat.length / 4
  const segments = Float64Array.from(flat)
  let minX = 0
  let minY = 0
  let maxX = 1
  let maxY = 1
  for (let i = 0; i < count; i++) {
    for (const k of [0, 2]) {
      minX = Math.min(minX, segments[4 * i + k] ?? 0)
      maxX = Math.max(maxX, segments[4 * i + k] ?? 0)
      minY = Math.min(minY, segments[4 * i + k + 1] ?? 0)
      maxY = Math.max(maxY, segments[4 * i + k + 1] ?? 0)
    }
  }
  const x0 = Math.floor(minX) - 1
  const y0 = Math.floor(minY) - 1
  const nx = Math.ceil(maxX) - x0 + 2
  const ny = Math.ceil(maxY) - y0 + 2
  const lists: number[][] = Array.from({ length: nx * ny }, () => [])
  for (let i = 0; i < count; i++) {
    const [ax, ay, bx, by] = [
      segments[4 * i] ?? 0,
      segments[4 * i + 1] ?? 0,
      segments[4 * i + 2] ?? 0,
      segments[4 * i + 3] ?? 0,
    ]
    const i0 = Math.max(0, Math.floor(Math.min(ax, bx) - EPS) - x0)
    const i1 = Math.min(nx - 1, Math.floor(Math.max(ax, bx) + EPS) - x0)
    const j0 = Math.max(0, Math.floor(Math.min(ay, by) - EPS) - y0)
    const j1 = Math.min(ny - 1, Math.floor(Math.max(ay, by) + EPS) - y0)
    for (let j = j0; j <= j1; j++) for (let k = i0; k <= i1; k++) lists[j * nx + k]?.push(i)
  }
  const start = new Int32Array(nx * ny + 1)
  for (let b = 0; b < nx * ny; b++) start[b + 1] = (start[b] ?? 0) + (lists[b]?.length ?? 0)
  const ids = new Int32Array(start[nx * ny] ?? 0)
  for (let b = 0; b < nx * ny; b++) ids.set(lists[b] ?? [], start[b] ?? 0)
  return { segments, count, x0, y0, nx, ny, start, ids }
}

/** The segments whose buckets overlap a box. */
export function inBox(ix: SegmentIndex, x0: number, y0: number, x1: number, y1: number): number[] {
  const i0 = Math.max(0, Math.floor(Math.min(x0, x1) - EPS) - ix.x0)
  const i1 = Math.min(ix.nx - 1, Math.floor(Math.max(x0, x1) + EPS) - ix.x0)
  const j0 = Math.max(0, Math.floor(Math.min(y0, y1) - EPS) - ix.y0)
  const j1 = Math.min(ix.ny - 1, Math.floor(Math.max(y0, y1) + EPS) - ix.y0)
  const seen = new Set<number>()
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const b = j * ix.nx + i
      for (let k = ix.start[b] ?? 0; k < (ix.start[b + 1] ?? 0); k++) seen.add(ix.ids[k] ?? 0)
    }
  }
  return [...seen]
}

/** The segments in the buckets a segment from `a` to `b` passes through (Amanatides-Woo traversal), in order. */
export function alongSegment(ix: SegmentIndex, a: readonly [number, number], b: readonly [number, number]): number[] {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  let i = Math.floor(a[0]) - ix.x0
  let j = Math.floor(a[1]) - ix.y0
  const iEnd = Math.floor(b[0]) - ix.x0
  const jEnd = Math.floor(b[1]) - ix.y0
  const si = dx > 0 ? 1 : dx < 0 ? -1 : 0
  const sj = dy > 0 ? 1 : dy < 0 ? -1 : 0
  const tdx = si === 0 ? Number.POSITIVE_INFINITY : Math.abs(1 / dx)
  const tdy = sj === 0 ? Number.POSITIVE_INFINITY : Math.abs(1 / dy)
  const fx = a[0] - Math.floor(a[0])
  const fy = a[1] - Math.floor(a[1])
  let tx = si === 0 ? Number.POSITIVE_INFINITY : (si > 0 ? 1 - fx : fx) * tdx
  let ty = sj === 0 ? Number.POSITIVE_INFINITY : (sj > 0 ? 1 - fy : fy) * tdy
  const seen = new Set<number>()
  const limit = Math.abs(iEnd - i) + Math.abs(jEnd - j) + 2
  for (let n = 0; n <= limit; n++) {
    // Each bucket also contributes its neighbours, so segments touching a bucket corner the line grazes are seen.
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        const bi = i + di
        const bj = j + dj
        if (bi < 0 || bj < 0 || bi >= ix.nx || bj >= ix.ny) continue
        const bucket = bj * ix.nx + bi
        for (let k = ix.start[bucket] ?? 0; k < (ix.start[bucket + 1] ?? 0); k++) seen.add(ix.ids[k] ?? 0)
      }
    }
    if (i === iEnd && j === jEnd) break
    if (tx < ty) {
      tx += tdx
      i += si
    } else {
      ty += tdy
      j += sj
    }
  }
  return [...seen]
}

/** Segment `id` of an index as two points. */
export function segmentOf(ix: SegmentIndex, id: number): [[number, number], [number, number]] {
  const s = ix.segments
  return [
    [s[4 * id] ?? 0, s[4 * id + 1] ?? 0],
    [s[4 * id + 2] ?? 0, s[4 * id + 3] ?? 0],
  ]
}

/**
 * Whether closed segments ab and cd share a point (touching and collinear overlap included). Each orientation within a
 * relative 1e-12 of zero (relative to the lengths it multiplies) counts as collinear, so lines through the rounded
 * corners of hexagons are judged consistently.
 */
export function segmentsTouch(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
  d: readonly [number, number],
): boolean {
  const side = (p: readonly [number, number], q: readonly [number, number], r: readonly [number, number]): number => {
    const o = orient(p, q, r)
    const tol = 1e-12 * Math.hypot(q[0] - p[0], q[1] - p[1]) * Math.hypot(r[0] - p[0], r[1] - p[1])
    return o > tol ? 1 : o < -tol ? -1 : 0
  }
  const o1 = side(a, b, c)
  const o2 = side(a, b, d)
  const o3 = side(c, d, a)
  const o4 = side(c, d, b)
  if (o1 * o2 > 0 || o3 * o4 > 0) return false
  if (o1 !== 0 || o2 !== 0 || o3 !== 0 || o4 !== 0) return true
  // Collinear: the projections on both axes must overlap.
  const overlap = (p: number, q: number, r: number, s: number): boolean =>
    Math.max(Math.min(p, q), Math.min(r, s)) <= Math.min(Math.max(p, q), Math.max(r, s)) + 1e-12
  return overlap(a[0], b[0], c[0], d[0]) && overlap(a[1], b[1], c[1], d[1])
}

/** The closest point of segment ab to p, and its distance. */
export function closestOnSegment(
  p: readonly [number, number],
  a: readonly [number, number],
  b: readonly [number, number],
): { readonly point: [number, number]; readonly distance: number } {
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const len2 = ex * ex + ey * ey
  const u = len2 === 0 ? 0 : Math.min(1, Math.max(0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2))
  const point: [number, number] = [a[0] + u * ex, a[1] + u * ey]
  return { point, distance: Math.hypot(p[0] - point[0], p[1] - point[1]) }
}

/** The ray parameter `t >= 0` at which `o + t d` meets segment ab (Infinity when it misses), with the hit normal. */
export function rayHit(
  o: readonly [number, number],
  d: readonly [number, number],
  a: readonly [number, number],
  b: readonly [number, number],
): { readonly t: number; readonly normal: [number, number] } {
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const denom = d[0] * ey - d[1] * ex
  const none = { t: Number.POSITIVE_INFINITY, normal: [0, 0] as [number, number] }
  if (denom === 0) return none
  const ax = a[0] - o[0]
  const ay = a[1] - o[1]
  const t = (ax * ey - ay * ex) / denom
  const u = (ax * d[1] - ay * d[0]) / denom
  if (t < 0 || u < 0 || u > 1) return none
  const len = Math.hypot(ex, ey)
  let nx = -ey / len
  let ny = ex / len
  if (nx * d[0] + ny * d[1] > 0) {
    nx = -nx
    ny = -ny
  }
  return { t, normal: [nx, ny] }
}

/**
 * The first time `t` in `[0, 1]` at which a circle of radius `r` moving from `p` by `d` touches segment ab, with the
 * contact normal (from the wall towards the circle); Infinity when it does not. A circle already touching and moving
 * closer contacts at `t = 0`.
 */
export function circleHit(
  p: readonly [number, number],
  d: readonly [number, number],
  r: number,
  a: readonly [number, number],
  b: readonly [number, number],
): { readonly t: number; readonly normal: [number, number] } {
  let best = Number.POSITIVE_INFINITY
  let normal: [number, number] = [0, 0]
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const len = Math.hypot(ex, ey)
  if (len > 0) {
    let nx = -ey / len
    let ny = ex / len
    let s = (p[0] - a[0]) * nx + (p[1] - a[1]) * ny
    if (s < 0) {
      nx = -nx
      ny = -ny
      s = -s
    }
    const vn = d[0] * nx + d[1] * ny
    if (vn < 0) {
      const t = Math.max(0, (s - r) / -vn)
      if (t <= 1) {
        const cx = p[0] + t * d[0] - a[0]
        const cy = p[1] + t * d[1] - a[1]
        const u = (cx * ex + cy * ey) / (len * len)
        if (u >= 0 && u <= 1 && t < best) {
          best = t
          normal = [nx, ny]
        }
      }
    }
  }
  for (const e of [a, b]) {
    const fx = p[0] - e[0]
    const fy = p[1] - e[1]
    const qa = d[0] * d[0] + d[1] * d[1]
    const qb = 2 * (fx * d[0] + fy * d[1])
    const qc = fx * fx + fy * fy - r * r
    if (qb >= 0 || qa === 0) continue
    let t: number
    if (qc <= 0) t = 0
    else {
      const disc = qb * qb - 4 * qa * qc
      if (disc < 0) continue
      t = (-qb - Math.sqrt(disc)) / (2 * qa)
    }
    if (t <= 1 && t < best) {
      const cx = p[0] + t * d[0] - e[0]
      const cy = p[1] + t * d[1] - e[1]
      const n = Math.hypot(cx, cy)
      best = t
      normal = n > 0 ? [cx / n, cy / n] : [-d[0] / Math.sqrt(qa), -d[1] / Math.sqrt(qa)]
    }
  }
  return { t: best, normal }
}
