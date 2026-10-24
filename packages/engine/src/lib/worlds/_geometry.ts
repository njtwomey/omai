// A bucket index over wall segments for continuous worlds: box and along-a-line queries. Private to worlds
// (continuous.ts is the public face); the segment predicates themselves are numerics/geometry's segments.ts.

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
