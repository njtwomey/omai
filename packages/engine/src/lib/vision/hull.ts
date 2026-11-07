/**
 * Convex hulls of point sets as OpenCV returns them (`cv2.convexHull` by Sklansky's scan over the points sorted by x,
 * with OpenCV's starting vertex, orientation and index order) and the convexity defects of a contour against its hull:
 * for each hull edge, the contour point deepest inside it and its depth.
 */

import { MdmlError, type Vec2 } from '../base/index.ts'
import { f32, type PointsLike, pointsOf, roundHalfEven } from './_image.ts'

type P = Vec2

/** OpenCV's Sklansky scan of one quarter of the hull (from start towards end), into `stack`; returns its size. */
function sklansky(
  pts: readonly P[],
  ptr: readonly number[],
  start: number,
  end0: number,
  stack: number[],
  base: number,
  nsign: number,
  sign2: number,
  float: boolean,
): number {
  const at = (i: number): P => pts[ptr[i] ?? 0] as P
  const incr = end0 > start ? 1 : -1
  let pprev = start
  let pcur = pprev + incr
  let pnext = pcur + incr
  let size = 3
  const s = at(start)
  const e = at(end0)
  if (start === end0 || (s[0] === e[0] && s[1] === e[1])) {
    stack[base] = start
    return 1
  }
  stack[base] = pprev
  stack[base + 1] = pcur
  stack[base + 2] = pnext
  const end = end0 + incr
  while (pnext !== end) {
    const cury = at(pcur)[1]
    const nexty = at(pnext)[1]
    const by = nexty - cury
    if (Math.sign(by) !== nsign) {
      let ax = at(pcur)[0] - at(pprev)[0]
      let ay = cury - at(pprev)[1]
      let bx = at(pnext)[0] - at(pcur)[0]
      let byy = by
      if (float) {
        const na = Math.hypot(ax, ay)
        const nb = Math.hypot(bx, byy)
        if (na > 0) {
          ax = f32(ax / na)
          ay = f32(ay / na)
        }
        if (nb > 0) {
          bx = f32(bx / nb)
          byy = f32(byy / nb)
        }
      }
      const convexity = ay * bx - ax * byy
      if (Math.sign(convexity) === sign2 && (ax !== 0 || ay !== 0)) {
        pprev = pcur
        pcur = pnext
        pnext += incr
        stack[base + size] = pnext
        size++
      } else if (pprev === start) {
        pcur = pnext
        stack[base + 1] = pcur
        pnext += incr
        stack[base + 2] = pnext
      } else {
        stack[base + size - 2] = pnext
        pcur = pprev
        pprev = stack[base + size - 4] ?? 0
        size--
      }
    } else {
      pnext += incr
      stack[base + size - 1] = pnext
    }
  }
  return size - 1
}

/**
 * The convex hull of planar points as OpenCV computes it (cv2.convexHull with returnPoints false): Sklansky's scan of
 * the upper and lower chains of the x-sorted points, as indices into the points, counter-clockwise on the screen by
 * default (clockwise in OpenCV's y-down sense is the option), rotated so the indices run monotonically when they can.
 * Collinear points on the boundary are dropped. The same set of vertices as `convexHull` of `numerics/geometry`
 * (scipy's), which orders them from the lowest-leftmost point; this one keeps OpenCV's order and orientation, which
 * the contour functions of this module (`convexityDefects`) depend on.
 *
 * @param points - Points `[n, 2]`.
 * @param options - clockwise (default false).
 * @returns Hull vertex indices.
 * @concept convex-hull
 * @example
 * ```ts
 * import { convexHull } from 'mdml-engine/vision'
 * convexHull([[0, 0], [4, 0], [2, 1], [4, 4], [0, 4]]) // => [0, 1, 3, 4]
 * ```
 */
export function convexHull(points: PointsLike, options: { readonly clockwise?: boolean } = {}): number[] {
  const pts = pointsOf(points, 'convexHull')
  const total = pts.length
  if (total === 0) return []
  const clockwise = options.clockwise ?? false
  const float = pts.some(([x, y]) => !Number.isInteger(x) || !Number.isInteger(y))
  const ptr = pts
    .map((_, i) => i)
    .sort((a, b) => {
      const pa = pts[a] as P
      const pb = pts[b] as P
      return pa[0] !== pb[0] ? pa[0] - pb[0] : pa[1] !== pb[1] ? pa[1] - pb[1] : a - b
    })
  const at = (i: number): P => pts[ptr[i] ?? 0] as P
  let miny = 0
  let maxy = 0
  for (let i = 1; i < total; i++) {
    const y = at(i)[1]
    if (at(miny)[1] > y) miny = i
    if (at(maxy)[1] < y) maxy = i
  }
  const hull: number[] = []
  if (at(0)[0] === at(total - 1)[0] && at(0)[1] === at(total - 1)[1]) hull.push(0)
  else {
    const stack = new Array<number>(total + 2).fill(0)
    let tlBase = 0
    let tlCount = sklansky(pts, ptr, 0, maxy, stack, 0, -1, 1, float)
    let trBase = tlCount
    let trCount = sklansky(pts, ptr, total - 1, maxy, stack, trBase, -1, -1, float)
    if (!clockwise) {
      ;[tlBase, trBase] = [trBase, tlBase]
      ;[tlCount, trCount] = [trCount, tlCount]
    }
    for (let i = 0; i < tlCount - 1; i++) hull.push(stack[tlBase + i] ?? 0)
    for (let i = trCount - 1; i > 0; i--) hull.push(stack[trBase + i] ?? 0)
    const stopIdx = trCount > 2 ? (stack[trBase + 1] ?? 0) : tlCount > 2 ? (stack[tlBase + tlCount - 2] ?? 0) : -1
    let blBase = 0
    let blCount = sklansky(pts, ptr, 0, miny, stack, 0, 1, -1, float)
    let brBase = blCount
    let brCount = sklansky(pts, ptr, total - 1, miny, stack, brBase, 1, 1, float)
    if (clockwise) {
      ;[blBase, brBase] = [brBase, blBase]
      ;[blCount, brCount] = [brCount, blCount]
    }
    if (stopIdx >= 0) {
      const checkIdx =
        blCount > 2 ? (stack[blBase + 1] ?? 0) : blCount + brCount > 2 ? (stack[brBase + 2 - blCount] ?? 0) : -1
      if (
        checkIdx === stopIdx ||
        (checkIdx >= 0 && at(checkIdx)[0] === at(stopIdx)[0] && at(checkIdx)[1] === at(stopIdx)[1])
      ) {
        // All points on one line: the lower chain mirrors the upper one.
        blCount = Math.min(blCount, 2)
        brCount = Math.min(brCount, 2)
      }
    }
    for (let i = 0; i < blCount - 1; i++) hull.push(stack[blBase + i] ?? 0)
    for (let i = brCount - 1; i > 0; i--) hull.push(stack[brBase + i] ?? 0)
    // Keep the indices monotone where duplicates allow (OpenCV's returnPoints = false pass, which compares the
    // points' original positions).
    const nout = hull.length
    const orig = (s: number): number => ptr[s] ?? 0
    for (let i = 0; i < nout; i++) {
      const prev = orig(hull[(i === 0 ? nout : i) - 1] ?? 0)
      const next = orig(hull[(i + 1) % nout] ?? 0)
      const cur = hull[i] ?? 0
      const oc = orig(cur)
      if ((prev < oc && oc < next) || (prev > oc && oc > next)) continue
      for (let j = cur + 1; j < total; j++) {
        if (at(cur)[0] === at(j)[0] && at(cur)[1] === at(j)[1]) {
          const oj = orig(j)
          if ((prev < oj && oj < next) || (prev > oj && oj > next)) {
            hull[i] = j
            break
          }
        } else break
      }
    }
  }
  const out = hull.map((i) => ptr[i] ?? 0)
  // Rotate so the indices form an ascending or descending run when they can.
  const nout = out.length
  if (nout >= 3) {
    let minIdx = 0
    let maxIdx = 0
    let lt = 0
    for (let i = 1; i < nout; i++) {
      const idx = out[i] ?? 0
      if ((out[i - 1] ?? 0) < idx) lt++
      if (lt > 1 && lt <= i - 2) break
      if (idx < (out[minIdx] ?? 0)) minIdx = i
      if (idx > (out[maxIdx] ?? 0)) maxIdx = i
    }
    const mmdist = Math.abs(maxIdx - minIdx)
    if ((mmdist === 1 || mmdist === nout - 1) && (lt <= 1 || lt >= nout - 2)) {
      const ascending = (maxIdx + 1) % nout === minIdx
      const i0 = ascending ? minIdx : maxIdx
      if (i0 > 0) {
        const rotated: number[] = []
        let j = i0
        let i = 0
        for (; i < nout; i++) {
          const curr = out[j] ?? 0
          rotated.push(curr)
          const nj = j + 1 < nout ? j + 1 : 0
          if (i < nout - 1 && ascending !== curr < (out[nj] ?? 0)) break
          j = nj
        }
        if (i === nout) return rotated
      }
    }
  }
  return out
}

/** A convexity defect: the hull edge (start, end), the deepest contour point between them and its depth. */
export interface ConvexityDefect {
  readonly start: number
  readonly end: number
  readonly farthest: number
  /** The distance of the farthest point from the hull edge. */
  readonly depth: number
}

/**
 * The convexity defects of a contour (cv2.convexityDefects): for each edge of its hull (given as monotone indices,
 * from {@link convexHull}), the contour point between the edge's ends farthest from the edge, when any lies off it;
 * OpenCV reports the depth in 1/256 pixel, rounded, and so does this.
 *
 * @param contour - Contour points `[n, 2]` (more than three).
 * @param hull - Hull indices into the contour.
 * @returns The defects.
 * @throws MdmlError with code `value` when the hull indices are not monotone (a self-intersecting contour).
 * @concept convexity
 * @example
 * ```ts
 * import { convexHull, convexityDefects } from 'mdml-engine/vision'
 * const c = [[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]]
 * convexityDefects(c, convexHull(c)) // => [{ start: 2, end: 4, farthest: 3, depth: 3 }]
 * ```
 */
export function convexityDefects(contour: PointsLike, hull: readonly number[]): ConvexityDefect[] {
  const pts = pointsOf(contour, 'convexityDefects')
  const n = pts.length
  const h = hull.length
  if (n <= 3 || h < 3) return []
  const reversed =
    ((hull[1] ?? 0) > (hull[0] ?? 0) ? 1 : 0) +
      ((hull[2] ?? 0) > (hull[1] ?? 0) ? 1 : 0) +
      ((hull[0] ?? 0) > (hull[2] ?? 0) ? 1 : 0) !==
    2
  let hcurr = hull[reversed ? 0 : h - 1] ?? 0
  let increasing = -1
  const out: ConvexityDefect[] = []
  for (let i = 0; i < h; i++) {
    const hnext = hull[reversed ? h - i - 1 : i] ?? 0
    if (hcurr < 0 || hcurr >= n || hnext < 0 || hnext >= n)
      throw new MdmlError('value', 'convexityDefects: hull index out of range')
    const p0 = pts[hcurr] as P
    const p1 = pts[hnext] as P
    // The first edge closes the loop (last index to first), so it runs against the others' direction.
    if (increasing < 0) increasing = hcurr < hnext ? 0 : 1
    else if (increasing !== (hcurr < hnext ? 1 : 0))
      throw new MdmlError('value', 'convexityDefects: the hull indices are not monotone')
    const dx0 = p1[0] - p0[0]
    const dy0 = p1[1] - p0[1]
    const scale = dx0 === 0 && dy0 === 0 ? 0 : 1 / Math.sqrt(dx0 * dx0 + dy0 * dy0)
    let deepest = -1
    let depth = 0
    for (let j = (hcurr + 1) % n; j !== hnext; j = (j + 1) % n) {
      const q = pts[j] as P
      const d = Math.abs(-dy0 * (q[0] - p0[0]) + dx0 * (q[1] - p0[1])) * scale
      if (d > depth) {
        depth = d
        deepest = j
      }
    }
    if (deepest >= 0) out.push({ start: hcurr, end: hnext, farthest: deepest, depth: roundHalfEven(depth * 256) / 256 })
    hcurr = hnext
  }
  return out
}
