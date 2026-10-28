/**
 * Contours of binary images, as OpenCV finds and measures them: `findContours` (Suzuki and Abe's border following,
 * with OpenCV's starting points, directions, retrieval modes and hierarchy), polygon simplification by
 * Douglas-Peucker (`approxPolyDP`), perimeter (`arcLength`), area (`contourArea`), the upright bounding rectangle,
 * the convexity test and the signed distance of a point to a contour (`pointPolygonTest`). Points are (x, y) pairs.
 */

// TODO: the Teh-Chin chain approximations (CHAIN_APPROX_TC89_L1 and KCOS) and RETR_FLOODFILL.

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { f32, greyOf, type PointsLike, pointsOf } from './_image.ts'

/** A contour or polygon: its vertices as (x, y) pairs. */
export type Contour = [number, number][]

/** Which contours {@link findContours} retrieves (cv2.RETR_*). */
export type RetrievalMode = 'external' | 'list' | 'ccomp' | 'tree'

/** How {@link findContours} stores a contour (cv2.CHAIN_APPROX_*): every border pixel, or segment end points. */
export type ChainApproximation = 'none' | 'simple'

/** The contours of an image and their hierarchy: per contour [next, previous, first child, parent] (-1 for none). */
export interface ContourTree {
  readonly contours: Contour[]
  readonly hierarchy: [number, number, number, number][]
}

// Chain-code directions: 0 east, then counter-clockwise on the screen (y down) in steps of 45 degrees.
const DX = [1, 1, 0, -1, -1, -1, 0, 1]
const DY = [0, -1, -1, -1, 0, 1, 1, 1]
const RIGHT = 0x80
const NEW = 2

interface Node {
  readonly hole: boolean
  readonly points: Contour
  /** Bounding rectangle in padded coordinates (x, y, width, height). */
  readonly rect: [number, number, number, number]
  /** The start pixel (padded coordinates) for re-tracing. */
  readonly origin: [number, number]
  parent: number
  firstChild: number
  prev: number
  next: number
  tableNext: number
}

/**
 * Suzuki and Abe's border following as OpenCV implements it (contours_new.cpp): a raster scan over the image padded
 * by a zero frame finds outer borders (a 0 to 1 step) and hole borders (a 1 to 0 step), each traced clockwise around
 * its pixels and marked with its label, the right-edge flag recording where the trace leaves on the right.
 */
class Scanner {
  private readonly img: Int8Array
  private readonly step: number
  private readonly height: number
  private readonly simple: boolean
  readonly nodes: Node[]
  private readonly table = new Int32Array(128).fill(-1)
  private nbd = 2
  private x = 1
  private y = 1
  private lnbd: [number, number] = [0, 1]

  constructor(
    data: Float64Array,
    h: number,
    w: number,
    private readonly mode: RetrievalMode,
    private readonly direct: boolean,
  ) {
    this.step = w + 2
    this.height = h + 2
    this.img = new Int8Array(this.step * this.height)
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) if ((data[r * w + c] ?? 0) !== 0) this.img[(r + 1) * this.step + c + 1] = 1
    this.simple = mode === 'external' || mode === 'list'
    this.nodes = [
      {
        hole: true,
        points: [],
        rect: [0, 0, w, h],
        origin: [0, 0],
        parent: -1,
        firstChild: -1,
        prev: -1,
        next: -1,
        tableNext: -1,
      },
    ]
  }

  private at(x: number, y: number): number {
    return this.img[y * this.step + x] ?? 0
  }

  private delta(s: number): number {
    return (DX[s & 7] ?? 0) + (DY[s & 7] ?? 0) * this.step
  }

  /** Trace a border from its start pixel, marking it and collecting its points (icvFetchContourEx). */
  private fetch(start: number, nbd: number, hole: boolean, origin: [number, number]): Node {
    const img = this.img
    const i0 = start
    let pt: [number, number] = [origin[0], origin[1]]
    const points: Contour = []
    // rect as OpenCV keeps it during the trace: x, y minima and x, y maxima.
    const rect: [number, number, number, number] = [pt[0], pt[1], pt[0], pt[1]]
    let sEnd = hole ? 0 : 4
    let s = sEnd
    let i1 = 0
    do {
      s = (s - 1) & 7
      i1 = i0 + this.delta(s)
    } while ((img[i1] ?? 0) === 0 && s !== sEnd)
    if (s === sEnd) {
      img[i0] = nbd | RIGHT
      points.push([pt[0], pt[1]])
    } else {
      let i3 = i0
      let i4 = 0
      let prevS = s ^ 4
      for (;;) {
        sEnd = s
        s = Math.min(s, 15)
        while (s < 15) {
          s++
          i4 = i3 + this.delta(s)
          if ((img[i4] ?? 0) !== 0) break
        }
        s &= 7
        if (s - 1 >= 0 && s - 1 < sEnd) img[i3] = nbd | RIGHT
        else if ((img[i3] ?? 0) === 1) img[i3] = nbd
        if (s !== prevS || this.direct) points.push([pt[0], pt[1]])
        if (s !== prevS) {
          if (pt[0] < rect[0]) rect[0] = pt[0]
          else if (pt[0] > rect[2]) rect[2] = pt[0]
          if (pt[1] < rect[1]) rect[1] = pt[1]
          else if (pt[1] > rect[3]) rect[3] = pt[1]
        }
        prevS = s
        pt = [pt[0] + (DX[s] ?? 0), pt[1] + (DY[s] ?? 0)]
        if (i4 === i0 && i3 === i1) break
        i3 = i4
        s = (s + 4) & 7
      }
    }
    rect[2] -= rect[0] - 1
    rect[3] -= rect[1] - 1
    return {
      hole,
      points,
      rect,
      origin: [0, 0],
      parent: -1,
      firstChild: -1,
      prev: -1,
      next: -1,
      tableNext: -1,
    }
  }

  /** Whether the border from `origin` passes through `end` (icvTraceContour), without marking. */
  private trace(origin: [number, number], end: [number, number], hole: boolean): boolean {
    const img = this.img
    const stop = end[1] * this.step + end[0]
    const i0 = origin[1] * this.step + origin[0]
    const sEnd = hole ? 0 : 4
    let s = sEnd
    let i1 = 0
    do {
      s = (s - 1) & 7
      i1 = i0 + this.delta(s)
    } while ((img[i1] ?? 0) === 0 && s !== sEnd)
    let i3 = i0
    if (s === sEnd) return i3 === stop
    let i4 = 0
    for (;;) {
      s = Math.min(s, 15)
      while (s < 15) {
        s++
        i4 = i3 + this.delta(s)
        if ((img[i4] ?? 0) !== 0) break
      }
      if (i3 === stop) {
        if (((img[i3] ?? 0) & RIGHT) === 0) return true
        let t = s
        for (;;) {
          t = (t - 1) & 7
          if ((img[i3 + this.delta(t)] ?? 0) !== 0) break
          if (t === 0) return true
        }
      }
      if (i4 === i0 && i3 === i1) break
      i3 = i4
      s = (s + 4) & 7
    }
    return false
  }

  private boundingContour(last: [number, number], y: number, lval: number): number {
    const end: [number, number] = [last[0], y]
    let res = -1
    let cur = this.table[lval] ?? -1
    while (cur !== -1) {
      const e = this.nodes[cur] as Node
      if (last[0] - e.rect[0] < e.rect[2] && last[1] - e.rect[1] < e.rect[3]) {
        if (res !== -1) {
          const r = this.nodes[res] as Node
          if (this.trace(r.origin, end, r.hole)) break
        }
        res = cur
      }
      cur = e.tableNext
    }
    return res
  }

  private addChild(parent: number, child: number): void {
    const p = this.nodes[parent] as Node
    const c = this.nodes[child] as Node
    if (p.firstChild !== -1) {
      ;(this.nodes[p.firstChild] as Node).prev = child
      c.next = p.firstChild
    }
    p.firstChild = child
    c.parent = parent
    c.prev = -1
  }

  /** contourScan: start a contour at (x, y) if the step prev -> p begins a border the mode retrieves. */
  private scan(prev: number, p: number, last: [number, number], x: number, y: number): boolean {
    let hole = false
    if (!(prev === 0 && p === 1)) {
      if (p !== 0 || prev < 1) return false
      if ((prev & 0xfe) !== 0) last[0] = x - 1
      hole = true
    }
    if (this.mode === 'external' && (hole || this.at(last[0], last[1]) > 0)) return false
    let parent = -1
    if (this.simple || (!hole && this.mode === 'ccomp') || last[0] <= 0) parent = 0
    else {
      const lval = this.at(last[0], last[1]) & 0x7f
      parent = this.boundingContour(last, y, lval)
      const pe = this.nodes[parent] as Node | undefined
      if (pe === undefined) throw new MdmlError('value', 'findContours: inconsistent contour labels')
      if (pe.hole === hole) parent = pe.parent !== -1 ? pe.parent : 0
    }
    last[0] = x - (hole ? 1 : 0)
    const sx = x - (hole ? 1 : 0)
    const origin: [number, number] = [sx - 1, y - 1]
    let node: Node
    if (this.simple) node = this.fetch(y * this.step + sx, NEW, hole, origin)
    else {
      const lval = this.nbd
      this.nbd = (this.nbd + 1) & 0x7f
      if (this.nbd === 0) this.nbd = 3
      node = this.fetch(y * this.step + sx, lval, hole, origin)
      node.rect[0] += 1
      node.rect[1] += 1
      node.tableNext = this.table[lval] ?? -1
      this.table[lval] = this.nodes.length
    }
    const withOrigin: Node = { ...node, origin: [sx, y] }
    this.nodes.push(withOrigin)
    this.addChild(parent, this.nodes.length - 1)
    this.x = x + 1
    this.y = y
    return true
  }

  /** findNext: continue the raster scan to the next contour; false at the end of the image. */
  next(): boolean {
    let x = this.x
    let y = this.y
    const width = this.step - 1
    const height = this.height - 1
    const last: [number, number] = [this.lnbd[0], this.lnbd[1]]
    let prev = this.at(x - 1, y)
    for (; y < height; y++) {
      let p = 0
      for (; x < width; x++) {
        // Skip the run of pixels equal to the previous one (OpenCV's findNextX).
        while (x < width) {
          p = this.at(x, y)
          if (p !== prev) break
          x++
        }
        if (x >= width) break
        if (this.scan(prev, p, last, x, y)) {
          this.lnbd = [last[0], last[1]]
          return true
        }
        prev = p
        if ((prev & 0xfe) !== 0) last[0] = x
      }
      last[0] = 0
      last[1] = y + 1
      x = 1
      prev = 0
    }
    return false
  }
}

/**
 * Find the contours of a binary image (cv2.findContours, Suzuki and Abe 1985): border following over the image framed
 * by zeros, with OpenCV's conventions (outer borders traced from their top-left pixel, holes from the pixel left of
 * the first background step, contours listed depth first with the latest-found sibling first). 'external' keeps
 * outermost borders only, 'list' all without nesting, 'ccomp' outer borders and their holes in two levels, 'tree'
 * the full nesting.
 *
 * @param x - Binary image `[H, W]` (nonzero is foreground).
 * @param options - Retrieval mode (default 'tree') and approximation (default 'simple').
 * @returns The contours (points (x, y)) and the hierarchy [next, previous, first child, parent].
 * @concept contours
 * @example
 * ```ts
 * import { findContours } from 'mdml-engine/vision'
 * findContours([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).contours // => [[[1, 1], [1, 2], [2, 2], [2, 1]]]
 * ```
 */
export function findContours(
  x: TensorLike,
  options: { readonly mode?: RetrievalMode; readonly method?: ChainApproximation } = {},
): ContourTree {
  const r = greyOf(x, 'findContours')
  const mode = options.mode ?? 'tree'
  if (!['external', 'list', 'ccomp', 'tree'].includes(mode))
    throw new MdmlError('value', `findContours: unknown mode ${String(mode)}`)
  const scanner = new Scanner(r.data, r.h, r.w, mode, (options.method ?? 'simple') === 'none')
  while (scanner.next()) {
    // Each step finds and traces one contour.
  }
  const nodes = scanner.nodes
  const order: number[] = []
  const stack = [0]
  while (stack.length > 0) {
    const i = stack.pop() as number
    order.push(i)
    // Push the children from the last sibling back, so the first child is visited first.
    const kids: number[] = []
    for (let c = nodes[i]?.firstChild ?? -1; c !== -1; c = nodes[c]?.next ?? -1) kids.push(c)
    for (let k = kids.length - 1; k >= 0; k--) stack.push(kids[k] as number)
  }
  const index = new Map<number, number>()
  const kept = order.filter((i) => i !== 0)
  kept.forEach((i, k) => {
    index.set(i, k)
  })
  const map = (i: number): number => index.get(i) ?? -1
  return {
    contours: kept.map((i) => (nodes[i] as Node).points),
    hierarchy: kept.map((i) => {
      const n = nodes[i] as Node
      return [map(n.next), map(n.prev), map(n.firstChild), map(n.parent)]
    }),
  }
}

/**
 * Simplify a curve by the Ramer-Douglas-Peucker algorithm as OpenCV runs it (cv2.approxPolyDP): for a closed curve
 * the recursion starts from two roughly farthest points, segments are split at the point farthest from their chord
 * until every point lies within epsilon, and a final pass drops vertices that sit on a straight line.
 *
 * @param curve - Points `[n, 2]`.
 * @param epsilon - The largest allowed distance from the simplified curve.
 * @param closed - Whether the curve is closed.
 * @returns The kept vertices.
 * @concept polygon-simplification
 * @example
 * ```ts
 * import { approxPolyDP } from 'mdml-engine/vision'
 * approxPolyDP([[0, 0], [1, 0.1], [2, 0], [2, 2], [0, 2]], 0.5, true) // => [[0, 0], [2, 0], [2, 2], [0, 2]]
 * ```
 */
export function approxPolyDP(curve: PointsLike, epsilon: number, closed: boolean): Contour {
  const src = pointsOf(curve, 'approxPolyDP')
  const count0 = src.length
  if (count0 === 0) return []
  const eps2 = epsilon * epsilon
  const dst: Contour = []
  const stack: [number, number][] = []
  let count = count0
  let isClosed = closed
  let initIters = 3
  let startPt: readonly [number, number] = [-1e6, -1e6]
  let pos = 0
  const read = (): readonly [number, number] => {
    const p = src[pos] as readonly [number, number]
    pos++
    if (pos >= count) pos = 0
    return p
  }
  let rightStart = 0
  if (!isClosed) {
    rightStart = count
    const end = src[0] as readonly [number, number]
    startPt = src[count - 1] as readonly [number, number]
    if (startPt[0] !== end[0] || startPt[1] !== end[1]) stack.push([0, count - 1])
    else {
      isClosed = true
      initIters = 1
    }
  }
  if (isClosed) {
    rightStart = 0
    let leEps = false
    for (let i = 0; i < initIters; i++) {
      let maxDist = 0
      pos = (pos + rightStart) % count
      startPt = read()
      for (let j = 1; j < count; j++) {
        const p = read()
        const d = (p[0] - startPt[0]) ** 2 + (p[1] - startPt[1]) ** 2
        if (d > maxDist) {
          maxDist = d
          rightStart = j
        }
      }
      leEps = maxDist <= eps2
    }
    if (!leEps) {
      const sliceStart = pos % count
      const sliceEnd = (rightStart + sliceStart) % count
      stack.push([sliceEnd, sliceStart], [sliceStart, sliceEnd])
    } else dst.push([startPt[0], startPt[1]])
  }
  while (stack.length > 0) {
    const [s0, s1] = stack.pop() as [number, number]
    const endPt = src[s1] as readonly [number, number]
    pos = s0
    startPt = read()
    let leEps: boolean
    let split = 0
    if (pos !== s1) {
      const dx = endPt[0] - startPt[0]
      const dy = endPt[1] - startPt[1]
      const len2 = dx * dx + dy * dy
      let maxD = 0
      while (pos !== s1) {
        const p = read()
        const proj = (p[0] - startPt[0]) * dx + (p[1] - startPt[1]) * dy
        let d: number
        if (proj < 0) d = ((p[0] - startPt[0]) ** 2 + (p[1] - startPt[1]) ** 2) * len2
        else if (proj > len2) d = ((p[0] - endPt[0]) ** 2 + (p[1] - endPt[1]) ** 2) * len2
        else {
          const c = (p[1] - startPt[1]) * dx - (p[0] - startPt[0]) * dy
          d = c * c
        }
        if (d > maxD) {
          maxD = d
          split = (pos + count - 1) % count
        }
      }
      leEps = maxD <= eps2 * len2
    } else {
      leEps = true
      startPt = src[s0] as readonly [number, number]
    }
    if (leEps) dst.push([startPt[0], startPt[1]])
    else stack.push([split, s1], [s0, split])
  }
  if (!isClosed) {
    const last = src[count - 1] as readonly [number, number]
    dst.push([last[0], last[1]])
  }
  // Final pass: remove points on (almost) straight lines.
  isClosed = closed
  count = dst.length
  let newCount = count
  pos = isClosed ? count - 1 : 0
  const readDst = (): [number, number] => {
    const p = dst[pos] as [number, number]
    pos++
    if (pos >= count) pos = 0
    return p
  }
  let start = readDst()
  let wpos = pos
  let pt = readDst()
  for (let i = isClosed ? 0 : 1; i < count - (isClosed ? 0 : 1) && newCount > 2; i++) {
    const end = readDst()
    const dx = end[0] - start[0]
    const dy = end[1] - start[1]
    const dist = Math.abs((pt[0] - start[0]) * dy - (pt[1] - start[1]) * dx)
    const inner = (pt[0] - start[0]) * (end[0] - pt[0]) + (pt[1] - start[1]) * (end[1] - pt[1])
    if (dist * dist <= 0.5 * eps2 * (dx * dx + dy * dy) && dx !== 0 && dy !== 0 && inner >= 0) {
      newCount--
      dst[wpos] = end
      start = end
      wpos++
      if (wpos >= count) wpos = 0
      pt = readDst()
      i++
      continue
    }
    dst[wpos] = pt
    start = pt
    wpos++
    if (wpos >= count) wpos = 0
    pt = end
  }
  if (!isClosed) dst[wpos] = pt
  return dst.slice(0, newCount)
}

/**
 * The length of a curve (cv2.arcLength): the sum of its segment lengths, closing it back to the first point when
 * closed (segment components in single precision, as OpenCV).
 *
 * @param curve - Points `[n, 2]`.
 * @param closed - Whether the curve is closed.
 * @returns The length.
 * @concept contours
 * @example
 * ```ts
 * import { arcLength } from 'mdml-engine/vision'
 * arcLength([[0, 0], [3, 0], [3, 4]], true) // => 12
 * ```
 */
export function arcLength(curve: PointsLike, closed: boolean): number {
  const p = pointsOf(curve, 'arcLength')
  if (p.length <= 1) return 0
  let prev = p[closed ? p.length - 1 : 0] as readonly [number, number]
  let total = 0
  for (const q of p) {
    const dx = f32(f32(q[0]) - f32(prev[0]))
    const dy = f32(f32(q[1]) - f32(prev[1]))
    total += f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy))))
    prev = q
  }
  return total
}

/**
 * The area of a polygon by the shoelace formula (cv2.contourArea); with `oriented`, the signed area, positive for
 * vertices running clockwise on the screen (y down), as the hole borders of {@link findContours} do (its outer borders
 * come out negative).
 *
 * @param contour - Vertices `[n, 2]`.
 * @param oriented - Return the signed area (default false).
 * @returns The area.
 * @concept contours
 * @example
 * ```ts
 * import { contourArea } from 'mdml-engine/vision'
 * contourArea([[0, 0], [4, 0], [4, 3], [0, 3]]) // => 12
 * contourArea([[0, 0], [4, 0], [4, 3], [0, 3]], true) // => 12
 * ```
 */
export function contourArea(contour: PointsLike, oriented = false): number {
  const p = pointsOf(contour, 'contourArea')
  if (p.length === 0) return 0
  let prev = p[p.length - 1] as readonly [number, number]
  let a = 0
  for (const q of p) {
    a += f32(prev[0]) * f32(q[1]) - f32(prev[1]) * f32(q[0])
    prev = q
  }
  a *= 0.5
  return oriented ? a : Math.abs(a)
}

/**
 * The upright bounding rectangle of points (cv2.boundingRect): [x, y, width, height] over the floored extremes, both
 * ends inclusive (so a single point has size 1 x 1).
 *
 * @param points - Points `[n, 2]`.
 * @returns The rectangle [x, y, width, height].
 * @concept bounding-box
 * @example
 * ```ts
 * import { boundingRect } from 'mdml-engine/vision'
 * boundingRect([[1, 2], [4, 3], [2, 6]]) // => [1, 2, 4, 5]
 * ```
 */
export function boundingRect(points: PointsLike): [number, number, number, number] {
  const p = pointsOf(points, 'boundingRect')
  if (p.length === 0) return [0, 0, 0, 0]
  let x0 = Number.POSITIVE_INFINITY
  let y0 = Number.POSITIVE_INFINITY
  let x1 = Number.NEGATIVE_INFINITY
  let y1 = Number.NEGATIVE_INFINITY
  for (const [x, y] of p) {
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  }
  const fx0 = Math.floor(x0)
  const fy0 = Math.floor(y0)
  return [fx0, fy0, Math.floor(x1) - fx0 + 1, Math.floor(y1) - fy0 + 1]
}

/**
 * Whether a polygon is convex (cv2.isContourConvex): every turn the same way, with no self-intersection of its turns.
 *
 * @param contour - Vertices `[n, 2]`.
 * @returns True for a convex polygon.
 * @concept convexity
 * @example
 * ```ts
 * import { isContourConvex } from 'mdml-engine/vision'
 * isContourConvex([[0, 0], [2, 0], [2, 2], [0, 2]]) // => true
 * isContourConvex([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]]) // => false
 * ```
 */
export function isContourConvex(contour: PointsLike): boolean {
  const p = pointsOf(contour, 'isContourConvex')
  const n = p.length
  if (n < 3) return false
  // OpenCV's test: every turn has the same strict sign (a straight or reversed step fails).
  let prev = p[(n - 2 + n) % n] as readonly [number, number]
  let cur = p[n - 1] as readonly [number, number]
  let dx0 = cur[0] - prev[0]
  let dy0 = cur[1] - prev[1]
  let orientation = 0
  for (let i = 0; i < n; i++) {
    prev = cur
    cur = p[i] as readonly [number, number]
    const dx = cur[0] - prev[0]
    const dy = cur[1] - prev[1]
    const dxdy0 = dx * dy0
    const dydx0 = dy * dx0
    orientation |= dydx0 > dxdy0 ? 1 : dydx0 < dxdy0 ? 2 : 3
    if (orientation === 3) return false
    dx0 = dx
    dy0 = dy
  }
  return true
}

/**
 * The position of a point relative to a contour (cv2.pointPolygonTest): +1 inside, -1 outside, 0 on an edge (by the
 * crossing rule), or with measureDist the signed distance to the nearest edge (positive inside).
 *
 * @param contour - Vertices `[n, 2]`.
 * @param point - The point (x, y).
 * @param measureDist - Return the signed distance (default false).
 * @returns The classification or the signed distance.
 * @concept point-in-polygon
 * @example
 * ```ts
 * import { pointPolygonTest } from 'mdml-engine/vision'
 * const square = [[0, 0], [4, 0], [4, 4], [0, 4]]
 * pointPolygonTest(square, [1, 2]) // => 1
 * pointPolygonTest(square, [1, 2], true) // => 1
 * pointPolygonTest(square, [6, 2], true) // => -2
 * ```
 */
export function pointPolygonTest(contour: PointsLike, point: readonly [number, number], measureDist = false): number {
  const p = pointsOf(contour, 'pointPolygonTest')
  const total = p.length
  if (total === 0) return measureDist ? -Number.MAX_VALUE : -1
  const [px, py] = point
  let counter = 0
  let v = p[total - 1] as readonly [number, number]
  if (!measureDist) {
    for (let i = 0; i < total; i++) {
      const v0 = v
      v = p[i] as readonly [number, number]
      if ((v0[1] <= py && v[1] <= py) || (v0[1] > py && v[1] > py) || (v0[0] < px && v[0] < px)) {
        if (
          py === v[1] &&
          (px === v[0] || (py === v0[1] && ((v0[0] <= px && px <= v[0]) || (v[0] <= px && px <= v0[0]))))
        )
          return 0
        continue
      }
      let dist = (py - v0[1]) * (v[0] - v0[0]) - (px - v0[0]) * (v[1] - v0[1])
      if (dist === 0) return 0
      if (v[1] < v0[1]) dist = -dist
      if (dist > 0) counter++
    }
    return counter % 2 === 0 ? -1 : 1
  }
  let minNum = Number.MAX_VALUE
  let minDen = 1
  for (let i = 0; i < total; i++) {
    const v0 = v
    v = p[i] as readonly [number, number]
    const dx = v[0] - v0[0]
    const dy = v[1] - v0[1]
    const dx1 = px - v0[0]
    const dy1 = py - v0[1]
    const dx2 = px - v[0]
    const dy2 = py - v[1]
    let num: number
    let den = 1
    if (dx1 * dx + dy1 * dy <= 0) num = dx1 * dx1 + dy1 * dy1
    else if (dx2 * dx + dy2 * dy >= 0) num = dx2 * dx2 + dy2 * dy2
    else {
      num = (dy1 * dx - dx1 * dy) ** 2
      den = dx * dx + dy * dy
    }
    if (num * minDen < minNum * den) {
      minNum = num
      minDen = den
      if (minNum === 0) break
    }
    if ((v0[1] <= py && v[1] <= py) || (v0[1] > py && v[1] > py) || (v0[0] < px && v[0] < px)) continue
    let cross = dy1 * dx - dx1 * dy
    if (dy < 0) cross = -cross
    if (cross > 0) counter++
  }
  const d = Math.sqrt(minNum / minDen)
  return counter % 2 === 0 ? -d : d
}
