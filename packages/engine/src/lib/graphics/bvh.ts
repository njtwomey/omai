/**
 * Bounding-volume hierarchies: a tree of nested boxes over a scene's shapes, so a ray tests only the few shapes whose
 * boxes it enters instead of every one. Construction by the binned surface-area heuristic, nearest-hit and any-hit
 * (shadow) traversal with statistics of the work done, and the hierarchy as a drawable tree.
 */

import { WhiteboxError } from '../base/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure, type Vec3 } from '../scene/index.ts'
import type { Ray } from './camera.ts'
import { bounds, type Hit, intersect, RAY_EPSILON, type Shape, slabs } from './intersect.ts'

/**
 * A node of a bounding-volume hierarchy: its box, and either two children (`left`, `right`) or, for a leaf, the range
 * `[start, start + count)` of the hierarchy's `order` listing its shapes.
 *
 * @concept bounding-volume-hierarchy
 */
export interface BvhNode {
  readonly min: Vec3
  readonly max: Vec3
  /** Child node indices (-1 in leaves). */
  readonly left: number
  readonly right: number
  /** Leaf contents in `order` (count 0 for interior nodes). */
  readonly start: number
  readonly count: number
  readonly depth: number
}

/** A bounding-volume hierarchy over shapes: node 0 is the root, and `order` lists shape indices leaf by leaf. */
export interface Bvh {
  readonly shapes: readonly Shape[]
  readonly nodes: readonly BvhNode[]
  readonly order: readonly number[]
}

/** Options of {@link buildBvh}. */
export interface BvhOptions {
  /** The most shapes a leaf may hold when a split is possible (default 2). */
  readonly maxLeaf?: number
  /** `'sah'` (default): the binned surface-area heuristic; `'median'`: split at the median centroid. */
  readonly split?: 'sah' | 'median'
  /** Bins of the surface-area heuristic (default 12). */
  readonly bins?: number
}

interface Box {
  min: [number, number, number]
  max: [number, number, number]
}

const emptyBox = (): Box => ({
  min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
  max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
})

function grow(b: Box, min: Vec3, max: Vec3): void {
  for (let k = 0; k < 3; k++) {
    if ((min[k] ?? 0) < (b.min[k] ?? 0)) b.min[k] = min[k] ?? 0
    if ((max[k] ?? 0) > (b.max[k] ?? 0)) b.max[k] = max[k] ?? 0
  }
}

const area = (b: Box): number => {
  const dx = Math.max(0, b.max[0] - b.min[0])
  const dy = Math.max(0, b.max[1] - b.min[1])
  const dz = Math.max(0, b.max[2] - b.min[2])
  return 2 * (dx * dy + dy * dz + dz * dx)
}

/**
 * Build a bounding-volume hierarchy over bounded shapes (spheres, triangles, boxes): top-down, splitting each node on
 * the axis of greatest centroid spread, by the binned surface-area heuristic (cost `1 + (A_L n_L + A_R n_R) / A`,
 * kept as a leaf when splitting does not pay) or at the median. The build is iterative, so deep trees cannot overflow
 * the stack, and every split separates at least one shape, so it terminates.
 *
 * @param shapes - The shapes (no planes; unbounded shapes are traced separately).
 * @param options - Leaf size, split rule and bins.
 * @returns The hierarchy.
 * @throws WhiteboxError with code `value` for a plane, no shapes, or a leaf size below 1.
 * @concept bounding-volume-hierarchy
 * @example
 * ```ts
 * import { buildBvh, sphere } from 'whitebox-engine/graphics'
 * const bvh = buildBvh([sphere([0, 0, 0], 1), sphere([5, 0, 0], 1), sphere([10, 0, 0], 1)], { maxLeaf: 1 })
 * [bvh.nodes.length, bvh.nodes[0].min.map(Math.round), bvh.nodes[0].max.map(Math.round)] // => [5, [-1, -1, -1], [11, 1, 1]]
 * ```
 */
export function buildBvh(shapes: readonly Shape[], options: BvhOptions = {}): Bvh {
  if (shapes.length === 0) throw new WhiteboxError('value', 'buildBvh: need at least one shape')
  const maxLeaf = options.maxLeaf ?? 2
  if (!(Number.isInteger(maxLeaf) && maxLeaf >= 1)) throw new WhiteboxError('value', 'buildBvh: maxLeaf must be >= 1')
  const bins = Math.max(2, Math.floor(options.bins ?? 12))
  // Boxes are padded by a relative 1e-9 so traversal stays conservative for hits found within rounding of a face.
  const boxes = shapes.map((s) => {
    const b = bounds(s)
    const pad =
      1e-9 *
      (1 +
        Math.max(
          Math.abs(b.min[0]),
          Math.abs(b.min[1]),
          Math.abs(b.min[2]),
          Math.abs(b.max[0]),
          Math.abs(b.max[1]),
          Math.abs(b.max[2]),
        ))
    return {
      min: [b.min[0] - pad, b.min[1] - pad, b.min[2] - pad] as Vec3,
      max: [b.max[0] + pad, b.max[1] + pad, b.max[2] + pad] as Vec3,
    }
  })
  const centroid = boxes.map(
    (b): Vec3 => [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2],
  )
  const order = shapes.map((_, i) => i)
  const nodes: BvhNode[] = []
  const tasks: { node: number; start: number; end: number; depth: number }[] = []
  const open = (start: number, end: number, depth: number): number => {
    const b = emptyBox()
    for (let i = start; i < end; i++) {
      const bb = boxes[order[i] ?? 0]
      if (bb !== undefined) grow(b, bb.min, bb.max)
    }
    nodes.push({ min: b.min, max: b.max, left: -1, right: -1, start, count: end - start, depth })
    tasks.push({ node: nodes.length - 1, start, end, depth })
    return nodes.length - 1
  }
  open(0, shapes.length, 0)
  while (tasks.length > 0) {
    const task = tasks.pop()
    if (task === undefined) break
    const { node, start, end, depth } = task
    const n = end - start
    if (n <= maxLeaf) continue
    const cb = emptyBox()
    for (let i = start; i < end; i++) {
      const c = centroid[order[i] ?? 0] ?? [0, 0, 0]
      grow(cb, c, c)
    }
    const extent = [cb.max[0] - cb.min[0], cb.max[1] - cb.min[1], cb.max[2] - cb.min[2]]
    const axis =
      (extent[0] ?? 0) >= (extent[1] ?? 0) && (extent[0] ?? 0) >= (extent[2] ?? 0)
        ? 0
        : (extent[1] ?? 0) >= (extent[2] ?? 0)
          ? 1
          : 2
    const span = extent[axis] ?? 0
    // All centroids coincide: no plane separates them, so the node stays a leaf.
    if (!(span > 0)) continue
    const key = (i: number): number => (centroid[i] ?? [0, 0, 0])[axis] ?? 0
    let mid = -1
    if (options.split !== 'median') {
      const binOf = (i: number): number =>
        Math.min(bins - 1, Math.floor(((key(i) - (cb.min[axis] ?? 0)) / span) * bins))
      const counts = new Array<number>(bins).fill(0)
      const binBoxes = Array.from({ length: bins }, emptyBox)
      for (let i = start; i < end; i++) {
        const s = order[i] ?? 0
        const k = binOf(s)
        counts[k] = (counts[k] ?? 0) + 1
        const bb = boxes[s]
        const target = binBoxes[k]
        if (bb !== undefined && target !== undefined) grow(target, bb.min, bb.max)
      }
      const parent = nodes[node]
      const parentArea = parent === undefined ? 1 : area({ min: [...parent.min], max: [...parent.max] })
      let bestCost = Number.POSITIVE_INFINITY
      let bestSplit = -1
      for (let split = 1; split < bins; split++) {
        const l = emptyBox()
        const r = emptyBox()
        let nl = 0
        let nr = 0
        for (let k = 0; k < bins; k++) {
          const bb = binBoxes[k]
          if (bb === undefined || (counts[k] ?? 0) === 0) continue
          if (k < split) {
            grow(l, bb.min, bb.max)
            nl += counts[k] ?? 0
          } else {
            grow(r, bb.min, bb.max)
            nr += counts[k] ?? 0
          }
        }
        if (nl === 0 || nr === 0) continue
        const cost = 1 + (area(l) * nl + area(r) * nr) / (parentArea || 1)
        if (cost < bestCost) {
          bestCost = cost
          bestSplit = split
        }
      }
      if (bestSplit > 0) {
        if (bestCost >= n && n <= 4 * maxLeaf) continue
        const part = order.slice(start, end)
        const left = part.filter((s) => binOf(s) < bestSplit)
        const right = part.filter((s) => binOf(s) >= bestSplit)
        order.splice(start, n, ...left, ...right)
        mid = start + left.length
      }
    }
    if (mid <= start || mid >= end) {
      const part = order.slice(start, end).sort((a, b) => key(a) - key(b) || a - b)
      order.splice(start, n, ...part)
      mid = start + Math.floor(n / 2)
    }
    const left = open(start, mid, depth + 1)
    const right = open(mid, end, depth + 1)
    const cur = nodes[node]
    if (cur !== undefined) nodes[node] = { ...cur, left, right, count: 0 }
  }
  return { shapes, nodes, order }
}

/** What a traversal did: nodes whose boxes were tested and shapes intersected. */
export interface TraversalStats {
  readonly nodesVisited: number
  readonly shapeTests: number
}

/**
 * The nearest hit of a ray in a hierarchy: depth-first, nearer child first, skipping every box that starts beyond the
 * nearest hit so far. Equal to testing every shape ({@link nearestHit}, ties going to the lowest index), at a fraction
 * of the tests.
 *
 * @param bvh - The hierarchy.
 * @param ray - The ray.
 * @param options - The accepted parameter range (default `[RAY_EPSILON, Infinity]`).
 * @returns The hit with its shape index (or null), and the traversal counts.
 * @concept bounding-volume-hierarchy
 * @example
 * ```ts
 * import { buildBvh, intersectBvh, sphere } from 'whitebox-engine/graphics'
 * const bvh = buildBvh([sphere([0, 0, 0], 1), sphere([5, 0, 0], 1), sphere([10, 0, 0], 1)], { maxLeaf: 1 })
 * const r = intersectBvh(bvh, { origin: [10, 0, 5], direction: [0, 0, -1] })
 * [r.hit?.index, r.hit?.t, r.stats.shapeTests] // => [2, 4, 1]
 * ```
 */
export function intersectBvh(
  bvh: Bvh,
  ray: Ray,
  options: { readonly tMin?: number; readonly tMax?: number } = {},
): { readonly hit: (Hit & { readonly index: number }) | null; readonly stats: TraversalStats } {
  const tMin = options.tMin ?? RAY_EPSILON
  let tMax = options.tMax ?? Number.POSITIVE_INFINITY
  let best: (Hit & { readonly index: number }) | null = null
  let nodesVisited = 0
  let shapeTests = 0
  const enter = (i: number): number => {
    const nd = bvh.nodes[i]
    if (nd === undefined) return Number.POSITIVE_INFINITY
    const s = slabs(ray.origin, ray.direction, nd.min, nd.max)
    if (s === null || s[1] < tMin || s[0] > tMax) return Number.POSITIVE_INFINITY
    return Math.max(s[0], tMin)
  }
  const stack: { node: number; t: number }[] = []
  const rootT = enter(0)
  nodesVisited++
  if (rootT < Number.POSITIVE_INFINITY) stack.push({ node: 0, t: rootT })
  while (stack.length > 0) {
    const top = stack.pop()
    if (top === undefined) break
    if (top.t > tMax) continue
    const nd = bvh.nodes[top.node]
    if (nd === undefined) continue
    if (nd.count > 0) {
      for (let k = nd.start; k < nd.start + nd.count; k++) {
        const index = bvh.order[k] ?? 0
        const s = bvh.shapes[index]
        if (s === undefined) continue
        shapeTests++
        const h = intersect(ray, s, tMin, tMax)
        // h.t <= tMax = best.t; a tie goes to the lower index, as nearestHit, so the answer is independent of the
        // traversal order.
        if (h !== null && (best === null || h.t < best.t || index < best.index)) {
          best = { ...h, index }
          tMax = h.t
        }
      }
      continue
    }
    const tl = enter(nd.left)
    const tr = enter(nd.right)
    nodesVisited += 2
    // Push the farther child first so the nearer is processed next.
    const near = tl <= tr ? { node: nd.left, t: tl } : { node: nd.right, t: tr }
    const far = tl <= tr ? { node: nd.right, t: tr } : { node: nd.left, t: tl }
    if (far.t < Number.POSITIVE_INFINITY) stack.push(far)
    if (near.t < Number.POSITIVE_INFINITY) stack.push(near)
  }
  return { hit: best, stats: { nodesVisited, shapeTests } }
}

/**
 * Whether anything in a hierarchy blocks a ray within a parameter range (the any-hit query of shadow rays, which can
 * stop at the first blocker).
 *
 * @param bvh - The hierarchy.
 * @param ray - The ray.
 * @param tMax - The far end (the distance to the light).
 * @param tMin - The near end (default {@link RAY_EPSILON}).
 * @returns Whether the ray is blocked.
 * @concept bounding-volume-hierarchy
 * @example
 * ```ts
 * import { buildBvh, occludedBvh, sphere } from 'whitebox-engine/graphics'
 * const bvh = buildBvh([sphere([0, 0, 0], 1)])
 * [occludedBvh(bvh, { origin: [0, 0, 5], direction: [0, 0, -1] }, 10), occludedBvh(bvh, { origin: [0, 0, 5], direction: [0, 0, -1] }, 3)] // => [true, false]
 * ```
 */
export function occludedBvh(bvh: Bvh, ray: Ray, tMax: number, tMin: number = RAY_EPSILON): boolean {
  const stack: number[] = [0]
  while (stack.length > 0) {
    const i = stack.pop()
    if (i === undefined) break
    const nd = bvh.nodes[i]
    if (nd === undefined) continue
    const s = slabs(ray.origin, ray.direction, nd.min, nd.max)
    if (s === null || s[1] < tMin || s[0] > tMax) continue
    if (nd.count > 0) {
      for (let k = nd.start; k < nd.start + nd.count; k++) {
        const shape = bvh.shapes[bvh.order[k] ?? 0]
        if (shape !== undefined && intersect(ray, shape, tMin, tMax) !== null) return true
      }
      continue
    }
    stack.push(nd.left, nd.right)
  }
  return false
}

/**
 * The hierarchy as a {@link Structure} (a tree of boxes, leaves listing their shapes), for drawing.
 *
 * @param bvh - The hierarchy.
 * @returns The structure.
 * @concept bounding-volume-hierarchy
 * @example
 * ```ts
 * import { buildBvh, bvhStructure, sphere } from 'whitebox-engine/graphics'
 * const s = bvhStructure(buildBvh([sphere([0, 0, 0], 1), sphere([5, 0, 0], 1)], { maxLeaf: 1 }))
 * [s.nodes.length, s.edges.length] // => [3, 2]
 * ```
 */
export function bvhStructure(bvh: Bvh): Structure {
  const nodes: StructureNode[] = bvh.nodes.map((nd, i) => ({
    id: `n${i}`,
    label: nd.count > 0 ? `leaf ${bvh.order.slice(nd.start, nd.start + nd.count).join(', ')}` : `node ${i}`,
    role: nd.count > 0 ? 'leaf' : 'internal',
    badges: { depth: nd.depth, shapes: nd.count },
    data: { min: nd.min, max: nd.max },
  }))
  const edges: StructureEdge[] = []
  for (const [i, nd] of bvh.nodes.entries()) {
    if (nd.count > 0) continue
    edges.push(
      { from: `n${i}`, to: `n${nd.left}`, directed: true },
      { from: `n${i}`, to: `n${nd.right}`, directed: true },
    )
  }
  return structure('bvh', nodes, edges, { title: 'bounding-volume hierarchy', direction: 'down' })
}
