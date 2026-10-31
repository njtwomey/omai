/**
 * Region segmentation of images: SLIC superpixels (Achanta et al. 2012) as a traced algorithm of assignment and update
 * steps with connectivity enforced at the end, seeded region growing, mean-shift segmentation in the joint
 * space-colour domain (Comaniciu and Meer 2002), and colour quantisation by k-means (over `cluster`). SLIC's exact
 * labels are implementation-defined, so it is tested by its invariants rather than against scikit-image.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { initialCentres, lloydAlgorithm } from '../cluster/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'

import type { Tensor, TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { imageTensor, type Raster, rasterOf, rasterTensor } from './_image.ts'
import { rgbToLab } from './colour.ts'

/** The state of {@link slicAlgorithm}: cluster centres (features then row, column) and pixel labels. */
export interface SlicState {
  readonly centres: number[][]
  readonly labels: Int32Array
  /** The largest centre movement of the last update. */
  readonly moved: number
}

/** Options of {@link slic}. */
export interface SlicOptions {
  /** The approximate number of superpixels (default 100). */
  readonly nSegments?: number
  /** The weight of space against colour (default 10, for Lab or 0..100-scaled grey). */
  readonly compactness?: number
  /** Convert colour images to Lab first (default true). */
  readonly convertToLab?: boolean
}

/** Features per pixel: Lab (or the channels) for colour, 100 times the value for grey. */
function featuresOf(x: TensorLike, convert: boolean): Raster {
  const r = rasterOf(x, 'slic')
  if (r.c === 3 && convert) return rasterOf(rgbToLab(x), 'slic')
  if (r.c === 1) return { ...r, data: Float64Array.from(r.data, (v) => 100 * v) }
  return r
}

/**
 * SLIC superpixels as an algorithm (Achanta et al. 2012): centres start on a regular grid of step S = sqrt(N / k)
 * (moved to the lowest-gradient pixel of their 3 x 3 neighbourhood); each step assigns every pixel within 2S of a
 * centre to the nearest by d^2 = d_colour^2 + (d_space / S)^2 m^2 and moves the centres to their pixels' means.
 *
 * @param x - Image `[H, W]` (in [0, 1]) or `[H, W, 3]` (sRGB).
 * @param options - Segments, compactness and the Lab conversion.
 * @returns The algorithm.
 * @concept superpixels
 * @example
 * ```ts
 * import { slicAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const img = Array.from({ length: 12 }, () => Array.from({ length: 12 }, (_, j) => (j < 6 ? 0.1 : 0.9)))
 * runAlgorithm(slicAlgorithm(img, { nSegments: 4 }), { maxSteps: 5 }).state.centres.length // => 4
 * ```
 */
export function slicAlgorithm(x: TensorLike, options: SlicOptions = {}): Algorithm<SlicState> {
  const f = featuresOf(x, options.convertToLab ?? true)
  const { h, w, c } = f
  const k = Math.max(1, options.nSegments ?? 100)
  const S = Math.max(1, Math.sqrt((h * w) / k))
  const m = options.compactness ?? 10
  const feat = (i: number, q: number): number => f.data[i * c + q] ?? 0
  const grad = (y: number, xx: number): number => {
    let g = 0
    for (let q = 0; q < c; q++) {
      const dx = feat(y * w + Math.min(w - 1, xx + 1), q) - feat(y * w + Math.max(0, xx - 1), q)
      const dy = feat(Math.min(h - 1, y + 1) * w + xx, q) - feat(Math.max(0, y - 1) * w + xx, q)
      g += dx * dx + dy * dy
    }
    return g
  }
  // A grid of about k cells shaped like the image.
  const cols = Math.max(1, Math.round(Math.sqrt((k * w) / h)))
  const rows = Math.max(1, Math.round(k / cols))
  return {
    name: 'slic',
    config: { nSegments: k, compactness: m } as Json,
    schema: channels({
      metrics: { moved: { description: 'largest centre movement of the update', direction: 'lower' } },
      tensors: { labels: { role: 'state', description: 'the superpixel of each pixel', shape: ['h', 'w'] } },
    }),
    init: () => {
      const centres: number[][] = []
      for (let a = 0; a < rows; a++)
        for (let b = 0; b < cols; b++) {
          let y = Math.min(h - 1, Math.floor(((a + 0.5) * h) / rows))
          let xx = Math.min(w - 1, Math.floor(((b + 0.5) * w) / cols))
          let best = grad(y, xx)
          const [y0, x0] = [y, xx]
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const yy = y0 + dy
              const x2 = x0 + dx
              if (yy < 0 || yy >= h || x2 < 0 || x2 >= w) continue
              const g = grad(yy, x2)
              if (g < best) {
                best = g
                y = yy
                xx = x2
              }
            }
          centres.push([...Array.from({ length: c }, (_, q) => feat(y * w + xx, q)), y, xx])
        }
      return { centres, labels: new Int32Array(h * w).fill(-1), moved: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const best = new Float64Array(h * w).fill(Number.POSITIVE_INFINITY)
      const labels = new Int32Array(h * w).fill(-1)
      s.centres.forEach((ctr, ci) => {
        const cy = ctr[c] ?? 0
        const cx = ctr[c + 1] ?? 0
        for (let y = Math.max(0, Math.floor(cy - 2 * S)); y <= Math.min(h - 1, Math.ceil(cy + 2 * S)); y++)
          for (let xx = Math.max(0, Math.floor(cx - 2 * S)); xx <= Math.min(w - 1, Math.ceil(cx + 2 * S)); xx++) {
            const i = y * w + xx
            let dc = 0
            for (let q = 0; q < c; q++) dc += (feat(i, q) - (ctr[q] ?? 0)) ** 2
            const d = dc + (((y - cy) ** 2 + (xx - cx) ** 2) / (S * S)) * m * m
            if (d < (best[i] ?? 0)) {
              best[i] = d
              labels[i] = ci
            }
          }
      })
      const sums = s.centres.map(() => new Float64Array(c + 3))
      for (let i = 0; i < h * w; i++) {
        const l = labels[i] ?? -1
        if (l < 0) continue
        const acc = sums[l] as Float64Array
        for (let q = 0; q < c; q++) acc[q] = (acc[q] ?? 0) + feat(i, q)
        acc[c] = (acc[c] ?? 0) + Math.floor(i / w)
        acc[c + 1] = (acc[c + 1] ?? 0) + (i % w)
        acc[c + 2] = (acc[c + 2] ?? 0) + 1
      }
      let moved = 0
      const centres = s.centres.map((ctr, ci) => {
        const acc = sums[ci] as Float64Array
        const n = acc[c + 2] ?? 0
        if (n === 0) return ctr
        const next = Array.from({ length: c + 2 }, (_, q) => (acc[q] ?? 0) / n)
        moved = Math.max(moved, Math.hypot((next[c] ?? 0) - (ctr[c] ?? 0), (next[c + 1] ?? 0) - (ctr[c + 1] ?? 0)))
        return next
      })
      return { centres, labels, moved }
    },
    done: (s) => (s.moved < 1e-3 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { moved: Number.isFinite(s.moved) ? s.moved : -1 },
      tensors: { labels: { role: 'state', value: Array.from(s.labels), shape: [h, w] } },
    }),
  }
}

/** Relabel so every label is one 4-connected region, merging regions below minSize into a neighbour. */
function enforceConnectivity(labels: Int32Array, h: number, w: number, minSize: number): Int32Array {
  const out = new Int32Array(h * w).fill(-1)
  let next = 0
  const queue: number[] = []
  for (let start = 0; start < h * w; start++) {
    if ((out[start] ?? 0) >= 0) continue
    const lab = labels[start] ?? 0
    queue.length = 0
    queue.push(start)
    out[start] = next
    let adjacent = -1
    for (let q = 0; q < queue.length; q++) {
      const p = queue[q] as number
      const y = Math.floor(p / w)
      const x = p % w
      for (const [yy, xx] of [
        [y - 1, x],
        [y + 1, x],
        [y, x - 1],
        [y, x + 1],
      ] as const) {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue
        const n = yy * w + xx
        if ((out[n] ?? 0) >= 0) {
          if ((out[n] ?? 0) !== next) adjacent = out[n] ?? adjacent
          continue
        }
        if ((labels[n] ?? 0) !== lab) continue
        out[n] = next
        queue.push(n)
      }
    }
    if (queue.length < minSize && adjacent >= 0) for (const p of queue) out[p] = adjacent
    else next++
  }
  // Renumber 0 .. n - 1 in raster order of first appearance.
  const ids = new Map<number, number>()
  return Int32Array.from(out, (v) => {
    let id = ids.get(v)
    if (id === undefined) {
      id = ids.size
      ids.set(v, id)
    }
    return id
  })
}

/**
 * SLIC superpixels ({@link slicAlgorithm} run for maxIter steps or to convergence), then each label made one connected
 * region, fragments smaller than a quarter of the expected size merged into a neighbour; labels from 0 in raster order.
 *
 * @param x - Image `[H, W]` or `[H, W, 3]`.
 * @param options - Segments, compactness, Lab conversion, iterations (default 10) and tracing.
 * @returns The labels `[H, W]`.
 * @concept superpixels
 * @example
 * ```ts
 * import { slic } from 'mdml-engine/vision'
 * import { toFlat } from 'mdml-engine/tensor'
 * const img = Array.from({ length: 12 }, () => Array.from({ length: 24 }, (_, j) => (j < 12 ? 0.1 : 0.9)))
 * new Set(toFlat(slic(img, { nSegments: 2 }))).size // => 2
 * ```
 */
export function slic(
  x: TensorLike,
  options: SlicOptions & { readonly maxIter?: number } & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): Tensor {
  const r = rasterOf(x, 'slic')
  const st = runAlgorithm(slicAlgorithm(x, options), {
    maxSteps: options.maxIter ?? 10,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  const k = Math.max(1, options.nSegments ?? 100)
  const labels = enforceConnectivity(st.labels, r.h, r.w, Math.floor((r.h * r.w) / k / 4))
  return imageTensor(Float64Array.from(labels), r.h, r.w)
}

/**
 * Seeded region growing (Adams and Bischof 1994): from each seed, repeatedly absorb the 4-neighbour of the region
 * closest to the region's running mean, while that difference is below the threshold; regions grow in turn by
 * priority, each pixel joining at most one.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param seeds - Seed pixels (x, y), one region each.
 * @param options - The largest colour distance to the region mean (default 0.1).
 * @returns Labels `[H, W]`: 1 .. seeds for the regions, 0 for unreached pixels.
 * @concept region-growing
 * @example
 * ```ts
 * import { regionGrowing } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(regionGrowing([[0.1, 0.12, 0.9], [0.11, 0.1, 0.92]], [[0, 0], [2, 0]])) // => [[1, 1, 2], [1, 1, 2]]
 * ```
 */
export function regionGrowing(
  x: TensorLike,
  seeds: readonly (readonly [number, number])[],
  options: { readonly threshold?: number } = {},
): Tensor {
  const r = rasterOf(x, 'regionGrowing')
  const { h, w, c } = r
  const thr = options.threshold ?? 0.1
  const labels = new Int32Array(h * w)
  const sums = seeds.map(() => new Float64Array(c))
  const counts = seeds.map(() => 0)
  // A frontier of (distance, pixel, region) kept sorted by distance (small images: a sorted insert suffices).
  const frontier: [number, number, number][] = []
  const dist = (p: number, reg: number): number => {
    let s = 0
    for (let k = 0; k < c; k++) s += ((r.data[p * c + k] ?? 0) - (sums[reg]?.[k] ?? 0) / (counts[reg] ?? 1)) ** 2
    return Math.sqrt(s)
  }
  const push = (p: number, reg: number): void => {
    const d = dist(p, reg)
    if (d > thr) return
    let lo = 0
    let hi = frontier.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((frontier[mid]?.[0] ?? 0) <= d) lo = mid + 1
      else hi = mid
    }
    frontier.splice(lo, 0, [d, p, reg])
  }
  const absorb = (p: number, reg: number): void => {
    labels[p] = reg + 1
    for (let k = 0; k < c; k++) (sums[reg] as Float64Array)[k] = (sums[reg]?.[k] ?? 0) + (r.data[p * c + k] ?? 0)
    counts[reg] = (counts[reg] ?? 0) + 1
    const y = Math.floor(p / w)
    const xx = p % w
    for (const [yy, x2] of [
      [y - 1, xx],
      [y + 1, xx],
      [y, xx - 1],
      [y, xx + 1],
    ] as const)
      if (yy >= 0 && yy < h && x2 >= 0 && x2 < w && (labels[yy * w + x2] ?? 0) === 0) push(yy * w + x2, reg)
  }
  seeds.forEach(([sx, sy], reg) => {
    if (!(sx >= 0 && sy >= 0 && sx < w && sy < h)) throw new MdmlError('value', 'regionGrowing: a seed lies outside')
    absorb(Math.round(sy) * w + Math.round(sx), reg)
  })
  while (frontier.length > 0) {
    const [, p, reg] = frontier.shift() as [number, number, number]
    if ((labels[p] ?? 0) !== 0) continue
    if (dist(p, reg) > thr) continue
    absorb(p, reg)
  }
  return imageTensor(Float64Array.from(labels), h, w)
}

/**
 * Mean-shift segmentation (Comaniciu and Meer 2002): every pixel climbs to the mode of the joint space-colour density
 * (a flat kernel of spatial radius sp and colour radius sr, iterated until it moves less than eps), and pixels whose
 * modes lie within half the radii are labelled together.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Spatial radius (default 5), colour radius (default 0.1), iterations (default 10).
 * @returns The labels `[H, W]` (from 0) and the filtered image (each pixel replaced by its mode's colour).
 * @concept mean-shift
 * @example
 * ```ts
 * import { meanShiftSegmentation } from 'mdml-engine/vision'
 * import { toFlat } from 'mdml-engine/tensor'
 * const img = Array.from({ length: 6 }, () => Array.from({ length: 8 }, (_, j) => (j < 4 ? 0.2 : 0.8)))
 * new Set(toFlat(meanShiftSegmentation(img, { spatialRadius: 2 }).labels)).size // => 2
 * ```
 */
export function meanShiftSegmentation(
  x: TensorLike,
  options: { readonly spatialRadius?: number; readonly colourRadius?: number; readonly maxIter?: number } = {},
): { readonly labels: Tensor; readonly filtered: Tensor } {
  const r = rasterOf(x, 'meanShiftSegmentation')
  const { h, w, c } = r
  const sp = options.spatialRadius ?? 5
  const sr = options.colourRadius ?? 0.1
  const iters = options.maxIter ?? 10
  const modes = new Float64Array(h * w * (c + 2))
  for (let p = 0; p < h * w; p++) {
    const mode = startingMode(r, p)
    for (let it = 0; it < iters; it++) if (shiftMode(r, sp, sr, mode) < 1e-6) break
    modes.set(mode, p * (c + 2))
  }
  return { labels: imageTensor(groupModes(modes, h, w, c, sp, sr), h, w), filtered: filteredOf(r, modes) }
}

/** A pixel's starting mode: its position (row, column) and colour. */
function startingMode(r: Raster, p: number): Float64Array {
  const mode = new Float64Array(r.c + 2)
  mode[0] = Math.floor(p / r.w)
  mode[1] = p % r.w
  for (let k = 0; k < r.c; k++) mode[2 + k] = r.data[p * r.c + k] ?? 0
  return mode
}

/**
 * One mean-shift iteration of a mode in place: the mean position and colour of the pixels within the spatial radius
 * and the colour radius. Returns the squared distance moved, or -1 when the window held no pixel (the mode stays).
 */
function shiftMode(r: Raster, sp: number, sr: number, mode: Float64Array): number {
  const { h, w, c } = r
  const my = mode[0] ?? 0
  const mx = mode[1] ?? 0
  let sy = 0
  let sx = 0
  const sc = new Float64Array(c)
  let n = 0
  for (let y = Math.max(0, Math.ceil(my - sp)); y <= Math.min(h - 1, Math.floor(my + sp)); y++)
    for (let xx = Math.max(0, Math.ceil(mx - sp)); xx <= Math.min(w - 1, Math.floor(mx + sp)); xx++) {
      let d = 0
      for (let k = 0; k < c; k++) d += ((r.data[(y * w + xx) * c + k] ?? 0) - (mode[2 + k] ?? 0)) ** 2
      if (d > sr * sr) continue
      sy += y
      sx += xx
      for (let k = 0; k < c; k++) sc[k] = (sc[k] ?? 0) + (r.data[(y * w + xx) * c + k] ?? 0)
      n++
    }
  if (n === 0) return -1
  const ny = sy / n
  const nx = sx / n
  let shift = (ny - my) ** 2 + (nx - mx) ** 2
  for (let k = 0; k < c; k++) {
    const v = (sc[k] ?? 0) / n
    shift += (v - (mode[2 + k] ?? 0)) ** 2
    mode[2 + k] = v
  }
  mode[0] = ny
  mode[1] = nx
  return shift
}

/** Labels (from 0, in raster order of first appearance) by union of 4-neighbours whose modes are within half the radii. */
function groupModes(modes: Float64Array, h: number, w: number, c: number, sp: number, sr: number): Float64Array {
  const parent = Int32Array.from({ length: h * w }, (_, i) => i)
  const find = (i: number): number => {
    let k = i
    while ((parent[k] ?? k) !== k) k = parent[k] ?? k
    return k
  }
  const near = (a: number, b: number): boolean => {
    const A = a * (c + 2)
    const B = b * (c + 2)
    let ds = 0
    for (let k = 0; k < 2; k++) ds += ((modes[A + k] ?? 0) - (modes[B + k] ?? 0)) ** 2
    let dc = 0
    for (let k = 2; k < c + 2; k++) dc += ((modes[A + k] ?? 0) - (modes[B + k] ?? 0)) ** 2
    return ds <= (sp / 2) ** 2 && dc <= (sr / 2) ** 2
  }
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++) {
      const p = y * w + xx
      for (const q of [xx + 1 < w ? p + 1 : -1, y + 1 < h ? p + w : -1])
        if (q >= 0 && near(p, q)) {
          const a = find(p)
          const b = find(q)
          if (a !== b) parent[Math.max(a, b)] = Math.min(a, b)
        }
    }
  const ids = new Map<number, number>()
  const labels = new Float64Array(h * w)
  for (let p = 0; p < h * w; p++) {
    const root = find(p)
    let id = ids.get(root)
    if (id === undefined) {
      id = ids.size
      ids.set(root, id)
    }
    labels[p] = id
  }
  return labels
}

/** The image with each pixel replaced by its mode's colour. */
function filteredOf(r: Raster, modes: Float64Array): Tensor {
  const filtered = new Float64Array(r.h * r.w * r.c)
  for (let p = 0; p < r.h * r.w; p++)
    for (let k = 0; k < r.c; k++) filtered[p * r.c + k] = modes[p * (r.c + 2) + 2 + k] ?? 0
  return rasterTensor({ ...r, data: filtered })
}

/** The state of {@link meanShiftSegmentationAlgorithm}. */
export interface MeanShiftSegmentationState {
  /** Every pixel's mode so far: row, column and colour, `[H W][2 + C]` flat. */
  readonly modes: Float64Array
  /** Whether each pixel's window has stopped moving. */
  readonly settled: Uint8Array
  /** The pixels still moving. */
  readonly moving: number
  /** The regions the modes form now. */
  readonly labels: Float64Array
  readonly regions: number
  readonly iteration: number
}

/** The paths of {@link meanShiftSegmentationAlgorithm}. */
export type MeanShiftSegmentationPath = 'shift/moving' | 'shift/regions' | 'shift/positions' | 'shift/labels'

/** What a mean-shift segmentation run logs. */
export const meanShiftSegmentationContract: Contract<MeanShiftSegmentationState, MeanShiftSegmentationPath> =
  defineContract({
    'shift/moving': scalar({ description: 'the pixels whose window is still moving', direction: 'lower' }),
    'shift/regions': scalar({ description: 'the regions the current modes form', direction: 'lower' }),
    'shift/positions': tensorSource({
      role: 'state',
      shape: ['pixels', 'dims'],
      description: 'every pixel’s mode: row, column, colour',
    }),
    'shift/labels': tensorSource({ role: 'output', shape: ['height', 'width'], description: 'the regions now' }),
  })

/**
 * Mean-shift segmentation as an algorithm: each step moves every pixel's window one iteration up the joint
 * space-colour density, and groups the modes reached so far into regions, so the modes can be watched merging. Each
 * pixel follows exactly the iteration of {@link meanShiftSegmentation}, so the last state's labels are its labels.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Spatial radius (default 5), colour radius (default 0.1) and the most iterations (default 10).
 * @returns The algorithm; it ends when no window moves or after `maxIter` steps.
 * @concept mean-shift
 * @example
 * ```ts
 * import { meanShiftSegmentationAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const img = Array.from({ length: 6 }, () => Array.from({ length: 8 }, (_, j) => (j < 4 ? 0.2 : 0.8)))
 * runAlgorithm(meanShiftSegmentationAlgorithm(img, { spatialRadius: 2 }), { maxSteps: 20 }).state.regions // => 2
 * ```
 */
export function meanShiftSegmentationAlgorithm(
  x: TensorLike,
  options: { readonly spatialRadius?: number; readonly colourRadius?: number; readonly maxIter?: number } = {},
): Algorithm<MeanShiftSegmentationState, MeanShiftSegmentationPath> {
  const r = rasterOf(x, 'meanShiftSegmentationAlgorithm')
  const { h, w, c } = r
  const sp = options.spatialRadius ?? 5
  const sr = options.colourRadius ?? 0.1
  const iters = options.maxIter ?? 10
  const d = c + 2
  const regionsOf = (labels: Float64Array): number => labels.reduce((m, v) => Math.max(m, v), -1) + 1
  return {
    name: 'meanShiftSegmentation',
    config: { spatialRadius: sp, colourRadius: sr, maxIter: iters } as Json,
    init: () => {
      const modes = new Float64Array(h * w * d)
      for (let p = 0; p < h * w; p++) modes.set(startingMode(r, p), p * d)
      const labels = groupModes(modes, h, w, c, sp, sr)
      return { modes, settled: new Uint8Array(h * w), moving: h * w, labels, regions: regionsOf(labels), iteration: 0 }
    },
    step: (s, ctx) => {
      const modes = Float64Array.from(s.modes)
      const settled = Uint8Array.from(s.settled)
      let moving = 0
      for (let p = 0; p < h * w; p++) {
        if (settled[p] === 1) continue
        const mode = modes.subarray(p * d, (p + 1) * d)
        const shift = shiftMode(r, sp, sr, mode)
        if (shift < 1e-6) settled[p] = 1
        else moving++
      }
      const labels = groupModes(modes, h, w, c, sp, sr)
      const regions = regionsOf(labels)
      ctx.log('shift/moving', moving)
      ctx.log('shift/regions', regions)
      if (ctx.wants('shift/positions'))
        ctx.log('shift/positions', rasterTensor({ h: h * w, w: d, c: 1, rank: 2, data: modes }))
      if (ctx.wants('shift/labels')) ctx.log('shift/labels', imageTensor(labels, h, w))
      return { modes, settled, moving, labels, regions, iteration: s.iteration + 1 }
    },
    done: (s) => (s.iteration > 0 && s.moving === 0 ? 'converged' : s.iteration >= iters ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { moving: s.moving, regions: s.regions },
      tensors: { labels: { role: 'output', value: s.labels, shape: [h, w] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MeanShiftSegmentationState,
  }
}

/**
 * Colour quantisation by k-means (Lloyd's algorithm of `cluster`, k-means++ seeded): every pixel replaced by its
 * cluster's mean colour.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param k - The palette size.
 * @param options - The key of the seeding (default key 0) and the most Lloyd iterations (default 50).
 * @returns The quantised image, the labels `[H, W]` and the palette `[k][C]`.
 * @concept colour-quantisation
 * @example
 * ```ts
 * import { kmeansQuantise } from 'mdml-engine/vision'
 * const img = [[[0, 0, 0], [0.1, 0, 0], [1, 1, 1], [0.9, 1, 1]]]
 * kmeansQuantise(img, 2).palette.length // => 2
 * ```
 */
export function kmeansQuantise(
  x: TensorLike,
  k: number,
  options: { readonly key?: Key; readonly maxIter?: number } = {},
): { readonly image: Tensor; readonly labels: Tensor; readonly palette: number[][] } {
  const r = rasterOf(x, 'kmeansQuantise')
  const n = r.h * r.w
  if (!(Number.isInteger(k) && k >= 1 && k <= n)) throw new MdmlError('value', 'kmeansQuantise: k must be in 1..pixels')
  const rows = Array.from({ length: n }, (_, i) => Array.from(r.data.subarray(i * r.c, (i + 1) * r.c)))
  const init = initialCentres(rows, k, 'k-means++', options.key ?? makeKey(0))
  const st = runAlgorithm(lloydAlgorithm(rows, init), { maxSteps: options.maxIter ?? 50 }).state
  const palette = st.centres.map((c) => [...c])
  const out = new Float64Array(n * r.c)
  for (const [i, l] of st.labels.entries()) out.set(palette[l] ?? [], i * r.c)
  return {
    image: rasterTensor({ ...r, data: out }),
    labels: imageTensor(Float64Array.from(st.labels), r.h, r.w),
    palette,
  }
}
