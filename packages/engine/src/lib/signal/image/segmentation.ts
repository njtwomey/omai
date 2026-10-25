/**
 * Image segmentation: global thresholds by Otsu's method and Li's minimum cross-entropy, marker-based watershed
 * flooding, Felzenszwalb and Huttenlocher's graph-based segmentation (as scikit-image), and exact binary segmentation
 * by a minimum graph cut of a data term plus a contrast-sensitive smoothness term (Boykov and Jolly).
 */

import { MdmlError } from '../../base/index.ts'
import { digraph, minCut } from '../../graph/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { float } from '../_shared.ts'
import { gaussianFilter } from './filters.ts'

// Image segmentation: histogram thresholds (Otsu, Li), marker-based watershed by priority flooding (as
// scikit-image), Felzenszwalb and Huttenlocher's graph-based segmentation, and binary graph cuts (an exact energy
// minimum by maximum flow). TODO: SLIC superpixels (k-means in Lab space with connectivity enforcement).

function valuesOf(x: TensorLike): { data: Float64Array; shape: number[] } {
  const v = primalOf(float(x))
  return { data: Float64Array.from(v.data, Number), shape: [...v.shape] }
}

/** numpy.histogram over the data range: counts and bin centres. */
function histogram(data: Float64Array, nbins: number): { counts: Float64Array; centres: Float64Array } {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of data) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const counts = new Float64Array(nbins)
  const step = (hi - lo) / nbins
  // numpy.linspace edges (the last exactly hi) and numpy's bin assignment with its edge corrections.
  const edges = Float64Array.from({ length: nbins + 1 }, (_, i) => (i === nbins ? hi : i * step + lo))
  const norm = nbins / (hi - lo)
  for (const v of data) {
    let b = Math.trunc((v - lo) * norm)
    if (b === nbins) b--
    if (v < (edges[b] ?? 0)) b--
    else if (b !== nbins - 1 && v >= (edges[b + 1] ?? 0)) b++
    counts[b] = (counts[b] ?? 0) + 1
  }
  const centres = Float64Array.from({ length: nbins }, (_, i) => ((edges[i] ?? 0) + (edges[i + 1] ?? 0)) / 2)
  return { counts, centres }
}

/**
 * Otsu's threshold (skimage.filters.threshold_otsu): the bin centre maximising the between-class variance of the
 * image histogram.
 *
 * @param x - Image.
 * @param nbins - Histogram bins (default 256).
 * @returns The threshold (pixels above it are foreground).
 * @concept thresholding
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * image.thresholdOtsu([0, 0, 0, 1, 10, 10, 11, 10], 4) // => 1.375
 * ```
 */
export function thresholdOtsu(x: TensorLike, nbins = 256): number {
  const { data } = valuesOf(x)
  if (data.every((v) => v === data[0])) return data[0] ?? 0
  const { counts, centres } = histogram(data, nbins)
  const n = counts.length
  const w1 = new Float64Array(n)
  const w2 = new Float64Array(n)
  const m1 = new Float64Array(n)
  const m2 = new Float64Array(n)
  let a = 0
  let s = 0
  for (let i = 0; i < n; i++) {
    a += counts[i] ?? 0
    s += (counts[i] ?? 0) * (centres[i] ?? 0)
    w1[i] = a
    m1[i] = s / a
  }
  a = 0
  s = 0
  for (let i = n - 1; i >= 0; i--) {
    a += counts[i] ?? 0
    s += (counts[i] ?? 0) * (centres[i] ?? 0)
    w2[i] = a
    m2[i] = s / a
  }
  let best = 0
  let bestVar = Number.NEGATIVE_INFINITY
  for (let i = 0; i < n - 1; i++) {
    const v = (w1[i] ?? 0) * (w2[i + 1] ?? 0) * ((m1[i] ?? 0) - (m2[i + 1] ?? 0)) ** 2
    if (v > bestVar) {
      bestVar = v
      best = i
    }
  }
  return centres[best] ?? 0
}

/**
 * Li's minimum cross-entropy threshold (skimage.filters.threshold_li): iterate t = (mu_b - mu_f) / (ln mu_b -
 * ln mu_f) on the image shifted to start at zero, until t moves less than half the smallest gap between values.
 *
 * @param x - Image.
 * @returns The threshold.
 * @concept thresholding
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * const t = image.thresholdLi([1, 1.2, 0.9, 1.1, 7, 7.5, 6.8, 7.2])
 * t > 1.2 && t < 6.8 // => true
 * ```
 */
export function thresholdLi(x: TensorLike): number {
  const { data } = valuesOf(x)
  const finite = data.filter(Number.isFinite)
  if (finite.length === 0) return 0
  if (finite.every((v) => v === finite[0])) return finite[0] ?? 0
  let lo = Number.POSITIVE_INFINITY
  for (const v of finite) lo = Math.min(lo, v)
  const shifted = Float64Array.from(finite, (v) => v - lo)
  const unique = Array.from(new Set(shifted)).sort((p, q) => p - q)
  let gap = Number.POSITIVE_INFINITY
  for (let i = 1; i < unique.length; i++) gap = Math.min(gap, (unique[i] ?? 0) - (unique[i - 1] ?? 0))
  const tol = gap / 2
  let next = shifted.reduce((p, v) => p + v, 0) / shifted.length
  let curr = -2 * tol
  for (let it = 0; it < 10_000 && Math.abs(next - curr) > tol; it++) {
    curr = next
    let sf = 0
    let nf = 0
    let sb = 0
    let nb = 0
    for (const v of shifted)
      if (v > curr) {
        sf += v
        nf++
      } else {
        sb += v
        nb++
      }
    const mf = sf / nf
    const mb = sb / nb
    if (mb === 0) break
    next = (mb - mf) / (Math.log(mb) - Math.log(mf))
  }
  return next + lo
}

/** A binary min-heap of (value, age, index) for priority flooding. */
class Heap {
  private readonly v: number[] = []
  private readonly a: number[] = []
  private readonly i: number[] = []
  get size(): number {
    return this.v.length
  }
  private less(p: number, q: number): boolean {
    const vp = this.v[p] ?? 0
    const vq = this.v[q] ?? 0
    return vp < vq || (vp === vq && (this.a[p] ?? 0) < (this.a[q] ?? 0))
  }
  private swap(p: number, q: number): void {
    for (const arr of [this.v, this.a, this.i]) {
      const t = arr[p] ?? 0
      arr[p] = arr[q] ?? 0
      arr[q] = t
    }
  }
  push(value: number, age: number, index: number): void {
    this.v.push(value)
    this.a.push(age)
    this.i.push(index)
    let k = this.v.length - 1
    while (k > 0) {
      const parent = (k - 1) >> 1
      if (!this.less(k, parent)) break
      this.swap(k, parent)
      k = parent
    }
  }
  pop(): number {
    const top = this.i[0] ?? 0
    const last = this.v.length - 1
    this.swap(0, last)
    this.v.pop()
    this.a.pop()
    this.i.pop()
    let k = 0
    for (;;) {
      const l = 2 * k + 1
      const r = l + 1
      let m = k
      if (l < this.v.length && this.less(l, m)) m = l
      if (r < this.v.length && this.less(r, m)) m = r
      if (m === k) break
      this.swap(k, m)
      k = m
    }
    return top
  }
}

/**
 * Marker-based watershed (skimage.segmentation.watershed): basins flooded from the labelled markers in order of
 * image value (ties first-come), each pixel taking the label of the neighbour that reached it.
 *
 * @param x - Image `[H, W]` (for example a gradient magnitude).
 * @param markers - Labelled markers (0 is unlabelled), the shape of x.
 * @param options - Connectivity (1: 4-neighbours, default; 2: 8-neighbours) and a mask of pixels to label.
 * @returns The labels.
 * @concept watershed
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(image.watershed([[0, 1, 5, 1, 0]], [[1, 0, 0, 0, 2]])) // => [[1, 1, 1, 2, 2]]
 * ```
 */
export function watershed(
  x: TensorLike,
  markers: TensorLike,
  options: { readonly connectivity?: 1 | 2; readonly mask?: TensorLike } = {},
): Tensor {
  const img = valuesOf(x)
  if (img.shape.length !== 2) throw new MdmlError('shape', 'watershed: images are [H, W]')
  const [h = 0, w = 0] = img.shape
  const lab = valuesOf(markers).data
  if (lab.length !== h * w) throw new MdmlError('shape', 'watershed: markers must have the image shape')
  const mask = options.mask === undefined ? new Float64Array(h * w).fill(1) : valuesOf(options.mask).data
  const out = Float64Array.from(lab, (v, i) => ((mask[i] ?? 0) !== 0 ? v : 0))
  // Neighbours ordered as scikit-image: by distance, then by raveled offset.
  const offsets: [number, number][] =
    (options.connectivity ?? 1) === 1
      ? [
          [-1, 0],
          [0, -1],
          [0, 1],
          [1, 0],
        ]
      : [
          [-1, 0],
          [0, -1],
          [0, 1],
          [1, 0],
          [-1, -1],
          [-1, 1],
          [1, -1],
          [1, 1],
        ]
  const heap = new Heap()
  for (let i = 0; i < h * w; i++) if ((out[i] ?? 0) !== 0) heap.push(img.data[i] ?? 0, 0, i)
  let age = 1
  while (heap.size > 0) {
    const p = heap.pop()
    const r = Math.floor(p / w)
    const c = p % w
    for (const [dr, dc] of offsets) {
      const rr = r + dr
      const cc = c + dc
      if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue
      const q = rr * w + cc
      if ((mask[q] ?? 0) === 0 || (out[q] ?? 0) !== 0) continue
      age++
      out[q] = out[p] ?? 0
      heap.push(img.data[q] ?? 0, age, q)
    }
  }
  return fromStorage(out, [h, w], 'float64')
}

/**
 * Felzenszwalb and Huttenlocher's graph-based segmentation (skimage.segmentation.felzenszwalb): Gaussian smoothing,
 * 8-connected edges weighted by colour distance taken in increasing order, two components merged when the edge is
 * lighter than both internal differences plus scale / size, then components below minSize merged away.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Scale (default 1), smoothing sigma (0.8) and minimum size (20).
 * @returns Labels 0 .. k - 1, numbered by first pixel.
 * @concept graph-segmentation
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 10 }, () => Array.from({ length: 10 }, (_, j) => (j < 5 ? 0 : 1)))
 * new Set(toFlat(image.felzenszwalb(x, { minSize: 20 }))).size // => 2
 * ```
 */
export function felzenszwalb(
  x: TensorLike,
  options: { readonly scale?: number; readonly sigma?: number; readonly minSize?: number } = {},
): Tensor {
  const v = valuesOf(x)
  const [h = 0, w = 0, c = 1] = v.shape
  if (v.shape.length < 2 || v.shape.length > 3)
    throw new MdmlError('shape', 'felzenszwalb: images are [H, W] or [H, W, C]')
  const sigma = options.sigma ?? 0.8
  const smoothed = Float64Array.from(
    primalOf(gaussianFilter(fromStorage(v.data, [h, w, c], 'float64'), [sigma, sigma, 0], { mode: 'reflect' })).data,
    Number,
  )
  const cost = (p: number, q: number): number => {
    let s = 0
    for (let k = 0; k < c; k++) s += ((smoothed[p * c + k] ?? 0) - (smoothed[q * c + k] ?? 0)) ** 2
    return Math.sqrt(s)
  }
  const ea: number[] = []
  const eb: number[] = []
  const ew: number[] = []
  const add = (a: number, b: number): void => {
    ea.push(a)
    eb.push(b)
    ew.push(cost(a, b))
  }
  for (let r = 0; r < h; r++) for (let col = 1; col < w; col++) add(r * w + col, r * w + col - 1)
  for (let r = 1; r < h; r++) for (let col = 0; col < w; col++) add(r * w + col, (r - 1) * w + col)
  for (let r = 1; r < h; r++) for (let col = 1; col < w; col++) add(r * w + col, (r - 1) * w + col - 1)
  for (let r = 0; r < h - 1; r++) for (let col = 1; col < w; col++) add(r * w + col, (r + 1) * w + col - 1)
  const order = ew.map((_, i) => i).sort((p, q) => (ew[p] ?? 0) - (ew[q] ?? 0) || p - q)
  const n = h * w
  const parent = Int32Array.from({ length: n }, (_, i) => i)
  const find = (i: number): number => {
    let r = i
    while ((parent[r] ?? r) !== r) r = parent[r] ?? r
    let k = i
    while ((parent[k] ?? k) !== r) {
      const nx = parent[k] ?? k
      parent[k] = r
      k = nx
    }
    return r
  }
  const join = (a: number, b: number): number => {
    const ra = find(a)
    const rb = find(b)
    const root = Math.min(ra, rb)
    parent[ra] = root
    parent[rb] = root
    return root
  }
  const size = new Float64Array(n).fill(1)
  const inner = new Float64Array(n)
  // As scikit-image (after the reference implementation, whose intensities are 0..255): scale / 255.
  const scale = (options.scale ?? 1) / 255
  for (const e of order) {
    const s0 = find(ea[e] ?? 0)
    const s1 = find(eb[e] ?? 0)
    if (s0 === s1) continue
    const wt = ew[e] ?? 0
    if (wt < Math.min((inner[s0] ?? 0) + scale / (size[s0] ?? 1), (inner[s1] ?? 0) + scale / (size[s1] ?? 1))) {
      const root = join(s0, s1)
      size[root] = (size[s0] ?? 0) + (size[s1] ?? 0)
      inner[root] = wt
    }
  }
  const minSize = options.minSize ?? 20
  for (const e of order) {
    const s0 = find(ea[e] ?? 0)
    const s1 = find(eb[e] ?? 0)
    if (s0 === s1) continue
    if ((size[s0] ?? 0) < minSize || (size[s1] ?? 0) < minSize) {
      const root = join(s0, s1)
      size[root] = (size[s0] ?? 0) + (size[s1] ?? 0)
    }
  }
  const roots = Array.from({ length: n }, (_, i) => find(i))
  const ids = Array.from(new Set(roots)).sort((p, q) => p - q)
  const index = new Map(ids.map((r, i) => [r, i]))
  return fromStorage(
    Float64Array.from(roots, (r) => index.get(r) ?? 0),
    [h, w],
    'float64',
  )
}

/** Options of {@link graphCut}. */
export interface GraphCutOptions {
  /** Mean intensity of the foreground and background models. */
  readonly foreground: number
  readonly background: number
  /** Weight of the smoothness term (default 1). */
  readonly lambda?: number
  /** Intensity scale of the contrast-sensitive smoothness (default the image standard deviation). */
  readonly sigma?: number
}

/**
 * Binary segmentation by graph cut (Boykov and Jolly 2001): the exact minimum of sum_p (I_p - mu_label)^2 + lambda
 * sum_{p~q} exp(-(I_p - I_q)^2 / (2 sigma^2)) [labels differ] over 4-connected labellings, by a minimum s-t cut.
 *
 * @param x - Image `[H, W]`.
 * @param options - The two intensity models, smoothness weight and contrast scale.
 * @returns A 0/1 foreground mask and the minimum energy.
 * @concept graph-cuts
 * @example
 * ```ts
 * import { image } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(image.graphCut([[0.1, 0.9, 0.8], [0.2, 0.7, 0.1]], { foreground: 0.8, background: 0.1, lambda: 0.05 }).mask) // => [[0, 1, 1], [0, 1, 0]]
 * ```
 */
export function graphCut(x: TensorLike, options: GraphCutOptions): { readonly mask: Tensor; readonly energy: number } {
  const v = valuesOf(x)
  if (v.shape.length !== 2) throw new MdmlError('shape', 'graphCut: images are [H, W]')
  const [h = 0, w = 0] = v.shape
  const n = h * w
  const lambda = options.lambda ?? 1
  let sigma = options.sigma
  if (sigma === undefined) {
    const mean = v.data.reduce((a, b) => a + b, 0) / n
    sigma = Math.sqrt(v.data.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1
  }
  const s = n
  const t = n + 1
  const edges: [number, number, number][] = []
  for (let p = 0; p < n; p++) {
    const ip = v.data[p] ?? 0
    // Cutting s->p labels p background (pays the background cost); cutting p->t labels it foreground.
    edges.push([s, p, (ip - options.background) ** 2])
    edges.push([p, t, (ip - options.foreground) ** 2])
  }
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const p = r * w + c
      for (const q of [c + 1 < w ? p + 1 : -1, r + 1 < h ? p + w : -1]) {
        if (q < 0) continue
        const wt = lambda * Math.exp(-(((v.data[p] ?? 0) - (v.data[q] ?? 0)) ** 2) / (2 * sigma * sigma))
        edges.push([p, q, wt], [q, p, wt])
      }
    }
  const cut = minCut(digraph(edges), s, t)
  const mask = new Float64Array(n)
  for (const p of cut.partition[0]) if (p < n) mask[p] = 1
  return { mask: fromStorage(mask, [h, w], 'float64'), energy: cut.value }
}
