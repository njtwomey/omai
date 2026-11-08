/**
 * Mathematical morphology, scipy.ndimage and scikit-image style: binary and grey erosion, dilation, opening and closing
 * with a structuring element, the morphological gradient and top-hats, connected-component labelling, the exact
 * Euclidean distance transform, and thinning and skeletonisation to one-pixel-wide shapes.
 */

import { OmaiError } from '../../base/index.ts'
import { type BorderMode, extendIndex, padModeOf } from '../../numerics/convolution/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { float } from '../_shared.ts'

// Mathematical morphology as scipy.ndimage and scikit-image define it: binary and grey erosion, dilation, opening and
// closing with structuring elements and origins, morphological gradients and top-hats, connected-component labelling,
// the exact Euclidean distance transform, and thinning and skeletonisation. Integer-valued outputs; not
// differentiable.

interface Grid {
  readonly data: Float64Array
  readonly shape: readonly number[]
  readonly strides: readonly number[]
}

function gridOf(x: TensorLike): Grid {
  const v = primalOf(float(x))
  const shape = [...v.shape]
  const strides = new Array<number>(shape.length).fill(1)
  for (let d = shape.length - 2; d >= 0; d--) strides[d] = (strides[d + 1] ?? 1) * (shape[d + 1] ?? 1)
  return { data: Float64Array.from(v.data, Number), shape, strides }
}

function coords(flat: number, shape: readonly number[], out: number[]): void {
  let rem = flat
  for (let d = shape.length - 1; d >= 0; d--) {
    out[d] = rem % (shape[d] ?? 1)
    rem = Math.floor(rem / (shape[d] ?? 1))
  }
}

/** A structuring element: its shape, the offsets of its true cells and their flat indices. */
interface Structure {
  readonly shape: readonly number[]
  readonly offsets: readonly (readonly number[])[]
}

function structureOf(s: TensorLike | undefined, rank: number, name: string): Structure {
  const g = s === undefined ? gridOf(generateBinaryStructure(rank, 1)) : gridOf(s)
  if (g.shape.length !== rank) throw new OmaiError('shape', `${name}: the structure must have the input's rank`)
  const offsets: number[][] = []
  const c = new Array<number>(rank).fill(0)
  for (let i = 0; i < g.data.length; i++)
    if ((g.data[i] ?? 0) !== 0) {
      coords(i, g.shape, c)
      offsets.push([...c])
    }
  return { shape: g.shape, offsets }
}

function reflect(s: Structure): Structure {
  return { shape: s.shape, offsets: s.offsets.map((o) => o.map((v, d) => (s.shape[d] ?? 1) - 1 - v)) }
}

const originsOf = (origin: number | readonly number[] | undefined, rank: number): number[] =>
  origin === undefined
    ? new Array<number>(rank).fill(0)
    : typeof origin === 'number'
      ? new Array<number>(rank).fill(origin)
      : [...origin]

/** scipy's origin flip for the reflected element of a dilation: -origin, minus one on even-length axes. */
const flippedOrigins = (origin: readonly number[], shape: readonly number[]): number[] =>
  origin.map((o, d) => -o - ((shape[d] ?? 1) % 2 === 0 ? 1 : 0))

/**
 * A structuring element of the given rank and connectivity (scipy.ndimage.generate_binary_structure): cells of the
 * 3^rank cube within city-block distance `connectivity` of the centre.
 *
 * @param rank - Number of dimensions.
 * @param connectivity - 1 (faces) .. rank (full cube).
 * @returns A 0/1 tensor `[3, ..., 3]`.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.generateBinaryStructure(2, 1)) // => [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
 * ```
 */
export function generateBinaryStructure(rank: number, connectivity: number): Tensor {
  const size = 3 ** rank
  const out = new Float64Array(size)
  const c = new Array<number>(rank).fill(0)
  for (let i = 0; i < size; i++) {
    coords(i, new Array<number>(rank).fill(3), c)
    out[i] = c.reduce((s, v) => s + Math.abs(v - 1), 0) <= Math.max(1, connectivity) ? 1 : 0
  }
  return fromStorage(out, new Array<number>(rank).fill(3), 'float64')
}

/** Options of the binary morphology. */
export interface BinaryOptions {
  /** Structuring element (default the cross of connectivity 1). */
  readonly structure?: TensorLike
  /** Repetitions (default 1; below 1, until nothing changes). */
  readonly iterations?: number
  /** Value outside the image (default 0). */
  readonly borderValue?: number
  /** Origin of the element per axis or shared (default 0). */
  readonly origin?: number | readonly number[]
}

function binaryStep(
  x: Uint8Array,
  shape: readonly number[],
  strides: readonly number[],
  s: Structure,
  origin: readonly number[],
  border: number,
  dilate: boolean,
): Uint8Array {
  const rank = shape.length
  const out = new Uint8Array(x.length)
  const c = new Array<number>(rank).fill(0)
  for (let p = 0; p < x.length; p++) {
    coords(p, shape, c)
    let acc = dilate ? 0 : 1
    for (const off of s.offsets) {
      let idx = 0
      let outside = false
      for (let d = 0; d < rank; d++) {
        const i = (c[d] ?? 0) + (off[d] ?? 0) - Math.floor((s.shape[d] ?? 1) / 2) - (origin[d] ?? 0)
        if (i < 0 || i >= (shape[d] ?? 0)) {
          outside = true
          break
        }
        idx += i * (strides[d] ?? 1)
      }
      const v = outside ? (border !== 0 ? 1 : 0) : (x[idx] ?? 0)
      if (dilate && v === 1) {
        acc = 1
        break
      }
      if (!dilate && v === 0) {
        acc = 0
        break
      }
    }
    out[p] = acc
  }
  return out
}

function binaryOp(x: TensorLike, options: BinaryOptions, dilate: boolean, name: string): Tensor {
  const g = gridOf(x)
  const rank = g.shape.length
  let s = structureOf(options.structure, rank, name)
  let origin = originsOf(options.origin, rank)
  if (dilate) {
    origin = flippedOrigins(origin, s.shape)
    s = reflect(s)
    // Dilation is the complement of eroding the complement with the reflected element (scipy's invert = 1).
  }
  let cur: Uint8Array = Uint8Array.from(g.data, (v) => (v !== 0 ? 1 : 0))
  const iterations = options.iterations ?? 1
  const border = options.borderValue ?? 0
  for (let it = 0; iterations < 1 || it < iterations; it++) {
    const next = binaryStep(cur, g.shape, g.strides, s, origin, border, dilate)
    let changed = false
    for (let i = 0; i < next.length; i++) if (next[i] !== cur[i]) changed = true
    cur = next
    if (!changed) break
  }
  return fromStorage(Float64Array.from(cur), [...g.shape], 'float64')
}

/**
 * Binary erosion (scipy.ndimage.binary_erosion): a cell stays set when the structuring element placed on it lies
 * inside the set.
 *
 * @param x - Binary image (nonzero is set).
 * @param options - Element, iterations, border value and origin.
 * @returns A 0/1 image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.binaryErosion([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])) // => [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
 * ```
 */
export function binaryErosion(x: TensorLike, options: BinaryOptions = {}): Tensor {
  return binaryOp(x, options, false, 'binaryErosion')
}

/**
 * Binary dilation (scipy.ndimage.binary_dilation): every cell reached by the reflected element from a set cell.
 *
 * @param x - Binary image.
 * @param options - Element, iterations, border value and origin.
 * @returns A 0/1 image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.binaryDilation([[0, 0, 0], [0, 1, 0], [0, 0, 0]])) // => [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
 * ```
 */
export function binaryDilation(x: TensorLike, options: BinaryOptions = {}): Tensor {
  return binaryOp(x, options, true, 'binaryDilation')
}

/**
 * Binary opening (scipy.ndimage.binary_opening): erosion then dilation, removing specks smaller than the element.
 *
 * @param x - Binary image.
 * @param options - Element, iterations, border value and origin.
 * @returns A 0/1 image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.binaryOpening([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], { structure: [[1, 1], [1, 1]] })) // => [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
 * ```
 */
export function binaryOpening(x: TensorLike, options: BinaryOptions = {}): Tensor {
  return binaryDilation(binaryErosion(x, options), options)
}

/**
 * Binary closing (scipy.ndimage.binary_closing): dilation then erosion, filling gaps smaller than the element.
 *
 * @param x - Binary image.
 * @param options - Element, iterations, border value and origin.
 * @returns A 0/1 image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.binaryClosing([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], { structure: [[1, 1, 1]] })) // => [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
 * ```
 */
export function binaryClosing(x: TensorLike, options: BinaryOptions = {}): Tensor {
  return binaryErosion(binaryDilation(x, options), options)
}

/** Options of the grey morphology: a box size or a flat footprint, border and origin. */
export interface GreyOptions {
  /** Box size per axis or shared (when no footprint is given). */
  readonly size?: number | readonly number[]
  /** Flat footprint (nonzero cells take part). */
  readonly footprint?: TensorLike
  /** Border mode (default 'reflect'). */
  readonly mode?: BorderMode
  readonly cval?: number
  readonly origin?: number | readonly number[]
}

function greyOp(x: TensorLike, options: GreyOptions, dilate: boolean, name: string): Tensor {
  const g = gridOf(x)
  const rank = g.shape.length
  let s: Structure
  if (options.footprint !== undefined) s = structureOf(options.footprint, rank, name)
  else {
    const size = options.size ?? 3
    const shape = typeof size === 'number' ? new Array<number>(rank).fill(size) : [...size]
    const total = shape.reduce((a, b) => a * b, 1)
    const offsets: number[][] = []
    const c = new Array<number>(rank).fill(0)
    for (let i = 0; i < total; i++) {
      coords(i, shape, c)
      offsets.push([...c])
    }
    s = { shape, offsets }
  }
  let origin = originsOf(options.origin, rank)
  if (dilate) {
    origin = flippedOrigins(origin, s.shape)
    s = reflect(s)
  }
  const mode = padModeOf(options.mode ?? 'reflect')
  const cval = options.cval ?? 0
  const out = new Float64Array(g.data.length)
  const c = new Array<number>(rank).fill(0)
  for (let p = 0; p < g.data.length; p++) {
    coords(p, g.shape, c)
    let acc = dilate ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY
    for (const off of s.offsets) {
      let idx = 0
      let outside = false
      for (let d = 0; d < rank; d++) {
        const i = extendIndex(
          (c[d] ?? 0) + (off[d] ?? 0) - Math.floor((s.shape[d] ?? 1) / 2) - (origin[d] ?? 0),
          g.shape[d] ?? 1,
          mode,
        )
        if (i < 0) {
          outside = true
          break
        }
        idx += i * (g.strides[d] ?? 1)
      }
      const v = outside ? cval : (g.data[idx] ?? 0)
      acc = dilate ? Math.max(acc, v) : Math.min(acc, v)
    }
    out[p] = acc
  }
  return fromStorage(out, [...g.shape], 'float64')
}

/**
 * Grey erosion (scipy.ndimage.grey_erosion): the minimum over a box or flat footprint.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The eroded image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.greyErosion([3, 1, 4, 1, 5], { size: 3 })) // => [1, 1, 1, 1, 1]
 * ```
 */
export function greyErosion(x: TensorLike, options: GreyOptions = {}): Tensor {
  return greyOp(x, options, false, 'greyErosion')
}

/**
 * Grey dilation (scipy.ndimage.grey_dilation): the maximum over the reflected box or footprint.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The dilated image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.greyDilation([3, 1, 4, 1, 5], { size: 3 })) // => [3, 4, 4, 5, 5]
 * ```
 */
export function greyDilation(x: TensorLike, options: GreyOptions = {}): Tensor {
  return greyOp(x, options, true, 'greyDilation')
}

/**
 * Grey opening (scipy.ndimage.grey_opening): erosion then dilation, removing bright details smaller than the element.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The opened image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.greyOpening([1, 1, 9, 1, 1], { size: 3 })) // => [1, 1, 1, 1, 1]
 * ```
 */
export function greyOpening(x: TensorLike, options: GreyOptions = {}): Tensor {
  return greyDilation(greyErosion(x, options), options)
}

/**
 * Grey closing (scipy.ndimage.grey_closing): dilation then erosion, removing dark details smaller than the element.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The closed image.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.greyClosing([5, 5, 0, 5, 5], { size: 3 })) // => [5, 5, 5, 5, 5]
 * ```
 */
export function greyClosing(x: TensorLike, options: GreyOptions = {}): Tensor {
  return greyErosion(greyDilation(x, options), options)
}

const subtract = (a: Tensor, b: Tensor): Tensor =>
  fromStorage(
    Float64Array.from(a.data, (v, i) => Number(v) - Number(b.data[i] ?? 0)),
    [...a.shape],
    'float64',
  )

/**
 * The morphological gradient (scipy.ndimage.morphological_gradient): dilation minus erosion, bright on edges.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The gradient.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.morphologicalGradient([0, 0, 1, 1], { size: 3 })) // => [0, 1, 1, 0]
 * ```
 */
export function morphologicalGradient(x: TensorLike, options: GreyOptions = {}): Tensor {
  return subtract(greyDilation(x, options), greyErosion(x, options))
}

/**
 * The white top-hat (scipy.ndimage.white_tophat): the image minus its opening, the bright details smaller than the
 * element.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The top-hat.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.whiteTophat([1, 1, 9, 1, 1], { size: 3 })) // => [0, 0, 8, 0, 0]
 * ```
 */
export function whiteTophat(x: TensorLike, options: GreyOptions = {}): Tensor {
  return subtract(primalOf(float(x)) as Tensor, greyOpening(x, options))
}

/**
 * The black top-hat (scipy.ndimage.black_tophat): the closing minus the image, the dark details smaller than the
 * element.
 *
 * @param x - Image.
 * @param options - Size or footprint, border and origin.
 * @returns The top-hat.
 * @concept mathematical-morphology
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.blackTophat([5, 5, 0, 5, 5], { size: 3 })) // => [0, 0, 5, 0, 0]
 * ```
 */
export function blackTophat(x: TensorLike, options: GreyOptions = {}): Tensor {
  return subtract(greyClosing(x, options), primalOf(float(x)) as Tensor)
}

/**
 * Connected-component labelling (scipy.ndimage.label): cells connected through the structuring element share a
 * label 1 .. count, numbered in raster order of their first cell; background is 0.
 *
 * @param x - Binary image.
 * @param structure - Connectivity (default the cross of connectivity 1).
 * @returns The labels and the number of components.
 * @concept connected-components
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const r = image.label([[1, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]])
 * [toArray(r.labels), r.count] // => [[[1, 1, 0, 2], [0, 0, 0, 2], [3, 0, 0, 0]], 3]
 * ```
 */
export function label(x: TensorLike, structure?: TensorLike): { readonly labels: Tensor; readonly count: number } {
  const g = gridOf(x)
  const rank = g.shape.length
  const s = structureOf(structure, rank, 'label')
  const parent = Int32Array.from({ length: g.data.length }, (_, i) => i)
  const find = (i: number): number => {
    let r = i
    while ((parent[r] ?? r) !== r) r = parent[r] ?? r
    let k = i
    while ((parent[k] ?? k) !== r) {
      const next = parent[k] ?? k
      parent[k] = r
      k = next
    }
    return r
  }
  const c = new Array<number>(rank).fill(0)
  for (let p = 0; p < g.data.length; p++) {
    if ((g.data[p] ?? 0) === 0) continue
    coords(p, g.shape, c)
    for (const off of s.offsets) {
      let idx = 0
      let outside = false
      for (let d = 0; d < rank; d++) {
        const i = (c[d] ?? 0) + (off[d] ?? 0) - Math.floor((s.shape[d] ?? 1) / 2)
        if (i < 0 || i >= (g.shape[d] ?? 0)) {
          outside = true
          break
        }
        idx += i * (g.strides[d] ?? 1)
      }
      if (outside || idx >= p || (g.data[idx] ?? 0) === 0) continue
      const a = find(p)
      const b = find(idx)
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b)
    }
  }
  const labels = new Float64Array(g.data.length)
  const ids = new Map<number, number>()
  for (let p = 0; p < g.data.length; p++) {
    if ((g.data[p] ?? 0) === 0) continue
    const r = find(p)
    let id = ids.get(r)
    if (id === undefined) {
      id = ids.size + 1
      ids.set(r, id)
    }
    labels[p] = id
  }
  return { labels: fromStorage(labels, [...g.shape], 'float64'), count: ids.size }
}

/** Squared distances to the nearest zero along one line (Felzenszwalb and Huttenlocher's lower envelope). */
function edt1d(f: Float64Array, spacing: number): Float64Array {
  const n = f.length
  const d = new Float64Array(n)
  const v = new Int32Array(n)
  const z = new Float64Array(n + 1)
  let k = 0
  let first = -1
  for (let q = 0; q < n; q++)
    if (Number.isFinite(f[q] ?? Number.POSITIVE_INFINITY)) {
      first = q
      break
    }
  if (first < 0) return d.fill(Number.POSITIVE_INFINITY)
  v[0] = first
  z[0] = Number.NEGATIVE_INFINITY
  z[1] = Number.POSITIVE_INFINITY
  const s2 = spacing * spacing
  for (let q = first + 1; q < n; q++) {
    const fq = f[q] ?? Number.POSITIVE_INFINITY
    if (!Number.isFinite(fq)) continue
    let s = 0
    for (;;) {
      const vk = v[k] ?? 0
      s = (fq + s2 * q * q - ((f[vk] ?? 0) + s2 * vk * vk)) / (2 * s2 * (q - vk))
      if (s <= (z[k] ?? 0) && k > 0) k--
      else break
    }
    k++
    v[k] = q
    z[k] = s
    z[k + 1] = Number.POSITIVE_INFINITY
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while ((z[k + 1] ?? 0) < q) k++
    const vk = v[k] ?? 0
    d[q] = s2 * (q - vk) * (q - vk) + (f[vk] ?? 0)
  }
  return d
}

/**
 * The exact Euclidean distance transform (scipy.ndimage.distance_transform_edt): for every nonzero cell, the
 * distance to the nearest zero cell, with per-axis sampling; separable lower envelopes of parabolas in O(n).
 *
 * @param x - Binary image (nonzero is foreground).
 * @param sampling - Grid spacing per axis or shared (default 1).
 * @returns The distances.
 * @concept distance-transform
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.distanceTransformEdt([[0, 1, 1, 1, 1]])) // => [[0, 1, 2, 3, 4]]
 * ```
 */
export function distanceTransformEdt(x: TensorLike, sampling: number | readonly number[] = 1): Tensor {
  const g = gridOf(x)
  const rank = g.shape.length
  const spacing = typeof sampling === 'number' ? new Array<number>(rank).fill(sampling) : [...sampling]
  let f = Float64Array.from(g.data, (v) => (v === 0 ? 0 : Number.POSITIVE_INFINITY))
  for (let d = 0; d < rank; d++) {
    const n = g.shape[d] ?? 1
    const stride = g.strides[d] ?? 1
    const next = new Float64Array(f.length)
    const line = new Float64Array(n)
    const c = new Array<number>(rank).fill(0)
    for (let p = 0; p < f.length; p++) {
      coords(p, g.shape, c)
      if ((c[d] ?? 0) !== 0) continue
      for (let i = 0; i < n; i++) line[i] = f[p + i * stride] ?? 0
      const r = edt1d(line, spacing[d] ?? 1)
      for (let i = 0; i < n; i++) next[p + i * stride] = r[i] ?? 0
    }
    f = next
  }
  return fromStorage(Float64Array.from(f, Math.sqrt), [...g.shape], 'float64')
}

function lut(fn: (bits: boolean[]) => boolean): Uint8Array {
  return Uint8Array.from({ length: 256 }, (_, n) =>
    fn(Array.from({ length: 9 }, (_, i) => ((n >> i) & 1) === 1)) ? 1 : 0,
  )
}

const G12 = lut((b) => {
  let g1 = 0
  for (const i of [0, 2, 4, 6]) if (!b[i] && (b[i + 1] === true || b[(i + 2) % 8] === true)) g1++
  let n1 = 0
  let n2 = 0
  for (const k of [1, 3, 5, 7]) {
    if (b[k] === true || b[k - 1] === true) n1++
    if (b[k] === true || b[(k + 1) % 8] === true) n2++
  }
  const m = Math.min(n1, n2)
  return g1 === 1 && (m === 2 || m === 3)
})
const G3 = lut((b) => !((b[1] === true || b[2] === true || !b[7]) && b[0] === true))
const G3P = lut((b) => !((b[5] === true || b[6] === true || !b[3]) && b[4] === true))

/**
 * Morphological thinning (Guo and Hall 1989, skimage.morphology.thin): two sub-iterations of deletions by lookup
 * tables until no cell changes (or maxIter passes), leaving one-pixel-wide, topology-preserving curves.
 *
 * @param x - Binary image `[H, W]`.
 * @param maxIter - Most passes (default until stable).
 * @returns A 0/1 image.
 * @concept skeletonization
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(image.thin([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])) // => [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
 * ```
 */
export function thin(x: TensorLike, maxIter: number = Number.POSITIVE_INFINITY): Tensor {
  const g = gridOf(x)
  if (g.shape.length !== 2) throw new OmaiError('shape', 'thin: images are [H, W]')
  const [h = 0, w = 0] = g.shape
  const skel = Uint8Array.from(g.data, (v) => (v !== 0 ? 1 : 0))
  // Neighbour weights (row-major 3 x 3): [[8, 4, 2], [16, 0, 1], [32, 64, 128]].
  const weights = [8, 4, 2, 16, 0, 1, 32, 64, 128]
  const at = (y: number, xx: number): number => (y < 0 || y >= h || xx < 0 || xx >= w ? 0 : (skel[y * w + xx] ?? 0))
  let count = skel.reduce((a, b) => a + b, 0)
  for (let it = 0; it < maxIter; it++) {
    const before = count
    for (const table of [G3, G3P]) {
      const code = new Uint8Array(h * w)
      for (let y = 0; y < h; y++)
        for (let xx = 0; xx < w; xx++) {
          let n = 0
          for (let k = 0; k < 9; k++) if (at(y + Math.floor(k / 3) - 1, xx + (k % 3) - 1) === 1) n += weights[k] ?? 0
          code[y * w + xx] = n
        }
      for (let i = 0; i < h * w; i++) {
        const n = code[i] ?? 0
        if ((G12[n] ?? 0) === 1 && (table[n] ?? 0) === 1) skel[i] = 0
      }
    }
    count = skel.reduce((a, b) => a + b, 0)
    if (count === before) break
  }
  return fromStorage(Float64Array.from(skel), [h, w], 'float64')
}

/**
 * The skeleton of a binary image by Zhang and Suen's parallel thinning (1984; skimage.morphology.skeletonize): two
 * sub-iterations remove boundary pixels with 2 to 6 neighbours, one 0-to-1 transition around them, and the
 * sub-iteration's corner conditions, until nothing changes.
 *
 * @param x - Binary image `[H, W]`.
 * @returns A 0/1 image.
 * @concept skeletonization
 * @example
 * ```ts
 * import { image } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * const r = toArray(image.skeletonize([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]])) as number[][]
 * r.flat().reduce((a, b) => a + b, 0) <= 5 // => true
 * ```
 */
export function skeletonize(x: TensorLike): Tensor {
  const g = gridOf(x)
  if (g.shape.length !== 2) throw new OmaiError('shape', 'skeletonize: images are [H, W]')
  const [h = 0, w = 0] = g.shape
  const s = Uint8Array.from(g.data, (v) => (v !== 0 ? 1 : 0))
  const at = (y: number, xx: number): number => (y < 0 || y >= h || xx < 0 || xx >= w ? 0 : (s[y * w + xx] ?? 0))
  for (let changed = true; changed; ) {
    changed = false
    for (const pass of [0, 1]) {
      const remove: number[] = []
      for (let y = 0; y < h; y++)
        for (let xx = 0; xx < w; xx++) {
          if (at(y, xx) === 0) continue
          // P2 .. P9 clockwise from north.
          const p = [
            at(y - 1, xx),
            at(y - 1, xx + 1),
            at(y, xx + 1),
            at(y + 1, xx + 1),
            at(y + 1, xx),
            at(y + 1, xx - 1),
            at(y, xx - 1),
            at(y - 1, xx - 1),
          ]
          const b = p.reduce((acc, v) => acc + v, 0)
          let a = 0
          for (let k = 0; k < 8; k++) if (p[k] === 0 && p[(k + 1) % 8] === 1) a++
          const p2 = p[0] ?? 0
          const p4 = p[2] ?? 0
          const p6 = p[4] ?? 0
          const p8 = p[6] ?? 0
          const c1 = pass === 0 ? p2 * p4 * p6 : p2 * p4 * p8
          const c2 = pass === 0 ? p4 * p6 * p8 : p2 * p6 * p8
          if (b >= 2 && b <= 6 && a === 1 && c1 === 0 && c2 === 0) remove.push(y * w + xx)
        }
      for (const i of remove) s[i] = 0
      if (remove.length > 0) changed = true
    }
  }
  return fromStorage(Float64Array.from(s), [h, w], 'float64')
}
