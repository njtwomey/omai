/**
 * Image patches and pixel graphs, scikit-learn's `feature_extraction.image`: `extractPatches2d` cuts every
 * (or a random sample of) overlapping window of an image, `reconstructFromPatches2d` averages them back,
 * `patchExtractor` does the cutting for a batch of images as a transformer, and `imgToGraph` is the adjacency of the
 * pixel grid weighted by intensity differences, for spectral clustering of images. Images are `[H, W]` or `[H, W, C]`.
 */

import { OmaiError } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import { type Graph, graph as makeGraph } from '../graph/index.ts'
import { defineEstimator, type Estimator, type Fitted } from '../learn/index.ts'
import { foldIn, type Key, randint } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Raster, rasterOf, valuesOf } from './_image.ts'

type Size2 = readonly [height: number, width: number]

const sizeCheck = (who: string, name: string, [a, b]: Size2): void => {
  if (!(Number.isInteger(a) && Number.isInteger(b) && a >= 1 && b >= 1))
    throw new OmaiError('value', `${who}: ${name} must be two positive integers, got [${a}, ${b}]`)
}

/** How many patches a request asks for: all of them, a count, or a fraction of all. */
function patchCount(who: string, all: number, maxPatches: number | undefined): number | undefined {
  if (maxPatches === undefined) return undefined
  if (Number.isInteger(maxPatches) && maxPatches >= 1) {
    if (maxPatches > all)
      throw new OmaiError('value', `${who}: maxPatches ${maxPatches} is more than the ${all} patches`)
    return maxPatches
  }
  if (maxPatches > 0 && maxPatches < 1) return Math.floor(all * maxPatches)
  throw new OmaiError(
    'value',
    `${who}: maxPatches must be a positive integer or a fraction in (0, 1), got ${maxPatches}`,
  )
}

/** The top-left corners of the patches to take, in the order they are taken. */
function corners(
  who: string,
  r: Raster,
  [ph, pw]: Size2,
  options: { readonly maxPatches?: number; readonly key?: Key },
): [number, number][] {
  const rows = r.h - ph + 1
  const cols = r.w - pw + 1
  const out: [number, number][] = []
  const count = patchCount(who, rows * cols, options.maxPatches)
  if (count === undefined) {
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out.push([i, j])
    return out
  }
  if (options.key === undefined) throw new OmaiError('value', `${who}: a random sample of patches needs a key`)
  const [ki, kj] = [foldIn(options.key, 0), foldIn(options.key, 1)]
  const is = randint(ki, [count], 0, rows).data
  const js = randint(kj, [count], 0, cols).data
  for (let p = 0; p < count; p++) out.push([is[p] ?? 0, js[p] ?? 0])
  return out
}

function cut(r: Raster, [ph, pw]: Size2, at: readonly (readonly [number, number])[]): Float64Array {
  const out = new Float64Array(at.length * ph * pw * r.c)
  let o = 0
  for (const [i, j] of at)
    for (let y = 0; y < ph; y++)
      for (let x = 0; x < pw; x++)
        for (let c = 0; c < r.c; c++) out[o++] = r.data[((i + y) * r.w + (j + x)) * r.c + c] ?? 0
  return out
}

const patchShape = (r: Raster, [ph, pw]: Size2, n: number): number[] => (r.rank === 2 ? [n, ph, pw] : [n, ph, pw, r.c])

/**
 * The patches of an image (sklearn.feature_extraction.image.extract_patches_2d): every window of `patchSize` at a stride
 * of one, in raster order of its top-left corner, or with `maxPatches` that many windows at random corners (drawn with
 * replacement, as scikit-learn does; a fraction in (0, 1) means that share of all windows).
 *
 * @param image - The image `[H, W]` or `[H, W, C]`.
 * @param patchSize - The window `[height, width]`, no larger than the image.
 * @param options - `maxPatches` (a count or a fraction) and the `key` it draws from.
 * @returns The patches `[n, ph, pw]` or `[n, ph, pw, C]`.
 * @throws OmaiError with code `value` for a patch larger than the image, a bad `maxPatches`, or a random sample without a key.
 * @index function Every overlapping window of an image, or a random sample of them.
 * @example
 * ```ts
 * import { extractPatches2d } from 'omai-engine/vision'
 * extractPatches2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [2, 2]).shape // => [4, 2, 2]
 * ```
 */
export function extractPatches2d(
  image: TensorLike,
  patchSize: Size2,
  options: { readonly maxPatches?: number; readonly key?: Key } = {},
): Tensor {
  const who = 'extractPatches2d'
  sizeCheck(who, 'patchSize', patchSize)
  const r = rasterOf(image, who)
  if (patchSize[0] > r.h || patchSize[1] > r.w)
    throw new OmaiError(
      'value',
      `${who}: a ${patchSize[0]} x ${patchSize[1]} patch does not fit in a ${r.h} x ${r.w} image`,
    )
  const at = corners(who, r, patchSize, options)
  return fromStorage(cut(r, patchSize, at), patchShape(r, patchSize, at.length), 'float64')
}

/** How many windows of `size` starting at `0..starts - 1` cover position `at`. */
const cover = (at: number, size: number, starts: number): number =>
  Math.min(at, starts - 1) - Math.max(0, at - size + 1) + 1

/**
 * Rebuild an image from its overlapping patches (sklearn.feature_extraction.image.reconstruct_from_patches_2d): each
 * pixel is the mean of the patches that cover it (counting the windows that really cover it, which differs from
 * scikit-learn's formula when a patch is more than half the image). The patches must be all the windows of the image in raster order, as
 * `extractPatches2d` returns them without `maxPatches`, so `reconstructFromPatches2d(extractPatches2d(x, s), size)`
 * is `x`.
 *
 * @param patches - The patches `[n, ph, pw]` or `[n, ph, pw, C]`, one per window.
 * @param imageSize - The image `[height, width]`.
 * @returns The image `[H, W]` or `[H, W, C]`.
 * @throws OmaiError with code `shape` unless the number of patches is that of the windows of the image.
 * @index function Average overlapping patches back into an image.
 * @example
 * ```ts
 * import { extractPatches2d, reconstructFromPatches2d } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * const image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
 * toArray(reconstructFromPatches2d(extractPatches2d(image, [2, 2]), [3, 3])) // => image
 * ```
 */
export function reconstructFromPatches2d(patches: TensorLike, imageSize: Size2): Tensor {
  const who = 'reconstructFromPatches2d'
  sizeCheck(who, 'imageSize', imageSize)
  const p = patchesOf(patches, who)
  const [ih, iw] = imageSize
  const rows = ih - p.ph + 1
  const cols = iw - p.pw + 1
  if (rows < 1 || cols < 1 || p.n !== rows * cols)
    throw new OmaiError(
      'shape',
      `${who}: ${p.n} patches of ${p.ph} x ${p.pw} do not tile a ${ih} x ${iw} image (${rows * cols} windows)`,
    )
  const out = new Float64Array(ih * iw * p.c)
  for (let k = 0; k < p.n; k++) {
    const i = Math.floor(k / cols)
    const j = k % cols
    for (let y = 0; y < p.ph; y++)
      for (let x = 0; x < p.pw; x++)
        for (let c = 0; c < p.c; c++)
          out[((i + y) * iw + (j + x)) * p.c + c] =
            (out[((i + y) * iw + (j + x)) * p.c + c] ?? 0) + (p.data[((k * p.ph + y) * p.pw + x) * p.c + c] ?? 0)
  }
  for (let y = 0; y < ih; y++)
    for (let x = 0; x < iw; x++) {
      const covers = cover(y, p.ph, rows) * cover(x, p.pw, cols)
      for (let c = 0; c < p.c; c++) out[(y * iw + x) * p.c + c] = (out[(y * iw + x) * p.c + c] ?? 0) / covers
    }
  return fromStorage(out, p.rank === 3 ? [ih, iw] : [ih, iw, p.c], 'float64')
}

function patchesOf(
  patches: TensorLike,
  who: string,
): { n: number; ph: number; pw: number; c: number; rank: 3 | 4; data: Float64Array } {
  const t = valuesOf(patches)
  if (t.shape.length !== 3 && t.shape.length !== 4)
    throw new OmaiError('shape', `${who}: patches are [n, ph, pw] or [n, ph, pw, C]`)
  const [n = 0, ph = 0, pw = 0, c = 1] = t.shape
  return { n, ph, pw, c, rank: t.shape.length as 3 | 4, data: t.data }
}

/** Hyperparameters of {@link patchExtractor}. */
export interface PatchExtractorHyper {
  /** The window `[height, width]` (default a tenth of the height and width of the first image, as scikit-learn). */
  readonly patchSize?: Size2
  /** Patches per image: a count or a fraction of all windows (default all). */
  readonly maxPatches?: number
}

/** The fitted parameters of {@link patchExtractor}: the key its random corners are drawn from. */
export type PatchExtractorParams = { readonly keyWords: number[] }

/** The verb of a fitted patch extractor. */
export interface PatchExtractorVerbs {
  /** The patches of every image, stacked: `[n images * patches per image, ph, pw(, C)]`. */
  readonly transform: (images: TensorLike) => Tensor
}

/**
 * Patches of a batch of images as a transformer (sklearn.feature_extraction.image.PatchExtractor): `transform` takes
 * images `[n, H, W]` or `[n, H, W, C]` and returns the patches of each, one image after the other. With `maxPatches`
 * the corners are random, drawn from the key given to `fit` (a key per image), so the same fit gives the same patches.
 *
 * @param hyper - `patchSize` and `maxPatches`.
 * @returns The estimator (stateless apart from its key).
 * @index estimator Cut the patches of a batch of images, all windows or random ones.
 * @example
 * ```ts
 * import { patchExtractor } from 'omai-engine/vision'
 * const images = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
 * patchExtractor({ patchSize: [2, 2] }).fit({ x: images }).transform(images).shape // => [8, 2, 2]
 * ```
 */
export function patchExtractor(
  hyper: PatchExtractorHyper = {},
): Estimator<PatchExtractorHyper, Fitted<PatchExtractorParams> & PatchExtractorVerbs> {
  return defineEstimator<PatchExtractorHyper, PatchExtractorParams, PatchExtractorVerbs>(
    {
      name: 'patch-extractor',
      kind: 'transformer',
      fit: (_data, _h, options) => ({ keyWords: [...resolveKey(options, 'patchExtractor').words] }),
      methods: (params, h) => ({
        transform: (images) => {
          const who = 'patchExtractor'
          const t = valuesOf(images)
          if (t.shape.length !== 3 && t.shape.length !== 4)
            throw new OmaiError('shape', `${who}: images are [n, H, W] or [n, H, W, C]`)
          const [n = 0, ih = 0, iw = 0, c = 1] = t.shape
          const size: Size2 = h.patchSize ?? [Math.floor(ih / 10), Math.floor(iw / 10)]
          sizeCheck(who, 'patchSize', size)
          const key: Key = { kind: 'key', words: [params.keyWords[0] ?? 0, params.keyWords[1] ?? 0] }
          const per = ih * iw * c
          const parts: Float64Array[] = []
          let total = 0
          for (let i = 0; i < n; i++) {
            const r: Raster = {
              h: ih,
              w: iw,
              c,
              rank: t.shape.length === 3 ? 2 : 3,
              data: t.data.subarray(i * per, (i + 1) * per),
            }
            if (size[0] > ih || size[1] > iw) throw new OmaiError('value', `${who}: a patch does not fit in the image`)
            const at = corners(who, r, size, {
              ...(h.maxPatches === undefined ? {} : { maxPatches: h.maxPatches }),
              key: foldIn(key, i),
            })
            parts.push(cut(r, size, at))
            total += at.length
          }
          const out = new Float64Array(parts.reduce((a, q) => a + q.length, 0))
          let o = 0
          for (const q of parts) {
            out.set(q, o)
            o += q.length
          }
          return fromStorage(
            out,
            t.shape.length === 3 ? [total, size[0], size[1]] : [total, size[0], size[1], c],
            'float64',
          )
        },
      }),
    },
    hyper,
  )
}

/** What {@link imgToGraph} returns: the weighted pixel graph and the values scikit-learn puts on the diagonal. */
export interface ImageGraph {
  /** The pixels as nodes `0..n-1` (raster order; only those inside the mask), joined to their grid neighbours, weighted by the absolute difference of their values. */
  readonly graph: Graph
  /** The value of each node (the diagonal of scikit-learn's matrix). */
  readonly values: number[]
}

/**
 * The graph of the pixel grid of an image (sklearn.feature_extraction.image.img_to_graph): each pixel is joined to its
 * neighbours along every axis (4-connected for `[H, W]`; a third axis is a spatial one, as in scikit-learn), the edge
 * weighted by the absolute difference of the two values, for spectral clustering of images. An optional boolean mask
 * keeps only some pixels, and the edges between them.
 *
 * @param image - The image `[H, W]` or a volume `[X, Y, Z]`.
 * @param options - `mask`: a boolean array of the same shape (nonzero = keep).
 * @returns The graph and the node values.
 * @throws OmaiError with code `shape` for a mask of another shape.
 * @index function The pixel-grid graph of an image weighted by intensity differences.
 * @example
 * ```ts
 * import { imgToGraph } from 'omai-engine/vision'
 * const { graph, values } = imgToGraph([[1, 3], [2, 7]])
 * graph.edges.map((e) => [e.source, e.target, e.weight]) // => [[0, 1, 2], [0, 2, 1], [1, 3, 4], [2, 3, 5]]
 * values // => [1, 3, 2, 7]
 * ```
 */
export function imgToGraph(image: TensorLike, options: { readonly mask?: TensorLike } = {}): ImageGraph {
  const who = 'imgToGraph'
  const t = valuesOf(image)
  if (t.shape.length !== 2 && t.shape.length !== 3)
    throw new OmaiError('shape', `${who}: the image is [H, W] or [X, Y, Z]`)
  const [nx = 0, ny = 0, nz = 1] = t.shape
  const total = nx * ny * nz
  let kept: number[] = Array.from({ length: total }, (_, v) => v)
  if (options.mask !== undefined) {
    const m = valuesOf(options.mask)
    if (m.data.length !== total)
      throw new OmaiError('shape', `${who}: the mask has ${m.data.length} entries for ${total} pixels`)
    kept = kept.filter((v) => (m.data[v] ?? 0) !== 0)
  }
  const order = new Map(kept.map((v, i) => [v, i]))
  const edges: [number, number, number][] = []
  const at = (x: number, y: number, z: number): number => (x * ny + y) * nz + z
  const push = (a: number, b: number): void => {
    const u = order.get(a)
    const v = order.get(b)
    if (u !== undefined && v !== undefined) edges.push([u, v, Math.abs((t.data[a] ?? 0) - (t.data[b] ?? 0))])
  }
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        if (z + 1 < nz) push(at(x, y, z), at(x, y, z + 1))
        if (y + 1 < ny) push(at(x, y, z), at(x, y + 1, z))
        if (x + 1 < nx) push(at(x, y, z), at(x + 1, y, z))
      }
  return { graph: makeGraph(edges, { nodes: kept.length }), values: kept.map((v) => t.data[v] ?? 0) }
}
