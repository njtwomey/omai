/**
 * Axis-aligned boxes for detection, torchvision.ops and OpenCV dnn style: the xyxy, xywh and cxcywh formats and their
 * conversions, areas, clipping, scaling and small-box removal; non-maximum suppression (hard, per-class batched, and
 * Bodla's soft NMS as OpenCV runs it); anchors on a feature-map grid, the Faster R-CNN box encoding and its inverse,
 * and the IoU matcher of anchors to targets. The pairwise IoU, GIoU, DIoU and CIoU matrices are `metrics`'
 * `boxIou` family.
 */

import { MdmlError } from '../base/index.ts'
import { boxIou } from '../metrics/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { f32, roundHalfEven, valuesOf } from './_image.ts'

/** A box as [a, b, c, d] in some {@link BoxFormat}. */
export type Box = [number, number, number, number]

/** Box formats: corners (x1, y1, x2, y2), corner and size (x, y, w, h), centre and size (cx, cy, w, h). */
export type BoxFormat = 'xyxy' | 'xywh' | 'cxcywh'

/** Boxes as rows `[n, 4]`. */
export type BoxesLike = TensorLike | readonly (readonly number[])[]

/**
 * Boxes as rows of four numbers, from a tensor `[n, 4]` (or a single box `[4]`) or nested arrays.
 *
 * @param boxes - The boxes.
 * @param name - The caller's name for error messages.
 * @returns The rows.
 * @throws MdmlError with code `shape` when the boxes are not `[n, 4]`.
 * @concept bounding-box
 * @example
 * ```ts
 * import { boxesOf } from 'mdml-engine/vision'
 * boxesOf([1, 2, 3, 4]) // => [[1, 2, 3, 4]]
 * ```
 */
export function boxesOf(boxes: BoxesLike, name = 'boxes'): Box[] {
  if (Array.isArray(boxes) && boxes.length === 0) return []
  const { data, shape } = valuesOf(boxes as TensorLike)
  const n = shape.length === 1 ? (data.length === 0 ? 0 : 1) : (shape[0] ?? 0)
  if (data.length !== 4 * n || (shape.length === 2 && shape[1] !== 4))
    throw new MdmlError('shape', `${name}: boxes are [n, 4]`)
  return Array.from({ length: n }, (_, i) => [
    data[4 * i] ?? 0,
    data[4 * i + 1] ?? 0,
    data[4 * i + 2] ?? 0,
    data[4 * i + 3] ?? 0,
  ])
}

const toXyxy = (b: Box, from: BoxFormat): Box => {
  const [a, c, d, e] = b
  if (from === 'xyxy') return [a, c, d, e]
  if (from === 'xywh') return [a, c, a + d, c + e]
  return [a - 0.5 * d, c - 0.5 * e, a + 0.5 * d, c + 0.5 * e]
}

const fromXyxy = (b: Box, to: BoxFormat): Box => {
  const [x1, y1, x2, y2] = b
  if (to === 'xyxy') return [x1, y1, x2, y2]
  if (to === 'xywh') return [x1, y1, x2 - x1, y2 - y1]
  return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
}

/**
 * Convert boxes between formats (torchvision.ops.box_convert).
 *
 * @param boxes - Boxes `[n, 4]`.
 * @param from - Their format.
 * @param to - The wanted format.
 * @returns The converted boxes.
 * @concept bounding-box
 * @example
 * ```ts
 * import { boxConvert } from 'mdml-engine/vision'
 * boxConvert([[10, 20, 30, 60]], 'xyxy', 'cxcywh') // => [[20, 40, 20, 40]]
 * ```
 */
export function boxConvert(boxes: BoxesLike, from: BoxFormat, to: BoxFormat): Box[] {
  for (const f of [from, to])
    if (f !== 'xyxy' && f !== 'xywh' && f !== 'cxcywh') throw new MdmlError('value', `boxConvert: unknown format ${f}`)
  return boxesOf(boxes, 'boxConvert').map((b) => fromXyxy(toXyxy(b, from), to))
}

/**
 * Box areas (torchvision.ops.box_area): (x2 - x1)(y2 - y1) of xyxy boxes.
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @returns The areas.
 * @concept bounding-box
 * @example
 * ```ts
 * import { boxArea } from 'mdml-engine/vision'
 * boxArea([[0, 0, 2, 3], [1, 1, 2, 2]]) // => [6, 1]
 * ```
 */
export function boxArea(boxes: BoxesLike): number[] {
  return boxesOf(boxes, 'boxArea').map(([x1, y1, x2, y2]) => (x2 - x1) * (y2 - y1))
}

/**
 * Clip xyxy boxes to an image (torchvision.ops.clip_boxes_to_image): x into [0, width], y into [0, height].
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param size - The image size [height, width].
 * @returns The clipped boxes.
 * @concept bounding-box
 * @example
 * ```ts
 * import { clipBoxes } from 'mdml-engine/vision'
 * clipBoxes([[-5, 2, 30, 12]], [10, 20]) // => [[0, 2, 20, 10]]
 * ```
 */
export function clipBoxes(boxes: BoxesLike, size: readonly [number, number]): Box[] {
  const [h, w] = size
  const cx = (v: number): number => Math.min(w, Math.max(0, v))
  const cy = (v: number): number => Math.min(h, Math.max(0, v))
  return boxesOf(boxes, 'clipBoxes').map(([x1, y1, x2, y2]) => [cx(x1), cy(y1), cx(x2), cy(y2)])
}

/**
 * Scale xyxy boxes, as when resizing the image they belong to.
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param sx - The horizontal factor.
 * @param sy - The vertical factor (default sx).
 * @returns The scaled boxes.
 * @concept bounding-box
 * @example
 * ```ts
 * import { scaleBoxes } from 'mdml-engine/vision'
 * scaleBoxes([[1, 2, 3, 4]], 2, 0.5) // => [[2, 1, 6, 2]]
 * ```
 */
export function scaleBoxes(boxes: BoxesLike, sx: number, sy: number = sx): Box[] {
  return boxesOf(boxes, 'scaleBoxes').map(([x1, y1, x2, y2]) => [x1 * sx, y1 * sy, x2 * sx, y2 * sy])
}

/**
 * The boxes with both sides at least minSize (torchvision.ops.remove_small_boxes).
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param minSize - The smallest side kept.
 * @returns Indices of the kept boxes.
 * @concept bounding-box
 * @example
 * ```ts
 * import { removeSmallBoxes } from 'mdml-engine/vision'
 * removeSmallBoxes([[0, 0, 1, 5], [0, 0, 4, 4]], 2) // => [1]
 * ```
 */
export function removeSmallBoxes(boxes: BoxesLike, minSize: number): number[] {
  return boxesOf(boxes, 'removeSmallBoxes').flatMap(([x1, y1, x2, y2], i) =>
    x2 - x1 >= minSize && y2 - y1 >= minSize ? [i] : [],
  )
}

/** Indices sorted by decreasing score (stable, so equal scores keep their order). */
const byScore = (scores: readonly number[]): number[] =>
  scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0) || a - b)

/**
 * Hard non-maximum suppression (torchvision.ops.nms): take boxes in decreasing score, discarding every remaining box
 * whose IoU with a kept one exceeds the threshold.
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param scores - Scores `[n]`.
 * @param iouThreshold - The IoU above which a box is suppressed.
 * @returns Indices of the kept boxes, by decreasing score.
 * @concept non-maximum-suppression
 * @example
 * ```ts
 * import { nms } from 'mdml-engine/vision'
 * nms([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], [0.9, 0.8, 0.7], 0.5) // => [0, 2]
 * ```
 */
export function nms(boxes: BoxesLike, scores: TensorLike, iouThreshold: number): number[] {
  const b = boxesOf(boxes, 'nms')
  const s = Array.from(valuesOf(scores).data)
  if (s.length !== b.length) throw new MdmlError('shape', 'nms: one score per box')
  if (b.length === 0) return []
  const iou = boxIou(b, b)
  const removed = new Uint8Array(b.length)
  const keep: number[] = []
  for (const i of byScore(s)) {
    if (removed[i] === 1) continue
    keep.push(i)
    const row = iou[i] as number[]
    for (let j = 0; j < b.length; j++) if ((row[j] ?? 0) > iouThreshold) removed[j] = 1
  }
  return keep
}

/**
 * Non-maximum suppression within each class (torchvision.ops.batched_nms): boxes of different classes never suppress
 * each other.
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param scores - Scores `[n]`.
 * @param classes - A class index per box.
 * @param iouThreshold - The IoU above which a box is suppressed.
 * @returns Indices of the kept boxes, by decreasing score.
 * @concept non-maximum-suppression
 * @example
 * ```ts
 * import { batchedNms } from 'mdml-engine/vision'
 * batchedNms([[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8], [0, 1], 0.5) // => [0, 1]
 * ```
 */
export function batchedNms(
  boxes: BoxesLike,
  scores: TensorLike,
  classes: readonly number[],
  iouThreshold: number,
): number[] {
  const b = boxesOf(boxes, 'batchedNms')
  const s = Array.from(valuesOf(scores).data)
  if (classes.length !== b.length || s.length !== b.length)
    throw new MdmlError('shape', 'batchedNms: one score and class per box')
  const kept = new Set<number>()
  for (const c of new Set(classes)) {
    const idx = classes.flatMap((k, i) => (k === c ? [i] : []))
    for (const j of nms(
      idx.map((i) => b[i] as Box),
      idx.map((i) => s[i] ?? 0),
      iouThreshold,
    ))
      kept.add(idx[j] ?? 0)
  }
  return byScore(s).filter((i) => kept.has(i))
}

/** Options of {@link softNms}. */
export interface SoftNmsOptions {
  /** 'gaussian' (s * exp(-iou^2 / sigma), default) or 'linear' (s * (1 - iou) above the IoU threshold). */
  readonly method?: 'gaussian' | 'linear'
  /** The Gaussian's sigma (default 0.5). */
  readonly sigma?: number
  /** The IoU above which the linear method decays a score (default 0.3). */
  readonly iouThreshold?: number
  /** Boxes whose score falls below this are dropped (default 0.001). */
  readonly scoreThreshold?: number
  /** The most boxes kept (default all). */
  readonly topK?: number
}

/**
 * Soft non-maximum suppression (Bodla et al. 2017; cv2.dnn.softNMSBoxes): repeatedly keep the highest-scoring box and
 * decay the scores of the others by their overlap with it, rather than discarding them, until every score is below
 * the threshold. Scores are decayed in single precision, as OpenCV does.
 *
 * @param boxes - Boxes `[n, 4]` as xyxy.
 * @param scores - Scores `[n]`.
 * @param options - Method, sigma, thresholds and topK.
 * @returns Kept indices in selection order and their decayed scores.
 * @concept non-maximum-suppression
 * @example
 * ```ts
 * import { softNms } from 'mdml-engine/vision'
 * softNms([[0, 0, 10, 10], [0, 0, 10, 5]], [0.9, 0.8], { method: 'linear' }) // => { indices: [0, 1], scores: [0.8999999761581421, 0.4000000059604645] }
 * ```
 */
export function softNms(
  boxes: BoxesLike,
  scores: TensorLike,
  options: SoftNmsOptions = {},
): { readonly indices: number[]; readonly scores: number[] } {
  const b = boxesOf(boxes, 'softNms')
  const s = Array.from(valuesOf(scores).data, f32)
  if (s.length !== b.length) throw new MdmlError('shape', 'softNms: one score per box')
  const method = options.method ?? 'gaussian'
  const sigma = f32(options.sigma ?? 0.5)
  const thr = f32(options.iouThreshold ?? 0.3)
  const scoreThr = f32(options.scoreThreshold ?? 0.001)
  const topK = options.topK === undefined || options.topK === 0 ? b.length : Math.min(options.topK, b.length)
  const iou = boxIou(b, b)
  const order = s.map((v, i): [number, number] => [v, i])
  const indices: number[] = []
  const kept: number[] = []
  let start = 0
  while (indices.length < topK && start < order.length) {
    let best = start
    for (let i = start + 1; i < order.length; i++) {
      const [v, k] = order[i] as [number, number]
      const [bv, bk] = order[best] as [number, number]
      if (v > bv || (v === bv && k < bk)) best = i
    }
    const [bscore, bidx] = order[best] as [number, number]
    if (bscore < scoreThr) break
    indices.push(bidx)
    kept.push(bscore)
    ;[order[start], order[best]] = [order[best] as [number, number], order[start] as [number, number]]
    for (let i = start + 1; i < order.length; i++) {
      const e = order[i] as [number, number]
      if (e[0] < scoreThr) continue
      // OpenCV's overlap: 1 - (float)(1 - IoU), the Jaccard distance taken through single precision.
      const overlap = f32(1 - f32(1 - (iou[bidx]?.[e[1]] ?? 0)))
      if (method === 'linear') {
        if (overlap > thr) e[0] = f32(e[0] * f32(1 - overlap))
      } else e[0] = f32(e[0] * f32(Math.exp(f32(-f32(overlap * overlap) / sigma))))
    }
    start++
  }
  return { indices, scores: kept }
}

/**
 * The zero-centred anchors of one feature-map cell (torchvision's AnchorGenerator.generate_anchors): for every
 * aspect ratio r (height over width) and size s, a box of width s / sqrt(r) and height s sqrt(r), rounded.
 *
 * @param sizes - Anchor sizes (the square root of the area).
 * @param aspectRatios - Height-to-width ratios.
 * @returns Anchors `[ratios x sizes, 4]` as xyxy, ratio-major.
 * @concept anchor-boxes
 * @example
 * ```ts
 * import { generateAnchors } from 'mdml-engine/vision'
 * generateAnchors([32], [0.5, 1, 2]) // => [[-23, -11, 23, 11], [-16, -16, 16, 16], [-11, -23, 11, 23]]
 * ```
 */
export function generateAnchors(sizes: readonly number[], aspectRatios: readonly number[]): Box[] {
  const out: Box[] = []
  for (const r of aspectRatios) {
    const hr = f32(Math.sqrt(r))
    const wr = f32(1 / hr)
    for (const s of sizes) {
      const ws = f32(wr * s)
      const hs = f32(hr * s)
      out.push([roundHalfEven(-ws / 2), roundHalfEven(-hs / 2), roundHalfEven(ws / 2), roundHalfEven(hs / 2)])
    }
  }
  return out
}

/**
 * Anchors over a feature-map grid (torchvision's AnchorGenerator.grid_anchors): every cell's base anchors shifted by
 * the cell's offset (column times the stride, row times the stride), cells in row-major order.
 *
 * @param base - Zero-centred anchors `[a, 4]` (from {@link generateAnchors}).
 * @param gridSize - The feature map [height, width].
 * @param stride - The stride [vertical, horizontal] in image pixels.
 * @returns Anchors `[height x width x a, 4]` as xyxy.
 * @concept anchor-boxes
 * @example
 * ```ts
 * import { gridAnchors } from 'mdml-engine/vision'
 * gridAnchors([[-1, -1, 1, 1]], [1, 2], [8, 8]) // => [[-1, -1, 1, 1], [7, -1, 9, 1]]
 * ```
 */
export function gridAnchors(
  base: BoxesLike,
  gridSize: readonly [number, number],
  stride: readonly [number, number],
): Box[] {
  const b = boxesOf(base, 'gridAnchors')
  const [gh, gw] = gridSize
  const [sh, sw] = stride
  const out: Box[] = []
  for (let r = 0; r < gh; r++)
    for (let c = 0; c < gw; c++) {
      const x = c * sw
      const y = r * sh
      for (const [x1, y1, x2, y2] of b) out.push([x + x1, y + y1, x + x2, y + y2])
    }
  return out
}

/** Options of the box coder: the weights of (dx, dy, dw, dh) and the clip on the decoded log sizes. */
export interface BoxCoderOptions {
  /** Weights [wx, wy, ww, wh] (default [1, 1, 1, 1]; Faster R-CNN's heads use [10, 10, 5, 5]). */
  readonly weights?: readonly [number, number, number, number]
  /** The largest decoded dw and dh (default log(1000 / 16), torchvision's). */
  readonly clip?: number
}

/**
 * Encode target boxes relative to reference boxes (torchvision's BoxCoder.encode, the Faster R-CNN
 * parameterisation): dx = wx (cx - cx_ref) / w_ref, dy likewise, dw = ww log(w / w_ref), dh likewise.
 *
 * @param targets - Target boxes `[n, 4]` as xyxy.
 * @param references - Reference (anchor or proposal) boxes `[n, 4]` as xyxy.
 * @param options - The weights.
 * @returns The regression targets `[n, 4]`.
 * @concept box-encoding
 * @example
 * ```ts
 * import { encodeBoxes } from 'mdml-engine/vision'
 * encodeBoxes([[1, 0, 11, 20]], [[0, 0, 10, 10]]) // => [[0.1, 0.5, 0, 0.6931471805599453]]
 * ```
 */
export function encodeBoxes(targets: BoxesLike, references: BoxesLike, options: BoxCoderOptions = {}): Box[] {
  const t = boxesOf(targets, 'encodeBoxes')
  const r = boxesOf(references, 'encodeBoxes')
  if (t.length !== r.length) throw new MdmlError('shape', 'encodeBoxes: one reference per target')
  const [wx, wy, ww, wh] = options.weights ?? [1, 1, 1, 1]
  return t.map((g, i) => {
    const e = r[i] as Box
    const ew = e[2] - e[0]
    const eh = e[3] - e[1]
    const ex = e[0] + 0.5 * ew
    const ey = e[1] + 0.5 * eh
    const gw = g[2] - g[0]
    const gh = g[3] - g[1]
    const gx = g[0] + 0.5 * gw
    const gy = g[1] + 0.5 * gh
    return [(wx * (gx - ex)) / ew, (wy * (gy - ey)) / eh, ww * Math.log(gw / ew), wh * Math.log(gh / eh)]
  })
}

/**
 * Decode regression outputs into boxes (torchvision's BoxCoder.decode_single), the inverse of {@link encodeBoxes},
 * with dw and dh clipped so exp does not overflow.
 *
 * @param codes - Regression outputs `[n, 4]`.
 * @param references - Reference boxes `[n, 4]` as xyxy.
 * @param options - The weights and clip.
 * @returns The decoded boxes `[n, 4]` as xyxy.
 * @concept box-encoding
 * @example
 * ```ts
 * import { decodeBoxes } from 'mdml-engine/vision'
 * decodeBoxes([[0.1, 0.5, 0, 0.6931471805599453]], [[0, 0, 10, 10]]).map((b) => b.map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, 0, 11, 20]]
 * ```
 */
export function decodeBoxes(codes: BoxesLike, references: BoxesLike, options: BoxCoderOptions = {}): Box[] {
  const d = boxesOf(codes, 'decodeBoxes')
  const r = boxesOf(references, 'decodeBoxes')
  if (d.length !== r.length) throw new MdmlError('shape', 'decodeBoxes: one reference per code')
  const [wx, wy, ww, wh] = options.weights ?? [1, 1, 1, 1]
  const clip = options.clip ?? Math.log(1000 / 16)
  return d.map((c, i) => {
    const e = r[i] as Box
    const w = e[2] - e[0]
    const h = e[3] - e[1]
    const cx = e[0] + 0.5 * w
    const cy = e[1] + 0.5 * h
    const px = (c[0] / wx) * w + cx
    const py = (c[1] / wy) * h + cy
    const pw = Math.exp(Math.min(c[2] / ww, clip)) * w
    const ph = Math.exp(Math.min(c[3] / wh, clip)) * h
    return [px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph]
  })
}

/** The match of {@link matchAnchors} for an anchor below the low threshold. */
export const BELOW_LOW_THRESHOLD = -1
/** The match of {@link matchAnchors} for an anchor between the thresholds. */
export const BETWEEN_THRESHOLDS = -2

/**
 * Match anchors to targets by IoU (torchvision's detection Matcher): each anchor takes its best target when that IoU
 * reaches the high threshold, -1 below the low threshold and -2 in between; with allowLowQuality, every target's
 * best anchors (ties included) keep their best target whatever the IoU.
 *
 * @param targets - Target boxes `[m, 4]` as xyxy.
 * @param anchors - Anchor boxes `[n, 4]` as xyxy.
 * @param options - The thresholds (defaults 0.7 and 0.3) and allowLowQuality (default false).
 * @returns Per anchor, the matched target index or a negative code.
 * @concept anchor-matching
 * @example
 * ```ts
 * import { matchAnchors } from 'mdml-engine/vision'
 * matchAnchors([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15], [40, 40, 50, 50]]) // => [0, -1, -1]
 * ```
 */
export function matchAnchors(
  targets: BoxesLike,
  anchors: BoxesLike,
  options: { readonly highThreshold?: number; readonly lowThreshold?: number; readonly allowLowQuality?: boolean } = {},
): number[] {
  const t = boxesOf(targets, 'matchAnchors')
  const a = boxesOf(anchors, 'matchAnchors')
  if (t.length === 0 || a.length === 0) throw new MdmlError('value', 'matchAnchors: no targets or no anchors')
  const high = options.highThreshold ?? 0.7
  const low = options.lowThreshold ?? 0.3
  const q = boxIou(t, a)
  const best = a.map((_, j) => {
    let k = 0
    for (let i = 1; i < t.length; i++) if ((q[i]?.[j] ?? 0) > (q[k]?.[j] ?? 0)) k = i
    return k
  })
  const matches = best.map((k, j) => {
    const v = q[k]?.[j] ?? 0
    return v < low ? BELOW_LOW_THRESHOLD : v < high ? BETWEEN_THRESHOLDS : k
  })
  if (options.allowLowQuality === true)
    for (let i = 0; i < t.length; i++) {
      const row = q[i] as number[]
      const top = Math.max(...row)
      for (let j = 0; j < a.length; j++) if (row[j] === top) matches[j] = best[j] ?? 0
    }
  return matches
}
