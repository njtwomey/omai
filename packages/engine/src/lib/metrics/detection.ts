/**
 * Object detection metrics, torchvision and pycocotools style: pairwise box intersection over union and its
 * generalised, distance and complete variants, and `detectionMeanAveragePrecision`, the mean average precision of
 * detections over images under the COCO or Pascal VOC protocol.
 */

import { MdmlError, type MetricResult, type MetricSpec, toRows } from '../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { defineMetric, type MetricFunction, matrix, metricResult } from './metric.ts'

/** Boxes as rows [x1, y1, x2, y2] (corner format, x2 > x1, y2 > y1). */
type Boxes = TensorLike | readonly (readonly number[])[]

const rowsOf = (boxes: Boxes): number[][] => {
  const m = matrix(boxes as TensorLike)
  const n = m.shape.length === 1 ? (m.data.length === 0 ? 0 : 1) : (m.shape[0] ?? 0)
  return toRows(m.data, n, 4)
}

/** Intersection, union and the enclosing box of two boxes. */
function geometry(a: readonly number[], b: readonly number[]): { inter: number; union: number; encl: number[] } {
  const [ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0] = a
  const [bx1 = 0, by1 = 0, bx2 = 0, by2 = 0] = b
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1))
  const inter = iw * ih
  const union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
  return { inter, union, encl: [Math.min(ax1, bx1), Math.min(ay1, by1), Math.max(ax2, bx2), Math.max(ay2, by2)] }
}

/** A pairwise box measure `[N, M]` of two sets of boxes, as a float64 tensor. */
function pairwise(a: Boxes, b: Boxes, f: (p: readonly number[], q: readonly number[]) => number): Tensor<'float64'> {
  const ra = rowsOf(a)
  const rb = rowsOf(b)
  const out = new Float64Array(ra.length * rb.length)
  for (let i = 0; i < ra.length; i++)
    for (let j = 0; j < rb.length; j++) out[i * rb.length + j] = f(ra[i] as number[], rb[j] as number[])
  return fromStorage(out, [ra.length, rb.length], 'float64')
}

const iouOf = (p: readonly number[], q: readonly number[]): number => {
  const g = geometry(p, q)
  return g.inter / g.union
}

/**
 * Pairwise intersection over union of two sets of boxes (torchvision.ops.box_iou). The four box measures return tensors
 * (migration, D-009: were `number[][]`).
 *
 * @param boxesA - Boxes `[N, 4]` as [x1, y1, x2, y2].
 * @param boxesB - Boxes `[M, 4]`.
 * @returns A float64 tensor of shape `[N, M]`: the IoU of every pair.
 * @concept object-detection
 * @example
 * ```ts
 * import { boxIou } from 'mdml-engine/metrics'
 * boxIou([[0, 0, 2, 2]], [[1, 1, 3, 3]]) // => [[0.14285714285714285]]
 * ```
 */
export function boxIou(boxesA: Boxes, boxesB: Boxes): Tensor<'float64'> {
  return pairwise(boxesA, boxesB, iouOf)
}

/**
 * Pairwise generalised IoU (Rezatofighi et al., 2019; torchvision.ops.generalized_box_iou):
 * IoU - |C \ (A u B)| / |C| with C the smallest enclosing box, in [-1, 1]. Returns a tensor (migration, D-009: was
 * `number[][]`).
 *
 * @param boxesA - Boxes `[N, 4]`.
 * @param boxesB - Boxes `[M, 4]`.
 * @returns A float64 tensor of shape `[N, M]`: the GIoU of every pair.
 * @example
 * ```ts
 * import { generalizedBoxIou } from 'mdml-engine/metrics'
 * generalizedBoxIou([[0, 0, 1, 1]], [[2, 0, 3, 1]]) // => [[-0.3333333333333333]]
 * ```
 */
export function generalizedBoxIou(boxesA: Boxes, boxesB: Boxes): Tensor<'float64'> {
  return pairwise(boxesA, boxesB, (p, q) => {
    const g = geometry(p, q)
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = g.encl
    const area = (x2 - x1) * (y2 - y1)
    return g.inter / g.union - (area - g.union) / area
  })
}

/** The centre distance squared over the enclosing diagonal squared (the DIoU penalty). */
function centrePenalty(p: readonly number[], q: readonly number[], encl: readonly number[], eps: number): number {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = encl
  const diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps
  const cx = ((p[0] ?? 0) + (p[2] ?? 0)) / 2 - ((q[0] ?? 0) + (q[2] ?? 0)) / 2
  const cy = ((p[1] ?? 0) + (p[3] ?? 0)) / 2 - ((q[1] ?? 0) + (q[3] ?? 0)) / 2
  return (cx * cx + cy * cy) / diag
}

/**
 * Pairwise distance IoU (Zheng et al., 2020; torchvision.ops.distance_box_iou): IoU - rho^2(centres) / c^2 with c the
 * diagonal of the enclosing box. Returns a tensor (migration, D-009: was `number[][]`).
 *
 * @param boxesA - Boxes `[N, 4]`.
 * @param boxesB - Boxes `[M, 4]`.
 * @param options - eps added to the squared diagonal (default 1e-7, torchvision's).
 * @returns A float64 tensor of shape `[N, M]`: the DIoU of every pair.
 * @example
 * ```ts
 * import { distanceBoxIou } from 'mdml-engine/metrics'
 * distanceBoxIou([[0, 0, 1, 1]], [[0, 0, 1, 1]]) // => [[1]]
 * ```
 */
export function distanceBoxIou(
  boxesA: Boxes,
  boxesB: Boxes,
  options: { readonly eps?: number } = {},
): Tensor<'float64'> {
  const eps = options.eps ?? 1e-7
  return pairwise(boxesA, boxesB, (p, q) => {
    const g = geometry(p, q)
    return g.inter / g.union - centrePenalty(p, q, g.encl, eps)
  })
}

/**
 * Pairwise complete IoU (Zheng et al., 2020; torchvision.ops.complete_box_iou): DIoU minus alpha v, with v the aspect
 * ratio consistency (4 / pi^2)(atan(w_b / h_b) - atan(w_a / h_a))^2 and alpha = v / (1 - IoU + v). Returns a tensor
 * (migration, D-009: was `number[][]`).
 *
 * @param boxesA - Boxes `[N, 4]`.
 * @param boxesB - Boxes `[M, 4]`.
 * @param options - eps (default 1e-7).
 * @returns A float64 tensor of shape `[N, M]`: the CIoU of every pair.
 * @example
 * ```ts
 * import { completeBoxIou } from 'mdml-engine/metrics'
 * completeBoxIou([[0, 0, 1, 1]], [[0, 0, 1, 1]]) // => [[1]]
 * ```
 */
export function completeBoxIou(
  boxesA: Boxes,
  boxesB: Boxes,
  options: { readonly eps?: number } = {},
): Tensor<'float64'> {
  const eps = options.eps ?? 1e-7
  return pairwise(boxesA, boxesB, (p, q) => {
    const g = geometry(p, q)
    const iou = g.inter / g.union
    const wa = (p[2] ?? 0) - (p[0] ?? 0)
    const ha = (p[3] ?? 0) - (p[1] ?? 0)
    const wb = (q[2] ?? 0) - (q[0] ?? 0)
    const hb = (q[3] ?? 0) - (q[1] ?? 0)
    const v = (4 / Math.PI ** 2) * (Math.atan(wb / hb) - Math.atan(wa / ha)) ** 2
    const alpha = v / (1 - iou + v + eps)
    return iou - centrePenalty(p, q, g.encl, eps) - alpha * v
  })
}

/** The detections of one image: boxes [x1, y1, x2, y2], their scores and class labels. */
export interface ImageDetections {
  readonly boxes: Boxes
  readonly scores: readonly number[]
  readonly labels: readonly number[]
}

/** The ground truth of one image: boxes, labels, and optional crowd flags (COCO) or difficult flags (VOC). */
export interface ImageTargets {
  readonly boxes: Boxes
  readonly labels: readonly number[]
  /** COCO crowd regions: matches are free (neither true nor false positives). */
  readonly iscrowd?: readonly number[]
  /** VOC difficult objects: ignored by the evaluation. */
  readonly difficult?: readonly number[]
  /** Object areas for the COCO size ranges (default the box areas). */
  readonly area?: readonly number[]
}

/** Options of {@link detectionMeanAveragePrecision}. */
export interface DetectionMapOptions {
  /** 'coco' (default): pycocotools' COCOeval over IoU 0.5:0.95; 'voc' (all-point) or 'voc2007' (11-point) at IoU 0.5. */
  readonly protocol?: 'coco' | 'voc' | 'voc2007'
  /** VOC's IoU threshold (default 0.5). */
  readonly iouThreshold?: number
  /** COCO's detection caps (default [1, 10, 100]). */
  readonly maxDetections?: readonly [number, number, number]
}

const metricSpec: MetricSpec = {
  name: 'detectionMeanAveragePrecision',
  direction: 'higher',
  range: [0, 1],
  inputs: 'sets',
  category: 'detection',
  concept: 'object-detection',
}

interface Det {
  readonly box: number[]
  readonly score: number
  readonly area: number
}
interface Gt {
  readonly box: number[]
  readonly crowd: boolean
  readonly area: number
}

/** COCO's IoU of a detection with a ground truth: intersection over the detection's area for crowd regions. */
function cocoIou(d: Det, g: Gt): number {
  const geo = geometry(d.box, g.box)
  if (g.crowd) return geo.inter / ((d.box[2] ?? 0) - (d.box[0] ?? 0)) / ((d.box[3] ?? 0) - (d.box[1] ?? 0))
  return geo.inter / geo.union
}

/** The stable descending order of scores (numpy's argsort(-scores, kind='mergesort')). */
const byScore = <T extends { readonly score: number }>(xs: readonly T[]): T[] =>
  xs
    .map((x, i) => [x, i] as const)
    .sort((a, b) => b[0].score - a[0].score || a[1] - b[1])
    .map(([x]) => x)

/** pycocotools' COCOeval.evaluate + accumulate + summarize for box detections. */
function coco(
  preds: readonly ImageDetections[],
  targets: readonly ImageTargets[],
  options: DetectionMapOptions,
): Record<string, number> {
  const maxDets = options.maxDetections ?? [1, 10, 100]
  const iouStep = (0.95 - 0.5) / 9
  const iouThrs = Array.from({ length: 10 }, (_, i) => i * iouStep + 0.5)
  const recThrs = Array.from({ length: 101 }, (_, i) => i * 0.01)
  const areaRanges: readonly (readonly [number, number])[] = [
    [0, 1e10],
    [0, 32 ** 2],
    [32 ** 2, 96 ** 2],
    [96 ** 2, 1e10],
  ]
  const boxArea = (b: readonly number[]): number => ((b[2] ?? 0) - (b[0] ?? 0)) * ((b[3] ?? 0) - (b[1] ?? 0))
  const categories = [...new Set([...targets.flatMap((t) => t.labels), ...preds.flatMap((p) => p.labels)])].sort(
    (a, b) => a - b,
  )
  const T = iouThrs.length
  const R = recThrs.length
  const K = categories.length
  const A = areaRanges.length
  const M = maxDets.length
  const precision = new Float64Array(T * R * K * A * M).fill(-1)
  const recall = new Float64Array(T * K * A * M).fill(-1)
  const maxDet = maxDets[M - 1] ?? 100
  for (const [k, cat] of categories.entries()) {
    // Per image: detections (score-sorted, capped) and ground truths of this category, with their IoUs.
    const images = targets.map((t, im) => {
      const p = preds[im] ?? { boxes: [], scores: [], labels: [] }
      const tb = rowsOf(t.boxes)
      const gts: Gt[] = []
      for (const [j, l] of t.labels.entries()) {
        if (l !== cat) continue
        const box = tb[j] ?? [0, 0, 0, 0]
        gts.push({ box, crowd: (t.iscrowd?.[j] ?? 0) > 0, area: t.area?.[j] ?? boxArea(box) })
      }
      const pb = rowsOf(p.boxes)
      const dets: Det[] = []
      for (const [j, l] of p.labels.entries()) {
        if (l !== cat) continue
        const box = pb[j] ?? [0, 0, 0, 0]
        dets.push({ box, score: p.scores[j] ?? 0, area: boxArea(box) })
      }
      return { gts, dets: byScore(dets).slice(0, maxDet) }
    })
    for (const [a, [lo, hi]] of areaRanges.entries()) {
      // evaluateImg for every image of this category and area range.
      const evals = images.map(({ gts, dets }) => {
        if (gts.length === 0 && dets.length === 0) return undefined
        const ignore = gts.map((g) => g.crowd || g.area < lo || g.area > hi)
        const gOrder = gts.map((_, i) => i).sort((x, y) => Number(ignore[x]) - Number(ignore[y]) || x - y)
        const g = gOrder.map((i) => gts[i] as Gt)
        const gIg = gOrder.map((i) => ignore[i] ?? false)
        const D = dets.length
        const G = g.length
        const ious = dets.map((d) => g.map((gt) => cocoIou(d, gt)))
        const dtMatched = Array.from({ length: T }, () => new Array<boolean>(D).fill(false))
        const dtIg = Array.from({ length: T }, () => new Array<boolean>(D).fill(false))
        for (let t = 0; t < T; t++) {
          const gtm = new Array<boolean>(G).fill(false)
          for (let d = 0; d < D; d++) {
            let iou = Math.min(iouThrs[t] ?? 0, 1 - 1e-10)
            let m = -1
            for (let j = 0; j < G; j++) {
              if (gtm[j] && !(g[j] as Gt).crowd) continue
              if (m > -1 && !gIg[m] && gIg[j]) break
              const v = ious[d]?.[j] ?? 0
              if (v < iou) continue
              iou = v
              m = j
            }
            if (m === -1) continue
            ;(dtIg[t] as boolean[])[d] = gIg[m] ?? false
            ;(dtMatched[t] as boolean[])[d] = true
            gtm[m] = true
          }
        }
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < D; d++) {
            const area = (dets[d] as Det).area
            if (!(dtMatched[t] as boolean[])[d] && (area < lo || area > hi)) (dtIg[t] as boolean[])[d] = true
          }
        }
        return { scores: dets.map((d) => d.score), dtMatched, dtIg, gIg }
      })
      const present = evals.filter((e) => e !== undefined)
      if (present.length === 0) continue
      for (const [m, cap] of maxDets.entries()) {
        const rows: { score: number; image: number; d: number }[] = []
        for (const [im, e] of present.entries())
          for (let d = 0; d < Math.min(cap, e.scores.length); d++) rows.push({ score: e.scores[d] ?? 0, image: im, d })
        const order = byScore(rows)
        const npig = present.reduce((s, e) => s + e.gIg.filter((x) => !x).length, 0)
        if (npig === 0) continue
        for (let t = 0; t < T; t++) {
          let tp = 0
          let fp = 0
          const pr: number[] = []
          const rc: number[] = []
          for (const r of order) {
            const e = present[r.image]
            if (e === undefined) continue
            if ((e.dtIg[t] as boolean[])[r.d]) continue
            if ((e.dtMatched[t] as boolean[])[r.d]) tp++
            else fp++
            rc.push(tp / npig)
            pr.push(tp / (fp + tp + Number.EPSILON))
          }
          // Ignored detections only repeat the previous cumulative point in pycocotools, so skipping them is equivalent.
          recall[((t * K + k) * A + a) * M + m] = rc.length > 0 ? (rc[rc.length - 1] ?? 0) : 0
          for (let i = pr.length - 1; i > 0; i--) if ((pr[i] ?? 0) > (pr[i - 1] ?? 0)) pr[i - 1] = pr[i] ?? 0
          for (const [ri, thr] of recThrs.entries()) {
            let pi = 0
            while (pi < rc.length && (rc[pi] ?? 0) < thr) pi++
            const q = pi < pr.length ? (pr[pi] ?? 0) : 0
            precision[(((t * R + ri) * K + k) * A + a) * M + m] = q
          }
        }
      }
    }
  }
  const summarize = (ap: boolean, t: number | undefined, a: number, m: number): number => {
    let s = 0
    let n = 0
    for (let ti = 0; ti < T; ti++) {
      if (t !== undefined && ti !== t) continue
      for (let k = 0; k < K; k++) {
        if (ap) {
          for (let r = 0; r < R; r++) {
            const v = precision[(((ti * R + r) * K + k) * A + a) * M + m] ?? -1
            if (v > -1) {
              s += v
              n++
            }
          }
        } else {
          const v = recall[((ti * K + k) * A + a) * M + m] ?? -1
          if (v > -1) {
            s += v
            n++
          }
        }
      }
    }
    return n === 0 ? -1 : s / n
  }
  const last = M - 1
  return {
    map: summarize(true, undefined, 0, last),
    map50: summarize(true, 0, 0, last),
    map75: summarize(true, 5, 0, last),
    mapSmall: summarize(true, undefined, 1, last),
    mapMedium: summarize(true, undefined, 2, last),
    mapLarge: summarize(true, undefined, 3, last),
    mar1: summarize(false, undefined, 0, 0),
    mar10: summarize(false, undefined, 0, 1),
    mar100: summarize(false, undefined, 0, last),
    marSmall: summarize(false, undefined, 1, last),
    marMedium: summarize(false, undefined, 2, last),
    marLarge: summarize(false, undefined, 3, last),
  }
}

/** The PASCAL VOC average precision of each class (py-faster-rcnn's voc_eval), and their mean. */
function voc(
  preds: readonly ImageDetections[],
  targets: readonly ImageTargets[],
  options: DetectionMapOptions,
): Record<string, number> {
  const threshold = options.iouThreshold ?? 0.5
  const elevenPoint = options.protocol === 'voc2007'
  const categories = [...new Set(targets.flatMap((t) => t.labels))].sort((a, b) => a - b)
  const out: Record<string, number> = {}
  for (const cat of categories) {
    const gts = targets.map((t) => {
      const b = rowsOf(t.boxes)
      const rows = t.labels.map((l, j) => (l === cat ? j : -1)).filter((j) => j >= 0)
      return {
        boxes: rows.map((j) => b[j] ?? []),
        difficult: rows.map((j) => (t.difficult?.[j] ?? 0) > 0),
        used: rows.map(() => false),
      }
    })
    const npos = gts.reduce((s, g) => s + g.difficult.filter((d) => !d).length, 0)
    const dets = byScore(
      preds.flatMap((p, im) => {
        const b = rowsOf(p.boxes)
        return p.labels
          .map((l, j) => (l === cat ? { image: im, box: b[j] ?? [], score: p.scores[j] ?? 0 } : undefined))
          .filter((d) => d !== undefined)
      }),
    )
    const tp: number[] = []
    const fp: number[] = []
    for (const d of dets) {
      const g = gts[d.image]
      let best = -Infinity
      let jmax = -1
      for (const [j, box] of (g?.boxes ?? []).entries()) {
        const v = iouOf(d.box, box)
        if (v > best) {
          best = v
          jmax = j
        }
      }
      if (g !== undefined && best > threshold) {
        if (g.difficult[jmax]) {
          tp.push(0)
          fp.push(0)
          continue
        }
        if (!g.used[jmax]) {
          g.used[jmax] = true
          tp.push(1)
          fp.push(0)
        } else {
          tp.push(0)
          fp.push(1)
        }
      } else {
        tp.push(0)
        fp.push(1)
      }
    }
    let ctp = 0
    let cfp = 0
    const rec: number[] = []
    const prec: number[] = []
    for (const [i, t] of tp.entries()) {
      ctp += t
      cfp += fp[i] ?? 0
      rec.push(ctp / npos)
      prec.push(ctp / Math.max(ctp + cfp, Number.EPSILON))
    }
    let ap = 0
    if (elevenPoint) {
      for (let k = 0; k <= 10; k++) {
        // numpy.arange(0, 1.1, 0.1): k * 0.1, so 0.30000000000000004 rather than 0.3.
        const t = k * 0.1
        let p = 0
        for (const [i, r] of rec.entries()) if (r >= t) p = Math.max(p, prec[i] ?? 0)
        ap += p / 11
      }
    } else {
      const mrec = [0, ...rec, 1]
      const mpre = [0, ...prec, 0]
      for (let i = mpre.length - 1; i > 0; i--) mpre[i - 1] = Math.max(mpre[i - 1] ?? 0, mpre[i] ?? 0)
      for (let i = 1; i < mrec.length; i++)
        if (mrec[i] !== mrec[i - 1]) ap += ((mrec[i] ?? 0) - (mrec[i - 1] ?? 0)) * (mpre[i] ?? 0)
    }
    out[String(cat)] = ap
  }
  const values = Object.values(out)
  return { map: values.reduce((a, b) => a + b, 0) / values.length, ...out }
}

/**
 * Mean average precision of object detections. The 'coco' protocol reproduces pycocotools' COCOeval for boxes: AP
 * averaged over IoU thresholds 0.5:0.05:0.95 and 101 recall points, crowd regions, size ranges and detection caps,
 * with every summary statistic (map, map50, map75, mapSmall/Medium/Large, mar1/10/100, marSmall/Medium/Large) in
 * the breakdown. The 'voc' protocols give the PASCAL VOC AP at one IoU threshold (all-point or the 2007 11-point
 * interpolation), ignoring difficult objects, with per-class APs in the breakdown.
 *
 * @param preds - Detections per image.
 * @param targets - Ground truth per image (same order).
 * @param options - protocol, iouThreshold (VOC) and maxDetections (COCO).
 * @returns The metric result (value: COCO's AP@[.5:.95] or VOC's mAP).
 * @throws MdmlError with code `shape` when the numbers of images differ.
 * @concept object-detection
 * @example
 * ```ts
 * import { detectionMeanAveragePrecision } from 'mdml-engine/metrics'
 * const pred = [{ boxes: [[0, 0, 10, 10]], scores: [0.9], labels: [1] }]
 * const target = [{ boxes: [[0, 0, 10, 10]], labels: [1] }]
 * Number(detectionMeanAveragePrecision(pred, target).estimate.value.toFixed(6)) // => 1
 * ```
 */
export const detectionMeanAveragePrecision: MetricFunction<
  [readonly ImageDetections[], readonly ImageTargets[], DetectionMapOptions?]
> = defineMetric(
  metricSpec,
  (
    preds: readonly ImageDetections[],
    targets: readonly ImageTargets[],
    options: DetectionMapOptions = {},
  ): MetricResult => {
    if (preds.length !== targets.length) throw new MdmlError('shape', 'one detection set per image of ground truth')
    const stats = (options.protocol ?? 'coco') === 'coco' ? coco(preds, targets, options) : voc(preds, targets, options)
    return metricResult('detectionMeanAveragePrecision', stats['map'] ?? Number.NaN, targets.length, stats)
  },
)
