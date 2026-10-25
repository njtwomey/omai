/**
 * Image segmentation metrics: the Dice coefficient and IoU of binary masks, per-class mean IoU of label maps, the
 * Hausdorff distance between mask boundaries (MONAI style), the boundary F1 score and panoptic quality, which scores
 * matched segments of both things and stuff.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { classCounts, encodeLabels, labelValues } from './_labels.ts'
import { defineMetric, type MetricFunction, matrix, metricResult } from './metric.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower' = 'higher',
  range: readonly [number, number] = [0, 1],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'labels',
  category: 'segmentation',
  concept: 'segmentation-evaluation',
})

/** A binary mask read as a grid of booleans. */
interface Mask {
  readonly shape: readonly number[]
  readonly data: Uint8Array
}

const mask = (x: TensorLike): Mask => {
  const m = matrix(x)
  return { shape: m.shape, data: Uint8Array.from(m.data, (v) => (v > 0 ? 1 : 0)) }
}

const sameShape = (name: string, a: Mask, b: Mask): void => {
  if (a.shape.length !== b.shape.length || a.shape.some((d, i) => d !== b.shape[i]))
    throw new MdmlError('shape', `${name}: masks differ in shape`)
}

/**
 * The Dice coefficient 2 |A n B| / (|A| + |B|) of two binary masks (monai.metrics.DiceMetric for one class); 1 when
 * both are empty.
 *
 * @param pred - Predicted mask (non-zero is foreground), any shape.
 * @param target - True mask, the same shape.
 * @returns The metric result.
 * @concept segmentation-evaluation
 * @example
 * ```ts
 * import { diceCoefficient } from 'mdml-engine/metrics'
 * diceCoefficient([[1, 1], [0, 0]], [[1, 0], [0, 0]]).estimate.value // => 0.6666666666666666
 * ```
 */
export const diceCoefficient: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('diceCoefficient'),
  (pred: TensorLike, target: TensorLike): MetricResult => {
    const a = mask(pred)
    const b = mask(target)
    sameShape('diceCoefficient', a, b)
    let inter = 0
    let sa = 0
    let sb = 0
    for (let i = 0; i < a.data.length; i++) {
      inter += (a.data[i] ?? 0) & (b.data[i] ?? 0)
      sa += a.data[i] ?? 0
      sb += b.data[i] ?? 0
    }
    return metricResult('diceCoefficient', sa + sb === 0 ? 1 : (2 * inter) / (sa + sb), a.data.length)
  },
)

/**
 * Intersection over union |A n B| / |A u B| of two binary masks; 1 when both are empty.
 *
 * @param pred - Predicted mask.
 * @param target - True mask.
 * @returns The metric result.
 * @example
 * ```ts
 * import { maskIou } from 'mdml-engine/metrics'
 * maskIou([[1, 1], [0, 0]], [[1, 0], [0, 0]]).estimate.value // => 0.5
 * ```
 */
export const maskIou: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('maskIou'),
  (pred: TensorLike, target: TensorLike): MetricResult => {
    const a = mask(pred)
    const b = mask(target)
    sameShape('maskIou', a, b)
    let inter = 0
    let union = 0
    for (let i = 0; i < a.data.length; i++) {
      inter += (a.data[i] ?? 0) & (b.data[i] ?? 0)
      union += (a.data[i] ?? 0) | (b.data[i] ?? 0)
    }
    return metricResult('maskIou', union === 0 ? 1 : inter / union, a.data.length)
  },
)

/**
 * Per-class mean IoU of label maps (torchmetrics' MulticlassJaccardIndex with macro averaging; sklearn's jaccard_score
 * over pixels): the mean over classes of TP / (TP + FP + FN), classes absent from both maps excluded, ignored pixels
 * dropped.
 *
 * @param pred - Predicted class map (any shape).
 * @param target - True class map.
 * @param options - numClasses (default: the classes present) and ignoreIndex.
 * @returns The metric result, with each class's IoU in the breakdown.
 * @example
 * ```ts
 * import { meanIou } from 'mdml-engine/metrics'
 * meanIou([[0, 1], [1, 1]], [[0, 1], [0, 1]]).estimate.value // => 0.5833333333333333
 * ```
 */
export const meanIou: MetricFunction<
  [TensorLike, TensorLike, { readonly numClasses?: number; readonly ignoreIndex?: number }?]
> = defineMetric(
  spec('meanIou'),
  (
    pred: TensorLike,
    target: TensorLike,
    options: { readonly numClasses?: number; readonly ignoreIndex?: number } = {},
  ): MetricResult => {
    let p = labelValues(pred) as number[]
    let t = labelValues(target) as number[]
    if (p.length !== t.length) throw new MdmlError('shape', 'meanIou: maps differ in size')
    if (options.ignoreIndex !== undefined) {
      const keep = t.map((v) => v !== options.ignoreIndex)
      p = p.filter((_, i) => keep[i])
      t = t.filter((_, i) => keep[i])
    }
    const labels =
      options.numClasses === undefined ? undefined : Array.from({ length: options.numClasses }, (_, c) => c)
    const e = encodeLabels('meanIou', t, p, labels === undefined ? {} : { labels })
    const c = classCounts(e)
    const per: Record<string, number> = {}
    const values: number[] = []
    for (const [k, cls] of e.classes.entries()) {
      const den = (c.tp[k] ?? 0) + (c.fp[k] ?? 0) + (c.fn[k] ?? 0)
      if (den === 0) continue
      const v = (c.tp[k] ?? 0) / den
      per[String(cls)] = v
      values.push(v)
    }
    return metricResult('meanIou', values.reduce((a, b) => a + b, 0) / values.length, e.n, per)
  },
)

/** The boundary of a binary mask: foreground cells with a background (or out-of-grid) face neighbour. */
function edges(m: Mask): number[] {
  const shape = m.shape
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1))
  const out: number[] = []
  for (let idx = 0; idx < m.data.length; idx++) {
    if (!m.data[idx]) continue
    let edge = false
    for (const [ax, n] of shape.entries()) {
      const s = strides[ax] ?? 1
      const coord = Math.floor(idx / s) % n
      if (coord === 0 || coord === n - 1 || !m.data[idx - s] || !m.data[idx + s]) {
        edge = true
        break
      }
    }
    if (edge) out.push(idx)
  }
  return out
}

/** Coordinates of flat indices, scaled by the voxel spacing. */
const coordinates = (idx: readonly number[], shape: readonly number[], spacing: readonly number[]): number[][] => {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1))
  return idx.map((i) => shape.map((n, ax) => (Math.floor(i / (strides[ax] ?? 1)) % n) * (spacing[ax] ?? 1)))
}

/** Each point's distance to the nearest point of another set. */
const nearestDistances = (from: readonly number[][], to: readonly number[][]): number[] =>
  from.map((p) => {
    let best = Number.POSITIVE_INFINITY
    for (const q of to) {
      let d = 0
      for (const [k, v] of p.entries()) d += (v - (q[k] ?? 0)) ** 2
      if (d < best) best = d
    }
    return Math.sqrt(best)
  })

/** numpy.percentile (linear) of values. */
const percentile = (values: readonly number[], q: number): number => {
  const s = [...values].sort((a, b) => a - b)
  const pos = (q / 100) * (s.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, s.length - 1)
  return (s[lo] ?? 0) + (pos - lo) * ((s[hi] ?? 0) - (s[lo] ?? 0))
}

/** Options of {@link maskHausdorffDistance}. */
export interface MaskHausdorffOptions {
  /** The percentile of the boundary distances (default 100, the Hausdorff distance; 95 gives HD95). */
  readonly percentile?: number
  /** Only from the prediction's boundary to the target's (default false: the larger of both directions). */
  readonly directed?: boolean
  /** The voxel spacing along each axis (default 1). */
  readonly spacing?: readonly number[]
}

/**
 * The Hausdorff distance between the boundaries of two binary masks (monai.metrics.compute_hausdorff_distance): the
 * largest (or the given percentile, HD95 for 95) distance from a boundary point of one mask to the boundary of the
 * other, in either direction. Boundaries are the foreground cells removed by a face-connected erosion.
 *
 * @param pred - Predicted mask (2-D or 3-D).
 * @param target - True mask, the same shape.
 * @param options - percentile, directed and spacing.
 * @returns The metric result (Infinity when exactly one mask is empty, NaN when both are).
 * @example
 * ```ts
 * import { maskHausdorffDistance } from 'mdml-engine/metrics'
 * maskHausdorffDistance([[1, 0, 0, 0]], [[0, 0, 0, 1]]).estimate.value // => 3
 * ```
 */
export const maskHausdorffDistance: MetricFunction<[TensorLike, TensorLike, MaskHausdorffOptions?]> = defineMetric(
  spec('maskHausdorffDistance', 'lower', [0, Infinity]),
  (pred: TensorLike, target: TensorLike, options: MaskHausdorffOptions = {}): MetricResult => {
    const a = mask(pred)
    const b = mask(target)
    sameShape('maskHausdorffDistance', a, b)
    const ea = coordinates(edges(a), a.shape, options.spacing ?? [])
    const eb = coordinates(edges(b), b.shape, options.spacing ?? [])
    if (ea.length === 0 && eb.length === 0) return metricResult('maskHausdorffDistance', Number.NaN, 0)
    if (ea.length === 0 || eb.length === 0) return metricResult('maskHausdorffDistance', Number.POSITIVE_INFINITY, 0)
    const q = options.percentile ?? 100
    const forward = percentile(nearestDistances(ea, eb), q)
    const value = options.directed === true ? forward : Math.max(forward, percentile(nearestDistances(eb, ea), q))
    return metricResult('maskHausdorffDistance', value, ea.length + eb.length)
  },
)

/**
 * The boundary F1 score (Csurka et al., 2013; the DAVIS contour accuracy): the harmonic mean of the fraction of the
 * predicted boundary within `tolerance` of the true boundary and the fraction of the true boundary within
 * `tolerance` of the predicted one.
 *
 * @param pred - Predicted mask.
 * @param target - True mask.
 * @param options - tolerance in pixels (default 2).
 * @returns The metric result, with boundary precision and recall in the breakdown.
 * @example
 * ```ts
 * import { boundaryF1 } from 'mdml-engine/metrics'
 * boundaryF1([[1, 1], [1, 1]], [[1, 1], [1, 1]]).estimate.value // => 1
 * ```
 */
export const boundaryF1: MetricFunction<[TensorLike, TensorLike, { readonly tolerance?: number }?]> = defineMetric(
  spec('boundaryF1'),
  (pred: TensorLike, target: TensorLike, options: { readonly tolerance?: number } = {}): MetricResult => {
    const a = mask(pred)
    const b = mask(target)
    sameShape('boundaryF1', a, b)
    const tol = options.tolerance ?? 2
    const ea = coordinates(edges(a), a.shape, [])
    const eb = coordinates(edges(b), b.shape, [])
    if (ea.length === 0 && eb.length === 0) return metricResult('boundaryF1', 1, 0, { precision: 1, recall: 1 })
    const precision = ea.length === 0 ? 0 : nearestDistances(ea, eb).filter((d) => d <= tol).length / ea.length
    const recall = eb.length === 0 ? 0 : nearestDistances(eb, ea).filter((d) => d <= tol).length / eb.length
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall)
    return metricResult('boundaryF1', f1, ea.length + eb.length, { precision, recall })
  },
)

/** Options of {@link panopticQuality}. */
export interface PanopticOptions {
  /** Categories with instances. */
  readonly things: readonly number[]
  /** Amorphous categories (instance ids ignored). */
  readonly stuffs: readonly number[]
  /** Stuff categories scored with the modified PQ (Porzi et al., 2019): every target segment counts, IoU > 0 matches. */
  readonly modifiedStuffs?: readonly number[]
}

/**
 * Panoptic quality (Kirillov et al., 2019; torchmetrics' PanopticQuality): segments match when they share a category
 * and their IoU exceeds 0.5; per category PQ = sum IoU(TP) / (|TP| + |FP| / 2 + |FN| / 2) = SQ RQ, averaged over
 * categories present. Pixels of unknown categories are void: they leave the IoUs, and segments mostly void are not
 * counted as errors.
 *
 * @param pred - Predicted panoptic map `[..., 2]`: (category, instance) per pixel.
 * @param target - True panoptic map, the same shape.
 * @param options - things, stuffs and modified stuffs.
 * @returns The metric result: PQ, with SQ, RQ and the per-category PQ in the breakdown.
 * @example
 * ```ts
 * import { panopticQuality } from 'mdml-engine/metrics'
 * panopticQuality([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], [[[0, 1], [0, 1]], [[1, 0], [1, 0]]], { things: [0], stuffs: [1] }).estimate.value // => 1
 * ```
 */
export const panopticQuality: MetricFunction<[TensorLike, TensorLike, PanopticOptions]> = defineMetric(
  spec('panopticQuality'),
  (pred: TensorLike, target: TensorLike, options: PanopticOptions): MetricResult => {
    const things = new Set(options.things)
    const stuffs = new Set(options.stuffs)
    const modified = new Set(options.modifiedStuffs ?? [])
    const voidCat = 1 + [...things, ...stuffs].reduce((a, b) => Math.max(a, b), 0)
    const colours = (x: TensorLike): string[] => {
      const m = matrix(x)
      const n = m.data.length / 2
      return Array.from({ length: n }, (_, i) => {
        const cat = m.data[2 * i] ?? 0
        const inst = m.data[2 * i + 1] ?? 0
        if (stuffs.has(cat)) return `${cat},0`
        if (things.has(cat)) return `${cat},${inst}`
        return `${voidCat},0`
      })
    }
    const p = colours(pred)
    const t = colours(target)
    if (p.length !== t.length) throw new MdmlError('shape', 'panopticQuality: maps differ in shape')
    const voidColour = `${voidCat},0`
    const count = (xs: readonly string[]): Map<string, number> => {
      const m = new Map<string, number>()
      for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1)
      return m
    }
    const predArea = count(p)
    const targetArea = count(t)
    const inter = count(p.map((c, i) => `${c}|${t[i] ?? ''}`))
    const cats = [...[...things].sort((a, b) => a - b), ...[...stuffs].sort((a, b) => a - b)]
    const index = new Map(cats.map((c, i) => [c, i]))
    const iouSum = new Float64Array(cats.length)
    const tp = new Float64Array(cats.length)
    const fp = new Float64Array(cats.length)
    const fn = new Float64Array(cats.length)
    const catOf = (colour: string): number => Number(colour.split(',')[0])
    const predMatched = new Set<string>()
    const targetMatched = new Set<string>()
    for (const [key, area] of inter) {
      const [pc = '', tc = ''] = key.split('|')
      if (tc === voidColour || catOf(pc) !== catOf(tc)) continue
      const predVoid = inter.get(`${pc}|${voidColour}`) ?? 0
      const voidTarget = inter.get(`${voidColour}|${tc}`) ?? 0
      const union = (predArea.get(pc) ?? 0) - predVoid + (targetArea.get(tc) ?? 0) - voidTarget - area
      const iou = area / union
      const k = index.get(catOf(tc)) ?? 0
      if (!modified.has(catOf(tc)) && iou > 0.5) {
        predMatched.add(pc)
        targetMatched.add(tc)
        iouSum[k] = (iouSum[k] ?? 0) + iou
        tp[k] = (tp[k] ?? 0) + 1
      } else if (modified.has(catOf(tc)) && iou > 0) {
        iouSum[k] = (iouSum[k] ?? 0) + iou
      }
    }
    for (const [tc, area] of targetArea) {
      if (tc === voidColour || targetMatched.has(tc)) continue
      if ((inter.get(`${voidColour}|${tc}`) ?? 0) / area <= 0.5 && !modified.has(catOf(tc))) {
        const k = index.get(catOf(tc)) ?? 0
        fn[k] = (fn[k] ?? 0) + 1
      }
    }
    for (const [pc, area] of predArea) {
      if (pc === voidColour || predMatched.has(pc)) continue
      if ((inter.get(`${pc}|${voidColour}`) ?? 0) / area <= 0.5 && !modified.has(catOf(pc))) {
        const k = index.get(catOf(pc)) ?? 0
        fp[k] = (fp[k] ?? 0) + 1
      }
    }
    for (const tc of targetArea.keys()) {
      if (modified.has(catOf(tc))) {
        const k = index.get(catOf(tc)) ?? 0
        tp[k] = (tp[k] ?? 0) + 1
      }
    }
    const pq: number[] = []
    const sq: number[] = []
    const rq: number[] = []
    const per: Record<string, number> = {}
    for (const [k, cat] of cats.entries()) {
      const den = (tp[k] ?? 0) + 0.5 * (fp[k] ?? 0) + 0.5 * (fn[k] ?? 0)
      if (den === 0) continue
      const s = (tp[k] ?? 0) > 0 ? (iouSum[k] ?? 0) / (tp[k] ?? 1) : 0
      const r = (tp[k] ?? 0) / den
      sq.push(s)
      rq.push(r)
      pq.push(s * r)
      per[String(cat)] = s * r
    }
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    return metricResult('panopticQuality', mean(pq), p.length, {
      segmentationQuality: mean(sq),
      recognitionQuality: mean(rq),
      ...per,
    })
  },
)
