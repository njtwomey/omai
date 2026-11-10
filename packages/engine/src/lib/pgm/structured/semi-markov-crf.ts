/**
 * Semi-Markov conditional random fields (Sarawagi and Cohen 2004): a CRF whose labelling is a segmentation. Instead of
 * one label per item the output is a sequence of segments, each with a label, a start and an end and a length of at most
 * `maxLength`, and the score of a segmentation is the sum of the scores of its segments plus a weight for each pair
 * of consecutive segment labels. A segment's score is a feature of the whole segment: the attributes of the items inside
 * (summed, with a per-item prefix sum over the label's scores so each candidate segment costs O(1) once the items are
 * scored), optionally the attributes of its first and of its last item (boundary features), and its length. The model is
 * the CRF counterpart of the hidden semi-Markov model (`hsmmPosterior`, which is generative and has Gaussian emissions
 * and its own duration table); here the durations are features and the items are conditioned on, not generated.
 *
 * Forward and backward run over segment ends: O(n L K) for the segment scores and the sums over segments plus O(n K^2)
 * for the transitions, because a transition depends only on the labels on both sides of a boundary and not on the
 * segment lengths. Viterbi, the marginal of every segment, the expected feature counts and cost-augmented decoding
 * (token-level Hamming or segment-level loss) follow, all through {@link semiMarkovOracle} for the structured learners.
 * With `maxLength` 1 the model is the linear-chain CRF (same score and partition function, with the state weights as the
 * content weights); `segmentationToTags` and `tagsToSegmentation` convert between segmentations and BIO tags to compare.
 */

import { argmaxOf, logSumExpOf, OmaiError } from '../../base/index.ts'
import { entitySpans } from '../../metrics/index.ts'
import type { ItemAttributes } from '../../text/index.ts'
import type { StructuredOracle } from './oracle.ts'

/** The input of a semi-Markov CRF: the attributes of each item (as the item attributes of a linear-chain CRF). */
export type SegmentInput = readonly ItemAttributes[]

/** A segment: items `start` to `end - 1` (`end` exclusive) carry the label with index `label`. */
export interface Segment {
  readonly label: number
  readonly start: number
  readonly end: number
}

/** A segmentation of an input of n items: consecutive segments that cover items 0 to n - 1. */
export type Segmentation = readonly Segment[]

/**
 * A semi-Markov CRF. The weights are one vector (see {@link semiMarkovLayout}): content weights per attribute and label
 * (the attribute value summed over the items of a segment), boundary weights of the first and the last item of a segment
 * when `boundary` is on, one weight per label and length, and one per pair of consecutive segment labels.
 */
export interface SemiMarkovCrf {
  readonly kind: 'semiMarkovCrf'
  readonly labels: readonly string[]
  readonly attributes: readonly string[]
  /** The longest segment. */
  readonly maxLength: number
  /** Whether the first and the last item of a segment have their own weights. */
  readonly boundary: boolean
}

/**
 * Build a semi-Markov CRF (weights are given at each call, as for every `StructuredOracle`).
 *
 * @param labels - The segment labels.
 * @param attributes - The item attribute vocabulary (attributes outside it are ignored).
 * @param maxLength - The longest segment (at least 1).
 * @param options - `boundary` adds first-item and last-item attribute weights (default false).
 * @returns The model.
 * @throws OmaiError with code `value` for no labels, repeated labels or a maximum length below one.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovLayout } from 'omai-engine/pgm'
 * semiMarkovLayout(semiMarkovCrf(['A', 'B'], ['f'], 3)).size // => 12
 * ```
 */
export function semiMarkovCrf(
  labels: readonly string[],
  attributes: readonly string[],
  maxLength: number,
  options: { readonly boundary?: boolean } = {},
): SemiMarkovCrf {
  if (labels.length === 0 || new Set(labels).size !== labels.length)
    throw new OmaiError('value', 'semiMarkovCrf: labels must be non-empty and distinct')
  if (!(Number.isInteger(maxLength) && maxLength >= 1))
    throw new OmaiError('value', 'semiMarkovCrf: maxLength must be a positive integer')
  return {
    kind: 'semiMarkovCrf',
    labels: [...labels],
    attributes: [...attributes],
    maxLength,
    boundary: options.boundary === true,
  }
}

/** Where each block of the weight vector starts. */
export interface SemiMarkovLayout {
  /** Content weights, [attribute][label]. */
  readonly content: number
  /** First-item weights, [attribute][label] (empty without boundary features). */
  readonly begin: number
  /** Last-item weights, [attribute][label] (empty without boundary features). */
  readonly end: number
  /** Length weights, [label][length - 1]. */
  readonly length: number
  /** Transition weights between consecutive segment labels, [from][to]. */
  readonly transition: number
  /** The number of weights. */
  readonly size: number
}

/**
 * The layout of the weight vector of a model: content, begin, end, length, then transition weights.
 *
 * @param model - The model.
 * @returns The offset of each block and the total size.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovLayout } from 'omai-engine/pgm'
 * semiMarkovLayout(semiMarkovCrf(['A', 'B'], ['f', 'g'], 2, { boundary: true })).transition // => 16
 * ```
 */
export function semiMarkovLayout(model: SemiMarkovCrf): SemiMarkovLayout {
  const K = model.labels.length
  const A = model.attributes.length
  const block = model.boundary ? A * K : 0
  const content = 0
  const begin = A * K
  const end = begin + block
  const length = end + block
  const transition = length + K * model.maxLength
  return { content, begin, end, length, transition, size: transition + K * K }
}

const attributeMaps = new WeakMap<readonly string[], Map<string, number>>()
function attributeIndex(model: SemiMarkovCrf): Map<string, number> {
  let index = attributeMaps.get(model.attributes)
  if (index === undefined) {
    index = new Map(model.attributes.map((a, i) => [a, i]))
    attributeMaps.set(model.attributes, index)
  }
  return index
}

/** The (attribute index, value) pairs of every item. */
function itemsOf(model: SemiMarkovCrf, x: SegmentInput): [number, number][][] {
  const index = attributeIndex(model)
  return x.map((item) => {
    const out: [number, number][] = []
    for (const [a, v] of Object.entries(item)) {
      const i = index.get(a)
      if (i !== undefined) out.push([i, v])
    }
    return out
  })
}

/** Check that a segmentation tiles the n items with segments of valid labels and lengths. */
function checkSegmentation(model: SemiMarkovCrf, n: number, y: Segmentation, name: string): void {
  let at = 0
  for (const s of y) {
    if (s.start !== at || !(s.end > s.start) || s.end > n)
      throw new OmaiError('shape', `${name}: segments must tile the ${n} items in order`)
    if (s.end - s.start > model.maxLength)
      throw new OmaiError('value', `${name}: a segment is longer than maxLength ${model.maxLength}`)
    if (!(Number.isInteger(s.label) && s.label >= 0 && s.label < model.labels.length))
      throw new OmaiError('value', `${name}: unknown label index ${s.label}`)
    at = s.end
  }
  if (at !== n) throw new OmaiError('shape', `${name}: segments must tile the ${n} items in order`)
}

/** The scores of everything a segment can use, computed once per input and weight vector. */
interface Scores {
  readonly n: number
  readonly K: number
  readonly L: number
  /** Segment score of items s to e - 1 with label k. */
  segment(s: number, e: number, k: number): number
  /** Transition score from label i to label j. */
  transition(i: number, j: number): number
}

function scoresOf(
  model: SemiMarkovCrf,
  layout: SemiMarkovLayout,
  w: ArrayLike<number>,
  items: readonly (readonly (readonly [number, number])[])[],
): Scores {
  const n = items.length
  const K = model.labels.length
  const L = model.maxLength
  // per-item content scores and their prefix sums over items: a segment's content is a difference of two rows
  const prefix = new Float64Array((n + 1) * K)
  const begin = new Float64Array(model.boundary ? n * K : 0)
  const end = new Float64Array(model.boundary ? n * K : 0)
  for (const [i, item] of items.entries()) {
    for (let k = 0; k < K; k++) {
      let c = 0
      let b = 0
      let e = 0
      for (const [a, v] of item) {
        c += v * (w[layout.content + a * K + k] ?? 0)
        if (model.boundary) {
          b += v * (w[layout.begin + a * K + k] ?? 0)
          e += v * (w[layout.end + a * K + k] ?? 0)
        }
      }
      prefix[(i + 1) * K + k] = (prefix[i * K + k] ?? 0) + c
      if (model.boundary) {
        begin[i * K + k] = b
        end[i * K + k] = e
      }
    }
  }
  return {
    n,
    K,
    L,
    segment: (s, e, k) =>
      (prefix[e * K + k] ?? 0) -
      (prefix[s * K + k] ?? 0) +
      (model.boundary ? (begin[s * K + k] ?? 0) + (end[(e - 1) * K + k] ?? 0) : 0) +
      (w[layout.length + k * L + (e - s - 1)] ?? 0),
    transition: (i, j) => w[layout.transition + i * K + j] ?? 0,
  }
}

/** Extra score of a candidate segment (a cost for cost-augmented decoding). */
type Cost = (s: number, e: number, k: number) => number

interface Forward {
  readonly logZ: number
  /** alpha[e * K + k]: log total of the segmentations of items 0..e-1 whose last segment has label k. */
  readonly alpha: Float64Array
  /** into[s * K + k]: log total of the ways to reach a segment of label k starting at s (zero at s = 0). */
  readonly into: Float64Array
}

function forwardPass(sc: Scores): Forward {
  const { n, K, L } = sc
  const alpha = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  const into = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  for (let k = 0; k < K; k++) into[k] = 0
  const buf = new Float64Array(Math.max(L, K))
  for (let e = 1; e <= n; e++) {
    for (let k = 0; k < K; k++) {
      let m = 0
      for (let l = 1; l <= Math.min(L, e); l++) buf[m++] = sc.segment(e - l, e, k) + (into[(e - l) * K + k] ?? 0)
      alpha[e * K + k] = logSumExpOf(buf.subarray(0, m))
    }
    if (e === n) break
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < K; j++) buf[j] = (alpha[e * K + j] ?? 0) + sc.transition(j, k)
      into[e * K + k] = logSumExpOf(buf.subarray(0, K))
    }
  }
  return { logZ: logSumExpOf(alpha.subarray(n * K, (n + 1) * K)), alpha, into }
}

/** The backward messages: `beta` as below, and `start[s * K + k]`, the log total of the ways to finish items s..n-1 given that a segment of label k starts at s. */
function backwardPass(sc: Scores): { readonly beta: Float64Array; readonly start: Float64Array } {
  const { n, K, L } = sc
  // beta[e * K + k]: log total of the ways to finish items e..n-1 given that a segment of label k ended at e
  const beta = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  const start = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  for (let k = 0; k < K; k++) beta[n * K + k] = 0
  const buf = new Float64Array(Math.max(L, K))
  for (let s = n - 1; s >= 0; s--) {
    for (let k = 0; k < K; k++) {
      let m = 0
      for (let l = 1; l <= Math.min(L, n - s); l++) buf[m++] = sc.segment(s, s + l, k) + (beta[(s + l) * K + k] ?? 0)
      start[s * K + k] = logSumExpOf(buf.subarray(0, m))
    }
    if (s === 0) break
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < K; j++) buf[j] = sc.transition(k, j) + (start[s * K + j] ?? 0)
      beta[s * K + k] = logSumExpOf(buf.subarray(0, K))
    }
  }
  return { beta, start }
}

/** The best segmentation and its score (with a cost added to every segment). */
function bestSegmentation(sc: Scores, cost?: Cost): { readonly y: Segment[]; readonly score: number } {
  const { n, K, L } = sc
  if (n === 0) return { y: [], score: 0 }
  const best = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  const into = new Float64Array((n + 1) * K).fill(Number.NEGATIVE_INFINITY)
  const lengthAt = new Int32Array((n + 1) * K)
  const prevAt = new Int32Array((n + 1) * K)
  for (let k = 0; k < K; k++) into[k] = 0
  for (let e = 1; e <= n; e++) {
    for (let k = 0; k < K; k++) {
      let top = Number.NEGATIVE_INFINITY
      for (let l = 1; l <= Math.min(L, e); l++) {
        const v = sc.segment(e - l, e, k) + (cost?.(e - l, e, k) ?? 0) + (into[(e - l) * K + k] ?? 0)
        if (v > top) {
          top = v
          lengthAt[e * K + k] = l
        }
      }
      best[e * K + k] = top
    }
    if (e === n) break
    for (let k = 0; k < K; k++) {
      let top = Number.NEGATIVE_INFINITY
      for (let j = 0; j < K; j++) {
        const v = (best[e * K + j] ?? 0) + sc.transition(j, k)
        if (v > top) {
          top = v
          prevAt[e * K + k] = j
        }
      }
      into[e * K + k] = top
    }
  }
  let k = argmaxOf(best.subarray(n * K, (n + 1) * K))
  const score = best[n * K + k] ?? Number.NEGATIVE_INFINITY
  const y: Segment[] = []
  let e = n
  while (e > 0) {
    const l = lengthAt[e * K + k] ?? 1
    y.push({ label: k, start: e - l, end: e })
    const s = e - l
    if (s > 0) k = prevAt[s * K + k] ?? 0
    e = s
  }
  return { y: y.reverse(), score }
}

/** The marginals of every segment, the items' label marginals and the log-partition function. */
export interface SemiMarkovMarginals {
  readonly logPartition: number
  /** Every segment with positive probability under the model, with p(segment is in the segmentation | x). */
  readonly segments: readonly (Segment & { readonly probability: number })[]
  /** p(item i lies in a segment labelled k | x), `[n][K]`: what a chain CRF's marginals are at `maxLength` 1. */
  readonly labels: number[][]
}

interface Expectation {
  readonly logZ: number
  readonly segments: { readonly s: number; readonly e: number; readonly k: number; readonly p: number }[]
  /** Probability of a boundary after item s - 1 between labels i then j, `[s][i * K + j]` for s = 1..n-1. */
  readonly transitions: Float64Array[]
}

function expectation(sc: Scores): Expectation {
  const { n, K, L } = sc
  const f = forwardPass(sc)
  const b = backwardPass(sc)
  const segments: { s: number; e: number; k: number; p: number }[] = []
  for (let s = 0; s < n; s++)
    for (let k = 0; k < K; k++)
      for (let l = 1; l <= Math.min(L, n - s); l++) {
        const e = s + l
        const p = Math.exp((f.into[s * K + k] ?? 0) + sc.segment(s, e, k) + (b.beta[e * K + k] ?? 0) - f.logZ)
        if (p > 0) segments.push({ s, e, k, p })
      }
  const transitions: Float64Array[] = [new Float64Array(0)]
  for (let s = 1; s < n; s++) {
    const m = new Float64Array(K * K)
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++)
        m[i * K + j] = Math.exp((f.alpha[s * K + i] ?? 0) + sc.transition(i, j) + (b.start[s * K + j] ?? 0) - f.logZ)
    transitions.push(m)
  }
  return { logZ: f.logZ, segments, transitions }
}

/**
 * The marginal probability of every segment and of every item's label under the model, by the forward and backward
 * passes over segment ends.
 *
 * @param model - The model.
 * @param w - The weights (see {@link semiMarkovLayout}).
 * @param x - The item attributes.
 * @returns The log-partition function, the segment marginals and the label marginals of the items.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovMarginals } from 'omai-engine/pgm'
 * const model = semiMarkovCrf(['A'], [], 2)
 * semiMarkovMarginals(model, new Float64Array(3), [{}, {}]).segments.map((s) => [s.start, s.end, Math.round(s.probability * 100) / 100]) // => [[0, 1, 0.5], [0, 2, 0.5], [1, 2, 0.5]]
 * ```
 */
export function semiMarkovMarginals(model: SemiMarkovCrf, w: ArrayLike<number>, x: SegmentInput): SemiMarkovMarginals {
  const layout = semiMarkovLayout(model)
  const K = model.labels.length
  const e = expectation(scoresOf(model, layout, w, itemsOf(model, x)))
  const labels = x.map(() => new Array<number>(K).fill(0))
  for (const { s, e: end, k, p } of e.segments)
    for (let i = s; i < end; i++) (labels[i] as number[])[k] = ((labels[i] as number[])[k] ?? 0) + p
  return {
    logPartition: e.logZ,
    segments: e.segments.map(({ s, e: end, k, p }) => ({ label: k, start: s, end, probability: p })),
    labels,
  }
}

/** The unnormalised log score of a segmentation, w . phi(x, y). */
function scoreOf(sc: Scores, y: Segmentation): number {
  let total = 0
  for (const [j, s] of y.entries()) {
    total += sc.segment(s.start, s.end, s.label)
    if (j > 0) total += sc.transition((y[j - 1] as Segment).label, s.label)
  }
  return total
}

/**
 * The unnormalised log score of a segmentation: its segment scores plus the transition weights between consecutive
 * segment labels.
 *
 * @param model - The model.
 * @param w - The weights (see {@link semiMarkovLayout}).
 * @param x - The item attributes.
 * @param y - The segmentation.
 * @returns The score.
 * @throws OmaiError with code `shape` for segments that do not tile the items, `value` for a segment longer than the maximum or an unknown label.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovLayout, semiMarkovScore } from 'omai-engine/pgm'
 * const model = semiMarkovCrf(['A', 'B'], ['f'], 2)
 * const w = new Float64Array(semiMarkovLayout(model).size)
 * w[0] = 1 // content weight of f for A
 * semiMarkovScore(model, w, [{ f: 1 }, { f: 2 }], [{ label: 0, start: 0, end: 2 }]) // => 3
 * ```
 */
export function semiMarkovScore(model: SemiMarkovCrf, w: ArrayLike<number>, x: SegmentInput, y: Segmentation): number {
  checkSegmentation(model, x.length, y, 'semiMarkovScore')
  return scoreOf(scoresOf(model, semiMarkovLayout(model), w, itemsOf(model, x)), y)
}

/**
 * The feature counts of a segmentation (the vector whose inner product with the weights is its score).
 *
 * @param model - The model.
 * @param x - The item attributes.
 * @param y - The segmentation.
 * @returns The counts in layout order.
 * @throws OmaiError as {@link semiMarkovScore}.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovFeatureCounts } from 'omai-engine/pgm'
 * const model = semiMarkovCrf(['A', 'B'], ['f'], 2)
 * Array.from(semiMarkovFeatureCounts(model, [{ f: 1 }, { f: 2 }], [{ label: 1, start: 0, end: 2 }])) // => [0, 3, 0, 0, 0, 1, 0, 0, 0, 0]
 * ```
 */
export function semiMarkovFeatureCounts(model: SemiMarkovCrf, x: SegmentInput, y: Segmentation): Float64Array {
  checkSegmentation(model, x.length, y, 'semiMarkovFeatureCounts')
  const layout = semiMarkovLayout(model)
  const K = model.labels.length
  const items = itemsOf(model, x)
  const out = new Float64Array(layout.size)
  for (const [j, s] of y.entries()) {
    for (let i = s.start; i < s.end; i++)
      for (const [a, v] of items[i] ?? [])
        out[layout.content + a * K + s.label] = (out[layout.content + a * K + s.label] ?? 0) + v
    if (model.boundary) {
      for (const [a, v] of items[s.start] ?? [])
        out[layout.begin + a * K + s.label] = (out[layout.begin + a * K + s.label] ?? 0) + v
      for (const [a, v] of items[s.end - 1] ?? [])
        out[layout.end + a * K + s.label] = (out[layout.end + a * K + s.label] ?? 0) + v
    }
    const lengthAt = layout.length + s.label * model.maxLength + (s.end - s.start - 1)
    out[lengthAt] = (out[lengthAt] ?? 0) + 1
    if (j > 0) {
      const at = layout.transition + (y[j - 1] as Segment).label * K + s.label
      out[at] = (out[at] ?? 0) + 1
    }
  }
  return out
}

/**
 * The expected feature counts under the model and the log-partition function: the segment marginals spread over the
 * items they cover (through a difference array, so the content counts cost O(n K) beyond the marginals), the boundary
 * items, the lengths and the label pairs across each boundary.
 *
 * @param model - The model.
 * @param w - The weights (see {@link semiMarkovLayout}).
 * @param x - The item attributes.
 * @returns The expected counts (layout order) and log Z(x).
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovExpectedFeatureCounts, semiMarkovLayout } from 'omai-engine/pgm'
 * const model = semiMarkovCrf(['A'], ['f'], 2)
 * const r = semiMarkovExpectedFeatureCounts(model, new Float64Array(semiMarkovLayout(model).size), [{ f: 1 }, { f: 1 }])
 * [Math.round(Math.exp(r.logPartition)), Math.round(r.counts[0] ?? 0)] // => [2, 2]
 * ```
 */
export function semiMarkovExpectedFeatureCounts(
  model: SemiMarkovCrf,
  w: ArrayLike<number>,
  x: SegmentInput,
): { readonly counts: Float64Array; readonly logPartition: number } {
  const layout = semiMarkovLayout(model)
  const K = model.labels.length
  const L = model.maxLength
  const n = x.length
  const items = itemsOf(model, x)
  const out = new Float64Array(layout.size)
  if (n === 0) return { counts: out, logPartition: 0 }
  const e = expectation(scoresOf(model, layout, w, items))
  const diff = new Float64Array((n + 1) * K)
  const first = new Float64Array(n * K)
  const last = new Float64Array(n * K)
  for (const { s, e: end, k, p } of e.segments) {
    diff[s * K + k] = (diff[s * K + k] ?? 0) + p
    diff[end * K + k] = (diff[end * K + k] ?? 0) - p
    first[s * K + k] = (first[s * K + k] ?? 0) + p
    last[(end - 1) * K + k] = (last[(end - 1) * K + k] ?? 0) + p
    out[layout.length + k * L + (end - s - 1)] = (out[layout.length + k * L + (end - s - 1)] ?? 0) + p
  }
  const inside = new Float64Array(K)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < K; k++) inside[k] = (inside[k] ?? 0) + (diff[i * K + k] ?? 0)
    for (const [a, v] of items[i] ?? [])
      for (let k = 0; k < K; k++) {
        out[layout.content + a * K + k] = (out[layout.content + a * K + k] ?? 0) + v * (inside[k] ?? 0)
        if (model.boundary) {
          out[layout.begin + a * K + k] = (out[layout.begin + a * K + k] ?? 0) + v * (first[i * K + k] ?? 0)
          out[layout.end + a * K + k] = (out[layout.end + a * K + k] ?? 0) + v * (last[i * K + k] ?? 0)
        }
      }
  }
  for (const m of e.transitions)
    for (let q = 0; q < m.length; q++) out[layout.transition + q] = (out[layout.transition + q] ?? 0) + (m[q] ?? 0)
  return { counts: out, logPartition: e.logZ }
}

/**
 * The highest-scoring segmentation (Viterbi over segment ends) and its score.
 *
 * @param model - The model.
 * @param w - The weights (see {@link semiMarkovLayout}).
 * @param x - The item attributes.
 * @returns The segmentation and its unnormalised score.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovDecode, semiMarkovLayout } from 'omai-engine/pgm'
 * const model = semiMarkovCrf(['A', 'B'], ['f'], 3)
 * const w = new Float64Array(semiMarkovLayout(model).size)
 * w[semiMarkovLayout(model).length + 2] = 1 // a length-3 segment of A scores 1
 * semiMarkovDecode(model, w, [{ f: 1 }, { f: 1 }, { f: 1 }]).y // => [{ label: 0, start: 0, end: 3 }]
 * ```
 */
export function semiMarkovDecode(
  model: SemiMarkovCrf,
  w: ArrayLike<number>,
  x: SegmentInput,
): { readonly y: Segmentation; readonly score: number } {
  return bestSegmentation(scoresOf(model, semiMarkovLayout(model), w, itemsOf(model, x)))
}

/** The label of every item under a segmentation. */
function tokenLabels(y: Segmentation, n: number): number[] {
  const out = new Array<number>(n).fill(-1)
  for (const s of y) for (let i = s.start; i < s.end; i++) out[i] = s.label
  return out
}

/**
 * The BIO tags of a segmentation: `B-X` on the first item of a segment labelled X and `I-X` on the others, and `O` on
 * every item of a segment with the `outside` label.
 *
 * @param labels - The segment label names.
 * @param y - The segmentation.
 * @param options - The name of the outside label, if any.
 * @returns One tag per item.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { segmentationToTags } from 'omai-engine/pgm'
 * segmentationToTags(['O', 'PER'], [{ label: 1, start: 0, end: 2 }, { label: 0, start: 2, end: 3 }], { outside: 'O' }) // => ['B-PER', 'I-PER', 'O']
 * ```
 */
export function segmentationToTags(
  labels: readonly string[],
  y: Segmentation,
  options: { readonly outside?: string } = {},
): string[] {
  const out: string[] = []
  for (const s of y) {
    const name = labels[s.label] ?? ''
    for (let i = s.start; i < s.end; i++)
      out.push(name === options.outside ? 'O' : `${i === s.start ? 'B' : 'I'}-${name}`)
  }
  return out
}

/**
 * The segmentation a BIO tag sequence encodes (entities by the rules of `entitySpans`): each entity is a segment of its
 * type, and each other item is a segment of length 1 with the `outside` label, so converting back gives the same tags.
 *
 * @param labels - The segment label names.
 * @param tags - One tag per item (`B-X`, `I-X`, `O`).
 * @param options - The name of the outside label (needed when the tags contain `O`).
 * @returns The segmentation.
 * @throws OmaiError with code `value` for an entity type or an `O` that has no label.
 * @concept semi-markov-crf
 * @example
 * ```ts
 * import { tagsToSegmentation } from 'omai-engine/pgm'
 * tagsToSegmentation(['O', 'PER'], ['B-PER', 'I-PER', 'O'], { outside: 'O' }) // => [{ label: 1, start: 0, end: 2 }, { label: 0, start: 2, end: 3 }]
 * ```
 */
export function tagsToSegmentation(
  labels: readonly string[],
  tags: readonly string[],
  options: { readonly outside?: string } = {},
): Segment[] {
  const out: Segment[] = []
  const outside = options.outside === undefined ? -1 : labels.indexOf(options.outside)
  const fill = (from: number, to: number): void => {
    if (to > from && outside < 0) throw new OmaiError('value', 'tagsToSegmentation: O tags need an outside label')
    for (let i = from; i < to; i++) out.push({ label: outside, start: i, end: i + 1 })
  }
  let at = 0
  for (const span of entitySpans(tags)) {
    const label = labels.indexOf(span.type)
    if (label < 0) throw new OmaiError('value', `tagsToSegmentation: unknown entity type ${span.type}`)
    fill(at, span.start)
    out.push({ label, start: span.start, end: span.end })
    at = span.end
  }
  fill(at, tags.length)
  return out
}

/** Options of {@link semiMarkovOracle}. */
export interface SemiMarkovOracleOptions {
  /**
   * The task loss: `token` (default), the number of items whose label differs, or `segment`, the number of predicted
   * segments that are not in the gold segmentation. Cost-augmented decoding adds the same loss to the score.
   */
  readonly loss?: 'token' | 'segment'
}

/**
 * The structured oracle of a semi-Markov CRF over item attributes and segmentations. `expectations` are exact (the
 * forward and backward passes over segment ends) and `decode` is Viterbi over segmentations; with `against` the score of
 * each candidate segment is raised by its loss against the gold segmentation, which decomposes over segments for both
 * the token loss (the items of the segment whose gold label differs, from per-label prefix counts) and the segment loss,
 * so the cost-augmented maximum is exact. The weights follow {@link semiMarkovLayout}.
 *
 * To train through the solver slot give it to the generic structured objective, the negative log-likelihood
 * `logZ - w . phi(x, y)` with gradient `E[phi] - phi(x, y)`; this module has no solver of its own.
 *
 * @param model - The model.
 * @param options - The task loss.
 * @returns The oracle.
 * @concept semi-markov-crf
 * @index estimator Semi-Markov CRF oracle: segment features, exact expectations, cost-augmented Viterbi over segmentations.
 * @example
 * ```ts
 * import { semiMarkovCrf, semiMarkovOracle } from 'omai-engine/pgm'
 * const oracle = semiMarkovOracle(semiMarkovCrf(['A', 'B'], ['f'], 2))
 * Math.round(Math.exp(oracle.expectations(new Float64Array(oracle.size), [{ f: 1 }, { f: 1 }]).logPartition)) // => 6
 * ```
 */
export function semiMarkovOracle(
  model: SemiMarkovCrf,
  options: SemiMarkovOracleOptions = {},
): StructuredOracle<SegmentInput, Segmentation> {
  const layout = semiMarkovLayout(model)
  const K = model.labels.length
  const kind = options.loss ?? 'token'
  const costAgainst = (gold: Segmentation, n: number): Cost => {
    checkSegmentation(model, n, gold, 'semiMarkovOracle.decode')
    if (kind === 'segment') {
      const keys = new Set(gold.map((s) => `${s.start}:${s.end}:${s.label}`))
      return (s, e, k) => (keys.has(`${s}:${e}:${k}`) ? 0 : 1)
    }
    const goldTokens = tokenLabels(gold, n)
    const agree = new Int32Array((n + 1) * K)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) agree[(i + 1) * K + k] = (agree[i * K + k] ?? 0) + (goldTokens[i] === k ? 1 : 0)
    return (s, e, k) => e - s - ((agree[e * K + k] ?? 0) - (agree[s * K + k] ?? 0))
  }
  return {
    size: layout.size,
    features: (x, y) => semiMarkovFeatureCounts(model, x, y),
    expectations: (w, x) => semiMarkovExpectedFeatureCounts(model, w, x),
    decode: (w, x, decodeOptions = {}) => {
      const sc = scoresOf(model, layout, w, itemsOf(model, x))
      return bestSegmentation(
        sc,
        decodeOptions.against === undefined ? undefined : costAgainst(decodeOptions.against, x.length),
      )
    },
    loss: (gold, predicted) => {
      if (kind === 'segment') {
        const keys = new Set(gold.map((s) => `${s.start}:${s.end}:${s.label}`))
        return predicted.reduce((c, s) => c + (keys.has(`${s.start}:${s.end}:${s.label}`) ? 0 : 1), 0)
      }
      const n = gold.at(-1)?.end ?? 0
      const g = tokenLabels(gold, n)
      const p = tokenLabels(predicted, n)
      return g.reduce((c, l, i) => c + (l === p[i] ? 0 : 1), 0)
    },
  }
}
