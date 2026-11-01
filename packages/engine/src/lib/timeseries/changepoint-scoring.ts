/**
 * Scoring detected changepoints against true ones: precision and recall within a margin of tolerance (each true
 * changepoint is matched to at most one detection, and a maximum matching is found), the Hausdorff distance between the
 * two sets, and the covering of the true segmentation by the detected one (van den Burg and Williams, 2020). A
 * changepoint is the index of the first observation of a new segment, as `changepointSeries` and the detectors report
 * them; a series of length n has changepoints in 1..n-1.
 */

import { MdmlError } from '../base/index.ts'

/** One true changepoint and the detection matched to it. */
export interface ChangepointMatch {
  readonly truth: number
  readonly detected: number
}

function sortedChangepoints(who: string, xs: readonly number[]): number[] {
  if (xs.some((v) => !Number.isFinite(v))) throw new MdmlError('value', `${who}: changepoints must be finite`)
  return [...xs].sort((a, b) => a - b)
}

/**
 * A maximum one-to-one matching of detections to true changepoints within a margin: each true changepoint gets at most
 * one detection no further than the margin from it, each detection serves at most one, and no matching has more pairs.
 * On a line, going through both lists in order and pairing the first that fit is such a matching.
 *
 * @param truth - The true changepoints.
 * @param detected - The detected changepoints.
 * @param margin - The largest distance (inclusive) at which a detection counts as the true changepoint.
 * @returns The matched pairs, in order.
 * @throws MdmlError with code `value` for a negative margin or a changepoint that is not finite.
 * @example
 * ```ts
 * import { matchChangepoints } from 'mdml-engine/timeseries'
 * matchChangepoints([50, 100], [48, 52, 101], 5) // => [{ truth: 50, detected: 48 }, { truth: 100, detected: 101 }]
 * ```
 */
export function matchChangepoints(
  truth: readonly number[],
  detected: readonly number[],
  margin: number,
): ChangepointMatch[] {
  if (!(margin >= 0)) throw new MdmlError('value', 'matchChangepoints: the margin must be non-negative')
  const t = sortedChangepoints('matchChangepoints', truth)
  const d = sortedChangepoints('matchChangepoints', detected)
  const out: ChangepointMatch[] = []
  let i = 0
  let j = 0
  while (i < t.length && j < d.length) {
    const [a, b] = [t[i] ?? 0, d[j] ?? 0]
    if (Math.abs(a - b) <= margin) {
      out.push({ truth: a, detected: b })
      i++
      j++
    } else if (b < a) j++
    else i++
  }
  return out
}

/** The counts and rates of {@link changepointPrecisionRecall}. */
export interface ChangepointScores {
  /** Detections matched to a true changepoint. */
  readonly truePositives: number
  /** Detections matched to none. */
  readonly falsePositives: number
  /** True changepoints no detection matched. */
  readonly falseNegatives: number
  /** The share of detections that are right (1 when there are none: nothing wrong was claimed). */
  readonly precision: number
  /** The share of true changepoints found (1 when there are none to find). */
  readonly recall: number
  /** Their harmonic mean (0 when both are 0). */
  readonly f1: number
}

/**
 * Precision, recall and F1 of detected changepoints within a margin of tolerance, with each true changepoint matched to
 * at most one detection (see {@link matchChangepoints}); extra detections near an already matched one are false
 * positives. With no detections precision is 1, with no true changepoints recall is 1.
 *
 * @param truth - The true changepoints.
 * @param detected - The detected changepoints.
 * @param margin - The largest distance (inclusive) at which a detection counts as the true changepoint.
 * @returns The counts, precision, recall and F1.
 * @throws MdmlError with code `value` for a negative margin or a changepoint that is not finite.
 * @example
 * ```ts
 * import { changepointPrecisionRecall } from 'mdml-engine/timeseries'
 * const s = changepointPrecisionRecall([50, 100, 150], [48, 52, 101], 5)
 * [s.truePositives, s.falsePositives, s.falseNegatives] // => [2, 1, 1]
 * ```
 */
export function changepointPrecisionRecall(
  truth: readonly number[],
  detected: readonly number[],
  margin: number,
): ChangepointScores {
  const tp = matchChangepoints(truth, detected, margin).length
  const precision = detected.length === 0 ? 1 : tp / detected.length
  const recall = truth.length === 0 ? 1 : tp / truth.length
  return {
    truePositives: tp,
    falsePositives: detected.length - tp,
    falseNegatives: truth.length - tp,
    precision,
    recall,
    f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
  }
}

/**
 * The Hausdorff distance between the true and the detected changepoints: the largest distance from a changepoint of
 * one set to the nearest of the other, taken both ways. Given the series length `n`, the series' ends 0 and n are added
 * to both sets (as the change point benchmark does), so a detector that finds nothing is as far from the truth as the
 * farthest true changepoint is from an end; without it, one empty set against a non-empty one is infinitely far.
 *
 * @param truth - The true changepoints.
 * @param detected - The detected changepoints.
 * @param n - The series length, to add the ends 0 and n to both sets (optional).
 * @returns The distance in steps (0 when both sets are empty).
 * @throws MdmlError with code `value` for a changepoint that is not finite.
 * @example
 * ```ts
 * import { changepointHausdorff } from 'mdml-engine/timeseries'
 * [changepointHausdorff([50, 100], [48, 101]), changepointHausdorff([50, 100], [], 120)] // => [2, 50]
 * ```
 */
export function changepointHausdorff(truth: readonly number[], detected: readonly number[], n?: number): number {
  const ends = n === undefined ? [] : [0, n]
  const a = sortedChangepoints('changepointHausdorff', [...truth, ...ends])
  const b = sortedChangepoints('changepointHausdorff', [...detected, ...ends])
  if (a.length === 0 && b.length === 0) return 0
  if (a.length === 0 || b.length === 0) return Number.POSITIVE_INFINITY
  const reach = (from: readonly number[], to: readonly number[]): number =>
    Math.max(...from.map((x) => Math.min(...to.map((y) => Math.abs(x - y)))))
  return Math.max(reach(a, b), reach(b, a))
}

/**
 * The covering of the true segmentation by the detected one (van den Burg and Williams, 2020): each true segment is
 * credited with its best Jaccard overlap among the detected segments, weighted by its length. It is 1 when the
 * segmentations agree and falls as detected segments cut or merge the true ones.
 *
 * @param truth - The true changepoints.
 * @param detected - The detected changepoints.
 * @param n - The series length.
 * @returns The covering, in (0, 1].
 * @throws MdmlError with code `value` for a series length below 1 or a changepoint outside 1..n-1.
 * @example
 * ```ts
 * import { changepointCovering } from 'mdml-engine/timeseries'
 * [changepointCovering([50], [50], 100), changepointCovering([50], [], 100)] // => [1, 0.5]
 * ```
 */
export function changepointCovering(truth: readonly number[], detected: readonly number[], n: number): number {
  if (!(Number.isInteger(n) && n >= 1))
    throw new MdmlError('value', 'changepointCovering: n must be a positive integer')
  const segments = (who: string, cps: readonly number[]): [number, number][] => {
    const inside = sortedChangepoints(who, cps)
    if (inside.some((c) => c < 1 || c > n - 1)) throw new MdmlError('value', `${who}: changepoints must lie in 1..n-1`)
    const bounds = [0, ...new Set(inside), n]
    return bounds.slice(0, -1).map((s, k) => [s, bounds[k + 1] ?? n])
  }
  const real = segments('changepointCovering', truth)
  const found = segments('changepointCovering', detected)
  let total = 0
  for (const [a0, a1] of real) {
    let best = 0
    for (const [b0, b1] of found) {
      const overlap = Math.max(0, Math.min(a1, b1) - Math.max(a0, b0))
      best = Math.max(best, overlap / (a1 - a0 + (b1 - b0) - overlap))
    }
    total += (a1 - a0) * best
  }
  return total / n
}
