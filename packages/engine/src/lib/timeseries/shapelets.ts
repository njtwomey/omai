/**
 * Shapelets, the subsequences that tell classes of series apart (Ye and Keogh): the distance from a series to a
 * shapelet is the smallest (optionally z-normalised) mean squared difference over its windows, computed with early
 * abandoning; a candidate shapelet is scored by the information gain of the best threshold split of the series by that
 * distance, ties broken by the margin between the two sides; discovery scans candidates one per step as an
 * `Algorithm` keeping the best non-overlapping ones; and the shapelet transform turns each series into its vector of
 * distances to the chosen shapelets, on which any tabular classifier runs.
 */

import { MdmlError } from '../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** The distance from a series to a shapelet, and where it is attained. */
export interface ShapeletMatch {
  /** The smallest mean squared difference over windows. */
  readonly distance: number
  /** The start of the best window. */
  readonly position: number
  /** The squared differences that early abandoning skipped. */
  readonly skipped: number
}

function znormalise(v: ArrayLike<number>, start: number, length: number): Float64Array {
  let s = 0
  let ss = 0
  for (let i = 0; i < length; i++) {
    const a = v[start + i] ?? 0
    s += a
    ss += a * a
  }
  const mean = s / length
  const sd = Math.sqrt(Math.max(ss / length - mean * mean, 0))
  return Float64Array.from({ length }, (_, i) => (sd > 1e-12 ? ((v[start + i] ?? 0) - mean) / sd : 0))
}

/**
 * The distance from a series to a shapelet: the smallest mean squared difference between the shapelet and a window of
 * the series, with early abandoning (a window's running sum is dropped as soon as it exceeds the best so far) and
 * optional z-normalisation of every window and of the shapelet.
 *
 * @param series - The series.
 * @param shapelet - The shapelet.
 * @param zNormalise - Normalise each window and the shapelet to zero mean and unit variance (default false).
 * @returns The distance, its position and how many squared differences were skipped.
 * @example
 * ```ts
 * import { shapeletDistance } from 'mdml-engine/timeseries'
 * shapeletDistance([0, 0, 1, 2, 1, 0, 0], [1, 2, 1]) // => { distance: 0, position: 2, skipped: 4 }
 * ```
 */
export function shapeletDistance(
  series: ArrayLike<number>,
  shapelet: ArrayLike<number>,
  zNormalise = false,
): ShapeletMatch {
  const m = shapelet.length
  const n = series.length
  if (m < 1 || m > n) throw new MdmlError('value', 'shapeletDistance: the shapelet must be no longer than the series')
  const target = zNormalise ? znormalise(shapelet, 0, m) : shapelet
  let best = Number.POSITIVE_INFINITY
  let position = 0
  let skipped = 0
  for (let s = 0; s + m <= n; s++) {
    const window = zNormalise ? znormalise(series, s, m) : null
    let acc = 0
    let i = 0
    for (; i < m; i++) {
      const d = (window === null ? (series[s + i] ?? 0) : (window[i] ?? 0)) - (target[i] ?? 0)
      acc += d * d
      if (acc >= best * m) {
        i++
        break
      }
    }
    if (i < m) skipped += m - i
    else if (acc / m < best) {
      best = acc / m
      position = s
    }
  }
  return { distance: best, position, skipped }
}

const entropyOf = (counts: readonly number[]): number => {
  const total = counts.reduce((a, b) => a + b, 0)
  let h = 0
  for (const c of counts) if (c > 0) h -= (c / total) * Math.log2(c / total)
  return h
}

/** The best threshold split of series by their distance to a candidate. */
export interface ShapeletSplit {
  /** The information gain in bits. */
  readonly gain: number
  /** The distance threshold: series at or below it are on the near side. */
  readonly threshold: number
  /** The gap between the mean distances of the two sides over the threshold's neighbours (the tie-break). */
  readonly gap: number
}

/**
 * The best split of labelled distances: the threshold between sorted distances (midpoints of neighbours with different
 * labels) that maximises the information gain, ties broken by the larger gap between the sides' mean distances.
 *
 * @param distances - The distance of each series to the candidate.
 * @param labels - The class index of each series.
 * @returns The gain, threshold and gap.
 * @example
 * ```ts
 * import { shapeletSplit } from 'mdml-engine/timeseries'
 * shapeletSplit([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]).gain // => 1
 * ```
 */
export function shapeletSplit(distances: ArrayLike<number>, labels: ArrayLike<number>): ShapeletSplit {
  const n = distances.length
  const classes = Math.max(...Array.from(labels)) + 1
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (distances[a] ?? 0) - (distances[b] ?? 0))
  const total = new Array<number>(classes).fill(0)
  for (const l of Array.from(labels)) total[l] = (total[l] ?? 0) + 1
  const h0 = entropyOf(total)
  const left = new Array<number>(classes).fill(0)
  let best: ShapeletSplit = { gain: 0, threshold: distances[order[0] ?? 0] ?? 0, gap: 0 }
  let leftSum = 0
  const sum = Array.from(distances).reduce((a, b) => a + b, 0)
  for (let k = 0; k < n - 1; k++) {
    const i = order[k] ?? 0
    const l = labels[i] ?? 0
    left[l] = (left[l] ?? 0) + 1
    leftSum += distances[i] ?? 0
    const a = distances[i] ?? 0
    const b = distances[order[k + 1] ?? 0] ?? 0
    if (a === b) continue
    const right = total.map((c, j) => c - (left[j] ?? 0))
    const nl = k + 1
    const gain = h0 - (nl / n) * entropyOf(left) - ((n - nl) / n) * entropyOf(right)
    const gap = (sum - leftSum) / (n - nl) - leftSum / nl
    if (gain > best.gain + 1e-12 || (Math.abs(gain - best.gain) <= 1e-12 && gap > best.gap))
      best = { gain, threshold: (a + b) / 2, gap }
  }
  return best
}

/** A discovered shapelet. */
export interface Shapelet {
  readonly values: readonly number[]
  /** The series it was cut from, and where. */
  readonly source: number
  readonly start: number
  readonly gain: number
  readonly threshold: number
  readonly gap: number
}

/** Options of {@link shapeletDiscoveryAlgorithm}. */
export interface ShapeletDiscoveryOptions {
  /** The shortest candidate (default 3). */
  readonly minLength?: number
  /** The longest candidate (default a quarter of the series length). */
  readonly maxLength?: number
  /** The step between candidate lengths (default 1). */
  readonly lengthStep?: number
  /** The step between candidate starts (default 1). */
  readonly stride?: number
  /** How many shapelets to keep (default 3). */
  readonly k?: number
  /** Z-normalise windows and shapelets (default false). */
  readonly zNormalise?: boolean
}

/** The state of the discovery: the candidate cursor and the best shapelets so far. */
export interface ShapeletDiscoveryState {
  readonly index: number
  readonly best: readonly Shapelet[]
  /** Squared differences skipped by early abandoning so far. */
  readonly skipped: number
  /** Squared differences evaluated or skipped so far. */
  readonly work: number
  readonly lastGain: number
}

const overlaps = (a: Shapelet, b: Shapelet): boolean =>
  a.source === b.source && a.start < b.start + b.values.length && b.start < a.start + a.values.length

/**
 * Shapelet discovery as an algorithm: each step scores one candidate (a window of one training series) by the
 * information gain of its distance split over all training series, and keeps the best `k` that do not overlap each
 * other within a series. The state serialises, so the scan can be resumed or sought.
 *
 * @param series - The training series (equal length is not required).
 * @param labels - The class index of each series (0, 1, ...).
 * @param options - Lengths, strides, how many to keep, normalisation.
 * @returns The algorithm and the number of candidates.
 * @example
 * ```ts
 * import { shapeletDiscoveryAlgorithm } from 'mdml-engine/timeseries'
 * const x = [[0, 0, 1, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
 * shapeletDiscoveryAlgorithm(x, [1, 0, 1, 0], { minLength: 3, maxLength: 3 }).candidates // => 24
 * ```
 */
export function shapeletDiscoveryAlgorithm(
  series: readonly ArrayLike<number>[],
  labels: readonly number[],
  options: ShapeletDiscoveryOptions = {},
): { readonly algorithm: Algorithm<ShapeletDiscoveryState>; readonly candidates: number } {
  if (series.length !== labels.length || series.length < 2)
    throw new MdmlError('shape', 'shapelets: need labelled series')
  const shortest = Math.min(...series.map((s) => s.length))
  const minLength = options.minLength ?? 3
  const maxLength = Math.min(options.maxLength ?? Math.max(minLength, Math.floor(shortest / 4)), shortest)
  const lengthStep = options.lengthStep ?? 1
  const stride = options.stride ?? 1
  const keep = options.k ?? 3
  const z = options.zNormalise ?? false
  const list: { source: number; start: number; length: number }[] = []
  for (let length = minLength; length <= maxLength; length += lengthStep)
    for (let source = 0; source < series.length; source++)
      for (let start = 0; start + length <= (series[source]?.length ?? 0); start += stride)
        list.push({ source, start, length })
  const algorithm: Algorithm<ShapeletDiscoveryState> = {
    name: 'shapelet-discovery',
    config: { minLength, maxLength, lengthStep, stride, k: keep, zNormalise: z, nSeries: series.length },
    schema: channels({
      metrics: {
        candidateGain: { description: 'the information gain of the candidate just scored', direction: 'higher' },
        bestGain: { description: 'the best gain so far', direction: 'higher' },
        skippedShare: { description: 'the share of squared differences early abandoning skipped', direction: 'higher' },
      },
      tensors: {},
    }),
    init: () => ({ index: 0, best: [], skipped: 0, work: 0, lastGain: 0 }),
    step: (s) => {
      const c = list[s.index]
      if (c === undefined) return s
      const values = Array.from({ length: c.length }, (_, i) => series[c.source]?.[c.start + i] ?? 0)
      let skipped = s.skipped
      let work = s.work
      const distances = series.map((x) => {
        const m = shapeletDistance(x, values, z)
        skipped += m.skipped
        work += (x.length - c.length + 1) * c.length
        return m.distance
      })
      const split = shapeletSplit(distances, labels)
      const cand: Shapelet = {
        values,
        source: c.source,
        start: c.start,
        gain: split.gain,
        threshold: split.threshold,
        gap: split.gap,
      }
      const better = (a: Shapelet, b: Shapelet): boolean => a.gain > b.gain || (a.gain === b.gain && a.gap > b.gap)
      let best = [...s.best]
      const clash = best.findIndex((b) => overlaps(b, cand))
      if (clash >= 0) {
        if (better(cand, best[clash] as Shapelet)) best[clash] = cand
      } else best.push(cand)
      best = best.sort((a, b) => (better(a, b) ? -1 : better(b, a) ? 1 : 0)).slice(0, keep)
      return { index: s.index + 1, best, skipped, work, lastGain: split.gain }
    },
    done: (s) => (s.index >= list.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        candidateGain: s.lastGain,
        bestGain: s.best[0]?.gain ?? 0,
        skippedShare: s.work === 0 ? 0 : s.skipped / s.work,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ShapeletDiscoveryState,
  }
  return { algorithm, candidates: list.length }
}

/**
 * Find the shapelets that best separate the classes (Ye and Keogh's information-gain search).
 *
 * @param series - The training series.
 * @param labels - Class indices.
 * @param options - Lengths, strides, how many to keep, normalisation.
 * @returns The best shapelets, best first.
 * @example
 * ```ts
 * import { findShapelets } from 'mdml-engine/timeseries'
 * const bump = [0, 1, 2, 1, 0]
 * const x = [[0, 0, ...bump, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, ...bump, 0, 0, 0], [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0]]
 * findShapelets(x, [1, 0, 1, 0], { minLength: 5, maxLength: 5, k: 1 })[0]?.gain // => 1
 * ```
 */
export function findShapelets(
  series: readonly ArrayLike<number>[],
  labels: readonly number[],
  options: ShapeletDiscoveryOptions = {},
): Shapelet[] {
  const { algorithm, candidates } = shapeletDiscoveryAlgorithm(series, labels, options)
  let state = algorithm.init(undefined as never)
  for (let i = 0; i < candidates; i++) state = algorithm.step(state, undefined as never)
  return [...state.best]
}

/**
 * The shapelet transform: each series becomes its vector of distances to the shapelets.
 *
 * @param series - The series.
 * @param shapelets - The shapelets (as returned by {@link findShapelets}, or plain value arrays).
 * @param zNormalise - Z-normalise windows and shapelets (default false; use the setting of the discovery).
 * @returns One row per series, one column per shapelet.
 * @example
 * ```ts
 * import { shapeletTransform } from 'mdml-engine/timeseries'
 * shapeletTransform([[0, 1, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0]], [[1, 2, 1]]) // => [[0], [2]]
 * ```
 */
export function shapeletTransform(
  series: readonly ArrayLike<number>[],
  shapelets: readonly (Shapelet | ArrayLike<number>)[],
  zNormalise = false,
): number[][] {
  return series.map((x) =>
    shapelets.map((s) => shapeletDistance(x, 'gain' in s ? (s.values as ArrayLike<number>) : s, zNormalise).distance),
  )
}
