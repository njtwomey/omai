/**
 * Resampling for imbalanced classes, imbalanced-learn style: resamplers change the training rows (and are applied only
 * during fit in a pipeline). Random under- and over-sampling, SMOTE and its borderline variant and ADASYN, which
 * synthesise minority samples by interpolating between neighbours, Tomek link removal and NearMiss under-sampling.
 */

import { OmaiError } from '../../base/index.ts'
import { beta, choice as drawChoice, foldIn, type Key, randint, uniform } from '../../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'
import { keyFor } from '../_shared.ts'
import type { FitOptions, LearnData } from '../estimator.ts'
import { encodeLabels, type MatrixView, matrixView, takeRows } from '../validation.ts'

/** The result of resampling: new rows and labels, and for each row the original row it copies (-1 if synthetic). */
export interface Resampled {
  readonly x: Tensor
  readonly y: Tensor
  readonly indices: Int32Array
}

/**
 * A resampler changes the rows of the training data (imbalanced-learn's samplers). It is applied only while fitting:
 * in a pipeline it resamples the data passed to the next steps and is skipped when predicting.
 */
export interface Resampler {
  readonly name: string
  readonly kind: 'resampler'
  fitResample(data: LearnData, options?: FitOptions): Resampled
}

/**
 * Which classes to resample and to how many samples: imbalanced-learn's `sampling_strategy` as `'auto'` (all but the
 * minority when under-sampling, all but the majority when over-sampling), a named set of classes, a ratio (binary:
 * minority / majority after resampling) or explicit counts by label.
 */
export type SamplingStrategy =
  | 'auto'
  | 'minority'
  | 'majority'
  | 'not minority'
  | 'not majority'
  | 'all'
  | number
  | Readonly<Record<number, number>>

/**
 * Whether a value is a resampler.
 *
 * @param x - Any value.
 * @returns `true` for resamplers.
 * @example
 * ```ts
 * import { isResampler, randomUnderSampler } from 'omai-engine/learn/preprocessing'
 * isResampler(randomUnderSampler()) // => true
 * ```
 */
export function isResampler(x: unknown): x is Resampler {
  return typeof x === 'object' && x !== null && (x as Partial<Resampler>).kind === 'resampler'
}

interface Classes {
  readonly m: MatrixView
  readonly labels: Float64Array
  readonly classes: readonly number[]
  readonly members: number[][]
}

function classesOf(data: LearnData, who: string): Classes {
  const m = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, m.n)
  const members = classes.map(() => [] as number[])
  codes.forEach((c, i) => {
    members[c]?.push(i)
  })
  return { m, labels: Float64Array.from(codes, (c) => classes[c] ?? 0), classes, members }
}

/**
 * The target count of each class under a sampling strategy.
 *
 * @param counts - The class sizes (by class index).
 * @param classes - The class labels.
 * @param strategy - The strategy.
 * @param direction - Under- or over-sampling.
 * @returns Target counts by class index (undefined: leave the class alone).
 * @throws OmaiError with code `value` for a ratio with more than two classes or impossible counts.
 * @example
 * ```ts
 * import { samplingTargets } from 'omai-engine/learn/preprocessing'
 * samplingTargets([10, 4, 6], [0, 1, 2], 'auto', 'under') // => [4, undefined, 4]
 * samplingTargets([10, 4], [0, 1], 0.5, 'over') // => [undefined, 5]
 * ```
 */
export function samplingTargets(
  counts: readonly number[],
  classes: readonly number[],
  strategy: SamplingStrategy,
  direction: 'under' | 'over',
): (number | undefined)[] {
  const min = counts.reduce((a, b) => Math.min(a, b))
  const max = counts.reduce((a, b) => Math.max(a, b))
  const minority = counts.indexOf(min)
  const majority = counts.indexOf(max)
  if (typeof strategy === 'number') {
    if (counts.length !== 2) throw new OmaiError('value', 'resampling: a ratio needs exactly two classes')
    return counts.map((_, i) => {
      if (direction === 'over') return i === minority ? Math.floor(strategy * max) : undefined
      return i === majority ? Math.floor(min / strategy) : undefined
    })
  }
  if (typeof strategy === 'object') {
    return classes.map((label, i) => {
      const target = strategy[label]
      if (target === undefined) return undefined
      if (direction === 'under' ? target > (counts[i] ?? 0) : target < (counts[i] ?? 0)) {
        throw new OmaiError('value', `resampling: target ${target} for class ${label} goes the wrong way`)
      }
      return target
    })
  }
  const auto = strategy === 'auto' ? (direction === 'under' ? 'not minority' : 'not majority') : strategy
  const goal = direction === 'under' ? min : max
  return counts.map((_, i) => {
    const chosen =
      auto === 'all' ||
      (auto === 'minority' && i === minority) ||
      (auto === 'majority' && i === majority) ||
      (auto === 'not minority' && i !== minority) ||
      (auto === 'not majority' && i !== majority)
    return chosen ? goal : undefined
  })
}

function assemble(
  c: Classes,
  rows: readonly number[],
  synthetic: readonly number[][],
  syntheticLabels: readonly number[],
): Resampled {
  const base = takeRows(c.m, rows)
  const n = rows.length + synthetic.length
  const out = new Float64Array(n * c.m.d)
  out.set(base.data)
  synthetic.forEach((r, s) => {
    out.set(r, (rows.length + s) * c.m.d)
  })
  return {
    x: fromStorage(out, [n, c.m.d], 'float64'),
    y: fromStorage(Float64Array.from([...rows.map((i) => c.labels[i] ?? 0), ...syntheticLabels]), [n], 'float64'),
    indices: Int32Array.from([...rows, ...synthetic.map(() => -1)]),
  }
}

/** Options of the random samplers. */
export interface RandomSamplerOptions {
  readonly samplingStrategy?: SamplingStrategy
}

/**
 * Random under-sampling (imblearn RandomUnderSampler): each resampled class keeps a random subset of its rows.
 *
 * @param options - The sampling strategy (default `'auto'`).
 * @returns The resampler.
 * @example
 * ```ts
 * import { randomUnderSampler } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const r = randomUnderSampler().fitResample({ x: [[0], [1], [2], [3], [4]], y: [0, 0, 0, 1, 1] })
 * toFlat(r.y) // => [0, 0, 1, 1]
 * ```
 */
export function randomUnderSampler(options: RandomSamplerOptions = {}): Resampler {
  return {
    name: 'random-under-sampler',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'randomUnderSampler')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'under',
      )
      const key = keyFor(fit, 'randomUnderSampler')
      const rows = c.members.flatMap((members, k) => {
        const t = targets[k]
        if (t === undefined) return members
        return toFlat(drawChoice(foldIn(key, k), members.length, t, { replace: false }))
          .map((r) => members[r] ?? 0)
          .sort((a, b) => a - b)
      })
      return assemble(c, rows, [], [])
    },
  }
}

/**
 * Random over-sampling (imblearn RandomOverSampler): each resampled class gains rows drawn with replacement from its
 * own rows.
 *
 * @param options - The sampling strategy (default `'auto'`).
 * @returns The resampler.
 * @example
 * ```ts
 * import { randomOverSampler } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const r = randomOverSampler().fitResample({ x: [[0], [1], [2], [3]], y: [0, 0, 0, 1] })
 * toFlat(r.y) // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function randomOverSampler(options: RandomSamplerOptions = {}): Resampler {
  return {
    name: 'random-over-sampler',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'randomOverSampler')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'over',
      )
      const key = keyFor(fit, 'randomOverSampler')
      const extra = c.members.flatMap((members, k) => {
        const t = targets[k]
        if (t === undefined || t <= members.length) return []
        return toFlat(randint(foldIn(key, k), [t - members.length], 0, members.length)).map((r) => members[r] ?? 0)
      })
      return assemble(c, [...Array.from({ length: c.m.n }, (_, i) => i), ...extra], [], [])
    },
  }
}

/** Squared Euclidean distances from row `i` of `a` to every row of `b`. */
function distancesFrom(a: MatrixView, i: number, b: MatrixView): Float64Array {
  const out = new Float64Array(b.n)
  for (let r = 0; r < b.n; r++) {
    let s = 0
    for (let j = 0; j < a.d; j++) s += ((a.data[i * a.d + j] ?? 0) - (b.data[r * b.d + j] ?? 0)) ** 2
    out[r] = s
  }
  return out
}

/** The indices of the k nearest rows of `b` to row i of `a` (ties by index), optionally skipping one index. */
function nearest(a: MatrixView, i: number, b: MatrixView, k: number, skip = -1): number[] {
  const d = distancesFrom(a, i, b)
  return Array.from(d.keys())
    .filter((r) => r !== skip)
    .sort((p, q) => (d[p] ?? 0) - (d[q] ?? 0) || p - q)
    .slice(0, k)
}

/** Interpolate `count` points between rows and their neighbours (imblearn's `_make_samples`). */
function interpolate(
  key: Key,
  base: MatrixView,
  rows: readonly number[],
  neighbours: readonly (readonly number[])[],
  donors: MatrixView,
  count: number,
  stepScale: number,
): number[][] {
  const k = neighbours[0]?.length ?? 0
  if (count <= 0 || rows.length === 0 || k === 0) return []
  const picks = toFlat(randint(foldIn(key, 0), [count], 0, rows.length * k))
  const steps = toFlat(uniform(foldIn(key, 1), [count]))
  return picks.map((p, s) => {
    const r = Math.floor(p / k)
    const i = rows[r] ?? 0
    const nb = neighbours[r]?.[p % k] ?? 0
    const step = stepScale * (steps[s] ?? 0)
    return Array.from({ length: base.d }, (_, j) => {
      const x = base.data[i * base.d + j] ?? 0
      return x + step * ((donors.data[nb * donors.d + j] ?? 0) - x)
    })
  })
}

/** Options of {@link smote} and its variants. */
export interface SmoteOptions {
  readonly samplingStrategy?: SamplingStrategy
  /** Neighbours within the class to interpolate towards (default 5). */
  readonly kNeighbors?: number
}

/**
 * SMOTE (Chawla et al. 2002; imblearn SMOTE): each synthetic sample lies on the segment from a random sample of the
 * class to one of its k nearest neighbours within the class, at a uniform random position.
 *
 * @param options - The sampling strategy and kNeighbors.
 * @returns The resampler.
 * @concept class-imbalance
 * @example
 * ```ts
 * import { smote } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0.1, 0], [0.2, 0.1], [5, 5], [5.1, 5], [5, 5.2], [5.3, 5.1]]
 * const r = smote({ kNeighbors: 2 }).fitResample({ x, y: [1, 1, 1, 0, 0, 0, 0] })
 * toFlat(r.y).filter((v) => v === 1).length // => 4
 * ```
 */
export function smote(options: SmoteOptions = {}): Resampler {
  const k = options.kNeighbors ?? 5
  return {
    name: 'smote',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'smote')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'over',
      )
      const key = keyFor(fit, 'smote')
      const synthetic: number[][] = []
      const labels: number[] = []
      c.members.forEach((members, cls) => {
        const t = targets[cls]
        if (t === undefined || t <= members.length) return
        if (members.length <= k)
          throw new OmaiError('value', `smote: class ${c.classes[cls]} needs more than ${k} samples`)
        const xc = takeRows(c.m, members)
        const nbs = members.map((_, r) => nearest(xc, r, xc, k, r))
        const made = interpolate(foldIn(key, cls), xc, Array.from(members.keys()), nbs, xc, t - members.length, 1)
        synthetic.push(...made)
        labels.push(...made.map(() => c.classes[cls] ?? 0))
      })
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/**
 * The borderline ("danger") samples of a class (Han et al. 2005): those with at least half, but not all, of their m
 * nearest neighbours (over all data, excluding themselves) in other classes.
 *
 * @param x - Features `[n, d]`.
 * @param y - Labels `[n]`.
 * @param label - The class.
 * @param m - Neighbours examined (default 10).
 * @returns Indices (into the data) of the class's danger samples.
 * @example
 * ```ts
 * import { borderlineSamples } from 'omai-engine/learn/preprocessing'
 * const x = [[0], [1], [2], [3], [10], [11]]
 * borderlineSamples(x, [1, 0, 1, 1, 1, 1], 1, 2) // => [0, 2, 3]
 * ```
 */
export function borderlineSamples(x: LearnData['x'], y: LearnData['y'] & {}, label: number, m = 10): number[] {
  const c = classesOf({ x, y }, 'borderlineSamples')
  const cls = c.classes.indexOf(label)
  return (c.members[cls] ?? []).filter((i) => {
    const nb = nearest(c.m, i, c.m, m, i)
    const other = nb.filter((t) => c.labels[t] !== label).length
    return other >= m / 2 && other < m
  })
}

/** Options of {@link borderlineSmote}. */
export interface BorderlineSmoteOptions extends SmoteOptions {
  /** Neighbours examined to find the borderline (default 10). */
  readonly mNeighbors?: number
  /** Borderline-1 interpolates within the class; borderline-2 also towards other classes (half-way). */
  readonly kind?: 'borderline-1' | 'borderline-2'
}

/**
 * Borderline SMOTE (Han et al. 2005; imblearn BorderlineSMOTE): SMOTE from the class's danger samples only. Kind 1
 * interpolates towards neighbours in the class; kind 2 draws a Beta(10, 10) share from them and the rest towards
 * neighbours of other classes, at most half-way.
 *
 * @param options - Sampling strategy, kNeighbors, mNeighbors and kind.
 * @returns The resampler.
 * @example
 * ```ts
 * import { borderlineSmote } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.5], [1], [1.5], [2], [2.5], [3], [1.2], [1.4], [5]]
 * const r = borderlineSmote({ kNeighbors: 2, mNeighbors: 4 }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] })
 * toFlat(r.y).filter((v) => v === 1).length // => 7
 * ```
 */
export function borderlineSmote(options: BorderlineSmoteOptions = {}): Resampler {
  const k = options.kNeighbors ?? 5
  const mn = options.mNeighbors ?? 10
  const kind = options.kind ?? 'borderline-1'
  return {
    name: 'borderline-smote',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'borderlineSmote')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'over',
      )
      const key = keyFor(fit, 'borderlineSmote')
      const synthetic: number[][] = []
      const labels: number[] = []
      c.members.forEach((members, cls) => {
        const t = targets[cls]
        const label = c.classes[cls] ?? 0
        if (t === undefined || t <= members.length) return
        const danger = borderlineSamples(fromRows(c.m), c.labels, label, mn)
        if (danger.length === 0) return
        const xc = takeRows(c.m, members)
        const dangerView = takeRows(c.m, danger)
        const count = t - members.length
        const sameNbs = danger.map((_, r) => nearest(dangerView, r, xc, k + 1).slice(1))
        if (kind === 'borderline-1') {
          synthetic.push(...interpolate(foldIn(key, cls), dangerView, Array.from(danger.keys()), sameNbs, xc, count, 1))
        } else {
          const share = toFlat(beta(foldIn(key, 1000 + cls), [1], 10, 10))[0] ?? 0.5
          const nSame = Math.floor(share * count)
          const others = Array.from({ length: c.m.n }, (_, i) => i).filter((i) => c.labels[i] !== label)
          const xo = takeRows(c.m, others)
          const otherNbs = danger.map((_, r) => nearest(dangerView, r, xo, k))
          synthetic.push(...interpolate(foldIn(key, cls), dangerView, Array.from(danger.keys()), sameNbs, xc, nSame, 1))
          synthetic.push(
            ...interpolate(
              foldIn(key, 2000 + cls),
              dangerView,
              Array.from(danger.keys()),
              otherNbs,
              xo,
              count - nSame,
              0.5,
            ),
          )
        }
        while (labels.length < synthetic.length) labels.push(label)
      })
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

function fromRows(m: MatrixView): Tensor {
  return fromStorage(m.data, [m.n, m.d], 'float64')
}

/**
 * The number of synthetic samples ADASYN generates from each sample of a class (He et al. 2008): proportional to the
 * share of other classes among its k nearest neighbours (over all data), normalised and rounded (numpy's `rint`).
 *
 * @param x - Features.
 * @param y - Labels.
 * @param label - The class.
 * @param total - The number of samples to generate.
 * @param k - Neighbours (default 5).
 * @returns Counts per member of the class, in index order.
 * @throws OmaiError with code `value` when no sample of the class has a neighbour of another class.
 * @example
 * ```ts
 * import { adasynCounts } from 'omai-engine/learn/preprocessing'
 * adasynCounts([[0], [1], [2], [10], [11]], [0, 0, 1, 1, 1], 0, 3, 2) // => [2, 2]
 * ```
 */
export function adasynCounts(x: LearnData['x'], y: LearnData['y'] & {}, label: number, total: number, k = 5): number[] {
  const c = classesOf({ x, y }, 'adasynCounts')
  const members = c.members[c.classes.indexOf(label)] ?? []
  const ratio = members.map((i) => nearest(c.m, i, c.m, k, i).filter((t) => c.labels[t] !== label).length / k)
  const sum = ratio.reduce((a, b) => a + b, 0)
  if (sum === 0) throw new OmaiError('value', 'adasyn: no sample has a neighbour of another class; use SMOTE')
  // numpy's rint rounds half to even.
  return ratio.map((r) => {
    const v = (r / sum) * total
    const f = Math.floor(v)
    return v - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v)
  })
}

/**
 * ADASYN (He et al. 2008; imblearn ADASYN): like SMOTE, but each sample generates in proportion to how many of its
 * neighbours belong to other classes, focusing on the hard regions.
 *
 * @param options - The sampling strategy and kNeighbors.
 * @returns The resampler.
 * @example
 * ```ts
 * import { adasyn } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [1], [2], [3], [4], [5], [2.5], [2.6], [9]]
 * const r = adasyn({ kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1, 1, 1] })
 * toFlat(r.y).length > 9 // => true
 * ```
 */
export function adasyn(options: SmoteOptions = {}): Resampler {
  const k = options.kNeighbors ?? 5
  return {
    name: 'adasyn',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'adasyn')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'over',
      )
      const key = keyFor(fit, 'adasyn')
      const synthetic: number[][] = []
      const labels: number[] = []
      c.members.forEach((members, cls) => {
        const t = targets[cls]
        const label = c.classes[cls] ?? 0
        if (t === undefined || t <= members.length) return
        const counts = adasynCounts(fromRows(c.m), c.labels, label, t - members.length, k)
        const xc = takeRows(c.m, members)
        const kk = Math.min(k, members.length - 1)
        members.forEach((_, r) => {
          const made = interpolate(
            foldIn(foldIn(key, cls), r),
            xc,
            [r],
            [nearest(xc, r, xc, kk, r)],
            xc,
            counts[r] ?? 0,
            1,
          )
          synthetic.push(...made)
          labels.push(...made.map(() => label))
        })
      })
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/**
 * Remove Tomek links (Tomek 1976; imblearn TomekLinks): pairs of mutual nearest neighbours of different classes; the
 * member of a resampled class (by default every class but the minority) is dropped.
 *
 * @param options - The sampling strategy (default `'auto'`, every class but the minority).
 * @returns The resampler.
 * @example
 * ```ts
 * import { tomekLinks } from 'omai-engine/learn/preprocessing'
 * const r = tomekLinks().fitResample({ x: [[0], [0.4], [0.6], [3], [3.2]], y: [0, 0, 1, 0, 0] })
 * Array.from(r.indices) // => [0, 2, 3, 4]
 * ```
 */
export function tomekLinks(options: RandomSamplerOptions = {}): Resampler {
  return {
    name: 'tomek-links',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'tomekLinks')
      const targets = samplingTargets(
        c.members.map((m) => m.length),
        c.classes,
        options.samplingStrategy ?? 'auto',
        'under',
      )
      const nn = Array.from({ length: c.m.n }, (_, i) => nearest(c.m, i, c.m, 1, i)[0] ?? i)
      const keep = Array.from({ length: c.m.n }, (_, i) => i).filter((i) => {
        const cls = c.classes.indexOf(c.labels[i] ?? 0)
        if (targets[cls] === undefined) return true
        const j = nn[i] ?? i
        return !(c.labels[j] !== c.labels[i] && nn[j] === i)
      })
      return assemble(c, keep, [], [])
    },
  }
}

/** Options of {@link nearMiss}. */
export interface NearMissOptions extends RandomSamplerOptions {
  /** The heuristic (default 1). */
  readonly version?: 1 | 2 | 3
  /** Minority neighbours averaged over (default 3). */
  readonly nNeighbors?: number
  /** Majority neighbours kept per minority sample in version 3 (default 3). */
  readonly nNeighborsVer3?: number
}

/**
 * NearMiss under-sampling (Mani and Zhang 2003; imblearn NearMiss): keep the majority samples closest to the minority
 * class, by the summed distance to their nearest minority samples (version 1), to their farthest (version 2), or, in
 * version 3, among the nearest majority neighbours of each minority sample, those farthest from the minority.
 *
 * @param options - version, nNeighbors, nNeighborsVer3 and the sampling strategy.
 * @returns The resampler.
 * @example
 * ```ts
 * import { nearMiss } from 'omai-engine/learn/preprocessing'
 * const r = nearMiss({ nNeighbors: 1 }).fitResample({ x: [[0], [1], [5], [9], [10]], y: [1, 0, 0, 0, 1] })
 * Array.from(r.indices) // => [0, 1, 3, 4]
 * ```
 */
export function nearMiss(options: NearMissOptions = {}): Resampler {
  const version = options.version ?? 1
  const k = options.nNeighbors ?? 3
  const k3 = options.nNeighborsVer3 ?? 3
  return {
    name: 'near-miss',
    kind: 'resampler',
    fitResample: (data) => {
      const c = classesOf(data, 'nearMiss')
      const counts = c.members.map((m) => m.length)
      const targets = samplingTargets(counts, c.classes, options.samplingStrategy ?? 'auto', 'under')
      const minority = counts.indexOf(counts.reduce((a, b) => Math.min(a, b)))
      const minorityRows = c.members[minority] ?? []
      const xmin = takeRows(c.m, minorityRows)
      const score = (view: MatrixView, r: number, nNb: number, fromEnd: boolean): number => {
        const d = Array.from(distancesFrom(view, r, xmin), Math.sqrt).sort((a, b) => a - b)
        const all = d.slice(0, nNb)
        const used = fromEnd ? all.slice(all.length - k) : all
        return used.reduce((a, b) => a + b, 0)
      }
      const rows = c.members.flatMap((members, cls) => {
        const t = targets[cls]
        if (t === undefined) return members
        const xc = takeRows(c.m, members)
        let candidates = Array.from(members.keys())
        if (version === 3) {
          const chosen = new Set<number>()
          minorityRows.forEach((_, r) => {
            for (const nb of nearest(xmin, r, xc, k3)) chosen.add(nb)
          })
          candidates = Array.from(chosen).sort((a, b) => a - b)
        }
        const scored = candidates.map((r) => ({
          r,
          s: version === 2 ? score(xc, r, minorityRows.length, true) : score(xc, r, k, false),
        }))
        scored.sort((a, b) => (version === 3 ? b.s - a.s : a.s - b.s))
        return scored
          .slice(0, t)
          .map((e) => members[e.r] ?? 0)
          .sort((a, b) => a - b)
      })
      return assemble(
        c,
        rows.sort((a, b) => a - b),
        [],
        [],
      )
    },
  }
}
