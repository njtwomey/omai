/**
 * Cross-validation splitters for dependent data, where random folds would leak information: time-series splits with
 * expanding, sliding and blocked windows, purged k-fold with an embargo for overlapping labels (de Prado), and spatial
 * and cluster block splits that test on unseen regions. Also conversions between splitters and lists of index pairs.
 */

import { MdmlError } from '../../base/index.ts'
import { kmeansCodebook } from '../../numerics/spatial/index.ts'
import { foldIn } from '../../random/index.ts'
import { fromStorage } from '../../tensor/index.ts'
import { integerOption } from '../validation.ts'
import { type Fold, groupKFold, kFold, type SplitInput, type Splitter } from './splitters.ts'

function range(a: number, b: number): Int32Array {
  return Int32Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i)
}

/** Options of {@link timeSeriesSplit}. */
export interface TimeSeriesSplitOptions {
  /** The most training samples (the most recent ones); unlimited by default (an expanding window). */
  readonly maxTrainSize?: number
  /** Samples per test fold (default n / (nSplits + 1)). */
  readonly testSize?: number
  /** Samples left out between the end of training and the start of the test (default 0). */
  readonly gap?: number
}

/**
 * Expanding-window splits for ordered data (sklearn.model_selection.TimeSeriesSplit): the test folds are the last
 * `nSplits` blocks of `testSize` samples and each trains on everything before it (minus `gap`), or on the most recent
 * `maxTrainSize` samples.
 *
 * @param nSplits - The number of splits.
 * @param options - maxTrainSize, testSize and gap.
 * @returns The splitter.
 * @concept time-series-cross-validation
 * @example
 * ```ts
 * import { splitIndices, timeSeriesSplit } from 'mdml-engine/learn/model-selection'
 * splitIndices(timeSeriesSplit(3), { n: 8 }).map((f) => [f.train.length, Array.from(f.test)]) // => [[2, [2, 3]], [4, [4, 5]], [6, [6, 7]]]
 * ```
 */
export function timeSeriesSplit(nSplits = 5, options: TimeSeriesSplitOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'timeSeriesSplit', 2)
  const gap = options.gap ?? 0
  return {
    name: 'time-series-split',
    randomised: false,
    partition: false,
    nSplits: () => nSplits,
    split: (input) => {
      const n = input.n
      const testSize = options.testSize ?? Math.floor(n / (nSplits + 1))
      if (nSplits + 1 > n) throw new MdmlError('value', `timeSeriesSplit: ${nSplits + 1} folds for ${n} samples`)
      if (n - gap - testSize * nSplits <= 0 || testSize < 1) {
        throw new MdmlError('value', `timeSeriesSplit: too many splits for ${n} samples (test ${testSize}, gap ${gap})`)
      }
      const folds: Fold[] = []
      for (let start = n - nSplits * testSize; start < n; start += testSize) {
        const end = start - gap
        const from = options.maxTrainSize !== undefined && options.maxTrainSize < end ? end - options.maxTrainSize : 0
        folds.push({ train: range(from, end), test: range(start, start + testSize) })
      }
      return folds
    },
  }
}

/** Options of {@link slidingWindowSplit} and {@link expandingWindowSplit}. */
export interface WindowOptions {
  /** Samples per test window (default 1). */
  readonly testSize?: number
  /** Samples skipped between training and test (default 0). */
  readonly gap?: number
  /** How far each split moves on (default `testSize`). */
  readonly step?: number
}

/**
 * Fixed-length rolling windows (forecasting backtests, sktime's SlidingWindowSplitter): train on `trainSize`
 * consecutive samples, test on the `testSize` after a gap, then slide on by `step`; as many windows as fit.
 *
 * @param trainSize - The training window length.
 * @param options - testSize, gap and step.
 * @returns The splitter.
 * @example
 * ```ts
 * import { slidingWindowSplit, splitIndices } from 'mdml-engine/learn/model-selection'
 * splitIndices(slidingWindowSplit(3, { testSize: 2 }), { n: 8 }).map((f) => Array.from(f.train)) // => [[0, 1, 2], [2, 3, 4]]
 * ```
 */
export function slidingWindowSplit(trainSize: number, options: WindowOptions = {}): Splitter {
  integerOption(trainSize, 'trainSize', 'slidingWindowSplit', 1)
  return windowSplitter('sliding-window', trainSize, options, false)
}

/**
 * Expanding windows from a fixed origin (sktime's ExpandingWindowSplitter): train on the first `initial` samples, test
 * on the next `testSize` after a gap, then grow the training window by `step`.
 *
 * @param initial - The first training window length.
 * @param options - testSize, gap and step.
 * @returns The splitter.
 * @example
 * ```ts
 * import { expandingWindowSplit, splitIndices } from 'mdml-engine/learn/model-selection'
 * splitIndices(expandingWindowSplit(3, { testSize: 2 }), { n: 8 }).map((f) => f.train.length) // => [3, 5]
 * ```
 */
export function expandingWindowSplit(initial: number, options: WindowOptions = {}): Splitter {
  integerOption(initial, 'initial', 'expandingWindowSplit', 1)
  return windowSplitter('expanding-window', initial, options, true)
}

function windowSplitter(name: string, size: number, options: WindowOptions, expanding: boolean): Splitter {
  const testSize = integerOption(options.testSize ?? 1, 'testSize', name, 1)
  const gap = integerOption(options.gap ?? 0, 'gap', name, 0)
  const step = integerOption(options.step ?? testSize, 'step', name, 1)
  const folds = (n: number): Fold[] => {
    const out: Fold[] = []
    for (let s = 0; s + size + gap + testSize <= n; s += step) {
      const trainEnd = s + size
      out.push({ train: range(expanding ? 0 : s, trainEnd), test: range(trainEnd + gap, trainEnd + gap + testSize) })
    }
    if (out.length === 0) throw new MdmlError('value', `${name}: no window fits in ${n} samples`)
    return out
  }
  return {
    name,
    randomised: false,
    partition: false,
    nSplits: (input) => folds(input.n).length,
    split: (input) => folds(input.n),
  }
}

/** Options of {@link blockedTimeSeriesSplit}. */
export interface BlockedOptions {
  /** The fraction of each block used for training (default 0.8). */
  readonly trainFraction?: number
  /** Samples skipped between a block's training and test parts (default 0). */
  readonly gap?: number
}

/**
 * Blocked time-series splits: the series is cut into `nSplits` disjoint contiguous blocks and each block trains on its
 * first part and tests on the rest, so no fold's training data comes after another fold's test data within a block
 * and folds never share samples.
 *
 * @param nSplits - The number of blocks.
 * @param options - trainFraction and gap.
 * @returns The splitter.
 * @example
 * ```ts
 * import { blockedTimeSeriesSplit, splitIndices } from 'mdml-engine/learn/model-selection'
 * splitIndices(blockedTimeSeriesSplit(2, { trainFraction: 0.6 }), { n: 10 }).map((f) => Array.from(f.test)) // => [[3, 4], [8, 9]]
 * ```
 */
export function blockedTimeSeriesSplit(nSplits = 5, options: BlockedOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'blockedTimeSeriesSplit', 1)
  const frac = options.trainFraction ?? 0.8
  const gap = options.gap ?? 0
  return {
    name: 'blocked-time-series-split',
    randomised: false,
    partition: false,
    nSplits: () => nSplits,
    split: (input) => {
      const k = Math.floor(input.n / nSplits)
      if (k < 2) throw new MdmlError('value', `blockedTimeSeriesSplit: blocks of ${k} samples are too short`)
      return Array.from({ length: nSplits }, (_, b) => {
        const start = b * k
        const stop = b === nSplits - 1 ? input.n : start + k
        const mid = start + Math.max(1, Math.floor(frac * (stop - start)))
        if (mid + gap >= stop) throw new MdmlError('value', 'blockedTimeSeriesSplit: no test samples left in a block')
        return { train: range(start, mid), test: range(mid + gap, stop) }
      })
    },
  }
}

/** Options of {@link purgedKFold}. */
export interface PurgedOptions {
  /** How many later samples each label depends on (default 0): sample i's label spans [i, i + horizon]. */
  readonly horizon?: number
  /** Explicit label end indices (>= i), instead of `horizon`. */
  readonly labelEnds?: ArrayLike<number>
  /** Samples (a count, or a fraction of n below 1) dropped from training right after each test fold (default 0). */
  readonly embargo?: number
}

/**
 * k-fold for overlapping labels (de Prado's purged k-fold with embargo): contiguous test folds; training samples whose
 * label span overlaps the test span are purged, and an embargo drops those right after the test fold, so no training
 * label sees test information.
 *
 * @param nSplits - The number of folds.
 * @param options - horizon or labelEnds, and the embargo.
 * @returns The splitter.
 * @reference de Prado (2018), Advances in Financial Machine Learning, chapter 7.
 * @example
 * ```ts
 * import { purgedKFold, splitIndices } from 'mdml-engine/learn/model-selection'
 * const folds = splitIndices(purgedKFold(3, { horizon: 1, embargo: 1 }), { n: 9 })
 * Array.from(folds[1].train) // => [0, 1, 8]
 * ```
 */
export function purgedKFold(nSplits = 5, options: PurgedOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'purgedKFold', 2)
  const base = kFold(nSplits)
  return {
    name: 'purged-k-fold',
    randomised: false,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      const n = input.n
      const ends = Int32Array.from({ length: n }, (_, i) => {
        const e = options.labelEnds === undefined ? i + (options.horizon ?? 0) : (options.labelEnds[i] ?? i)
        if (e < i) throw new MdmlError('value', `purgedKFold: label end ${e} before sample ${i}`)
        return Math.min(e, n - 1)
      })
      const e = options.embargo ?? 0
      const embargo = e > 0 && e < 1 ? Math.ceil(e * n) : Math.floor(e)
      return base.split(input, key).map((fold) => {
        const a = fold.test[0] ?? 0
        let b = a
        for (const i of fold.test) b = Math.max(b, ends[i] ?? i)
        const last = fold.test[fold.test.length - 1] ?? a
        const train = Array.from(fold.train).filter(
          (i) => !(i <= b && (ends[i] ?? i) >= a) && !(i > last && i <= b + embargo),
        )
        return { train: Int32Array.from(train), test: fold.test }
      })
    },
  }
}

function coordinatesOf(input: SplitInput, who: string): { readonly data: ArrayLike<number>; readonly dims: number } {
  const c = input.coordinates
  if (c === undefined) throw new MdmlError('value', `${who}: needs coordinates`)
  if (c.data.length !== input.n * c.dims) throw new MdmlError('shape', `${who}: coordinates must be [n, dims]`)
  return c
}

/** Options of {@link spatialBlockSplit}. */
export interface SpatialBlockOptions {
  /** The side of the square (cubic) blocks, in coordinate units. */
  readonly blockSize: number
  /** Assign blocks to folds at random (default false: balanced, largest blocks first). */
  readonly shuffle?: boolean
}

/**
 * Spatial block cross-validation (Roberts et al. 2017): the coordinates are cut into a grid of blocks of side
 * `blockSize`, and whole blocks are assigned to folds, so nearby (autocorrelated) samples never straddle training and
 * test.
 *
 * @param nSplits - The number of folds.
 * @param options - blockSize and shuffle.
 * @returns The splitter (it needs `coordinates`).
 * @example
 * ```ts
 * import { spatialBlockSplit, splitIndices } from 'mdml-engine/learn/model-selection'
 * const coordinates = { data: [0.1, 0.1, 0.2, 0.3, 5.1, 5.2, 5.3, 5.1], dims: 2 }
 * splitIndices(spatialBlockSplit(2, { blockSize: 1 }), { n: 4, coordinates }).map((f) => Array.from(f.test)) // => [[2, 3], [0, 1]]
 * ```
 */
export function spatialBlockSplit(nSplits: number, options: SpatialBlockOptions): Splitter {
  if (!(options.blockSize > 0)) throw new MdmlError('value', 'spatialBlockSplit: blockSize must be positive')
  const inner = groupKFold(nSplits, { shuffle: options.shuffle ?? false })
  const blocks = (input: SplitInput): Float64Array => {
    const c = coordinatesOf(input, 'spatialBlockSplit')
    const lo = Array.from({ length: c.dims }, (_, d) => {
      let m = Number.POSITIVE_INFINITY
      for (let i = 0; i < input.n; i++) m = Math.min(m, c.data[i * c.dims + d] ?? 0)
      return m
    })
    const ids = new Map<string, number>()
    return Float64Array.from({ length: input.n }, (_, i) => {
      const cell = lo.map((l, d) => Math.floor(((c.data[i * c.dims + d] ?? 0) - l) / options.blockSize)).join(',')
      if (!ids.has(cell)) ids.set(cell, ids.size)
      return ids.get(cell) ?? 0
    })
  }
  return {
    name: 'spatial-block-split',
    randomised: options.shuffle ?? false,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => inner.split({ ...input, groups: blocks(input) }, key),
  }
}

/** Options of {@link clusterBlockSplit}. */
export interface ClusterBlockOptions {
  /** The number of k-means clusters (default 2 * nSplits). */
  readonly nClusters?: number
  /** Assign clusters to folds at random (default false). */
  readonly shuffle?: boolean
}

/**
 * Cluster block cross-validation: k-means on the coordinates (or any features) forms blocks, and whole clusters are
 * assigned to folds, testing extrapolation to unseen regions.
 *
 * @param nSplits - The number of folds.
 * @param options - nClusters and shuffle.
 * @returns The splitter (it needs `coordinates`; the key seeds k-means).
 * @example
 * ```ts
 * import { clusterBlockSplit, foldsAreDisjoint, splitIndices } from 'mdml-engine/learn/model-selection'
 * const coordinates = { data: [0, 0, 0.1, 0, 9, 9, 9.1, 9], dims: 2 }
 * const folds = splitIndices(clusterBlockSplit(2, { nClusters: 2 }), { n: 4, coordinates })
 * folds.map((f) => f.test.length) // => [2, 2]
 * ```
 */
export function clusterBlockSplit(nSplits: number, options: ClusterBlockOptions = {}): Splitter {
  const nClusters = options.nClusters ?? 2 * nSplits
  integerOption(nClusters, 'nClusters', 'clusterBlockSplit', nSplits)
  const inner = groupKFold(nSplits, { shuffle: options.shuffle ?? false })
  return {
    name: 'cluster-block-split',
    randomised: true,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      const c = coordinatesOf(input, 'clusterBlockSplit')
      const data = fromStorage(Float64Array.from(c.data), [input.n, c.dims], 'float64')
      const cb = kmeansCodebook(data, Math.min(nClusters, input.n), { key: foldIn(key, 0) })
      const groups = Float64Array.from(cb.assignments)
      return inner.split({ ...input, groups }, foldIn(key, 1))
    },
  }
}

/**
 * The test folds of any splitter turned into a list of `[train, test]` index pairs: a convenience for code written
 * against sklearn's `split` generator.
 *
 * @param folds - Folds.
 * @returns Plain index arrays.
 * @example
 * ```ts
 * import { asIndexPairs, kFold, splitIndices } from 'mdml-engine/learn/model-selection'
 * asIndexPairs(splitIndices(kFold(2), { n: 2 })) // => [[[1], [0]], [[0], [1]]]
 * ```
 */
export function asIndexPairs(folds: readonly Fold[]): [number[], number[]][] {
  return folds.map((f) => [Array.from(f.train), Array.from(f.test)])
}

/**
 * Turn index pairs (or any precomputed folds) into a splitter.
 *
 * @param pairs - `[train, test]` index pairs.
 * @returns The splitter.
 * @example
 * ```ts
 * import { fromIndexPairs, splitIndices } from 'mdml-engine/learn/model-selection'
 * splitIndices(fromIndexPairs([[[0, 1], [2]]]), { n: 3 })[0].test[0] // => 2
 * ```
 */
export function fromIndexPairs(pairs: readonly (readonly [readonly number[], readonly number[]])[]): Splitter {
  return {
    name: 'index-pairs',
    randomised: false,
    partition: false,
    nSplits: () => pairs.length,
    split: (input) =>
      pairs.map(([train, test]) => {
        for (const i of [...train, ...test]) {
          if (!(Number.isInteger(i) && i >= 0 && i < input.n))
            throw new MdmlError('value', `fromIndexPairs: index ${i} out of range`)
        }
        return { train: Int32Array.from(train), test: Int32Array.from(test) }
      }),
  }
}
