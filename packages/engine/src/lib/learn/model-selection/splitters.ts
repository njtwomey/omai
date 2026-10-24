import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { integerOption } from '../validation.ts'

/** What a splitter partitions: the number of samples and, when it needs them, labels, groups or coordinates. */
export interface SplitInput {
  readonly n: number
  /** Class labels (any numbers), for stratified splitters. */
  readonly y?: ArrayLike<number>
  /** Group ids (any numbers), for group splitters. */
  readonly groups?: ArrayLike<number>
  /** Row-major coordinates `[n * dims]` with `dims`, for spatial splitters. */
  readonly coordinates?: { readonly data: ArrayLike<number>; readonly dims: number }
}

/** One split: sorted training and test indices (and validation indices for three-way splits). */
export interface Fold {
  readonly train: Int32Array
  readonly test: Int32Array
  readonly validation?: Int32Array
}

/**
 * The splitter contract: `split(input, key)` returns the folds (a pure function of the input and the key; splitters
 * that do not shuffle ignore the key). `partition` is true when every sample is tested exactly once, as
 * `crossValPredict` needs.
 */
export interface Splitter {
  readonly name: string
  /** Whether the key matters. */
  readonly randomised: boolean
  /** Whether the test sets partition the samples. */
  readonly partition: boolean
  nSplits(input: SplitInput): number
  split(input: SplitInput, key: Key): Fold[]
}

/**
 * The folds of a splitter on some input, with a fixed default key for reproducibility.
 *
 * @param splitter - The splitter.
 * @param input - The number of samples (and labels, groups or coordinates when needed).
 * @param key - The key for shuffling splitters (default `key(0)`).
 * @returns The folds.
 * @example
 * ```ts
 * import { kFold, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(kFold(3), { n: 7 }).map((f) => Array.from(f.test)) // => [[0, 1, 2], [3, 4], [5, 6]]
 * ```
 */
export function splitIndices(splitter: Splitter, input: SplitInput, key: Key = makeKey(0)): Fold[] {
  integerOption(input.n, 'n', splitter.name, 1)
  return splitter.split(input, key)
}

/** Sorted indices as an Int32Array. */
function sorted(xs: Iterable<number>): Int32Array {
  return Int32Array.from(xs).sort()
}

/** The fold whose test set is `test` and whose training set is everything else. */
export function complementFold(n: number, test: Iterable<number>): Fold {
  const inTest = new Uint8Array(n)
  for (const i of test) inTest[i] = 1
  const tr: number[] = []
  const te: number[] = []
  for (let i = 0; i < n; i++) (inTest[i] === 1 ? te : tr).push(i)
  return { train: Int32Array.from(tr), test: Int32Array.from(te) }
}

function shuffled(key: Key, n: number): number[] {
  return toFlat(permutation(key, n))
}

function need<T>(value: T | undefined, what: string, who: string): T {
  if (value === undefined) throw new WhiteboxError('value', `${who}: needs ${what}`)
  return value
}

/** Encode values as 0..k-1 by sorted distinct value; returns codes and counts. */
function encode(values: ArrayLike<number>): { readonly codes: Int32Array; readonly counts: number[] } {
  const uniq = Array.from(new Set(Array.from(values))).sort((a, b) => a - b)
  const index = new Map(uniq.map((v, i) => [v, i]))
  const codes = Int32Array.from(values, (v) => index.get(v) ?? 0)
  const counts = uniq.map(() => 0)
  for (const c of codes) counts[c] = (counts[c] ?? 0) + 1
  return { codes, counts }
}

/** Options of {@link kFold} and its stratified and group variants. */
export interface KFoldOptions {
  /** Shuffle before splitting (default false: contiguous folds, as sklearn). */
  readonly shuffle?: boolean
}

/**
 * k contiguous folds (sklearn.model_selection.KFold): the first n mod k folds have one extra sample; with `shuffle`
 * the samples are permuted first.
 *
 * @param nSplits - The number of folds (at least 2).
 * @param options - Whether to shuffle.
 * @returns The splitter.
 * @concept cross-validation
 * @example
 * ```ts
 * import { kFold, splitIndices } from 'whitebox-engine/learn/model-selection'
 * Array.from(splitIndices(kFold(2), { n: 5 })[0].train) // => [3, 4]
 * ```
 */
export function kFold(nSplits = 5, options: KFoldOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'kFold', 2)
  const shuffle = options.shuffle ?? false
  return {
    name: 'k-fold',
    randomised: shuffle,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      if (nSplits > input.n) throw new WhiteboxError('value', `kFold: ${nSplits} folds for ${input.n} samples`)
      const order = shuffle ? shuffled(key, input.n) : Array.from({ length: input.n }, (_, i) => i)
      const folds: Fold[] = []
      let start = 0
      for (let f = 0; f < nSplits; f++) {
        const size = Math.floor(input.n / nSplits) + (f < input.n % nSplits ? 1 : 0)
        folds.push(complementFold(input.n, order.slice(start, start + size)))
        start += size
      }
      return folds
    },
  }
}

/**
 * k folds preserving the class proportions (sklearn.model_selection.StratifiedKFold): classes ordered by first
 * appearance, the sorted labels dealt round-robin to the folds, and each class's fold assignments laid out in order (or
 * shuffled per class with `shuffle`).
 *
 * @param nSplits - The number of folds (at least 2).
 * @param options - Whether to shuffle.
 * @returns The splitter (it needs `y`).
 * @throws WhiteboxError with code `value` when every class has fewer members than folds.
 * @example
 * ```ts
 * import { splitIndices, stratifiedKFold } from 'whitebox-engine/learn/model-selection'
 * const y = [0, 0, 0, 0, 1, 1]
 * splitIndices(stratifiedKFold(2), { n: 6, y }).map((f) => Array.from(f.test)) // => [[0, 1, 4], [2, 3, 5]]
 * ```
 */
export function stratifiedKFold(nSplits = 5, options: KFoldOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'stratifiedKFold', 2)
  const shuffle = options.shuffle ?? false
  return {
    name: 'stratified-k-fold',
    randomised: shuffle,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      const y = need(input.y, 'labels y', 'stratifiedKFold')
      // Classes in order of first appearance (sklearn's y_encoded).
      const first = new Map<number, number>()
      for (let i = 0; i < input.n; i++) if (!first.has(y[i] ?? 0)) first.set(y[i] ?? 0, first.size)
      const yEnc = Int32Array.from({ length: input.n }, (_, i) => first.get(y[i] ?? 0) ?? 0)
      const k = first.size
      const counts = new Array<number>(k).fill(0)
      for (const c of yEnc) counts[c] = (counts[c] ?? 0) + 1
      if (counts.every((c) => c < nSplits)) {
        throw new WhiteboxError('value', `stratifiedKFold: n_splits=${nSplits} exceeds the members of every class`)
      }
      const order = Array.from(yEnc).sort((a, b) => a - b)
      const allocation = Array.from({ length: nSplits }, (_, f) => {
        const row = new Array<number>(k).fill(0)
        for (let i = f; i < order.length; i += nSplits) row[order[i] ?? 0] = (row[order[i] ?? 0] ?? 0) + 1
        return row
      })
      const testFold = new Int32Array(input.n)
      for (let c = 0; c < k; c++) {
        let folds: number[] = []
        for (let f = 0; f < nSplits; f++) for (let r = 0; r < (allocation[f]?.[c] ?? 0); r++) folds.push(f)
        if (shuffle) {
          const perm = shuffled(foldIn(key, c), folds.length)
          folds = perm.map((p) => folds[p] ?? 0)
        }
        let t = 0
        for (let i = 0; i < input.n; i++) if (yEnc[i] === c) testFold[i] = folds[t++] ?? 0
      }
      return Array.from({ length: nSplits }, (_, f) =>
        complementFold(
          input.n,
          Array.from(testFold.keys()).filter((i) => testFold[i] === f),
        ),
      )
    },
  }
}

/**
 * k folds with no group split across folds (sklearn.model_selection.GroupKFold): groups taken largest first (ties by
 * the later group) and each placed in the currently lightest fold; with `shuffle`, the permuted groups are dealt in
 * nearly equal numbers instead.
 *
 * @param nSplits - The number of folds (at most the number of groups).
 * @param options - Whether to shuffle.
 * @returns The splitter (it needs `groups`).
 * @example
 * ```ts
 * import { groupKFold, splitIndices } from 'whitebox-engine/learn/model-selection'
 * const groups = [1, 1, 1, 2, 2, 3]
 * splitIndices(groupKFold(2), { n: 6, groups }).map((f) => Array.from(f.test)) // => [[0, 1, 2], [3, 4, 5]]
 * ```
 */
export function groupKFold(nSplits = 5, options: KFoldOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'groupKFold', 2)
  const shuffle = options.shuffle ?? false
  return {
    name: 'group-k-fold',
    randomised: shuffle,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      const { codes, counts } = encode(need(input.groups, 'groups', 'groupKFold'))
      const g = counts.length
      if (nSplits > g) throw new WhiteboxError('value', `groupKFold: ${nSplits} folds for ${g} groups`)
      const groupFold = new Int32Array(g)
      if (shuffle) {
        const perm = shuffled(key, g)
        let start = 0
        for (let f = 0; f < nSplits; f++) {
          const size = Math.floor(g / nSplits) + (f < g % nSplits ? 1 : 0)
          for (let r = start; r < start + size; r++) groupFold[perm[r] ?? 0] = f
          start += size
        }
      } else {
        const order = counts
          .map((c, i) => ({ c, i }))
          .sort((a, b) => a.c - b.c || a.i - b.i)
          .reverse()
        const load = new Array<number>(nSplits).fill(0)
        for (const { c, i } of order) {
          let lightest = 0
          for (let f = 1; f < nSplits; f++) if ((load[f] ?? 0) < (load[lightest] ?? 0)) lightest = f
          load[lightest] = (load[lightest] ?? 0) + c
          groupFold[i] = lightest
        }
      }
      return Array.from({ length: nSplits }, (_, f) =>
        complementFold(
          input.n,
          Array.from(codes.keys()).filter((i) => groupFold[codes[i] ?? 0] === f),
        ),
      )
    },
  }
}

function std(xs: readonly number[]): number {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length)
}

/**
 * k group folds that also balance the classes (sklearn.model_selection.StratifiedGroupKFold): groups taken in order of
 * decreasing spread of their class counts, each placed in the fold that minimises the mean (over classes) standard
 * deviation of the per-fold class fractions, ties going to the fold with fewer samples.
 *
 * @param nSplits - The number of folds.
 * @param options - Whether to shuffle the groups first.
 * @returns The splitter (it needs `y` and `groups`).
 * @example
 * ```ts
 * import { splitIndices, stratifiedGroupKFold } from 'whitebox-engine/learn/model-selection'
 * const y = [0, 0, 1, 1, 0, 1, 0, 1]
 * const groups = [1, 1, 2, 2, 3, 3, 4, 4]
 * splitIndices(stratifiedGroupKFold(2), { n: 8, y, groups }).map((f) => f.test.length) // => [4, 4]
 * ```
 */
export function stratifiedGroupKFold(nSplits = 5, options: KFoldOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'stratifiedGroupKFold', 2)
  const shuffle = options.shuffle ?? false
  return {
    name: 'stratified-group-k-fold',
    randomised: shuffle,
    partition: true,
    nSplits: () => nSplits,
    split: (input, key) => {
      const ys = encode(need(input.y, 'labels y', 'stratifiedGroupKFold'))
      const gs = encode(need(input.groups, 'groups', 'stratifiedGroupKFold'))
      const k = ys.counts.length
      const g = gs.counts.length
      if (nSplits > g) throw new WhiteboxError('value', `stratifiedGroupKFold: ${nSplits} folds for ${g} groups`)
      let perGroup = Array.from({ length: g }, () => new Array<number>(k).fill(0))
      for (let i = 0; i < input.n; i++) {
        const row = perGroup[gs.codes[i] ?? 0] as number[]
        row[ys.codes[i] ?? 0] = (row[ys.codes[i] ?? 0] ?? 0) + 1
      }
      let groupOf = Array.from(gs.codes)
      if (shuffle) {
        const perm = shuffled(key, g)
        perGroup = perm.map((p) => perGroup[p] as number[])
        const inv = new Array<number>(g)
        perm.forEach((p, i) => {
          inv[p] = i
        })
        groupOf = groupOf.map((c) => inv[c] ?? 0)
      }
      const order = perGroup.map((row, i) => ({ s: -std(row), i })).sort((a, b) => a.s - b.s || a.i - b.i)
      const perFold = Array.from({ length: nSplits }, () => new Array<number>(k).fill(0))
      const foldOfGroup = new Int32Array(g)
      for (const { i } of order) {
        const row = perGroup[i] as number[]
        let best = -1
        let minEval = Number.POSITIVE_INFINITY
        let minSamples = Number.POSITIVE_INFINITY
        for (let f = 0; f < nSplits; f++) {
          const fold = perFold[f] as number[]
          const stds: number[] = []
          for (let c = 0; c < k; c++) {
            stds.push(std(perFold.map((r, ff) => ((r[c] ?? 0) + (ff === f ? (row[c] ?? 0) : 0)) / (ys.counts[c] ?? 1))))
          }
          const evalF = stds.reduce((a, b) => a + b, 0) / k
          const samples = fold.reduce((a, b) => a + b, 0)
          const close = Math.abs(evalF - minEval) <= 1e-8 + 1e-5 * Math.abs(minEval)
          if (evalF < minEval || (close && samples < minSamples)) {
            minEval = evalF
            minSamples = samples
            best = f
          }
        }
        const fold = perFold[best] as number[]
        for (let c = 0; c < k; c++) fold[c] = (fold[c] ?? 0) + (row[c] ?? 0)
        foldOfGroup[i] = best
      }
      return Array.from({ length: nSplits }, (_, f) =>
        complementFold(
          input.n,
          groupOf.flatMap((c, i) => (foldOfGroup[c] === f ? [i] : [])),
        ),
      )
    },
  }
}

/**
 * Leave one sample out (sklearn.model_selection.LeaveOneOut): n folds, each testing one sample.
 *
 * @returns The splitter.
 * @example
 * ```ts
 * import { leaveOneOut, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(leaveOneOut(), { n: 3 }).map((f) => Array.from(f.train)) // => [[1, 2], [0, 2], [0, 1]]
 * ```
 */
export function leaveOneOut(): Splitter {
  return {
    name: 'leave-one-out',
    randomised: false,
    partition: true,
    nSplits: (input) => input.n,
    split: (input) => {
      if (input.n < 2) throw new WhiteboxError('value', 'leaveOneOut: needs at least 2 samples')
      return Array.from({ length: input.n }, (_, i) => complementFold(input.n, [i]))
    },
  }
}

/** All k-subsets of 0..n-1 in lexicographic order (itertools.combinations), capped. */
function combinations(n: number, k: number, cap: number, who: string): number[][] {
  const out: number[][] = []
  const c = Array.from({ length: k }, (_, i) => i)
  if (k > n) return out
  for (;;) {
    out.push([...c])
    if (out.length > cap) throw new WhiteboxError('value', `${who}: more than ${cap} splits`)
    let i = k - 1
    while (i >= 0 && c[i] === n - k + i) i--
    if (i < 0) return out
    c[i] = (c[i] ?? 0) + 1
    for (let j = i + 1; j < k; j++) c[j] = (c[j - 1] ?? 0) + 1
  }
}

/** The most splits the leave-p-out splitters enumerate. */
export const MAX_COMBINATION_SPLITS = 100000

/**
 * Leave every p-subset out (sklearn.model_selection.LeavePOut), in lexicographic order of the test sets; the test
 * sets overlap, so this is not a partition. Capped at {@link MAX_COMBINATION_SPLITS} splits.
 *
 * @param p - The test size.
 * @returns The splitter.
 * @example
 * ```ts
 * import { leavePOut, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(leavePOut(2), { n: 3 }).map((f) => Array.from(f.test)) // => [[0, 1], [0, 2], [1, 2]]
 * ```
 */
export function leavePOut(p: number): Splitter {
  integerOption(p, 'p', 'leavePOut', 1)
  return {
    name: 'leave-p-out',
    randomised: false,
    partition: p === 1,
    nSplits: (input) => combinations(input.n, p, MAX_COMBINATION_SPLITS, 'leavePOut').length,
    split: (input) => {
      if (input.n <= p) throw new WhiteboxError('value', `leavePOut: p=${p} needs more than ${p} samples`)
      return combinations(input.n, p, MAX_COMBINATION_SPLITS, 'leavePOut').map((c) => complementFold(input.n, c))
    },
  }
}

/**
 * Leave one group out at a time (sklearn.model_selection.LeaveOneGroupOut), groups in sorted order.
 *
 * @returns The splitter (it needs `groups`).
 * @example
 * ```ts
 * import { leaveOneGroupOut, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(leaveOneGroupOut(), { n: 4, groups: [7, 3, 7, 3] }).map((f) => Array.from(f.test)) // => [[1, 3], [0, 2]]
 * ```
 */
export function leaveOneGroupOut(): Splitter {
  return leavePGroupsOut(1, 'leave-one-group-out')
}

/**
 * Leave every p-subset of groups out (sklearn.model_selection.LeavePGroupsOut), subsets of the sorted groups in
 * lexicographic order.
 *
 * @param p - The number of groups per test set.
 * @param name - The splitter's name.
 * @returns The splitter (it needs `groups`).
 * @example
 * ```ts
 * import { leavePGroupsOut, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(leavePGroupsOut(2), { n: 3, groups: [0, 1, 2] }).map((f) => Array.from(f.test)) // => [[0, 1], [0, 2], [1, 2]]
 * ```
 */
export function leavePGroupsOut(p: number, name = 'leave-p-groups-out'): Splitter {
  integerOption(p, 'p', 'leavePGroupsOut', 1)
  return {
    name,
    randomised: false,
    partition: p === 1,
    nSplits: (input) => {
      const { counts } = encode(need(input.groups, 'groups', name))
      return combinations(counts.length, p, MAX_COMBINATION_SPLITS, name).length
    },
    split: (input) => {
      const { codes, counts } = encode(need(input.groups, 'groups', name))
      if (counts.length <= p) throw new WhiteboxError('value', `${name}: needs more than ${p} groups`)
      return combinations(counts.length, p, MAX_COMBINATION_SPLITS, name).map((chosen) => {
        const set = new Set(chosen)
        return complementFold(
          input.n,
          Array.from(codes.keys()).filter((i) => set.has(codes[i] ?? -1)),
        )
      })
    },
  }
}

/**
 * A fixed assignment of samples to test folds (sklearn.model_selection.PredefinedSplit): `testFold[i]` is the fold
 * sample i is tested in, or -1 for always training.
 *
 * @param testFold - The fold of each sample.
 * @returns The splitter (for exactly `testFold.length` samples).
 * @example
 * ```ts
 * import { predefinedSplit, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(predefinedSplit([0, 1, -1, 1]), { n: 4 }).map((f) => Array.from(f.test)) // => [[0], [1, 3]]
 * ```
 */
export function predefinedSplit(testFold: readonly number[]): Splitter {
  const folds = Array.from(new Set(testFold.filter((f) => f !== -1))).sort((a, b) => a - b)
  return {
    name: 'predefined-split',
    randomised: false,
    partition: !testFold.includes(-1),
    nSplits: () => folds.length,
    split: (input) => {
      if (input.n !== testFold.length) {
        throw new WhiteboxError('shape', `predefinedSplit: ${testFold.length} fold ids for ${input.n} samples`)
      }
      return folds.map((f) =>
        complementFold(
          input.n,
          testFold.flatMap((t, i) => (t === f ? [i] : [])),
        ),
      )
    },
  }
}

/**
 * Repeat a shuffling splitter with fresh keys (sklearn RepeatedKFold, RepeatedStratifiedKFold): repetition r uses the
 * key folded with r, so the folds of each repetition differ and the whole is reproducible.
 *
 * @param splitter - A splitter (normally shuffling).
 * @param nRepeats - The number of repetitions.
 * @returns The splitter.
 * @example
 * ```ts
 * import { kFold, repeated, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(repeated(kFold(3, { shuffle: true }), 2), { n: 9 }).length // => 6
 * ```
 */
export function repeated(splitter: Splitter, nRepeats = 10): Splitter {
  integerOption(nRepeats, 'nRepeats', 'repeated', 1)
  return {
    name: `repeated-${splitter.name}`,
    randomised: splitter.randomised,
    partition: false,
    nSplits: (input) => nRepeats * splitter.nSplits(input),
    split: (input, key) => Array.from({ length: nRepeats }, (_, r) => splitter.split(input, foldIn(key, r))).flat(),
  }
}

/**
 * Repeated shuffled k-fold (sklearn.model_selection.RepeatedKFold).
 *
 * @param nSplits - Folds per repetition.
 * @param nRepeats - Repetitions.
 * @returns The splitter.
 * @example
 * ```ts
 * import { repeatedKFold } from 'whitebox-engine/learn/model-selection'
 * repeatedKFold(5, 3).nSplits({ n: 20 }) // => 15
 * ```
 */
export function repeatedKFold(nSplits = 5, nRepeats = 10): Splitter {
  return repeated(kFold(nSplits, { shuffle: true }), nRepeats)
}

/**
 * Repeated shuffled stratified k-fold (sklearn.model_selection.RepeatedStratifiedKFold).
 *
 * @param nSplits - Folds per repetition.
 * @param nRepeats - Repetitions.
 * @returns The splitter (it needs `y`).
 * @example
 * ```ts
 * import { repeatedStratifiedKFold } from 'whitebox-engine/learn/model-selection'
 * repeatedStratifiedKFold(2, 2).nSplits({ n: 8 }) // => 4
 * ```
 */
export function repeatedStratifiedKFold(nSplits = 5, nRepeats = 10): Splitter {
  return repeated(stratifiedKFold(nSplits, { shuffle: true }), nRepeats)
}

/** Train and test sizes: a fraction in (0, 1) of the samples, or a count. */
export interface SizeOptions {
  readonly testSize?: number
  readonly trainSize?: number
}

/**
 * The train and test counts of a split (sklearn's `_validate_shuffle_split`): fractions round the test size up and the
 * train size down; a missing size is the complement.
 *
 * @param n - The number of samples.
 * @param options - The sizes.
 * @param defaultTest - The test size when neither is given.
 * @returns `[nTrain, nTest]`.
 * @throws WhiteboxError with code `value` for sizes that do not fit.
 * @example
 * ```ts
 * import { splitSizes } from 'whitebox-engine/learn/model-selection'
 * splitSizes(10, { testSize: 0.25 }) // => [7, 3]
 * splitSizes(10, { trainSize: 4, testSize: 3 }) // => [4, 3]
 * ```
 */
export function splitSizes(n: number, options: SizeOptions, defaultTest = 0.25): [number, number] {
  const count = (s: number, round: (v: number) => number, what: string): number => {
    if (s > 0 && s < 1) return round(s * n)
    if (Number.isInteger(s) && s >= 1 && s < n) return s
    throw new WhiteboxError('value', `split: ${what}=${s} must be a fraction in (0, 1) or a count in [1, ${n - 1}]`)
  }
  const test = options.testSize ?? (options.trainSize === undefined ? defaultTest : undefined)
  let nTest = test === undefined ? undefined : count(test, Math.ceil, 'testSize')
  let nTrain = options.trainSize === undefined ? undefined : count(options.trainSize, Math.floor, 'trainSize')
  if (nTrain === undefined) nTrain = n - (nTest ?? 0)
  if (nTest === undefined) nTest = n - nTrain
  if (nTrain + nTest > n || nTrain < 1) {
    throw new WhiteboxError('value', `split: train ${nTrain} + test ${nTest} do not fit in ${n} samples`)
  }
  return [nTrain, nTest]
}

/**
 * Random train/test splits (sklearn.model_selection.ShuffleSplit): each split permutes the samples with its own key
 * and takes the first `nTest` as test and the next `nTrain` as training.
 *
 * @param nSplits - The number of splits.
 * @param options - Sizes (default test 0.1).
 * @returns The splitter.
 * @example
 * ```ts
 * import { shuffleSplit, splitIndices } from 'whitebox-engine/learn/model-selection'
 * splitIndices(shuffleSplit(3, { testSize: 0.2 }), { n: 10 }).map((f) => f.test.length) // => [2, 2, 2]
 * ```
 */
export function shuffleSplit(nSplits = 10, options: SizeOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'shuffleSplit', 1)
  return {
    name: 'shuffle-split',
    randomised: true,
    partition: false,
    nSplits: () => nSplits,
    split: (input, key) => {
      const [nTrain, nTest] = splitSizes(input.n, options, 0.1)
      return Array.from({ length: nSplits }, (_, s) => {
        const perm = shuffled(foldIn(key, s), input.n)
        return { test: sorted(perm.slice(0, nTest)), train: sorted(perm.slice(nTest, nTest + nTrain)) }
      })
    },
  }
}

/**
 * Allocate `total` items over classes in proportion to their counts (largest remainders first, ties broken by a
 * key), never exceeding a class's count: sklearn's `_approximate_mode`.
 */
function approximateMode(counts: readonly number[], total: number, key: Key): number[] {
  const n = counts.reduce((a, b) => a + b, 0)
  const exact = counts.map((c) => (c * total) / n)
  const floor = exact.map(Math.floor)
  let left = total - floor.reduce((a, b) => a + b, 0)
  const jitter = toFlat(permutation(key, counts.length))
  const order = exact
    .map((e, i) => ({ r: e - (floor[i] ?? 0), i, j: jitter[i] ?? 0 }))
    .sort((a, b) => b.r - a.r || a.j - b.j)
  for (const { i } of order) {
    if (left <= 0) break
    if ((floor[i] ?? 0) < (counts[i] ?? 0)) {
      floor[i] = (floor[i] ?? 0) + 1
      left--
    }
  }
  return floor
}

/**
 * Random splits preserving class proportions (sklearn.model_selection.StratifiedShuffleSplit): each class contributes
 * its share of the test and training sets (largest remainders, ties at random), drawn by permutation.
 *
 * @param nSplits - The number of splits.
 * @param options - Sizes (default test 0.1).
 * @returns The splitter (it needs `y`).
 * @example
 * ```ts
 * import { splitIndices, stratifiedShuffleSplit } from 'whitebox-engine/learn/model-selection'
 * const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
 * const [f] = splitIndices(stratifiedShuffleSplit(1, { testSize: 0.5 }), { n: 10, y })
 * Array.from(f.test).filter((i) => y[i] === 1).length // => 2
 * ```
 */
export function stratifiedShuffleSplit(nSplits = 10, options: SizeOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'stratifiedShuffleSplit', 1)
  return {
    name: 'stratified-shuffle-split',
    randomised: true,
    partition: false,
    nSplits: () => nSplits,
    split: (input, key) => {
      const { codes, counts } = encode(need(input.y, 'labels y', 'stratifiedShuffleSplit'))
      const [nTrain, nTest] = splitSizes(input.n, options, 0.1)
      if (nTrain < counts.length || nTest < counts.length) {
        throw new WhiteboxError('value', 'stratifiedShuffleSplit: train and test sets must each hold every class')
      }
      return Array.from({ length: nSplits }, (_, s) => {
        const k = foldIn(key, s)
        const testPer = approximateMode(counts, nTest, foldIn(k, 0))
        const rest = counts.map((c, i) => c - (testPer[i] ?? 0))
        const trainPer = approximateMode(rest, nTrain, foldIn(k, 1))
        const train: number[] = []
        const test: number[] = []
        counts.forEach((_, c) => {
          const members = Array.from(codes.keys()).filter((i) => codes[i] === c)
          const perm = shuffled(foldIn(k, 2 + c), members.length).map((p) => members[p] ?? 0)
          test.push(...perm.slice(0, testPer[c] ?? 0))
          train.push(...perm.slice(testPer[c] ?? 0, (testPer[c] ?? 0) + (trainPer[c] ?? 0)))
        })
        return { train: sorted(train), test: sorted(test) }
      })
    },
  }
}

/**
 * Random splits of whole groups (sklearn.model_selection.GroupShuffleSplit): the sizes count groups, not samples.
 *
 * @param nSplits - The number of splits.
 * @param options - Sizes in groups (default test 0.2).
 * @returns The splitter (it needs `groups`).
 * @example
 * ```ts
 * import { groupShuffleSplit, splitIndices } from 'whitebox-engine/learn/model-selection'
 * const groups = [0, 0, 1, 1, 2, 2, 3, 3]
 * const [f] = splitIndices(groupShuffleSplit(1, { testSize: 0.25 }), { n: 8, groups })
 * f.test.length // => 2
 * groups[f.test[0]] === groups[f.test[1]] // => true
 * ```
 */
export function groupShuffleSplit(nSplits = 5, options: SizeOptions = {}): Splitter {
  integerOption(nSplits, 'nSplits', 'groupShuffleSplit', 1)
  return {
    name: 'group-shuffle-split',
    randomised: true,
    partition: false,
    nSplits: () => nSplits,
    split: (input, key) => {
      const { codes, counts } = encode(need(input.groups, 'groups', 'groupShuffleSplit'))
      const [nTrain, nTest] = splitSizes(counts.length, options, 0.2)
      return Array.from({ length: nSplits }, (_, s) => {
        const perm = shuffled(foldIn(key, s), counts.length)
        const testGroups = new Set(perm.slice(0, nTest))
        const trainGroups = new Set(perm.slice(nTest, nTest + nTrain))
        const idx = Array.from(codes.keys())
        return {
          test: sorted(idx.filter((i) => testGroups.has(codes[i] ?? -1))),
          train: sorted(idx.filter((i) => trainGroups.has(codes[i] ?? -1))),
        }
      })
    },
  }
}

/** Options of {@link holdout}. */
export interface HoldoutOptions extends SizeOptions {
  /** Shuffle before splitting (default true, as sklearn's train_test_split). */
  readonly shuffle?: boolean
  /** Stratify by these labels (implies shuffling). */
  readonly stratify?: ArrayLike<number>
}

/**
 * One train/test split (sklearn.model_selection.train_test_split on indices): without shuffling, the first `nTrain`
 * samples train and the next `nTest` test; with shuffling, a random (optionally stratified) split.
 *
 * @param options - Sizes (default test 0.25), shuffling and stratification labels.
 * @returns The splitter (one fold).
 * @example
 * ```ts
 * import { holdout, splitIndices } from 'whitebox-engine/learn/model-selection'
 * const [f] = splitIndices(holdout({ testSize: 0.3, shuffle: false }), { n: 10 })
 * Array.from(f.test) // => [7, 8, 9]
 * ```
 */
export function holdout(options: HoldoutOptions = {}): Splitter {
  const shuffle = options.shuffle ?? true
  return {
    name: 'holdout',
    randomised: shuffle,
    partition: false,
    nSplits: () => 1,
    split: (input, key) => {
      const [nTrain, nTest] = splitSizes(input.n, options)
      if (options.stratify !== undefined) {
        return stratifiedShuffleSplit(1, { testSize: nTest, trainSize: nTrain }).split(
          { ...input, y: options.stratify },
          key,
        )
      }
      if (!shuffle) {
        return [
          {
            train: Int32Array.from({ length: nTrain }, (_, i) => i),
            test: Int32Array.from({ length: nTest }, (_, i) => nTrain + i),
          },
        ]
      }
      return shuffleSplit(1, { testSize: nTest, trainSize: nTrain }).split(input, key)
    },
  }
}

/** Options of {@link trainValidationTestSplit}: fractions or counts of the validation and test sets. */
export interface ThreeWayOptions {
  readonly validationSize?: number
  readonly testSize?: number
  readonly shuffle?: boolean
}

/**
 * One train/validation/test split: the test set is held out first, then the validation set from the rest; without
 * shuffling the three are contiguous in that order (train, validation, test).
 *
 * @param options - Validation size (default 0.2), test size (default 0.2) and shuffling (default true).
 * @returns The splitter (one fold with `validation`).
 * @example
 * ```ts
 * import { splitIndices, trainValidationTestSplit } from 'whitebox-engine/learn/model-selection'
 * const [f] = splitIndices(trainValidationTestSplit({ shuffle: false }), { n: 10 })
 * [f.train.length, f.validation.length, f.test.length] // => [6, 2, 2]
 * ```
 */
export function trainValidationTestSplit(options: ThreeWayOptions = {}): Splitter {
  const shuffle = options.shuffle ?? true
  return {
    name: 'train-validation-test',
    randomised: shuffle,
    partition: false,
    nSplits: () => 1,
    split: (input, key) => {
      const [, nTest] = splitSizes(input.n, { testSize: options.testSize ?? 0.2 })
      const v = options.validationSize ?? 0.2
      const nVal = v > 0 && v < 1 ? Math.ceil(v * input.n) : v
      const nTrain = input.n - nTest - nVal
      if (!(nTrain >= 1 && nVal >= 1)) {
        throw new WhiteboxError('value', `trainValidationTestSplit: sizes leave ${nTrain} training samples`)
      }
      const order = shuffle ? shuffled(key, input.n) : Array.from({ length: input.n }, (_, i) => i)
      return [
        {
          train: sorted(order.slice(0, nTrain)),
          validation: sorted(order.slice(nTrain, nTrain + nVal)),
          test: sorted(order.slice(nTrain + nVal)),
        },
      ]
    },
  }
}

/**
 * Check that folds never test a training sample (no leakage) and stay in range.
 *
 * @param folds - The folds.
 * @param n - The number of samples.
 * @returns `true` when every fold's sets are disjoint and in [0, n).
 * @example
 * ```ts
 * import { foldsAreDisjoint, kFold, splitIndices } from 'whitebox-engine/learn/model-selection'
 * foldsAreDisjoint(splitIndices(kFold(3), { n: 9 }), 9) // => true
 * ```
 */
export function foldsAreDisjoint(folds: readonly Fold[], n: number): boolean {
  return folds.every((f) => {
    const seen = new Uint8Array(n)
    for (const part of [f.train, f.test, f.validation ?? new Int32Array(0)]) {
      for (const i of part) {
        if (i < 0 || i >= n || seen[i] === 1) return false
        seen[i] = 1
      }
    }
    return true
  })
}
