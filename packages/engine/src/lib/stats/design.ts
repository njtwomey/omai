/**
 * Experimental design, randomization schemes, and Analysis of Variance: simple, permuted-block, and stratified
 * randomization; Pocock-Simon covariate-adaptive minimization as a stepped algorithm (`minimizationAlgorithm`,
 * `minimizationAllocation`); Mahalanobis and Euclidean matched-pairs design (`matchedPairsDesign`); full and
 * fractional $2^{k-p}$ factorial designs with defining relations and alias chains (`fullFactorialDesign`,
 * `fractionalFactorialDesign`, `aliasStructure`); Latin squares (`latinSquareDesign`); and one-way and two-way ANOVA
 * with Type I, II, and III sums of squares (`oneWayAnova`, `twoWayAnova`).
 */

import { type Json, MdmlError, meanOf, type TestResult } from '../base/index.ts'
import { categorical, foldIn, type Key, key as makeKey, permutation, split, uniform } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Observation,
  type RunStatus,
  runAlgorithm,
  type StepContext,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { covarianceOfColumns, invSpdOrPinv, ols, orthonormalContrasts } from './_linear.ts'
import { fSf, groupsOf, mat, rowsOf, sample, vec } from './_util.ts'
import { testResult } from './tests/result.ts'

/**
 * Options for {@link simpleRandomisation}.
 */
export interface SimpleRandomisationOptions {
  /** Number of treatment arms $K \ge 2$ (default `2`). */
  readonly nArms?: number
  /** Allocation probabilities of length $K$ (default equal allocation $1 / K$). */
  readonly probabilities?: readonly number[]
  /** When true, enforces exact integer quotas $\lfloor n p_k \rfloor$ via a random permutation instead of independent draws. */
  readonly exact?: boolean
}

/**
 * Assign $n$ experimental units across $K$ arms by completely randomized (simple) allocation.
 *
 * By default (`exact: false`), each unit is assigned independently according to `probabilities` (or $1/K$). When
 * `exact: true`, arm quotas are fixed to match `probabilities` as closely as possible and shuffled uniformly with
 * Fisher-Yates.
 *
 * @param k - PRNG key.
 * @param n - Number of units $n \ge 1$ to assign.
 * @param options - `nArms` (default `2`), `probabilities`, and `exact` (default `false`).
 * @returns An `int32` tensor of shape `[n]` with arm indices in `0 .. K - 1`.
 * @throws MdmlError with code `domain` if `n < 1`, `nArms < 2`, or probabilities are invalid.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { simpleRandomisation } from 'mdml-engine/stats'
 * const arms = simpleRandomisation(key(0), 6, { exact: true })
 * arms.data.filter((a) => a === 1).length // => 3
 * ```
 */
export function simpleRandomisation(k: Key, n: number, options: SimpleRandomisationOptions = {}): Tensor<'int32'> {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `simpleRandomisation: n must be a positive integer, got ${n}`)
  }
  const nArms = options.nArms ?? (options.probabilities !== undefined ? options.probabilities.length : 2)
  if (!Number.isInteger(nArms) || nArms < 2) {
    throw new MdmlError('domain', `simpleRandomisation: nArms must be an integer >= 2, got ${nArms}`)
  }
  const probs = options.probabilities ?? new Array<number>(nArms).fill(1 / nArms)
  if (probs.length !== nArms) {
    throw new MdmlError('shape', `simpleRandomisation: probabilities length ${probs.length} must equal nArms=${nArms}`)
  }
  let pSum = 0
  for (let a = 0; a < nArms; a++) {
    const p = probs[a] ?? 0
    if (!(p > 0) || !Number.isFinite(p)) {
      throw new MdmlError('domain', `simpleRandomisation: probabilities[${a}] must be > 0, got ${p}`)
    }
    pSum += p
  }

  if (options.exact ?? false) {
    const pool = new Int32Array(n)
    let pos = 0
    const remainders: { readonly arm: number; readonly rem: number }[] = []
    for (let a = 0; a < nArms; a++) {
      const exactCount = (n * (probs[a] ?? 0)) / pSum
      const count = Math.floor(exactCount)
      remainders.push({ arm: a, rem: exactCount - count })
      for (let c = 0; c < count; c++) pool[pos++] = a
    }
    remainders.sort((a, b) => b.rem - a.rem || a.arm - b.arm)
    let rIdx = 0
    while (pos < n) {
      pool[pos++] = remainders[rIdx % nArms]?.arm ?? 0
      rIdx++
    }
    const perm = permutation(k, n).data
    const out = new Int32Array(n)
    for (let i = 0; i < n; i++) out[i] = pool[perm[i] ?? 0] ?? 0
    return fromStorage(out, [n], 'int32')
  }

  return categorical(k, probs, [n])
}

/**
 * Alias for {@link simpleRandomisation}.
 *
 * @param k - PRNG key.
 * @param n - Number of units $n \ge 1$.
 * @param options - Simple randomisation options.
 * @returns An `int32` tensor of shape `[n]` with arm indices in `0 .. K - 1`.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { completelyRandomised } from 'mdml-engine/stats'
 * completelyRandomised(key(1), 8, { exact: true }).shape // => [8]
 * ```
 */
export function completelyRandomised(k: Key, n: number, options: SimpleRandomisationOptions = {}): Tensor<'int32'> {
  return simpleRandomisation(k, n, options)
}

/**
 * Options for {@link permutedBlockRandomisation} and {@link stratifiedRandomisation}.
 */
export interface PermutedBlockOptions {
  /** Number of treatment arms $K \ge 2$ (default `2`). */
  readonly nArms?: number
  /** Integer allocation ratios across the $K$ arms within each block (default `[1, 1, ..., 1]`). */
  readonly ratios?: readonly number[]
  /** Block size (must be a positive multiple of `sum(ratios)`; defaults to `2 * sum(ratios)`). */
  readonly blockSize?: number
}

function buildBlockTemplate(options: PermutedBlockOptions, name: string): Int32Array {
  const nArms = options.nArms ?? (options.ratios !== undefined ? options.ratios.length : 2)
  if (!Number.isInteger(nArms) || nArms < 2) {
    throw new MdmlError('domain', `${name}: nArms must be an integer >= 2, got ${nArms}`)
  }
  const ratios = options.ratios ?? new Array<number>(nArms).fill(1)
  if (ratios.length !== nArms) {
    throw new MdmlError('shape', `${name}: ratios length ${ratios.length} must equal nArms=${nArms}`)
  }
  let ratioSum = 0
  for (let a = 0; a < nArms; a++) {
    const r = ratios[a] ?? 0
    if (!Number.isInteger(r) || r < 1) {
      throw new MdmlError('domain', `${name}: ratios[${a}] must be a positive integer, got ${r}`)
    }
    ratioSum += r
  }
  const blockSize = options.blockSize ?? 2 * ratioSum
  if (!Number.isInteger(blockSize) || blockSize < ratioSum || blockSize % ratioSum !== 0) {
    throw new MdmlError(
      'domain',
      `${name}: blockSize (${blockSize}) must be a positive integer multiple of sum(ratios)=${ratioSum}`,
    )
  }
  const mult = blockSize / ratioSum
  const template = new Int32Array(blockSize)
  let pos = 0
  for (let a = 0; a < nArms; a++) {
    const count = (ratios[a] ?? 1) * mult
    for (let c = 0; c < count; c++) template[pos++] = a
  }
  return template
}

/**
 * Permuted-block randomisation: divides the sequence of $n$ arrivals into contiguous blocks of size `blockSize` and
 * uniformly permutes a balanced allocation template within each block so running treatment counts never drift by more
 * than half a block.
 *
 * @param k - PRNG key.
 * @param n - Number of units $n \ge 1$.
 * @param options - `nArms` (default `2`), integer `ratios` (default all `1`), and `blockSize` (default `2 * sum(ratios)`).
 * @returns An `int32` tensor of shape `[n]` with arm indices in `0 .. K - 1`.
 * @throws MdmlError with code `domain` if `blockSize` is not a positive multiple of `sum(ratios)`.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { permutedBlockRandomisation } from 'mdml-engine/stats'
 * const arms = permutedBlockRandomisation(key(0), 8, { blockSize: 4 })
 * arms.data.slice(0, 4).reduce((a, b) => a + b, 0) // => 2
 * arms.data.slice(4, 8).reduce((a, b) => a + b, 0) // => 2
 * ```
 */
export function permutedBlockRandomisation(k: Key, n: number, options: PermutedBlockOptions = {}): Tensor<'int32'> {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `permutedBlockRandomisation: n must be a positive integer, got ${n}`)
  }
  const template = buildBlockTemplate(options, 'permutedBlockRandomisation')
  const B = template.length
  const nBlocks = Math.ceil(n / B)
  const out = new Int32Array(n)
  let idx = 0
  for (let b = 0; b < nBlocks; b++) {
    const perm = permutation(foldIn(k, b), B).data
    for (let j = 0; j < B && idx < n; j++) {
      out[idx++] = template[perm[j] ?? 0] ?? 0
    }
  }
  return fromStorage(out, [n], 'int32')
}

/**
 * Alias for {@link permutedBlockRandomisation}.
 *
 * @param k - PRNG key.
 * @param n - Number of units $n \ge 1$.
 * @param options - Permuted-block options.
 * @returns An `int32` tensor of shape `[n]` with arm indices in `0 .. K - 1`.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { blockedDesign } from 'mdml-engine/stats'
 * blockedDesign(key(0), 4, { blockSize: 4 }).data.reduce((a, b) => a + b, 0) // => 2
 * ```
 */
export function blockedDesign(k: Key, n: number, options: PermutedBlockOptions = {}): Tensor<'int32'> {
  return permutedBlockRandomisation(k, n, options)
}

/**
 * Stratified permuted-block randomisation: partitions units into discrete strata defined by a 1-D vector or 2-D matrix
 * of categorical covariate values and runs independent permuted-block randomisation within each stratum.
 *
 * @param k - PRNG key.
 * @param strata - 1-D vector (length $n$) or 2-D matrix (shape `[n, p]`) of discrete stratum identifiers.
 * @param options - `nArms` (default `2`), integer `ratios`, and `blockSize`.
 * @returns An `int32` tensor of shape `[n]` with arm indices in `0 .. K - 1`.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { stratifiedRandomisation } from 'mdml-engine/stats'
 * const strata = [0, 0, 0, 0, 1, 1, 1, 1]
 * const arms = stratifiedRandomisation(key(7), strata, { blockSize: 4 })
 * arms.data.slice(0, 4).reduce((a, b) => a + b, 0) // => 2
 * arms.data.slice(4, 8).reduce((a, b) => a + b, 0) // => 2
 * ```
 */
export function stratifiedRandomisation(
  k: Key,
  strata: TensorLike,
  options: PermutedBlockOptions = {},
): Tensor<'int32'> {
  const m = rowsOf('stratifiedRandomisation', strata)
  const n = m.rows
  const p = m.cols
  if (n < 1) {
    throw new MdmlError('shape', 'stratifiedRandomisation: strata must have at least 1 row')
  }
  buildBlockTemplate(options, 'stratifiedRandomisation')

  const groups = new Map<string, number[]>()
  for (let i = 0; i < n; i++) {
    const parts: number[] = []
    for (let j = 0; j < p; j++) parts.push(m.data[i * p + j] ?? 0)
    const sKey = parts.join(',')
    const existing = groups.get(sKey)
    if (existing === undefined) groups.set(sKey, [i])
    else existing.push(i)
  }

  const sortedKeys = Array.from(groups.keys()).sort()
  const out = new Int32Array(n)
  for (let sIdx = 0; sIdx < sortedKeys.length; sIdx++) {
    const sKey = sortedKeys[sIdx] ?? ''
    const indices = groups.get(sKey) ?? []
    const subArms = permutedBlockRandomisation(foldIn(k, sIdx), indices.length, options).data
    for (let j = 0; j < indices.length; j++) {
      out[indices[j] ?? 0] = subArms[j] ?? 0
    }
  }
  return fromStorage(out, [n], 'int32')
}

/**
 * Options for Pocock-Simon covariate-adaptive minimization ({@link minimizationAlgorithm}, {@link minimizationAllocation}).
 */
export interface MinimizationOptions {
  /** Number of treatment arms $K \ge 2$ (default `2`). */
  readonly nArms?: number
  /** Biased-coin probability $p \in [1/K, 1]$ of choosing the imbalance-minimizing arm (default `0.8`; `1` is deterministic). */
  readonly pBiased?: number
  /** Optional relative importance weights across the $p$ covariates (default all `1`). */
  readonly weights?: readonly number[]
}

/**
 * State of the stepped Pocock-Simon covariate-adaptive minimization algorithm. Plain arrays, so it serialises.
 */
export interface MinimizationState {
  /** Number of units allocated so far (`0 .. n`). */
  readonly step: number
  /** Total number of units $n$. */
  readonly n: number
  /** Number of covariates $p$. */
  readonly nCovariates: number
  /** Number of treatment arms $K$. */
  readonly nArms: number
  /** Biased-coin probability $p_{\text{biased}}$. */
  readonly pBiased: number
  /** Row-major integer-coded covariate matrix of shape `[n, p]`. */
  readonly covariates: Float64Array
  /** Covariate importance weights of shape `[p]`. */
  readonly weights: Float64Array
  /** Arm assignments of length $n$ (`-1` for not-yet-assigned units). */
  readonly assignments: Float64Array
  /** Arm allocated to the most recently processed unit (`-1` at `step === 0`). */
  readonly lastAssignedArm: number
  /** Total weighted marginal range imbalance after the latest allocation. */
  readonly imbalance: number
  /** Total marginal imbalance after each processed unit (length `n`; entries past `step` are zero). */
  readonly imbalanceHistory: Float64Array
}

function computeTotalImbalance(
  covariates: Float64Array,
  nCovariates: number,
  weights: Float64Array,
  assignments: Float64Array,
  uptoStep: number,
  nArms: number,
): number {
  let total = 0
  for (let j = 0; j < nCovariates; j++) {
    const wj = weights[j] ?? 1
    const levelCounts = new Map<number, Float64Array>()
    for (let i = 0; i < uptoStep; i++) {
      const arm = assignments[i] ?? -1
      if (arm < 0) continue
      const lvl = covariates[i * nCovariates + j] ?? 0
      let counts = levelCounts.get(lvl)
      if (counts === undefined) {
        counts = new Float64Array(nArms)
        levelCounts.set(lvl, counts)
      }
      counts[arm] = (counts[arm] ?? 0) + 1
    }
    for (const counts of levelCounts.values()) {
      let minC = Number.POSITIVE_INFINITY
      let maxC = Number.NEGATIVE_INFINITY
      for (let a = 0; a < nArms; a++) {
        const c = counts[a] ?? 0
        if (c < minC) minC = c
        if (c > maxC) maxC = c
      }
      total += wj * (maxC - minC)
    }
  }
  return total
}

function candidateImbalanceForUnit(
  covariates: Float64Array,
  nCovariates: number,
  weights: Float64Array,
  assignments: Float64Array,
  unitIndex: number,
  candidateArm: number,
  nArms: number,
): number {
  let score = 0
  for (let j = 0; j < nCovariates; j++) {
    const wj = weights[j] ?? 1
    const targetLevel = covariates[unitIndex * nCovariates + j] ?? 0
    const counts = new Float64Array(nArms)
    for (let i = 0; i < unitIndex; i++) {
      const arm = assignments[i] ?? -1
      if (arm >= 0 && (covariates[i * nCovariates + j] ?? 0) === targetLevel) {
        counts[arm] = (counts[arm] ?? 0) + 1
      }
    }
    counts[candidateArm] = (counts[candidateArm] ?? 0) + 1
    let minC = Number.POSITIVE_INFINITY
    let maxC = Number.NEGATIVE_INFINITY
    for (let a = 0; a < nArms; a++) {
      const c = counts[a] ?? 0
      if (c < minC) minC = c
      if (c > maxC) maxC = c
    }
    score += wj * (maxC - minC)
  }
  return score
}

/**
 * Create a stepped {@link Algorithm} for Pocock-Simon (1975) covariate-adaptive minimization.
 *
 * Each step allocates the next arriving unit $i \in \{0, \dots, n - 1\}$ by evaluating the weighted marginal range
 * imbalance across all covariates that would result from assigning unit $i$ to each arm $a \in \{0, \dots, K - 1\}$,
 * then selecting the imbalance-minimizing arm with probability `pBiased` (ties are broken, and the biased coin is
 * flipped, with the step's key, so a resumed run repeats the same allocations).
 *
 * @param covariates - 1-D vector or 2-D matrix of discrete covariate levels for the $n$ sequential arrivals.
 * @param options - `nArms` (default `2`), `pBiased` (default `0.8`) and covariate `weights`.
 * @returns A stepped `Algorithm<MinimizationState>` advancing one unit arrival per step; run it for `n` steps.
 * @example
 * ```ts
 * import { minimizationAlgorithm } from 'mdml-engine/stats'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const algo = minimizationAlgorithm([[0, 1], [0, 1], [1, 0], [1, 0]], { pBiased: 1 })
 * runAlgorithm(algo, { seed: 0, maxSteps: 4 }).state.imbalance // => 0
 * ```
 */
export function minimizationAlgorithm(
  covariates: TensorLike,
  options: MinimizationOptions = {},
): Algorithm<MinimizationState> {
  const m = rowsOf('minimizationAlgorithm', covariates)
  const n = m.rows
  const p = m.cols
  if (n < 1) {
    throw new MdmlError('shape', 'minimizationAlgorithm: covariates must have at least 1 row')
  }
  const nArms = options.nArms ?? 2
  if (!Number.isInteger(nArms) || nArms < 2) {
    throw new MdmlError('domain', `minimizationAlgorithm: nArms must be an integer >= 2, got ${nArms}`)
  }
  const pBiased = options.pBiased ?? 0.8
  if (!(pBiased >= 1 / nArms && pBiased <= 1) || !Number.isFinite(pBiased)) {
    throw new MdmlError('domain', `minimizationAlgorithm: pBiased must be in [1/nArms, 1], got ${pBiased}`)
  }
  const wArr = new Float64Array(p)
  if (options.weights !== undefined) {
    if (options.weights.length !== p) {
      throw new MdmlError('shape', `minimizationAlgorithm: weights length ${options.weights.length} must equal p=${p}`)
    }
    for (let j = 0; j < p; j++) {
      const wj = options.weights[j] ?? 0
      if (!(wj > 0) || !Number.isFinite(wj)) {
        throw new MdmlError('domain', `minimizationAlgorithm: weights[${j}] must be > 0, got ${wj}`)
      }
      wArr[j] = wj
    }
  } else {
    wArr.fill(1)
  }

  return {
    name: 'minimization',
    config: { n, nCovariates: p, nArms, pBiased, weights: Array.from(wArr) },
    init(): MinimizationState {
      return {
        step: 0,
        n,
        nCovariates: p,
        nArms,
        pBiased,
        covariates: Float64Array.from(m.data),
        weights: wArr,
        assignments: new Float64Array(n).fill(-1),
        lastAssignedArm: -1,
        imbalance: 0,
        imbalanceHistory: new Float64Array(n),
      }
    },
    step(state: MinimizationState, ctx: StepContext): MinimizationState {
      if (state.step >= state.n) return state
      const i = state.step
      const nextAssign = Float64Array.from(state.assignments)

      const armScores = new Float64Array(state.nArms)
      let minScore = Number.POSITIVE_INFINITY
      for (let a = 0; a < state.nArms; a++) {
        const s = candidateImbalanceForUnit(
          state.covariates,
          state.nCovariates,
          state.weights,
          nextAssign,
          i,
          a,
          state.nArms,
        )
        armScores[a] = s
        if (s < minScore) minScore = s
      }

      const bestArms: number[] = []
      const otherArms: number[] = []
      for (let a = 0; a < state.nArms; a++) {
        if (Math.abs((armScores[a] ?? 0) - minScore) <= 1e-12) bestArms.push(a)
        else otherArms.push(a)
      }

      const [uCoin, uPick] = uniform(ctx.key, [2]).data
      let chosenArm: number
      if (otherArms.length === 0 || (uCoin ?? 0) < state.pBiased) {
        const idx = Math.min(bestArms.length - 1, Math.floor((uPick ?? 0) * bestArms.length))
        chosenArm = bestArms[idx] ?? 0
      } else {
        const idx = Math.min(otherArms.length - 1, Math.floor((uPick ?? 0) * otherArms.length))
        chosenArm = otherArms[idx] ?? 0
      }

      nextAssign[i] = chosenArm
      const imb = computeTotalImbalance(
        state.covariates,
        state.nCovariates,
        state.weights,
        nextAssign,
        i + 1,
        state.nArms,
      )
      const nextHistory = Float64Array.from(state.imbalanceHistory)
      nextHistory[i] = imb
      return {
        ...state,
        step: i + 1,
        assignments: nextAssign,
        lastAssignedArm: chosenArm,
        imbalance: imb,
        imbalanceHistory: nextHistory,
      }
    },
    done: (state: MinimizationState): RunStatus | undefined => (state.step >= state.n ? 'completed' : undefined),
    observe: (state: MinimizationState): Observation => ({
      metrics: { imbalance: state.imbalance, assignedArm: state.lastAssignedArm },
    }),
    serialise: (state: MinimizationState): Json => stateToJson(state),
    deserialise: (json: Json): MinimizationState => stateFromJson(json) as MinimizationState,
  }
}

/**
 * Result of {@link minimizationAllocation}.
 */
export interface MinimizationResult {
  /** Final `int32` arm assignments of shape `[n]`. */
  readonly assignments: Tensor<'int32'>
  /** Final weighted marginal range imbalance across all covariates. */
  readonly imbalance: number
  /** Trajectory of total marginal imbalance after each sequential allocation (`[n]`). */
  readonly imbalanceHistory: Tensor
  /** Final stepped algorithm state. */
  readonly state: MinimizationState
}

/**
 * Allocate $n$ units across $K$ arms using Pocock-Simon (1975) covariate-adaptive minimization.
 *
 * @param covariates - 1-D vector or 2-D matrix of discrete covariate levels for the $n$ units.
 * @param options - `nArms` (default `2`), `pBiased` (default `0.8`), covariate `weights` and the PRNG `key` (default `key(0)`).
 * @returns `MinimizationResult` with `assignments`, `imbalance`, `imbalanceHistory`, and final `state`.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { minimizationAllocation } from 'mdml-engine/stats'
 * const cov = [[0, 0], [0, 0], [1, 1], [1, 1]]
 * const res = minimizationAllocation(cov, { key: key(0), pBiased: 1 })
 * res.imbalance // => 0
 * res.assignments.data.reduce((a, b) => a + b, 0) // => 2
 * ```
 */
export function minimizationAllocation(
  covariates: TensorLike,
  options: MinimizationOptions & { readonly key?: Key } = {},
): MinimizationResult {
  const algo = minimizationAlgorithm(covariates, options)
  const state0 = algo.init(options.key ?? makeKey(0))
  const { state } = runAlgorithm(algo, { key: options.key ?? makeKey(0), maxSteps: state0.n })
  return {
    assignments: fromStorage(Int32Array.from(state.assignments), [state.n], 'int32'),
    imbalance: state.imbalance,
    imbalanceHistory: vec(state.imbalanceHistory),
    state,
  }
}

/**
 * Options for {@link matchedPairsDesign}.
 */
export interface MatchedPairsOptions {
  /** Distance metric between unit covariate vectors: `'mahalanobis'` (default) or `'euclidean'` (standardized). */
  readonly distance?: 'mahalanobis' | 'euclidean'
  /** Optional PRNG key to randomize which member of each matched pair receives treatment (`1`) vs control (`0`). */
  readonly key?: Key
}

/**
 * Result of {@link matchedPairsDesign}.
 */
export interface MatchedPairsResult {
  /** Matched pairs `[i, j]` of row indices (`0 <= i < j < n`). */
  readonly pairs: readonly (readonly [number, number])[]
  /** Pairwise covariate distance for each matched pair (shape `[nPairs]`). */
  readonly distances: Tensor
  /** Average pairwise covariate distance across all matched pairs. */
  readonly meanDistance: number
  /** Pair index `0 .. nPairs - 1` for each unit (`-1` if $n$ is odd and one unit is left unmatched). */
  readonly pairIds: Tensor<'int32'>
  /** Treatment assignments (`0` or `1`) of shape `[n]` with exactly one treated and one control per pair (`-1` if unmatched). */
  readonly assignments: Tensor<'int32'>
}

/**
 * Construct a matched-pairs experimental design by pairing units with minimum covariate distance (Mahalanobis or
 * standardized Euclidean), using exact dynamic programming minimum-weight perfect matching for $n \le 16$ and greedy
 * nearest-neighbor matching for larger $n$, then assigning one unit per pair to treatment (`1`) and the other to
 * control (`0`).
 *
 * @param covariates - 1-D vector or 2-D matrix of baseline covariates of shape `[n, p]` ($n \ge 2$).
 * @param options - `distance` (`'mahalanobis'` default or `'euclidean'`) and optional PRNG `key`.
 * @returns Matched pairs, pairwise distances, `meanDistance`, `pairIds`, and binary `assignments`.
 * @throws MdmlError with code `shape` if fewer than 2 units are provided.
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { matchedPairsDesign } from 'mdml-engine/stats'
 * const res = matchedPairsDesign([1.0, 1.1, 10.0, 10.2], { key: key(0) })
 * res.pairs // => [[0, 1], [2, 3]]
 * res.assignments.data[0] !== res.assignments.data[1] // => true
 * ```
 */
export function matchedPairsDesign(covariates: TensorLike, options: MatchedPairsOptions = {}): MatchedPairsResult {
  const m = rowsOf('matchedPairsDesign', covariates)
  const n = m.rows
  const p = m.cols
  if (n < 2) {
    throw new MdmlError('shape', `matchedPairsDesign: need at least 2 units, got ${n}`)
  }

  const metric = options.distance ?? 'mahalanobis'
  const cov = covarianceOfColumns(m.data, n, p, 1)
  let weightMat: Float64Array
  if (metric === 'euclidean') {
    weightMat = new Float64Array(p * p)
    for (let j = 0; j < p; j++) {
      const v = cov[j * p + j] ?? 0
      weightMat[j * p + j] = v > 1e-12 ? 1 / v : 1
    }
  } else {
    weightMat = invSpdOrPinv(cov, p)
  }

  const distMat = new Float64Array(n * n)
  const diff = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let c = 0; c < p; c++) {
        diff[c] = (m.data[i * p + c] ?? 0) - (m.data[j * p + c] ?? 0)
      }
      let quad = 0
      for (let a = 0; a < p; a++) {
        for (let b = 0; b < p; b++) {
          quad += (diff[a] ?? 0) * (weightMat[a * p + b] ?? 0) * (diff[b] ?? 0)
        }
      }
      const d = Math.sqrt(Math.max(0, quad))
      distMat[i * n + j] = d
      distMat[j * n + i] = d
    }
  }

  const pairs: [number, number][] = []
  const nPairs = Math.floor(n / 2)

  if (n <= 16 && n % 2 === 0) {
    // Exact minimum-weight perfect matching via bitmask DP over subsets with even popcount.
    const totalMasks = 1 << n
    const dp = new Float64Array(totalMasks)
    dp.fill(Number.POSITIVE_INFINITY)
    const choiceJ = new Int32Array(totalMasks)
    dp[0] = 0
    for (let mask = 0; mask < totalMasks; mask++) {
      const baseCost = dp[mask] ?? Number.POSITIVE_INFINITY
      if (!Number.isFinite(baseCost)) continue
      // Find lowest unset bit i
      let i = 0
      while (i < n && (mask & (1 << i)) !== 0) i++
      if (i >= n) continue
      for (let j = i + 1; j < n; j++) {
        if ((mask & (1 << j)) === 0) {
          const nextMask = mask | (1 << i) | (1 << j)
          const cand = baseCost + (distMat[i * n + j] ?? 0)
          if (cand < (dp[nextMask] ?? Number.POSITIVE_INFINITY)) {
            dp[nextMask] = cand
            choiceJ[nextMask] = (i << 8) | j
          }
        }
      }
    }
    let curMask = totalMasks - 1
    while (curMask > 0) {
      const packed = choiceJ[curMask] ?? 0
      const i = packed >> 8
      const j = packed & 0xff
      pairs.push([i, j])
      curMask &= ~((1 << i) | (1 << j))
    }
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  } else {
    // Greedy minimum-distance pairing across all candidate edges.
    const candidates: { readonly i: number; readonly j: number; readonly d: number }[] = []
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        candidates.push({ i, j, d: distMat[i * n + j] ?? 0 })
      }
    }
    candidates.sort((a, b) => a.d - b.d || a.i - b.i || a.j - b.j)
    const used = new Uint8Array(n)
    for (let c = 0; c < candidates.length && pairs.length < nPairs; c++) {
      const edge = candidates[c]
      if (edge === undefined) continue
      if (used[edge.i] === 0 && used[edge.j] === 0) {
        used[edge.i] = 1
        used[edge.j] = 1
        pairs.push([edge.i, edge.j])
      }
    }
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  }

  const distances = new Float64Array(nPairs)
  const pairIds = new Int32Array(n)
  pairIds.fill(-1)
  const assignments = new Int32Array(n)
  assignments.fill(-1)

  const coins = options.key !== undefined ? uniform(options.key, [nPairs]).data : undefined
  for (let pIdx = 0; pIdx < nPairs; pIdx++) {
    const [i, j] = pairs[pIdx] ?? [0, 1]
    distances[pIdx] = distMat[i * n + j] ?? 0
    pairIds[i] = pIdx
    pairIds[j] = pIdx
    const firstTreated = coins !== undefined ? (coins[pIdx] ?? 0) < 0.5 : pIdx % 2 === 0
    assignments[i] = firstTreated ? 1 : 0
    assignments[j] = firstTreated ? 0 : 1
  }

  return {
    pairs,
    distances: vec(distances),
    meanDistance: nPairs > 0 ? meanOf(distances) : 0,
    pairIds: fromStorage(pairIds, [n], 'int32'),
    assignments: fromStorage(assignments, [n], 'int32'),
  }
}

/**
 * Alias for {@link matchedPairsDesign}.
 *
 * @param covariates - 1-D vector or 2-D matrix of baseline covariates.
 * @param options - Matched-pairs options.
 * @returns Matched-pairs design result.
 * @example
 * ```ts
 * import { matchedPairs } from 'mdml-engine/stats'
 * matchedPairs([0, 0.1, 5, 5.1]).pairs // => [[0, 1], [2, 3]]
 * ```
 */
export function matchedPairs(covariates: TensorLike, options: MatchedPairsOptions = {}): MatchedPairsResult {
  return matchedPairsDesign(covariates, options)
}

/**
 * Options for {@link fullFactorialDesign}.
 */
export interface FullFactorialOptions {
  /**
   * Level coding: `'coded'` (default for two-level designs, mapping `0 -> -1` and `1 -> +1`, or evenly spaced in
   * `[-1, +1]`) or `'index'` (`0 .. L_j - 1`).
   */
  readonly coding?: 'coded' | 'index'
}

/**
 * Construct a full factorial design matrix:
 * - Given an integer $k \ge 1$, returns the $2^k \times k$ two-level full factorial matrix with entries in `{-1, +1}`
 *   (or `{0, 1}` when `coding: 'index'`).
 * - Given an array of level counts `[L_1, ..., L_k]`, returns the $(\prod_{j=1}^k L_j) \times k$ grid across all
 *   factor combinations.
 *
 * @param levels - Number of two-level factors $k \ge 1$, or array of level counts `[L_1, ..., L_k]` ($L_j \ge 2$).
 * @param options - `coding` (`'coded'` default or `'index'`).
 * @returns Design matrix `Tensor` of shape `[nRuns, k]`.
 * @throws MdmlError with code `domain` for invalid factor counts or level counts.
 * @example
 * ```ts
 * import { fullFactorialDesign } from 'mdml-engine/stats'
 * const X = fullFactorialDesign(3)
 * X.shape // => [8, 3]
 * Array.from(X.data.slice(0, 3)) // => [-1, -1, -1]
 * ```
 */
export function fullFactorialDesign(levels: number | readonly number[], options: FullFactorialOptions = {}): Tensor {
  const levelArr: number[] =
    typeof levels === 'number'
      ? (() => {
          if (!Number.isInteger(levels) || levels < 1 || levels > 20) {
            throw new MdmlError('domain', `fullFactorialDesign: k must be an integer in 1..20, got ${levels}`)
          }
          return new Array<number>(levels).fill(2)
        })()
      : [...levels]

  const k = levelArr.length
  if (k < 1) {
    throw new MdmlError('domain', 'fullFactorialDesign: levels array must be non-empty')
  }
  let nRuns = 1
  for (let j = 0; j < k; j++) {
    const lj = levelArr[j] ?? 0
    if (!Number.isInteger(lj) || lj < 2) {
      throw new MdmlError('domain', `fullFactorialDesign: levels[${j}] must be an integer >= 2, got ${lj}`)
    }
    nRuns *= lj
  }

  const coding = options.coding ?? 'coded'
  const data = new Float64Array(nRuns * k)
  for (let r = 0; r < nRuns; r++) {
    let rem = r
    for (let j = k - 1; j >= 0; j--) {
      const lj = levelArr[j] ?? 2
      const idx = rem % lj
      rem = Math.floor(rem / lj)
      data[r * k + j] = coding === 'index' ? idx : (2 * idx) / (lj - 1) - 1
    }
  }
  return mat(data, nRuns, k)
}

/**
 * Alias for {@link fullFactorialDesign}.
 *
 * @param levels - Number of two-level factors $k$ or array of factor level counts.
 * @param options - Coding options.
 * @returns Design matrix `Tensor` of shape `[nRuns, k]`.
 * @example
 * ```ts
 * import { fullFactorial } from 'mdml-engine/stats'
 * fullFactorial(2).shape // => [4, 2]
 * ```
 */
export function fullFactorial(levels: number | readonly number[], options: FullFactorialOptions = {}): Tensor {
  return fullFactorialDesign(levels, options)
}

function factorLabel(index: number): string {
  return String.fromCharCode(65 + index)
}

function maskToWord(mask: number, k: number): string {
  if (mask === 0) return 'I'
  let s = ''
  for (let j = 0; j < k; j++) {
    if ((mask & (1 << j)) !== 0) s += factorLabel(j)
  }
  return s
}

function popcount(mask: number): number {
  let c = 0
  let m = mask
  while (m > 0) {
    m &= m - 1
    c++
  }
  return c
}

function parseGeneratorSpecs(
  k: number,
  generators: readonly (string | readonly number[])[],
): { readonly nBase: number; readonly genMasks: readonly number[]; readonly definingMasks: readonly number[] } {
  const p = generators.length
  if (p < 1 || p >= k) {
    throw new MdmlError('domain', `fractionalFactorialDesign: number of generators p=${p} must satisfy 1 <= p < k=${k}`)
  }
  const nBase = k - p
  const genMasks: number[] = []
  const definingWords: number[] = []

  for (let gIdx = 0; gIdx < p; gIdx++) {
    const spec = generators[gIdx]
    const targetFactor = nBase + gIdx
    let rhsMask = 0
    if (typeof spec === 'string') {
      const cleaned = spec.replace(/\s+/g, '').toUpperCase()
      const parts = cleaned.split('=')
      const rhs = parts.length === 2 ? (parts[1] ?? '') : (parts[0] ?? '')
      if (parts.length === 2) {
        const lhs = parts[0] ?? ''
        const lhsIdx = lhs.charCodeAt(0) - 65
        if (lhs.length !== 1 || lhsIdx !== targetFactor) {
          throw new MdmlError(
            'domain',
            `fractionalFactorialDesign: generator '${spec}' must define factor ${factorLabel(targetFactor)}`,
          )
        }
      }
      for (let c = 0; c < rhs.length; c++) {
        const fIdx = rhs.charCodeAt(c) - 65
        if (fIdx < 0 || fIdx >= nBase) {
          throw new MdmlError(
            'domain',
            `fractionalFactorialDesign: generator '${spec}' RHS must reference base factors A..${factorLabel(nBase - 1)}`,
          )
        }
        rhsMask ^= 1 << fIdx
      }
    } else if (Array.isArray(spec)) {
      for (let c = 0; c < spec.length; c++) {
        const fIdx = spec[c] ?? -1
        if (!Number.isInteger(fIdx) || fIdx < 0 || fIdx >= nBase) {
          throw new MdmlError('domain', `fractionalFactorialDesign: generator index ${fIdx} must be in 0..${nBase - 1}`)
        }
        rhsMask ^= 1 << fIdx
      }
    }
    if (rhsMask === 0) {
      throw new MdmlError('domain', `fractionalFactorialDesign: generator ${gIdx} cannot be empty`)
    }
    genMasks.push(rhsMask)
    definingWords.push(rhsMask | (1 << targetFactor))
  }

  return { nBase, genMasks, definingMasks: definingWords }
}

/**
 * Defining relation and alias structure of a $2^{k-p}$ fractional factorial design.
 */
export interface AliasStructureResult {
  /** Total number of factors $k$. */
  readonly k: number
  /** Number of fraction generators $p$. */
  readonly p: number
  /** Number of runs $2^{k-p}$. */
  readonly nRuns: number
  /** Design resolution $R$ (shortest word length in the non-trivial defining relation, e.g. `3`, `4`, `5`). */
  readonly resolution: number
  /** Defining relation words `['I', ...]` in the $2^p$-element defining subgroup. */
  readonly definingRelation: readonly string[]
  /** Map from each main effect and two-factor interaction (e.g. `'A'`, `'AB'`) to its aliased effects. */
  readonly aliases: Readonly<Record<string, readonly string[]>>
}

/**
 * Compute the complete defining relation subgroup, resolution, and alias chains of a $2^{k-p}$ fractional factorial
 * design specified by $p$ generators.
 *
 * @param k - Total number of two-level factors ($3 \le k \le 15$).
 * @param generators - Array of $p$ generator specifications (e.g. `['D=ABC']` or `[[0, 1, 2]]`).
 * @returns `AliasStructureResult` with `resolution`, `definingRelation`, and `aliases` for main effects and 2FIs.
 * @throws MdmlError with code `domain` if generators are invalid.
 * @example
 * ```ts
 * import { aliasStructure } from 'mdml-engine/stats'
 * const info = aliasStructure(4, ['D=ABC'])
 * info.resolution // => 4
 * info.definingRelation // => ['I', 'ABCD']
 * info.aliases['A'] // => ['BCD']
 * info.aliases['AB'] // => ['CD']
 * ```
 */
export function aliasStructure(k: number, generators: readonly (string | readonly number[])[]): AliasStructureResult {
  if (!Number.isInteger(k) || k < 3 || k > 15) {
    throw new MdmlError('domain', `aliasStructure: k must be an integer in 3..15, got ${k}`)
  }
  const { nBase, definingMasks } = parseGeneratorSpecs(k, generators)
  const p = generators.length
  const subgroupSize = 1 << p
  const subgroupMasks: number[] = []
  let minWordLen = Number.POSITIVE_INFINITY

  for (let s = 0; s < subgroupSize; s++) {
    let mask = 0
    for (let g = 0; g < p; g++) {
      if ((s & (1 << g)) !== 0) {
        mask ^= definingMasks[g] ?? 0
      }
    }
    subgroupMasks.push(mask)
    if (mask !== 0) {
      const len = popcount(mask)
      if (len < minWordLen) minWordLen = len
    }
  }

  const definingRelation = subgroupMasks.map((m) => maskToWord(m, k))
  const aliases: Record<string, readonly string[]> = {}

  // Populate alias chains for all main effects (order 1) and two-factor interactions (order 2).
  const targetMasks: number[] = []
  for (let i = 0; i < k; i++) targetMasks.push(1 << i)
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      targetMasks.push((1 << i) | (1 << j))
    }
  }

  for (const effMask of targetMasks) {
    const label = maskToWord(effMask, k)
    const chain: number[] = []
    for (let s = 1; s < subgroupSize; s++) {
      const aliased = effMask ^ (subgroupMasks[s] ?? 0)
      if (aliased !== 0 && aliased !== effMask) chain.push(aliased)
    }
    chain.sort((a, b) => popcount(a) - popcount(b) || a - b)
    aliases[label] = chain.map((m) => maskToWord(m, k))
  }

  return {
    k,
    p,
    nRuns: 1 << nBase,
    resolution: minWordLen,
    definingRelation,
    aliases,
  }
}

/**
 * Result of {@link fractionalFactorialDesign}.
 */
export interface FractionalFactorialResult extends AliasStructureResult {
  /** Design matrix of shape `[2^(k-p), k]` with entries in `{-1, +1}`. */
  readonly design: Tensor
}

/**
 * Construct a $2^{k-p}$ two-level fractional factorial design matrix in `{-1, +1}` together with its defining relation,
 * resolution, and alias chains.
 *
 * @param k - Total number of factors ($3 \le k \le 15$).
 * @param generators - Array of $p$ generator specifications (e.g. `['D=ABC']` for a $2^{4-1}_{\text{IV}}$ design or `['C=AB']` for a $2^{3-1}_{\text{III}}$ design).
 * @returns `FractionalFactorialResult` containing the `[2^(k-p), k]` `design` matrix, `resolution`, `definingRelation`, and `aliases`.
 * @example
 * ```ts
 * import { fractionalFactorialDesign } from 'mdml-engine/stats'
 * const ff = fractionalFactorialDesign(4, ['D=ABC'])
 * ff.design.shape // => [8, 4]
 * ff.resolution // => 4
 * ```
 */
export function fractionalFactorialDesign(
  k: number,
  generators: readonly (string | readonly number[])[],
): FractionalFactorialResult {
  const info = aliasStructure(k, generators)
  const { nBase, genMasks } = parseGeneratorSpecs(k, generators)
  const p = generators.length
  const baseMat = fullFactorialDesign(nBase)
  const nRuns = 1 << nBase
  const out = new Float64Array(nRuns * k)

  for (let r = 0; r < nRuns; r++) {
    for (let j = 0; j < nBase; j++) {
      out[r * k + j] = baseMat.data[r * nBase + j] ?? 1
    }
    for (let g = 0; g < p; g++) {
      const mask = genMasks[g] ?? 0
      let prod = 1
      for (let j = 0; j < nBase; j++) {
        if ((mask & (1 << j)) !== 0) {
          prod *= out[r * k + j] ?? 1
        }
      }
      out[r * k + nBase + g] = prod
    }
  }

  return {
    ...info,
    design: mat(out, nRuns, k),
  }
}

/**
 * Alias for {@link fractionalFactorialDesign}.
 *
 * @param k - Total number of factors.
 * @param generators - Array of $p$ generator specifications.
 * @returns Fractional factorial design matrix and alias structure.
 * @example
 * ```ts
 * import { fractionalFactorial } from 'mdml-engine/stats'
 * fractionalFactorial(3, ['C=AB']).resolution // => 3
 * ```
 */
export function fractionalFactorial(
  k: number,
  generators: readonly (string | readonly number[])[],
): FractionalFactorialResult {
  return fractionalFactorialDesign(k, generators)
}

/**
 * Construct an $n \times n$ Latin square design where each treatment symbol `0 .. n - 1` appears exactly once in every
 * row and every column. When `options.key` is supplied, rows, columns, and treatment symbols are independently permuted
 * to draw a random Latin square from the cyclic isotopic class.
 *
 * @param n - Side length / number of treatments ($n \ge 1$).
 * @param options - Optional PRNG `key` for randomizing rows, columns, and symbols.
 * @returns An `int32` tensor of shape `[n, n]` with entries in `0 .. n - 1`.
 * @throws MdmlError with code `domain` if `n < 1`.
 * @example
 * ```ts
 * import { latinSquareDesign } from 'mdml-engine/stats'
 * const sq = latinSquareDesign(3)
 * sq.shape // => [3, 3]
 * Array.from(sq.data) // => [0, 1, 2, 1, 2, 0, 2, 0, 1]
 * ```
 */
export function latinSquareDesign(n: number, options: { readonly key?: Key } = {}): Tensor<'int32'> {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `latinSquareDesign: n must be a positive integer, got ${n}`)
  }
  let rowPerm: Int32Array = new Int32Array(n)
  let colPerm: Int32Array = new Int32Array(n)
  let symPerm: Int32Array = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    rowPerm[i] = i
    colPerm[i] = i
    symPerm[i] = i
  }
  if (options.key !== undefined) {
    const [kRow, kCol, kSym] = split(options.key, 3)
    if (kRow !== undefined) rowPerm = permutation(kRow, n).data
    if (kCol !== undefined) colPerm = permutation(kCol, n).data
    if (kSym !== undefined) symPerm = permutation(kSym, n).data
  }
  const out = new Int32Array(n * n)
  for (let r = 0; r < n; r++) {
    const pr = rowPerm[r] ?? r
    for (let c = 0; c < n; c++) {
      const pc = colPerm[c] ?? c
      out[r * n + c] = symPerm[(pr + pc) % n] ?? 0
    }
  }
  return fromStorage(out, [n, n], 'int32')
}

/**
 * Result of {@link oneWayAnova}.
 */
export interface OneWayAnovaResult extends TestResult {
  /** Between-groups sum of squares $SS_B$. */
  readonly ssBetween: number
  /** Within-groups (residual) sum of squares $SS_W$. */
  readonly ssWithin: number
  /** Total sum of squares $SS_T = SS_B + SS_W$. */
  readonly ssTotal: number
  /** Between-groups degrees of freedom $k - 1$. */
  readonly dfBetween: number
  /** Within-groups degrees of freedom $N - k$. */
  readonly dfWithin: number
  /** Between-groups mean square $MS_B = SS_B / (k - 1)$. */
  readonly msBetween: number
  /** Within-groups mean square $MS_W = SS_W / (N - k)$. */
  readonly msWithin: number
  /** Proportion of variance explained $\eta^2 = SS_B / SS_T$. */
  readonly etaSquared: number
  /** Unbiased effect-size estimate $\omega^2 = (SS_B - (k - 1) MS_W) / (SS_T + MS_W)$. */
  readonly omegaSquared: number
  /** Sample mean of each group (shape `[k]`). */
  readonly groupMeans: Tensor
}

/**
 * One-way Analysis of Variance (ANOVA) $F$-test for equality of means across $k \ge 2$ groups, returning the full
 * ANOVA table ($SS_B$, $SS_W$, $MS_B$, $MS_W$), degrees of freedom, and $\eta^2$ and $\omega^2$ effect sizes.
 *
 * @param groups - Array of $k \ge 2$ 1-D group samples.
 * @param options - Optional confidence `level`.
 * @returns `OneWayAnovaResult` (`TestResult` plus ANOVA table and effect sizes).
 * @throws MdmlError with code `shape` if fewer than 2 groups or total degrees of freedom $N - k < 1$.
 * @example
 * ```ts
 * import { oneWayAnova } from 'mdml-engine/stats'
 * const res = oneWayAnova([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 * Number(res.statistic.toFixed(4)) // => 27
 * Number(res.etaSquared.toFixed(4)) // => 0.9
 * ```
 */
export function oneWayAnova(groups: readonly TensorLike[]): OneWayAnovaResult {
  const gs = groupsOf('oneWayAnova', groups, 1, 2)
  const k = gs.length
  let N = 0
  let grandSum = 0
  const means = new Float64Array(k)
  for (let g = 0; g < k; g++) {
    const arr = gs[g]
    if (arr === undefined) continue
    N += arr.length
    let s = 0
    for (let i = 0; i < arr.length; i++) s += arr[i] ?? 0
    grandSum += s
    means[g] = s / arr.length
  }
  const dfBetween = k - 1
  const dfWithin = N - k
  if (dfWithin < 1) {
    throw new MdmlError('shape', `oneWayAnova: need total sample size N > k (got N=${N}, k=${k})`)
  }
  const grandMean = grandSum / N
  let ssBetween = 0
  let ssWithin = 0
  for (let g = 0; g < k; g++) {
    const arr = gs[g]
    if (arr === undefined) continue
    const gm = means[g] ?? 0
    const db = gm - grandMean
    ssBetween += arr.length * db * db
    for (let i = 0; i < arr.length; i++) {
      const dw = (arr[i] ?? 0) - gm
      ssWithin += dw * dw
    }
  }
  const ssTotal = ssBetween + ssWithin
  const msBetween = ssBetween / dfBetween
  const msWithin = ssWithin / dfWithin
  const fStat = msWithin > 0 ? msBetween / msWithin : ssBetween > 0 ? Number.POSITIVE_INFINITY : 0
  const pVal = Number.isFinite(fStat) ? fSf(dfBetween, dfWithin, fStat) : 0
  const etaSquared = ssTotal > 0 ? ssBetween / ssTotal : 0
  const omegaSquared =
    ssTotal + msWithin > 0 ? Math.max(0, (ssBetween - dfBetween * msWithin) / (ssTotal + msWithin)) : 0

  const base = testResult({
    test: 'One-way ANOVA',
    statistic: fStat,
    pValue: pVal,
    alternative: 'greater',
    df: [dfBetween, dfWithin],
  })
  return {
    ...base,
    ssBetween,
    ssWithin,
    ssTotal,
    dfBetween,
    dfWithin,
    msBetween,
    msWithin,
    etaSquared,
    omegaSquared,
    groupMeans: vec(means),
  }
}

/**
 * Single term row in a two-way ANOVA table (`factorA`, `factorB`, or `interaction`).
 */
export interface AnovaTermResult extends TestResult {
  /** Term sum of squares ($SS$). */
  readonly ss: number
  /** Numerator degrees of freedom. */
  readonly dfTerm: number
  /** Mean square $MS = SS / \text{dfTerm}$. */
  readonly ms: number
  /** Partial eta-squared $\eta_p^2 = SS / (SS + SS_{\text{error}})$. */
  readonly partialEtaSquared: number
}

/**
 * Result of {@link twoWayAnova}.
 */
export interface TwoWayAnovaResult {
  /** Sum-of-squares convention used (`'I'`, `'II'`, or `'III'`). */
  readonly ssType: 'I' | 'II' | 'III'
  /** Main effect of Factor A. */
  readonly factorA: AnovaTermResult
  /** Main effect of Factor B. */
  readonly factorB: AnovaTermResult
  /** Interaction effect $A \times B$ (present when `interaction: true`, the default). */
  readonly interaction?: AnovaTermResult
  /** Residual (within-cell error) sum of squares, degrees of freedom, and mean square. */
  readonly residual: { readonly ss: number; readonly df: number; readonly ms: number }
  /** Total sum of squares and degrees of freedom $N - 1$. */
  readonly total: { readonly ss: number; readonly df: number }
  /** Multiple $R^2 = 1 - SS_{\text{error}} / SS_{\text{total}}$. */
  readonly rSquared: number
  /** Adjusted $R^2$. */
  readonly adjustedRSquared: number
}

/**
 * Options for {@link twoWayAnova}.
 */
export interface TwoWayAnovaOptions {
  /** Sum-of-squares type: `1`/`'I'` (sequential), `2`/`'II'` (hierarchical), or `3`/`'III'` (marginal orthogonal contrasts, default `'III'`). */
  readonly ssType?: 1 | 2 | 3 | 'I' | 'II' | 'III'
  /** Whether to include the $A \times B$ interaction term (default `true`). */
  readonly interaction?: boolean
}

function buildContrastColumns(
  levels: ArrayLike<number>,
  uniqueLevels: readonly number[],
): { readonly cols: number; readonly data: Float64Array } {
  const n = levels.length
  const k = uniqueLevels.length
  const cMat = orthonormalContrasts(k) // k x (k - 1)
  const q = k - 1
  const out = new Float64Array(n * q)
  const indexMap = new Map<number, number>()
  for (let j = 0; j < k; j++) indexMap.set(uniqueLevels[j] ?? 0, j)
  for (let i = 0; i < n; i++) {
    const lvlIdx = indexMap.get(levels[i] ?? 0) ?? 0
    for (let c = 0; c < q; c++) {
      out[i * q + c] = cMat[lvlIdx * q + c] ?? 0
    }
  }
  return { cols: q, data: out }
}

function combineDesignBlocks(
  n: number,
  blocks: readonly { readonly cols: number; readonly data: Float64Array }[],
): {
  readonly cols: number
  readonly data: Float64Array
} {
  let totalCols = 1 // intercept
  for (const b of blocks) totalCols += b.cols
  const X = new Float64Array(n * totalCols)
  for (let i = 0; i < n; i++) {
    X[i * totalCols] = 1
    let colOff = 1
    for (const b of blocks) {
      for (let c = 0; c < b.cols; c++) {
        X[i * totalCols + colOff + c] = b.data[i * b.cols + c] ?? 0
      }
      colOff += b.cols
    }
  }
  return { cols: totalCols, data: X }
}

/**
 * Two-way Analysis of Variance (ANOVA) for a continuous response $y$ crossed by categorical factors $A$ ($a \ge 2$
 * levels) and $B$ ($b \ge 2$ levels), supporting balanced and unbalanced designs under Type I (sequential), Type II
 * (hierarchical marginal), and Type III (Helmert/orthonormal contrast marginal, default) sums of squares.
 *
 * @param y - 1-D response vector of length $n$.
 * @param factorA - 1-D vector of numeric/integer level labels for Factor A (length $n$).
 * @param factorB - 1-D vector of numeric/integer level labels for Factor B (length $n$).
 * @param options - `ssType` (`'I'`, `'II'`, or `'III'`, default `'III'`), and `interaction` (default `true`).
 * @returns `TwoWayAnovaResult` with `factorA`, `factorB`, optional `interaction`, `residual`, `total`, and $R^2$.
 * @throws MdmlError with code `shape` or `domain` if a factor has fewer than 2 levels or residual degrees of freedom $< 1$.
 * @example
 * ```ts
 * import { twoWayAnova } from 'mdml-engine/stats'
 * const y = [2, 3, 4, 5, 6, 7, 8, 9]
 * const a = [0, 0, 0, 0, 1, 1, 1, 1]
 * const b = [0, 0, 1, 1, 0, 0, 1, 1]
 * const res = twoWayAnova(y, a, b)
 * Number(res.factorA.ss.toFixed(4)) // => 32
 * Number(res.factorB.ss.toFixed(4)) // => 8
 * Number((res.interaction?.ss ?? 0).toFixed(4)) // => 0
 * ```
 */
export function twoWayAnova(
  y: TensorLike,
  factorA: TensorLike,
  factorB: TensorLike,
  options: TwoWayAnovaOptions = {},
): TwoWayAnovaResult {
  const ys = sample('twoWayAnova.y', y, 4)
  const n = ys.length
  const fa = sample('twoWayAnova.factorA', factorA, n)
  const fb = sample('twoWayAnova.factorB', factorB, n)
  if (fa.length !== n || fb.length !== n) {
    throw new MdmlError('shape', `twoWayAnova: factorA (${fa.length}) and factorB (${fb.length}) must match y (${n})`)
  }

  const levelsA = Array.from(new Set(fa)).sort((u, v) => u - v)
  const levelsB = Array.from(new Set(fb)).sort((u, v) => u - v)
  const a = levelsA.length
  const b = levelsB.length
  if (a < 2 || b < 2) {
    throw new MdmlError('domain', `twoWayAnova: both factors must have at least 2 levels (got a=${a}, b=${b})`)
  }

  const includeInteraction = options.interaction ?? true
  const rawType = options.ssType ?? 'III'
  const ssType: 'I' | 'II' | 'III' =
    rawType === 1 || rawType === 'I' ? 'I' : rawType === 2 || rawType === 'II' ? 'II' : 'III'

  const blockA = buildContrastColumns(fa, levelsA)
  const blockB = buildContrastColumns(fb, levelsB)
  const dfA = a - 1
  const dfB = b - 1
  const dfAB = dfA * dfB

  const blockABData = new Float64Array(n * dfAB)
  for (let i = 0; i < n; i++) {
    let idx = 0
    for (let ca = 0; ca < dfA; ca++) {
      const va = blockA.data[i * dfA + ca] ?? 0
      for (let cb = 0; cb < dfB; cb++) {
        blockABData[i * dfAB + idx++] = va * (blockB.data[i * dfB + cb] ?? 0)
      }
    }
  }
  const blockAB = { cols: dfAB, data: blockABData }

  const meanY = meanOf(ys)
  let ssTotal = 0
  for (let i = 0; i < n; i++) {
    const d = (ys[i] ?? 0) - meanY
    ssTotal += d * d
  }

  const fullBlocks = includeInteraction ? [blockA, blockB, blockAB] : [blockA, blockB]
  const fullDesign = combineDesignBlocks(n, fullBlocks)
  if (n <= fullDesign.cols) {
    throw new MdmlError(
      'shape',
      `twoWayAnova: need n > ${fullDesign.cols} observations for residual degrees of freedom, got n=${n}`,
    )
  }

  const fitFull = ols(fullDesign.data, n, fullDesign.cols, ys)
  const rssFull = fitFull.rss
  const dfResid = n - fullDesign.cols
  const msResid = rssFull / dfResid

  const fitAOnly = ols(combineDesignBlocks(n, [blockA]).data, n, 1 + dfA, ys)
  const fitBOnly = ols(combineDesignBlocks(n, [blockB]).data, n, 1 + dfB, ys)
  const fitABMain = includeInteraction
    ? ols(combineDesignBlocks(n, [blockA, blockB]).data, n, 1 + dfA + dfB, ys)
    : fitFull

  let ssA: number
  let ssB: number
  let ssAB = 0

  if (ssType === 'I') {
    ssA = Math.max(0, ssTotal - fitAOnly.rss)
    ssB = Math.max(0, fitAOnly.rss - fitABMain.rss)
    if (includeInteraction) ssAB = Math.max(0, fitABMain.rss - rssFull)
  } else if (ssType === 'II') {
    ssA = Math.max(0, fitBOnly.rss - fitABMain.rss)
    ssB = Math.max(0, fitAOnly.rss - fitABMain.rss)
    if (includeInteraction) ssAB = Math.max(0, fitABMain.rss - rssFull)
  } else {
    // Type III: drop each term from the full orthonormal-contrast model.
    if (includeInteraction) {
      const fitNoA = ols(combineDesignBlocks(n, [blockB, blockAB]).data, n, 1 + dfB + dfAB, ys)
      const fitNoB = ols(combineDesignBlocks(n, [blockA, blockAB]).data, n, 1 + dfA + dfAB, ys)
      ssA = Math.max(0, fitNoA.rss - rssFull)
      ssB = Math.max(0, fitNoB.rss - rssFull)
      ssAB = Math.max(0, fitABMain.rss - rssFull)
    } else {
      ssA = Math.max(0, fitBOnly.rss - rssFull)
      ssB = Math.max(0, fitAOnly.rss - rssFull)
    }
  }

  const makeTerm = (label: string, ss: number, dfTerm: number): AnovaTermResult => {
    const ms = ss / dfTerm
    const fVal = msResid > 0 ? ms / msResid : ss > 0 ? Number.POSITIVE_INFINITY : 0
    const pVal = Number.isFinite(fVal) ? fSf(dfTerm, dfResid, fVal) : 0
    const partialEtaSquared = ss + rssFull > 0 ? ss / (ss + rssFull) : 0
    const base = testResult({
      test: label,
      statistic: fVal,
      pValue: pVal,
      alternative: 'greater',
      df: [dfTerm, dfResid],
    })
    return {
      ...base,
      ss,
      dfTerm,
      ms,
      partialEtaSquared,
    }
  }

  const rSquared = ssTotal > 0 ? Math.max(0, 1 - rssFull / ssTotal) : 1
  const adjustedRSquared = ssTotal > 0 ? 1 - rssFull / dfResid / (ssTotal / (n - 1)) : 1

  return {
    ssType,
    factorA: makeTerm('Factor A', ssA, dfA),
    factorB: makeTerm('Factor B', ssB, dfB),
    ...(includeInteraction ? { interaction: makeTerm('A x B interaction', ssAB, dfAB) } : {}),
    residual: { ss: rssFull, df: dfResid, ms: msResid },
    total: { ss: ssTotal, df: n - 1 },
    rSquared,
    adjustedRSquared,
  }
}
