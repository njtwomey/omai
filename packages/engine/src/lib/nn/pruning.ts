/**
 * Pruning, which zeroes weights to make a network sparse: unstructured magnitude masks per tensor or ranked globally,
 * structured masks that remove whole channels, N:M semi-structured sparsity, sparsity reports, and the lottery-ticket
 * procedure of iterative magnitude pruning with rewinding (Frankle and Carbin 2019). Also constant, linear and
 * polynomial (Zhu and Gupta) sparsity schedules for gradual pruning, as in torch.nn.utils.prune.
 */

import { MdmlError } from '../base/index.ts'
import {
  asTensor,
  fromStorage,
  mul,
  primalOf,
  type Tensor,
  type TensorLike,
  toFlat,
  treeFlatten,
  treeMap,
  treePaths,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import type { Params } from './module/module.ts'

/** A tree of 0/1 masks with the structure of a parameter tree. */
export type Masks = Params

const checkSparsity = (s: number, name: string): void => {
  if (!(s >= 0 && s <= 1)) throw new MdmlError('value', `${name}: sparsity must be in [0, 1], got ${s}`)
}

/** Indices of the k smallest scores (ties broken by position, so results are deterministic). */
function smallest(scores: readonly number[], k: number): number[] {
  return Array.from({ length: scores.length }, (_, i) => i)
    .sort((a, b) => (scores[a] ?? 0) - (scores[b] ?? 0) || a - b)
    .slice(0, k)
}

const maskOf = (shape: readonly number[], zeroed: Iterable<number>, size: number): Tensor => {
  const m = new Float64Array(size).fill(1)
  for (const i of zeroed) m[i] = 0
  return fromStorage(m, [...shape], 'float64')
}

/**
 * The mask of unstructured magnitude pruning (torch.nn.utils.prune.l1_unstructured): zero the round(sparsity n)
 * entries of smallest absolute value.
 *
 * @param x - The tensor.
 * @param sparsity - The fraction to prune.
 * @returns A 0/1 mask with x's shape.
 * @concept pruning
 * @example
 * ```ts
 * import { magnitudeMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(magnitudeMask([0.1, -3, 0.5, 2], 0.5)) // => [0, 1, 0, 1]
 * ```
 */
export function magnitudeMask(x: TensorLike, sparsity: number): Tensor {
  checkSparsity(sparsity, 'magnitudeMask')
  const t = primalOf(asTensor(x))
  const v = toFlat(t)
  return maskOf(t.shape, smallest(v.map(Math.abs), Math.round(sparsity * v.length)), v.length)
}

/** Options of {@link globalMagnitudeMasks}. */
export interface GlobalPruneOptions {
  /** Which leaves to prune (default: those with at least two axes, the weights). */
  readonly filter?: (path: string, value: Value) => boolean
  /** Existing masks: already pruned entries stay pruned. */
  readonly masks?: Masks
}

/**
 * Global magnitude pruning (torch.nn.utils.prune.global_unstructured): rank the eligible weights of a whole parameter
 * tree together and zero the smallest fraction, so layers lose different fractions; other leaves get all-ones masks.
 *
 * @param params - The parameter tree.
 * @param sparsity - The fraction of eligible weights to prune.
 * @param options - A leaf filter and existing masks.
 * @returns Masks with the tree's structure.
 * @example
 * ```ts
 * import { globalMagnitudeMasks } from 'mdml-engine/nn'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const m = globalMagnitudeMasks({ a: tensor([[1, 2]]), b: tensor([[3, 0.5]]) }, 0.5)
 * toArray(m['a'] as never) // => [[0, 1]]
 * ```
 */
export function globalMagnitudeMasks(params: Params, sparsity: number, options: GlobalPruneOptions = {}): Masks {
  checkSparsity(sparsity, 'globalMagnitudeMasks')
  const filter = options.filter ?? ((_p: string, v: Value) => v.shape.length >= 2)
  const entries = Object.entries(treePaths(params))
  const old = options.masks === undefined ? undefined : treePaths(options.masks)
  const scores: number[] = []
  const owner: [number, number][] = []
  for (const [li, [path, v]] of entries.entries()) {
    if (!filter(path, v)) continue
    const data = toFlat(primalOf(v))
    const m = old?.[path] === undefined ? undefined : toFlat(old[path] as Value)
    for (const [i, x] of data.entries()) {
      scores.push(m !== undefined && m[i] === 0 ? -1 : Math.abs(x))
      owner.push([li, i])
    }
  }
  const zeroed = new Map<number, number[]>()
  for (const s of smallest(scores, Math.round(sparsity * scores.length))) {
    const [li, i] = owner[s] as [number, number]
    zeroed.set(li, [...(zeroed.get(li) ?? []), i])
  }
  const { def } = treeFlatten(params)
  const leaves = entries.map(([, v], li) => {
    const size = v.shape.reduce((a, b) => a * b, 1)
    return maskOf(v.shape, zeroed.get(li) ?? [], size)
  })
  return treeUnflatten(def, leaves) as Masks
}

/**
 * The mask of structured pruning (torch.nn.utils.prune.ln_structured): zero whole slices along an axis (output
 * channels, rows) with the smallest L1 or L2 norm, so the layer can be physically shrunk.
 *
 * @param x - The tensor.
 * @param sparsity - The fraction of slices to prune.
 * @param options - The axis (default 0) and norm order (1 or 2, default 2).
 * @returns A 0/1 mask with x's shape.
 * @example
 * ```ts
 * import { structuredMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(structuredMask([[1, 1], [0.1, 0.2], [3, 0]], 1 / 3)) // => [[1, 1], [0, 0], [1, 1]]
 * ```
 */
export function structuredMask(
  x: TensorLike,
  sparsity: number,
  options: { readonly axis?: number; readonly norm?: 1 | 2 } = {},
): Tensor {
  checkSparsity(sparsity, 'structuredMask')
  const t = primalOf(asTensor(x))
  const rank = t.shape.length
  const axis = (options.axis ?? 0) < 0 ? (options.axis ?? 0) + rank : (options.axis ?? 0)
  const n = t.shape[axis] ?? 1
  const inner = t.shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const data = toFlat(t)
  const norms = new Array<number>(n).fill(0)
  for (const [i, v] of data.entries()) {
    const c = Math.floor(i / inner) % n
    norms[c] = (norms[c] ?? 0) + (options.norm === 1 ? Math.abs(v) : v * v)
  }
  const pruned = new Set(smallest(norms, Math.round(sparsity * n)))
  const zeroed: number[] = []
  for (let i = 0; i < data.length; i++) if (pruned.has(Math.floor(i / inner) % n)) zeroed.push(i)
  return maskOf(t.shape, zeroed, data.length)
}

/**
 * The mask of N:M semi-structured sparsity (2:4 on NVIDIA Ampere): in every group of m consecutive entries along the
 * last axis, keep the n of largest magnitude.
 *
 * @param x - The tensor (last axis divisible by m).
 * @param n - Entries kept per group.
 * @param m - Group size.
 * @returns A 0/1 mask with x's shape.
 * @example
 * ```ts
 * import { nmMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(nmMask([[0.1, -3, 0.5, 2]], 2, 4)) // => [[0, 1, 0, 1]]
 * ```
 */
export function nmMask(x: TensorLike, n: number, m: number): Tensor {
  const t = primalOf(asTensor(x))
  const last = t.shape[t.shape.length - 1] ?? 0
  if (!(n >= 0 && n <= m) || last % m !== 0)
    throw new MdmlError('value', `nmMask: need 0 <= n <= m and the last axis (${last}) divisible by m`)
  const data = toFlat(t)
  const zeroed: number[] = []
  for (let g = 0; g < data.length; g += m) {
    const group = data.slice(g, g + m).map(Math.abs)
    // Prune the m - n smallest of the group.
    for (const i of smallest(group, m - n)) zeroed.push(g + i)
  }
  return maskOf(t.shape, zeroed, data.length)
}

/**
 * Multiply parameters by their masks (leaves without a mask are kept).
 *
 * @param params - The parameter tree.
 * @param masks - Masks with the same structure.
 * @returns The pruned parameters, differentiable in params (pruned entries get zero gradient).
 * @example
 * ```ts
 * import { applyMasks } from 'mdml-engine/nn'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(applyMasks({ w: tensor([1, 2]) }, { w: tensor([0, 1]) })['w'] as never) // => [0, 2]
 * ```
 */
export function applyMasks(params: Params, masks: Masks): Params {
  return treeMap((p, m) => mul(p, m), params, masks)
}

/**
 * The fraction of exactly-zero entries of a tensor.
 *
 * @param x - The tensor.
 * @returns zeros / size.
 * @example
 * ```ts
 * import { sparsityOf } from 'mdml-engine/nn'
 * sparsityOf([0, 1, 0, 2]) // => 0.5
 * ```
 */
export function sparsityOf(x: TensorLike): number {
  const v = toFlat(primalOf(asTensor(x)))
  return v.length === 0 ? 0 : v.filter((a) => a === 0).length / v.length
}

/** A per-tensor sparsity report. */
export interface SparsityReport {
  readonly tensors: Readonly<
    Record<string, { readonly zeros: number; readonly size: number; readonly sparsity: number }>
  >
  /** Over every tensor. */
  readonly global: number
}

/**
 * Zeros, sizes and sparsity of every tensor of a tree, and overall.
 *
 * @param params - The parameter tree.
 * @returns The report.
 * @example
 * ```ts
 * import { sparsityReport } from 'mdml-engine/nn'
 * import { tensor } from 'mdml-engine/tensor'
 * sparsityReport({ a: tensor([0, 1]), b: tensor([0, 0]) }).global // => 0.75
 * ```
 */
export function sparsityReport(params: Params): SparsityReport {
  const tensors: Record<string, { zeros: number; size: number; sparsity: number }> = {}
  let zeros = 0
  let size = 0
  for (const [path, v] of Object.entries(treePaths(params))) {
    const data = toFlat(primalOf(v))
    const z = data.filter((a) => a === 0).length
    tensors[path] = { zeros: z, size: data.length, sparsity: data.length === 0 ? 0 : z / data.length }
    zeros += z
    size += data.length
  }
  return { tensors, global: size === 0 ? 0 : zeros / size }
}

/**
 * Lottery-ticket rewinding (Frankle and Carbin 2019; Frankle et al. 2020): keep the mask and reset the surviving
 * weights to their values at initialisation (or at an early step).
 *
 * @param rewindTo - The parameters to rewind to (initial or early-training).
 * @param masks - The pruning masks.
 * @returns The masked rewound parameters.
 * @reference Frankle and Carbin (2019), The lottery ticket hypothesis: finding sparse, trainable neural networks.
 * @example
 * ```ts
 * import { rewind } from 'mdml-engine/nn'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(rewind({ w: tensor([5, 6]) }, { w: tensor([1, 0]) })['w'] as never) // => [5, 0]
 * ```
 */
export function rewind(rewindTo: Params, masks: Masks): Params {
  return applyMasks(rewindTo, masks)
}

/** Options of {@link iterativeMagnitudePruning}. */
export interface IterativePruningOptions {
  /** Pruning rounds (default 3). */
  readonly rounds?: number
  /** The sparsity after the last round. */
  readonly finalSparsity: number
  /** Rewind to these parameters after each round (default the initial ones). */
  readonly rewindTo?: Params
  /** Which leaves to prune (default: at least two axes). */
  readonly filter?: (path: string, value: Value) => boolean
}

/** Result of {@link iterativeMagnitudePruning}. */
export interface IterativePruningResult {
  /** The final masks. */
  readonly masks: Masks
  /** The winning ticket: the rewound parameters under the final masks, ready to retrain. */
  readonly ticket: Params
  /** Per round: the target sparsity and the trained (masked) parameters. */
  readonly rounds: readonly { readonly sparsity: number; readonly trained: Params }[]
}

/**
 * Iterative magnitude pruning with rewinding (the lottery-ticket procedure): train, prune a further fraction of the
 * surviving weights globally (geometric targets 1 - (1 - s)^(r / R)), rewind the survivors, repeat.
 *
 * @param initial - The initial parameters.
 * @param train - Trains from parameters under masks (it should keep pruned weights at zero) and returns the result.
 * @param options - Rounds, final sparsity, rewinding point, leaf filter.
 * @returns The final masks, the ticket and each round.
 * @example
 * ```ts
 * import { iterativeMagnitudePruning } from 'mdml-engine/nn'
 * import { tensor } from 'mdml-engine/tensor'
 * iterativeMagnitudePruning({ w: tensor([[1, 2, 3, 4]]) }, (p) => p, { finalSparsity: 0.5, rounds: 2 }).rounds.length // => 2
 * ```
 */
export function iterativeMagnitudePruning(
  initial: Params,
  train: (params: Params, masks: Masks, round: number) => Params,
  options: IterativePruningOptions,
): IterativePruningResult {
  checkSparsity(options.finalSparsity, 'iterativeMagnitudePruning')
  const rounds = options.rounds ?? 3
  const target = options.rewindTo ?? initial
  let masks: Masks = treeMap((v) => onesOf(v), initial)
  let params = initial
  const history: { sparsity: number; trained: Params }[] = []
  for (let r = 1; r <= rounds; r++) {
    const trained = applyMasks(train(params, masks, r - 1), masks)
    const sparsity = 1 - (1 - options.finalSparsity) ** (r / rounds)
    masks = globalMagnitudeMasks(trained, sparsity, {
      masks,
      ...(options.filter === undefined ? {} : { filter: options.filter }),
    })
    history.push({ sparsity, trained })
    params = rewind(target, masks)
  }
  return { masks, ticket: params, rounds: history }
}

const onesOf = (v: Value): Tensor =>
  fromStorage(new Float64Array(v.shape.reduce((a, b) => a * b, 1)).fill(1), [...v.shape], 'float64')

/** A sparsity schedule: the target sparsity at a step. */
export type SparsitySchedule = (step: number) => number

/**
 * A constant sparsity from a step on (0 before).
 *
 * @param sparsity - The target.
 * @param start - The first pruning step (default 0).
 * @returns The schedule.
 * @example
 * ```ts
 * import { constantSparsity } from 'mdml-engine/nn'
 * [constantSparsity(0.5, 10)(5), constantSparsity(0.5, 10)(10)] // => [0, 0.5]
 * ```
 */
export function constantSparsity(sparsity: number, start = 0): SparsitySchedule {
  checkSparsity(sparsity, 'constantSparsity')
  return (step) => (step < start ? 0 : sparsity)
}

/**
 * Sparsity rising linearly from `initial` at `start` to `final` at `end`, constant outside.
 *
 * @param initial - Sparsity at the start.
 * @param final - Sparsity at the end.
 * @param start - First step.
 * @param end - Last step.
 * @returns The schedule.
 * @example
 * ```ts
 * import { linearSparsity } from 'mdml-engine/nn'
 * linearSparsity(0, 0.8, 0, 10)(5) // => 0.4
 * ```
 */
export function linearSparsity(initial: number, final: number, start: number, end: number): SparsitySchedule {
  return polynomialSparsity(initial, final, start, end, 1)
}

/**
 * The gradual pruning schedule of Zhu and Gupta (2017): s_t = s_f + (s_i - s_f) (1 - (t - t0) / (t1 - t0))^power,
 * pruning fast early and slowly near the end (power 3 in the paper), constant outside [start, end].
 *
 * @param initial - Sparsity at the start.
 * @param final - Sparsity at the end.
 * @param start - First step.
 * @param end - Last step.
 * @param power - The exponent (default 3).
 * @returns The schedule.
 * @reference Zhu and Gupta (2017), To prune, or not to prune: exploring the efficacy of pruning for model compression.
 * @example
 * ```ts
 * import { polynomialSparsity } from 'mdml-engine/nn'
 * polynomialSparsity(0, 0.8, 0, 10)(10) // => 0.8
 * ```
 */
export function polynomialSparsity(
  initial: number,
  final: number,
  start: number,
  end: number,
  power = 3,
): SparsitySchedule {
  checkSparsity(initial, 'polynomialSparsity')
  checkSparsity(final, 'polynomialSparsity')
  if (!(end > start)) throw new MdmlError('value', 'polynomialSparsity: need end > start')
  return (step) => {
    if (step <= start) return initial
    if (step >= end) return final
    return final + (initial - final) * (1 - (step - start) / (end - start)) ** power
  }
}
