/**
 * The streaming Bayesian tree: rows arrive one at a time or in batches, each is routed to a leaf, and the leaf's
 * conjugate statistics are updated by addition, so the leaf posterior after any prefix of the stream is exactly the batch
 * posterior of the rows seen. A leaf's precision (its evidence, in observations) only grows, and its step size, the
 * reciprocal of the precision, only falls: a leaf that has seen few rows moves a lot per new row, a well-visited leaf
 * barely moves. When a leaf has gathered enough new rows it asks the exact split question on the rows it holds, and splits
 * when the posterior odds say the split pays for itself (the same rule as {@link growBayesTree}).
 */

import {
  add,
  arange,
  concat,
  lessEqual,
  logicalNot,
  nonzero,
  sum,
  type TensorLike,
  take,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { childRules, splitLogOdds } from './grow.ts'
import type { LeafModel, LeafPosterior } from './leaf.ts'
import type { TreePrior } from './prior.ts'
import { treePrior } from './prior.ts'
import { searchBayesSplits } from './search.ts'
import {
  type BayesData,
  type BayesTree,
  bayesLeaf,
  bayesSplit,
  column,
  inputMatrix,
  type LeafBox,
  leafBoxes,
} from './tree.ts'

/** The rows a leaf holds and its accumulated statistics. */
export interface StreamLeaf {
  /** The inputs of the rows the leaf holds, `[m, d]`. */
  readonly x: Value
  /** The per-row statistics of those rows, `[m, size]`. */
  readonly rows: Value
  /** The accumulated statistics `[size]`: the sum of the rows' statistics, updated by addition. */
  readonly stats: Value
  readonly n: number
  /** Rows that arrived since the leaf last asked whether to split. */
  readonly since: number
  readonly depth: number
}

/** A node of the streaming tree. */
export type StreamNode =
  | { readonly kind: 'leaf'; readonly leaf: StreamLeaf }
  | {
      readonly kind: 'split'
      readonly feature: number
      readonly threshold: number
      readonly left: StreamNode
      readonly right: StreamNode
    }

/** A split made while rows arrived. */
export interface StreamSplit {
  /** The rows seen when it was made. */
  readonly seen: number
  readonly feature: number
  readonly threshold: number
  readonly depth: number
  /** The rows the leaf held. */
  readonly n: number
  readonly gain: number
  readonly logOdds: number
}

/** The state of a streaming tree. */
export interface StreamState {
  readonly root: StreamNode
  readonly seen: number
  /** The splits so far, in order. */
  readonly splits: readonly StreamSplit[]
}

/** The settings of a streaming tree. */
export interface StreamOptions {
  readonly prior?: TreePrior
  /** A leaf asks whether to split after this many new rows (default 10). */
  readonly checkEvery?: number
}

/**
 * The streaming tree before any row: one empty leaf.
 *
 * @param model - The leaf model.
 * @param d - The number of features.
 * @returns The state.
 * @concept bayesian-trees
 * @index function Streaming Bayesian tree: start empty, update leaves conjugately as rows arrive.
 * @example
 * ```ts
 * import { normalGammaLeaf, streamInit } from 'omai-engine/trees'
 * const state = streamInit(normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 }), 1)
 * state.seen // => 0
 * ```
 */
export function streamInit(model: LeafModel, d: number): StreamState {
  const leaf: StreamLeaf = {
    x: zeros([0, d]),
    rows: zeros([0, model.size]),
    stats: zeros([model.size]),
    n: 0,
    since: 0,
    depth: 0,
  }
  return { root: { kind: 'leaf', leaf }, seen: 0, splits: [] }
}

/**
 * Take in a batch of rows (a batch of one is one row arriving). Each leaf adds the statistics of its rows to its own;
 * a leaf with enough new rows then asks the split question and may split, and its children ask in turn.
 *
 * @param state - The state.
 * @param model - The leaf model.
 * @param x - The batch's inputs, `[m, d]`.
 * @param y - The batch's targets.
 * @param options - The prior and how often a leaf asks.
 * @returns The new state.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { normalGammaLeaf, streamInit, streamLeaves, streamUpdate } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const first = streamUpdate(streamInit(model, 1), model, [[0.2]], [1])
 * const second = streamUpdate(first, model, [[0.4]], [3])
 * streamLeaves(second, model).map((l) => l.stepSize) // => [0.3333333333333333]
 * ```
 */
export function streamUpdate(
  state: StreamState,
  model: LeafModel,
  x: TensorLike,
  y: TensorLike,
  options: StreamOptions = {},
): StreamState {
  return advance(state, model, inputMatrix(x), model.statistics(y), options)
}

/** The rows of a matrix where a mask is true. */
const select = (x: Value, mask: Value): Value => take(x, nonzero(mask)[0] as Value, 0)

/** One update with the per-row statistics already made. */
function advance(state: StreamState, model: LeafModel, xm: Value, rows: Value, options: StreamOptions): StreamState {
  const prior = options.prior ?? treePrior()
  const every = options.checkEvery ?? 10
  const seen = state.seen + (xm.shape[0] ?? 0)
  const splits: StreamSplit[] = [...state.splits]
  const settle = (leaf: StreamLeaf): StreamNode => {
    if (leaf.since < every || leaf.n < 2 * prior.minLeaf) return { kind: 'leaf', leaf }
    const d = leaf.x.shape[1] ?? 1
    const found = searchBayesSplits({ x: leaf.x, rows: leaf.rows, model, n: leaf.n, d }, undefined, prior.minLeaf)
    const best = found.best
    const checked: StreamLeaf = { ...leaf, since: 0 }
    if (best === undefined) return { kind: 'leaf', leaf: checked }
    const here = { x: leaf.x, rows: leaf.rows, model, n: leaf.n, d }
    const logOdds = splitLogOdds(
      prior,
      leaf.depth,
      found.available,
      best.gain,
      childRules(here, undefined, best.feature, best.threshold, prior.minLeaf),
    )
    if (!(logOdds > 0)) return { kind: 'leaf', leaf: checked }
    const { feature, threshold, gain } = best
    splits.push({ seen, feature, threshold, depth: leaf.depth, n: leaf.n, gain, logOdds })
    const goesLeft = lessEqual(column(leaf.x, feature), threshold)
    const part = (mask: Value): StreamLeaf => {
      const px = select(leaf.x, mask)
      const pr = select(leaf.rows, mask)
      const n = px.shape[0] ?? 0
      return { x: px, rows: pr, stats: sum(pr, 0), n, since: n, depth: leaf.depth + 1 }
    }
    return {
      kind: 'split',
      feature,
      threshold,
      left: settle(part(goesLeft)),
      right: settle(part(logicalNot(goesLeft))),
    }
  }
  const route = (node: StreamNode, bx: Value, br: Value): StreamNode => {
    const m = bx.shape[0] ?? 0
    if (m === 0) return node
    if (node.kind === 'leaf') {
      const leaf: StreamLeaf = {
        x: concat([node.leaf.x, bx], 0),
        rows: concat([node.leaf.rows, br], 0),
        stats: add(node.leaf.stats, sum(br, 0)),
        n: node.leaf.n + m,
        since: node.leaf.since + m,
        depth: node.leaf.depth,
      }
      return settle(leaf)
    }
    const goesLeft = lessEqual(column(bx, node.feature), node.threshold)
    const goesRight = logicalNot(goesLeft)
    return {
      ...node,
      left: route(node.left, select(bx, goesLeft), select(br, goesLeft)),
      right: route(node.right, select(bx, goesRight), select(br, goesRight)),
    }
  }
  return { root: route(state.root, xm, rows), seen, splits }
}

/**
 * The tree the stream has grown so far.
 *
 * @param state - The state.
 * @returns The tree.
 * @concept bayesian-trees
 * @example
 * ```ts
 * import { normalGammaLeaf, streamInit, streamTree } from 'omai-engine/trees'
 * streamTree(streamInit(normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 }), 1)).kind // => 'leaf'
 * ```
 */
export function streamTree(state: StreamState): BayesTree {
  const build = (node: StreamNode): BayesTree =>
    node.kind === 'leaf' ? bayesLeaf : bayesSplit(node.feature, node.threshold, build(node.left), build(node.right))
  return build(state.root)
}

/** A leaf of the streaming tree: its box, its posterior and its step size. */
export interface StreamLeafPosterior extends LeafBox, LeafPosterior {}

/**
 * The leaves of the streaming tree with their posteriors, in preorder: how many rows each holds, its posterior mean, its
 * precision and its step size (one over the precision).
 *
 * @param state - The state.
 * @param model - The leaf model.
 * @returns One entry per leaf.
 * @concept bayesian-trees
 * @index function Leaves of a streaming tree: posterior, precision and the adaptive step size 1/precision.
 * @example
 * ```ts
 * import { normalGammaLeaf, streamInit, streamLeaves, streamUpdate } from 'omai-engine/trees'
 * const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
 * const state = streamUpdate(streamInit(model, 1), model, [[0.2], [0.4], [0.9]], [1, 1, 1])
 * streamLeaves(state, model).map((l) => [l.n, l.precision]) // => [[3, 4]]
 * ```
 */
export function streamLeaves(state: StreamState, model: LeafModel): StreamLeafPosterior[] {
  const leaves: StreamLeaf[] = []
  const walk = (node: StreamNode): void => {
    if (node.kind === 'leaf') leaves.push(node.leaf)
    else {
      walk(node.left)
      walk(node.right)
    }
  }
  walk(state.root)
  const d = leaves[0]?.x.shape[1] ?? 1
  const boxes = leafBoxes(streamTree(state), d)
  return leaves.map((leaf, i) => ({
    ...(boxes[i] as LeafBox),
    ...model.describe(toFlat(leaf.stats)),
  }))
}

/** The paths the streaming algorithm logs. */
export type StreamPath = 'stream/seen' | 'stream/leaves' | 'stream/split' | 'stream/stepSizes'

/**
 * The streaming tree as a stepped algorithm: step `t` takes in the rows `t * batch .. (t + 1) * batch` of the data, in
 * order, as if they had arrived then. Logs `stream/seen`, `stream/leaves`, `stream/stepSizes` (every leaf's step size) and
 * `stream/split` for each split made in the step.
 *
 * @param data - The stream, as data (rows in arrival order).
 * @param options - The batch size (default 1), the prior and how often a leaf asks.
 * @returns The algorithm.
 * @concept bayesian-trees
 * @index algorithm Streaming Bayesian tree: conjugate leaf updates, splits by posterior odds.
 * @example
 * ```ts
 * import { normalGammaLeaf, streamAlgorithm, bayesData } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = normalGammaLeaf({ m0: 2.5, kappa0: 1, alpha0: 2, beta0: 2 })
 * const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
 * runAlgorithm(streamAlgorithm(data, { batch: 2 }), { maxSteps: 3 }).state.seen // => 6
 * ```
 */
export function streamAlgorithm(
  data: BayesData,
  options: StreamOptions & { readonly batch?: number } = {},
): Algorithm<StreamState, StreamPath> {
  const batch = options.batch ?? 1
  const model = data.model
  return {
    name: 'streamTree',
    config: { batch, checkEvery: options.checkEvery ?? 10, ...(options.prior ?? treePrior()) },
    init: () => streamInit(model, data.d),
    step: (state, ctx) => {
      const from = ctx.t * batch
      const to = Math.min(data.n, from + batch)
      const rows = take(data.x, indices(from, to), 0)
      const stats = take(data.rows, indices(from, to), 0)
      const next = advance(state, model, rows, stats, options)
      if (ctx.wants('stream/seen')) ctx.log('stream/seen', next.seen)
      if (ctx.wants('stream/leaves')) ctx.log('stream/leaves', streamLeaves(next, model).length)
      if (ctx.wants('stream/stepSizes'))
        ctx.log(
          'stream/stepSizes',
          streamLeaves(next, model).map((l) => l.stepSize),
        )
      if (ctx.wants('stream/split'))
        for (const s of next.splits.slice(state.splits.length)) ctx.log('stream/split', { ...s })
      return next
    },
    done: (state) => (state.seen >= data.n ? 'converged' : undefined),
  }
}

const indices = (from: number, to: number): Value => arange(from, to, 1, { dtype: 'int32' })
