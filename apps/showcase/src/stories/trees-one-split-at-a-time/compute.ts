/**
 * The computation of 'Trees, one split at a time, and all trees at once': a staircase with noise is fitted by one tree grown
 * greedily, one split at a time, and by the posterior over all trees (exactly, for trees up to two levels deep, which have
 * up to three cuts, over thirty candidate cuts, and by a sampler that draws whole trees). The results are turned into plain numbers for the figures: the
 * tree after each split, the average over all trees, the chance of a cut at each position, and how closely the sampler
 * agrees with the exact answer. It is a pure function of the seed, and it runs in the story's worker. The regions are the
 * code the page shows.
 */

// region imports
import { key } from 'omai-engine/random'
import { linspace } from 'omai-engine/tensor'
import {
  bayesData,
  drawnCurves,
  enumerateTrees,
  growBayesTree,
  normalGammaLeaf,
  predictEnumerated,
  predictSampled,
  probabilityOfCutIn,
  quantileRules,
  regressionPrior,
  rulePosterior,
  sampleTrees,
  topTrees,
  treePrior,
  treeSpace,
} from 'omai-engine/trees'
// endregion

import { toFlat } from 'omai-engine/tensor'
import {
  type BayesData,
  type BayesGrowState,
  type BayesTree,
  fittedLeaves,
  treeAfter,
  treeSummary,
} from 'omai-engine/trees'
import type { Say } from '../../kit/protocol.ts'
import { NEAR, simulate, X_RANGE } from './data.ts'

/** A leaf of a tree: where it reaches, how many points it holds, its posterior mean and the step size 1/precision. */
export interface Leaf {
  readonly from: number
  readonly to: number
  readonly n: number
  readonly mean: number
  readonly sd: number
  readonly stepSize: number
}

/** A tree as a curve over the grid, with the leaves it is made of. */
export interface Stage {
  /** The cuts of the tree, left to right. */
  readonly cuts: readonly number[]
  readonly leaves: readonly Leaf[]
  /** The mean curve and a band of two standard deviations of the mean, over the grid. */
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
}

/** One split of the greedy tree. */
export interface Cut {
  readonly threshold: number
  /** The log Bayes factor of the split against the leaf it cut. */
  readonly gain: number
  /** The log posterior odds of the split, the prior included; above 0 means it pays for itself. */
  readonly logOdds: number
  /** Points in the leaf that was cut, and in the two leaves it made. */
  readonly n: number
  readonly nLeft: number
  readonly nRight: number
  /** The chance, under the posterior over all trees, that some cut lies within `NEAR` of this one. */
  readonly chanceNear: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  readonly grid: readonly number[]
  readonly greedy: {
    /** The tree before any split and after each split. */
    readonly stages: readonly Stage[]
    readonly cuts: readonly Cut[]
    /** The best split still on offer when the growth stopped, with its odds (below 0), or none. */
    readonly next: { readonly threshold: number; readonly gain: number; readonly logOdds: number } | undefined
  }
  readonly posterior: {
    /** The candidate cuts. */
    readonly thresholds: readonly number[]
    /** The expected number of times a cut is at each candidate, anywhere in the tree (what the figure draws). */
    readonly usage: readonly number[]
    /** The chance that some cut of the tree is at each candidate. */
    readonly cutChance: readonly number[]
    /** The chance that the root cut is at each candidate. */
    readonly root: readonly number[]
    /** The chance of no cut, one cut, two cuts and three cuts: they add to one. */
    readonly splits: readonly number[]
    readonly trees: number
    readonly mean: readonly number[]
    readonly lower: readonly number[]
    readonly upper: readonly number[]
    /** The most probable trees: their cuts and probability. */
    readonly top: readonly { readonly cuts: readonly number[]; readonly probability: number }[]
  }
  readonly sampled: {
    /** The mean curve of some of the drawn trees. */
    readonly curves: readonly (readonly number[])[]
    readonly draws: number
    readonly chains: number
    /** How often a proposed move was accepted. */
    readonly acceptance: number
    readonly rhat: number
    /** The largest gap between the sampler's mean curve and the exact one, and the total variation of the root cut. */
    readonly curveGap: number
    readonly rootGap: number
  }
}

/** How much to sample: the run uses {@link DEFAULT_SIZE}, a test a small one. */
export interface Size {
  readonly grid: number
  readonly cuts: number
  readonly chains: number
  readonly iterations: number
  readonly warmup: number
  readonly thin: number
  readonly curves: number
}

/** The size of the real run: thirty candidate cuts, four chains of twenty thousand moves. */
export const DEFAULT_SIZE: Size = {
  grid: 120,
  cuts: 30,
  chains: 4,
  iterations: 20000,
  warmup: 500,
  thin: 20,
  curves: 30,
}

const cutsOf = (tree: BayesTree): number[] =>
  tree.kind === 'leaf' ? [] : [...cutsOf(tree.left), tree.threshold, ...cutsOf(tree.right)].sort((a, b) => a - b)

/** A tree as a curve over the grid, with its leaves. */
function stageOf(data: BayesData, tree: BayesTree, grid: readonly number[]): Stage {
  const read = data.model.read(treeSummary(data, tree, grid))
  const sd = read.sd as readonly number[]
  return {
    cuts: cutsOf(tree),
    leaves: fittedLeaves(data, tree).map((l) => ({
      from: Math.max(l.lower[0] as number, X_RANGE[0]),
      to: Math.min(l.upper[0] as number, X_RANGE[1]),
      n: l.n,
      mean: l.value[0] as number,
      sd: l.sd ?? 0,
      stepSize: l.stepSize,
    })),
    mean: read.mean,
    lower: read.mean.map((m, i) => m - 2 * (sd[i] as number)),
    upper: read.mean.map((m, i) => m + 2 * (sd[i] as number)),
  }
}

/** The best split a growth left unmade, when it stopped. */
function stopped(state: BayesGrowState): Result['greedy']['next'] {
  const open = state.nodes.filter((n) => n.splitAt === undefined && n.best !== undefined)
  const best = open.reduce<(typeof open)[number] | undefined>(
    (a, b) => (a === undefined || (b.best?.logOdds ?? 0) > (a.best?.logOdds ?? 0) ? b : a),
    undefined,
  )
  return best?.best === undefined
    ? undefined
    : { threshold: best.best.threshold, gain: best.best.gain, logOdds: best.best.logOdds }
}

/**
 * The computation, from a seed: grow the greedy tree, enumerate all trees, sample trees, and summarise.
 *
 * @param seed - The sampler's seed (the data do not change with it).
 * @param say - Says what is happening, for the play button.
 * @param size - How much to compute.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const points = simulate()
  say('Growing one tree, a split at a time')
  // region greedy
  const prior = treePrior({ maxDepth: 2, minLeaf: 3 })
  const model = normalGammaLeaf(regressionPrior(points.y))
  const data = bayesData(points.x, points.y, model)
  // Take the split with the best odds, again and again, until none pays for itself.
  const grown = growBayesTree(data, { prior })
  // endregion
  say('Weighing every tree up to two levels deep')
  // region exact
  // Thirty cuts, spread over the data, that a tree may use.
  const rules = quantileRules(points.x, size.cuts)
  const space = treeSpace(data, prior, rules)
  // Every tree up to two levels deep (up to three cuts), each weighted by its prior and the exact evidence.
  const exact = enumerateTrees(space)
  // The prediction of the average over all of those trees.
  const grid = linspace(X_RANGE[0], X_RANGE[1], size.grid)
  const average = predictEnumerated(exact, grid)
  // endregion
  say('Drawing trees from the same posterior')
  // region sampled
  const draws = sampleTrees(space, {
    key: key(seed),
    chains: size.chains,
    iterations: size.iterations,
    warmup: size.warmup,
    thin: size.thin,
  })
  const curves = drawnCurves(draws, grid, size.curves)
  // endregion
  say('Reading the results')
  const gridValues = toFlat(grid)
  const stages = grown.steps.map((_, k) => k + 1)
  const found = rulePosterior(draws)
  const sampledMean = predictSampled(draws, grid).mean
  const sd = average.sd as readonly number[]
  return {
    grid: gridValues,
    greedy: {
      stages: [0, ...stages].map((k) => stageOf(data, treeAfter(grown.state, k), gridValues)),
      cuts: grown.steps.map((s) => ({
        threshold: s.threshold,
        gain: s.gain,
        logOdds: s.logOdds,
        n: s.n,
        nLeft: s.nLeft,
        nRight: s.nRight,
        chanceNear: probabilityOfCutIn(exact, s.threshold - NEAR, s.threshold + NEAR),
      })),
      next: stopped(grown.state),
    },
    posterior: {
      thresholds: rules.thresholds,
      usage: exact.usage,
      cutChance: rules.thresholds.map((t) => probabilityOfCutIn(exact, t, t)),
      root: exact.root,
      splits: exact.splits,
      trees: exact.trees,
      mean: average.mean,
      lower: average.mean.map((m, i) => m - 2 * (sd[i] as number)),
      upper: average.mean.map((m, i) => m + 2 * (sd[i] as number)),
      top: topTrees(exact, 3).map((t) => ({ cuts: cutsOf(t.tree), probability: t.probability })),
    },
    sampled: {
      curves,
      draws: draws.draws.length,
      chains: size.chains,
      acceptance: draws.acceptance,
      rhat: draws.diagnostics.logPosterior.rhat,
      curveGap: Math.max(...average.mean.map((m, i) => Math.abs(m - (sampledMean[i] as number)))),
      rootGap:
        0.5 *
        (Math.abs(exact.stump - found.stump) +
          exact.root.reduce((a, p, i) => a + Math.abs(p - (found.root[i] as number)), 0)),
    },
  }
}

/**
 * The run of the story: the computation at its full size.
 *
 * @param seed - The sampler's seed.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
