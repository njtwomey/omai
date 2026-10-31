/**
 * The Hoeffding tree (Domingos and Hulten 2000; river's `HoeffdingTreeClassifier`): a decision tree learnt from a
 * stream, one example per step. Each leaf keeps class counts and per-feature statistics (a Gaussian per class for a
 * numeric feature, counts per value and class for a categorical one); every `gracePeriod` examples a leaf asks which
 * split would gain most, and splits when its best candidate beats the runner-up by more than the Hoeffding bound
 * sqrt(R^2 ln(1/delta) / 2n), or when the bound has shrunk below a tie threshold. The tree is the same binary `Tree`
 * as the batch grower's, so prediction, drawing and importances are shared. The stream is the rows of the data in order.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import { type LearnData, withDefaultScore } from '../learn/estimator.ts'
import { encodeLabels, matrixView, rowsOf } from '../learn/validation.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type Algorithm, derived, Log, message, scalar, stateFromJson, stateToJson } from '../trace/index.ts'
import { type TreeVerbs, treeVerbs } from './decision-tree.ts'
import { type Tree, treeFeatureImportances } from './tree.ts'

/** Options of the Hoeffding tree. */
export interface HoeffdingOptions {
  /** Examples a leaf sees between attempts to split (river: 200). */
  readonly gracePeriod: number
  /** The confidence of the Hoeffding bound (river: 1e-7). */
  readonly delta: number
  /** Split anyway when the bound falls below this (river: 0.05). */
  readonly tieThreshold: number
  /** `entropy` (information gain) or `gini`. */
  readonly criterion: 'entropy' | 'gini'
  /** Candidate thresholds per numeric feature (river: 10). */
  readonly nSplitPoints: number
  /** A split needs two branches holding at least this fraction of the examples (river: 0.01). */
  readonly minBranchFraction: number
  readonly maxDepth?: number
  /** Indices of the categorical features (split one value against the rest). */
  readonly categorical: readonly number[]
  /** Passes over the data (the stream repeats; default 1). */
  readonly passes: number
}

/** Welford statistics of a feature within a class. */
type Gauss = { readonly n: number; readonly mean: number; readonly m2: number }

/** What a leaf has learnt: counts per class, and per feature its statistics. */
export type LeafStats = {
  readonly counts: readonly number[]
  /** Examples since the last attempt to split. */
  readonly since: number
  /** Per feature: min, max (numeric) over the leaf's examples. */
  readonly min: readonly number[]
  readonly max: readonly number[]
  /** Numeric features: `gauss[j][c]`. */
  readonly gauss: readonly (readonly Gauss[])[]
  /** Categorical features: `values[j]` maps a value to its counts per class. */
  readonly values: readonly (readonly (readonly [number, readonly number[]])[])[]
}

/** A node of the tree being learnt; leaves that can still split keep their statistics. */
export interface HoeffdingNode {
  readonly id: number
  readonly parent: number
  readonly depth: number
  /** Class counts (frozen when the node splits). */
  readonly counts: readonly number[]
  readonly feature: number
  readonly threshold: number
  readonly categories: readonly number[]
  readonly left: number
  readonly right: number
  readonly stats?: LeafStats
}

/** The state of {@link hoeffdingTreeAlgorithm}. */
export interface HoeffdingState {
  readonly nodes: readonly HoeffdingNode[]
  /** Examples seen and how many the tree classified correctly before learning from them (prequential). */
  readonly seen: number
  readonly correct: number
}

const LOG2 = Math.log(2)

function impurityOf(counts: readonly number[], criterion: 'entropy' | 'gini'): number {
  const total = counts.reduce((a, b) => a + b, 0)
  if (total <= 0) return 0
  let s = 0
  for (const c of counts) {
    const p = c / total
    if (criterion === 'gini') s += p * p
    else if (p > 0) s -= (p * Math.log(p)) / LOG2
  }
  return criterion === 'gini' ? 1 - s : s
}

/** The standard normal CDF (Abramowitz and Stegun 26.2.17, accurate to 1e-7: enough to rank splits). */
function normalCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z))
  const d = 0.3989423 * Math.exp((-z * z) / 2)
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  return z > 0 ? 1 - p : p
}

function emptyStats(d: number, k: number): LeafStats {
  return {
    counts: new Array<number>(k).fill(0),
    since: 0,
    min: new Array<number>(d).fill(Number.POSITIVE_INFINITY),
    max: new Array<number>(d).fill(Number.NEGATIVE_INFINITY),
    gauss: Array.from({ length: d }, () => Array.from({ length: k }, () => ({ n: 0, mean: 0, m2: 0 }))),
    values: Array.from({ length: d }, () => []),
  }
}

/** A split suggestion: where, and its merit (impurity decrease). */
interface Suggestion {
  readonly feature: number
  readonly threshold: number
  readonly categories: readonly number[]
  readonly merit: number
  readonly left: readonly number[]
  readonly right: readonly number[]
}

function suggestions(stats: LeafStats, options: HoeffdingOptions, categorical: ReadonlySet<number>): Suggestion[] {
  const k = stats.counts.length
  const total = stats.counts.reduce((a, b) => a + b, 0)
  const before = impurityOf(stats.counts, options.criterion)
  const out: Suggestion[] = []
  const consider = (feature: number, threshold: number, categories: readonly number[], left: number[]): void => {
    const right = stats.counts.map((c, i) => Math.max(0, c - (left[i] ?? 0)))
    const wl = left.reduce((a, b) => a + b, 0)
    const wr = right.reduce((a, b) => a + b, 0)
    if (wl < options.minBranchFraction * total || wr < options.minBranchFraction * total) return
    const merit =
      before - (wl * impurityOf(left, options.criterion) + wr * impurityOf(right, options.criterion)) / total
    out.push({ feature, threshold, categories, merit, left, right })
  }
  for (let j = 0; j < stats.min.length; j++) {
    if (categorical.has(j)) {
      for (const [v, counts] of stats.values[j] ?? []) consider(j, -2, [v], [...counts])
      continue
    }
    const lo = stats.min[j] ?? 0
    const hi = stats.max[j] ?? 0
    if (!(hi > lo)) continue
    for (let p = 1; p <= options.nSplitPoints; p++) {
      const t = lo + ((hi - lo) * p) / (options.nSplitPoints + 1)
      const left = Array.from({ length: k }, (_, c) => {
        const g = stats.gauss[j]?.[c] as Gauss
        if (g.n === 0) return 0
        const sd = Math.sqrt(g.m2 / g.n)
        return g.n * (sd > 0 ? normalCdf((t - g.mean) / sd) : t >= g.mean ? 1 : 0)
      })
      consider(j, t, [], left)
    }
  }
  return out
}

function learn(stats: LeafStats, row: ArrayLike<number>, label: number, categorical: ReadonlySet<number>): LeafStats {
  const counts = [...stats.counts]
  counts[label] = (counts[label] ?? 0) + 1
  const min = [...stats.min]
  const max = [...stats.max]
  const gauss = stats.gauss.map((g) => [...g])
  const values = stats.values.map((v) => [...v])
  for (let j = 0; j < min.length; j++) {
    const x = row[j] ?? 0
    if (Number.isNaN(x)) continue
    if (categorical.has(j)) {
      const at = (values[j] ?? []).findIndex(([v]) => v === x)
      const list = values[j] ?? []
      if (at < 0) {
        const c = new Array<number>(counts.length).fill(0)
        c[label] = 1
        list.push([x, c])
      } else {
        const entry = list[at] as readonly [number, readonly number[]]
        const c = [...entry[1]]
        c[label] = (c[label] ?? 0) + 1
        list[at] = [x, c]
      }
      continue
    }
    min[j] = Math.min(min[j] ?? x, x)
    max[j] = Math.max(max[j] ?? x, x)
    const g = (gauss[j] as Gauss[])[label] as Gauss
    const n = g.n + 1
    const delta = x - g.mean
    const mean = g.mean + delta / n
    ;(gauss[j] as Gauss[])[label] = { n, mean, m2: g.m2 + delta * (x - mean) }
  }
  return { counts, since: stats.since + 1, min, max, gauss, values }
}

function routeTo(nodes: readonly HoeffdingNode[], row: ArrayLike<number>): number {
  let id = 0
  for (;;) {
    const n = nodes[id] as HoeffdingNode
    if (n.left < 0) return id
    const x = row[n.feature] ?? 0
    const goLeft = n.categories.length > 0 ? n.categories.includes(x) : !Number.isNaN(x) && x <= n.threshold
    id = goLeft ? n.left : n.right
  }
}

/**
 * The Hoeffding tree as an algorithm, one stream example per step: the example is first classified by the tree so far
 * (test-then-train, which gives the prequential accuracy), then taught to its leaf, and the leaf tries to split when it
 * has seen `gracePeriod` examples since it last tried. Logs, when subscribed: `tree/split` (the leaf that split, its
 * feature, threshold, merit, the runner-up's merit and the bound), `tree/leaves`, `tree/depth` and `eval/accuracy` (the
 * prequential accuracy so far).
 *
 * @param data - The stream: features `[n, d]` and class labels, in arrival order.
 * @param options - The Hoeffding bound's confidence, grace period and the rest.
 * @returns The algorithm; the tree of a state is {@link hoeffdingTreeOf}.
 * @concept decision-trees
 * @example
 * ```ts
 * import { hoeffdingTreeAlgorithm, hoeffdingTreeOf } from 'mdml-engine/trees'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const options = { gracePeriod: 10, delta: 1e-3, tieThreshold: 0.05, criterion: 'entropy' as const, nSplitPoints: 10, minBranchFraction: 0.01, categorical: [], passes: 1 }
 * const data = { x: Array.from({ length: 100 }, (_, i) => [i / 100]), y: Array.from({ length: 100 }, (_, i) => (i < 50 ? 0 : 1)) }
 * const run = runAlgorithm(hoeffdingTreeAlgorithm(data, options), { maxSteps: 100 })
 * hoeffdingTreeOf(run.state, { classes: [0, 1], nFeatures: 1 }).feature[0] // => 0
 * ```
 */
export function hoeffdingTreeAlgorithm(data: LearnData, options: HoeffdingOptions): Algorithm<HoeffdingState> {
  const m = matrixView(data.x, 'hoeffdingTree', { allowNaN: true })
  if (data.y === undefined) throw new MdmlError('value', 'hoeffdingTree: needs targets y')
  const { classes, codes } = encodeLabels(data.y, 'hoeffdingTree', m.n)
  const k = classes.length
  const categorical = new Set(options.categorical)
  const range = options.criterion === 'entropy' ? Math.log2(Math.max(k, 2)) : 1
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  const rowAt = (t: number): number[] => {
    const i = t % m.n
    return Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
  }
  return {
    name: 'hoeffdingTree',
    config: { gracePeriod: options.gracePeriod, delta: options.delta },
    init: () => ({
      nodes: [
        {
          id: 0,
          parent: -1,
          depth: 0,
          counts: new Array<number>(k).fill(0),
          feature: -1,
          threshold: -2,
          categories: [],
          left: -1,
          right: -1,
          stats: emptyStats(m.d, k),
        },
      ],
      seen: 0,
      correct: 0,
    }),
    step: (s, ctx) => {
      const t = s.seen
      const row = rowAt(t)
      const label = codes[t % m.n] ?? 0
      const nodes = [...s.nodes]
      const id = routeTo(nodes, row)
      const leaf = nodes[id] as HoeffdingNode
      const stats = leaf.stats as LeafStats
      let majority = 0
      for (let c = 1; c < k; c++) if ((stats.counts[c] ?? 0) > (stats.counts[majority] ?? 0)) majority = c
      const correct = s.correct + (majority === label && (stats.counts[majority] ?? 0) > 0 ? 1 : 0)
      let learnt = learn(stats, row, label, categorical)
      nodes[id] = { ...leaf, counts: learnt.counts, stats: learnt }
      const n = learnt.counts.reduce((a, b) => a + b, 0)
      const pure = learnt.counts.filter((c) => c > 0).length < 2
      if (learnt.since >= options.gracePeriod && !pure && leaf.depth < maxDepth) {
        learnt = { ...learnt, since: 0 }
        // The contest is between features: each is represented by its best split.
        const perFeature = new Map<number, Suggestion>()
        for (const c of suggestions(learnt, options, categorical)) {
          const have = perFeature.get(c.feature)
          if (have === undefined || c.merit > have.merit) perFeature.set(c.feature, c)
        }
        const all = [...perFeature.values()].sort((a, b) => b.merit - a.merit)
        const best = all[0]
        const second = all[1]
        const bound = Math.sqrt((range * range * Math.log(1 / options.delta)) / (2 * n))
        const runnerUp = Math.max(0, second?.merit ?? 0)
        if (best !== undefined && best.merit > 0 && (best.merit - runnerUp > bound || bound < options.tieThreshold)) {
          const mk = (nid: number, counts: readonly number[]): HoeffdingNode => ({
            id: nid,
            parent: id,
            depth: leaf.depth + 1,
            counts,
            feature: -1,
            threshold: -2,
            categories: [],
            left: -1,
            right: -1,
            stats: { ...emptyStats(m.d, k), counts },
          })
          const left = mk(nodes.length, best.left)
          const right = mk(nodes.length + 1, best.right)
          // The children start with the split's class estimates but learn their statistics from new examples.
          nodes[id] = {
            id,
            parent: leaf.parent,
            depth: leaf.depth,
            counts: learnt.counts,
            feature: best.feature,
            threshold: best.threshold,
            categories: best.categories,
            left: left.id,
            right: right.id,
          }
          nodes.push(left, right)
          if (ctx.wants('tree/split'))
            ctx.log('tree/split', {
              node: id,
              depth: leaf.depth,
              samples: n,
              feature: best.feature,
              threshold: best.threshold,
              merit: best.merit,
              runnerUp,
              bound,
            })
        } else nodes[id] = { ...leaf, counts: learnt.counts, stats: learnt }
      }
      const seen = s.seen + 1
      if (ctx.wants('tree/leaves')) ctx.log('tree/leaves', nodes.filter((x) => x.left < 0).length)
      if (ctx.wants('tree/depth')) ctx.log('tree/depth', Math.max(...nodes.map((x) => x.depth)))
      if (ctx.wants('eval/accuracy')) ctx.log('eval/accuracy', correct / seen)
      return { nodes, seen, correct }
    },
    done: (s) => (s.seen >= options.passes * m.n ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HoeffdingState,
  }
}

/**
 * The tree of a state as a `Tree` (nodes in the order they were created): leaves predict their majority class.
 *
 * @param state - A state of {@link hoeffdingTreeAlgorithm}.
 * @param info - The class labels and the number of features.
 * @returns The tree.
 * @example
 * ```ts
 * import { hoeffdingTreeOf } from 'mdml-engine/trees'
 * const root = { id: 0, parent: -1, depth: 0, counts: [3, 1], feature: -1, threshold: -2, categories: [], left: -1, right: -1 }
 * hoeffdingTreeOf({ nodes: [root], seen: 4, correct: 3 }, { classes: [0, 1], nFeatures: 1 }).value // => [[3, 1]]
 * ```
 */
export function hoeffdingTreeOf(
  state: HoeffdingState,
  info: { readonly classes: readonly number[]; readonly nFeatures: number },
): Tree {
  const nodes = state.nodes
  const impurity = nodes.map((n) => impurityOf(n.counts, 'entropy'))
  const weight = nodes.map((n) => n.counts.reduce((a, b) => a + b, 0))
  return {
    task: 'classification',
    classes: [...info.classes],
    nFeatures: info.nFeatures,
    feature: nodes.map((n) => (n.left >= 0 ? n.feature : -2)),
    threshold: nodes.map((n) => (n.left >= 0 ? n.threshold : -2)),
    left: nodes.map((n) => n.left),
    right: nodes.map((n) => n.right),
    value: nodes.map((n) => [...n.counts]),
    impurity,
    nSamples: weight,
    weightedNSamples: weight,
    missingLeft: nodes.map(() => false),
    categories: nodes.map((n) => (n.left >= 0 ? [...n.categories] : [])),
  }
}

/** A Hoeffding tree estimator on the iterative base. */
export type HoeffdingEstimator = IterativeEstimator<HoeffdingOptions, HoeffdingState, Tree, TreeVerbs>

/**
 * A Hoeffding tree classifier (river's HoeffdingTreeClassifier) on the iterative base: a fit streams the rows once in
 * order, one example per step; the session's `fittedAt(t)` is the tree after t examples. The paths are `tree/split`,
 * `tree/leaves`, `tree/depth` and the prequential `eval/accuracy`, and the probe `model/importances`.
 *
 * @param hyper - Grace period (default 200), delta (1e-7), tie threshold (0.05), criterion (entropy), split points (10),
 *   depth limit, categorical features and passes.
 * @returns The estimator.
 * @concept decision-trees
 * @example
 * ```ts
 * import { hoeffdingTreeClassifier } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 200 }, (_, i) => [i / 200])
 * const y = x.map(([v = 0]) => (v < 0.5 ? 0 : 1))
 * const fitted = hoeffdingTreeClassifier({ gracePeriod: 20, delta: 1e-3 }).fit({ x, y })
 * toFlat(fitted.predict([[0.1], [0.9]])) // => [0, 1]
 * ```
 */
export function hoeffdingTreeClassifier(hyper: Partial<HoeffdingOptions> = {}): HoeffdingEstimator {
  const defaults: HoeffdingOptions = {
    gracePeriod: hyper.gracePeriod ?? 200,
    delta: hyper.delta ?? 1e-7,
    tieThreshold: hyper.tieThreshold ?? 0.05,
    criterion: hyper.criterion ?? 'entropy',
    nSplitPoints: hyper.nSplitPoints ?? 10,
    minBranchFraction: hyper.minBranchFraction ?? 0.01,
    ...(hyper.maxDepth === undefined ? {} : { maxDepth: hyper.maxDepth }),
    categorical: hyper.categorical ?? [],
    passes: hyper.passes ?? 1,
  }
  const classesOf = (data: LearnData): readonly number[] => encodeLabels(data.y ?? [], 'hoeffdingTree').classes
  return iterative<HoeffdingOptions, HoeffdingState, Tree, TreeVerbs>(
    {
      name: 'hoeffding-tree-classifier',
      kind: 'classifier',
      defaults,
      hyperSpace: (h) =>
        dict({
          gracePeriod: int(1, Math.max(1000, h.gracePeriod), { default: h.gracePeriod, label: 'grace period' }),
          delta: real(1e-10, 0.5, { log: true, default: h.delta, label: 'delta' }),
          tieThreshold: real(0.001, 0.5, { log: true, default: h.tieThreshold, label: 'tie threshold' }),
          criterion: choice(['entropy', 'gini'] as const, { default: h.criterion }),
        }),
      algorithm: (data, h) => hoeffdingTreeAlgorithm(data, h),
      params: (state, data) =>
        hoeffdingTreeOf(state, {
          classes: classesOf(data),
          nFeatures: matrixView(data.x, 'hoeffdingTree', { allowNaN: true }).d,
        }),
      verbs: (tree) => withDefaultScore('classifier', treeVerbs(tree)),
      maxSteps: (h, data) => h.passes * rowsOf(data.x),
      contract: (data) => ({
        'tree/split': message({
          description: 'a leaf split: its feature and threshold, the merit, the runner-up and the bound',
        }),
        'tree/leaves': scalar({ description: 'leaves so far' }),
        'tree/depth': scalar({ description: 'depth of the deepest node' }),
        'eval/accuracy': scalar({
          description: 'prequential accuracy: each example classified before it is learnt',
          direction: 'higher',
        }),
        'model/importances': derived<HoeffdingState>({
          description: 'impurity importance of each feature in the tree so far',
          compute: (state) =>
            treeFeatureImportances(
              hoeffdingTreeOf(state, {
                classes: classesOf(data),
                nFeatures: matrixView(data.x, 'hoeffdingTree', { allowNaN: true }).d,
              }),
            ),
        }),
      }),
      logs: { 'tree/split': Log(), 'tree/leaves': Log(), 'eval/accuracy': Log() },
    },
    defaults,
  )
}
