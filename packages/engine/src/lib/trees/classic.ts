/**
 * The classic tree growers, for comparison with CART: ID3 (Quinlan 1986: multiway splits on categorical features by
 * information gain, each feature used once on a path), C4.5 (Quinlan 1993: gain ratio among the features whose gain is
 * at least average, multiway splits on categorical features and binary thresholds on continuous ones, pessimistic
 * error pruning) and CHAID (Kass 1980: categories whose class distributions do not differ significantly are merged,
 * then the feature with the smallest Bonferroni-adjusted chi-square p-value splits the node, if it is significant).
 * The growth is a stepped algorithm, one node per step. Trees have any number of children per node, so they are their
 * own `ClassicTree` type rather than a `FittedTree`; ID3 and CHAID read continuous features through equal-frequency bins.
 */

import { argmaxOf, maxOf, OmaiError } from '../base/index.ts'
import { type SplitCandidate, type TreeNode, type TreeRecord, treeRecord } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import type { FittedFacts, LearnData, Verbs } from '../learn/estimator.ts'
import { withDefaultScore } from '../learn/estimator.ts'
import { classNamesOf, featureNamesOf } from '../learn/explained.ts'
import { encodeLabels, matrixView, rowsOf, sampleWeightsOf } from '../learn/validation.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { chi2Contingency } from '../stats/tests/contingency.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  derived,
  Log,
  message,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** Which classic grower. */
export type ClassicMode = 'id3' | 'c45' | 'chaid'

/** Options of the classic growers. */
export interface ClassicOptions {
  readonly mode: ClassicMode
  /** Indices of the features whose values are category codes (others are continuous; ID3 and CHAID bin them). */
  readonly categorical: readonly number[]
  readonly maxDepth?: number
  readonly minSamplesSplit: number
  /** A split needs at least two branches with this many cases (C4.5's minimum), or every branch (ID3, CHAID). */
  readonly minSamplesLeaf: number
  /** Equal-frequency bins for the continuous features of ID3 and CHAID. */
  readonly bins: number
  /** CHAID: the p-value above which two categories merge, and the adjusted p-value below which a node splits. */
  readonly alphaMerge: number
  readonly alphaSplit: number
  /** C4.5's confidence factor for pessimistic pruning (smaller prunes more), and whether to prune. */
  readonly confidence: number
  readonly prune: boolean
}

/**
 * A tree with any number of children per node. Node i tests `feature[i]`: a continuous C4.5 split has two children
 * (`x <= threshold`, then `x > threshold`), any other split has one child per group of category codes listed in
 * `branches[i]`. Leaves have no children. `cuts` are the bin edges of binned features (null: the value is the code).
 */
export type ClassicTree = {
  readonly mode: ClassicMode
  readonly classes: number[]
  readonly nFeatures: number
  readonly cuts: (number[] | null)[]
  readonly feature: number[]
  readonly threshold: number[]
  readonly children: number[][]
  readonly branches: number[][][]
  readonly value: number[][]
  readonly impurity: number[]
  readonly nSamples: number[]
  readonly depth: number[]
}

/** A node in growth. */
interface ClassicNode {
  readonly id: number
  readonly parent: number
  readonly depth: number
  readonly rows: Int32Array
  readonly value: readonly number[]
  readonly impurity: number
  /** Features already tested on the path to the node (ID3 and CHAID use each once). */
  readonly used: readonly number[]
  readonly candidates: readonly SplitCandidate[]
  readonly feature: number
  readonly threshold: number
  readonly children: readonly number[]
  readonly branches: readonly (readonly number[])[]
  readonly order: number
}

/** The state of {@link classicTreeAlgorithm}. */
export interface ClassicState {
  readonly nodes: readonly ClassicNode[]
  readonly open: readonly number[]
}

const LOG2 = Math.log(2)

function entropyBits(counts: ArrayLike<number>): number {
  let total = 0
  for (let i = 0; i < counts.length; i++) total += counts[i] ?? 0
  if (total <= 0) return 0
  let h = 0
  for (let i = 0; i < counts.length; i++) {
    const p = (counts[i] ?? 0) / total
    if (p > 0) h -= (p * Math.log(p)) / LOG2
  }
  return h
}

/** A candidate multiway or binary split of a node: groups of rows, ranked by gain and a score. */
interface SplitOption {
  readonly feature: number
  readonly threshold: number
  readonly branches: readonly (readonly number[])[]
  readonly groups: readonly (readonly number[])[]
  readonly gain: number
  readonly ratio: number
  readonly pValue: number
  readonly score: number
}

function bandsOf(values: readonly number[], bins: number): number[] {
  const sorted = Float64Array.from(values).sort()
  const cuts: number[] = []
  for (let b = 1; b < bins; b++) {
    const v = sorted[Math.min(sorted.length - 1, Math.floor((b * sorted.length) / bins))] ?? 0
    const prev = sorted[Math.max(0, Math.floor((b * sorted.length) / bins) - 1)] ?? v
    const cut = (v + prev) / 2
    if (
      cut > (cuts[cuts.length - 1] ?? Number.NEGATIVE_INFINITY) &&
      cut > (sorted[0] ?? 0) &&
      cut < (sorted[sorted.length - 1] ?? 0)
    )
      cuts.push(cut)
  }
  return cuts
}

function codeOf(value: number, cuts: readonly number[] | null): number {
  if (cuts === null) return value
  let c = 0
  while (c < cuts.length && value > (cuts[c] ?? 0)) c++
  return c
}

/** The chi-square p-value of a table of class counts per group, dropping empty classes (1 when nothing to test). */
function pValueOf(counts: readonly (readonly number[])[]): number {
  const k = counts[0]?.length ?? 0
  const live = Array.from({ length: k }, (_, c) => c).filter((c) => counts.some((g) => (g[c] ?? 0) > 0))
  if (counts.length < 2 || live.length < 2) return 1
  return chi2Contingency(
    counts.map((g) => live.map((c) => g[c] ?? 0)),
    { correction: false },
  ).pValue
}

/** Bonferroni multiplier of CHAID for a nominal predictor: the ways c categories merge into r groups. */
function bonferroni(c: number, r: number): number {
  const fact = (m: number): number => (m <= 1 ? 1 : m * fact(m - 1))
  let b = 0
  for (let i = 0; i < r; i++) b += ((-1) ** i * (r - i) ** c) / (fact(i) * fact(r - i))
  return b
}

/**
 * The growth of an ID3, C4.5 or CHAID tree as an algorithm, one node per step, depth-first. Logs, when subscribed:
 * `tree/split` (the node, its feature, how many branches and the gain), `tree/leaves`, and for CHAID `tree/merge`
 * (the categories merged at the node, with the p-value that allowed it).
 *
 * @param data - Features `[n, d]`, class labels and optional weights.
 * @param options - The grower and its limits.
 * @returns The algorithm; build the tree of a state with {@link classicTreeOf}.
 * @concept decision-trees
 * @example
 * ```ts
 * import { classicTreeAlgorithm, classicTreeOf } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const options = { mode: 'id3' as const, categorical: [0], minSamplesSplit: 2, minSamplesLeaf: 1, bins: 4, alphaMerge: 0.05, alphaSplit: 0.05, confidence: 0.25, prune: false }
 * const data = { x: [[0], [0], [1], [1], [2], [2]], y: [0, 0, 1, 1, 0, 0] }
 * const alg = classicTreeAlgorithm(data, options)
 * classicTreeOf(runAlgorithm(alg, { maxSteps: 20 }).state, data, options).children[0] // => [1, 2, 3]
 * ```
 */
export function classicTreeAlgorithm(
  data: LearnData,
  options: ClassicOptions,
): Algorithm<ClassicState, ClassicGrowPath> {
  const m = matrixView(data.x, 'classicTree', { allowNaN: true })
  if (data.y === undefined) throw new OmaiError('value', 'classicTree: needs targets y')
  const { classes, codes } = encodeLabels(data.y, 'classicTree', m.n)
  const w = sampleWeightsOf(data.weights, m.n, 'classicTree')
  const k = classes.length
  const categorical = new Set(options.mode === 'c45' ? options.categorical : Array.from({ length: m.d }, (_, j) => j))
  const cuts = cutsOf(m, options, categorical)
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  const code = (r: number, j: number): number => codeOf(m.data[r * m.d + j] ?? 0, cuts[j] ?? null)
  const countsOf = (rows: ArrayLike<number>): number[] => {
    const c = new Array<number>(k).fill(0)
    for (let i = 0; i < rows.length; i++)
      c[codes[rows[i] ?? 0] ?? 0] = (c[codes[rows[i] ?? 0] ?? 0] ?? 0) + (w[rows[i] ?? 0] ?? 0)
    return c
  }
  const born = (id: number, parent: number, depth: number, rows: Int32Array, used: readonly number[]): ClassicNode => {
    const value = countsOf(rows)
    return {
      id,
      parent,
      depth,
      rows,
      value,
      impurity: entropyBits(value),
      used,
      candidates: [],
      feature: -1,
      threshold: -2,
      children: [],
      branches: [],
      order: -1,
    }
  }
  /** The options at a node for one feature. */
  const optionsFor = (node: ClassicNode, j: number): SplitOption | undefined => {
    const total = node.value.reduce((a, b) => a + b, 0)
    const rows = Array.from(node.rows).filter((r) => !Number.isNaN(m.data[r * m.d + j] ?? 0))
    if (rows.length < 2) return undefined
    if (!categorical.has(j)) {
      // C4.5, continuous: the cut of the best gain between consecutive distinct values.
      const sorted = rows.sort((a, b) => (m.data[a * m.d + j] ?? 0) - (m.data[b * m.d + j] ?? 0))
      const left = new Array<number>(k).fill(0)
      const right = countsOf(sorted)
      let best: { gain: number; cut: number; at: number } | undefined
      for (let p = 1; p < sorted.length; p++) {
        const prev = sorted[p - 1] ?? 0
        left[codes[prev] ?? 0] = (left[codes[prev] ?? 0] ?? 0) + (w[prev] ?? 0)
        right[codes[prev] ?? 0] = (right[codes[prev] ?? 0] ?? 0) - (w[prev] ?? 0)
        const a = m.data[prev * m.d + j] ?? 0
        const b = m.data[(sorted[p] ?? 0) * m.d + j] ?? 0
        if (b <= a + 1e-7 || p < options.minSamplesLeaf || sorted.length - p < options.minSamplesLeaf) continue
        const wl = left.reduce((x, y) => x + y, 0)
        const wr = right.reduce((x, y) => x + y, 0)
        const gain = node.impurity - (wl * entropyBits(left) + wr * entropyBits(right)) / (wl + wr)
        if (best === undefined || gain > best.gain) best = { gain, cut: a / 2 + b / 2, at: p }
      }
      if (best === undefined) return undefined
      const l = sorted.slice(0, best.at)
      const r = sorted.slice(best.at)
      const wl = countsOf(l).reduce((x, y) => x + y, 0)
      const info = entropyBits([wl, total - wl])
      return {
        feature: j,
        threshold: best.cut,
        branches: [[], []],
        groups: [l, r],
        gain: best.gain,
        ratio: info > 1e-12 ? best.gain / info : 0,
        pValue: 1,
        score: best.gain,
      }
    }
    const byCode = new Map<number, number[]>()
    for (const r of rows) {
      const c = code(r, j)
      const list = byCode.get(c)
      if (list === undefined) byCode.set(c, [r])
      else list.push(r)
    }
    let groups: { codes: number[]; rows: number[] }[] = [...byCode.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([c, r]) => ({ codes: [c], rows: r }))
    if (groups.length < 2) return undefined
    let pValue = 1
    if (options.mode === 'chaid') {
      const original = groups.length
      for (;;) {
        if (groups.length <= 2) break
        let worst = { p: -1, a: 0, b: 1 }
        for (let a = 0; a < groups.length; a++)
          for (let b = a + 1; b < groups.length; b++) {
            const p = pValueOf([countsOf(groups[a]?.rows ?? []), countsOf(groups[b]?.rows ?? [])])
            if (p > worst.p) worst = { p, a, b }
          }
        if (worst.p <= options.alphaMerge) break
        const merged = {
          codes: [...(groups[worst.a]?.codes ?? []), ...(groups[worst.b]?.codes ?? [])].sort((x, y) => x - y),
          rows: [...(groups[worst.a]?.rows ?? []), ...(groups[worst.b]?.rows ?? [])],
        }
        groups = groups.filter((_, i) => i !== worst.a && i !== worst.b)
        groups.push(merged)
        groups.sort((x, y) => (x.codes[0] ?? 0) - (y.codes[0] ?? 0))
      }
      pValue = Math.min(1, pValueOf(groups.map((g) => countsOf(g.rows))) * bonferroni(original, groups.length))
    }
    if (groups.length < 2 || groups.some((g) => g.rows.length < options.minSamplesLeaf)) {
      if (options.mode !== 'c45') return undefined
      if (groups.filter((g) => g.rows.length >= options.minSamplesLeaf).length < 2) return undefined
    }
    const counts = groups.map((g) => countsOf(g.rows))
    const sizes = counts.map((c) => c.reduce((a, b) => a + b, 0))
    const sw = sizes.reduce((a, b) => a + b, 0)
    const gain = node.impurity - counts.reduce((s, c, i) => s + ((sizes[i] ?? 0) / sw) * entropyBits(c), 0)
    const info = entropyBits(sizes)
    return {
      feature: j,
      threshold: -2,
      branches: groups.map((g) => g.codes),
      groups: groups.map((g) => g.rows),
      gain,
      ratio: info > 1e-12 ? gain / info : 0,
      pValue,
      score: options.mode === 'chaid' ? -pValue : options.mode === 'c45' ? gain : gain,
    }
  }
  return {
    name: `classicTree-${options.mode}`,
    config: { mode: options.mode },
    init: () => ({
      nodes: [
        born(
          0,
          -1,
          0,
          Int32Array.from({ length: m.n }, (_, i) => i),
          [],
        ),
      ],
      open: [0],
    }),
    step: (s, ctx) => {
      const nodes = [...s.nodes]
      const open = [...s.open]
      const id = open.pop() ?? 0
      const node = nodes[id] as ClassicNode
      let chosen: SplitOption | undefined
      const candidates: SplitCandidate[] = []
      const stop = node.depth >= maxDepth || node.rows.length < options.minSamplesSplit || node.impurity <= 1e-12
      if (!stop) {
        const all: SplitOption[] = []
        for (let j = 0; j < m.d; j++) {
          if (options.mode !== 'c45' && node.used.includes(j)) continue
          const o = optionsFor(node, j)
          if (o !== undefined) all.push(o)
        }
        for (const o of all) {
          const wTotal = o.groups.reduce((a, g) => a + countsOf(g).reduce((x, y) => x + y, 0), 0)
          const weighted = o.groups.reduce(
            (a, g) => a + (countsOf(g).reduce((x, y) => x + y, 0) / wTotal) * entropyBits(countsOf(g)),
            0,
          )
          candidates.push({
            feature: o.feature,
            threshold: o.threshold,
            impurityLeft: weighted,
            impurityRight: weighted,
            decrease: o.gain,
            score: options.mode === 'chaid' ? o.pValue : o.ratio,
          })
        }
        if (options.mode === 'id3') chosen = best(all, (o) => o.gain)
        else if (options.mode === 'chaid') {
          const significant = all.filter((o) => o.pValue < options.alphaSplit)
          chosen = best(significant, (o) => -o.pValue)
        } else {
          const average = all.length === 0 ? 0 : all.reduce((a, o) => a + o.gain, 0) / all.length
          chosen = best(
            all.filter((o) => o.gain >= average - 1e-12 && o.gain > 1e-12),
            (o) => o.ratio,
          )
        }
        if (chosen !== undefined && !(chosen.gain > 1e-12 || options.mode === 'chaid')) chosen = undefined
      }
      let decided: ClassicNode = { ...node, candidates, order: s.nodes.filter((n) => n.order >= 0).length }
      if (chosen !== undefined) {
        const kids = chosen.groups.map((rows, i) =>
          born(nodes.length + i, id, node.depth + 1, Int32Array.from(rows), [...node.used, chosen.feature]),
        )
        decided = {
          ...decided,
          feature: chosen.feature,
          threshold: chosen.threshold,
          children: kids.map((c) => c.id),
          branches: chosen.branches,
        }
        nodes.push(...kids)
        open.push(...kids.map((c) => c.id).reverse())
      }
      nodes[id] = decided
      if (ctx.wants('tree/split'))
        ctx.log('tree/split', {
          node: id,
          depth: node.depth,
          samples: node.rows.length,
          impurity: node.impurity,
          ...(chosen === undefined
            ? { leaf: true }
            : {
                leaf: false,
                feature: chosen.feature,
                branches: chosen.groups.length,
                gain: chosen.gain,
                pValue: chosen.pValue,
              }),
        })
      if (ctx.wants('tree/merge') && chosen !== undefined && options.mode === 'chaid')
        ctx.log('tree/merge', {
          node: id,
          feature: chosen.feature,
          groups: chosen.branches as unknown as number[][],
          pValue: chosen.pValue,
        })
      if (ctx.wants('tree/leaves')) ctx.log('tree/leaves', nodes.filter((n) => n.children.length === 0).length)
      return { nodes, open }
    },
    done: (s) => (s.open.length === 0 ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ClassicState,
  }
}

function best<T>(items: readonly T[], score: (t: T) => number): T | undefined {
  let out: T | undefined
  let top = Number.NEGATIVE_INFINITY
  for (const t of items) {
    const v = score(t)
    if (v > top) {
      top = v
      out = t
    }
  }
  return out
}

function cutsOf(
  m: { readonly n: number; readonly d: number; readonly data: Float64Array },
  options: ClassicOptions,
  categorical: ReadonlySet<number>,
): (number[] | null)[] {
  return Array.from({ length: m.d }, (_, j) => {
    if (options.mode === 'c45' || options.categorical.includes(j)) return null
    if (!categorical.has(j)) return null
    const col = Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0).filter((v) => !Number.isNaN(v))
    return bandsOf(col, options.bins)
  })
}

/**
 * The tree of a state, partial or complete.
 *
 * @param state - A state of {@link classicTreeAlgorithm}.
 * @param data - The data the algorithm ran on (for the classes and bin edges).
 * @param options - The options it ran with.
 * @returns The tree; pessimistic pruning is applied for C4.5 when `options.prune`.
 * @example
 * ```ts
 * import { classicTreeAlgorithm, classicTreeOf } from 'omai-engine/trees'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const options = { mode: 'c45' as const, categorical: [], minSamplesSplit: 2, minSamplesLeaf: 1, bins: 4, alphaMerge: 0.05, alphaSplit: 0.05, confidence: 0.25, prune: false }
 * const data = { x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] }
 * const run = runAlgorithm(classicTreeAlgorithm(data, options), { maxSteps: 20 })
 * classicTreeOf(run.state, data, options).children[0] // => [1, 2]
 * ```
 */
export function classicTreeOf(state: ClassicState, data: LearnData, options: ClassicOptions): ClassicTree {
  const m = matrixView(data.x, 'classicTree', { allowNaN: true })
  const { classes } = encodeLabels(data.y ?? [], 'classicTree', m.n)
  const categorical = new Set(options.mode === 'c45' ? options.categorical : Array.from({ length: m.d }, (_, j) => j))
  const tree: ClassicTree = {
    mode: options.mode,
    classes: [...classes],
    nFeatures: m.d,
    cuts: cutsOf(m, options, categorical),
    feature: state.nodes.map((n) => n.feature),
    threshold: state.nodes.map((n) => n.threshold),
    children: state.nodes.map((n) => [...n.children]),
    branches: state.nodes.map((n) => n.branches.map((b) => [...b])),
    value: state.nodes.map((n) => [...n.value]),
    impurity: state.nodes.map((n) => n.impurity),
    nSamples: state.nodes.map((n) => n.rows.length),
    depth: state.nodes.map((n) => n.depth),
  }
  return options.mode === 'c45' && options.prune ? pessimisticPrune(tree, options.confidence) : tree
}

/**
 * Quinlan's extra errors for a leaf of n cases and e observed errors at confidence cf (C4.5's `AddErrs`): the upper
 * confidence limit of the error count minus the count.
 *
 * @param n - Cases at the leaf.
 * @param e - Cases misclassified.
 * @param cf - The confidence factor (default 0.25).
 * @returns The extra errors to add to e.
 * @example
 * ```ts
 * import { pessimisticErrors } from 'omai-engine/trees'
 * pessimisticErrors(6, 0, 0.25).toFixed(4) // => '1.2378'
 * ```
 */
export function pessimisticErrors(n: number, e: number, cf = 0.25): number {
  if (n <= 0) return 0
  // The normal deviate for the upper confidence limit (0.6745 for 25%).
  const coeff = normalDeviate(cf)
  if (e < 1e-6) return n * (1 - Math.exp(Math.log(cf) / n))
  if (e < 0.9999) {
    const v0 = n * (1 - Math.exp(Math.log(cf) / n))
    return v0 + e * (pessimisticErrors(n, 1, cf) - v0)
  }
  if (e + 0.5 >= n) return 0.67 * (n - e)
  const pr =
    (e + 0.5 + (coeff * coeff) / 2 + Math.sqrt(coeff * ((e + 0.5) * (1 - (e + 0.5) / n) + (coeff * coeff) / 4))) /
    (n + coeff * coeff)
  return n * pr - e
}

function normalDeviate(cf: number): number {
  // The upper-tail quantile by bisection on the normal survival function (Abramowitz-Stegun 7.1.26 for erf).
  const erf = (x: number): number => {
    const t = 1 / (1 + 0.3275911 * Math.abs(x))
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
    return x >= 0 ? y : -y
  }
  let lo = 0
  let hi = 8
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    if (0.5 * (1 - erf(mid / Math.SQRT2)) > cf) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/**
 * C4.5's pessimistic error pruning: a subtree becomes a leaf when the upper confidence bound of the leaf's training
 * errors does not exceed the sum over the subtree's leaves (plus C4.5's small tolerance).
 *
 * @param tree - The tree.
 * @param confidence - The confidence factor (C4.5's default 0.25; smaller prunes more).
 * @returns The pruned tree, renumbered.
 * @example
 * ```ts
 * import { pessimisticPrune } from 'omai-engine/trees'
 * const leaf = (v: number[]) => ({ v, n: v[0] + v[1] })
 * const tree = { mode: 'c45' as const, classes: [0, 1], nFeatures: 1, cuts: [null], feature: [0, -1, -1], threshold: [0.5, -2, -2], children: [[1, 2], [], []], branches: [[[], []], [], []], value: [[5, 5], [3, 2], [2, 3]], impurity: [1, 0.97, 0.97], nSamples: [10, 5, 5], depth: [0, 1, 1] }
 * pessimisticPrune(tree, 0.25).children // => [[]]
 * ```
 */
export function pessimisticPrune(tree: ClassicTree, confidence: number): ClassicTree {
  const n = tree.feature.length
  const isLeafNow = new Array<boolean>(n).fill(false)
  const errors = (node: number): number => {
    const v = tree.value[node] ?? []
    const total = v.reduce((a, b) => a + b, 0)
    const e = total - maxOf(v)
    return e + pessimisticErrors(total, e, confidence)
  }
  const estimate = (node: number): number => {
    const kids = tree.children[node] ?? []
    if (kids.length === 0) return errors(node)
    const below = kids.reduce((s, c) => s + estimate(c), 0)
    const asLeaf = errors(node)
    if (asLeaf <= below + 0.1) {
      isLeafNow[node] = true
      return asLeaf
    }
    return below
  }
  estimate(0)
  const keep: number[] = []
  const walk = (node: number): void => {
    keep.push(node)
    if (!isLeafNow[node]) for (const c of tree.children[node] ?? []) walk(c)
  }
  walk(0)
  const index = new Map(keep.map((old, i) => [old, i]))
  const leafNow = (o: number): boolean => isLeafNow[o] === true || (tree.children[o] ?? []).length === 0
  return {
    ...tree,
    feature: keep.map((o) => (leafNow(o) ? -1 : (tree.feature[o] ?? -1))),
    threshold: keep.map((o) => (leafNow(o) ? -2 : (tree.threshold[o] ?? -2))),
    children: keep.map((o) => (leafNow(o) ? [] : (tree.children[o] ?? []).map((c) => index.get(c) ?? -1))),
    branches: keep.map((o) => (leafNow(o) ? [] : (tree.branches[o] ?? []).map((b) => [...b]))),
    value: keep.map((o) => tree.value[o] ?? []),
    impurity: keep.map((o) => tree.impurity[o] ?? 0),
    nSamples: keep.map((o) => tree.nSamples[o] ?? 0),
    depth: keep.map((o) => tree.depth[o] ?? 0),
  }
}

/**
 * The leaf each row of `x` reaches in a classic tree; a row whose category was not seen at a node (or whose value is
 * missing) stops at that node.
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns The node ids.
 * @example
 * ```ts
 * import { applyClassic } from 'omai-engine/trees'
 * const tree = { mode: 'id3' as const, classes: [0, 1], nFeatures: 1, cuts: [null], feature: [0, -1, -1], threshold: [-2, -2, -2], children: [[1, 2], [], []], branches: [[[0], [1]], [], []], value: [[1, 1], [1, 0], [0, 1]], impurity: [1, 0, 0], nSamples: [2, 1, 1], depth: [0, 1, 1] }
 * Array.from(applyClassic(tree, [[0], [1], [7]])) // => [1, 2, 0]
 * ```
 */
export function applyClassic(tree: ClassicTree, x: TensorLike): Int32Array {
  const m = matrixView(x, 'applyClassic', { nFeatures: tree.nFeatures, allowNaN: true })
  const out = new Int32Array(m.n)
  for (let i = 0; i < m.n; i++) {
    let node = 0
    for (;;) {
      const kids = tree.children[node] ?? []
      if (kids.length === 0) break
      const j = tree.feature[node] ?? 0
      const v = m.data[i * m.d + j] ?? 0
      if (Number.isNaN(v)) break
      let next = -1
      if ((tree.threshold[node] ?? -2) !== -2 || (tree.branches[node]?.[0]?.length ?? 0) === 0) {
        next = v <= (tree.threshold[node] ?? 0) ? (kids[0] ?? -1) : (kids[1] ?? -1)
      } else {
        const c = codeOf(v, tree.cuts[j] ?? null)
        const at = (tree.branches[node] ?? []).findIndex((b) => b.includes(c))
        next = at < 0 ? -1 : (kids[at] ?? -1)
      }
      if (next < 0) break
      node = next
    }
    out[i] = node
  }
  return out
}

/**
 * Class probabilities `[m, k]` of a classic tree (the class proportions at the node each row stops at).
 *
 * @param tree - The tree.
 * @param x - Features `[m, d]`.
 * @returns The probabilities.
 * @example
 * ```ts
 * import { classicProba } from 'omai-engine/trees'
 * const tree = { mode: 'id3' as const, classes: [0, 1], nFeatures: 1, cuts: [null], feature: [0, -1, -1], threshold: [-2, -2, -2], children: [[1, 2], [], []], branches: [[[0], [1]], [], []], value: [[1, 1], [3, 1], [0, 1]], impurity: [1, 0, 0], nSamples: [2, 1, 1], depth: [0, 1, 1] }
 * Array.from(classicProba(tree, [[0]]).data) // => [0.75, 0.25]
 * ```
 */
export function classicProba(tree: ClassicTree, x: TensorLike): Tensor {
  const leaves = applyClassic(tree, x)
  const k = tree.classes.length
  const out = new Float64Array(leaves.length * k)
  leaves.forEach((l, i) => {
    const v = tree.value[l] ?? []
    const total = v.reduce((a, b) => a + b, 0) || 1
    for (let c = 0; c < k; c++) out[i * k + c] = (v[c] ?? 0) / total
  })
  return fromStorage(out, [leaves.length, k], 'float64')
}

/**
 * The record of a classic tree's state: every node in birth order, with its rows, class counts, entropy, the candidate
 * splits it considered (a multiway candidate lists the weighted child entropy as both impurities and the gain ratio, or
 * CHAID's p-value, as its score) and its branches.
 *
 * @param state - A state of {@link classicTreeAlgorithm}.
 * @param names - The criterion label, feature names and class names.
 * @returns The record.
 * @example
 * ```ts
 * import { classicRecordOf } from 'omai-engine/trees'
 * const node = { id: 0, parent: -1, depth: 0, rows: Int32Array.of(0), value: [1], impurity: 0, used: [], candidates: [], feature: -1, threshold: -2, children: [], branches: [], order: -1 }
 * classicRecordOf({ nodes: [node], open: [] }, { criterion: 'entropy', features: ['x'] }).nodes.length // => 1
 * ```
 */
export function classicRecordOf(
  state: ClassicState,
  names: { readonly criterion: string; readonly features: readonly string[]; readonly classes?: readonly string[] },
): TreeRecord {
  const nodes: TreeNode[] = state.nodes.map((n) => ({
    id: n.id,
    parent: n.parent < 0 ? null : n.parent,
    depth: n.depth,
    rows: Array.from(n.rows),
    value: [...n.value],
    impurity: n.impurity,
    candidates: n.candidates,
    order: n.order < 0 ? state.nodes.length + n.id : n.order,
    ...(n.children.length === 0
      ? {}
      : {
          split: {
            feature: n.feature,
            threshold: n.threshold,
            left: n.children[0] ?? -1,
            right: n.children[1] ?? n.children[0] ?? -1,
            ...(n.threshold === -2 || n.branches[0]?.length
              ? { branches: n.children.map((child, i) => ({ child, categories: [...(n.branches[i] ?? [])] })) }
              : {}),
          },
        }),
  }))
  return treeRecord({
    criterion: names.criterion,
    features: [...names.features],
    ...(names.classes === undefined ? {} : { classes: [...names.classes] }),
    nodes,
  })
}

/** The hyperparameters of the classic tree estimators. */
export type ClassicHyper = ClassicOptions

/** Verbs of a fitted classic tree. */
export interface ClassicVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly apply: (x: TensorLike) => Int32Array
  readonly nFeaturesIn: number
}

/** The paths {@link classicTreeAlgorithm} logs: the node just decided, the leaves so far, and CHAID's merged categories. */
export type ClassicGrowPath = 'tree/split' | 'tree/leaves' | 'tree/merge'

/** The paths a classic tree estimator adds to the base's: the grower's and the tree so far. */
export type ClassicPath = ClassicGrowPath | 'model/structure'

/** A classic tree estimator on the iterative base. */
export type ClassicEstimator = IterativeEstimator<ClassicHyper, ClassicState, ClassicTree, ClassicVerbs, ClassicPath>

/**
 * A classic decision tree classifier (ID3, C4.5 or CHAID) on the iterative base: a session of the growth, one node per
 * step, with the paths `tree/split`, `tree/leaves` and `model/structure` (a tree record whose multiway splits list their
 * branches).
 *
 * @param mode - The grower.
 * @param hyper - Overrides of the options (`categorical` lists the category-coded features).
 * @returns The estimator.
 * @concept decision-trees
 * @example
 * ```ts
 * import { classicTreeClassifier } from 'omai-engine/trees'
 * import { toFlat } from 'omai-engine/tensor'
 * const data = { x: [[0], [0], [1], [1], [2], [2]], y: [0, 0, 1, 1, 0, 0] }
 * const fitted = classicTreeClassifier('id3', { categorical: [0] }).fit(data)
 * toFlat(fitted.predict([[0], [1], [2]])) // => [0, 1, 0]
 * ```
 */
export function classicTreeClassifier(mode: ClassicMode, hyper: Partial<ClassicHyper> = {}): ClassicEstimator {
  const defaults: ClassicHyper = {
    mode,
    categorical: hyper.categorical ?? [],
    ...(hyper.maxDepth === undefined ? {} : { maxDepth: hyper.maxDepth }),
    minSamplesSplit: hyper.minSamplesSplit ?? 2,
    minSamplesLeaf: hyper.minSamplesLeaf ?? (mode === 'c45' ? 2 : 1),
    bins: hyper.bins ?? 5,
    alphaMerge: hyper.alphaMerge ?? 0.05,
    alphaSplit: hyper.alphaSplit ?? 0.05,
    confidence: hyper.confidence ?? 0.25,
    prune: hyper.prune ?? mode === 'c45',
  }
  return iterative<ClassicHyper, ClassicState, ClassicTree, ClassicVerbs, ClassicPath>(
    {
      name: `${mode}-classifier`,
      kind: 'classifier',
      defaults,
      hyperSpace: (h) =>
        dict({
          minSamplesLeaf: int(1, Math.max(50, h.minSamplesLeaf), {
            default: h.minSamplesLeaf,
            label: 'minimum branch size',
          }),
          ...(mode === 'c45'
            ? {
                confidence: real(0.01, 0.5, { default: h.confidence, label: 'pruning confidence' }),
                prune: choice([true, false], { default: h.prune }),
              }
            : {}),
          ...(mode === 'chaid'
            ? { alphaSplit: real(0.001, 0.2, { default: h.alphaSplit, label: 'split significance' }) }
            : {}),
        }),
      algorithm: (data, h) => classicTreeAlgorithm(data, h),
      params: (state, data, h) => classicTreeOf(state, data, h),
      verbs: (tree) => {
        const importances = new Array<number>(tree.nFeatures).fill(0)
        for (let i = 0; i < tree.feature.length; i++) {
          const kids = tree.children[i] ?? []
          if (kids.length === 0) continue
          const n = tree.nSamples[i] ?? 0
          const below = kids.reduce((s, c) => s + (tree.nSamples[c] ?? 0) * (tree.impurity[c] ?? 0), 0)
          importances[tree.feature[i] ?? 0] =
            (importances[tree.feature[i] ?? 0] ?? 0) + n * (tree.impurity[i] ?? 0) - below
        }
        const total = importances.reduce((a, b) => a + b, 0)
        return withDefaultScore('classifier', {
          predict: (x: TensorLike) => {
            const leaves = applyClassic(tree, x)
            return fromStorage(
              Float64Array.from(leaves, (l) => tree.classes[argmaxOf(tree.value[l] ?? [])] ?? 0),
              [leaves.length],
              'float64',
            )
          },
          predictProba: (x: TensorLike) => classicProba(tree, x),
          apply: (x: TensorLike) => applyClassic(tree, x),
          classes: tree.classes,
          nFeaturesIn: tree.nFeatures,
          featureImportances: total > 0 ? importances.map((v) => v / total) : importances,
        })
      },
      maxSteps: (_h, data) => 4 * rowsOf(data.x) + 10,
      contract: (data): Contract<ClassicState, ClassicPath> => {
        const names = featureNamesOf(data, matrixView(data.x, 'classicTree', { allowNaN: true }).d)
        const classes = classNamesOf(data, encodeLabels(data.y ?? [], 'classicTree').classes)
        // `tree/merge` is CHAID's only, so it is in the contract of that mode alone.
        return {
          'tree/split': message({ description: 'the node just decided: its feature, how many branches, the gain' }),
          'tree/leaves': scalar({ description: 'leaves so far' }),
          ...(mode === 'chaid'
            ? {
                'tree/merge': message({
                  description: 'the category groups CHAID formed at the node and the adjusted p-value',
                }),
              }
            : {}),
          'model/structure': derived<ClassicState>({
            description: 'the tree grown so far with its multiway branches',
            compute: (state) =>
              classicRecordOf(state, {
                criterion: mode === 'chaid' ? 'chi-square' : mode === 'c45' ? 'gain ratio' : 'information gain',
                features: names,
                classes,
              }),
          }),
        } as Contract<ClassicState, ClassicPath>
      },
      records: (state, data) => {
        const names = featureNamesOf(data, matrixView(data.x, 'classicTree', { allowNaN: true }).d)
        const classes = classNamesOf(data, encodeLabels(data.y ?? [], 'classicTree').classes)
        return [classicRecordOf(state, { criterion: mode, features: names, classes })]
      },
      logs: { 'tree/split': Log(), 'tree/leaves': Log() },
    },
    defaults,
  )
}
