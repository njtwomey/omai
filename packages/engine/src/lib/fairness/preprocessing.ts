/**
 * Fairness before training, as in AIF360: reweighing, which weights each row so the label is independent of the group
 * in the weighted data; disparate-impact repair, which moves each feature towards a distribution shared by all groups
 * while keeping ranks within each group; massaging and preferential sampling (Kamiran and Calders), which relabel or
 * resample the rows nearest a ranker's boundary; optimised pre-processing of the labels (after Calmon et al.), a
 * linear programme; and learning fair representations (Zemel et al.), prototypes the groups use alike, whose smooth loss
 * is minimised by L-BFGS or by any other solver of the slot (`fairRepresentationAlgorithm`).
 */

import { type Json, MdmlError, maxOf } from '../base/index.ts'
import {
  infNorm,
  linprog,
  type SlotState,
  type SmoothProblem,
  slotAlgorithm,
  smoothSolverConfig,
} from '../optim/index.ts'
import { type Key, key as makeKey, permutation } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  abs,
  add,
  clip,
  fromStorage,
  log,
  matmul,
  mean,
  mul,
  neg,
  reshape,
  sigmoid,
  sliceAxis,
  softmax,
  square,
  sub,
  sum,
  type Tensor,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'

/** One cell of {@link reweighingWeights}: a group and a label, how many rows it has and expects, and its weight. */
export interface ReweighingCell {
  readonly group: number
  readonly label: number
  readonly count: number
  /** The count were group and label independent: n P(group) P(label). */
  readonly expected: number
  /** expected / count. */
  readonly weight: number
}

/**
 * Reweighing (Kamiran and Calders, 2012; AIF360's `Reweighing`): weight each row by `P(group) P(label) / P(group,
 * label)`, so that in the weighted data the label is independent of the group.
 *
 * @param labels - Each row's binary label.
 * @param groups - Each row's group index.
 * @returns A weight per row, and the group-and-label cells (group-major, label 0 then 1).
 * @throws MdmlError `shape` when the two differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { reweighingWeights } from 'mdml-engine/fairness'
 * reweighingWeights([1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]).weights // => [0.75, 0.75, 1.5, 0.75, 0.75, 1.5]
 * ```
 */
export function reweighingWeights(
  labels: readonly number[],
  groups: readonly number[],
): { readonly weights: number[]; readonly cells: ReweighingCell[] } {
  if (labels.length !== groups.length) throw new MdmlError('shape', 'reweighingWeights: one group per label')
  const n = labels.length
  const k = Math.max(-1, maxOf(groups)) + 1
  const joint = Array.from({ length: k }, () => [0, 0])
  for (const [i, g] of groups.entries()) {
    const row = joint[g] as number[]
    const y = labels[i] === 1 ? 1 : 0
    row[y] = (row[y] ?? 0) + 1
  }
  const byGroup = joint.map((r) => (r[0] ?? 0) + (r[1] ?? 0))
  const byLabel = [0, 1].map((y) => joint.reduce((s, r) => s + (r[y] ?? 0), 0))
  const cells: ReweighingCell[] = joint.flatMap((r, g) =>
    [0, 1].map((y) => {
      const count = r[y] ?? 0
      const expected = ((byGroup[g] ?? 0) * (byLabel[y] ?? 0)) / n
      return { group: g, label: y, count, expected, weight: count === 0 ? 0 : expected / count }
    }),
  )
  return { weights: groups.map((g, i) => cells[2 * g + (labels[i] === 1 ? 1 : 0)]?.weight ?? 1), cells }
}

/** The empirical quantile (type 7, numpy's default) of sorted values. */
function quantileSorted(sorted: readonly number[], q: number): number {
  const h = (sorted.length - 1) * q
  const lo = Math.floor(h)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return (sorted[lo] ?? 0) + (h - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
}

/** The position of v in sorted values: the number below it plus half of the others equal to it. */
function midRank(sorted: readonly number[], v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((sorted[mid] ?? 0) < v) lo = mid + 1
    else hi = mid
  }
  let end = lo
  while (end < sorted.length && sorted[end] === v) end++
  return lo + (end - lo - 1) / 2
}

/**
 * Disparate-impact repair (Feldman et al., 2015; AIF360's `DisparateImpactRemover`): move each feature column towards
 * a distribution shared by every group while keeping each value's rank within its group: a value at quantile u of its
 * group goes towards the median, across groups, of the groups' quantiles at u, by `level`.
 *
 * @param x - Row-major features `[n, d]`.
 * @param d - The number of columns.
 * @param groups - Each row's group index.
 * @param options - The repair level in [0, 1] (default 1) and columns to leave as they are (the protected one).
 * @returns The repaired features, row-major.
 * @throws MdmlError `value` for a level outside [0, 1]; `shape` when the groups do not match the rows.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { repairFeatures } from 'mdml-engine/fairness'
 * Array.from(repairFeatures([0, 1, 10, 11], 1, [0, 0, 1, 1])) // => [5, 6, 5, 6]
 * ```
 */
export function repairFeatures(
  x: ArrayLike<number>,
  d: number,
  groups: readonly number[],
  options: { readonly level?: number; readonly skip?: readonly number[] } = {},
): Float64Array {
  const level = options.level ?? 1
  if (!(level >= 0 && level <= 1)) throw new MdmlError('value', 'repairFeatures: the level is in [0, 1]')
  const n = groups.length
  if (x.length !== n * d) throw new MdmlError('shape', `repairFeatures: ${x.length} values for ${n} rows of ${d}`)
  const k = Math.max(-1, maxOf(groups)) + 1
  const members = Array.from({ length: k }, (): number[] => [])
  for (const [i, g] of groups.entries()) members[g]?.push(i)
  const skip = new Set(options.skip ?? [])
  const out = Float64Array.from(x)
  for (let j = 0; j < d; j++) {
    if (skip.has(j)) continue
    const sorted = members.map((rows) => rows.map((i) => x[i * d + j] ?? 0).sort((a, b) => a - b))
    const present = sorted.filter((s) => s.length > 0)
    for (const [g, rows] of members.entries()) {
      const own = sorted[g] ?? []
      for (const i of rows) {
        const v = x[i * d + j] ?? 0
        const u = own.length <= 1 ? 0.5 : midRank(own, v) / (own.length - 1)
        const target = quantileSorted(
          present.map((s) => quantileSorted(s, u)).sort((a, b) => a - b),
          0.5,
        )
        out[i * d + j] = (1 - level) * v + level * target
      }
    }
  }
  return out
}

/** The result of {@link massageLabels}. */
export interface Massaged {
  /** The massaged labels. */
  readonly labels: number[]
  /** Unprivileged negatives relabelled positive (the highest scored). */
  readonly promoted: number[]
  /** Privileged positives relabelled negative (the lowest scored). */
  readonly demoted: number[]
}

/**
 * Massaging (Kamiran and Calders, 2012): relabel the fewest rows that make the label rate equal across the privileged
 * and unprivileged groups, choosing those a ranker is least sure of: promote the unprivileged negatives with the highest
 * scores and demote the same number of privileged positives with the lowest. The number of swaps is
 * `round((P_p n_u - P_u n_p) / n)`, where P counts positives and n rows.
 *
 * @param labels - Binary labels `[N]`.
 * @param scores - A ranker's scores `[N]` (higher means more likely positive).
 * @param groups - Group indices `[N]`.
 * @param options - The privileged group's index (default 0); every other group is unprivileged.
 * @returns The new labels and the rows promoted and demoted.
 * @throws MdmlError `shape` when the inputs differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { massageLabels } from 'mdml-engine/fairness'
 * const r = massageLabels([1, 1, 1, 0, 1, 0, 0, 0], [0.9, 0.8, 0.6, 0.3, 0.7, 0.5, 0.4, 0.1], [0, 0, 0, 0, 1, 1, 1, 1])
 * r.promoted // => [5]
 * r.demoted // => [2]
 * ```
 */
export function massageLabels(
  labels: readonly number[],
  scores: readonly number[],
  groups: readonly number[],
  options: { readonly privileged?: number } = {},
): Massaged {
  if (labels.length !== scores.length || labels.length !== groups.length)
    throw new MdmlError('shape', 'massageLabels: inputs differ in length')
  const privileged = options.privileged ?? 0
  const n = labels.length
  const isPriv = groups.map((g) => g === privileged)
  let np = 0
  let pp = 0
  let pu = 0
  for (const [i, y] of labels.entries()) {
    if (isPriv[i] === true) {
      np++
      if (y === 1) pp++
    } else if (y === 1) pu++
  }
  const nu = n - np
  const m = Math.max(0, Math.round((pp * nu - pu * np) / n))
  const candidates = (priv: boolean, label: number): number[] =>
    labels.flatMap((y, i) => ((isPriv[i] === true) === priv && y === label ? [i] : []))
  const promoted = candidates(false, 0)
    .sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0))
    .slice(0, m)
  const demoted = candidates(true, 1)
    .sort((a, b) => (scores[a] ?? 0) - (scores[b] ?? 0))
    .slice(0, m)
  const out = labels.map((y) => (y === 1 ? 1 : 0))
  for (const i of promoted) out[i] = 1
  for (const i of demoted) out[i] = 0
  return { labels: out, promoted, demoted }
}

/** One cell of {@link preferentialSample}: a group and a label, its size before and after. */
export interface SampleCell {
  readonly group: number
  readonly label: number
  readonly count: number
  /** n P(group) P(label), rounded: the size were group and label independent. */
  readonly expected: number
}

/**
 * Preferential sampling (Kamiran and Calders, 2012): resample every group-and-label cell to the size it would have if
 * label and group were independent, removing or duplicating the rows nearest the decision boundary (by a ranker's
 * score): a cell of positives that is too large loses its lowest-scored rows and one too small duplicates them; a cell
 * of negatives the same with its highest-scored rows.
 *
 * @param labels - Binary labels `[N]`.
 * @param scores - A ranker's scores `[N]`.
 * @param groups - Group indices `[N]`.
 * @returns The row indices of the resample (with repeats), and each cell's size before and after.
 * @throws MdmlError `shape` when the inputs differ in length.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { preferentialSample } from 'mdml-engine/fairness'
 * const r = preferentialSample([1, 1, 1, 0, 1, 0, 0, 0], [0.9, 0.8, 0.6, 0.3, 0.7, 0.5, 0.4, 0.1], [0, 0, 0, 0, 1, 1, 1, 1])
 * r.rows // => [0, 1, 3, 3, 4, 4, 6, 7]
 * ```
 */
export function preferentialSample(
  labels: readonly number[],
  scores: readonly number[],
  groups: readonly number[],
): { readonly rows: number[]; readonly cells: SampleCell[] } {
  if (labels.length !== scores.length || labels.length !== groups.length)
    throw new MdmlError('shape', 'preferentialSample: inputs differ in length')
  const { cells: weights } = reweighingWeights(labels, groups)
  const rows: number[] = []
  const cells: SampleCell[] = []
  for (const c of weights) {
    const members = labels.flatMap((y, i) => (groups[i] === c.group && (y === 1 ? 1 : 0) === c.label ? [i] : []))
    // Nearest the boundary first: the lowest-scored positives, the highest-scored negatives.
    members.sort((a, b) => (c.label === 1 ? (scores[a] ?? 0) - (scores[b] ?? 0) : (scores[b] ?? 0) - (scores[a] ?? 0)))
    const expected = Math.round(c.expected)
    cells.push({ group: c.group, label: c.label, count: members.length, expected })
    if (members.length === 0) continue
    if (expected <= members.length) rows.push(...members.slice(members.length - expected))
    else {
      rows.push(...members)
      for (let k = 0; k < expected - members.length; k++) rows.push(members[k % members.length] ?? 0)
    }
  }
  return { rows: rows.sort((a, b) => a - b), cells }
}

/** Options of {@link optimisedLabelMapping}. */
export interface LabelMappingOptions {
  /** The largest gap allowed between any group's positive rate and a shared rate (default 0.02). */
  readonly epsilon?: number
  /** The largest probability with which any group's labels of one class may be flipped (default 1). */
  readonly maxFlip?: number
}

/** A randomised relabelling of each group's labels. */
export interface LabelMapping {
  /** positive[g][y]: the probability a row of group g with label y is relabelled positive. */
  readonly positive: readonly (readonly [number, number])[]
  /** Each group's positive rate after the mapping. */
  readonly rates: readonly number[]
  /** The shared rate the groups are held within epsilon of. */
  readonly target: number
  /** The expected share of labels changed (the distortion minimised). */
  readonly distortion: number
}

/**
 * Optimised pre-processing of the labels (Calmon et al., 2017, in the form that maps the label alone): the randomised
 * relabelling P(new label | group, label) that changes the fewest labels in expectation subject to every group's
 * positive rate lying within `epsilon` of a shared rate (statistical parity) and to a cap on how often any group's
 * labels of one class are flipped (the individual distortion constraint), solved as a linear program. Calmon et al.
 * also map the features through the same programme; this form leaves them as they are.
 *
 * @param labels - Binary labels `[N]`.
 * @param groups - Group indices `[N]`.
 * @param options - epsilon and the flip cap.
 * @returns The mapping, each group's new rate, the shared rate and the distortion.
 * @throws MdmlError `value` when the programme has no solution (a flip cap too tight for epsilon).
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { optimisedLabelMapping } from 'mdml-engine/fairness'
 * const m = optimisedLabelMapping([1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], { epsilon: 0 })
 * Math.abs((m.rates[0] ?? 0) - (m.rates[1] ?? 1)) < 1e-9 // => true
 * m.distortion.toFixed(3) // => '0.250'
 * ```
 */
export function optimisedLabelMapping(
  labels: readonly number[],
  groups: readonly number[],
  options: LabelMappingOptions = {},
): LabelMapping {
  if (labels.length !== groups.length) throw new MdmlError('shape', 'optimisedLabelMapping: one group per label')
  const n = labels.length
  const k = Math.max(-1, maxOf(groups)) + 1
  const eps = options.epsilon ?? 0.02
  const cap = options.maxFlip ?? 1
  // P(group, label).
  const joint = Array.from({ length: k }, () => [0, 0])
  for (const [i, g] of groups.entries()) {
    const row = joint[g] as number[]
    const y = labels[i] === 1 ? 1 : 0
    row[y] = (row[y] ?? 0) + 1 / n
  }
  const pg = joint.map((r) => (r[0] ?? 0) + (r[1] ?? 0))
  // Variables: p[g][0], p[g][1] for each group, then the shared rate t.
  const v = 2 * k + 1
  const c = new Array<number>(v).fill(0)
  for (let g = 0; g < k; g++) {
    c[2 * g] = joint[g]?.[0] ?? 0
    c[2 * g + 1] = -(joint[g]?.[1] ?? 0)
  }
  const aUb: number[][] = []
  const bUb: number[] = []
  for (let g = 0; g < k; g++) {
    if ((pg[g] ?? 0) === 0) continue
    const row = new Array<number>(v).fill(0)
    row[2 * g] = (joint[g]?.[0] ?? 0) / (pg[g] ?? 1)
    row[2 * g + 1] = (joint[g]?.[1] ?? 0) / (pg[g] ?? 1)
    aUb.push([...row.slice(0, v - 1), -1])
    bUb.push(eps)
    aUb.push([...row.slice(0, v - 1).map((x) => -x), 1])
    bUb.push(eps)
  }
  const bounds: [number, number][] = Array.from({ length: v }, (_, j) =>
    j === v - 1 ? [0, 1] : j % 2 === 0 ? [0, Math.min(1, cap)] : [Math.max(0, 1 - cap), 1],
  )
  const r = linprog({ c, aUb, bUb, bounds })
  if (!r.success) throw new MdmlError('value', `optimisedLabelMapping: no mapping meets the constraints (${r.message})`)
  const x = Array.from(toFlat(r.x))
  const positive = Array.from({ length: k }, (_, g): [number, number] => [x[2 * g] ?? 0, x[2 * g + 1] ?? 1])
  const rates = positive.map(
    ([p0, p1], g) => ((joint[g]?.[0] ?? 0) * p0 + (joint[g]?.[1] ?? 0) * p1) / Math.max(pg[g] ?? 0, 1e-300),
  )
  const distortion = positive.reduce(
    (s, [p0, p1], g) => s + (joint[g]?.[0] ?? 0) * p0 + (joint[g]?.[1] ?? 0) * (1 - p1),
    0,
  )
  return { positive, rates, target: x[v - 1] ?? 0, distortion }
}

/** Options of {@link learnFairRepresentation}. */
export interface FairRepresentationOptions {
  /** Prototypes (default 5). */
  readonly prototypes?: number
  /** Weights of the reconstruction, prediction and parity losses (default 0.01, 1, 50, as AIF360's Ax, Ay, Az). */
  readonly weights?: { readonly reconstruction?: number; readonly prediction?: number; readonly parity?: number }
  readonly maxIterations?: number
  readonly key?: Key
  /** How the loss is minimised: a value of `smoothSolvers.space` (default L-BFGS to a gradient tolerance of 1e-7). */
  readonly solver?: SpaceValue
}

/** A learned fair representation: prototypes in feature space, each with a label probability. */
export interface FairRepresentation {
  /** The prototypes `[K, d]`, in the features' units. */
  readonly prototypes: readonly (readonly number[])[]
  /** Each prototype's probability of the positive label. */
  readonly labels: readonly number[]
  /** The losses at the solution. */
  readonly losses: { readonly reconstruction: number; readonly prediction: number; readonly parity: number }
}

/**
 * The memberships of rows in a fair representation's prototypes: a softmax of minus each squared distance.
 *
 * @param rep - The representation.
 * @param x - Rows `[n, d]`.
 * @returns memberships[i][k].
 * @example
 * ```ts
 * import { fairRepresentationMemberships } from 'mdml-engine/fairness'
 * const rep = { prototypes: [[0], [10]], labels: [0, 1], losses: { reconstruction: 0, prediction: 0, parity: 0 } }
 * fairRepresentationMemberships(rep, [[0]])[0]?.[0] // => 1
 * ```
 */
export function fairRepresentationMemberships(rep: FairRepresentation, x: readonly (readonly number[])[]): number[][] {
  return x.map((row) => {
    const d = rep.prototypes.map((v) => -row.reduce((s, xj, j) => s + (xj - (v[j] ?? 0)) ** 2, 0))
    const top = maxOf(d)
    const e = d.map((v) => Math.exp(v - top))
    const z = e.reduce((s, v) => s + v, 0)
    return e.map((v) => v / z)
  })
}

/** A learned-fair-representation problem: the smooth objective over the prototypes and their label logits. */
export interface FairRepresentationProblem extends SmoothProblem {
  /** The number of prototypes K. */
  readonly prototypes: number
  /** The number of features d. */
  readonly features: number
  /** The three losses at a parameter vector (the prototypes row-major, then the K label logits). */
  losses(p: ArrayLike<number>): {
    readonly reconstruction: number
    readonly prediction: number
    readonly parity: number
  }
}

/**
 * The learning-fair-representations objective as a smooth problem for the solver slot: the weighted sum of the
 * reconstruction, prediction and parity losses over the prototypes (`K d` numbers) followed by their `K` label
 * logits, started at `K` rows drawn by the key.
 *
 * @param x - Features `[n, d]`.
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]` (0 the privileged group).
 * @param options - Prototypes, loss weights and key.
 * @returns The problem: the objective, the start and the losses of a parameter vector.
 * @throws MdmlError `shape` when labels or groups differ in length from the rows.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairRepresentationProblem } from 'mdml-engine/fairness'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * fairRepresentationProblem(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { prototypes: 2 }).prototypes // => 2
 * ```
 */
export function fairRepresentationProblem(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: FairRepresentationOptions = {},
): FairRepresentationProblem {
  const n = x.length
  const d = x[0]?.length ?? 0
  if (labels.length !== n || groups.length !== n)
    throw new MdmlError('shape', 'learnFairRepresentation: one label and group per row')
  const K = options.prototypes ?? 5
  const ax = options.weights?.reconstruction ?? 0.01
  const ay = options.weights?.prediction ?? 1
  const az = options.weights?.parity ?? 50
  const X = fromStorage(Float64Array.from(x.flat()), [n, d], 'float64')
  const y = fromStorage(
    Float64Array.from(labels, (v) => (v === 1 ? 1 : 0)),
    [n, 1],
    'float64',
  )
  const z = groups.map((g) => (g === 0 ? 0 : 1))
  const n1 = z.reduce((s: number, v) => s + v, 0)
  const n0 = n - n1
  // Each row's weight in its group's mean membership, signed: +1/n0 for the privileged group, -1/n1 for the rest.
  const signed = fromStorage(
    Float64Array.from(z, (v) => (v === 1 ? -1 / Math.max(1, n1) : 1 / Math.max(1, n0))),
    [1, n],
    'float64',
  )
  const sqX = fromStorage(
    Float64Array.from({ length: n }, (_, i) => (x[i] ?? []).reduce((s, v) => s + v * v, 0)),
    [n, 1],
    'float64',
  )
  const unpack = (p: Value) => {
    const V = reshape(sliceAxis(p, 0, 0, K * d), [K, d])
    const w = reshape(sliceAxis(p, 0, K * d, K * d + K), [K, 1])
    return { V, w }
  }
  const parts = (p: Value) => {
    const { V, w } = unpack(p)
    // Squared distances ||x||^2 - 2 x.v + ||v||^2, then memberships by softmax over prototypes.
    const sqV = reshape(sum(square(V), 1), [1, K])
    const dist = add(sub(sqX, mul(matmul(X, transpose(V)), 2)), sqV)
    const M = softmax(neg(dist), 1)
    const recon = mean(sum(square(sub(X, matmul(M, V))), 1))
    const prob = clip(matmul(M, sigmoid(w)), 1e-6, 1 - 1e-6)
    const pred = neg(mean(add(mul(y, log(prob)), mul(sub(1, y), log(sub(1, prob))))))
    const parity = sum(abs(matmul(signed, M)))
    return { recon, pred, parity }
  }
  const objective = (p: Value): Value => {
    const { recon, pred, parity } = parts(p)
    return add(add(mul(recon, ax), mul(pred, ay)), mul(parity, az))
  }
  // Start the prototypes at rows drawn by the key, the labels at 0.
  const k = options.key ?? makeKey(0)
  const order = Array.from(toFlat(permutation(k, n))).slice(0, K)
  const p0 = new Float64Array(K * d + K)
  order.forEach((row, j) => {
    for (let c = 0; c < d; c++) p0[j * d + c] = x[row]?.[c] ?? 0
  })
  const scalar = (v: Value): number => toFlat(v as Tensor)[0] ?? Number.NaN
  return {
    objective: (p) => objective(p as Value),
    x0: fromStorage(p0, [K * d + K], 'float64'),
    prototypes: K,
    features: d,
    losses: (p) => {
      const { recon, pred, parity } = parts(fromStorage(Float64Array.from(p), [K * d + K], 'float64'))
      return { reconstruction: scalar(recon), prediction: scalar(pred), parity: scalar(parity) }
    },
  }
}

/** The state of {@link fairRepresentationAlgorithm}: the parameters, the objective and the solver's state. */
export interface FairRepresentationState extends SlotState {
  /** The prototypes row-major, then the label logits. */
  readonly x: number[]
  readonly objective: number
  /** The largest gradient entry. */
  readonly gradient: number
  readonly evaluations: number
  readonly stalled: boolean
}

/** The solver of {@link learnFairRepresentation} unless another is given: L-BFGS to a gradient tolerance of 1e-7. */
export const fairRepresentationSolver: SpaceValue = smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-7 })

/**
 * Learning fair representations as a traced algorithm over the solver slot: any solver of `smoothSolvers` (L-BFGS by
 * default, as AIF360's) minimises the objective of {@link fairRepresentationProblem}. It logs `objective`,
 * `gradient` (the largest gradient entry), `gradNorm`, `step` and the tensor `parameters`.
 *
 * @param x - Features `[n, d]`.
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]` (0 the privileged group).
 * @param options - Prototypes, loss weights, key and `solver` (a value of `smoothSolvers.space`).
 * @returns The algorithm.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairRepresentationAlgorithm } from 'mdml-engine/fairness'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * const a = fairRepresentationAlgorithm(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { prototypes: 2 })
 * runAlgorithm(a, { maxSteps: 3 }).state.x.length // => 4
 * ```
 */
export function fairRepresentationAlgorithm(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: FairRepresentationOptions = {},
): Algorithm<FairRepresentationState> {
  const problem = fairRepresentationProblem(x, labels, groups, options)
  const solver = options.solver ?? fairRepresentationSolver
  const m = problem.prototypes * problem.features + problem.prototypes
  return slotAlgorithm<FairRepresentationState>({
    name: 'fair-representation',
    config: { prototypes: problem.prototypes, weights: options.weights ?? null, solver } as Json,
    schema: {
      metrics: {
        objective: {
          description: 'the weighted sum of the reconstruction, prediction and parity losses',
          direction: 'lower',
        },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { parameters: { role: 'param', description: 'prototypes row-major, then label logits', shape: [m] } },
    },
    problem,
    solver,
    ...(options.key === undefined ? {} : { key: options.key }),
    tol: 1e-7,
    view: (r, evaluations, stalled) => ({
      x: Array.from(r.x),
      objective: r.value,
      gradient: infNorm(r.grad),
      evaluations,
      stalled,
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { parameters: { role: 'param', value: s.x, shape: [m] } },
    }),
  })
}

/**
 * Learning fair representations (Zemel, Wu, Swersky, Pitassi and Dwork, 2013; AIF360's `LFR`): map each row to a
 * distribution over K prototypes (a softmax of minus its squared distances to them) so that the prototypes reconstruct
 * the features, predict the label (each prototype carries a label probability), and are used alike by both groups
 * (statistical parity of the memberships); the weighted sum of the three losses is minimised by L-BFGS, or by any
 * other solver of the slot (`options.solver`, see {@link fairRepresentationAlgorithm}). A classifier on the
 * memberships (or the prototypes' labels) is then fair by construction, as far as the parity weight demands.
 *
 * @param x - Features `[n, d]` (standardise them first; without the protected attribute).
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]` (0 the privileged group; others pooled).
 * @param options - Prototypes, loss weights, iteration cap, key and solver.
 * @returns The prototypes, their labels and the losses.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { learnFairRepresentation } from 'mdml-engine/fairness'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * learnFairRepresentation(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { prototypes: 2 }).prototypes.length // => 2
 * ```
 */
export function learnFairRepresentation(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: FairRepresentationOptions = {},
): FairRepresentation {
  const problem = fairRepresentationProblem(x, labels, groups, options)
  const { prototypes: K, features: d } = problem
  const r = runAlgorithm(fairRepresentationAlgorithm(x, labels, groups, options), {
    maxSteps: options.maxIterations ?? 300,
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const flat = r.state.x
  return {
    prototypes: Array.from({ length: K }, (_, j) => Array.from({ length: d }, (_, c) => flat[j * d + c] ?? 0)),
    labels: Array.from({ length: K }, (_, j) => 1 / (1 + Math.exp(-(flat[K * d + j] ?? 0)))),
    losses: problem.losses(flat),
  }
}
