/**
 * Multiclass reductions over any base estimator: one-versus-rest, one-versus-one, error-correcting output codes and
 * (ensembles of) nested dichotomies. Each reduces the problem to binary ones that the base fits, and each is an
 * estimator on the iterative base that solves one binary problem per step, logging `solver/problems`, so a session
 * shows the per-class boundaries arriving and `fittedAt(t)` is the classifier built from the problems solved so far.
 * The base is an ordinary `Estimator` (a classifier with a decision function or probabilities); wrapping decides
 * how its scores combine.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { type DictSpace, dict, real } from '../space/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { accuracyOf, type ScoreClassifierVerbs } from './_classify.ts'
import { confidence, hasDecision, matrix, predicted, type StagedState, stagedAlgorithm } from './_staged.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { Estimator, Fitted, LearnData } from './estimator.ts'
import { encodeLabels, matrixView, rowsOf, takeData } from './validation.ts'

/** The hyperparameters every reduction has: the base estimator that fits the binary problems. */
export interface MulticlassHyper {
  readonly estimator: Estimator
}

/** The fitted parameters of a reduction: the classes, the plan of its problems and the base's parameters for each solved. */
export type MulticlassParams<Plan extends Json> = {
  readonly classes: number[]
  readonly plan: Plan
  readonly models: Json[]
}

/** What a fitted reduction offers. */
export interface MulticlassVerbs extends Omit<ScoreClassifierVerbs, 'predictProba' | 'predictLogProba'> {
  /** The class probabilities, when the base gives them. */
  readonly predictProba?: (x: TensorLike) => Tensor
  /** The problems solved (and so the base's fitted values, in the plan's order). */
  readonly nProblems: number
}

type Problem = { readonly rows?: readonly number[]; readonly labels: ArrayLike<number> }

function stageOf(data: LearnData, p: Problem): LearnData {
  const base = p.rows === undefined ? data : takeData(data, p.rows)
  return { ...base, y: matrix(p.labels, [p.labels.length]) }
}

/** Shared pieces of the reductions: the classes and codes of the data, and the fitted bases of some parameters. */
function setup(data: LearnData, who: string): { readonly classes: readonly number[]; readonly codes: Int32Array } {
  const { classes, codes } = encodeLabels(data.y ?? [], who, rowsOf(data.x))
  if (classes.length < 2) throw new MdmlError('value', `${who}: needs at least two classes`)
  return { classes, codes }
}

const revived = (estimator: Estimator, models: readonly Json[]): Fitted[] => models.map((m) => estimator.revive(m))

function build<Plan extends Json, H extends MulticlassHyper>(
  name: string,
  hyper: H,
  spec: {
    readonly plan: (classes: readonly number[], codes: Int32Array, key: Key, h: H) => Plan
    readonly total: (plan: Plan) => number
    readonly problem: (plan: Plan, index: number, codes: Int32Array) => Problem
    readonly verbs: (p: MulticlassParams<Plan>, h: H) => MulticlassVerbs
  },
  space: (h: H) => DictSpace = () => dict({}),
): IterativeEstimator<H, StagedState<Plan>, MulticlassParams<Plan>, MulticlassVerbs> {
  return iterative<H, StagedState<Plan>, MulticlassParams<Plan>, MulticlassVerbs>(
    {
      name,
      kind: 'classifier',
      defaults: hyper,
      hyperSpace: space,
      algorithm: (data, h) => {
        const { classes, codes } = setup(data, name)
        return stagedAlgorithm<Plan>(h.estimator, {
          name,
          plan: (k) => spec.plan(classes, codes, k, h),
          total: spec.total,
          stage: (plan, index) => stageOf(data, spec.problem(plan, index, codes)),
        })
      },
      maxSteps: (_h, data) => {
        // The number of problems is known only from the plan, so this bound is generous: the algorithm ends itself.
        const k = setup(data, name).classes.length
        return Math.max(64, k * k * 4)
      },
      params: (s, data) => ({ classes: [...setup(data, name).classes], plan: s.plan, models: s.models }),
      verbs: spec.verbs,
      logs: { 'solver/problems': Log() },
    },
    hyper,
  )
}

function finish(
  p: MulticlassParams<Json>,
  d: number,
  predict: (x: TensorLike) => Tensor,
  decision: (x: TensorLike) => Tensor,
  proba?: (x: TensorLike) => Tensor,
): MulticlassVerbs {
  return {
    classes: p.classes,
    nFeaturesIn: d,
    predict,
    decisionFunction: decision,
    ...(proba === undefined ? {} : { predictProba: proba }),
    nProblems: p.models.length,
    score: accuracyOf(predict),
  }
}

const argmaxRow = (row: ArrayLike<number>): number => {
  let best = 0
  for (let c = 1; c < row.length; c++)
    if ((row[c] ?? Number.NEGATIVE_INFINITY) > (row[best] ?? Number.NEGATIVE_INFINITY)) best = c
  return best
}

/** One-versus-rest plan: the class index each problem separates from the others. */
export type OneVsRestPlan = { readonly positives: number[] }

/**
 * One-versus-rest (`sklearn.multiclass.OneVsRestClassifier`): one base classifier per class, fitted to separate it
 * from the rest, and the class whose classifier is most confident (its decision function, else its probability)
 * wins. With two classes one classifier suffices, as in scikit-learn. The probabilities are the base's, normalised
 * across classes. In a session, `fittedAt(t)` knows the classes of the first `t + 1` problems; the others score
 * minus infinity.
 *
 * @param estimator - The binary base classifier.
 * @returns The estimator.
 * @example
 * ```ts
 * import { oneVsRest, linearDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [5], [6], [10], [11]]
 * const f = oneVsRest(linearDiscriminant()).fit({ x, y: [0, 0, 1, 1, 2, 2] })
 * toFlat(f.predict([[0.5], [5.5], [10.5]])) // => [0, 1, 2]
 * ```
 */
export function oneVsRest(
  estimator: Estimator,
): IterativeEstimator<MulticlassHyper, StagedState<OneVsRestPlan>, MulticlassParams<OneVsRestPlan>, MulticlassVerbs> {
  return build<OneVsRestPlan, MulticlassHyper>(
    'one-vs-rest',
    { estimator },
    {
      plan: (classes) => ({ positives: classes.length === 2 ? [1] : classes.map((_, c) => c) }),
      total: (plan) => plan.positives.length,
      problem: (plan, i, codes) => ({
        labels: Float64Array.from(codes, (c) => (c === plan.positives[i] ? 1 : 0)),
      }),
      verbs: (p, h) => {
        const fits = revived(h.estimator, p.models)
        const k = p.classes.length
        const d = fits[0]?.nFeaturesIn ?? 0
        const positives = p.plan.positives
        const conf = (x: TensorLike): { readonly n: number; readonly c: Float64Array[] } => {
          const c = fits.map((f) => confidence(f, x))
          return { n: c[0]?.length ?? matrixView(x, 'oneVsRest').n, c }
        }
        const threshold = fits[0] !== undefined && hasDecision(fits[0]) ? 0 : 0.5
        const scores = (x: TensorLike): Float64Array => {
          const { n, c } = conf(x)
          const out = new Float64Array(n * k).fill(Number.NEGATIVE_INFINITY)
          c.forEach((col, i) => {
            for (let r = 0; r < n; r++) out[r * k + (positives[i] ?? 0)] = col[r] ?? 0
          })
          return out
        }
        const predict = (x: TensorLike): Tensor => {
          const { n, c } = conf(x)
          if (k === 2) {
            return matrix(
              Array.from({ length: n }, (_, r) => p.classes[(c[0]?.[r] ?? 0) > threshold ? 1 : 0] ?? 0),
              [n],
            )
          }
          const s = scores(x)
          return matrix(
            Array.from({ length: n }, (_, r) => p.classes[argmaxRow(s.subarray(r * k, (r + 1) * k))] ?? 0),
            [n],
          )
        }
        const decision = (x: TensorLike): Tensor => {
          const { n, c } = conf(x)
          return k === 2 ? matrix(c[0] ?? [], [n]) : matrix(scores(x), [n, k])
        }
        const canProba = fits.length > 0 && fits.every((f) => f.predictProba !== undefined)
        const proba = (x: TensorLike): Tensor => {
          const cols = fits.map((f) => {
            const pr = (f.predictProba as (x: TensorLike) => Tensor)(x)
            const kk = pr.shape[1] ?? 2
            const flat = toFlat(pr)
            return Float64Array.from({ length: flat.length / kk }, (_, r) => flat[r * kk + 1] ?? 0)
          })
          const n = cols[0]?.length ?? 0
          const out = new Float64Array(n * k)
          if (k === 2) {
            for (let r = 0; r < n; r++) {
              out[r * 2] = 1 - (cols[0]?.[r] ?? 0)
              out[r * 2 + 1] = cols[0]?.[r] ?? 0
            }
          } else {
            cols.forEach((col, i) => {
              for (let r = 0; r < n; r++) out[r * k + (positives[i] ?? 0)] = col[r] ?? 0
            })
            for (let r = 0; r < n; r++) {
              const sum = out.subarray(r * k, (r + 1) * k).reduce((a, b) => a + b, 0)
              if (sum !== 0) for (let c = 0; c < k; c++) out[r * k + c] = (out[r * k + c] ?? 0) / sum
            }
          }
          return matrix(out, [n, k])
        }
        return finish(p, d, predict, decision, canProba ? proba : undefined)
      },
    },
  )
}

/** One-versus-one plan: the pair of class indices each problem separates (the second is the positive). */
export type OneVsOnePlan = { readonly pairs: [number, number][] }

/**
 * One-versus-one (`sklearn.multiclass.OneVsOneClassifier`): one base classifier for every pair of classes, fitted on
 * the rows of those two; a prediction is the class with most pairwise votes, ties broken by the sum of the pairwise
 * confidences (scikit-learn's `_ovr_decision_function`). The decision function is that votes-plus-confidence score.
 *
 * @param estimator - The binary base classifier.
 * @returns The estimator.
 * @example
 * ```ts
 * import { oneVsOne, linearDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [5], [6], [10], [11]]
 * const f = oneVsOne(linearDiscriminant()).fit({ x, y: [0, 0, 1, 1, 2, 2] })
 * toFlat(f.predict([[0.5], [5.5], [10.5]])) // => [0, 1, 2]
 * ```
 */
export function oneVsOne(
  estimator: Estimator,
): IterativeEstimator<MulticlassHyper, StagedState<OneVsOnePlan>, MulticlassParams<OneVsOnePlan>, MulticlassVerbs> {
  return build<OneVsOnePlan, MulticlassHyper>(
    'one-vs-one',
    { estimator },
    {
      plan: (classes) => ({
        pairs: classes.flatMap((_, i) => classes.slice(i + 1).map((_, o): [number, number] => [i, i + 1 + o])),
      }),
      total: (plan) => plan.pairs.length,
      problem: (plan, index, codes) => {
        const [a, b] = plan.pairs[index] ?? [0, 1]
        const rows = Array.from(codes.keys()).filter((i) => codes[i] === a || codes[i] === b)
        return { rows, labels: rows.map((i) => (codes[i] === b ? 1 : 0)) }
      },
      verbs: (p, h) => {
        const fits = revived(h.estimator, p.models)
        const k = p.classes.length
        const d = fits[0]?.nFeaturesIn ?? 0
        const threshold = fits[0] !== undefined && hasDecision(fits[0]) ? 0 : 0.5
        const decide = (x: TensorLike): { readonly n: number; readonly y: Float64Array } => {
          const conf = fits.map((f) => confidence(f, x))
          const hard = fits.map((f) => predicted(f, x))
          const n = conf[0]?.length ?? 0
          const votes = new Float64Array(n * k)
          const sums = new Float64Array(n * k)
          fits.forEach((_, m) => {
            const [a, b] = p.plan.pairs[m] ?? [0, 1]
            for (let r = 0; r < n; r++) {
              const c = conf[m]?.[r] ?? 0
              sums[r * k + a] = (sums[r * k + a] ?? 0) - c
              sums[r * k + b] = (sums[r * k + b] ?? 0) + c
              const target = (hard[m]?.[r] ?? 0) === 0 ? a : b
              votes[r * k + target] = (votes[r * k + target] ?? 0) + 1
            }
          })
          const y = votes.map((v, i) => v + (sums[i] ?? 0) / (3 * (Math.abs(sums[i] ?? 0) + 1)))
          return { n, y }
        }
        const predict = (x: TensorLike): Tensor => {
          const { n, y } = decide(x)
          if (k === 2)
            return matrix(
              Array.from({ length: n }, (_, r) => p.classes[(y[r * 2 + 1] ?? 0) > threshold ? 1 : 0] ?? 0),
              [n],
            )
          return matrix(
            Array.from({ length: n }, (_, r) => p.classes[argmaxRow(y.subarray(r * k, (r + 1) * k))] ?? 0),
            [n],
          )
        }
        const decision = (x: TensorLike): Tensor => {
          const { n, y } = decide(x)
          return k === 2
            ? matrix(
                Array.from({ length: n }, (_, r) => y[r * 2 + 1] ?? 0),
                [n],
              )
            : matrix(y, [n, k])
        }
        return finish(p, d, predict, decision)
      },
    },
  )
}

/** The hyperparameters of {@link outputCode}. */
export interface OutputCodeHyper extends MulticlassHyper {
  /** The number of binary problems as a multiple of the number of classes (scikit-learn's `code_size`). */
  readonly codeSize: number
  /** A code book `[k][n_codes]` of 0 and 1 (default random from the key). */
  readonly codeBook?: readonly (readonly number[])[]
}

/** Output-code plan: the code of each class, one 0/1 column per problem. */
export type OutputCodePlan = { readonly codeBook: number[][] }

function randomCodeBook(k: number, size: number, key: Key): number[][] {
  const u = toFlat(uniform(key, [k, size]))
  const book = Array.from({ length: k }, (_, c) =>
    Array.from({ length: size }, (_, j) => ((u[c * size + j] ?? 0) > 0.5 ? 1 : 0)),
  )
  // A column of one value separates nothing: flip the entry of one class so every problem is binary.
  for (let j = 0; j < size; j++) {
    const ones = book.reduce((s, row) => s + (row[j] ?? 0), 0)
    if (ones === 0 && book[0] !== undefined) book[0][j] = 1
    if (ones === k && book[0] !== undefined) book[0][j] = 0
  }
  return book
}

/**
 * Error-correcting output codes (`sklearn.multiclass.OutputCodeClassifier`, Dietterich and Bakiri): every class gets
 * a binary code word, one base classifier learns each bit, and a prediction is the class whose code word is nearest
 * (Euclidean) to the bits' scores, so a few wrong bits are corrected. The code book is random from the run's key
 * with `codeSize * k` bits, or given. Scores are decision functions where the base has one (codes ±1), else
 * probabilities (codes 0/1). The decision function is minus the distance to each code word.
 *
 * @param estimator - The binary base classifier.
 * @param hyper - `codeSize` (default 1.5) and an optional `codeBook`.
 * @returns The estimator.
 * @reference T. G. Dietterich and G. Bakiri, Solving multiclass learning problems via error-correcting output codes,
 * JAIR 2 (1995).
 * @example
 * ```ts
 * import { outputCode, linearDiscriminant } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [5], [6], [10], [11]]
 * const book = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
 * const f = outputCode(linearDiscriminant(), { codeBook: book }).fit({ x, y: [0, 0, 1, 1, 2, 2] })
 * toFlat(f.predict([[0.5], [10.5]])) // => [0, 2]
 * ```
 */
export function outputCode(
  estimator: Estimator,
  hyper: Partial<Omit<OutputCodeHyper, 'estimator'>> = {},
): IterativeEstimator<OutputCodeHyper, StagedState<OutputCodePlan>, MulticlassParams<OutputCodePlan>, MulticlassVerbs> {
  return build<OutputCodePlan, OutputCodeHyper>(
    'output-code',
    { estimator, codeSize: 1.5, ...hyper },
    {
      plan: (classes, _codes, key, h) => ({
        codeBook:
          h.codeBook === undefined
            ? randomCodeBook(classes.length, Math.max(1, Math.floor(classes.length * h.codeSize)), key)
            : h.codeBook.map((row) => [...row]),
      }),
      total: (plan) => plan.codeBook[0]?.length ?? 0,
      problem: (plan, index, codes) => ({ labels: Float64Array.from(codes, (c) => plan.codeBook[c]?.[index] ?? 0) }),
      verbs: (p, h) => {
        const fits = revived(h.estimator, p.models)
        const k = p.classes.length
        const d = fits[0]?.nFeaturesIn ?? 0
        const signed = fits[0] !== undefined && hasDecision(fits[0])
        const code = (c: number, j: number): number => {
          const bit = p.plan.codeBook[c]?.[j] ?? 0
          return signed ? 2 * bit - 1 : bit
        }
        const distances = (x: TensorLike): { readonly n: number; readonly dist: Float64Array } => {
          const conf = fits.map((f) => confidence(f, x))
          const n = conf[0]?.length ?? 0
          const dist = new Float64Array(n * k)
          for (let r = 0; r < n; r++)
            for (let c = 0; c < k; c++) {
              let s = 0
              fits.forEach((_, j) => {
                s += ((conf[j]?.[r] ?? 0) - code(c, j)) ** 2
              })
              dist[r * k + c] = Math.sqrt(s)
            }
          return { n, dist }
        }
        const predict = (x: TensorLike): Tensor => {
          const { n, dist } = distances(x)
          return matrix(
            Array.from(
              { length: n },
              (_, r) => p.classes[argmaxRow(dist.subarray(r * k, (r + 1) * k).map((v) => -v))] ?? 0,
            ),
            [n],
          )
        }
        const decision = (x: TensorLike): Tensor => {
          const { n, dist } = distances(x)
          return matrix(
            dist.map((v) => -v),
            [n, k],
          )
        }
        return finish(p, d, predict, decision)
      },
    },
    (h) => dict({ codeSize: real(0.5, Math.max(4, h.codeSize), { default: h.codeSize }) }),
  )
}

/** Nested-dichotomy plan: for each tree, its internal nodes as the classes sent left and right. */
export type NestedDichotomiesPlan = { readonly trees: { readonly left: number[]; readonly right: number[] }[][] }

/** The hyperparameters of {@link nestedDichotomies}. */
export interface NestedDichotomiesHyper extends MulticlassHyper {
  /** The number of random class hierarchies averaged (an ensemble of nested dichotomies). */
  readonly trees: number
  /** One given hierarchy (internal nodes, parents before children) in place of random ones. */
  readonly hierarchy?: readonly { readonly left: readonly number[]; readonly right: readonly number[] }[]
}

function randomTree(classes: readonly number[], key: Key): { left: number[]; right: number[] }[] {
  const nodes: { left: number[]; right: number[] }[] = []
  const grow = (set: readonly number[], k: Key, depth: number): void => {
    if (set.length < 2) return
    const order = toFlat(permutation(k, set.length))
    const shuffled = Array.from(order, (i) => set[i] ?? 0)
    const cut =
      1 + Math.min(set.length - 2, Math.floor((toFlat(uniform(foldIn(k, 77), [1]))[0] ?? 0) * (set.length - 1)))
    const left = shuffled.slice(0, cut).sort((a, b) => a - b)
    const right = shuffled.slice(cut).sort((a, b) => a - b)
    nodes.push({ left, right })
    grow(left, foldDepth(k, depth, 0), depth + 1)
    grow(right, foldDepth(k, depth, 1), depth + 1)
  }
  grow(classes, key, 0)
  return nodes
}

const foldDepth = (k: Key, depth: number, side: number): Key => foldIn(k, depth * 2 + side + 1)

/**
 * Nested dichotomies (Frank and Kramer): the classes are split at random into two groups, a base classifier learns to
 * tell the groups apart, and each group is split again until single classes remain; the probability of a class is the
 * product of the branch probabilities on its path. `trees` random hierarchies are averaged (an ensemble of nested
 * dichotomies). The base must give probabilities. A node not yet solved, in a session, splits its rows evenly.
 *
 * @param estimator - The binary base classifier, with `predictProba`.
 * @param hyper - `trees`.
 * @returns The estimator.
 * @reference E. Frank and S. Kramer, Ensembles of nested dichotomies for multi-class problems, ICML (2004).
 * @example
 * ```ts
 * import { nestedDichotomies, naiveBayes } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [5], [6], [10], [11]]
 * const f = nestedDichotomies(naiveBayes()).fit({ x, y: [0, 0, 1, 1, 2, 2] })
 * toFlat(f.predict([[0.5], [5.5], [10.5]])) // => [0, 1, 2]
 * ```
 */
export function nestedDichotomies(
  estimator: Estimator,
  hyper: Partial<Omit<NestedDichotomiesHyper, 'estimator'>> = {},
): IterativeEstimator<
  NestedDichotomiesHyper,
  StagedState<NestedDichotomiesPlan>,
  MulticlassParams<NestedDichotomiesPlan>,
  MulticlassVerbs
> {
  return build<NestedDichotomiesPlan, NestedDichotomiesHyper>(
    'nested-dichotomies',
    { estimator, trees: 1, ...hyper },
    {
      plan: (classes, _codes, key, h) => ({
        trees:
          h.hierarchy !== undefined
            ? [h.hierarchy.map((n) => ({ left: [...n.left], right: [...n.right] }))]
            : Array.from({ length: Math.max(1, h.trees) }, (_, t) =>
                randomTree(
                  classes.map((_, c) => c),
                  foldIn(key, 1000 + t),
                ),
              ),
      }),
      total: (plan) => plan.trees.reduce((s, t) => s + t.length, 0),
      problem: (plan, index, codes) => {
        let rest = index
        let node = plan.trees[0]?.[0]
        for (const tree of plan.trees) {
          if (rest < tree.length) {
            node = tree[rest]
            break
          }
          rest -= tree.length
        }
        const left = new Set(node?.left ?? [])
        const right = new Set(node?.right ?? [])
        const rows = Array.from(codes.keys()).filter((i) => left.has(codes[i] ?? -1) || right.has(codes[i] ?? -1))
        return { rows, labels: rows.map((i) => (right.has(codes[i] ?? -1) ? 1 : 0)) }
      },
      verbs: (p, h) => {
        const fits = revived(h.estimator, p.models)
        const k = p.classes.length
        const d = fits[0]?.nFeaturesIn ?? 0
        if (fits.some((f) => f.predictProba === undefined)) {
          throw new MdmlError('value', 'nestedDichotomies: the base estimator must give probabilities (predictProba)')
        }
        const logProba = (x: TensorLike): { readonly n: number; readonly p: Float64Array } => {
          const m = matrixView(x, 'nestedDichotomies')
          const out = new Float64Array(m.n * k)
          let index = 0
          for (const tree of p.plan.trees) {
            const right = tree.map((_, i) => {
              const f = fits[index + i]
              if (f === undefined) return new Float64Array(m.n).fill(0.5)
              const pr = (f.predictProba as (x: TensorLike) => Tensor)(x)
              const kk = pr.shape[1] ?? 2
              const flat = toFlat(pr)
              return Float64Array.from({ length: m.n }, (_, r) => flat[r * kk + (kk - 1)] ?? 0.5)
            })
            index += tree.length
            for (let r = 0; r < m.n; r++)
              for (let c = 0; c < k; c++) {
                let prob = 1
                tree.forEach((node, i) => {
                  const q = right[i]?.[r] ?? 0.5
                  if (node.right.includes(c)) prob *= q
                  else if (node.left.includes(c)) prob *= 1 - q
                })
                out[r * k + c] = (out[r * k + c] ?? 0) + prob / p.plan.trees.length
              }
          }
          return { n: m.n, p: out }
        }
        const proba = (x: TensorLike): Tensor => {
          const { n, p: pr } = logProba(x)
          return matrix(pr, [n, k])
        }
        const predict = (x: TensorLike): Tensor => {
          const { n, p: pr } = logProba(x)
          return matrix(
            Array.from({ length: n }, (_, r) => p.classes[argmaxRow(pr.subarray(r * k, (r + 1) * k))] ?? 0),
            [n],
          )
        }
        const decision = (x: TensorLike): Tensor => {
          const { n, p: pr } = logProba(x)
          return k === 2
            ? matrix(
                Array.from({ length: n }, (_, r) => Math.log((pr[r * 2 + 1] ?? 0) / (pr[r * 2] ?? 1))),
                [n],
              )
            : matrix(pr.map(Math.log), [n, k])
        }
        return finish(p, d, predict, decision, proba)
      },
    },
  )
}
