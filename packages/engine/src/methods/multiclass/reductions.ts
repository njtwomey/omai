/**
 * Reductions of a multiclass problem to binary ones, as in scikit-learn's `multiclass` module: one-versus-rest (one
 * classifier per class against the others), one-versus-one (one per pair of classes, combined by votes) and
 * error-correcting output codes (one per bit of a random binary code for each class, decoded to the nearest code). Any
 * binary classifier can be the member.
 */

import { argmaxOf, type Json, WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type Exposition, ensembleStages } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type Estimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  requireVerb,
  takeData,
} from '../../lib/learn/index.ts'
import { foldIn, uniform } from '../../lib/random/index.ts'
import { dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'

// TODO: nested dichotomies; multilabel (binary relevance, classifier chains, label powerset) and multi-output
// regression (multilabel.ts, multioutput.ts).

/** The reduction of a multiclass problem to binary ones. */
export type Reduction = 'ovr' | 'ovo' | 'ecoc'

/** Hyperparameters of the reductions. */
export interface ReductionHyper {
  /** The binary classifier (default logistic regression). */
  readonly estimator: Estimator
  /** Output codes: code length as a multiple of the number of classes (default 1.5). */
  readonly codeSize: number
}

/** Parameters of a fitted reduction: the binary problems' codes and their members' parameters. */
export type ReductionParams = {
  readonly classes: number[]
  /** codes[c][b]: the target (1 or 0) of class c in binary problem b; -1 when the class is left out (one-versus-one). */
  readonly codes: number[][]
  readonly members: Json[]
  readonly nFeatures: number
}

/** Verbs of a fitted reduction. */
export interface ReductionVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** Each binary problem's probability of its positive side, `[problems][m]`. */
  readonly memberScores: (x: TensorLike) => number[][]
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted reduction. */
export type FittedReduction = Fitted<ReductionParams> & ReductionVerbs

/** The probability of the positive label (1) from a binary member, or its 0/1 prediction. */
function positiveScore(f: Fitted, x: TensorLike): number[] {
  if (f.predictProba !== undefined && f.classes !== undefined) {
    const p = toFlat(f.predictProba(x))
    const k = f.classes.length
    const col = f.classes.indexOf(1)
    return Array.from({ length: p.length / k }, (_, i) => (col < 0 ? 0 : (p[i * k + col] ?? 0)))
  }
  return toFlat(requireVerb(f, 'predict', 'reduction')(x)).map((v) => (v === 1 ? 1 : 0))
}

function combine(
  kind: Reduction,
  codes: readonly (readonly number[])[],
  scores: readonly (readonly number[])[],
  m: number,
): Float64Array {
  const k = codes.length
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    const row = new Float64Array(k)
    for (let c = 0; c < k; c++) {
      const code = codes[c] ?? []
      if (kind === 'ovr') row[c] = scores[c]?.[i] ?? 0
      else if (kind === 'ovo') {
        // Votes plus a small share of the summed confidences (scikit-learn's tie-break).
        let votes = 0
        let conf = 0
        for (const [b, t] of code.entries()) {
          if (t < 0) continue
          const s = scores[b]?.[i] ?? 0.5
          const mine = t === 1 ? s : 1 - s
          if (mine > 0.5) votes++
          conf += mine
        }
        row[c] = votes + conf / (3 * (conf + 1))
      } else {
        // Output codes: the class whose code is nearest (in L1) to the members' scores.
        let dist = 0
        for (const [b, t] of code.entries()) dist += Math.abs((scores[b]?.[i] ?? 0) - t)
        row[c] = -dist
      }
    }
    // Normalise into probabilities: shares of the positive scores (ovr) or a softmax of the votes / code distances.
    if (kind === 'ovr') {
      const total = row.reduce((a, b) => a + b, 0)
      for (let c = 0; c < k; c++) out[i * k + c] = total > 0 ? (row[c] ?? 0) / total : 1 / k
    } else {
      const max = Math.max(...row)
      const e = Array.from(row, (v) => Math.exp(v - max))
      const z = e.reduce((a, b) => a + b, 0)
      for (let c = 0; c < k; c++) out[i * k + c] = (e[c] ?? 0) / z
    }
  }
  return out
}

function reduction(
  name: string,
  kind: Reduction,
  hyper: ReductionHyper,
): ExplainedEstimator<ReductionHyper, FittedReduction> {
  const est: ExplainedEstimator<ReductionHyper, FittedReduction> = defineExplainedEstimator(
    {
      name,
      kind: 'classifier',
      hyperSpace: (h: ReductionHyper): Space =>
        kind === 'ecoc' ? dict({ codeSize: real(0.5, 4, { default: h.codeSize, label: 'code size' }) }) : dict({}),
      update: (h, changes) => {
        const own = Object.fromEntries(Object.entries(changes).filter(([k]) => !k.startsWith('estimator.')))
        const inner = Object.fromEntries(
          Object.entries(changes).flatMap(([k, v]) => (k.startsWith('estimator.') ? [[k.slice(10), v]] : [])),
        )
        return {
          ...h,
          ...own,
          estimator: Object.keys(inner).length > 0 ? h.estimator.with(inner) : h.estimator,
        } as ReductionHyper
      },
      fit: (data: LearnData, h: ReductionHyper, options: FitOptions) => {
        const m = matrixView(data.x, name)
        const { classes, codes } = encodeLabels(data.y ?? [], name, m.n)
        const k = classes.length
        if (k < 2) throw new WhiteboxError('value', `${name}: needs two classes`)
        const key = resolveKey(options, name)
        let table: number[][]
        if (kind === 'ovr') table = classes.map((_, c) => classes.map((_, b) => (b === c ? 1 : 0)))
        else if (kind === 'ovo') {
          const pairs: [number, number][] = []
          for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) pairs.push([a, b])
          table = classes.map((_, c) => pairs.map(([a, b]) => (c === a ? 1 : c === b ? 0 : -1)))
        } else {
          const bits = Math.max(1, Math.round(h.codeSize * k))
          // Random codes, as scikit-learn's OutputCodeClassifier, redrawn until every column has both labels.
          table = classes.map(() => new Array<number>(bits).fill(0))
          for (let b = 0; b < bits; b++) {
            let attempt = 0
            let column: number[]
            do {
              column = Array.from(toFlat(uniform(foldIn(foldIn(key, b), attempt++), [k]))).map((u) => (u > 0.5 ? 1 : 0))
            } while (new Set(column).size < 2 && attempt < 100)
            for (let c = 0; c < k; c++) (table[c] as number[])[b] = column[c] ?? 0
          }
        }
        const problems = table[0]?.length ?? 0
        const members: Fitted[] = []
        const trainScores: number[] = []
        for (let b = 0; b < problems; b++) {
          const rows = Array.from(codes.keys()).filter((i) => (table[codes[i] ?? 0]?.[b] ?? -1) >= 0)
          const targets = rows.map((i) => table[codes[i] ?? 0]?.[b] ?? 0)
          const sub = takeData(
            { x: data.x, y: Array.from(codes), ...(data.weights === undefined ? {} : { weights: data.weights }) },
            rows,
          )
          const f = h.estimator.fit({ ...sub, y: targets }, { key: foldIn(key, 1000 + b) })
          members.push(f)
          const s = positiveScore(f, sub.x)
          trainScores.push(s.reduce((a, v, i) => a + ((v > 0.5 ? 1 : 0) === targets[i] ? 1 : 0), 0) / rows.length)
        }
        const params: ReductionParams = {
          classes: [...classes],
          codes: table,
          members: members.map((f) => f.params),
          nFeatures: m.d,
        }
        const exposition: Exposition[] = [
          ensembleStages({
            method: name,
            weights: members.map(() => 1),
            trainScores,
            scoreName: 'member training accuracy',
            higherIsBetter: true,
          }),
        ]
        if (m.d === 2)
          exposition.push(
            decisionSurfaceOf(est.revive(params), data.x, {
              featureNames: featureNamesOf(data, m.d),
              classNames: classNamesOf(data, classes),
            }),
          )
        return { params, exposition }
      },
      methods: (p: ReductionParams, h: ReductionHyper): ReductionVerbs => {
        const members = p.members.map((m) => h.estimator.revive(m))
        const memberScores = (x: TensorLike): number[][] => members.map((f) => positiveScore(f, x))
        const proba = (x: TensorLike): Tensor => {
          const n = matrixView(x, name, { nFeatures: p.nFeatures }).n
          return fromStorage(combine(kind, p.codes, memberScores(x), n), [n, p.classes.length], 'float64')
        }
        return {
          predict: (x) => {
            const pr = proba(x)
            const k = p.classes.length
            const n = pr.shape[0] ?? 0
            return fromStorage(
              Float64Array.from(
                { length: n },
                (_, i) => p.classes[argmaxOf(pr.data.subarray(i * k, (i + 1) * k))] ?? 0,
              ),
              [n],
              'float64',
            )
          },
          predictProba: proba,
          predictDistribution: (x) => categorical({ probs: proba(x) }),
          memberScores,
          classes: p.classes,
          nFeaturesIn: p.nFeatures,
        }
      },
    },
    hyper,
  )
  return est
}

const defaults = (hyper: Partial<ReductionHyper>): ReductionHyper => ({
  estimator: hyper.estimator ?? logisticRegression(),
  codeSize: hyper.codeSize ?? 1.5,
})

/**
 * One-versus-rest (sklearn.multiclass.OneVsRestClassifier): one binary classifier per class against the others; the
 * class with the largest positive probability wins (probabilities are the normalised shares). Exposition: the binary
 * members as stages (each member's training accuracy on its own problem) and, for two features, the decision surface.
 *
 * @param hyper - The binary estimator (default logistic regression).
 * @returns The estimator.
 */
export function oneVsRestClassifier(
  hyper: Partial<ReductionHyper> = {},
): ExplainedEstimator<ReductionHyper, FittedReduction> {
  return reduction('one-vs-rest', 'ovr', defaults(hyper))
}

/**
 * One-versus-one (sklearn.multiclass.OneVsOneClassifier): one binary classifier per pair of classes on that pair's
 * rows, combined by votes with summed confidences breaking ties. Exposition as {@link oneVsRestClassifier}.
 *
 * @param hyper - The binary estimator (default logistic regression).
 * @returns The estimator.
 */
export function oneVsOneClassifier(
  hyper: Partial<ReductionHyper> = {},
): ExplainedEstimator<ReductionHyper, FittedReduction> {
  return reduction('one-vs-one', 'ovo', defaults(hyper))
}

/**
 * Error-correcting output codes (sklearn.multiclass.OutputCodeClassifier): keyed random binary codes for the classes,
 * one classifier per code bit, decoded to the nearest class code. Exposition as {@link oneVsRestClassifier}.
 *
 * @param hyper - The binary estimator and the code size (bits per class, default 1.5).
 * @returns The estimator.
 */
export function outputCodeClassifier(
  hyper: Partial<ReductionHyper> = {},
): ExplainedEstimator<ReductionHyper, FittedReduction> {
  return reduction('output-code', 'ecoc', defaults(hyper))
}

/** {@link oneVsRestClassifier} as a class. */
export class OneVsRestClassifier extends ExplainedEstimatorClass<ReductionHyper, FittedReduction> {
  /** @param hyper - As {@link oneVsRestClassifier}. */
  constructor(hyper: Partial<ReductionHyper> = {}) {
    super(oneVsRestClassifier(hyper))
  }
}

/** {@link oneVsOneClassifier} as a class. */
export class OneVsOneClassifier extends ExplainedEstimatorClass<ReductionHyper, FittedReduction> {
  /** @param hyper - As {@link oneVsOneClassifier}. */
  constructor(hyper: Partial<ReductionHyper> = {}) {
    super(oneVsOneClassifier(hyper))
  }
}
