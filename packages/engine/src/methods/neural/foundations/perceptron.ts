import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import { coefficientTable, type Exposition, learningCurve } from '../../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type Fitted,
  featureNamesOf,
  matrixView,
} from '../../../lib/learn/index.ts'
import { ddot, dnrm2 } from '../../../lib/numerics/linalg/index.ts'
import { foldIn, key as makeKey, permutation } from '../../../lib/random/index.ts'
import { bool, dict, int, real, type Space } from '../../../lib/space/index.ts'
import { type Tensor, type TensorLike, tensor, toFlat } from '../../../lib/tensor/index.ts'

/** Hyperparameters of {@link rosenblattPerceptron}. */
export interface RosenblattHyper {
  /** The most passes over the data (default 50). */
  readonly maxEpochs: number
  /** The step of each update (default 1; the perceptron's mistakes do not depend on it when the weights start at 0). */
  readonly eta: number
  readonly fitIntercept: boolean
  /** Visit the rows in a fresh random order each epoch (default true). */
  readonly shuffle: boolean
  /** The most updates kept in the record (default 2000). */
  readonly maxRecorded: number
}

/** One update of the perceptron: the misclassified row, its label (±1) and the weights after the update. */
export type PerceptronUpdate = {
  readonly epoch: number
  /** The row's index in the training data. */
  readonly row: number
  readonly label: number
  /** The weights (then the intercept) after the update. */
  readonly weights: readonly number[]
}

/** The learnt parameters of {@link rosenblattPerceptron}. */
export interface RosenblattParams {
  readonly [key: string]: Json
  readonly classes: readonly number[]
  readonly weights: readonly number[]
  readonly intercept: number
  readonly updates: number
  readonly epochs: number
  readonly converged: boolean
  /** Every update in order, up to `maxRecorded`. */
  readonly history: readonly PerceptronUpdate[]
}

/** The verbs and study results of a fitted {@link rosenblattPerceptron}. */
export interface RosenblattVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

/**
 * Novikoff's bound on the perceptron's mistakes for separable data: at most (R / γ)² updates, R the largest norm of a
 * row (with the constant 1 appended when fitting an intercept) and γ the margin of a unit-norm separating hyperplane.
 * Given the perceptron's own final hyperplane (which separates when it converged), γ is its margin, so the bound holds
 * for that γ; a wider hyperplane would give a tighter bound.
 *
 * @param rows - The rows (augmented as they were trained).
 * @param labels - ±1 per row.
 * @param w - A separating weight vector (augmented likewise).
 * @returns R, γ and the bound (infinite when w does not separate).
 */
export function novikoffBound(
  rows: readonly (readonly number[])[],
  labels: readonly number[],
  w: readonly number[],
): { readonly radius: number; readonly margin: number; readonly bound: number } {
  const norm = dnrm2(w)
  let radius = 0
  let margin = Number.POSITIVE_INFINITY
  for (const [i, r] of rows.entries()) {
    radius = Math.max(radius, dnrm2(r))
    margin = Math.min(margin, ((labels[i] ?? 0) * ddot(w, r)) / (norm || 1))
  }
  return { radius, margin, bound: margin > 0 ? (radius / margin) ** 2 : Number.POSITIVE_INFINITY }
}

/**
 * Rosenblatt's perceptron on two classes, as a study of its learning rule: rows are visited in turn and every
 * misclassified row (y·(w·x + b) ≤ 0) moves the weights by η·y·x, until an epoch makes no mistake (convergence, which
 * Novikoff's theorem guarantees for separable data within (R/γ)² updates) or the epochs run out. Exposition: the
 * decision surface, the mistakes and updates per epoch, and the final weights with the Novikoff bound; the parameters
 * carry every update (the row that triggered it and the weights after), so a page can step through the rule.
 *
 * @param hyper - Epochs, step, intercept, shuffling and how many updates to record.
 * @returns The estimator.
 */
export function rosenblattPerceptron(
  hyper: Partial<RosenblattHyper> = {},
): ExplainedEstimator<RosenblattHyper, Fitted<RosenblattParams> & RosenblattVerbs> {
  return defineExplainedEstimator(
    {
      name: 'rosenblattPerceptron',
      kind: 'classifier',
      hyperSpace: (h: RosenblattHyper): Space =>
        dict({
          maxEpochs: int(1, 500, { log: true, default: h.maxEpochs }),
          eta: real(1e-3, 10, { log: true, default: h.eta }),
          shuffle: bool({ default: h.shuffle }),
          fitIntercept: bool({ default: h.fitIntercept }),
        }),
      fit: (data, h, options) => {
        const x = matrixView(data.x, 'rosenblattPerceptron')
        const enc = encodeLabels(data.y ?? [], 'rosenblattPerceptron', x.n)
        if (enc.classes.length !== 2)
          throw new WhiteboxError(
            'value',
            `rosenblattPerceptron: needs exactly two classes, got ${enc.classes.length} (use methods/linear's perceptron for more)`,
          )
        const { n, d } = x
        const m = h.fitIntercept ? d + 1 : d
        const rows = Array.from({ length: n }, (_, i) => {
          const r = Array.from(x.data.subarray(i * d, (i + 1) * d))
          return h.fitIntercept ? [...r, 1] : r
        })
        const labels = Array.from(enc.codes, (c) => (c === 1 ? 1 : -1))
        const w = new Float64Array(m)
        const history: PerceptronUpdate[] = []
        const mistakes: number[] = []
        const cumulative: number[] = []
        let updates = 0
        let converged = false
        const base = options.key ?? makeKey(0)
        let epoch = 0
        while (epoch < h.maxEpochs && !converged) {
          epoch++
          const order = h.shuffle ? toFlat(permutation(foldIn(base, epoch), n)) : Array.from({ length: n }, (_, i) => i)
          let wrong = 0
          for (const i of order) {
            const r = rows[i] as number[]
            const y = labels[i] ?? 0
            if (y * ddot(w, r) > 0) continue
            wrong++
            updates++
            for (let j = 0; j < m; j++) w[j] = (w[j] ?? 0) + h.eta * y * (r[j] ?? 0)
            if (history.length < h.maxRecorded) history.push({ epoch, row: i, label: y, weights: Array.from(w) })
          }
          mistakes.push(wrong)
          cumulative.push(updates)
          converged = wrong === 0
        }
        const weights = Array.from(w.subarray(0, d))
        const intercept = h.fitIntercept ? (w[d] ?? 0) : 0
        const params: RosenblattParams = {
          classes: [...enc.classes],
          weights,
          intercept,
          updates,
          epochs: epoch,
          converged,
          history,
        }
        const bound = novikoffBound(rows, labels, Array.from(w))
        const featureNames = featureNamesOf(data, d)
        const fitted = { estimator: 'rosenblattPerceptron', kind: 'classifier' as const, params, ...verbs(params) }
        const exposition: Exposition[] = [
          decisionSurfaceOf(fitted, data.x, { featureNames, classNames: classNamesOf(data, params.classes) }),
          learningCurve({
            sizes: mistakes.map((_, e) => e + 1),
            train: mistakes,
            validation: cumulative,
            resource: 'epoch',
            labels: { train: 'mistakes in the epoch', validation: 'updates so far' },
          }),
          coefficientTable({
            names: [
              ...featureNames,
              ...(h.fitIntercept ? ['intercept'] : []),
              'R',
              'margin γ',
              'bound (R/γ)²',
              'updates',
            ],
            estimates: [
              ...weights,
              ...(h.fitIntercept ? [intercept] : []),
              bound.radius,
              bound.margin,
              bound.bound,
              updates,
            ],
          }),
        ]
        // TODO: once DecisionSurface records steps (the records branch), the surface after geometric updates, each
        // highlighting the row that triggered it.
        return { params, exposition }
      },
      methods: (p) => verbs(p),
    },
    { maxEpochs: 50, eta: 1, fitIntercept: true, shuffle: true, maxRecorded: 2000, ...hyper },
  )
}

function verbs(p: RosenblattParams): RosenblattVerbs {
  const score = (x: TensorLike): number[] => {
    const m = matrixView(x, 'rosenblattPerceptron')
    return Array.from({ length: m.n }, (_, i) => ddot(p.weights, m.data.subarray(i * m.d, (i + 1) * m.d)) + p.intercept)
  }
  return {
    classes: p.classes,
    nFeaturesIn: p.weights.length,
    decisionFunction: (x) => tensor(score(x)),
    predict: (x) => tensor(score(x).map((s) => (s > 0 ? (p.classes[1] ?? 1) : (p.classes[0] ?? 0)))),
  }
}

/** The class form of {@link rosenblattPerceptron}. */
export class RosenblattPerceptron extends ExplainedEstimatorClass<
  RosenblattHyper,
  Fitted<RosenblattParams> & RosenblattVerbs
> {
  /** @param hyper - As {@link rosenblattPerceptron}. */
  constructor(hyper: Partial<RosenblattHyper> = {}) {
    super(rosenblattPerceptron(hyper))
  }
}
