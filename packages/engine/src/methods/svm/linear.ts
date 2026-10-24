import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { coefficientTable, type Exposition, marginRecord } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  collecting,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { ddot } from '../../lib/numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../../lib/random/index.ts'
import { choice, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'

// TODO: linear SVR, Pegasos, Crammer-Singer multiclass.

/** Hyperparameters of {@link linearSvc}. */
export interface LinearSvcHyper {
  readonly C: number
  /** The hinge (L1) or squared hinge (L2, scikit-learn's default) loss. */
  readonly loss: 'hinge' | 'squared_hinge'
  readonly fitIntercept: boolean
  /** The constant feature appended for the intercept, which is penalised with the weights (LIBLINEAR's bias). */
  readonly interceptScaling: number
  readonly tol: number
  readonly maxIter: number
}

/** The state of dual coordinate descent. */
interface DcdState {
  readonly alpha: Float64Array
  readonly w: Float64Array
  readonly gap: number
  readonly objective: number
}

/**
 * Dual coordinate descent for one binary linear SVM (Hsieh et al. 2008, LIBLINEAR): one keyed sweep over the
 * coordinates per step, each solved exactly within its box; stops when the projected-gradient spread falls to tol.
 */
function dcdAlgorithm(x: MatrixView, bias: number, y: Float64Array, h: LinearSvcHyper): Algorithm<DcdState> {
  const { n, d } = x
  const dim = d + (bias > 0 ? 1 : 0)
  const row = (i: number): Float64Array => {
    const r = new Float64Array(dim)
    r.set(x.data.subarray(i * d, (i + 1) * d))
    if (bias > 0) r[d] = bias
    return r
  }
  const rows = Array.from({ length: n }, (_, i) => row(i))
  const diag = h.loss === 'hinge' ? 0 : 1 / (2 * h.C)
  const upper = h.loss === 'hinge' ? h.C : Number.POSITIVE_INFINITY
  const qd = rows.map((r) => ddot(r, r) + diag)
  const objectiveOf = (alpha: Float64Array, w: Float64Array): number => {
    let s = 0.5 * ddot(w, w)
    for (let i = 0; i < n; i++) s += 0.5 * diag * (alpha[i] ?? 0) ** 2 - (alpha[i] ?? 0)
    return s
  }
  return {
    name: 'dualCoordinateDescent',
    config: { C: h.C, loss: h.loss },
    schema: channels({
      metrics: {
        objective: { description: 'dual objective (minimised)', direction: 'lower' },
        gap: { description: 'projected-gradient spread' },
      },
    }),
    init: () => ({ alpha: new Float64Array(n), w: new Float64Array(dim), gap: Number.POSITIVE_INFINITY, objective: 0 }),
    step: (s, ctx) => {
      const alpha = Float64Array.from(s.alpha)
      const w = Float64Array.from(s.w)
      let pgMax = Number.NEGATIVE_INFINITY
      let pgMin = Number.POSITIVE_INFINITY
      for (const i of toFlat(permutation(ctx.key, n))) {
        const r = rows[i] as Float64Array
        const yi = y[i] ?? 1
        const a = alpha[i] ?? 0
        const g = yi * ddot(w, r) - 1 + diag * a
        const pg = a === 0 ? Math.min(g, 0) : a >= upper ? Math.max(g, 0) : g
        pgMax = Math.max(pgMax, pg)
        pgMin = Math.min(pgMin, pg)
        if (Math.abs(pg) > 1e-12) {
          const next = Math.min(Math.max(a - g / (qd[i] ?? 1), 0), upper)
          const delta = (next - a) * yi
          for (let j = 0; j < dim; j++) w[j] = (w[j] ?? 0) + delta * (r[j] ?? 0)
          alpha[i] = next
        }
      }
      return { alpha, w, gap: pgMax - pgMin, objective: objectiveOf(alpha, w) }
    },
    done: (s) => (s.gap <= h.tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { objective: s.objective, gap: Number.isFinite(s.gap) ? s.gap : -1 } }),
  }
}

/** Parameters of a fitted linear SVM: one score row per class (one row for two classes). */
export type LinearSvcParams = { readonly classes: number[]; readonly coef: number[][]; readonly intercept: number[] }

/** Verbs of a fitted linear SVM. */
export interface LinearSvcVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly classes: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted linear SVM. */
export type FittedLinearSvc = Fitted<LinearSvcParams> & LinearSvcVerbs

/**
 * A linear SVM (sklearn.svm.LinearSVC, LIBLINEAR): the hinge or squared-hinge loss with an L2 penalty, solved in the
 * dual by traced coordinate descent, one-versus-rest for several classes; the intercept is a penalised constant
 * feature, as LIBLINEAR's. Exposition: the coefficient table, one margin record per binary problem (support vectors,
 * dual coefficients, margins and the dual objective per sweep) and, for two features, the decision surface.
 *
 * @param hyper - C (default 1), loss (squared hinge), intercept and its scaling, tolerance (1e-4) and sweeps.
 * @returns The estimator.
 */
export function linearSvc(hyper: Partial<LinearSvcHyper> = {}): ExplainedEstimator<LinearSvcHyper, FittedLinearSvc> {
  const est: ExplainedEstimator<LinearSvcHyper, FittedLinearSvc> = defineExplainedEstimator(
    {
      name: 'linear-svc',
      kind: 'classifier',
      hyperSpace: (h: LinearSvcHyper): Space =>
        dict({
          C: real(1e-3, 1e3, { log: true, default: h.C }),
          loss: choice(['hinge', 'squared_hinge'], { default: h.loss }),
        }),
      fit: (data: LearnData, h: LinearSvcHyper, options: FitOptions) => {
        const x = matrixView(data.x, 'linearSvc')
        const { classes, codes } = encodeLabels(data.y ?? [], 'linearSvc', x.n)
        if (classes.length < 2) throw new WhiteboxError('value', 'linearSvc: needs two classes')
        const bias = h.fitIntercept ? h.interceptScaling : 0
        const key: Key = resolveKey(options, 'linearSvc')
        const problems = classes.length === 2 ? [1] : classes.map((_, c) => c)
        const exposition: Exposition[] = []
        const coef: number[][] = []
        const intercept: number[] = []
        for (const [s, positive] of problems.entries()) {
          const y = Float64Array.from(codes, (c) => (c === positive ? 1 : -1))
          const tracked = collecting(dcdAlgorithm(x, bias, y, h), (state) => state.objective)
          const r = runAlgorithm(tracked.algorithm, {
            maxSteps: h.maxIter,
            key: foldIn(key, s),
            ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
            ...(options.policy === undefined ? {} : { policy: options.policy }),
          })
          const objectives = tracked.history.slice(1)
          const w = Array.from(r.state.w)
          const b = bias > 0 ? (w[x.d] ?? 0) * bias : 0
          coef.push(w.slice(0, x.d))
          intercept.push(b)
          const support = Array.from(r.state.alpha.keys()).filter((i) => (r.state.alpha[i] ?? 0) > 0)
          exposition.push(
            marginRecord({
              support,
              dualCoefficients: support.map((i) => (r.state.alpha[i] ?? 0) * (y[i] ?? 1)),
              intercept: b,
              margins: Array.from(
                { length: x.n },
                (_, i) => (y[i] ?? 1) * (b + ddot(x.data.subarray(i * x.d, (i + 1) * x.d), w.slice(0, x.d))),
              ),
              dualObjective: objectives,
            }),
          )
        }
        const params: LinearSvcParams = { classes: [...classes], coef, intercept }
        const features = featureNamesOf(data, x.d)
        const classNames = classNamesOf(data, classes)
        exposition.unshift(
          coefficientTable({
            names:
              coef.length === 1
                ? [...features, 'intercept']
                : classNames.flatMap((c) => [...features.map((f) => `${c}:${f}`), `${c}:intercept`]),
            estimates: coef.flatMap((row, s) => [...row, intercept[s] ?? 0]),
          }),
        )
        if (x.d === 2)
          exposition.push(decisionSurfaceOf(est.revive(params), data.x, { featureNames: features, classNames }))
        return { params, exposition }
      },
      methods: (p: LinearSvcParams): LinearSvcVerbs => {
        const d = p.coef[0]?.length ?? 0
        const scores = (x: TensorLike): { m: number; s: Float64Array } => {
          const v = matrixView(x, 'linearSvc', { nFeatures: d })
          const r = p.coef.length
          const s = new Float64Array(v.n * r)
          for (let i = 0; i < v.n; i++)
            for (let k = 0; k < r; k++)
              s[i * r + k] = (p.intercept[k] ?? 0) + ddot(v.data.subarray(i * d, (i + 1) * d), p.coef[k] ?? [])
          return { m: v.n, s }
        }
        return {
          predict: (x) => {
            const { m, s } = scores(x)
            const r = p.coef.length
            return fromStorage(
              Float64Array.from({ length: m }, (_, i) =>
                r === 1
                  ? (p.classes[(s[i] ?? 0) > 0 ? 1 : 0] ?? 0)
                  : (p.classes[argmaxOf(s.subarray(i * r, (i + 1) * r))] ?? 0),
              ),
              [m],
              'float64',
            )
          },
          decisionFunction: (x) => {
            const { m, s } = scores(x)
            return p.coef.length === 1 ? fromStorage(s, [m], 'float64') : fromStorage(s, [m, p.coef.length], 'float64')
          },
          classes: p.classes,
          nFeaturesIn: d,
          featureImportances: Array.from({ length: d }, (_, j) =>
            p.coef.reduce((a, row) => a + Math.abs(row[j] ?? 0), 0),
          ),
        }
      },
    },
    {
      C: hyper.C ?? 1,
      loss: hyper.loss ?? 'squared_hinge',
      fitIntercept: hyper.fitIntercept ?? true,
      interceptScaling: hyper.interceptScaling ?? 1,
      tol: hyper.tol ?? 1e-4,
      maxIter: hyper.maxIter ?? 1000,
    },
  )
  return est
}

/** {@link linearSvc} as a class. */
export class LinearSVC extends ExplainedEstimatorClass<LinearSvcHyper, FittedLinearSvc> {
  /** @param hyper - As {@link linearSvc}. */
  constructor(hyper: Partial<LinearSvcHyper> = {}) {
    super(linearSvc(hyper))
  }
}
