/**
 * Linear support vector machines in the primal (scikit-learn's `LinearSVC` and `LinearSVR`, liblinear's models) as
 * estimators on the iterative base: a linear function of the features whose weights minimise a penalty plus the sum of
 * a hinge-type loss over the rows, with the intercept as a constant feature (`interceptScaling`) that is penalised with
 * the weights, as liblinear does. The model is the objective and the solver is a slot (`solver`, see `linearSvcSolvers`
 * and `linearSvrSolvers`): dual coordinate descent by default (one sweep per step, the dual and the duality gap on the
 * tape), orthant-wise L-BFGS for the L1 penalty, and every solver of `smoothSolvers` on the smooth squared-hinge and
 * squared epsilon-insensitive primals. More than two classes are one-versus-rest, one problem after another.
 */

import { OmaiError } from '../base/index.ts'
import {
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  sampleWeightsOf,
  targetVector,
} from '../learn/index.ts'
import { solverConfigOf, type VectorParams } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { choice, dict, real, type SpaceValue } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type ModeTable, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  checkLinearSvc,
  checkLinearSvr,
  type LinearDesign,
  type LinearPenalty,
  type LinearSvcLoss,
  type LinearSvcProblem,
  type LinearSvrLoss,
  type LinearSvrProblem,
  linearDesign,
  linearSvcSolvers,
  linearSvrSolvers,
  svcPrimal,
  svrPrimal,
} from './_linear.ts'

export {
  type LinearPenalty,
  type LinearSvcLoss,
  type LinearSvrLoss,
  linearSvcSolvers,
  linearSvrSolvers,
} from './_linear.ts'

/** The state of a linear SVM fit: the problem being solved, its solver's state, and the weights of every problem so far. */
export interface LinearSvmState {
  /** The problem being solved (the last one once all are). */
  readonly problem: number
  /** Steps taken in this problem. */
  readonly steps: number
  /** The solver's own state. */
  readonly inner: unknown
  /** The weights and scaled intercept `(w, b / s)` of each problem, the unsolved ones zero. */
  readonly weights: number[][]
  /** The primal objective of the problem last stepped, at its weights. */
  readonly objective: number
  readonly finished: boolean
  /** Whether some problem stopped at the iteration cap before its solver converged. */
  readonly capped: boolean
}

/** What the runner of problems in turn needs. */
interface InTurn<P> {
  readonly name: string
  readonly problems: readonly P[]
  readonly table: ModeTable<P, VectorParams>
  readonly solver: SpaceValue
  readonly maxIter: number
  readonly width: number
  readonly primal: (problem: P, w: ArrayLike<number>) => number
}

/** The problems solved one after another by the slot's solver, as one algorithm; each step is one step of the current solver. */
function inTurn<P>(spec: InTurn<P>): Algorithm<LinearSvmState> {
  const { problems, table, solver } = spec
  const algorithms = new Map<number, Algorithm<unknown>>()
  const algorithmOf = (k: number, key: Key): Algorithm<unknown> => {
    let a = algorithms.get(k)
    if (a === undefined) {
      a = table.make(problems[k] as P, solver, key)
      algorithms.set(k, a)
    }
    return a
  }
  const zeros = (): number[][] => problems.map(() => new Array<number>(spec.width).fill(0))
  return {
    name: spec.name,
    config: { problems: problems.length, solver: solver as never, maxIter: spec.maxIter },
    schema: channels({
      metrics: {
        objective: {
          description: 'the primal objective of the problem being solved, at its weights',
          direction: 'lower',
        },
        problem: { description: 'the index of the problem being solved' },
      },
      tensors: {
        weights: {
          role: 'param',
          description: 'the weights and the scaled intercept of each problem',
          shape: ['problems', 'width'],
        },
      },
    }),
    init: (key) => ({
      problem: 0,
      steps: 0,
      inner: algorithmOf(0, key).init(key),
      weights: zeros(),
      objective: spec.primal(problems[0] as P, new Array<number>(spec.width).fill(0)),
      finished: false,
      capped: false,
    }),
    step: (s, ctx) => {
      const alg = algorithmOf(s.problem, ctx.key)
      const inner = alg.step(s.inner, ctx)
      const w = table.params(solver, inner).x
      const weights = s.weights.map((row, k) => (k === s.problem ? w : row))
      const steps = s.steps + 1
      const solved = alg.done?.(inner, steps) !== undefined
      const capped = !solved && steps >= spec.maxIter
      const objective = spec.primal(problems[s.problem] as P, w)
      if (!(solved || capped)) return { ...s, inner, steps, weights, objective }
      const last = s.problem + 1 >= problems.length
      return {
        problem: last ? s.problem : s.problem + 1,
        steps: last ? steps : 0,
        inner: last ? inner : algorithmOf(s.problem + 1, ctx.key).init(ctx.key),
        weights,
        objective,
        finished: last,
        capped: s.capped || capped,
      }
    },
    done: (s) => (s.finished ? (s.capped ? 'completed' : 'converged') : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, problem: s.problem },
      tensors: { weights: { role: 'param', value: s.weights.flat(), shape: [problems.length, spec.width] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LinearSvmState,
  }
}

const designOf = (data: LearnData, who: string, fitIntercept: boolean, scaling: number): LinearDesign => {
  const m = matrixView(data.x, who)
  return linearDesign(m.data, m.n, m.d, fitIntercept, scaling)
}

/** `(coef, intercept)` of a problem's weights `(w, b / s)`. */
const splitWeights = (w: readonly number[], d: LinearDesign): { coef: number[]; intercept: number } => ({
  coef: w.slice(0, d.d),
  intercept: d.fitIntercept ? (w[d.d] ?? 0) * d.scaling : 0,
})

/** The rows of a design that have weight, and the problem over them. */
const keep = (design: LinearDesign, multiplier: Float64Array): number[] =>
  Array.from({ length: design.n }, (_, i) => i).filter((i) => (multiplier[i] ?? 0) > 0)

const subDesign = (d: LinearDesign, rows: readonly number[]): LinearDesign => {
  const raw = new Float64Array(rows.length * d.d)
  for (const [i, r] of rows.entries()) raw.set(d.raw.subarray(r * d.d, (r + 1) * d.d), i * d.d)
  return linearDesign(raw, rows.length, d.d, d.fitIntercept, d.scaling)
}

// --- the classifier -----------------------------------------------------------------------------------------------

/** The hyperparameters of {@link linearSvc}. */
export interface LinearSvcHyper {
  /** The penalty of the losses against the penalty on the weights (default 1). */
  readonly C: number
  /** The hinge (dual coordinate descent only) or the squared hinge (liblinear's L2 loss, the default). */
  readonly loss: LinearSvcLoss
  /** The norm of the weights that is penalised: `l2` (default) or `l1` (the squared hinge only, which gives sparse weights). */
  readonly penalty: LinearPenalty
  /** The solvers' stopping tolerance (default 1e-4). */
  readonly tol: number
  /** The cap on the steps of each binary problem (default 1000). */
  readonly maxIter: number
  readonly fitIntercept: boolean
  /** The value of the constant feature that carries the intercept: a larger one penalises the intercept less (default 1). */
  readonly interceptScaling: number
  /** `balanced` (`n / (classes * count)`) or a weight per class label, multiplying `C`. */
  readonly classWeight: 'balanced' | Readonly<Record<number, number>> | null
  /**
   * How a problem is fitted: a value of `linearSvcSolvers.space`. `dcd` (default) is liblinear's dual coordinate
   * descent, either loss under the L2 penalty (under `penalty: 'l1'` the default becomes `owlqn`); with the squared
   * hinge and the L2 penalty every solver of `smoothSolvers` minimises the primal.
   */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link linearSvc}. */
export type LinearSvcParams = {
  readonly classes: number[]
  /** The weights: one row for two classes (positive for the second class), else one per class (versus the rest). */
  readonly coef: number[][]
  readonly intercept: number[]
  /** The cap-or-converged steps the last problem took, and whether every problem converged. */
  readonly converged: boolean
}

/** The verbs of a fitted {@link linearSvc}. */
export interface LinearSvcVerbs {
  /** `[m]` for two classes (positive favours the second), else `[m, classes]` (one-versus-rest scores). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predict: (x: TensorLike) => Tensor
  /** The accuracy on a labelled dataset. */
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

const WHO_SVC = 'linearSvc'

/** The default `dcd` solver stands for what liblinear uses with the penalty: under L1 it is `owlqn`. */
function svcSolverOf(h: LinearSvcHyper): SpaceValue {
  const name = linearSvcSolvers.modeOf(h.solver).name
  const solver = name === 'dcd' && h.penalty === 'l1' ? solverConfigOf(linearSvcSolvers, { kind: 'owlqn' }) : h.solver
  checkLinearSvc(WHO_SVC, linearSvcSolvers.modeOf(solver).name, h.loss, h.penalty)
  return solver
}

function svcProblems(
  data: LearnData,
  h: LinearSvcHyper,
): { problems: LinearSvcProblem[]; classes: number[]; design: LinearDesign } {
  if (!(h.C > 0)) throw new OmaiError('value', `${WHO_SVC}: C must be positive`)
  if (!(h.interceptScaling > 0)) throw new OmaiError('value', `${WHO_SVC}: interceptScaling must be positive`)
  const design = designOf(data, WHO_SVC, h.fitIntercept, h.interceptScaling)
  const labels = targetVector(data.y ?? [], WHO_SVC, design.n)
  const classes = Array.from(new Set(labels)).sort((a, b) => a - b)
  if (classes.length < 2) throw new OmaiError('value', `${WHO_SVC}: needs at least two classes`)
  const sampleWeights = sampleWeightsOf(data.weights, design.n, WHO_SVC)
  const counts = classes.map((c) => labels.filter((v) => v === c).length)
  const multiplier = Float64Array.from(labels, (label, i) => {
    const c = classes.indexOf(label)
    let weight = 1
    if (h.classWeight === 'balanced') weight = design.n / (classes.length * (counts[c] ?? 1))
    else if (h.classWeight !== null) weight = h.classWeight[label] ?? 1
    return weight * (sampleWeights[i] ?? 1)
  })
  const rows = keep(design, multiplier)
  const kept = rows.length === design.n ? design : subDesign(design, rows)
  const weights = Float64Array.from(rows, (i) => multiplier[i] ?? 1)
  const positives = classes.length === 2 ? [classes[1] as number] : classes
  const problems = positives.map((positive): LinearSvcProblem => {
    const y = Float64Array.from(rows, (i) => (labels[i] === positive ? 1 : -1))
    if (!y.includes(1) || !y.includes(-1))
      throw new OmaiError('value', `${WHO_SVC}: class ${positive} and the rest both need rows with weight`)
    const problem: Omit<LinearSvcProblem, 'objective' | 'x0'> = {
      design: kept,
      y,
      C: h.C,
      weights,
      loss: h.loss,
      penalty: h.penalty,
      tol: h.tol,
      maxIter: h.maxIter,
    }
    return problem as LinearSvcProblem
  })
  return { problems, classes, design: kept }
}

function svcVerbs(p: LinearSvcParams, d: number): LinearSvcVerbs {
  const k = p.coef.length
  const decision = (x: TensorLike): Tensor => {
    const m = matrixView(x, WHO_SVC, { nFeatures: d })
    const out = new Float64Array(m.n * k)
    for (let i = 0; i < m.n; i++)
      for (let c = 0; c < k; c++) {
        let z = p.intercept[c] ?? 0
        const w = p.coef[c] as number[]
        for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (m.data[i * d + j] ?? 0)
        out[i * k + c] = z
      }
    return fromStorage(out, k === 1 ? [m.n] : [m.n, k], 'float64')
  }
  const predict = (x: TensorLike): Tensor => {
    const dec = toFlat(decision(x))
    const n = k === 1 ? dec.length : dec.length / k
    const label = (i: number): number => {
      if (k === 1) return p.classes[(dec[i] ?? 0) > 0 ? 1 : 0] as number
      let best = 0
      for (let c = 1; c < k; c++) if ((dec[i * k + c] ?? 0) > (dec[i * k + best] ?? 0)) best = c
      return p.classes[best] as number
    }
    return fromStorage(
      Float64Array.from({ length: n }, (_, i) => label(i)),
      [n],
      'float64',
    )
  }
  return {
    decisionFunction: decision,
    predict,
    score: (data) => {
      const predicted = toFlat(predict(data.x))
      const truth = toFlat(asTensor(data.y ?? []))
      return predicted.reduce((s, v, i) => s + (v === truth[i] ? 1 : 0), 0) / Math.max(1, predicted.length)
    },
    classes: p.classes,
    nFeaturesIn: d,
  }
}

/**
 * Linear support vector classification (scikit-learn `LinearSVC`, liblinear): `1/2 |w|^2 + C sum_i max(0, 1 - y_i (w.x_i +
 * b))^2` by default, with the squared hinge or the hinge, the L2 or L1 penalty, and the intercept penalised as a constant
 * feature. The solver is a slot: dual coordinate descent (default; one sweep per step, `solver/objective` the primal),
 * `owlqn` for the L1 penalty, or any solver of `smoothSolvers` on the squared-hinge L2 primal; they reach the same
 * weights (the L2 squared-hinge objective is strictly convex). More than two classes are one-versus-rest, one
 * problem after another, `solver/problem` saying which; the weights of every problem are the `solver/weights` tensor.
 *
 * @concept support-vector-machines
 * @reference R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang and C.-J. Lin, LIBLINEAR: a library for large linear classification (2008).
 * @index estimator Linear support vector classifier by dual coordinate descent or any smooth solver.
 * @example
 * ```ts
 * import { linearSvc } from 'omai-engine/svm'
 * import { toFlat } from 'omai-engine/tensor'
 * const fitted = linearSvc.with({ C: 10 }).fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] })
 * toFlat(fitted.predict([[0.2], [2.8]])) // => [0, 1]
 * ```
 */
export const linearSvc: IterativeEstimator<LinearSvcHyper, LinearSvmState, LinearSvcParams, LinearSvcVerbs> = iterative<
  LinearSvcHyper,
  LinearSvmState,
  LinearSvcParams,
  LinearSvcVerbs
>({
  name: 'svm/linear-svc',
  kind: 'classifier',
  defaults: {
    C: 1,
    loss: 'squaredHinge',
    penalty: 'l2',
    tol: 1e-4,
    maxIter: 1000,
    fitIntercept: true,
    interceptScaling: 1,
    classWeight: null,
    solver: solverConfigOf(linearSvcSolvers, { kind: 'dcd' }),
  },
  hyperSpace: (h) =>
    dict({
      C: real(1e-2, 1e3, { log: true, default: Math.min(Math.max(h.C, 1e-2), 1e3), label: 'C' }),
      loss: choice(['squaredHinge', 'hinge'] as const, { default: h.loss, label: 'loss' }),
      penalty: choice(['l2', 'l1'] as const, { default: h.penalty, label: 'penalty' }),
      solver: linearSvcSolvers.space,
    }),
  algorithm: (data, h) => {
    const { problems, design } = svcProblems(data, h)
    return inTurn<LinearSvcProblem>({
      name: 'linear-svc',
      problems,
      table: linearSvcSolvers,
      solver: svcSolverOf(h),
      maxIter: h.maxIter,
      width: design.width,
      primal: svcPrimal,
    })
  },
  maxSteps: (h, data) => {
    const k = new Set(targetVector(data.y ?? [], WHO_SVC)).size
    return h.maxIter * (k === 2 ? 1 : k)
  },
  params: (s, data, h) => {
    const m = matrixView(data.x, WHO_SVC)
    const classes = Array.from(new Set(targetVector(data.y ?? [], WHO_SVC, m.n))).sort((a, b) => a - b)
    const design = linearDesign(new Float64Array(0), 0, m.d, h.fitIntercept, h.interceptScaling)
    const parts = s.weights.map((w) => splitWeights(w, design))
    return {
      classes,
      coef: parts.map((q) => q.coef),
      intercept: parts.map((q) => q.intercept),
      converged: s.finished && !s.capped,
    }
  },
  verbs: (p) => svcVerbs(p, p.coef[0]?.length ?? 0),
})

// --- the regressor ------------------------------------------------------------------------------------------------

/** The hyperparameters of {@link linearSvr}. */
export interface LinearSvrHyper {
  readonly C: number
  /** The half-width of the tube inside which a residual costs nothing (default 0). */
  readonly epsilon: number
  /** The epsilon-insensitive loss (dual coordinate descent only, the default) or its square (liblinear's L2 loss). */
  readonly loss: LinearSvrLoss
  readonly tol: number
  readonly maxIter: number
  readonly fitIntercept: boolean
  readonly interceptScaling: number
  /** How the problem is fitted: a value of `linearSvrSolvers.space` (`dcd` by default). */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link linearSvr}. */
export type LinearSvrParams = {
  readonly coef: number[]
  readonly intercept: number
  readonly converged: boolean
}

/** The verbs of a fitted {@link linearSvr}. */
export interface LinearSvrVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The coefficient of determination on a labelled dataset. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

const WHO_SVR = 'linearSvr'

function svrProblem(data: LearnData, h: LinearSvrHyper): LinearSvrProblem {
  if (!(h.C > 0)) throw new OmaiError('value', `${WHO_SVR}: C must be positive`)
  if (!(h.epsilon >= 0)) throw new OmaiError('value', `${WHO_SVR}: epsilon must be non-negative`)
  if (!(h.interceptScaling > 0)) throw new OmaiError('value', `${WHO_SVR}: interceptScaling must be positive`)
  const design = designOf(data, WHO_SVR, h.fitIntercept, h.interceptScaling)
  const y = targetVector(data.y ?? [], WHO_SVR, design.n)
  const sampleWeights = sampleWeightsOf(data.weights, design.n, WHO_SVR)
  const rows = keep(design, sampleWeights)
  const problem: Omit<LinearSvrProblem, 'objective' | 'x0'> = {
    design: rows.length === design.n ? design : subDesign(design, rows),
    y: Float64Array.from(rows, (i) => y[i] ?? 0),
    C: h.C,
    weights: Float64Array.from(rows, (i) => sampleWeights[i] ?? 1),
    loss: h.loss,
    epsilon: h.epsilon,
    tol: h.tol,
    maxIter: h.maxIter,
  }
  return problem as LinearSvrProblem
}

/**
 * Linear support vector regression (scikit-learn `LinearSVR`, liblinear): `1/2 |w|^2 + C sum_i max(0, |w.x_i + b - y_i| -
 * epsilon)` (or its square), the intercept penalised as a constant feature. The solver is a slot: dual coordinate descent
 * (default; one sweep per step, with the dual objective and the gap on the tape), or any solver of `smoothSolvers` on
 * the smooth `squaredEpsilonInsensitive` primal; both reach the same weights.
 *
 * @concept support-vector-machines
 * @reference C.-H. Ho and C.-J. Lin, Large-scale linear support vector regression (2012).
 * @index estimator Linear support vector regressor by dual coordinate descent or any smooth solver.
 * @example
 * ```ts
 * import { linearSvr } from 'omai-engine/svm'
 * import { toFlat } from 'omai-engine/tensor'
 * const fitted = linearSvr.with({ C: 100, tol: 1e-8, maxIter: 5000 }).fit({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] })
 * toFlat(fitted.predict([[4]])).map((v) => Math.round(v)) // => [9]
 * ```
 */
export const linearSvr: IterativeEstimator<LinearSvrHyper, LinearSvmState, LinearSvrParams, LinearSvrVerbs> = iterative<
  LinearSvrHyper,
  LinearSvmState,
  LinearSvrParams,
  LinearSvrVerbs
>({
  name: 'svm/linear-svr',
  kind: 'regressor',
  defaults: {
    C: 1,
    epsilon: 0,
    loss: 'epsilonInsensitive',
    tol: 1e-4,
    maxIter: 1000,
    fitIntercept: true,
    interceptScaling: 1,
    solver: solverConfigOf(linearSvrSolvers, { kind: 'dcd' }),
  },
  hyperSpace: (h) =>
    dict({
      C: real(1e-2, 1e3, { log: true, default: Math.min(Math.max(h.C, 1e-2), 1e3), label: 'C' }),
      epsilon: real(0, 10, { default: Math.min(h.epsilon, 10), label: 'epsilon' }),
      loss: choice(['epsilonInsensitive', 'squaredEpsilonInsensitive'] as const, { default: h.loss, label: 'loss' }),
      solver: linearSvrSolvers.space,
    }),
  algorithm: (data, h) => {
    checkLinearSvr(WHO_SVR, linearSvrSolvers.modeOf(h.solver).name, h.loss)
    const problem = svrProblem(data, h)
    return inTurn<LinearSvrProblem>({
      name: 'linear-svr',
      problems: [problem],
      table: linearSvrSolvers,
      solver: h.solver,
      maxIter: h.maxIter,
      width: problem.design.width,
      primal: svrPrimal,
    })
  },
  maxSteps: (h) => h.maxIter,
  params: (s, data, h) => {
    const design = linearDesign(
      new Float64Array(0),
      0,
      matrixView(data.x, WHO_SVR).d,
      h.fitIntercept,
      h.interceptScaling,
    )
    const { coef, intercept } = splitWeights(s.weights[0] ?? [], design)
    return { coef, intercept, converged: s.finished && !s.capped }
  },
  verbs: (p) => {
    const d = p.coef.length
    const predict = (x: TensorLike): Tensor => {
      const m = matrixView(x, WHO_SVR, { nFeatures: d })
      const out = Float64Array.from({ length: m.n }, (_, i) => {
        let z = p.intercept
        for (let j = 0; j < d; j++) z += (p.coef[j] ?? 0) * (m.data[i * d + j] ?? 0)
        return z
      })
      return fromStorage(out, [m.n], 'float64')
    }
    return {
      predict,
      score: (data) => {
        const truth = toFlat(asTensor(data.y ?? []))
        const fit = toFlat(predict(data.x))
        const mean = truth.reduce((a, b) => a + b, 0) / Math.max(1, truth.length)
        let rss = 0
        let tss = 0
        for (const [i, v] of truth.entries()) {
          rss += (v - (fit[i] ?? 0)) ** 2
          tss += (v - mean) ** 2
        }
        return tss === 0 ? (rss === 0 ? 1 : 0) : 1 - rss / tss
      },
      nFeaturesIn: d,
    }
  },
})
