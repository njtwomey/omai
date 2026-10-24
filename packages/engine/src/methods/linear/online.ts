// TODO: averaged SGD (`average`), early stopping on a validation split, class weights, partial_fit, and the
// one-class SGD SVM.
import { WhiteboxError } from '../../lib/base/index.ts'
import { type CoefficientTable, coefficientTable, type Exposition, learningCurve } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  fitRunOptions,
  geometricSteps,
  type LearnData,
  type MatrixView,
  matrixView,
  type SurfaceStepsInput,
  targetVector,
} from '../../lib/learn/index.ts'
import { daxpy, ddot, dscal } from '../../lib/numerics/linalg/index.ts'
import { permutation } from '../../lib/random/index.ts'
import { bool, choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { linearPredictor, vectorTensor } from './_design.ts'
import {
  classifierTable,
  classifierVerbs,
  type FittedLinearClassifier,
  type LinearClassifierParams,
} from './classification.ts'

/** The per-sample losses of scikit-learn's SGD (classification losses take labels in {-1, +1}). */
export type SgdLoss =
  | 'hinge'
  | 'log_loss'
  | 'modified_huber'
  | 'squared_hinge'
  | 'perceptron'
  | 'squared_error'
  | 'huber'
  | 'epsilon_insensitive'
  | 'squared_epsilon_insensitive'

/** Learning-rate schedules: scikit-learn's, plus the passive-aggressive step sizes PA-I and PA-II. */
export type SgdLearningRate = 'optimal' | 'constant' | 'invscaling' | 'adaptive' | 'pa1' | 'pa2'

/** A per-sample loss and its derivative in the prediction p. */
export interface ScalarLoss {
  readonly loss: (y: number, p: number) => number
  readonly dloss: (y: number, p: number) => number
  /** Classification by margin (hinge family): passive-aggressive steps take the label's sign. */
  readonly hinge: boolean
}

/**
 * The scalar loss of scikit-learn's `_sgd_fast` by name.
 *
 * @param name - The loss.
 * @param epsilon - The width of the Huber and epsilon-insensitive losses.
 * @returns Its value and derivative.
 */
export function sgdLoss(name: SgdLoss, epsilon = 0.1): ScalarLoss {
  const hingeAt = (threshold: number): ScalarLoss => ({
    loss: (y, p) => (p * y <= threshold ? threshold - p * y : 0),
    dloss: (y, p) => (p * y <= threshold ? -y : 0),
    hinge: true,
  })
  switch (name) {
    case 'hinge':
      return hingeAt(1)
    case 'perceptron':
      return hingeAt(0)
    case 'squared_hinge':
      return {
        loss: (y, p) => Math.max(0, 1 - p * y) ** 2,
        dloss: (y, p) => (1 - p * y > 0 ? -2 * y * (1 - p * y) : 0),
        hinge: false,
      }
    case 'log_loss':
      return {
        loss: (y, p) => {
          const z = p * y
          return z > 18 ? Math.exp(-z) : z < -18 ? -z : Math.log1p(Math.exp(-z))
        },
        dloss: (y, p) => {
          const z = p * y
          return z > 18 ? -y * Math.exp(-z) : z < -18 ? -y : -y / (Math.exp(z) + 1)
        },
        hinge: false,
      }
    case 'modified_huber':
      return {
        loss: (y, p) => {
          const z = p * y
          return z >= 1 ? 0 : z >= -1 ? (1 - z) * (1 - z) : -4 * z
        },
        dloss: (y, p) => {
          const z = p * y
          return z >= 1 ? 0 : z >= -1 ? 2 * (1 - z) * -y : -4 * y
        },
        hinge: false,
      }
    case 'squared_error':
      return { loss: (y, p) => 0.5 * (p - y) * (p - y), dloss: (y, p) => p - y, hinge: false }
    case 'huber':
      return {
        loss: (y, p) => {
          const r = Math.abs(p - y)
          return r <= epsilon ? 0.5 * r * r : epsilon * r - 0.5 * epsilon * epsilon
        },
        dloss: (y, p) => {
          const r = p - y
          return Math.abs(r) <= epsilon ? r : r > 0 ? epsilon : -epsilon
        },
        hinge: false,
      }
    case 'epsilon_insensitive':
      return {
        loss: (y, p) => Math.max(0, Math.abs(y - p) - epsilon),
        dloss: (y, p) => (y - p > epsilon ? -1 : p - y > epsilon ? 1 : 0),
        hinge: false,
      }
    case 'squared_epsilon_insensitive':
      return {
        loss: (y, p) => Math.max(0, Math.abs(y - p) - epsilon) ** 2,
        dloss: (y, p) => {
          const z = y - p
          return z > epsilon ? -2 * (z - epsilon) : z < -epsilon ? 2 * (-z - epsilon) : 0
        },
        hinge: false,
      }
  }
}

/** The settings of plain SGD shared by the classifier, the regressor and the passive-aggressive models. */
export interface SgdSettings {
  readonly loss: SgdLoss
  readonly penalty: 'none' | 'l2' | 'l1' | 'elasticnet'
  /** Regularisation strength (default 1e-4). */
  readonly alpha: number
  /** The L1 share of the elastic-net penalty (default 0.15). */
  readonly l1Ratio: number
  readonly fitIntercept: boolean
  /** Epochs (default 1000). */
  readonly maxIter: number
  /** Stop when the epoch objective fails to improve by tol for `nIterNoChange` epochs (default 1e-3; null never). */
  readonly tol: number | null
  readonly nIterNoChange: number
  /** Shuffle the rows each epoch (default true). */
  readonly shuffle: boolean
  readonly learningRate: SgdLearningRate
  /** The initial rate (constant, invscaling, adaptive) or the aggressiveness C (pa1, pa2). */
  readonly eta0: number
  /** The exponent of invscaling, eta0 / t^powerT. */
  readonly powerT: number
  /** The Huber and epsilon-insensitive width (default 0.1). */
  readonly epsilon: number
  /** Learning-curve checkpoints within the first epoch (default 10; every later epoch adds one). */
  readonly checkpoints: number
}

/** One problem of plain SGD (a binary score of one-versus-rest, or a regression), updated in place. */
export interface SgdProblem {
  readonly w: Float64Array
  b: number
  readonly q: Float64Array
  u: number
  t: number
  eta: number
  sum: number
  best: number
  stall: number
  active: boolean
  epochs: number
}

/** The state of plain SGD: every problem, one epoch per step, and the learning curve so far. */
export interface SgdState {
  readonly problems: readonly SgdProblem[]
  readonly epoch: number
  readonly seen: number
  readonly curve: {
    readonly seen: number[]
    readonly progressive: number[]
    readonly training: number[]
    readonly rate: number[]
    /** coefficients[t][p]: problem p's coefficients then its intercept at checkpoint t (the trajectory of the fit). */
    readonly coefficients: number[][][]
  }
}

const MAX_DLOSS = 1e12

/**
 * Plain stochastic gradient descent as scikit-learn's `_plain_sgd` runs it, as a traced algorithm with one epoch per
 * step: for every row in turn the prediction, the step size from the schedule (optimal 1 / (alpha (t0 + t)),
 * constant, inverse scaling, adaptive, or the passive-aggressive PA-I and PA-II steps), weight decay for L2, the
 * gradient step and the cumulative L1 penalty of Tsuruoka et al. Several problems (one-versus-rest scores) run in
 * lock-step over the same rows, each stopping on its own when its epoch objective stalls. The state carries a
 * learning curve: the progressive (predict-then-learn) error over each window of rows and the training error at
 * each checkpoint.
 *
 * @param x - Rows.
 * @param targets - Per problem, the target of each row ({-1, +1} for classification).
 * @param settings - The SGD settings.
 * @param evaluate - The error of the current problems on row i (for the progressive error) and on all rows.
 * @returns The algorithm.
 */
export function plainSgdAlgorithm(
  x: MatrixView,
  targets: readonly Float64Array[],
  settings: SgdSettings,
  evaluate: {
    readonly row: (scores: readonly number[], i: number) => number
    readonly all: (problems: readonly SgdProblem[]) => number
  },
): Algorithm<SgdState> {
  const { n, d } = x
  const loss = sgdLoss(settings.loss, settings.epsilon)
  const penalty = settings.penalty
  const l1Ratio = penalty === 'l2' ? 0 : penalty === 'l1' ? 1 : settings.l1Ratio
  const alpha = settings.alpha
  const lr = settings.learningRate
  const pa = lr === 'pa1' || lr === 'pa2'
  if ((lr === 'constant' || lr === 'invscaling' || lr === 'adaptive' || pa) && !(settings.eta0 > 0))
    throw new WhiteboxError('value', `sgd: eta0 must be positive for the ${lr} schedule`)
  if (lr === 'optimal' && !(alpha > 0)) throw new WhiteboxError('value', 'sgd: the optimal schedule needs alpha > 0')
  const typw = Math.sqrt(1 / Math.sqrt(alpha))
  const optimalInit = lr === 'optimal' ? 1 / ((typw / Math.max(1, loss.dloss(1, -typw))) * alpha) : 0
  const tol = settings.tol ?? Number.NEGATIVE_INFINITY
  const firstEpochEvery = Math.max(1, Math.round(n / Math.max(1, settings.checkpoints)))
  const sqnorms = Float64Array.from({ length: n }, (_, i) => {
    const row = x.data.subarray(i * d, (i + 1) * d)
    return ddot(row, row)
  })
  return {
    name: pa ? 'passive-aggressive' : 'sgd',
    config: { loss: settings.loss, penalty, alpha, learningRate: lr, eta0: settings.eta0, problems: targets.length },
    schema: channels({
      metrics: {
        objective: { description: 'mean loss (with penalty) over the epoch', direction: 'lower' },
        progressive: { description: 'progressive (predict-then-learn) error over the epoch', direction: 'lower' },
        eta: { description: 'the step size at the end of the epoch' },
      },
    }),
    init: () => {
      const problems: SgdProblem[] = targets.map(() => ({
        w: new Float64Array(d),
        b: 0,
        q: new Float64Array(d),
        u: 0,
        t: 1,
        eta: settings.eta0,
        sum: 0,
        best: Number.POSITIVE_INFINITY,
        stall: 0,
        active: true,
        epochs: 0,
      }))
      return {
        problems,
        epoch: 0,
        seen: 0,
        curve: { seen: [], progressive: [], training: [], rate: [], coefficients: [] },
      }
    },
    step: (s, ctx) => {
      // Copy the problems so that every state stays a value.
      const problems: SgdProblem[] = s.problems.map((p) => ({
        ...p,
        w: Float64Array.from(p.w),
        q: Float64Array.from(p.q),
      }))
      const order = settings.shuffle
        ? Array.from(toFlat(permutation(ctx.key, n)))
        : Array.from({ length: n }, (_, i) => i)
      const curve = {
        seen: [...s.curve.seen],
        progressive: [...s.curve.progressive],
        training: [...s.curve.training],
        rate: [...s.curve.rate],
        coefficients: [...s.curve.coefficients],
      }
      let seen = s.seen
      let windowErr = 0
      let windowRows = 0
      for (const p of problems) p.sum = 0
      const every = s.epoch === 0 ? firstEpochEvery : n
      for (const [j, i] of order.entries()) {
        const row = x.data.subarray(i * d, (i + 1) * d)
        const scores = problems.map((p) => ddot(row, p.w) + p.b)
        const err = evaluate.row(scores, i)
        windowErr += err
        windowRows++
        for (const [r, p] of problems.entries()) {
          if (!p.active) continue
          const y = targets[r]?.[i] ?? 0
          const pred = scores[r] ?? 0
          if (lr === 'optimal') p.eta = 1 / (alpha * (optimalInit + p.t - 1))
          else if (lr === 'invscaling') p.eta = settings.eta0 / p.t ** settings.powerT
          const value = loss.loss(y, pred)
          p.sum += value
          if (!pa && penalty !== 'none') {
            let l1 = 0
            for (const v of p.w) l1 += Math.abs(v)
            p.sum += alpha * ((1 - l1Ratio) * 0.5 * ddot(p.w, p.w) + l1Ratio * l1)
          }
          let update: number
          if (lr === 'pa1') {
            const sq = sqnorms[i] ?? 0
            if (sq === 0) continue
            update = Math.min(settings.eta0, value / sq)
          } else if (lr === 'pa2') update = value / ((sqnorms[i] ?? 0) + 0.5 / settings.eta0)
          else update = -p.eta * Math.min(MAX_DLOSS, Math.max(-MAX_DLOSS, loss.dloss(y, pred)))
          if (pa) {
            if (loss.hinge) update *= y
            else if (y - pred < 0) update *= -1
          }
          if (penalty === 'l2' || penalty === 'elasticnet')
            p.w.set(dscal(Math.max(0, 1 - (1 - l1Ratio) * p.eta * alpha), p.w))
          if (update !== 0) p.w.set(daxpy(update, row, p.w))
          if (settings.fitIntercept && update !== 0) p.b += update
          if (penalty === 'l1' || penalty === 'elasticnet') {
            p.u += l1Ratio * p.eta * alpha
            for (let k = 0; k < d; k++) {
              const z = p.w[k] ?? 0
              const qk = p.q[k] ?? 0
              if (z > 0) p.w[k] = Math.max(0, z - (p.u + qk))
              else if (z < 0) p.w[k] = Math.min(0, z + (p.u - qk))
              p.q[k] = qk + ((p.w[k] ?? 0) - z)
            }
          }
          p.t++
        }
        seen++
        if ((j + 1) % every === 0 || j === n - 1) {
          curve.seen.push(seen)
          curve.progressive.push(windowErr / windowRows)
          curve.training.push(evaluate.all(problems))
          curve.rate.push(problems[0]?.eta ?? 0)
          curve.coefficients.push(problems.map((p) => [...p.w, p.b]))
          windowErr = 0
          windowRows = 0
        }
      }
      for (const p of problems) {
        if (!p.active) continue
        p.epochs++
        const objective = p.sum / n
        if (tol > Number.NEGATIVE_INFINITY && objective > p.best - tol) p.stall++
        else p.stall = 0
        if (objective < p.best) p.best = objective
        if (p.stall >= settings.nIterNoChange) {
          if (lr === 'adaptive' && p.eta > 1e-6) {
            p.eta /= 5
            p.stall = 0
          } else p.active = false
        }
      }
      return { problems, epoch: s.epoch + 1, seen, curve }
    },
    done: (s) => (s.problems.every((p) => !p.active) ? 'converged' : undefined),
    observe: (s) => {
      const p = s.problems[0]
      const last = s.curve.progressive.length - 1
      return {
        metrics:
          p === undefined || s.epoch === 0
            ? {}
            : { objective: p.sum / n, progressive: s.curve.progressive[last] ?? 0, eta: p.eta },
      }
    },
  }
}

/** Hyperparameters of {@link sgdClassifier} (scikit-learn's names and defaults). */
export type SgdClassifierHyper = SgdSettings

/** Hyperparameters of {@link sgdRegressor}. */
export type SgdRegressorHyper = SgdSettings

/** Parameters of a fitted SGD regressor. */
export type SgdRegressorParams = {
  readonly coef: number[]
  readonly intercept: number
  /** Epochs run. */
  readonly nIter: number
  /** Updates made (scikit-learn's t_ - 1). */
  readonly t: number
}

/** Verbs of a fitted SGD regressor. */
export interface SgdRegressorVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** A fitted SGD regressor. */
export type FittedSgdRegressor = Fitted<SgdRegressorParams> & SgdRegressorVerbs

const classifierDefaults = (h: Partial<SgdSettings>): SgdSettings => ({
  loss: h.loss ?? 'hinge',
  penalty: h.penalty ?? 'l2',
  alpha: h.alpha ?? 1e-4,
  l1Ratio: h.l1Ratio ?? 0.15,
  fitIntercept: h.fitIntercept ?? true,
  maxIter: h.maxIter ?? 1000,
  tol: h.tol === undefined ? 1e-3 : h.tol,
  nIterNoChange: h.nIterNoChange ?? 5,
  shuffle: h.shuffle ?? true,
  learningRate: h.learningRate ?? 'optimal',
  eta0: h.eta0 ?? 0.01,
  powerT: h.powerT ?? 0.5,
  epsilon: h.epsilon ?? 0.1,
  checkpoints: h.checkpoints ?? 10,
})

const sgdSpace = (h: SgdSettings, losses: readonly SgdLoss[]): Space =>
  dict({
    loss: choice([...losses], {
      label: 'loss',
      description:
        'The loss each update descends: hinge, log, modified Huber, squared hinge or perceptron (classifiers), squared error, Huber or ε-insensitive (regressors).',
      default: h.loss,
    }),
    penalty: choice(['none', 'l2', 'l1', 'elasticnet'], {
      label: 'penalty',
      description: 'The regularisation on the coefficients: none, L2, L1 or elastic net.',
      default: h.penalty,
    }),
    alpha: real(1e-6, 1, {
      label: 'α (penalty)',
      description: 'The regularisation strength (also sets the optimal learning rate).',
      log: true,
      default: h.alpha,
    }),
    learningRate: choice(['optimal', 'constant', 'invscaling', 'adaptive'], {
      default: h.learningRate,
      label: 'learning-rate schedule',
    }),
    eta0: real(1e-4, 10, { log: true, default: h.eta0, label: 'initial rate' }),
    maxIter: int(1, 1000, { default: h.maxIter, label: 'epochs' }),
    shuffle: bool({ label: 'shuffle', description: 'Shuffle the rows before each epoch.', default: h.shuffle }),
  })

const paSpace = (h: SgdSettings, losses: readonly SgdLoss[]): Space =>
  dict({
    eta0: real(1e-3, 100, { log: true, default: h.eta0, label: 'C (aggressiveness)' }),
    learningRate: choice(['pa1', 'pa2'], { default: h.learningRate, label: 'PA-I or PA-II' }),
    loss: choice([...losses], {
      label: 'loss',
      description: 'Hinge (classifiers) or ε-insensitive (regressors) loss, or their squares.',
      default: h.loss,
    }),
    maxIter: int(1, 1000, { default: h.maxIter, label: 'epochs' }),
    shuffle: bool({ label: 'shuffle', description: 'Shuffle the rows before each epoch.', default: h.shuffle }),
  })

/** A coefficient table with the first problem's coefficients (then intercept) at each checkpoint of the run. */
function withTrajectory(table: CoefficientTable, state: SgdState): CoefficientTable {
  const values = state.curve.coefficients.map((c) => c[0] ?? [])
  if (values.length < 2 || values.some((v) => v.length !== table.names.length)) return table
  return coefficientTable({ ...table, trajectories: { resource: 'samples seen', at: [...state.curve.seen], values } })
}

/** The learning curve of an SGD run: progressive and training error per checkpoint. */
function sgdCurve(state: SgdState, what: string): Exposition {
  return learningCurve({
    sizes: state.curve.seen,
    train: state.curve.progressive,
    validation: state.curve.training,
    resource: 'samples seen',
    labels: { train: `progressive ${what}`, validation: `training ${what}` },
  })
}

function fitSgdClassifier(
  name: string,
  data: LearnData,
  h: SgdSettings,
  options: FitOptions,
): { params: LinearClassifierParams; exposition: Exposition[]; state: SgdState } {
  const x = matrixView(data.x, name)
  const { classes: labels, codes } = encodeLabels(data.y ?? [], name, x.n)
  const classes = [...labels]
  if (classes.length < 2) throw new WhiteboxError('value', `${name}: needs at least two classes`)
  const rows = classes.length === 2 ? 1 : classes.length
  const targets = Array.from({ length: rows }, (_, r) =>
    Float64Array.from(codes, (c) => (rows === 1 ? (c === 1 ? 1 : -1) : c === r ? 1 : -1)),
  )
  const predictCode = (scores: readonly number[]): number => {
    if (rows === 1) return (scores[0] ?? 0) > 0 ? 1 : 0
    let best = 0
    for (let r = 1; r < rows; r++) if ((scores[r] ?? 0) > (scores[best] ?? 0)) best = r
    return best
  }
  const evaluate = {
    row: (scores: readonly number[], i: number) => (predictCode(scores) === codes[i] ? 0 : 1),
    all: (problems: readonly SgdProblem[]) => {
      let wrong = 0
      for (let i = 0; i < x.n; i++) {
        const row = x.data.subarray(i * x.d, (i + 1) * x.d)
        if (predictCode(problems.map((p) => ddot(row, p.w) + p.b)) !== codes[i]) wrong++
      }
      return wrong / x.n
    },
  }
  const out = runAlgorithm(plainSgdAlgorithm(x, targets, h, evaluate), {
    ...fitRunOptions(options, name),
    maxSteps: h.maxIter,
  })
  const s = out.state
  const params: LinearClassifierParams = {
    classes,
    coef: s.problems.map((p) => Array.from(p.w)),
    intercept: s.problems.map((p) => p.b),
    link: h.loss === 'log_loss' ? (rows === 1 ? 'binary' : 'ovr') : 'none',
    iterations: Math.max(...s.problems.map((p) => p.epochs)),
  }
  const table = classifierTable(params, featureNamesOf(data, x.d), classNamesOf(data, classes))
  // One score row (two classes): the coefficients' trajectory over the samples seen.
  return { params, exposition: [sgdCurve(s, 'error'), rows === 1 ? withTrajectory(table, s) : table], state: s }
}

function classifierEstimator(
  name: string,
  h: SgdSettings,
  space: (h: SgdSettings) => Space,
): ExplainedEstimator<SgdSettings, FittedLinearClassifier> {
  const est: ExplainedEstimator<SgdSettings, FittedLinearClassifier> = defineExplainedEstimator(
    {
      name,
      kind: 'classifier',
      hyperSpace: space,
      fit: (data: LearnData, hy: SgdSettings, options: FitOptions) => {
        const r = fitSgdClassifier(name, data, hy, options)
        const d = r.params.coef[0]?.length ?? 0
        const fitted = est.revive(r.params)
        // The surface at up to 40 of the run's checkpoints (geometric in the samples seen), the rows the
        // model then misclassifies highlighted, beside the progressive error.
        const curve = r.state.curve
        const record = options.explain !== false && curve.coefficients.length > 1
        const chosen = record ? geometricSteps(curve.coefficients.length - 1, 40) : []
        const steps: SurfaceStepsInput | undefined = record
          ? {
              models: chosen.map((t) => {
                const c = curve.coefficients[t] ?? []
                return t === curve.coefficients.length - 1
                  ? fitted
                  : est.revive({ ...r.params, coef: c.map((w) => w.slice(0, d)), intercept: c.map((w) => w[d] ?? 0) })
              }),
              iterations: chosen.map((t) => curve.seen[t] ?? t),
              objective: chosen.map((t) => curve.progressive[t] ?? Number.NaN),
              highlight: 'misclassified',
            }
          : undefined
        const surface =
          d === 2
            ? [
                decisionSurfaceOf(fitted, data.x, {
                  featureNames: featureNamesOf(data, d),
                  classNames: classNamesOf(data, r.params.classes),
                  ...(data.y === undefined ? {} : { y: data.y }),
                  ...(steps === undefined
                    ? {}
                    : {
                        steps,
                        stepName: 'samples seen',
                        objective: {
                          name: 'progressive error',
                          iterations: [...curve.seen],
                          values: [...curve.progressive],
                        },
                      }),
                }),
              ]
            : []
        return { params: r.params, exposition: [...r.exposition, ...surface] }
      },
      methods: (p: LinearClassifierParams) => classifierVerbs(p, name, p.link !== 'none'),
    },
    h,
  )
  return est
}

/**
 * A linear classifier trained by stochastic gradient descent (sklearn.linear_model.SGDClassifier): hinge (a linear
 * SVM), log loss (logistic regression, with probabilities), modified Huber, squared hinge or the perceptron loss; L2,
 * L1 or elastic-net penalties; scikit-learn's learning-rate schedules; one-versus-rest for several classes. Trained by
 * {@link plainSgdAlgorithm}, one traced epoch per step. Exposition: a `learningCurve` of the progressive error (each
 * row predicted before it is learned) and the training error against the samples seen, the coefficient table and, for
 * two features, the decision surface.
 *
 * @param hyper - Loss, penalty, alpha, l1Ratio, intercept, epochs, tolerance, shuffling and the schedule.
 * @returns The estimator.
 */
export function sgdClassifier(
  hyper: Partial<SgdClassifierHyper> = {},
): ExplainedEstimator<SgdClassifierHyper, FittedLinearClassifier> {
  return classifierEstimator('sgd-classifier', classifierDefaults(hyper), (h) =>
    sgdSpace(h, ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']),
  )
}

/**
 * The online perceptron: {@link sgdClassifier} with the perceptron loss, a constant rate of 1 and no penalty (the
 * configuration scikit-learn's Perceptron is), so its learning curve shows the mistakes made row by row.
 *
 * @param hyper - As {@link sgdClassifier} (loss, rate and penalty preset).
 * @returns The estimator.
 */
export function onlinePerceptron(
  hyper: Partial<SgdClassifierHyper> = {},
): ExplainedEstimator<SgdClassifierHyper, FittedLinearClassifier> {
  return classifierEstimator(
    'online-perceptron',
    classifierDefaults({ loss: 'perceptron', learningRate: 'constant', eta0: 1, penalty: 'none', ...hyper }),
    (h) =>
      dict({
        eta0: real(1e-3, 10, { log: true, default: h.eta0, label: 'learning rate' }),
        maxIter: int(1, 1000, { default: h.maxIter, label: 'epochs' }),
        shuffle: bool({ label: 'shuffle', description: 'Shuffle the rows before each epoch.', default: h.shuffle }),
      }),
  )
}

/**
 * The passive-aggressive classifier (Crammer et al. 2006; sklearn.linear_model.PassiveAggressiveClassifier): on each
 * row, stay put if the margin is at least one (passive), else move just far enough to fix it, capped by C (PA-I,
 * `pa1`) or softened by C (PA-II, `pa2`). Exposition as {@link sgdClassifier}.
 *
 * @param hyper - C as `eta0` (default 1), the PA variant as `learningRate`, epochs, tolerance and shuffling.
 * @returns The estimator.
 */
export function passiveAggressiveClassifier(
  hyper: Partial<SgdClassifierHyper> = {},
): ExplainedEstimator<SgdClassifierHyper, FittedLinearClassifier> {
  return classifierEstimator(
    'passive-aggressive-classifier',
    classifierDefaults({ eta0: 1, learningRate: 'pa1', ...hyper, loss: 'hinge', penalty: 'none' }),
    (h) => paSpace(h, ['hinge']),
  )
}

function regressorEstimator(
  name: string,
  h: SgdSettings,
  space: (h: SgdSettings) => Space,
): ExplainedEstimator<SgdSettings, FittedSgdRegressor> {
  const verbs = (p: SgdRegressorParams): SgdRegressorVerbs => ({
    predict: (x) => vectorTensor(linearPredictor(x, p.coef, p.intercept, name)),
    nFeaturesIn: p.coef.length,
  })
  return defineExplainedEstimator<SgdSettings, SgdRegressorParams, SgdRegressorVerbs>(
    {
      name,
      kind: 'regressor',
      hyperSpace: space,
      fit: (data, hy, options) => {
        const x = matrixView(data.x, name)
        const y = targetVector(data.y ?? [], name, x.n)
        const sq = (scores: readonly number[], i: number) => ((scores[0] ?? 0) - (y[i] ?? 0)) ** 2
        const evaluate = {
          row: sq,
          all: (problems: readonly SgdProblem[]) => {
            const p = problems[0] as SgdProblem
            let s = 0
            for (let i = 0; i < x.n; i++) s += sq([ddot(x.data.subarray(i * x.d, (i + 1) * x.d), p.w) + p.b], i)
            return s / x.n
          },
        }
        const out = runAlgorithm(plainSgdAlgorithm(x, [y], hy, evaluate), {
          ...fitRunOptions(options, name),
          maxSteps: hy.maxIter,
        })
        const p = out.state.problems[0] as SgdProblem
        const params: SgdRegressorParams = { coef: Array.from(p.w), intercept: p.b, nIter: p.epochs, t: p.t - 1 }
        const names = featureNamesOf(data, x.d)
        return {
          params,
          exposition: [
            sgdCurve(out.state, 'squared error'),
            withTrajectory(
              coefficientTable({ names: [...names, 'intercept'], estimates: [...params.coef, params.intercept] }),
              out.state,
            ),
          ],
        }
      },
      methods: verbs,
    },
    h,
  )
}

/**
 * A linear regressor trained by stochastic gradient descent (sklearn.linear_model.SGDRegressor): squared error, Huber
 * or (squared) epsilon-insensitive losses, the penalties and schedules of {@link sgdClassifier} (default inverse
 * scaling, eta0 / t^0.25). Exposition: the learning curve (progressive and training squared error against samples
 * seen) and the coefficient table.
 *
 * @param hyper - As {@link sgdClassifier}, with regression losses.
 * @returns The estimator.
 */
export function sgdRegressor(
  hyper: Partial<SgdRegressorHyper> = {},
): ExplainedEstimator<SgdRegressorHyper, FittedSgdRegressor> {
  return regressorEstimator(
    'sgd-regressor',
    classifierDefaults({ loss: 'squared_error', learningRate: 'invscaling', powerT: 0.25, ...hyper }),
    (h) => sgdSpace(h, ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
  )
}

/**
 * The passive-aggressive regressor (sklearn.linear_model.PassiveAggressiveRegressor): epsilon-insensitive steps of
 * PA-I or PA-II. Exposition as {@link sgdRegressor}.
 *
 * @param hyper - C as `eta0` (default 1), the variant, epsilon, epochs and shuffling.
 * @returns The estimator.
 */
export function passiveAggressiveRegressor(
  hyper: Partial<SgdRegressorHyper> = {},
): ExplainedEstimator<SgdRegressorHyper, FittedSgdRegressor> {
  return regressorEstimator(
    'passive-aggressive-regressor',
    classifierDefaults({ eta0: 1, learningRate: 'pa1', ...hyper, loss: 'epsilon_insensitive', penalty: 'none' }),
    (h) => paSpace(h, ['epsilon_insensitive']),
  )
}

/** {@link sgdClassifier} as a class. */
export class SGDClassifier extends ExplainedEstimatorClass<SgdClassifierHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link sgdClassifier}. */
  constructor(hyper: Partial<SgdClassifierHyper> = {}) {
    super(sgdClassifier(hyper))
  }
}

/** {@link sgdRegressor} as a class. */
export class SGDRegressor extends ExplainedEstimatorClass<SgdRegressorHyper, FittedSgdRegressor> {
  /** @param hyper - As {@link sgdRegressor}. */
  constructor(hyper: Partial<SgdRegressorHyper> = {}) {
    super(sgdRegressor(hyper))
  }
}

/** {@link passiveAggressiveClassifier} as a class. */
export class PassiveAggressiveClassifier extends ExplainedEstimatorClass<SgdClassifierHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link passiveAggressiveClassifier}. */
  constructor(hyper: Partial<SgdClassifierHyper> = {}) {
    super(passiveAggressiveClassifier(hyper))
  }
}

/** {@link passiveAggressiveRegressor} as a class. */
export class PassiveAggressiveRegressor extends ExplainedEstimatorClass<SgdRegressorHyper, FittedSgdRegressor> {
  /** @param hyper - As {@link passiveAggressiveRegressor}. */
  constructor(hyper: Partial<SgdRegressorHyper> = {}) {
    super(passiveAggressiveRegressor(hyper))
  }
}
