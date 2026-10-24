import { type Json, linspaceNumbers, WhiteboxError } from '../base/index.ts'
import { resolveKey, resolvePolicy, resolveTracer } from '../context/index.ts'
import {
  type DecisionSurface,
  decisionSurface,
  type Explained,
  type Exposition,
  type InputRows,
  inputRows,
  type SurfaceStep,
} from '../exposition/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Algorithm, RunOptions } from '../trace/index.ts'
import {
  defineEstimator,
  type Estimator,
  EstimatorClass,
  type EstimatorSpec,
  type FitOptions,
  type Fitted,
  type FittedFacts,
  type Hyper,
  type LearnData,
  type Verbs,
} from './estimator.ts'
import { matrixView } from './validation.ts'

/** What an explained estimator's `fit` learns: the parameters and the exposition records explaining how. */
export interface ExplainedFit<P extends Json> {
  readonly params: P
  readonly exposition: readonly Exposition[]
}

/** The spec of {@link defineExplainedEstimator}: an estimator spec whose `fit` also returns exposition records. */
export interface ExplainedSpec<H extends Hyper, P extends Json, V extends Verbs & FittedFacts>
  extends Omit<EstimatorSpec<H, P, V>, 'fit'> {
  fit(data: LearnData, hyper: H, options: FitOptions): ExplainedFit<P>
}

/** An estimator whose fitted values carry their exposition records (empty after `revive`). */
export type ExplainedEstimator<H extends Hyper, F extends Fitted> = Estimator<H, Explained<F>>

/** The fitted value of an explained estimator with parameters P and verbs V. */
export type ExplainedFitted<P extends Json = Json, V = object> = Explained<Fitted<P> & V>

/**
 * Build an estimator whose fitted values carry exposition records (docs/structure.md 5.0): `fit` returns the fitted
 * value of {@link defineEstimator} plus `exposition`, the plain data a view or a sentence of explanation needs (paths,
 * trees, stages, margins). Revived values have no records (only the parameters are serialised).
 *
 * @param spec - As for {@link defineEstimator}, with `fit` returning `{ params, exposition }`.
 * @param hyper - The hyperparameters.
 * @returns The estimator.
 * @concept estimator-protocol
 * @example
 * ```ts
 * import { defineExplainedEstimator } from 'whitebox-engine/learn'
 * import { coefficientTable } from 'whitebox-engine/exposition'
 * import { full, item, mean } from 'whitebox-engine/tensor'
 * const meanRegressor = defineExplainedEstimator(
 *   {
 *     name: 'mean',
 *     kind: 'regressor',
 *     fit: (data) => {
 *       const m = item(mean(data.y))
 *       return { params: { mean: m }, exposition: [coefficientTable({ names: ['mean'], estimates: [m] })] }
 *     },
 *     methods: (p) => ({ predict: (x) => full([x.length], p.mean) }),
 *   },
 *   {},
 * )
 * const fitted = meanRegressor.fit({ x: [[0], [1]], y: [1, 3] })
 * fitted.exposition[0]?.kind // => 'coefficientTable'
 * meanRegressor.revive({ mean: 4 }).exposition // => []
 * ```
 */
export function defineExplainedEstimator<H extends Hyper, P extends Json, V extends Verbs & FittedFacts>(
  spec: ExplainedSpec<H, P, V>,
  hyper: H,
): ExplainedEstimator<H, Fitted<P> & V> {
  const { fit: explainedFit, fitTransform: crossFit, ...rest } = spec
  const base = defineEstimator<H, P, V>(
    { ...rest, fit: (data, h, options) => explainedFit(data, h, options).params },
    hyper,
  )
  const fit = (data: LearnData, options: FitOptions = {}): Explained<Fitted<P> & V> => {
    const r = explainedFit(data, hyper, options)
    return { ...base.revive(r.params), exposition: r.exposition }
  }
  return {
    name: base.name,
    kind: base.kind,
    hyper,
    ...(base.hyperSpace === undefined ? {} : { hyperSpace: base.hyperSpace }),
    ...(crossFit === undefined
      ? {}
      : {
          fitTransform: (data: LearnData, options?: FitOptions) => {
            const fitted = fit(data, options)
            return { fitted, transformed: crossFit(fitted, data) }
          },
        }),
    fit,
    revive: (params) => ({ ...base.revive(params as P), exposition: [] }),
    with: (changes) => defineExplainedEstimator(spec, spec.update?.(hyper, changes) ?? ({ ...hyper, ...changes } as H)),
  }
}

/**
 * The base of class wrappers for explained estimators: an {@link EstimatorClass} that also exposes the latest fit's
 * exposition records.
 *
 * @concept estimator-protocol
 * @example
 * ```ts
 * import { defineExplainedEstimator, ExplainedEstimatorClass } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const zero = defineExplainedEstimator(
 *   { name: 'zero', kind: 'regressor', fit: () => ({ params: {}, exposition: [] }), methods: () => ({ predict: (x) => full([x.length], 0) }) },
 *   {},
 * )
 * new ExplainedEstimatorClass(zero).fit([[1]], [2]).exposition // => []
 * ```
 */
export class ExplainedEstimatorClass<H extends Hyper = Hyper, F extends Fitted = Fitted> extends EstimatorClass<
  H,
  Explained<F>
> {
  /** The exposition records of the latest fit. */
  get exposition(): readonly Exposition[] {
    return this.model.exposition
  }
}

/** The rows of a surface's training data drawn beside it: the default cap on how many rows a record carries. */
export const ROWS_LIMIT = 5000

/** The recorded steps of training, as fitted models, for {@link decisionSurfaceOf}. */
export interface SurfaceStepsInput {
  /** The model after each recorded step (the last one the fitted model). */
  readonly models: readonly Fitted[]
  /** The iteration each model was taken after. */
  readonly iterations: readonly number[]
  /** The training objective at each step. */
  readonly objective?: readonly number[]
  /** Rows that matter at each step (support vectors, the rows just seen), or the rows the model then misclassifies. */
  readonly highlight?: readonly (readonly number[])[] | 'misclassified'
  /** A weight per row at each step (sample weights, residuals). */
  readonly rowWeights?: readonly (readonly number[])[]
  /** What the highlighted rows are (default 'misclassified' for those, else none). */
  readonly highlightName?: string
}

/**
 * The recorded steps of a model built stage by stage (boosting, forests): the model with its first m members at up to
 * `limit` stage counts spaced geometrically from 1 to all of them, with what each stage did.
 *
 * @param stages - The number of members.
 * @param modelAt - The model with its first m members (m from 1 to `stages`; the last is the fitted model).
 * @param extras - Per stage (index m - 1): the objective and the row weights the member was fitted to; and whether to
 * highlight the rows the model then misclassifies.
 * @param limit - The most steps (default 40).
 * @returns The steps, or `undefined` for fewer than two stages.
 * @example
 * ```ts
 * import { stagedSteps } from 'whitebox-engine/learn'
 * const model = { estimator: 'm', kind: 'regressor', params: {} }
 * stagedSteps(100, () => model, {}, 4)?.iterations // => [1, 5, 22, 100]
 * ```
 */
export function stagedSteps(
  stages: number,
  modelAt: (stages: number) => Fitted,
  extras: {
    readonly objective?: readonly number[]
    readonly rowWeights?: readonly (readonly number[])[]
    readonly misclassified?: boolean
  } = {},
  limit = 40,
): SurfaceStepsInput | undefined {
  if (stages < 2) return undefined
  const counts = geometricSteps(stages - 1, limit).map((t) => t + 1)
  const objective = extras.objective
  const weights = extras.rowWeights
  return {
    models: counts.map((m) => modelAt(m)),
    iterations: counts,
    ...(objective === undefined || objective.length < stages
      ? {}
      : { objective: counts.map((m) => objective[m - 1] ?? 0) }),
    ...(weights === undefined || weights.length < stages
      ? {}
      : { rowWeights: counts.map((m) => weights[m - 1] ?? []) }),
    ...(extras.misclassified === true ? { highlight: 'misclassified' as const } : {}),
  }
}

/** Options of {@link decisionSurfaceOf}. */
export interface DecisionSurfaceOptions {
  /** The two feature columns spanning the grid (default [0, 1]); other features are held at their means. */
  readonly features?: readonly [number, number]
  /** Grid points per axis (default 30). */
  readonly resolution?: number
  /** Padding beyond the data range, as a fraction of the range (default 0.1). */
  readonly padding?: number
  /** Feature names (default `x0, x1, ...`). */
  readonly featureNames?: readonly string[]
  /** Class names (default the fitted classes as strings). */
  readonly classNames?: readonly string[]
  /** The training targets, to label the rows the record carries. */
  readonly y?: TensorLike
  /** The most rows the record carries (default {@link ROWS_LIMIT}; 0 for none). */
  readonly rowsLimit?: number
  /** Levels of a binary decision function to mark besides 0 (an SVM's margins, -1 and 1). */
  readonly levels?: readonly number[]
  /** The model at recorded steps of training, to record the surface at each. */
  readonly steps?: SurfaceStepsInput
  /** What a step is ('SMO step', 'IRLS iteration', 'epoch', 'stage'). */
  readonly stepName?: string
  /** The objective at every recorded iteration, with its name (finer than the steps). */
  readonly objective?: {
    readonly name: string
    readonly iterations: readonly number[]
    readonly values: readonly number[]
    readonly lowerIsBetter?: boolean
  }
}

/** The fields of a surface for one model on a grid. */
type SurfaceFields = Pick<DecisionSurface, 'labels' | 'probabilities' | 'decision' | 'values'>

/**
 * The input rows of a record: two columns of a feature matrix with their names and, given classes, the rows' class
 * indices; none beyond the row limit.
 *
 * @param x - The features `[n, d]`.
 * @param options - The two columns (default the first two; one feature is paired with zeros), names, targets, the
 * fitted classes and class names, and the row limit.
 * @returns The rows, or `undefined` when there are more than the limit.
 * @example
 * ```ts
 * import { inputRowsOf } from 'whitebox-engine/learn'
 * inputRowsOf([[1, 2, 3], [4, 5, 6]], { y: [7, 9], classes: [7, 9] })?.labels // => [0, 1]
 * inputRowsOf([[1, 2], [3, 4]], { limit: 1 }) // => undefined
 * ```
 */
export function inputRowsOf(
  x: TensorLike,
  options: {
    readonly columns?: readonly [number, number]
    readonly featureNames?: readonly string[]
    readonly y?: TensorLike
    readonly classes?: readonly number[]
    readonly classNames?: readonly string[]
    readonly limit?: number
  } = {},
): InputRows | undefined {
  const m = matrixView(x, 'inputRowsOf', { allowNaN: true })
  if (m.n > (options.limit ?? ROWS_LIMIT)) return undefined
  const [a, b] = options.columns ?? [0, Math.min(1, m.d - 1)]
  const single = m.d === 1
  const points = Array.from({ length: m.n }, (_, i) => [
    m.data[i * m.d + a] ?? 0,
    single ? 0 : (m.data[i * m.d + b] ?? 0),
  ])
  const names = options.featureNames
  const features: [string, string] = [names?.[a] ?? `x${a}`, single ? '' : (names?.[b] ?? `x${b}`)]
  let labels: number[] | undefined
  if (options.y !== undefined && options.classes !== undefined) {
    const index = new Map(options.classes.map((c, i) => [c, i]))
    const y = toFlat(asTensor(options.y))
    if (y.length === m.n) labels = Array.from(y, (v) => index.get(v) ?? -1)
  }
  const classes = options.classNames ?? options.classes?.map(String)
  return inputRows({
    points,
    features,
    ...(labels === undefined ? {} : { labels }),
    ...(labels === undefined || classes === undefined ? {} : { classes }),
  })
}

/**
 * At most `limit` of the steps 0 to `total`, spaced geometrically (dense early, where training changes fastest, sparse
 * late), always with the first and the last: the steps a method records a surface or a snapshot at.
 *
 * @param total - The last step.
 * @param limit - The most steps kept (default 40).
 * @returns The steps, increasing.
 * @example
 * ```ts
 * import { geometricSteps } from 'whitebox-engine/learn'
 * geometricSteps(3) // => [0, 1, 2, 3]
 * geometricSteps(1000, 5) // => [0, 5, 31, 177, 1000]
 * ```
 */
export function geometricSteps(total: number, limit = 40): number[] {
  const last = Math.max(0, Math.floor(total))
  if (last + 1 <= limit) return Array.from({ length: last + 1 }, (_, t) => t)
  const out = new Set<number>([0, last])
  // Spaced evenly in log(t + 1): the gaps grow by a constant ratio.
  for (let k = 1; k < limit - 1; k++) out.add(Math.round((last + 1) ** (k / (limit - 1)) - 1))
  return [...out].sort((p, q) => p - q)
}

/**
 * Whether step t is on a geometric schedule (every step up to 8, then about one in each 15% stretch): for snapshotting
 * a run whose length is not known in advance, keeping a few dozen snapshots of even a long run.
 *
 * @param t - The step.
 * @param ratio - The schedule's growth per kept step (default 1.15).
 * @returns Whether to snapshot.
 * @example
 * ```ts
 * import { onGeometricSchedule } from 'whitebox-engine/learn'
 * [1, 2, 9, 12, 13, 14].map((t) => onGeometricSchedule(t)) // => [true, true, true, false, true, false]
 * ```
 */
export function onGeometricSchedule(t: number, ratio = 1.15): boolean {
  if (t <= 8) return true
  return Math.floor(Math.log(t) / Math.log(ratio)) !== Math.floor(Math.log(t - 1) / Math.log(ratio))
}

/**
 * A fitted model's predictions on a grid over two features of the training data: predicted class indices,
 * probabilities and (for binary problems) the decision function for classifiers, predictions for others; with the
 * training rows when the data is small, and the surface at recorded steps of training when given the models there.
 *
 * @param fitted - A fitted estimator with `predict`.
 * @param x - The data whose range the grid spans `[n, d]` (d at least 1; a single feature gives a one-row grid).
 * @param options - Features, resolution, padding, names, targets, decision levels and the steps of training.
 * @returns The decision surface.
 * @throws WhiteboxError `value` when the estimator cannot predict.
 * @example
 * ```ts
 * import { decisionSurfaceOf, defineEstimator } from 'whitebox-engine/learn'
 * import { asTensor, tensor, toArray } from 'whitebox-engine/tensor'
 * const signOfFirst = (x) => tensor(toArray(asTensor(x)).map((r) => Math.sign(r[0])))
 * const sign = defineEstimator(
 *   { name: 'sign', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: signOfFirst }) },
 *   {},
 * ).fit({ x: [[-1, 0], [1, 1]], y: [0, 1] })
 * decisionSurfaceOf(sign, [[-1, 0], [1, 1]], { resolution: 2, padding: 0 }).values // => [[-1, 1], [-1, 1]]
 * ```
 */
export function decisionSurfaceOf(
  fitted: Fitted,
  x: TensorLike,
  options: DecisionSurfaceOptions = {},
): DecisionSurface {
  if (fitted.predict === undefined)
    throw new WhiteboxError('value', `decisionSurfaceOf: ${fitted.estimator} cannot predict`)
  const m = matrixView(x, 'decisionSurfaceOf')
  const [fi, fj] = options.features ?? [0, Math.min(1, m.d - 1)]
  const res = options.resolution ?? 30
  const pad = options.padding ?? 0.1
  const means = Array.from({ length: m.d }, (_, j) => {
    let s = 0
    for (let i = 0; i < m.n; i++) s += m.data[i * m.d + j] ?? 0
    return s / m.n
  })
  const axis = (j: number, n: number): number[] => {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < m.n; i++) {
      const v = m.data[i * m.d + j] ?? 0
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    const span = hi > lo ? hi - lo : 1
    return linspaceNumbers(lo - pad * span, hi + pad * span, n)
  }
  const single = m.d === 1
  const gx = axis(fi, res)
  const gy = single ? [means[0] ?? 0] : axis(fj, res)
  const cells = gx.length * gy.length
  const grid = new Float64Array(cells * m.d)
  for (let a = 0; a < gy.length; a++)
    for (let b = 0; b < gx.length; b++) {
      const r = a * gx.length + b
      for (let j = 0; j < m.d; j++) grid[r * m.d + j] = means[j] ?? 0
      grid[r * m.d + fi] = gx[b] ?? 0
      if (!single) grid[r * m.d + fj] = gy[a] ?? 0
    }
  const input = fromStorage(grid, [cells, m.d], 'float64')
  const reshape = (flat: ArrayLike<number>, offset = 0, stride = 1): number[][] =>
    gy.map((_, a) => gx.map((_, b) => flat[(a * gx.length + b) * stride + offset] ?? 0))
  const names = options.featureNames ?? Array.from({ length: m.d }, (_, j) => `x${j}`)
  const features: [string, string] = [names[fi] ?? `x${fi}`, single ? '' : (names[fj] ?? `x${fj}`)]
  const classes = fitted.classes
  const classifier = fitted.kind === 'classifier' && classes !== undefined
  // A binary decision function, when the model has one ([m] scores for the positive class).
  const binaryDecision = (model: Fitted, at: TensorLike): Float64Array | undefined => {
    if (!classifier || classes.length !== 2 || model.decisionFunction === undefined) return undefined
    const f = model.decisionFunction(at) as Tensor
    return f.shape.length === 1 ? Float64Array.from(toFlat(f)) : undefined
  }
  const fieldsOf = (model: Fitted): SurfaceFields => {
    const predicted = toFlat((model.predict as (x: TensorLike) => Tensor)(input))
    if (!classifier) return { values: reshape(predicted) }
    const index = new Map(classes.map((c, i) => [c, i]))
    const labels = reshape(predicted.map((v) => index.get(v) ?? 0))
    const proba = model.predictProba === undefined ? undefined : (model.predictProba(input) as Tensor)
    const k = classes.length
    const probabilities = proba === undefined ? undefined : classes.map((_, c) => reshape(toFlat(proba), c, k))
    const decision = binaryDecision(model, input)
    return {
      labels,
      ...(probabilities === undefined ? {} : { probabilities }),
      ...(decision === undefined ? {} : { decision: reshape(decision) }),
    }
  }
  const limit = options.rowsLimit ?? ROWS_LIMIT
  const rows =
    limit <= 0
      ? undefined
      : inputRowsOf(x, {
          columns: [fi, fj],
          featureNames: names,
          limit,
          ...(options.y === undefined || !classifier ? {} : { y: options.y, classes }),
          ...(options.classNames === undefined ? {} : { classNames: options.classNames }),
        })
  // The rows a model gets wrong (indices into the rows the record carries).
  const misclassified = (model: Fitted): number[] | undefined => {
    const labels = rows?.labels
    if (labels === undefined || !classifier) return undefined
    const index = new Map(classes.map((c, i) => [c, i]))
    const predicted = toFlat((model.predict as (x: TensorLike) => Tensor)(x))
    return labels.flatMap((l, i) => (l >= 0 && index.get(predicted[i] ?? Number.NaN) !== l ? [i] : []))
  }
  const steps: SurfaceStep[] | undefined = options.steps?.models.map((model, t) => {
    const s = options.steps as SurfaceStepsInput
    const objective = s.objective?.[t]
    const highlight = s.highlight === 'misclassified' ? misclassified(model) : s.highlight?.[t]
    const rowWeights = s.rowWeights?.[t]
    const rowValues = rows === undefined ? undefined : binaryDecision(model, x)
    return {
      iteration: s.iterations[t] ?? t,
      ...fieldsOf(model),
      ...(objective === undefined || !Number.isFinite(objective) ? {} : { objective }),
      ...(rowValues === undefined ? {} : { rowValues: Array.from(rowValues) }),
      ...(highlight === undefined ? {} : { highlight }),
      ...(rowWeights === undefined || rowWeights.length !== rows?.points.length ? {} : { rowWeights }),
    }
  })
  const highlightName =
    options.steps?.highlightName ?? (options.steps?.highlight === 'misclassified' ? 'misclassified' : undefined)
  return decisionSurface({
    features,
    x: gx,
    y: gy,
    ...fieldsOf(fitted),
    ...(classifier ? { classes: options.classNames ?? classes.map(String) } : {}),
    ...(options.levels === undefined ? {} : { levels: options.levels }),
    ...(rows === undefined ? {} : { rows }),
    ...(steps === undefined ? {} : { steps }),
    ...(options.stepName === undefined ? {} : { stepName: options.stepName }),
    ...(highlightName === undefined ? {} : { highlightName }),
    ...(options.objective === undefined ? {} : { objective: options.objective }),
  })
}

/** An algorithm that also collects a summary of every state it steps through, with the summaries collected. */
export interface Collecting<S, T> {
  readonly algorithm: Algorithm<S>
  /** One entry per state, the initial state first; filled as the algorithm runs. */
  readonly history: readonly T[]
}

/**
 * Wrap an algorithm so that running it also collects a summary of each state (the initial one and every step's),
 * whatever the trace policy records: what a method needs to build its exposition records (a dual objective per SMO
 * step, a deviance per IRLS iteration). The wrapper has the algorithm's name, schema and behaviour.
 *
 * @param algorithm - The algorithm.
 * @param summarise - A summary of a state (plain data).
 * @returns The wrapped algorithm and the (live) history.
 * @concept algorithm-protocol
 * @example
 * ```ts
 * import { collecting } from 'whitebox-engine/learn'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const halve = { name: 'halve', config: {}, init: () => 8, step: (x) => x / 2 }
 * const c = collecting(halve, (x) => x)
 * runAlgorithm(c.algorithm, { maxSteps: 3 })
 * c.history // => [8, 4, 2, 1]
 * ```
 */
export function collecting<S, T>(algorithm: Algorithm<S>, summarise: (state: S, t: number) => T): Collecting<S, T> {
  const history: T[] = []
  return {
    history,
    algorithm: {
      ...algorithm,
      init: (key) => {
        history.length = 0
        const s = algorithm.init(key)
        history.push(summarise(s, 0))
        return s
      },
      step: (state, ctx) => {
        const s = algorithm.step(state, ctx)
        history.push(summarise(s, ctx.t + 1))
        return s
      },
    },
  }
}

/**
 * The feature names of a dataset passed as learn data: its `featureNames` when it carries them (every tabular
 * dataset does) with the right length, else sklearn's `x0, x1, ...`.
 *
 * @param data - The data (a dataset or plain learn data).
 * @param d - The number of features.
 * @returns The names.
 * @example
 * ```ts
 * import { featureNamesOf } from 'whitebox-engine/learn'
 * featureNamesOf({ x: [[1, 2]] }, 2) // => ['x0', 'x1']
 * featureNamesOf({ x: [[1, 2]], featureNames: ['a', 'b'] }, 2) // => ['a', 'b']
 * ```
 */
export function featureNamesOf(data: LearnData, d: number): string[] {
  const names = (data as { readonly featureNames?: unknown }).featureNames
  if (Array.isArray(names) && names.length === d && names.every((s) => typeof s === 'string')) return [...names]
  return Array.from({ length: d }, (_, j) => `x${j}`)
}

/**
 * The names of fitted classes: a classification dataset's `classes` (names indexed by label) when it carries them,
 * else the labels as strings.
 *
 * @param data - The data (a dataset or plain learn data).
 * @param classes - The fitted class labels.
 * @returns One name per class.
 * @example
 * ```ts
 * import { classNamesOf } from 'whitebox-engine/learn'
 * classNamesOf({ x: [[0]], classes: ['cat', 'dog'] }, [0, 1]) // => ['cat', 'dog']
 * classNamesOf({ x: [[0]] }, [2, 5]) // => ['2', '5']
 * ```
 */
export function classNamesOf(data: LearnData, classes: readonly number[]): string[] {
  const names = (data as { readonly classes?: unknown }).classes
  return classes.map((c) =>
    Array.isArray(names) && Number.isInteger(c) && typeof names[c] === 'string' ? (names[c] as string) : String(c),
  )
}

/**
 * Run options for a traced fit (section 3.2's rules): the caller's parent run, or the resolved key, tracer and policy.
 *
 * @param options - The fit options.
 * @param name - The name the key is derived under when none is given.
 * @returns Options to spread into `runAlgorithm`.
 * @example
 * ```ts
 * import { fitRunOptions } from 'whitebox-engine/learn'
 * import { key } from 'whitebox-engine/random'
 * fitRunOptions({ key: key(0) }, 'kmeans').key !== undefined // => true
 * ```
 */
export function fitRunOptions(
  options: FitOptions,
  name: string,
): Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'> {
  const key = resolveKey(options, name)
  if (options.parent !== undefined) return { key, parent: options.parent }
  return { key, tracer: resolveTracer(options), policy: resolvePolicy(options) }
}
