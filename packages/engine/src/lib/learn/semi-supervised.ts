/**
 * Semi-supervised classification and regression: a few labelled rows and many unlabelled ones (label `-1`, or NaN, in
 * `y`). The wrappers (self-training, co-training, tri-training) put pseudo-labels on the unlabelled rows round by round
 * over any base estimator; label propagation and spreading diffuse the labels over a similarity graph
 * (`sklearn.semi_supervised`); Laplacian-regularised least squares adds a graph smoothness penalty to kernel ridge
 * regression; the transductive SVM (S3VM) switches the labels of unlabelled rows in pairs while the weight of their
 * margin violations grows, over any weighted binary base. Each is an estimator on the iterative base, one round (or one
 * diffusion step) per step, so a session shows the labels spread and `fittedAt(t)` is the model at round `t`.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type KernelChoice, kernelMatrix, kernelSpace, type ResolvedKernel, resolveKernel } from '../kernels/index.ts'
import { dgemm, solve } from '../numerics/linalg/index.ts'
import { kNearest } from '../numerics/spatial/index.ts'
import { foldIn, type Key, permutation, randint } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { accuracyOf, r2Of } from './_classify.ts'
import { confidence, predicted } from './_staged.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import { type Estimator, type Fitted, type LearnData, requireTargets, requireVerb } from './estimator.ts'
import { fromView, type MatrixView, matrixView, takeColumns, takeData, takeRows } from './validation.ts'

/** The label that marks an unlabelled row of a classification target. */
export const UNLABELLED = -1

/**
 * Whether a target entry marks an unlabelled row: `-1` (scikit-learn's convention) or NaN (for regression targets).
 *
 * @param y - A target entry.
 * @returns `true` for an unlabelled row.
 * @example
 * ```ts
 * import { isUnlabelled } from 'mdml-engine/learn'
 * isUnlabelled(-1) // => true
 * isUnlabelled(Number.NaN) // => true
 * isUnlabelled(0) // => false
 * ```
 */
export function isUnlabelled(y: number): boolean {
  return y === UNLABELLED || Number.isNaN(y)
}

// ---------------------------------------------------------------------------------------------------------------------
// Shared pieces (not exported from the domain beyond this file)

/** The training data of a semi-supervised fit: the features, the targets with the unlabelled marker, and the classes. */
type Partial_ = { readonly m: MatrixView; readonly y: Float64Array; readonly classes: number[] }

function readPartial(data: LearnData, who: string, needClasses = true): Partial_ {
  const m = matrixView(data.x, who)
  const y = Float64Array.from(toFlat(asTensor(requireTargets(data, who))))
  if (y.length !== m.n) throw new MdmlError('shape', `${who}: ${m.n} rows but ${y.length} targets`)
  const seen = new Set<number>()
  for (const v of y) if (!isUnlabelled(v)) seen.add(v)
  if (seen.size === 0) throw new MdmlError('value', `${who}: needs at least one labelled row`)
  const classes = needClasses ? Array.from(seen).sort((a, b) => a - b) : []
  return { m, y, classes }
}

const labelledRows = (labels: ArrayLike<number>): number[] =>
  Array.from({ length: labels.length }, (_, i) => i).filter((i) => !isUnlabelled(labels[i] ?? UNLABELLED))

const unlabelledRows = (labels: ArrayLike<number>): number[] =>
  Array.from({ length: labels.length }, (_, i) => i).filter((i) => isUnlabelled(labels[i] ?? UNLABELLED))

/** The rows of the data, with the targets replaced by labels (indexed by the global row). */
function relabelled(data: LearnData, rows: readonly number[], labels: ArrayLike<number>): LearnData {
  const sub = takeData(data, rows)
  return {
    ...sub,
    y: fromStorage(
      Float64Array.from(rows, (r) => labels[r] ?? 0),
      [rows.length],
      'float64',
    ),
  }
}

const tensorOfRows = (m: MatrixView, rows: readonly number[]): Tensor => fromView(takeRows(m, rows))

/** Class probabilities of a fitted base on some rows, as a flat `[n, k]` array; the base must offer them. */
function probaOf(f: Fitted, x: TensorLike, who: string): { readonly p: Float64Array; readonly k: number } {
  const t = requireVerb(f, 'predictProba', who)(x)
  return { p: Float64Array.from(toFlat(t)), k: t.shape[1] ?? 1 }
}

const argmaxRow = (p: ArrayLike<number>, from: number, k: number): number => {
  let best = 0
  for (let c = 1; c < k; c++) if ((p[from + c] ?? 0) > (p[from + best] ?? 0)) best = c
  return best
}

const rowsTensor = (values: ArrayLike<number>, shape: readonly number[]): Tensor =>
  fromStorage(Float64Array.from(values), [...shape], 'float64')

/** What a fitted semi-supervised wrapper offers: the base's verbs, and the labels it put on the training rows. */
export interface PseudoLabelVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly decisionFunction?: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The label of every training row at the end: given, or pseudo (the transduction of scikit-learn). */
  readonly transduction: readonly number[]
  /** The round at which each training row got its label: 0 for the given ones, -1 for those still unlabelled. */
  readonly labelledIter: readonly number[]
}

/** Why a wrapper stopped. */
export type PseudoLabelStop = 'running' | 'all-labelled' | 'no-change' | 'max-iter'

const doneOf = (stop: PseudoLabelStop): 'converged' | 'completed' | undefined =>
  stop === 'running' ? undefined : stop === 'max-iter' ? 'completed' : 'converged'

function delegated(
  f: Fitted,
  d: number,
  who: string,
  transduction: readonly number[],
  iter: readonly number[],
): PseudoLabelVerbs {
  const predict = requireVerb(f, 'predict', who)
  return {
    predict,
    ...(f.predictProba === undefined ? {} : { predictProba: f.predictProba }),
    ...(f.decisionFunction === undefined ? {} : { decisionFunction: f.decisionFunction }),
    score: accuracyOf(predict),
    classes: f.classes ?? [],
    nFeaturesIn: d,
    transduction,
    labelledIter: iter,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Self-training

/** The hyperparameters of {@link selfTraining}. */
export interface SelfTrainingHyper {
  /** The base classifier: it must give class probabilities. */
  readonly base: Estimator
  /** `threshold` adds the rows whose top probability exceeds `threshold`; `k_best` the `kBest` most confident. */
  readonly criterion: 'threshold' | 'k_best'
  /** The probability a pseudo-label must exceed (scikit-learn's 0.75). */
  readonly threshold: number
  /** The rows added per round under `k_best`. */
  readonly kBest: number
  /** The most rounds (scikit-learn's `max_iter`, 10). */
  readonly maxIter: number
}

/** The state of {@link selfTrainingAlgorithm}: the label of every row, the round each got it, and the model fitted on them. */
export type SelfTrainingState = {
  readonly labels: number[]
  readonly iter: number[]
  readonly round: number
  readonly added: number
  /** The mean top probability of the rows added this round. */
  readonly meanConfidence: number
  /** The parameters of the base fitted on the rows labelled so far. */
  readonly model: Json
  readonly stop: PseudoLabelStop
}

/** The fitted parameters of {@link selfTraining}. */
export type SelfTrainingParams = SelfTrainingState & { readonly classes: number[]; readonly d: number }

/**
 * Self-training as an algorithm (`sklearn.semi_supervised.SelfTrainingClassifier`, Yarowsky 1995): fit the base on the
 * labelled rows, predict the unlabelled ones, give the confident ones their predicted label for good and refit; one round
 * per step, the first fit at step 0. It stops when no row is added, none is left, or `maxIter` rounds are done. Logs
 * `labelled` (rows with a label), `added` and `meanConfidence` of the round.
 *
 * @param data - Features, and targets with `-1` for the unlabelled rows.
 * @param hyper - The base and the selection rule.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { gaussianNaiveBayes, selfTrainingAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.2], [0.4], [5], [5.2], [5.4]]
 * const alg = selfTrainingAlgorithm({ x, y: [0, -1, -1, 1, -1, -1] }, { base: gaussianNaiveBayes(), criterion: 'threshold', threshold: 0.75, kBest: 10, maxIter: 10 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function selfTrainingAlgorithm(data: LearnData, hyper: SelfTrainingHyper): Algorithm<SelfTrainingState> {
  const who = 'selfTraining'
  const { m, y } = readPartial(data, who, false)
  const fitOn = (labels: readonly number[], key: Key): Json =>
    hyper.base.fit(relabelled(data, labelledRows(labels), labels), { key, explain: false }).params
  const stopOf = (labels: readonly number[], round: number, added: number): PseudoLabelStop =>
    unlabelledRows(labels).length === 0
      ? 'all-labelled'
      : round > 0 && added === 0
        ? 'no-change'
        : round >= hyper.maxIter
          ? 'max-iter'
          : 'running'
  return {
    name: 'self-training',
    config: { base: hyper.base.name, criterion: hyper.criterion, threshold: hyper.threshold, maxIter: hyper.maxIter },
    schema: channels({
      metrics: {
        labelled: { description: 'rows with a label, given or pseudo', direction: 'higher' },
        added: { description: 'pseudo-labels added this round' },
        meanConfidence: { description: 'mean top probability of the pseudo-labels added this round' },
      },
    }),
    init: (key) => {
      const labels = Array.from(y)
      return {
        labels,
        iter: labels.map((v) => (isUnlabelled(v) ? -1 : 0)),
        round: 0,
        added: 0,
        meanConfidence: 0,
        model: fitOn(labels, foldIn(key, 0)),
        stop: stopOf(labels, 0, 0),
      }
    },
    step: (s, ctx) => {
      const round = s.round + 1
      const rest = unlabelledRows(s.labels)
      const fitted = hyper.base.revive(s.model)
      const { p, k } = probaOf(fitted, tensorOfRows(m, rest), who)
      const classes = fitted.classes ?? []
      const top = rest.map((_, r) => p[r * k + argmaxRow(p, r * k, k)] ?? 0)
      const order = rest.map((_, r) => r)
      let chosen: number[]
      if (hyper.criterion === 'threshold') chosen = order.filter((r) => (top[r] ?? 0) > hyper.threshold)
      else
        chosen = order
          .sort((a, b) => (top[b] ?? 0) - (top[a] ?? 0) || a - b)
          .slice(0, Math.min(hyper.kBest, rest.length))
      const labels = [...s.labels]
      const iter = [...s.iter]
      let confSum = 0
      for (const r of chosen) {
        const row = rest[r] ?? 0
        labels[row] = classes[argmaxRow(p, r * k, k)] ?? UNLABELLED
        iter[row] = round
        confSum += top[r] ?? 0
      }
      const model = chosen.length === 0 ? s.model : fitOn(labels, foldIn(ctx.key, round))
      return {
        labels,
        iter,
        round,
        added: chosen.length,
        meanConfidence: chosen.length === 0 ? 0 : confSum / chosen.length,
        model,
        stop: stopOf(labels, round, chosen.length),
      }
    },
    done: (s) => doneOf(s.stop),
    observe: (s) => ({
      metrics: {
        labelled: labelledRows(s.labels).length,
        added: s.added,
        meanConfidence: s.meanConfidence,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SelfTrainingState,
  }
}

const SELF_DEFAULTS = {
  criterion: 'threshold',
  threshold: 0.75,
  kBest: 10,
  maxIter: 10,
} as const satisfies Omit<SelfTrainingHyper, 'base'>

/**
 * Self-training over any probabilistic classifier (`sklearn.semi_supervised.SelfTrainingClassifier`). The fitted value
 * predicts with the base refitted on the given and pseudo-labelled rows, and keeps `transduction` (the label of every
 * training row) and `labelledIter` (the round each got it).
 *
 * @param base - The base classifier (it must offer `predictProba`).
 * @param hyper - `criterion`, `threshold`, `kBest`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { gaussianNaiveBayes, selfTraining } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [0.2], [0.4], [5], [5.2], [5.4]]
 * const f = selfTraining(gaussianNaiveBayes()).fit({ x, y: [0, -1, -1, 1, -1, -1] })
 * f.transduction // => [0, 0, 0, 1, 1, 1]
 * toFlat(f.predict([[0.1], [5.1]])) // => [0, 1]
 * ```
 */
export function selfTraining(
  base: Estimator,
  hyper: Partial<Omit<SelfTrainingHyper, 'base'>> = {},
): IterativeEstimator<SelfTrainingHyper, SelfTrainingState, SelfTrainingParams, PseudoLabelVerbs> {
  const h: SelfTrainingHyper = { base, ...SELF_DEFAULTS, ...hyper }
  return iterative<SelfTrainingHyper, SelfTrainingState, SelfTrainingParams, PseudoLabelVerbs>(
    {
      name: 'self-training',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          threshold: real(0.5, 1, { default: c.threshold, label: 'confidence threshold' }),
          maxIter: int(1, 50, { default: c.maxIter, label: 'rounds' }),
        }),
      algorithm: (data, c) => selfTrainingAlgorithm(data, c),
      maxSteps: (c) => c.maxIter + 2,
      params: (s, data) => ({
        ...s,
        classes: readPartial(data, 'selfTraining').classes,
        d: matrixView(data.x, 'selfTraining').d,
      }),
      verbs: (p, c) => delegated(c.base.revive(p.model), p.d, 'selfTraining', p.labels, p.iter),
      logs: { 'solver/labelled': Log(), 'solver/added': Log(), 'solver/meanConfidence': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Co-training

/** The hyperparameters of {@link coTraining}. */
export interface CoTrainingHyper {
  /** The base classifier of both views (it must give class probabilities). */
  readonly base: Estimator
  /** The columns of the first view (default the first half of the features). */
  readonly viewA: readonly number[] | null
  /** The columns of the second view (default the rest). */
  readonly viewB: readonly number[] | null
  /** The rows each classifier labels per round (Blum and Mitchell's p and n together). */
  readonly growth: number
  /** The size of the pool of unlabelled rows the classifiers choose from. */
  readonly pool: number
  /** The least probability a classifier needs to label a row. */
  readonly threshold: number
  readonly maxRounds: number
}

/** The state of {@link coTrainingAlgorithm}. */
export type CoTrainingState = {
  readonly labels: number[]
  readonly iter: number[]
  readonly round: number
  readonly added: number
  /** The fraction of the pool on which the two views disagree. */
  readonly disagreement: number
  readonly modelA: Json
  readonly modelB: Json
  /** The order the unlabelled rows enter the pool in (fixed by the key). */
  readonly order: number[]
  readonly stop: PseudoLabelStop
}

/** The fitted parameters of {@link coTraining}. */
export type CoTrainingParams = CoTrainingState & {
  readonly classes: number[]
  readonly d: number
  readonly viewA: number[]
  readonly viewB: number[]
}

/** The two views of `d` features: the given columns, or the first half and the rest. */
function viewsOf(d: number, a: readonly number[] | null, b: readonly number[] | null): [number[], number[]] {
  if (a !== null && b !== null) return [[...a], [...b]]
  const mid = Math.max(1, Math.ceil(d / 2))
  const cols = Array.from({ length: d }, (_, j) => j)
  return [cols.slice(0, mid), cols.slice(mid).length === 0 ? cols.slice(0, mid) : cols.slice(mid)]
}

/**
 * Co-training as an algorithm (Blum and Mitchell 1998): two classifiers, one per view of the features, share one
 * labelled set; each round each labels the `growth` rows of a pool of unlabelled ones it is most sure of (when above the
 * threshold), a row chosen by both goes to the more confident, and both are refitted. Logs `labelled`, `added` and
 * `disagreement` (the fraction of the pool the views label differently).
 *
 * @param data - Features, and targets with `-1` for the unlabelled rows.
 * @param hyper - The base, the views and the schedule.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { coTrainingAlgorithm, gaussianNaiveBayes } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 0], [0.2, 0.1], [0.1, 0.3], [5, 5], [5.2, 5.1], [5.1, 5.3]]
 * const alg = coTrainingAlgorithm({ x, y: [0, -1, -1, 1, -1, -1] }, { base: gaussianNaiveBayes(), viewA: null, viewB: null, growth: 2, pool: 10, threshold: 0.5, maxRounds: 5 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function coTrainingAlgorithm(data: LearnData, hyper: CoTrainingHyper): Algorithm<CoTrainingState> {
  const who = 'coTraining'
  const { m, y } = readPartial(data, who, false)
  const [va, vb] = viewsOf(m.d, hyper.viewA, hyper.viewB)
  const views: [MatrixView, MatrixView] = [takeColumns(m, va), takeColumns(m, vb)]
  const fitView = (v: 0 | 1, labels: readonly number[], key: Key): Json => {
    const rows = labelledRows(labels)
    const sub = takeData({ x: fromView(views[v]), y: data.y ?? [] } as LearnData, rows)
    const yy = fromStorage(
      Float64Array.from(rows, (r) => labels[r] ?? 0),
      [rows.length],
      'float64',
    )
    return hyper.base.fit({ ...sub, y: yy }, { key, explain: false }).params
  }
  const stopOf = (labels: readonly number[], round: number, added: number): PseudoLabelStop =>
    unlabelledRows(labels).length === 0
      ? 'all-labelled'
      : round > 0 && added === 0
        ? 'no-change'
        : round >= hyper.maxRounds
          ? 'max-iter'
          : 'running'
  return {
    name: 'co-training',
    config: { base: hyper.base.name, growth: hyper.growth, pool: hyper.pool, maxRounds: hyper.maxRounds },
    schema: channels({
      metrics: {
        labelled: { description: 'rows with a label, given or pseudo', direction: 'higher' },
        added: { description: 'pseudo-labels added this round' },
        disagreement: { description: 'fraction of the pool the two views label differently' },
      },
    }),
    init: (key) => {
      const labels = Array.from(y)
      const order = Array.from(permutation(foldIn(key, 1), m.n).data).filter((i) => isUnlabelled(labels[i] ?? 0))
      return {
        labels,
        iter: labels.map((v) => (isUnlabelled(v) ? -1 : 0)),
        round: 0,
        added: 0,
        disagreement: 0,
        modelA: fitView(0, labels, foldIn(key, 2)),
        modelB: fitView(1, labels, foldIn(key, 3)),
        order,
        stop: stopOf(labels, 0, 0),
      }
    },
    step: (s, ctx) => {
      const round = s.round + 1
      const pool = s.order.filter((i) => isUnlabelled(s.labels[i] ?? 0)).slice(0, hyper.pool)
      const fa = hyper.base.revive(s.modelA)
      const fb = hyper.base.revive(s.modelB)
      const pa = probaOf(fa, fromView(takeRows(views[0], pool)), who)
      const pb = probaOf(fb, fromView(takeRows(views[1], pool)), who)
      const classes = fa.classes ?? []
      const pick = (pr: { p: Float64Array; k: number }, r: number): { label: number; conf: number } => {
        const c = argmaxRow(pr.p, r * pr.k, pr.k)
        return { label: classes[c] ?? UNLABELLED, conf: pr.p[r * pr.k + c] ?? 0 }
      }
      const answers = pool.map((_, r) => [pick(pa, r), pick(pb, r)] as const)
      const disagree = answers.filter(([a, b]) => a.label !== b.label).length
      const chosen = new Map<number, { label: number; conf: number }>()
      for (const v of [0, 1] as const) {
        const ranked = pool
          .map((_, r) => r)
          .filter((r) => (answers[r]?.[v].conf ?? 0) >= hyper.threshold)
          .sort((a, b) => (answers[b]?.[v].conf ?? 0) - (answers[a]?.[v].conf ?? 0) || a - b)
          .slice(0, hyper.growth)
        for (const r of ranked) {
          const a = answers[r]?.[v]
          const have = chosen.get(r)
          if (a !== undefined && (have === undefined || a.conf > have.conf)) chosen.set(r, a)
        }
      }
      const labels = [...s.labels]
      const iter = [...s.iter]
      for (const [r, a] of chosen) {
        const row = pool[r] ?? 0
        labels[row] = a.label
        iter[row] = round
      }
      const key = foldIn(ctx.key, round)
      return {
        labels,
        iter,
        round,
        added: chosen.size,
        disagreement: pool.length === 0 ? 0 : disagree / pool.length,
        modelA: chosen.size === 0 ? s.modelA : fitView(0, labels, foldIn(key, 0)),
        modelB: chosen.size === 0 ? s.modelB : fitView(1, labels, foldIn(key, 1)),
        order: s.order,
        stop: stopOf(labels, round, chosen.size),
      }
    },
    done: (s) => doneOf(s.stop),
    observe: (s) => ({
      metrics: { labelled: labelledRows(s.labels).length, added: s.added, disagreement: s.disagreement },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CoTrainingState,
  }
}

/** The product of the two views' probabilities, normalised (Blum and Mitchell's combination). */
function coVerbs(p: CoTrainingParams, h: CoTrainingHyper): PseudoLabelVerbs {
  const fa = h.base.revive(p.modelA)
  const fb = h.base.revive(p.modelB)
  const classes = p.classes
  const joint = (x: TensorLike): { readonly out: Float64Array; readonly n: number; readonly k: number } => {
    const m = matrixView(x, 'coTraining', { nFeatures: p.d })
    const a = probaOf(fa, fromView(takeColumns(m, p.viewA)), 'coTraining')
    const b = probaOf(fb, fromView(takeColumns(m, p.viewB)), 'coTraining')
    const out = new Float64Array(a.p.length)
    for (let r = 0; r < m.n; r++) {
      let z = 0
      for (let c = 0; c < a.k; c++) {
        const v = (a.p[r * a.k + c] ?? 0) * (b.p[r * a.k + c] ?? 0)
        out[r * a.k + c] = v
        z += v
      }
      for (let c = 0; c < a.k; c++) out[r * a.k + c] = z > 0 ? (out[r * a.k + c] ?? 0) / z : 1 / a.k
    }
    return { out, n: m.n, k: a.k }
  }
  const predict = (x: TensorLike): Tensor => {
    const { out, n, k } = joint(x)
    return rowsTensor(
      Array.from(
        { length: n },
        (_, r) => fa.classes?.[argmaxRow(out, r * k, k)] ?? classes[argmaxRow(out, r * k, k)] ?? 0,
      ),
      [n],
    )
  }
  return {
    predict,
    predictProba: (x) => {
      const { out, n, k } = joint(x)
      return rowsTensor(out, [n, k])
    },
    score: accuracyOf(predict),
    classes: fa.classes ?? classes,
    nFeaturesIn: p.d,
    transduction: p.labels,
    labelledIter: p.iter,
  }
}

const CO_DEFAULTS = {
  viewA: null,
  viewB: null,
  growth: 5,
  pool: 75,
  threshold: 0.5,
  maxRounds: 30,
} as const satisfies Omit<CoTrainingHyper, 'base'>

/**
 * Co-training over any probabilistic classifier (Blum and Mitchell 1998): the features split into two views, a
 * classifier on each, each teaching the other. The fitted value combines the views by the product of their
 * probabilities; `transduction` holds the labels the rows got.
 *
 * @param base - The base classifier (it must offer `predictProba`).
 * @param hyper - The views (`viewA`, `viewB` as column indices), `growth`, `pool`, `threshold`, `maxRounds`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { coTraining, gaussianNaiveBayes } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [0.2, 0.1], [0.1, 0.3], [0.3, 0.2], [5, 5], [5.2, 5.1], [5.1, 5.3], [5.3, 5.2]]
 * const f = coTraining(gaussianNaiveBayes(), { growth: 2 }).fit({ x, y: [0, 0, -1, -1, 1, 1, -1, -1] })
 * toFlat(f.predict([[0.1, 0.1], [5.1, 5.2]])) // => [0, 1]
 * ```
 */
export function coTraining(
  base: Estimator,
  hyper: Partial<Omit<CoTrainingHyper, 'base'>> = {},
): IterativeEstimator<CoTrainingHyper, CoTrainingState, CoTrainingParams, PseudoLabelVerbs> {
  const h: CoTrainingHyper = { base, ...CO_DEFAULTS, ...hyper }
  return iterative<CoTrainingHyper, CoTrainingState, CoTrainingParams, PseudoLabelVerbs>(
    {
      name: 'co-training',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          growth: int(1, 50, { default: c.growth, label: 'rows labelled per round' }),
          threshold: real(0, 1, { default: c.threshold, label: 'confidence threshold' }),
        }),
      algorithm: (data, c) => coTrainingAlgorithm(data, c),
      maxSteps: (c) => c.maxRounds + 2,
      params: (s, data, c) => {
        const mm = matrixView(data.x, 'coTraining')
        const [va, vb] = viewsOf(mm.d, c.viewA, c.viewB)
        return { ...s, classes: readPartial(data, 'coTraining').classes, d: mm.d, viewA: va, viewB: vb }
      },
      verbs: coVerbs,
      logs: { 'solver/labelled': Log(), 'solver/added': Log(), 'solver/disagreement': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Tri-training

/** The hyperparameters of {@link triTraining}. */
export interface TriTrainingHyper {
  /** The base classifier of all three members. */
  readonly base: Estimator
  readonly maxRounds: number
}

/** The state of {@link triTrainingAlgorithm}: three models and what Zhou and Li track for each. */
export type TriTrainingState = {
  readonly round: number
  readonly models: Json[]
  /** The error estimate e' of each member at its last update. */
  readonly errors: number[]
  /** The number of pseudo-labelled rows l' of each member at its last update. */
  readonly sizes: number[]
  /** The error of each pair's agreement on the labelled rows, as estimated this round (for member i, the other two). */
  readonly agreement: number[]
  /** How many members were refitted this round. */
  readonly updated: number
  readonly stop: PseudoLabelStop
}

/** The fitted parameters of {@link triTraining}. */
export type TriTrainingParams = TriTrainingState & {
  readonly classes: number[]
  readonly d: number
  readonly transduction: number[]
}

/**
 * Tri-training as an algorithm (Zhou and Li 2005): three classifiers fitted on bootstrap samples of the labelled rows;
 * each round, each is retrained on the labelled rows plus the unlabelled rows on which the other two agree, but only
 * when the estimated error of that agreement fell (the paper's conditions, with the subsampling that keeps the noise
 * rate in check). It stops when no member changes. Logs `updated`, `errorA`, `errorB` and `errorC` (the agreement error
 * of the other two members on the labelled rows).
 *
 * @param data - Features, and targets with `-1` for the unlabelled rows.
 * @param hyper - The base and the round budget.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { gaussianNaiveBayes, triTrainingAlgorithm } from 'mdml-engine/learn'
 * triTrainingAlgorithm({ x: [[0], [1], [2]], y: [0, 1, -1] }, { base: gaussianNaiveBayes(), maxRounds: 3 }).name // => 'tri-training'
 * ```
 */
export function triTrainingAlgorithm(data: LearnData, hyper: TriTrainingHyper): Algorithm<TriTrainingState> {
  const who = 'triTraining'
  const { m, y } = readPartial(data, who, false)
  const lab = labelledRows(y)
  const unl = unlabelledRows(y)
  const xl = tensorOfRows(m, lab)
  const xu = tensorOfRows(m, unl)
  const labelOf = (r: number): number => y[lab[r] ?? 0] ?? 0
  const fitOn = (rowsL: readonly number[], pseudo: readonly { u: number; label: number }[], key: Key): Json => {
    const rows = [...rowsL.map((r) => lab[r] ?? 0), ...pseudo.map((p) => unl[p.u] ?? 0)]
    const labels = [...rowsL.map(labelOf), ...pseudo.map((p) => p.label)]
    const sub = takeData(data, rows)
    return hyper.base.fit({ ...sub, y: rowsTensor(labels, [rows.length]) }, { key, explain: false }).params
  }
  return {
    name: 'tri-training',
    config: { base: hyper.base.name, maxRounds: hyper.maxRounds },
    schema: channels({
      metrics: {
        updated: { description: 'members refitted this round' },
        errorA: {
          description: 'estimated error of members B and C agreeing, on the labelled rows',
          direction: 'lower',
        },
        errorB: {
          description: 'estimated error of members A and C agreeing, on the labelled rows',
          direction: 'lower',
        },
        errorC: {
          description: 'estimated error of members A and B agreeing, on the labelled rows',
          direction: 'lower',
        },
      },
    }),
    init: (key) => {
      const allRows = lab.map((_, r) => r)
      const classesL = new Set(allRows.map(labelOf)).size
      const models = [0, 1, 2].map((i) => {
        // A bootstrap sample of the labelled rows; one that misses a class falls back to all of them.
        const boot = Array.from(randint(foldIn(key, 10 + i), [lab.length], 0, lab.length).data)
        const rowsL = new Set(boot.map(labelOf)).size < classesL ? allRows : boot
        return fitOn(rowsL, [], foldIn(key, 20 + i))
      })
      return {
        round: 0,
        models,
        errors: [0.5, 0.5, 0.5],
        sizes: [0, 0, 0],
        agreement: [0.5, 0.5, 0.5],
        updated: 0,
        stop: unl.length === 0 ? 'all-labelled' : 'running',
      }
    },
    step: (s, ctx) => {
      const round = s.round + 1
      const fits = s.models.map((p) => hyper.base.revive(p))
      const onL = fits.map((f) => predicted(f, xl))
      const onU = fits.map((f) => predicted(f, xu))
      const models = [...s.models]
      const errors = [...s.errors]
      const sizes = [...s.sizes]
      const currentErrors = [0.5, 0.5, 0.5]
      let updated = 0
      for (let i = 0; i < 3; i++) {
        const j = (i + 1) % 3
        const k = (i + 2) % 3
        let agree = 0
        let wrong = 0
        for (let r = 0; r < lab.length; r++) {
          if (onL[j]?.[r] === onL[k]?.[r]) {
            agree++
            if (onL[j]?.[r] !== labelOf(r)) wrong++
          }
        }
        const e = agree === 0 ? 0.5 : wrong / agree
        currentErrors[i] = e
        const ePrev = s.errors[i] ?? 0.5
        if (!(e < ePrev)) continue
        let pseudo = unl
          .map((_, u) => u)
          .filter((u) => onU[j]?.[u] === onU[k]?.[u])
          .map((u) => ({ u, label: onU[j]?.[u] ?? 0 }))
        let lPrev = s.sizes[i] ?? 0
        if (lPrev === 0) lPrev = Math.floor(e / (ePrev - e) + 1)
        let update = false
        if (lPrev < pseudo.length) {
          if (e * pseudo.length < ePrev * lPrev) update = true
          else if (lPrev > e / (ePrev - e)) {
            const keep = Math.ceil((ePrev * lPrev) / e - 1)
            const order = Array.from(permutation(foldIn(ctx.key, 100 * round + i), pseudo.length).data).slice(0, keep)
            pseudo = order.map((r) => pseudo[r] ?? { u: 0, label: 0 })
            update = true
          }
        }
        if (update) {
          models[i] = fitOn(
            lab.map((_, r) => r),
            pseudo,
            foldIn(ctx.key, 1000 * round + i),
          )
          errors[i] = e
          sizes[i] = pseudo.length
          updated++
        } else sizes[i] = lPrev
      }
      return {
        round,
        models,
        errors,
        sizes,
        updated,
        agreement: currentErrors,
        stop: updated === 0 ? 'no-change' : round >= hyper.maxRounds ? 'max-iter' : 'running',
      }
    },
    done: (s) => doneOf(s.stop),
    observe: (s) => ({
      metrics: {
        updated: s.updated,
        errorA: s.agreement[0] ?? 0.5,
        errorB: s.agreement[1] ?? 0.5,
        errorC: s.agreement[2] ?? 0.5,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TriTrainingState,
  }
}

function triVerbs(p: TriTrainingParams, h: TriTrainingHyper): PseudoLabelVerbs {
  const fits = p.models.map((q) => h.base.revive(q))
  const classes = p.classes
  const vote = (x: TensorLike): Tensor => {
    const votes = fits.map((f) => predicted(f, x))
    const n = votes[0]?.length ?? 0
    const proba = fits.every((f) => f.predictProba !== undefined)
    const mean = proba ? meanProba(x) : undefined
    return rowsTensor(
      Array.from({ length: n }, (_, r) => {
        const tally = new Map<number, number>()
        for (const v of votes) tally.set(v[r] ?? 0, (tally.get(v[r] ?? 0) ?? 0) + 1)
        const best = Math.max(...tally.values())
        const tied = Array.from(tally)
          .filter(([, c]) => c === best)
          .map(([label]) => label)
        if (tied.length === 1 || mean === undefined) return Math.min(...tied)
        const k = mean.k
        return tied.reduce((a, b) => {
          const ia = classes.indexOf(a)
          const ib = classes.indexOf(b)
          return (mean.p[r * k + ia] ?? 0) >= (mean.p[r * k + ib] ?? 0) ? a : b
        })
      }),
      [n],
    )
  }
  const meanProba = (x: TensorLike): { readonly p: Float64Array; readonly k: number } => {
    const parts = fits.map((f) => probaOf(f, x, 'triTraining'))
    const k = parts[0]?.k ?? 1
    const out = new Float64Array(parts[0]?.p.length ?? 0)
    for (const q of parts) for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (q.p[i] ?? 0) / parts.length
    return { p: out, k }
  }
  const canProba = fits.every((f) => f.predictProba !== undefined)
  return {
    predict: vote,
    ...(canProba
      ? {
          predictProba: (x: TensorLike): Tensor => {
            const { p: out, k } = meanProba(x)
            return rowsTensor(out, [out.length / k, k])
          },
        }
      : {}),
    score: accuracyOf(vote),
    classes: fits[0]?.classes ?? classes,
    nFeaturesIn: p.d,
    transduction: p.transduction,
    labelledIter: p.transduction.map(() => 0),
  }
}

/**
 * Tri-training over any classifier (Zhou and Li 2005): three members that label unlabelled rows for one another when
 * the other two agree. The fitted value is their majority vote (ties broken by the mean probability when the base has
 * one); `transduction` holds the given labels and, for the unlabelled rows, the vote.
 *
 * @param base - The base classifier.
 * @param hyper - `maxRounds`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { gaussianNaiveBayes, triTraining } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [0.1], [0.2], [0.3], [5], [5.1], [5.2], [5.3]]
 * const f = triTraining(gaussianNaiveBayes()).fit({ x, y: [0, 0, -1, -1, 1, 1, -1, -1] })
 * toFlat(f.predict([[0.15], [5.15]])) // => [0, 1]
 * ```
 */
export function triTraining(
  base: Estimator,
  hyper: Partial<Omit<TriTrainingHyper, 'base'>> = {},
): IterativeEstimator<TriTrainingHyper, TriTrainingState, TriTrainingParams, PseudoLabelVerbs> {
  const h: TriTrainingHyper = { base, maxRounds: 30, ...hyper }
  return iterative<TriTrainingHyper, TriTrainingState, TriTrainingParams, PseudoLabelVerbs>(
    {
      name: 'tri-training',
      kind: 'classifier',
      defaults: h,
      algorithm: (data, c) => triTrainingAlgorithm(data, c),
      maxSteps: (c) => c.maxRounds + 2,
      params: (s, data, c) => {
        const { m, y, classes } = readPartial(data, 'triTraining')
        const votes = triVerbs({ ...s, classes, d: m.d, transduction: [] }, c).predict(fromView(m))
        const v = toFlat(votes)
        return {
          ...s,
          classes,
          d: m.d,
          transduction: Array.from(y, (label, i) => (isUnlabelled(label) ? (v[i] ?? label) : label)),
        }
      },
      verbs: triVerbs,
      logs: { 'solver/updated': Log(), 'solver/errorA': Log(), 'solver/errorB': Log(), 'solver/errorC': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Label propagation and label spreading

/** The hyperparameters of {@link labelPropagation} and {@link labelSpreading}. */
export interface GraphLabelHyper {
  /** `rbf` weights every pair by exp(-gamma |x - y|^2); `knn` connects each row to its `nNeighbors` nearest (itself included). */
  readonly kernel: 'rbf' | 'knn'
  readonly gamma: number
  readonly nNeighbors: number
  /** The share of the neighbours' labels kept by spreading, in (0, 1) (ignored by propagation). */
  readonly alpha: number
  readonly maxIter: number
  /** Stop when the labels' total change falls below this. */
  readonly tol: number
}

/** The state of the diffusion: the label distributions `[n, k]` and how much they moved at the last step. */
export type GraphLabelState = {
  readonly f: Float64Array
  readonly change: number
  readonly iter: number
}

/** The fitted parameters of {@link labelPropagation} and {@link labelSpreading}. */
export type GraphLabelParams = {
  readonly classes: number[]
  /** The training rows, row-major. */
  readonly x: number[]
  readonly n: number
  readonly d: number
  /** The label distributions `[n, k]` row-major, each row normalised to sum to one. */
  readonly distributions: number[]
  readonly kernel: 'rbf' | 'knn'
  readonly gamma: number
  readonly nNeighbors: number
  readonly iter: number
}

/** What a fitted label propagation offers. */
export interface GraphLabelVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The label of each training row: the class with most mass. */
  readonly transduction: readonly number[]
  /** The label distributions `[n, k]` of the training rows. */
  readonly labelDistributions: readonly (readonly number[])[]
}

/** The weights of the similarity graph `[n, n]`, row-major: the kernel, or the 0/1 neighbour graph. */
function similarity(m: MatrixView, h: GraphLabelHyper): Float64Array {
  if (h.kernel === 'rbf') {
    const resolved: ResolvedKernel = { kernel: 'rbf', gamma: h.gamma, degree: 3, coef0: 1 }
    return kernelMatrix({ kernel: 'rbf' }, resolved, fromView(m))
  }
  const k = Math.min(h.nNeighbors, m.n)
  const nb = kNearest(fromView(m), undefined, k)
  const idx = nb.indices.data
  const w = new Float64Array(m.n * m.n)
  for (let i = 0; i < m.n; i++) for (let c = 0; c < k; c++) w[i * m.n + (idx[i * k + c] ?? 0)] = 1
  return w
}

/**
 * The matrix a diffusion step multiplies by. Propagation: the weights divided by the column sums (scikit-learn's
 * normalisation: row `i` by the sum of column `i` for the dense kernel, entry `(i, j)` by the sum of column `j` for the
 * neighbour graph). Spreading: `W_ij / sqrt(c_i c_j)` off the diagonal and zero on it, `c` the column sums without the
 * diagonal (the normalised Laplacian of scipy, negated).
 */
function transitionOf(
  w: Float64Array,
  n: number,
  h: GraphLabelHyper,
  variant: 'propagation' | 'spreading',
): Float64Array {
  const colSum = new Float64Array(n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) colSum[j] = (colSum[j] ?? 0) + (w[i * n + j] ?? 0)
  const out = new Float64Array(n * n)
  if (variant === 'propagation') {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const z = h.kernel === 'rbf' ? (colSum[i] ?? 1) : (colSum[j] ?? 1)
        out[i * n + j] = (w[i * n + j] ?? 0) / z
      }
    return out
  }
  const c = Float64Array.from(colSum, (v, j) => {
    const off = v - (w[j * n + j] ?? 0)
    return off === 0 ? 1 : off
  })
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j) out[i * n + j] = (w[i * n + j] ?? 0) / Math.sqrt((c[i] ?? 1) * (c[j] ?? 1))
  return out
}

function graphAlgorithm(
  variant: 'propagation' | 'spreading',
  data: LearnData,
  h: GraphLabelHyper,
): Algorithm<GraphLabelState> {
  const who = variant === 'propagation' ? 'labelPropagation' : 'labelSpreading'
  const { m, y, classes } = readPartial(data, who)
  if (variant === 'spreading' && !(h.alpha > 0 && h.alpha < 1)) {
    throw new MdmlError('value', `${who}: alpha must be in (0, 1), got ${h.alpha}`)
  }
  const k = classes.length
  const n = m.n
  const g = transitionOf(similarity(m, h), n, h, variant)
  const labelled = Uint8Array.from(y, (v) => (isUnlabelled(v) ? 0 : 1))
  const clamp = new Float64Array(n * k)
  for (let i = 0; i < n; i++) if (labelled[i] === 1) clamp[i * k + classes.indexOf(y[i] ?? 0)] = 1
  const initial = Float64Array.from(clamp)
  const total = (a: Float64Array, b: Float64Array): number => {
    let s = 0
    for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0))
    return s
  }
  return {
    name: variant === 'propagation' ? 'label-propagation' : 'label-spreading',
    config: { kernel: h.kernel, gamma: h.gamma, nNeighbors: h.nNeighbors, alpha: h.alpha, rows: n, classes: k },
    schema: channels({
      metrics: {
        change: { description: 'total absolute change of the label distributions this step', direction: 'lower' },
        clampedMass: { description: 'label mass on the labelled rows' },
      },
    }),
    init: () => ({ f: initial, change: total(initial, new Float64Array(n * k)), iter: 0 }),
    step: (s) => {
      const next = dgemm(g, s.f, n, n, k)
      for (let i = 0; i < n; i++) {
        if (variant === 'propagation') {
          let z = 0
          for (let c = 0; c < k; c++) z += next[i * k + c] ?? 0
          const div = z === 0 ? 1 : z
          for (let c = 0; c < k; c++)
            next[i * k + c] = labelled[i] === 1 ? (clamp[i * k + c] ?? 0) : (next[i * k + c] ?? 0) / div
        } else {
          for (let c = 0; c < k; c++)
            next[i * k + c] =
              h.alpha * (next[i * k + c] ?? 0) + (1 - h.alpha) * (clamp[i * k + c] ?? 0) * (labelled[i] === 1 ? 1 : 0)
        }
      }
      return { f: next, change: total(next, s.f), iter: s.iter + 1 }
    },
    done: (s) => (s.change < h.tol ? 'converged' : s.iter >= h.maxIter ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        change: s.change,
        clampedMass: labelledRows(y).reduce((a, i) => a + classes.reduce((b, _, c) => b + (s.f[i * k + c] ?? 0), 0), 0),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GraphLabelState,
  }
}

/** Normalise the rows of an `[n, k]` array to sum to one (rows of zeros become uniform). */
function normaliseRows(f: ArrayLike<number>, k: number): Float64Array {
  const out = new Float64Array(f.length)
  for (let i = 0; i < f.length / k; i++) {
    let z = 0
    for (let c = 0; c < k; c++) z += f[i * k + c] ?? 0
    for (let c = 0; c < k; c++) out[i * k + c] = z > 0 ? (f[i * k + c] ?? 0) / z : 1 / k
  }
  return out
}

function graphVerbs(p: GraphLabelParams): GraphLabelVerbs {
  const k = p.classes.length
  const train = rowsTensor(p.x, [p.n, p.d])
  const proba = (x: TensorLike): Float64Array => {
    const q = matrixView(x, 'labelPropagation', { nFeatures: p.d })
    const out = new Float64Array(q.n * k)
    if (p.kernel === 'rbf') {
      const w = kernelMatrix(
        { kernel: 'rbf' },
        { kernel: 'rbf', gamma: p.gamma, degree: 3, coef0: 1 },
        fromView(q),
        train,
      )
      for (let r = 0; r < q.n; r++)
        for (let i = 0; i < p.n; i++)
          for (let c = 0; c < k; c++)
            out[r * k + c] = (out[r * k + c] ?? 0) + (w[r * p.n + i] ?? 0) * (p.distributions[i * k + c] ?? 0)
    } else {
      const kk = Math.min(p.nNeighbors, p.n)
      const idx = kNearest(train, fromView(q), kk).indices.data
      for (let r = 0; r < q.n; r++)
        for (let j = 0; j < kk; j++)
          for (let c = 0; c < k; c++)
            out[r * k + c] = (out[r * k + c] ?? 0) + (p.distributions[(idx[r * kk + j] ?? 0) * k + c] ?? 0)
    }
    return normaliseRows(out, k)
  }
  const predict = (x: TensorLike): Tensor => {
    const pr = proba(x)
    return rowsTensor(
      Array.from({ length: pr.length / k }, (_, r) => p.classes[argmaxRow(pr, r * k, k)] ?? 0),
      [pr.length / k],
    )
  }
  return {
    predict,
    predictProba: (x) => {
      const pr = proba(x)
      return rowsTensor(pr, [pr.length / k, k])
    },
    score: accuracyOf(predict),
    classes: p.classes,
    nFeaturesIn: p.d,
    transduction: Array.from({ length: p.n }, (_, i) => p.classes[argmaxRow(p.distributions, i * k, k)] ?? 0),
    labelDistributions: Array.from({ length: p.n }, (_, i) => p.distributions.slice(i * k, (i + 1) * k)),
  }
}

function graphEstimator(
  variant: 'propagation' | 'spreading',
  defaults: GraphLabelHyper,
  hyper: Partial<GraphLabelHyper>,
): IterativeEstimator<GraphLabelHyper, GraphLabelState, GraphLabelParams, GraphLabelVerbs> {
  const name = variant === 'propagation' ? 'label-propagation' : 'label-spreading'
  return iterative<GraphLabelHyper, GraphLabelState, GraphLabelParams, GraphLabelVerbs>(
    {
      name,
      kind: 'classifier',
      defaults: { ...defaults, ...hyper },
      hyperSpace: (c) =>
        dict({
          gamma: real(1e-2, 1e3, { log: true, default: c.gamma, label: 'rbf gamma' }),
          nNeighbors: int(1, 50, { default: c.nNeighbors, label: 'neighbours' }),
          ...(variant === 'spreading' ? { alpha: real(0.01, 0.99, { default: c.alpha, label: 'alpha' }) } : {}),
        }),
      algorithm: (data, c) => graphAlgorithm(variant, data, c),
      maxSteps: (c) => c.maxIter + 1,
      params: (s, data, c) => {
        const { m, classes } = readPartial(data, name)
        return {
          classes,
          x: Array.from(m.data),
          n: m.n,
          d: m.d,
          distributions: Array.from(normaliseRows(s.f, classes.length)),
          kernel: c.kernel,
          gamma: c.gamma,
          nNeighbors: c.nNeighbors,
          iter: s.iter,
        }
      },
      verbs: (p) => graphVerbs(p),
      logs: { 'solver/change': Log(), 'solver/clampedMass': Log() },
    },
    { ...defaults, ...hyper },
  )
}

/**
 * Label propagation as an algorithm (`sklearn.semi_supervised.LabelPropagation`, Zhu and Ghahramani 2002): the label
 * distributions of the rows are repeatedly replaced by the weighted average of their neighbours', renormalised, with
 * the labelled rows clamped to their label. One diffusion step per step; logs `change` and `clampedMass`.
 *
 * @param data - Features, and targets with `-1` for the unlabelled rows.
 * @param hyper - The graph (`kernel`, `gamma`, `nNeighbors`) and the stopping rule (`maxIter`, `tol`).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { labelPropagationAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = labelPropagationAlgorithm({ x: [[0], [1], [10], [11]], y: [0, -1, 1, -1] }, { kernel: 'rbf', gamma: 0.5, nNeighbors: 7, alpha: 0.2, maxIter: 1000, tol: 1e-3 })
 * runAlgorithm(alg, { maxSteps: 1000 }).status // => 'converged'
 * ```
 */
export function labelPropagationAlgorithm(data: LearnData, hyper: GraphLabelHyper): Algorithm<GraphLabelState> {
  return graphAlgorithm('propagation', data, hyper)
}

/**
 * Label spreading as an algorithm (`sklearn.semi_supervised.LabelSpreading`, Zhou et al. 2004): the label distributions
 * move to `alpha S F + (1 - alpha) Y` with `S` the symmetrically normalised similarity, so the given labels are held
 * softly rather than clamped. One step per iteration; logs `change` and `clampedMass`.
 *
 * @param data - Features, and targets with `-1` for the unlabelled rows.
 * @param hyper - As {@link labelPropagationAlgorithm}, with `alpha` in (0, 1).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { labelSpreadingAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = labelSpreadingAlgorithm({ x: [[0], [1], [10], [11]], y: [0, -1, 1, -1] }, { kernel: 'rbf', gamma: 0.5, nNeighbors: 7, alpha: 0.2, maxIter: 30, tol: 1e-3 })
 * runAlgorithm(alg, { maxSteps: 30 }).status // => 'converged'
 * ```
 */
export function labelSpreadingAlgorithm(data: LearnData, hyper: GraphLabelHyper): Algorithm<GraphLabelState> {
  return graphAlgorithm('spreading', data, hyper)
}

/**
 * Label propagation (`sklearn.semi_supervised.LabelPropagation`): labels diffuse over an RBF or neighbour graph with the
 * given labels clamped. Predicting a new row averages the training rows' distributions with its graph weights. The
 * defaults are scikit-learn's (`rbf`, gamma 20, 7 neighbours, 1000 iterations, tolerance 1e-3).
 *
 * @param hyper - `kernel`, `gamma`, `nNeighbors`, `maxIter`, `tol`.
 * @returns The estimator; the fitted value has `transduction` and `labelDistributions` of the training rows.
 * @index estimator Semi-supervised labelling by propagation over a neighbour graph.
 * @example
 * ```ts
 * import { labelPropagation } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const f = labelPropagation({ gamma: 0.5 }).fit({ x: [[0], [1], [2], [10], [11], [12]], y: [0, -1, -1, 1, -1, -1] })
 * f.transduction // => [0, 0, 0, 1, 1, 1]
 * toFlat(f.predict([[1.5], [10.5]])) // => [0, 1]
 * ```
 */
export function labelPropagation(
  hyper: Partial<GraphLabelHyper> = {},
): IterativeEstimator<GraphLabelHyper, GraphLabelState, GraphLabelParams, GraphLabelVerbs> {
  return graphEstimator(
    'propagation',
    { kernel: 'rbf', gamma: 20, nNeighbors: 7, alpha: 0.2, maxIter: 1000, tol: 1e-3 },
    hyper,
  )
}

/**
 * Label spreading (`sklearn.semi_supervised.LabelSpreading`): the soft-clamped, symmetrically normalised version of
 * {@link labelPropagation}, robust to a few wrong labels. The defaults are scikit-learn's (`alpha` 0.2, 30 iterations).
 *
 * @param hyper - `kernel`, `gamma`, `nNeighbors`, `alpha`, `maxIter`, `tol`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { labelSpreading } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const f = labelSpreading({ gamma: 0.5 }).fit({ x: [[0], [1], [2], [10], [11], [12]], y: [0, -1, -1, 1, -1, -1] })
 * toFlat(f.predict([[1.5], [10.5]])) // => [0, 1]
 * ```
 */
export function labelSpreading(
  hyper: Partial<GraphLabelHyper> = {},
): IterativeEstimator<GraphLabelHyper, GraphLabelState, GraphLabelParams, GraphLabelVerbs> {
  return graphEstimator(
    'spreading',
    { kernel: 'rbf', gamma: 20, nNeighbors: 7, alpha: 0.2, maxIter: 30, tol: 1e-3 },
    hyper,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Laplacian-regularised least squares

/** The hyperparameters of {@link laplacianRlsClassifier} and {@link laplacianRlsRegressor}. */
export interface LaplacianRlsHyper extends KernelChoice {
  /** The ambient (RKHS) penalty gamma_A. */
  readonly gammaA: number
  /** The intrinsic (graph smoothness) penalty gamma_I: zero gives plain kernel ridge on the labelled rows. */
  readonly gammaI: number
  /** The neighbours of the graph. */
  readonly nNeighbors: number
  /** The scale of the graph's heat-kernel weights exp(-graphGamma |x - y|^2). */
  readonly graphGamma: number
  /** Use the symmetric normalised Laplacian instead of D - W. */
  readonly normalised: boolean
}

/** The state of the closed-form solve: nothing at step 0, the coefficients after it. */
export type LaplacianRlsState = { readonly alpha: number[]; readonly solved: boolean }

/** The fitted parameters of {@link laplacianRlsClassifier}. */
export type LaplacianRlsParams = {
  readonly classes: number[]
  readonly kernel: ResolvedKernel
  readonly x: number[]
  readonly n: number
  readonly d: number
  /** The expansion coefficients `[n, outputs]` row-major. */
  readonly alpha: number[]
  readonly outputs: number
}

/** What a fitted Laplacian RLS offers. */
export interface LaplacianRlsVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The expansion's value: `[m]` for two classes or a regression, `[m, k]` otherwise. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes?: readonly number[]
  readonly nFeaturesIn: number
}

/** The Laplacian `[n, n]` of the symmetric kNN graph with heat-kernel weights. */
function graphLaplacian(m: MatrixView, h: LaplacianRlsHyper): Float64Array {
  const n = m.n
  const k = Math.min(h.nNeighbors + 1, n)
  const nb = kNearest(fromView(m), undefined, k, { excludeSelf: false })
  const idx = nb.indices.data
  const w = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < k; c++) {
      const j = idx[i * k + c] ?? 0
      if (j === i) continue
      let d2 = 0
      for (let f = 0; f < m.d; f++) d2 += ((m.data[i * m.d + f] ?? 0) - (m.data[j * m.d + f] ?? 0)) ** 2
      const v = Math.exp(-h.graphGamma * d2)
      w[i * n + j] = v
      w[j * n + i] = v
    }
  const deg = Float64Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < n; j++) s += w[i * n + j] ?? 0
    return s
  })
  const l = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (h.normalised) {
        const z = Math.sqrt((deg[i] ?? 0) * (deg[j] ?? 0))
        l[i * n + j] = (i === j && (deg[i] ?? 0) > 0 ? 1 : 0) - (z > 0 ? (w[i * n + j] ?? 0) / z : 0)
      } else l[i * n + j] = (i === j ? (deg[i] ?? 0) : 0) - (w[i * n + j] ?? 0)
    }
  return l
}

/**
 * Laplacian RLS as an algorithm: step 0 is the zero function, step 1 the closed-form solve of
 * `(J K + gammaA l I + gammaI l / n^2 L K) alpha = Y`. Logs `labelledError` (mean squared error on the labelled rows)
 * and `smoothness` (the graph penalty `f^T L f` of the expansion over all rows).
 *
 * @param data - Features and targets (`-1` or NaN for unlabelled rows).
 * @param h - The kernel, the two penalties and the graph.
 * @param task - Classification (targets +-1, one-versus-rest) or regression.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { laplacianRlsAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const h = { kernel: 'rbf', gamma: 1, degree: 3, coef0: 1, gammaA: 0.1, gammaI: 1, nNeighbors: 1, graphGamma: 1, normalised: false } as const
 * runAlgorithm(laplacianRlsAlgorithm({ x: [[0], [1], [5]], y: [0, -1, 1] }, h), { maxSteps: 3 }).steps // => 1
 * ```
 */
export function laplacianRlsAlgorithm(
  data: LearnData,
  h: LaplacianRlsHyper,
  task: 'classification' | 'regression' = 'classification',
): Algorithm<LaplacianRlsState> {
  const who = task === 'classification' ? 'laplacianRlsClassifier' : 'laplacianRlsRegressor'
  const m = matrixView(data.x, who)
  const y = Float64Array.from(toFlat(asTensor(requireTargets(data, who))))
  const n = m.n
  const lab = labelledRows(y)
  const classes = task === 'classification' ? Array.from(new Set(lab.map((i) => y[i] ?? 0))).sort((a, b) => a - b) : []
  if (lab.length === 0) throw new MdmlError('value', `${who}: needs at least one labelled row`)
  const outputs = task === 'regression' ? 1 : classes.length === 2 ? 1 : classes.length
  const resolved = resolveKernel(h, fromView(m))
  const kmat = kernelMatrix(h, resolved, fromView(m))
  const l = graphLaplacian(m, h)
  // The system (J K + gammaA l I + gammaI l / n^2 L K) alpha = Y of Belkin, Niyogi and Sindhwani, with J selecting the labelled rows.
  const lk = dgemm(l, kmat, n, n, n)
  const a = new Float64Array(n * n)
  const nl = lab.length
  for (let i = 0; i < n; i++) {
    const isLab = !isUnlabelled(y[i] ?? UNLABELLED)
    for (let j = 0; j < n; j++) {
      a[i * n + j] = (isLab ? (kmat[i * n + j] ?? 0) : 0) + ((h.gammaI * nl) / (n * n)) * (lk[i * n + j] ?? 0)
    }
    a[i * n + i] = (a[i * n + i] ?? 0) + h.gammaA * nl
  }
  const target = new Float64Array(n * outputs)
  for (const i of lab) {
    if (task === 'regression') target[i] = y[i] ?? 0
    else if (outputs === 1) target[i] = classes.indexOf(y[i] ?? 0) === 1 ? 1 : -1
    else for (let c = 0; c < outputs; c++) target[i * outputs + c] = classes.indexOf(y[i] ?? 0) === c ? 1 : -1
  }
  return {
    name: 'laplacian-rls',
    config: { gammaA: h.gammaA, gammaI: h.gammaI, nNeighbors: h.nNeighbors, rows: n, labelled: nl },
    schema: channels({
      metrics: {
        labelledError: { description: 'mean squared error of the expansion on the labelled rows', direction: 'lower' },
        smoothness: { description: 'the graph penalty f^T L f of the expansion over all rows', direction: 'lower' },
      },
    }),
    init: () => ({ alpha: new Array<number>(n * outputs).fill(0), solved: false }),
    step: () => {
      const sol = Float64Array.from(
        toFlat(asTensor(solve(fromView({ n, d: n, data: a }), rowsTensor(target, [n, outputs])))),
      )
      return { alpha: Array.from(sol), solved: true }
    },
    done: (s) => (s.solved ? 'converged' : undefined),
    observe: (s) => {
      const f = dgemm(kmat, s.alpha, n, n, outputs)
      let err = 0
      for (const i of lab)
        for (let c = 0; c < outputs; c++) err += ((f[i * outputs + c] ?? 0) - (target[i * outputs + c] ?? 0)) ** 2
      const lf = dgemm(l, f, n, n, outputs)
      let pen = 0
      for (let i = 0; i < n * outputs; i++) pen += (f[i] ?? 0) * (lf[i] ?? 0)
      return { metrics: { labelledError: err / (nl * outputs), smoothness: pen } }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LaplacianRlsState,
  }
}

function laplacianRlsVerbs(
  task: 'classification' | 'regression',
  p: LaplacianRlsParams,
  h: LaplacianRlsHyper,
): LaplacianRlsVerbs {
  const train = rowsTensor(p.x, [p.n, p.d])
  const values = (x: TensorLike): { readonly f: Float64Array; readonly n: number } => {
    const q = matrixView(x, 'laplacianRls', { nFeatures: p.d })
    const kq = kernelMatrix(h, p.kernel, fromView(q), train)
    return { f: dgemm(kq, p.alpha, q.n, p.n, p.outputs), n: q.n }
  }
  const decisionFunction = (x: TensorLike): Tensor => {
    const { f, n } = values(x)
    return rowsTensor(f, p.outputs === 1 ? [n] : [n, p.outputs])
  }
  const predict = (x: TensorLike): Tensor => {
    const { f, n } = values(x)
    if (task === 'regression') return rowsTensor(f, [n])
    return rowsTensor(
      Array.from({ length: n }, (_, r) =>
        p.outputs === 1
          ? (p.classes[(f[r] ?? 0) > 0 ? 1 : 0] ?? 0)
          : (p.classes[argmaxRow(f, r * p.outputs, p.outputs)] ?? 0),
      ),
      [n],
    )
  }
  return {
    predict,
    decisionFunction,
    score: task === 'classification' ? accuracyOf(predict) : r2Of(predict),
    ...(task === 'classification' ? { classes: p.classes } : {}),
    nFeaturesIn: p.d,
  }
}

const LAP_DEFAULTS: LaplacianRlsHyper = {
  kernel: 'rbf',
  gamma: 'scale',
  degree: 3,
  coef0: 1,
  gammaA: 1e-2,
  gammaI: 1,
  nNeighbors: 6,
  graphGamma: 1,
  normalised: false,
}

function laplacianRls(
  task: 'classification' | 'regression',
  hyper: Partial<LaplacianRlsHyper>,
): IterativeEstimator<LaplacianRlsHyper, LaplacianRlsState, LaplacianRlsParams, LaplacianRlsVerbs> {
  const name = task === 'classification' ? 'laplacian-rls-classifier' : 'laplacian-rls-regressor'
  return iterative<LaplacianRlsHyper, LaplacianRlsState, LaplacianRlsParams, LaplacianRlsVerbs>(
    {
      name,
      kind: task === 'classification' ? 'classifier' : 'regressor',
      defaults: { ...LAP_DEFAULTS, ...hyper },
      hyperSpace: (c) =>
        dict({
          ...kernelSpace(c),
          gammaA: real(1e-6, 10, { log: true, default: c.gammaA, label: 'ambient penalty' }),
          gammaI: real(0, 100, { default: c.gammaI, label: 'intrinsic (graph) penalty' }),
        }),
      algorithm: (data, c) => laplacianRlsAlgorithm(data, c, task),
      maxSteps: () => 2,
      params: (s, data, c) => {
        const m = matrixView(data.x, name)
        const y = Float64Array.from(toFlat(asTensor(requireTargets(data, name))))
        const classes =
          task === 'classification'
            ? Array.from(new Set(labelledRows(y).map((i) => y[i] ?? 0))).sort((a, b) => a - b)
            : []
        return {
          classes,
          kernel: resolveKernel(c, fromView(m)),
          x: Array.from(m.data),
          n: m.n,
          d: m.d,
          alpha: s.alpha,
          outputs: task === 'regression' || classes.length === 2 ? 1 : classes.length,
        }
      },
      verbs: (p, c) => laplacianRlsVerbs(task, p, c),
      logs: { 'solver/labelledError': Log(), 'solver/smoothness': Log() },
    },
    { ...LAP_DEFAULTS, ...hyper },
  )
}

/**
 * Laplacian-regularised least-squares classification (Belkin, Niyogi and Sindhwani 2006, LapRLS): the kernel expansion
 * `f(x) = sum_i alpha_i k(x_i, x)` minimising the squared error on the labelled rows (targets +-1, one-versus-rest for
 * more classes) plus `gammaA` times its RKHS norm plus `gammaI` times the graph smoothness `f^T L f` over all rows,
 * solved in closed form. With `gammaI = 0` it is kernel ridge on the labelled rows alone. Unlabelled rows are `-1`.
 *
 * @param hyper - A kernel (`kernel`, `gamma`, `degree`, `coef0`), `gammaA`, `gammaI`, the graph (`nNeighbors`, `graphGamma`, `normalised`).
 * @returns The estimator; `decisionFunction` gives the expansion's value.
 * @example
 * ```ts
 * import { laplacianRlsClassifier } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [0.5], [1], [1.5], [6], [6.5], [7], [7.5]]
 * const f = laplacianRlsClassifier({ gamma: 0.5, nNeighbors: 2 }).fit({ x, y: [0, -1, -1, -1, 1, -1, -1, -1] })
 * toFlat(f.predict([[0.2], [7.2]])) // => [0, 1]
 * ```
 */
export function laplacianRlsClassifier(
  hyper: Partial<LaplacianRlsHyper> = {},
): IterativeEstimator<LaplacianRlsHyper, LaplacianRlsState, LaplacianRlsParams, LaplacianRlsVerbs> {
  return laplacianRls('classification', hyper)
}

/**
 * Laplacian-regularised least-squares regression: {@link laplacianRlsClassifier} with real targets, NaN marking the
 * unlabelled rows.
 *
 * @param hyper - As {@link laplacianRlsClassifier}.
 * @returns The estimator.
 * @example
 * ```ts
 * import { laplacianRlsRegressor } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3], [4]]
 * const f = laplacianRlsRegressor({ gamma: 0.5, gammaA: 1e-6, gammaI: 0, nNeighbors: 2 }).fit({ x, y: [0, Number.NaN, 2, Number.NaN, 4] })
 * toFlat(f.predict([[2]]))[0].toFixed(1) // => '2.0'
 * ```
 */
export function laplacianRlsRegressor(
  hyper: Partial<LaplacianRlsHyper> = {},
): IterativeEstimator<LaplacianRlsHyper, LaplacianRlsState, LaplacianRlsParams, LaplacianRlsVerbs> {
  return laplacianRls('regression', hyper)
}

// ---------------------------------------------------------------------------------------------------------------------
// Transductive SVM

/** The hyperparameters of {@link s3vm}. */
export interface S3vmHyper {
  /** The base binary classifier: it must accept sample weights (the weight scales C per row) and give a decision function. */
  readonly base: Estimator
  /** The weight of the unlabelled rows' margin violations at the end, relative to the labelled rows' (Joachims' C*). */
  readonly unlabelledWeight: number
  /** The weight they start at, doubled whenever no pair of labels is left to switch. */
  readonly startWeight: number
  /** The fraction of the unlabelled rows labelled positive at the start (default: the labelled share). */
  readonly positiveFraction: number | null
  /** The most label switches and weight increases. */
  readonly maxSteps: number
}

/** The state of {@link s3vmAlgorithm}: labels as 0/1 codes of the two classes, the weight, the model fitted for them. */
export type S3vmState = {
  /** Codes (0 or 1) of every row: the given ones fixed, the unlabelled ones as currently assigned. */
  readonly codes: number[]
  readonly weight: number
  readonly model: Json
  readonly switches: number
  /** The margin slack `max(0, 1 - y f)` summed over the labelled rows. */
  readonly slackLabelled: number
  /** The same over the unlabelled rows under their current labels. */
  readonly slackUnlabelled: number
  readonly stop: 'running' | 'converged'
}

/** The fitted parameters of {@link s3vm}. */
export type S3vmParams = S3vmState & { readonly classes: number[]; readonly d: number }

/** What a fitted S3VM offers. */
export interface S3vmVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The label of every training row: given, or the one the search ended with. */
  readonly transduction: readonly number[]
}

/**
 * The transductive SVM as an algorithm (Joachims 1999, TSVM): fit the base on the labelled rows, label the unlabelled
 * rows by the sign of its decision function (the top `positiveFraction` positive), then repeat: refit with the unlabelled
 * rows at weight `w`, switch the labels of the worst pair (a positive and a negative row whose margin slacks sum above
 * 2), and when no pair is left double `w` up to `unlabelledWeight`. One switch or doubling per step; logs
 * `unlabelledWeight`, `switches` and the slack sums.
 *
 * @param data - Features and targets with `-1` for the unlabelled rows; two classes among the labelled ones.
 * @param hyper - The base, the weights and the fraction.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { s3vmAlgorithm } from 'mdml-engine/learn'
 * import { svc } from 'mdml-engine/svm'
 * const alg = s3vmAlgorithm({ x: [[0], [1], [5], [6]], y: [0, -1, -1, 1] }, { base: svc.with({ kernel: 'linear' }), unlabelledWeight: 1, startWeight: 0.1, positiveFraction: null, maxSteps: 50 })
 * alg.name // => 's3vm'
 * ```
 */
export function s3vmAlgorithm(data: LearnData, hyper: S3vmHyper): Algorithm<S3vmState> {
  const who = 's3vm'
  const { m, y, classes } = readPartial(data, who)
  if (classes.length !== 2) throw new MdmlError('value', `${who}: needs exactly two classes among the labelled rows`)
  const n = m.n
  const lab = labelledRows(y)
  const unl = unlabelledRows(y)
  const xAll = fromView(m)
  const given = Int8Array.from(y, (v) => (isUnlabelled(v) ? -1 : classes.indexOf(v)))
  const fraction = hyper.positiveFraction ?? lab.filter((i) => given[i] === 1).length / lab.length
  const fitFor = (codes: readonly number[], weight: number, key: Key): Json => {
    const labels = Float64Array.from(codes, (c) => classes[c] ?? 0)
    const w = Float64Array.from({ length: n }, (_, i) => (given[i] === -1 ? weight : 1))
    const rows = weight > 0 ? Array.from({ length: n }, (_, i) => i) : lab
    const sub = takeData({ ...data, weights: rowsTensor(w, [n]) }, rows)
    return hyper.base.fit(
      {
        ...sub,
        y: rowsTensor(
          rows.map((r) => labels[r] ?? 0),
          [rows.length],
        ),
      },
      { key, explain: false },
    ).params
  }
  const slacks = (model: Json, codes: readonly number[]): Float64Array => {
    const f = confidence(hyper.base.revive(model), xAll)
    return Float64Array.from(f, (v, i) => Math.max(0, 1 - (codes[i] === 1 ? v : -v)))
  }
  const sums = (xi: Float64Array): { readonly lab: number; readonly unl: number } => ({
    lab: lab.reduce((a, i) => a + (xi[i] ?? 0), 0),
    unl: unl.reduce((a, i) => a + (xi[i] ?? 0), 0),
  })
  return {
    name: 's3vm',
    config: {
      base: hyper.base.name,
      unlabelledWeight: hyper.unlabelledWeight,
      startWeight: hyper.startWeight,
      rows: n,
    },
    schema: channels({
      metrics: {
        unlabelledWeight: { description: 'the weight of the unlabelled rows’ margin violations' },
        switches: { description: 'label switches so far', direction: 'higher' },
        slackLabelled: { description: 'margin slack of the labelled rows', direction: 'lower' },
        slackUnlabelled: {
          description: 'margin slack of the unlabelled rows under their current labels',
          direction: 'lower',
        },
      },
    }),
    init: (key) => {
      const supervised = fitFor(
        Array.from(given, (c) => Math.max(0, c)),
        0,
        foldIn(key, 0),
      )
      const f = confidence(hyper.base.revive(supervised), tensorOfRows(m, unl))
      const positives = Math.round(fraction * unl.length)
      const order = unl.map((_, r) => r).sort((a, b) => (f[b] ?? 0) - (f[a] ?? 0) || a - b)
      const codes = Array.from(given, (c) => Math.max(0, c))
      order.forEach((r, rank) => {
        codes[unl[r] ?? 0] = rank < positives ? 1 : 0
      })
      const weight = Math.min(hyper.startWeight, hyper.unlabelledWeight)
      const model = fitFor(codes, weight, foldIn(key, 1))
      const s = sums(slacks(model, codes))
      return { codes, weight, model, switches: 0, slackLabelled: s.lab, slackUnlabelled: s.unl, stop: 'running' }
    },
    step: (s, ctx) => {
      const xi = slacks(s.model, s.codes)
      let best: { i: number; j: number; total: number } | undefined
      for (const i of unl)
        if (s.codes[i] === 1 && (xi[i] ?? 0) > 0)
          for (const j of unl)
            if (s.codes[j] === 0 && (xi[j] ?? 0) > 0) {
              const total = (xi[i] ?? 0) + (xi[j] ?? 0)
              if (total > 2 && (best === undefined || total > best.total)) best = { i, j, total }
            }
      let codes = s.codes
      let weight = s.weight
      let switches = s.switches
      if (best !== undefined) {
        codes = [...s.codes]
        codes[best.i] = 0
        codes[best.j] = 1
        switches++
      } else if (weight < hyper.unlabelledWeight) weight = Math.min(2 * weight, hyper.unlabelledWeight)
      else return { ...s, stop: 'converged' }
      const model = fitFor(
        codes,
        weight,
        foldIn(ctx.key, switches + 1000 * Math.round(Math.log2(weight / hyper.startWeight + 1))),
      )
      const sm = sums(slacks(model, codes))
      return { codes, weight, model, switches, slackLabelled: sm.lab, slackUnlabelled: sm.unl, stop: 'running' }
    },
    done: (s, t) => (s.stop === 'converged' ? 'converged' : t >= hyper.maxSteps ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        unlabelledWeight: s.weight,
        switches: s.switches,
        slackLabelled: s.slackLabelled,
        slackUnlabelled: s.slackUnlabelled,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as S3vmState,
  }
}

/**
 * The transductive SVM over any weighted binary classifier (Joachims 1999; Bennett and Demiriz 1998 for the idea): the
 * margin pushed through the low-density gap by labelling the unlabelled rows and improving the labelling pair by pair.
 * The fitted value is the final base model; `transduction` holds the labels the search ended with.
 *
 * @param base - A binary classifier with sample weights and a decision function, e.g. `svc` of `mdml-engine/svm`.
 * @param hyper - `unlabelledWeight`, `startWeight`, `positiveFraction`, `maxSteps`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { s3vm } from 'mdml-engine/learn'
 * import { svc } from 'mdml-engine/svm'
 * const f = s3vm(svc.with({ kernel: 'linear', C: 10 })).fit({ x: [[0], [0.5], [1], [5], [5.5], [6]], y: [0, -1, -1, -1, -1, 1] })
 * f.transduction // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export function s3vm(
  base: Estimator,
  hyper: Partial<Omit<S3vmHyper, 'base'>> = {},
): IterativeEstimator<S3vmHyper, S3vmState, S3vmParams, S3vmVerbs> {
  const h: S3vmHyper = { base, unlabelledWeight: 1, startWeight: 0.01, positiveFraction: null, maxSteps: 200, ...hyper }
  return iterative<S3vmHyper, S3vmState, S3vmParams, S3vmVerbs>(
    {
      name: 's3vm',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          unlabelledWeight: real(1e-3, 10, { log: true, default: c.unlabelledWeight, label: 'unlabelled weight' }),
        }),
      algorithm: (data, c) => s3vmAlgorithm(data, c),
      maxSteps: (c) => c.maxSteps + 1,
      params: (s, data) => ({ ...s, classes: readPartial(data, 's3vm').classes, d: matrixView(data.x, 's3vm').d }),
      verbs: (p, c) => {
        const f = c.base.revive(p.model)
        return {
          predict: requireVerb(f, 'predict', 's3vm'),
          decisionFunction: requireVerb(f, 'decisionFunction', 's3vm'),
          ...(f.predictProba === undefined ? {} : { predictProba: f.predictProba }),
          score: accuracyOf(requireVerb(f, 'predict', 's3vm')),
          classes: f.classes ?? p.classes,
          nFeaturesIn: p.d,
          transduction: p.codes.map((code) => p.classes[code] ?? 0),
        }
      },
      logs: {
        'solver/unlabelledWeight': Log(),
        'solver/switches': Log(),
        'solver/slackLabelled': Log(),
        'solver/slackUnlabelled': Log(),
      },
    },
    h,
  )
}
