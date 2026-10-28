// Scores and records shared by the ensembles: the score of a running sum of member outputs, class alignment of a
// member's probabilities, and the `ensemble/stages` projection built from what a fit logged.
import { argmaxOf } from '../base/index.ts'
import { type EnsembleStages, ensembleStages } from '../exposition/index.ts'
import { type Fitted, requireVerb } from '../learn/estimator.ts'
import { accuracyScore, r2Score } from '../metrics/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { metricSeries, type Run, tensorSeries } from '../trace/index.ts'

/** Accuracy of the argmax of summed class scores (k > 0), or R^2 of sum / total for values (k = 0), over some rows. */
export function scoreOfSum(
  sum: ArrayLike<number>,
  total: number | ArrayLike<number>,
  y: ArrayLike<number>,
  k: number,
  rows?: readonly number[],
): number {
  const all = rows ?? Array.from({ length: y.length }, (_, i) => i)
  if (all.length < 2) return Number.NaN
  const truth = all.map((i) => y[i] ?? 0)
  if (k > 0) {
    const pred = all.map((i) => argmaxOf(Array.from({ length: k }, (_, c) => sum[i * k + c] ?? 0)))
    return accuracyScore(truth, pred).estimate.value
  }
  const t = (i: number): number => (typeof total === 'number' ? total : (total[i] ?? 1))
  return r2Score(
    truth,
    all.map((i) => (sum[i] ?? 0) / t(i)),
  ).estimate.value
}

/** A member's class probabilities `[m * k]` aligned to `classes` (one-hot of its predictions when it has none). */
export function memberProba(f: Fitted, x: TensorLike, classes: readonly number[]): Float64Array {
  const k = classes.length
  if (f.predictProba !== undefined && f.classes !== undefined) {
    const p = toFlat(f.predictProba(x))
    const own = f.classes
    const m = p.length / own.length
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++)
      for (const [c, label] of own.entries()) out[i * k + classes.indexOf(label)] = p[i * own.length + c] ?? 0
    return out
  }
  const pred = toFlat(requireVerb(f, 'predict', 'ensemble')(x))
  const out = new Float64Array(pred.length * k)
  pred.forEach((v, i) => {
    out[i * k + classes.indexOf(v)] = 1
  })
  return out
}

/** What the `ensemble/stages` projection reads from a tape. */
export interface StagesSource {
  readonly method: string
  readonly scoreName: string
  readonly higherIsBetter: boolean
  /** The metric paths: member weights, the training score, the validation or out-of-bag score. */
  readonly weight?: string
  readonly train: string
  readonly valid?: string
  /** Tensor paths: the rows each member saw (counts per row), and what it was fitted to (weights or residuals). */
  readonly bootstrap?: string
  readonly target?: string
}

/**
 * The `ensembleStages` record of a fit from the paths it logged, or `undefined` when the tape holds no training score.
 */
export function stagesFromRun(run: Run, source: StagesSource): EnsembleStages | undefined {
  const train = metricSeries(run, source.train).values
  if (train.length === 0) return undefined
  const weights = source.weight === undefined ? [] : metricSeries(run, source.weight).values
  const valid = source.valid === undefined ? [] : metricSeries(run, source.valid).values
  const bags = source.bootstrap === undefined ? [] : tensorSeries(run, source.bootstrap)
  const targets = source.target === undefined ? [] : tensorSeries(run, source.target)
  const bootstraps = bags.flatMap((e) =>
    e.values === undefined ? [] : [e.values.flatMap((c, i) => (c > 0 ? [i] : []))],
  )
  const fitted = targets.flatMap((e) => (e.values === undefined ? [] : [[...e.values]]))
  const n = train.length
  return ensembleStages({
    method: source.method,
    weights: weights.length === n ? weights : train.map(() => 1),
    trainScores: train,
    scoreName: source.scoreName,
    higherIsBetter: source.higherIsBetter,
    ...(valid.length === n ? { validationScores: valid } : {}),
    ...(bootstraps.length === n ? { bootstraps } : {}),
    ...(fitted.length === n ? { targets: fitted } : {}),
  })
}
