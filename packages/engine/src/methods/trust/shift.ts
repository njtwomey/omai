/**
 * Dataset shift and its detection (Rabanser, Günnemann and Lipton, 2019): a classifier fitted on reference rows meets
 * current rows shifted by growing amounts in their features or labels, and the study records how accuracy falls and how
 * well per-feature Kolmogorov-Smirnov tests and a domain classifier detect the shift.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { curveSet, type Explained } from '../../lib/exposition/index.ts'
import { type Estimator, type LearnData, matrixView, requireVerb, targetVector } from '../../lib/learn/index.ts'
import { crossValPredict } from '../../lib/learn/model-selection/index.ts'
import { rocAucScore } from '../../lib/metrics/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../lib/random/index.ts'
import { ks2samp } from '../../lib/stats/tests/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { splitData } from './_split.ts'

/** The kinds of shift {@link shiftStudy} applies to the current rows. */
export type ShiftKind = 'covariate' | 'concept'

/** Options of {@link shiftStudy}. */
export interface ShiftStudyOptions {
  /** `covariate`: the current rows' features move; `concept`: their labels flip, the features stay (default covariate). */
  readonly kind?: ShiftKind
  /** Shift sizes: standard deviations added to every feature, or the share of labels flipped (default 0 to 1.5 or 0.5). */
  readonly magnitudes?: readonly number[]
  readonly key?: Key
}

/** One shift size of {@link shiftStudy}. */
export interface ShiftPoint {
  readonly magnitude: number
  /** The smallest per-feature Kolmogorov–Smirnov p-value (Bonferroni-corrected) between reference and current. */
  readonly ksPValue: number
  /** The cross-validated AUC of a classifier telling reference from current rows (0.5: indistinguishable). */
  readonly domainAuc: number
  /** The model's accuracy on the current rows. */
  readonly accuracy: number
}

/**
 * Dataset shift and its detection (Rabanser, Günnemann and Lipton, 2019): a classifier fitted on reference rows, then
 * the current rows shifted by growing amounts, either in their features (covariate shift) or in their labels (concept
 * shift). Two detectors watch the features only: per-feature two-sample Kolmogorov–Smirnov tests and a domain classifier
 * that tries to tell the two sets apart; the model's accuracy shows what the shift costs. Covariate shift is caught by
 * both (strongly when the shift breaks a correlation between features, as a shift of every feature by its own standard
 * deviation does to redundant features); concept shift leaves the features alone, so neither sees it while accuracy falls. Exposition: a `curves`
 * record against the shift's size.
 *
 * @param estimator - A classifier.
 * @param data - Labelled classification rows.
 * @param options - Kind, sizes and key.
 * @returns The points, with the record.
 */
export function shiftStudy(
  estimator: Estimator,
  data: LearnData,
  options: ShiftStudyOptions = {},
): Explained<{ readonly points: readonly ShiftPoint[] }> {
  if (data.y === undefined) throw new MdmlError('value', 'shiftStudy: the data needs labels')
  const kind = options.kind ?? 'covariate'
  const key = options.key ?? makeKey(0)
  const magnitudes =
    options.magnitudes ?? (kind === 'covariate' ? [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5] : [0, 0.1, 0.2, 0.3, 0.4, 0.5])
  const [reference, current] = splitData(data, [0.5, 0.5], foldIn(key, 0))
  if (reference === undefined || current === undefined) throw new MdmlError('value', 'shiftStudy: no rows')
  const model = estimator.fit(reference, { key: foldIn(key, 1), explain: false })
  const predict = requireVerb(model, 'predict', 'shiftStudy')
  const r = matrixView(reference.x, 'shiftStudy')
  const c = matrixView(current.x, 'shiftStudy')
  const sd = Array.from({ length: r.d }, (_, j) => {
    let s = 0
    let s2 = 0
    for (let i = 0; i < r.n; i++) {
      const v = r.data[i * r.d + j] ?? 0
      s += v
      s2 += v * v
    }
    return Math.sqrt(Math.max(0, s2 / r.n - (s / r.n) ** 2))
  })
  const y = Array.from(targetVector(current.y as NonNullable<LearnData['y']>, 'shiftStudy'))
  const classes = [...new Set(y)].sort((a, b) => a - b)
  const flips = uniform(foldIn(key, 2), [c.n]).data
  const points = magnitudes.map((mag, s) => {
    const shifted = Float64Array.from(c.data, (v, k) => (kind === 'covariate' ? v + mag * (sd[k % c.d] ?? 1) : v))
    const labels = y.map((v, i) => {
      if (kind !== 'concept' || (flips[i] ?? 1) >= mag) return v
      const at = classes.indexOf(v)
      return classes[(at + 1) % classes.length] ?? v
    })
    const x = fromStorage(shifted, [c.n, c.d], 'float64')
    let ks = 1
    for (let j = 0; j < c.d; j++) {
      const a = Array.from({ length: r.n }, (_, i) => r.data[i * r.d + j] ?? 0)
      const b = Array.from({ length: c.n }, (_, i) => shifted[i * c.d + j] ?? 0)
      ks = Math.min(ks, Math.min(1, ks2samp(a, b).pValue * c.d))
    }
    // The domain classifier: reference rows labelled 0, current rows 1, scored out of fold.
    const both = fromStorage(Float64Array.from([...r.data, ...shifted]), [r.n + c.n, c.d], 'float64')
    const domain = [...new Array<number>(r.n).fill(0), ...new Array<number>(c.n).fill(1)]
    const proba = toFlat(
      crossValPredict(
        logisticRegression(),
        { x: both, y: domain },
        { cv: 5, method: 'predictProba', key: foldIn(key, 10 + s) },
      ),
    )
    const auc = rocAucScore(
      domain,
      Array.from({ length: domain.length }, (_, i) => proba[2 * i + 1] ?? 0),
    ).estimate.value
    const pred = toFlat(predict(x))
    const accuracy = labels.filter((v, i) => v === pred[i]).length / Math.max(1, labels.length)
    return { magnitude: mag, ksPValue: ks, domainAuc: auc, accuracy }
  })
  return {
    points,
    exposition: [
      curveSet({
        x: magnitudes,
        xLabel: kind === 'covariate' ? 'shift (standard deviations)' : 'share of labels flipped',
        yLabel: 'accuracy, AUC',
        y2Label: '−log10 p (KS)',
        series: [
          { name: 'model accuracy on current rows', y: points.map((p) => p.accuracy) },
          { name: 'domain classifier AUC', y: points.map((p) => p.domainAuc) },
          { name: 'KS test, −log10 p', y: points.map((p) => -Math.log10(Math.max(1e-300, p.ksPValue))), axis: 'right' },
        ],
      }),
    ],
  }
}
