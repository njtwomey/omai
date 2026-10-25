/**
 * A robustness study of a fitted classifier: how its accuracy degrades as Gaussian noise of growing scale is added to
 * every feature, or to one feature at a time to show which it leans on, and as growing shares of its inputs are
 * corrupted.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { curveSet, type Explained } from '../../lib/exposition/index.ts'
import {
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  requireVerb,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../../lib/random/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'

/** Options of {@link robustnessStudy}. */
export interface RobustnessStudyOptions {
  /** Noise scales, in each feature's standard deviations (default 0 to 2 in 9 steps). */
  readonly noise?: readonly number[]
  /** Shares of missing values, filled with the feature's mean (default 0 to 0.8 in 9 steps). */
  readonly missing?: readonly number[]
  /** Noisy copies averaged per scale (default 3). */
  readonly repeats?: number
  readonly key?: Key
}

/** The result of {@link robustnessStudy}. */
export interface RobustnessStudy {
  /** Accuracy with every feature noisy, at each noise scale. */
  readonly noise: readonly number[]
  /** Accuracy with only one feature noisy, per feature and scale. */
  readonly perFeature: readonly (readonly number[])[]
  /** Accuracy with a share of values missing (mean-imputed), at each share. */
  readonly missing: readonly number[]
}

/**
 * How a fitted classifier's accuracy degrades under perturbation: Gaussian noise added to every feature (or to one
 * at a time, which shows the features it leans on) at growing scales, and growing shares of values missing and filled
 * with the feature's mean. Exposition: a `curves` record of accuracy against the noise scale (all features and each
 * one) and one against the missing share.
 *
 * @param model - A fitted classifier.
 * @param data - Labelled rows to perturb.
 * @param options - Noise scales, missing shares, repeats and key.
 * @returns The accuracies, with the records.
 */
export function robustnessStudy(
  model: Fitted,
  data: LearnData,
  options: RobustnessStudyOptions = {},
): Explained<RobustnessStudy> {
  if (data.y === undefined) throw new WhiteboxError('value', 'robustnessStudy: the data needs labels')
  const m = matrixView(data.x, 'robustnessStudy')
  const y = Array.from(targetVector(data.y, 'robustnessStudy', m.n))
  const key = options.key ?? makeKey(0)
  const scales = options.noise ?? Array.from({ length: 9 }, (_, i) => i / 4)
  const shares = options.missing ?? Array.from({ length: 9 }, (_, i) => i / 10)
  const repeats = options.repeats ?? 3
  const predict = requireVerb(model, 'predict', 'robustnessStudy')
  const stats = Array.from({ length: m.d }, (_, j) => {
    let s = 0
    let s2 = 0
    for (let i = 0; i < m.n; i++) {
      const v = m.data[i * m.d + j] ?? 0
      s += v
      s2 += v * v
    }
    const mean = s / m.n
    return { mean, sd: Math.sqrt(Math.max(0, s2 / m.n - mean * mean)) }
  })
  const accuracy = (x: Float64Array): number => {
    const p = toFlat(predict(fromStorage(x, [m.n, m.d], 'float64')))
    return y.filter((v, i) => v === p[i]).length / m.n
  }
  const noisy = (scale: number, only: number | undefined, k: Key): number => {
    let acc = 0
    for (let r = 0; r < repeats; r++) {
      const z = normal(foldIn(k, r), [m.n * m.d]).data
      acc += accuracy(
        Float64Array.from(m.data, (v, idx) => {
          const j = idx % m.d
          return only === undefined || only === j ? v + scale * (stats[j]?.sd ?? 1) * (z[idx] ?? 0) : v
        }),
      )
    }
    return acc / repeats
  }
  const all = scales.map((s, t) => noisy(s, undefined, foldIn(key, t)))
  const perFeature = stats.map((_, j) => scales.map((s, t) => noisy(s, j, foldIn(key, 100 * (j + 1) + t))))
  const missing = shares.map((share, t) => {
    let acc = 0
    for (let r = 0; r < repeats; r++) {
      const u = uniform(foldIn(key, 10_000 + 10 * t + r), [m.n * m.d]).data
      acc += accuracy(
        Float64Array.from(m.data, (v, idx) => ((u[idx] ?? 1) < share ? (stats[idx % m.d]?.mean ?? 0) : v)),
      )
    }
    return acc / repeats
  })
  const names = featureNamesOf(data, m.d)
  const shown = perFeature.slice(0, 8)
  return {
    noise: all,
    perFeature,
    missing,
    exposition: [
      curveSet({
        x: scales,
        xLabel: 'noise (standard deviations of each feature)',
        yLabel: 'accuracy',
        series: [
          { name: 'every feature noisy', y: all, group: 'all' },
          ...shown.map((acc, j) => ({ name: `only ${names[j] ?? `x${j}`}`, y: acc })),
        ],
      }),
      curveSet({
        x: shares,
        xLabel: 'share of values missing (mean-filled)',
        yLabel: 'accuracy',
        series: [{ name: 'accuracy', y: missing }],
      }),
    ],
  }
}
