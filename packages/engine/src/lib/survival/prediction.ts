/**
 * Scoring survival predictions at a time horizon when some subjects are censored: the Kaplan-Meier estimate of the
 * censoring distribution that inverse-probability-of-censoring weights come from, the Brier score of predicted
 * survival probabilities (Graf et al.) and its integral over a range of times, and the cumulative/dynamic AUC of a
 * risk score (Uno et al.). With no censoring the weights are 1 and they reduce to the plain Brier score and AUC.
 */

import { OmaiError } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type SurvivalSample, survivalSample } from './data.ts'

/**
 * The Kaplan-Meier estimate of the censoring distribution `G(t) = P(C > t)`, treating the censorings as the events
 * (an event at a tied time counts as before the censoring), as a function of a time that returns the left limit
 * `G(t-)`: the probability of not having been censored strictly before `t`.
 *
 * @param sample - The survival sample.
 * @returns The left-limit function.
 * @example
 * ```ts
 * import { censoringSurvival, survivalSample } from 'omai-engine/survival'
 * const g = censoringSurvival(survivalSample([1, 2, 3, 4], [1, 0, 1, 0]))
 * g(2) // => 1
 * g(2.5) // => 0.666667
 * ```
 */
export function censoringSurvival(sample: SurvivalSample): (t: number) => number {
  const order = Array.from({ length: sample.n }, (_, i) => i).sort(
    (a, b) => (sample.time[a] ?? 0) - (sample.time[b] ?? 0) || (sample.event[b] ?? 0) - (sample.event[a] ?? 0),
  )
  const times: number[] = []
  const levels: number[] = []
  let g = 1
  let i = 0
  while (i < order.length) {
    const t = sample.time[order[i] ?? 0] ?? 0
    let censored = 0
    let j = i
    while (j < order.length && (sample.time[order[j] ?? 0] ?? 0) === t) {
      if (sample.event[order[j] ?? 0] === 0) censored++
      j++
    }
    // Everyone at or after position i is at risk of being censored at t, those with an event at t included.
    if (censored > 0) {
      g *= 1 - censored / (sample.n - i)
      times.push(t)
      levels.push(g)
    }
    i = j
  }
  return (t) => {
    let value = 1
    for (let k = 0; k < times.length && (times[k] ?? Number.POSITIVE_INFINITY) < t; k++) value = levels[k] ?? value
    return value
  }
}

/**
 * The inverse-probability-of-censoring weighted Brier score at a horizon `t` (Graf et al. 1999): the mean squared error
 * between the predicted survival probability and the status at `t`, with subjects who failed before `t` weighted by
 * `1 / G(T_i-)` and subjects still at risk at `t` by `1 / G(t)`, and subjects censored before `t` dropped. Lower is
 * better; predicting 1/2 for everyone scores 1/4.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param survival - The predicted survival probability of every subject at `t`.
 * @param t - The horizon.
 * @returns The Brier score.
 * @throws OmaiError with code `value` when the censoring distribution has no mass left at `t`.
 * @example
 * ```ts
 * import { brierScore } from 'omai-engine/survival'
 * brierScore([1, 2, 3, 4], undefined, [0.1, 0.4, 0.6, 0.9], 2.5) // => 0.085
 * ```
 */
export function brierScore(time: TensorLike, event: TensorLike | undefined, survival: TensorLike, t: number): number {
  const sample = survivalSample(time, event, 'brierScore')
  const s = toFlat(asTensor(survival))
  if (s.length !== sample.n) throw new OmaiError('shape', `brierScore: ${s.length} predictions for ${sample.n} times`)
  const g = censoringSurvival(sample)
  const gt = g(t)
  let total = 0
  for (let i = 0; i < sample.n; i++) {
    const ti = sample.time[i] ?? 0
    const si = s[i] ?? 0
    if (ti <= t && sample.event[i] === 1) {
      const w = g(ti)
      if (!(w > 0)) throw new OmaiError('value', 'brierScore: the censoring distribution has no mass at an event time')
      total += (si * si) / w
    } else if (ti > t) {
      if (!(gt > 0)) throw new OmaiError('value', 'brierScore: the censoring distribution has no mass at the horizon')
      total += (1 - si) ** 2 / gt
    }
  }
  return total / sample.n
}

/**
 * The integrated Brier score: the Brier score over a grid of horizons, integrated by the trapezoidal rule and divided
 * by the length of the range.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param survival - Predicted survival probabilities, `n x m`: every subject at every horizon.
 * @param times - The `m` horizons, increasing.
 * @returns The integrated score.
 * @example
 * ```ts
 * import { integratedBrierScore } from 'omai-engine/survival'
 * integratedBrierScore([1, 2, 3, 4], undefined, [[0.9, 0.5], [0.8, 0.4], [0.9, 0.7], [1, 0.9]], [0.5, 2.5]) // => 0.07125
 * ```
 */
export function integratedBrierScore(
  time: TensorLike,
  event: TensorLike | undefined,
  survival: TensorLike,
  times: readonly number[],
): number {
  const sample = survivalSample(time, event, 'integratedBrierScore')
  const m = times.length
  if (m < 2) throw new OmaiError('value', 'integratedBrierScore: needs at least two horizons')
  const flat = toFlat(asTensor(survival))
  if (flat.length !== sample.n * m)
    throw new OmaiError('shape', `integratedBrierScore: expected ${sample.n} x ${m} predictions, got ${flat.length}`)
  const scores = times.map((t, k) =>
    brierScore(
      sample.time,
      sample.event,
      Array.from({ length: sample.n }, (_, i) => flat[i * m + k] ?? 0),
      t,
    ),
  )
  let area = 0
  for (let k = 1; k < m; k++)
    area += (((scores[k] ?? 0) + (scores[k - 1] ?? 0)) / 2) * ((times[k] ?? 0) - (times[k - 1] ?? 0))
  return area / ((times[m - 1] ?? 0) - (times[0] ?? 0))
}

/**
 * The cumulative/dynamic AUC at a horizon `t` (Uno et al. 2007): the probability that a subject who failed by `t` has a
 * higher risk score than one who survives past `t`, with the failures weighted by `1 / G(T_i-)^2` to undo the
 * censoring. 0.5 is a coin flip and 1 a perfect ranking at that horizon.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param risk - The risk score of every subject (higher: earlier failure).
 * @param t - The horizon.
 * @returns The AUC.
 * @throws OmaiError with code `value` when no subject has failed by `t` or none survives past it.
 * @example
 * ```ts
 * import { cumulativeDynamicAuc } from 'omai-engine/survival'
 * cumulativeDynamicAuc([1, 2, 3, 4], undefined, [4, 3, 2, 1], 2.5) // => 1
 * ```
 */
export function cumulativeDynamicAuc(
  time: TensorLike,
  event: TensorLike | undefined,
  risk: TensorLike,
  t: number,
): number {
  const sample = survivalSample(time, event, 'cumulativeDynamicAuc')
  const r = toFlat(asTensor(risk))
  if (r.length !== sample.n)
    throw new OmaiError('shape', `cumulativeDynamicAuc: ${r.length} scores for ${sample.n} times`)
  const g = censoringSurvival(sample)
  let numerator = 0
  let weightSum = 0
  let survivors = 0
  for (let j = 0; j < sample.n; j++) if ((sample.time[j] ?? 0) > t) survivors++
  for (let i = 0; i < sample.n; i++) {
    if (!(sample.event[i] === 1 && (sample.time[i] ?? 0) <= t)) continue
    const gi = g(sample.time[i] ?? 0)
    if (!(gi > 0))
      throw new OmaiError('value', 'cumulativeDynamicAuc: the censoring distribution has no mass at an event time')
    const w = 1 / (gi * gi)
    weightSum += w
    for (let j = 0; j < sample.n; j++) {
      if (!((sample.time[j] ?? 0) > t)) continue
      const a = r[i] ?? 0
      const b = r[j] ?? 0
      numerator += w * (a > b ? 1 : a === b ? 0.5 : 0)
    }
  }
  if (weightSum === 0 || survivors === 0)
    throw new OmaiError('value', 'cumulativeDynamicAuc: needs a failure by t and a survivor past it')
  return numerator / (weightSum * survivors)
}
