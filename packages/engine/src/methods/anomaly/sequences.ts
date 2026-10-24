import { WhiteboxError } from '../../lib/base/index.ts'
import { type AnomalyScores, anomalyScores } from '../../lib/exposition/index.ts'
import { asTensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { discords, stump } from '../timeseries/index.ts'

/**
 * Time-series discords as anomaly scores (Yeh et al. 2016): each window's score is its matrix-profile distance to its
 * nearest non-trivial match (timeseries' STUMP); the threshold is the k-th discord's score, so the top-k discords are
 * flagged. TODO: forecasting-residual detectors.
 *
 * @param series - The series `[n]`.
 * @param m - The window length.
 * @param k - Discords flagged (default 1).
 * @returns An `anomalyScores` record over the windows (scores by window start), with the discord starts as evidence.
 */
export function matrixProfileDiscords(series: TensorLike, m: number, k = 1): AnomalyScores {
  const t = toFlat(asTensor(series))
  if (!(Number.isInteger(m) && m >= 3 && m <= t.length / 2))
    throw new WhiteboxError('value', `matrixProfileDiscords: the window must be in 3..${Math.floor(t.length / 2)}`)
  const profile = stump(t, m)
  const scores = Array.from(profile.P, (v) => (Number.isFinite(v) ? v : 0))
  const found = discords(profile.P, m, k)
  const flagged = new Set(found.indices)
  return anomalyScores({
    method: 'matrixProfileDiscords',
    scores,
    threshold: Math.min(...found.distances) - 1e-12,
    evidence: { name: 'isDiscord', values: scores.map((_, i) => (flagged.has(i) ? 1 : 0)) },
  })
}
