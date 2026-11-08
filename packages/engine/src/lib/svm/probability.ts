/**
 * Class probabilities from one-versus-one machines: Platt scaling of each pair's decision values and the coupling of
 * the pairwise probabilities into one distribution over the classes (Wu, Lin and Weng's second method, libsvm's).
 */

import { OmaiError } from '../base/index.ts'

/**
 * Couple pairwise probabilities into class probabilities (libsvm's `multiclass_probability`, the second method of Wu,
 * Lin and Weng 2004): the distribution p minimising sum_i sum_{j != i} (r_ji p_i - r_ij p_j)^2 subject to sum p = 1,
 * found by the fixed-point iteration of the paper.
 *
 * @param r - `r[i][j]` is the probability of class i given that the class is i or j (`r[j][i] = 1 - r[i][j]`).
 * @returns The class probabilities.
 * @throws OmaiError `shape` unless `r` is square with at least two classes.
 * @concept support-vector-machines
 * @reference T.-F. Wu, C.-J. Lin and R. C. Weng, Probability estimates for multi-class classification by pairwise
 * coupling (2004).
 * @example
 * ```ts
 * import { pairwiseCoupling } from 'omai-engine/svm'
 * const p = pairwiseCoupling([[0, 0.8, 0.9], [0.2, 0, 0.6], [0.1, 0.4, 0]])
 * p.map((v) => Math.round(v * 1000) / 1000) // => [0.733, 0.175, 0.092]
 * ```
 */
export function pairwiseCoupling(r: readonly (readonly number[])[]): number[] {
  const k = r.length
  if (k < 2 || r.some((row) => row.length !== k))
    throw new OmaiError('shape', 'pairwiseCoupling: r must be k x k, k >= 2')
  const at = (i: number, j: number): number => r[i]?.[j] ?? 0
  const Q = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  for (let t = 0; t < k; t++) {
    let diag = 0
    for (let j = 0; j < k; j++) {
      if (j === t) continue
      diag += at(j, t) * at(j, t)
      ;(Q[t] as number[])[j] = -at(j, t) * at(t, j)
    }
    ;(Q[t] as number[])[t] = diag
  }
  const q = (a: number, b: number): number => Q[a]?.[b] ?? 0
  const p = new Array<number>(k).fill(1 / k)
  const Qp = new Array<number>(k).fill(0)
  const eps = 0.005 / k
  let pQp = 0
  for (let iter = 0; iter < Math.max(100, k); iter++) {
    pQp = 0
    for (let t = 0; t < k; t++) {
      Qp[t] = 0
      for (let j = 0; j < k; j++) Qp[t] = (Qp[t] ?? 0) + q(t, j) * (p[j] ?? 0)
      pQp += (p[t] ?? 0) * (Qp[t] ?? 0)
    }
    let maxError = 0
    for (let t = 0; t < k; t++) maxError = Math.max(maxError, Math.abs((Qp[t] ?? 0) - pQp))
    if (maxError < eps) break
    for (let t = 0; t < k; t++) {
      const diff = (-(Qp[t] ?? 0) + pQp) / q(t, t)
      p[t] = (p[t] ?? 0) + diff
      pQp = (pQp + diff * (diff * q(t, t) + 2 * (Qp[t] ?? 0))) / ((1 + diff) * (1 + diff))
      for (let j = 0; j < k; j++) {
        Qp[j] = ((Qp[j] ?? 0) + diff * q(t, j)) / (1 + diff)
        p[j] = (p[j] ?? 0) / (1 + diff)
      }
    }
  }
  return p
}
