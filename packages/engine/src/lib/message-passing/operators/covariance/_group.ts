// What the factors of a group of Gaussian vectors with a shared Gaussian mean (`mvn-wishart-group`, `mvn-sd-corr-group`) have in common: the
// closed-form integral over the vectors and the mean at one precision matrix, with the conditionals it gives, and the key of a tilt the
// ports of one sweep share.

import type { OperatorContext } from '../../core/types.ts'
import { choleskyInPlace, choleskyLogDet, choleskySolveInPlace } from '../../numerics/_bartlett.ts'
import type { GaussianSide } from './_sides.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/** What the closed-form integral over the vectors and the mean reads at one matrix: the arguments as Gaussian sides. */
export interface Sides {
  readonly d: number
  readonly groups: readonly GaussianSide[]
  readonly mean: GaussianSide
}

/**
 * The integral of the factor against the cavities of the vectors and the mean, as a function of the precision matrix at the nodes of a
 * quadrature: `log I(L)` and, when asked, the mean and covariance of each Gaussian under the tilted conditional given the matrix. With
 * cavities `exp(h_j x - x' W_j x / 2)` for each vector, `exp(h_0 m - m' W_0 m / 2)` for the mean and `T_j = (W_j + L)^-1`:
 * integrating `x_j` leaves `c_j + (L T_j h_j) m - m' (L - L T_j L) m / 2` with `c_j = (log |L| - log |W_j + L| + h_j' T_j h_j) / 2`, a Gaussian in
 * the mean of precision `P = W_0 + sum (L - L T_j L)` and linear term `g = h_0 + sum L T_j h_j`.
 */
export function nodeEvaluator(
  s: Sides,
): (lambda: Float64Array, offset: number, logDet: number, conditionals: Float64Array | undefined) => number {
  const { d, groups, mean } = s
  const dd = d * d
  const count = groups.length
  // Exempt from the tensor rule, on purpose: the scratch buffers of the per-node closure below, which the tilt loop calls for every
  // node of the quadrature; a tensor op per node would allocate on each call.
  const t = new Float64Array(count * dd)
  const a = new Float64Array(count * dd)
  const u = new Float64Array(count * d)
  const m = new Float64Array(dd)
  const p = new Float64Array(dd)
  const g = new Float64Array(d)
  const q = new Float64Array(d)
  const col = new Float64Array(d)
  const la = new Float64Array(dd)
  return (lambda, offset, logDet, conditionals) => {
    p.set(mean.precision)
    g.set(mean.h)
    let logI = 0
    for (let j = 0; j < count; j++) {
      const x = groups[j] as GaussianSide
      if (x.point) {
        // N(x; mean, L^-1) as a function of the mean
        let quad = 0
        for (let i = 0; i < d; i++) {
          let li = 0
          for (let k = 0; k < d; k++) {
            const l = lambda[offset + i * d + k] as number
            li += l * (x.mean[k] as number)
            p[i * d + k] = (p[i * d + k] as number) + l
          }
          g[i] = (g[i] as number) + li
          quad += li * (x.mean[i] as number)
        }
        logI += 0.5 * logDet - 0.5 * d * LOG_2PI - 0.5 * quad
        continue
      }
      for (let e = 0; e < dd; e++) m[e] = (x.precision[e] as number) + (lambda[offset + e] as number)
      if (!choleskyInPlace(m, d)) return Number.NEGATIVE_INFINITY
      const logDetM = choleskyLogDet(m, d)
      const at = j * dd
      for (let k = 0; k < d; k++) {
        col.fill(0)
        col[k] = 1
        choleskySolveInPlace(m, d, col)
        for (let i = 0; i < d; i++) t[at + i * d + k] = col[i] as number
      }
      let hu = 0
      for (let i = 0; i < d; i++) {
        let s2 = 0
        for (let k = 0; k < d; k++) s2 += (t[at + i * d + k] as number) * (x.h[k] as number)
        u[j * d + i] = s2
        hu += s2 * (x.h[i] as number)
      }
      logI += 0.5 * (logDet - logDetM + hu)
      // A = T L, and L - L A (= L T L symmetric) added to the precision of the mean, L u to its linear term
      for (let i = 0; i < d; i++)
        for (let k = 0; k < d; k++) {
          let s2 = 0
          for (let l = 0; l < d; l++) s2 += (t[at + i * d + l] as number) * (lambda[offset + l * d + k] as number)
          a[at + i * d + k] = s2
        }
      for (let i = 0; i < d; i++) {
        for (let k = 0; k < d; k++) {
          let s2 = 0
          for (let l = 0; l < d; l++) s2 += (lambda[offset + i * d + l] as number) * (a[at + l * d + k] as number)
          p[i * d + k] = (p[i * d + k] as number) + (lambda[offset + i * d + k] as number) - s2
        }
        let li = 0
        for (let k = 0; k < d; k++) li += (lambda[offset + i * d + k] as number) * (u[j * d + k] as number)
        g[i] = (g[i] as number) + li
      }
    }
    la.set(p)
    if (!choleskyInPlace(la, d)) return Number.NEGATIVE_INFINITY
    q.set(g)
    choleskySolveInPlace(la, d, q)
    let gq = 0
    for (let i = 0; i < d; i++) gq += (g[i] as number) * (q[i] as number)
    logI += 0.5 * gq - 0.5 * choleskyLogDet(la, d) + 0.5 * d * LOG_2PI
    if (conditionals !== undefined) {
      // the conditionals given the matrix: the mean N(q, P^-1) in the last slot, vector j N(u_j + A_j q, T_j + A_j P^-1 A_j')
      const mu = count * (d + dd)
      for (let k = 0; k < d; k++) {
        col.fill(0)
        col[k] = 1
        choleskySolveInPlace(la, d, col)
        for (let i = 0; i < d; i++) conditionals[mu + d + i * d + k] = col[i] as number
      }
      for (let i = 0; i < d; i++) conditionals[mu + i] = q[i] as number
      for (let j = 0; j < count; j++) {
        if ((groups[j] as GaussianSide).point) continue
        const at = j * dd
        const out = j * (d + dd)
        for (let i = 0; i < d; i++) {
          let s2 = u[j * d + i] as number
          for (let k = 0; k < d; k++) s2 += (a[at + i * d + k] as number) * (q[k] as number)
          conditionals[out + i] = s2
        }
        for (let i = 0; i < d; i++)
          for (let k = 0; k < d; k++) {
            let s2 = t[at + i * d + k] as number
            for (let l = 0; l < d; l++)
              for (let r = 0; r < d; r++)
                s2 +=
                  (a[at + i * d + l] as number) *
                  (conditionals[mu + d + l * d + r] as number) *
                  (a[at + k * d + r] as number)
            conditionals[out + d + i * d + k] = s2
          }
      }
    }
    return logI
  }
}

/**
 * The key of a tilt: the arguments to ten digits of the largest entry of each. The cavity of a port is its belief divided by its own message, so
 * when the message of another port of the factor has just been sent the cavities are the same numbers to rounding, not to the last bit (and an
 * entry that is zero comes back as 1e-17); a key that said every bit would miss for each of the group's ports and integrate once per port
 * instead of once per sweep.
 */
export const tiltKey = (prefix: string, c: OperatorContext): string =>
  `${prefix}:${c.args
    .map((a) => {
      const v = Array.from((a.value ?? a.eta ?? []) as ArrayLike<number> | number[])
      const unit = Math.max(...v.map((x) => Math.abs(x)), 1e-300) * 1e-10
      return v.map((x) => Math.round(x / unit)).join(',')
    })
    .join(';')}`
