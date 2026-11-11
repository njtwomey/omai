/**
 * The Gaussian transition factor between two Gaussian vectors, `x ~ N(F w + c, Q)` for a constant matrix `F`, offset `c` and
 * covariance `Q`: the dynamics of a linear dynamical system (a position and a velocity that moves by its velocity each step) or any
 * linear Gaussian step on a vector. It is the dense counterpart of the scalar `x_t = phi x_(t-1) + noise` chain: expectation
 * propagation on a chain of them with linear Gaussian readings is the Kalman smoother (exact), and each message is a closed form in
 * the natural parameters.
 */

import { dgemm, dgemv } from '../../numerics/linalg/index.ts'
import { spd } from '../_dense.ts'
import { gaussianMvEta, gaussianMvMoments } from '../families-vector.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { allProper, etaOf, ImproperMessageError, massOf, vectorParam } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

/** The transition's parameters: the matrix (rows = the size of x), the offset (zero when absent) and the covariance. */
function parameters(c: OperatorContext): {
  readonly F: Float64Array
  readonly offset: Float64Array
  readonly Q: Float64Array
  readonly rows: number
  readonly cols: number
} {
  const rows = arg(c, 0).family.dim
  const cols = arg(c, 1).family.dim
  const F = Float64Array.from(vectorParam(c.params, 'matrix'))
  const offset =
    c.params['offset'] === undefined ? new Float64Array(rows) : Float64Array.from(vectorParam(c.params, 'offset'))
  return { F, offset, Q: Float64Array.from(vectorParam(c.params, 'covariance')), rows, cols }
}

/**
 * x ~ N(F w + c, Q) for Gaussian vectors x (`rows` entries) and w (`cols` entries), the matrix `F` row-major (`matrix`), the optional
 * offset `c` (`offset`) and the covariance `Q` (`covariance`, row-major, positive definite). The message to `x` is the Gaussian
 * pushforward of the cavity of `w` with the noise added (`N(F m + c, Q + F S F')`); the message to `w` is exact in natural
 * parameters whatever the cavity of `x` (a flat or a partly informed one included): `(F' (h - J c'), F' J F)` with the information
 * of `x` passed through the noise by `J' = J - J (J + Q^-1)^-1 J`. EP on a chain of these with linear Gaussian readings is exact.
 *
 * @concept factor-operators
 * @index structure A Gaussian vector step x ~ N(F w + c, Q): the dynamics of a linear dynamical system.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, gaussianTransition, messageArg } from 'omai-engine/message-passing'
 * const g = gaussianMv(2)
 * const args = [messageArg(g, g.flat), messageArg(g, gaussianMvEta([1, 0], [1, 0, 0, 1]))]
 * const params = { matrix: [1, 1, 0, 1], covariance: [0.1, 0, 0, 0.1] }
 * gaussianTransition.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message.length // => 6
 * ```
 */
export const gaussianTransition: FactorOperator = {
  name: 'gaussian-transition',
  description: 'x ~ N(F w + c, Q) for Gaussian vectors x and w',
  ports: [
    { name: 'x', accepts: ['gaussian-mv'] },
    { name: 'w', accepts: ['gaussian-mv'] },
  ],
  support: { ep: 'exact' },
  check: (params, families) => {
    const rows = families[0]?.dim ?? 0
    const cols = families[1]?.dim ?? 0
    if (vectorParam(params, 'matrix').length !== rows * cols)
      return 'gaussian-transition: matrix must have (size of x) x (size of w) entries'
    if (vectorParam(params, 'covariance').length !== rows * rows)
      return 'gaussian-transition: covariance must be (size of x) squared'
    if (params['offset'] !== undefined && vectorParam(params, 'offset').length !== rows)
      return 'gaussian-transition: offset must have the size of x'
    return undefined
  },
  ep: (c): OperatorResult => {
    const { F, offset, Q, rows, cols } = parameters(c)
    if (c.to === 0) {
      let moments: ReturnType<typeof gaussianMvMoments>
      try {
        moments = gaussianMvMoments(etaOf(arg(c, 1)), cols)
      } catch {
        throw new ImproperMessageError('gaussian-transition: the cavity of w has no density')
      }
      const mean = dgemv(F, rows, cols, moments.mean)
      const FS = dgemm(F, moments.covariance, rows, cols, cols)
      const FSF = dgemm(FS, F, rows, cols, rows, { transposeB: true })
      const cov = Array.from(FSF, (v, i) => v + (Q[i] ?? 0))
      try {
        return {
          message: gaussianMvEta(
            Array.from(mean, (v, i) => v + (offset[i] ?? 0)),
            cov,
          ),
        }
      } catch {
        throw new ImproperMessageError('gaussian-transition: the step is degenerate')
      }
    }
    // To w: the information of x's cavity passed through the noise, then through F.
    const eta = etaOf(arg(c, 0))
    const hx = eta.slice(0, rows)
    const Jx = Float64Array.from(eta.slice(rows))
    const factorQ = spd(Q, rows)
    if (factorQ === undefined)
      throw new ImproperMessageError('gaussian-transition: the covariance is not positive definite')
    const QInv = factorQ.inverse()
    const sum = Float64Array.from(Jx, (v, i) => v + (QInv[i] ?? 0))
    const factorM = spd(sum, rows)
    if (factorM === undefined) throw new ImproperMessageError('gaussian-transition: the information of x is not usable')
    const M = factorM.inverse()
    const JM = dgemm(Jx, M, rows, rows, rows)
    const JMJ = dgemm(JM, Jx, rows, rows, rows)
    const Jm = Float64Array.from(Jx, (v, i) => v - (JMJ[i] ?? 0))
    const JMh = dgemv(JM, rows, rows, hx)
    const hm = Float64Array.from(hx, (v, i) => v - (JMh[i] ?? 0))
    const Jc = dgemv(Jm, rows, rows, offset)
    const rhs = Float64Array.from(hm, (v, i) => v - (Jc[i] ?? 0))
    const h = dgemv(F, rows, cols, rhs, true)
    const JF = dgemm(Jm, F, rows, rows, cols)
    const J = dgemm(F, JF, cols, rows, cols, { transposeA: true })
    return { message: [...h, ...J] }
  },
  // The evidence of the factor under the cavities: N(m_x; F m_w + c, Q + S_x + F S_w F') times the masses of the cavities.
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const { F, offset, Q, rows, cols } = parameters(c)
    const x = gaussianMvMoments(etaOf(arg(c, 0)), rows)
    const w = gaussianMvMoments(etaOf(arg(c, 1)), cols)
    const FS = dgemm(F, w.covariance, rows, cols, cols)
    const FSF = dgemm(FS, F, rows, cols, rows, { transposeB: true })
    const spread = Array.from(FSF, (v, i) => v + (Q[i] ?? 0) + (x.covariance[i] ?? 0))
    const factor = spd(spread, rows)
    if (factor === undefined) return Number.NaN
    const image = dgemv(F, rows, cols, w.mean)
    const gap = Array.from(x.mean, (v, i) => v - (image[i] ?? 0) - (offset[i] ?? 0))
    const solved = factor.solve(gap)
    let quad = 0
    for (let i = 0; i < rows; i++) quad += (gap[i] ?? 0) * (solved[i] ?? 0)
    return massOf(arg(c, 0)) + massOf(arg(c, 1)) - 0.5 * (rows * Math.log(2 * Math.PI) + factor.logDet + quad)
  },
}
