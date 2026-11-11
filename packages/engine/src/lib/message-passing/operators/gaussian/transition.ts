/**
 * The Gaussian transition factor between two Gaussian vectors, `x ~ N(F w + c, Q)` for a constant matrix `F`, offset `c` and
 * covariance `Q`: the dynamics of a linear dynamical system (a position and a velocity that moves by its velocity each step) or any
 * linear Gaussian step on a vector. It is the dense counterpart of the scalar `x_t = phi x_(t-1) + noise` chain: expectation
 * propagation on a chain of them with linear Gaussian readings is the Kalman smoother (exact), and each message is a closed form in
 * the natural parameters.
 */

import { dgemm, dgemv } from '../../../numerics/linalg/index.ts'
import { arg } from '../../core/_messages.ts'
import { etaOf, vectorParam } from '../../core/args.ts'
import { ImproperMessageError } from '../../core/errors.ts'
import { gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'

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

/** The mean and covariance of a Gaussian-vector belief; a belief with no density is an improper message. */
function momentsOfBelief(a: Arg, d: number): ReturnType<typeof gaussianMvMoments> {
  try {
    return gaussianMvMoments(etaOf(a), d)
  } catch {
    throw new ImproperMessageError('gaussian-transition: a belief has no density')
  }
}

/**
 * x ~ N(F w + c, Q) for Gaussian vectors x (`rows` entries) and w (`cols` entries), the matrix `F` row-major (`matrix`), the optional
 * offset `c` (`offset`) and the covariance `Q` (`covariance`, row-major, positive definite). The message to `x` is the Gaussian
 * pushforward of the cavity of `w` with the noise added (`N(F m + c, Q + F S F')`); the message to `w` is exact in natural
 * parameters whatever the cavity of `x` (a flat or a partly informed one included): `(F' (h - J c'), F' J F)` with the information
 * of `x` passed through the noise by `J' = J - J (J + Q^-1)^-1 J`. EP on a chain of these with linear Gaussian readings is exact. Under
 * variational message passing the log density is quadratic in both vectors, so the messages are the conjugate ones from the other's mean
 * (the mean field of a dynamical system: the beliefs of the states are independent, so their spread is under-estimated).
 *
 * @concept factor-operators
 * @index structure Gaussian vector step x ~ N(F w + c, Q), the dynamics of a linear system.
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
  params: { matrix: param.vector(), covariance: param.vector(), offset: param.vector({ optional: true }) },
  support: { ep: 'exact', vmp: 'exact' },
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
  // The mean field of the stochastic step: log N(x; F w + c, Q) is quadratic in both vectors, so the messages are the expected
  // log under the other's belief, exact and with no cross term (the beliefs are independent): to x the information form of
  // `N(F E w + c, Q)`, to w that of `F' Q^-1 (E x - c)` with precision `F' Q^-1 F`.
  vmp: (c): OperatorResult => {
    const { F, offset, Q, rows, cols } = parameters(c)
    const factorQ = spd(Q, rows)
    if (factorQ === undefined)
      throw new ImproperMessageError('gaussian-transition: the covariance is not positive definite')
    const QInv = factorQ.inverse()
    if (c.to === 0) {
      const mean = momentsOfBelief(arg(c, 1), cols).mean
      const image = dgemv(F, rows, cols, mean)
      const mu = Float64Array.from(image, (v, i) => v + (offset[i] ?? 0))
      return { message: [...dgemv(QInv, rows, rows, mu), ...QInv] }
    }
    const mean = momentsOfBelief(arg(c, 0), rows).mean
    const gap = Float64Array.from(mean, (v, i) => v - (offset[i] ?? 0))
    const h = dgemv(F, rows, cols, dgemv(QInv, rows, rows, gap), true)
    const QF = dgemm(QInv, F, rows, rows, cols)
    const J = dgemm(F, QF, cols, rows, cols, { transposeA: true })
    return { message: [...h, ...J] }
  },
  expectedLog: (c) => {
    const { F, offset, Q, rows, cols } = parameters(c)
    const factorQ = spd(Q, rows)
    if (factorQ === undefined) return Number.NaN
    const QInv = factorQ.inverse()
    const x = momentsOfBelief(arg(c, 0), rows)
    const w = momentsOfBelief(arg(c, 1), cols)
    const image = dgemv(F, rows, cols, w.mean)
    const gap = Float64Array.from(x.mean, (v, i) => v - (image[i] ?? 0) - (offset[i] ?? 0))
    const FS = dgemm(F, w.covariance, rows, cols, cols)
    const FSF = dgemm(FS, F, rows, cols, rows, { transposeB: true })
    // E (x - F w - c)' Q^-1 (x - F w - c) = gap' Q^-1 gap + tr(Q^-1 (S_x + F S_w F')).
    const solved = dgemv(QInv, rows, rows, gap)
    let quad = 0
    let trace = 0
    for (let i = 0; i < rows; i++) {
      quad += (gap[i] ?? 0) * (solved[i] ?? 0)
      for (let j = 0; j < rows; j++)
        trace += (QInv[i * rows + j] ?? 0) * ((x.covariance[j * rows + i] ?? 0) + (FSF[j * rows + i] ?? 0))
    }
    return -0.5 * (quad + trace + rows * Math.log(2 * Math.PI) + factorQ.logDet)
  },
  // The evidence of the factor under the cavities as unnormalised densities exp(h x - x J x / 2): the Gaussian integral of the joint of
  // (x, w), in information form so that a cavity that is only partly informative (the readings see the position, not the velocity)
  // still has one. Not a number when even the joint has no density.
  logEvidence: (c) => {
    const { F, offset, Q, rows, cols } = parameters(c)
    const ex = arg(c, 0).eta
    const ew = arg(c, 1).eta
    if (ex === undefined || ew === undefined) return Number.NaN
    const factorQ = spd(Q, rows)
    if (factorQ === undefined) return Number.NaN
    const QInv = factorQ.inverse()
    const n = rows + cols
    const J = new Float64Array(n * n)
    const h = new Float64Array(n)
    const QInvF = dgemm(QInv, F, rows, rows, cols)
    const FQF = dgemm(F, QInvF, cols, rows, cols, { transposeA: true })
    const QInvC = dgemv(QInv, rows, rows, offset)
    const FQc = dgemv(F, rows, cols, QInvC, true)
    for (let i = 0; i < rows; i++) {
      h[i] = (ex[i] ?? 0) + (QInvC[i] ?? 0)
      for (let j = 0; j < rows; j++) J[i * n + j] = (ex[rows + i * rows + j] ?? 0) + (QInv[i * rows + j] ?? 0)
      for (let j = 0; j < cols; j++) {
        J[i * n + rows + j] = -(QInvF[i * cols + j] ?? 0)
        J[(rows + j) * n + i] = -(QInvF[i * cols + j] ?? 0)
      }
    }
    for (let i = 0; i < cols; i++) {
      h[rows + i] = (ew[i] ?? 0) - (FQc[i] ?? 0)
      for (let j = 0; j < cols; j++)
        J[(rows + i) * n + rows + j] = (ew[cols + i * cols + j] ?? 0) + (FQF[i * cols + j] ?? 0)
    }
    const joint = spd(J, n)
    if (joint === undefined) return Number.NaN
    const solved = joint.solve(h)
    let hJh = 0
    for (let i = 0; i < n; i++) hJh += (h[i] ?? 0) * (solved[i] ?? 0)
    let cQc = 0
    for (let i = 0; i < rows; i++) cQc += (offset[i] ?? 0) * (QInvC[i] ?? 0)
    return 0.5 * hJh - 0.5 * joint.logDet + 0.5 * cols * Math.log(2 * Math.PI) - 0.5 * factorQ.logDet - 0.5 * cQc
  },
}
