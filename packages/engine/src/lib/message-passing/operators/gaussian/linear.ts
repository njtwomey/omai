/**
 * The linear-Gaussian operators: an affine combination of Gaussians (and its special cases, the sum, the difference
 * and the scaling by a constant), a Gaussian projection of a Gaussian vector (a dot product and a matrix product with
 * constants), a vector assembled from scalar variables, a regression row, and a joint Gaussian potential over scalar variables. All are closed forms under
 * expectation propagation, so EP is exact on a tree of them. The affine ones (`linear`, `plus`, `difference`,
 * `times-constant`, `sum`, `dot-constant`) are deterministic: they define their output (`FactorOperator.defines`), and
 * under variational message passing the output is the pushforward of the inputs' beliefs while the messages to the
 * inputs are the expected log of the output's cavity, which is the mean field of the model with the output substituted.
 */

import { OmaiError } from '../../../base/index.ts'
import { ddot, dgemv } from '../../../numerics/linalg/index.ts'
import { mul, outer, toFlat } from '../../../tensor/index.ts'
import { arg } from '../../core/_messages.ts'
import {
  allProper,
  etaOf,
  gaussianMessage,
  gaussianOf,
  logGaussianIntegral,
  logNormal,
  massOf,
  numberParam,
  rawSum,
  scalarOf,
  vectorParam,
} from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { ImproperMessageError } from '../../core/errors.ts'
import { gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import { type ParamSpec, param } from '../../core/params.ts'
import type { FactorOperator, Family, OperatorContext, OperatorResult, Port } from '../../core/types.ts'
import { constantArray, rangeArray, unitArray, zeroArray } from '../../numerics/_arrays.ts'
import { spd } from '../../numerics/_dense.ts'

const gaussianPort = (name: string): Port => ({ name, accepts: ['gaussian'], constant: true })

/** The EP message of y = d + sum c_i a_i, to the output (port 0) or to an input. */
function affineMessage(c: OperatorContext, coeffs: readonly number[], offset: number): OperatorResult {
  const inputs = coeffs.map((_, i) => gaussianOf(arg(c, i + 1)))
  if (c.to === 0) {
    let mean = offset
    let variance = 0
    for (const [i, k] of coeffs.entries()) {
      mean += k * (inputs[i]?.mean ?? 0)
      variance += k * k * (inputs[i]?.variance ?? 0)
    }
    return { message: gaussianMessage(mean, variance) }
  }
  const j = c.to - 1
  const k = coeffs[j] ?? 0
  if (k === 0) return { message: [0, 0] }
  const y = gaussianOf(arg(c, 0))
  let mean = y.mean - offset
  let variance = y.variance
  for (const [i, ki] of coeffs.entries()) {
    if (i === j) continue
    mean -= ki * (inputs[i]?.mean ?? 0)
    variance += ki * ki * (inputs[i]?.variance ?? 0)
  }
  return { message: gaussianMessage(mean / k, variance / (k * k)) }
}

/**
 * log of the mass of y = d + sum c_i a_i under the incoming messages of every port, in natural parameters: the integral of the
 * messages over the inputs with the output substituted (or, for an observed output, with one input eliminated by the constraint and
 * the Jacobian `1 / |c_j|`), a Gaussian integral that is finite whenever the joint precision is positive definite, though no cavity
 * need be proper by itself (the site of a scale mixture or a mixture can have a negative precision, and a flat one says nothing and
 * integrates as 1). NaN when it is not.
 */
function affineEvidenceNatural(c: OperatorContext, coeffs: readonly number[], offset: number): number {
  let shift = offset
  const free: number[] = []
  for (const [i, k] of coeffs.entries()) {
    const a = arg(c, i + 1)
    if (a.value !== undefined) shift += k * scalarOf(a)
    else free.push(i)
  }
  const y = arg(c, 0)
  const h = free.map((i) => etaOf(arg(c, i + 1))[0] ?? 0)
  const j = free.map((i) => etaOf(arg(c, i + 1))[1] ?? 0)
  const k = free.map((i) => coeffs[i] ?? 0)
  const m = free.length
  const precision = (cross: readonly number[], scale: number): number[] => {
    const p = zeroArray(m * m)
    for (let r = 0; r < m; r++)
      for (let q = 0; q < m; q++) p[r * m + q] = (r === q ? (j[r] ?? 0) : 0) + scale * (cross[r] ?? 0) * (cross[q] ?? 0)
    return p
  }
  if (y.value === undefined) {
    const [hy, jy] = [etaOf(y)[0] ?? 0, etaOf(y)[1] ?? 0]
    const constant = hy * shift - 0.5 * jy * shift * shift
    if (m === 0) return constant
    return (
      constant +
      logGaussianIntegral(
        h.map((v, r) => v + (k[r] ?? 0) * (hy - jy * shift)),
        precision(k, jy),
        m,
      )
    )
  }
  // An observed output: the hyperplane `sum c_i a_i = y - shift`, parameterised by the others once the input of the largest
  // coefficient is solved for.
  let pivot = -1
  for (let r = 0; r < m; r++) if (pivot < 0 || Math.abs(k[r] ?? 0) > Math.abs(k[pivot] ?? 0)) pivot = r
  if (pivot < 0 || (k[pivot] ?? 0) === 0) return Number.NaN
  const kp = k[pivot] as number
  const u0 = (scalarOf(y) - shift) / kp
  const rest = rangeArray(m).filter((r) => r !== pivot)
  // a_pivot = u0 + sum_r B_r a_r, B_r = -c_r / c_pivot; the exponent in the remaining inputs a_r.
  const b = rest.map((r) => -(k[r] ?? 0) / kp)
  const hp = h[pivot] ?? 0
  const jp = j[pivot] ?? 0
  const constant = hp * u0 - 0.5 * jp * u0 * u0 - Math.log(Math.abs(kp))
  const n = rest.length
  if (n === 0) return constant
  const hh = rest.map((r, t) => (h[r] ?? 0) + (b[t] ?? 0) * (hp - jp * u0))
  const p = zeroArray(n * n)
  for (let r = 0; r < n; r++)
    for (let q = 0; q < n; q++)
      p[r * n + q] = (r === q ? (j[rest[r] as number] ?? 0) : 0) + jp * (b[r] ?? 0) * (b[q] ?? 0)
  return constant + logGaussianIntegral(hh, p, n)
}

/** log of the mass of y = d + sum c_i a_i under the incoming messages of every port. */
function affineEvidence(c: OperatorContext, coeffs: readonly number[], offset: number): number {
  if (!allProper(c.args)) return affineEvidenceNatural(c, coeffs, offset)
  const y = gaussianOf(arg(c, 0))
  let mean = offset
  let variance = 0
  for (const [i, k] of coeffs.entries()) {
    const a = gaussianOf(arg(c, i + 1))
    mean += k * a.mean
    variance += k * k * a.variance
  }
  let log = logNormal(y.mean, mean, y.variance + variance)
  for (const a of c.args) log += massOf(a)
  return log
}

/**
 * The variational message of y = d + sum c_i a_i, which defines y: to the output, the pushforward of the inputs' beliefs
 * (the output's belief is that alone); to input j, what the mean field makes of the cavity `N(h, J)` of the output (the
 * messages its other factors send it) once the other inputs are averaged out: `exp E[log cavity(d + sum c_i a_i)]` as a
 * function of a_j, which is `[c_j (h - J (d + sum_{i != j} c_i E[a_i])), J c_j^2]`. The variances of the other inputs only
 * add a constant, so they do not appear.
 */
function affineVariational(c: OperatorContext, coeffs: readonly number[], offset: number): OperatorResult {
  if (c.to === 0) return affineMessage(c, coeffs, offset)
  const j = c.to - 1
  const k = coeffs[j] ?? 0
  if (k === 0) return { message: [0, 0] }
  const out = arg(c, 0)
  if (out.eta === undefined)
    throw new OmaiError('not-implemented', 'a deterministic factor with an observed output has no variational message')
  let rest = offset
  for (const [i, ki] of coeffs.entries()) if (i !== j) rest += ki * gaussianOf(arg(c, i + 1)).mean
  const h = out.eta[0] ?? 0
  const lambda = out.eta[1] ?? 0
  return { message: [k * (h - lambda * rest), lambda * k * k] }
}

function affineOperator(
  name: string,
  description: string,
  ports: readonly Port[] | ((params: OperatorContext['params']) => readonly Port[]),
  declared: ParamSpec,
  form: (params: OperatorContext['params']) => { readonly coeffs: readonly number[]; readonly offset: number },
): FactorOperator {
  return defineOperator({
    name,
    description,
    ports,
    params: declared,
    rules: {
      ep: messageRule((c) => {
        const f = form(c.params)
        return affineMessage(c, f.coeffs, f.offset)
      }),
      vmp: messageRule((c) => {
        const f = form(c.params)
        return affineVariational(c, f.coeffs, f.offset)
      }),
    },
    evidence: {
      ep: (c) => {
        const f = form(c.params)
        return affineEvidence(c, f.coeffs, f.offset)
      },
      vmp: () => 0,
    },
    traits: { defines: 0 },
  })
}

/**
 * y = d + sum_i c_i a_i for Gaussian variables: the affine combination, with its coefficients `coeffs` and offset
 * `offset` as parameters. The one operator behind the sum, difference and scaling; under EP the messages are the exact
 * pushforward (to the output) and its inverse (to an input), and every message is a closed form.
 *
 * @concept factor-operators
 * @index structure An affine combination of Gaussian variables y = d + sum c_i a_i.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, linear, messageArg } from 'omai-engine/message-passing'
 * const a = messageArg(gaussian, gaussianEta(1, 1))
 * const b = messageArg(gaussian, gaussianEta(2, 0.25))
 * const args = [messageArg(gaussian, gaussian.flat), a, b]
 * linear.ep?.({ mode: 'ep', to: 0, args, params: { coeffs: [2, 1], offset: 1 }, power: 1 }).message.map((v) => Math.round(v * 1e6) / 1e6) // => [1.176471, 0.235294]
 * ```
 */
export const linear: FactorOperator = affineOperator(
  'linear',
  'y = offset + sum of coefficient times argument, for Gaussians',
  (params) => [gaussianPort('y'), ...vectorParam(params, 'coeffs').map((_, i) => gaussianPort(`a${i}`))],
  { coeffs: param.vector(), offset: param.number({ optional: true }) },
  (params) => ({ coeffs: vectorParam(params, 'coeffs'), offset: numberParam(params, 'offset', 0) }),
)

/**
 * y = a + b for Gaussians.
 *
 * @concept factor-operators
 * @index structure The sum of two Gaussian variables.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, plus } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(1, 1)), messageArg(gaussian, gaussianEta(2, 1))]
 * plus.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message // => [1.5, 0.5]
 * ```
 */
export const plus: FactorOperator = affineOperator(
  'plus',
  'y = a + b for Gaussians',
  [gaussianPort('y'), gaussianPort('a'), gaussianPort('b')],
  {},
  () => ({ coeffs: [1, 1], offset: 0 }),
)

/**
 * y = a - b for Gaussians.
 *
 * @concept factor-operators
 * @index structure The difference of two Gaussian variables.
 * @example
 * ```ts
 * import { difference, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(5, 1)), messageArg(gaussian, gaussianEta(2, 1))]
 * difference.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message // => [1.5, 0.5]
 * ```
 */
export const difference: FactorOperator = affineOperator(
  'difference',
  'y = a - b for Gaussians',
  [gaussianPort('y'), gaussianPort('a'), gaussianPort('b')],
  {},
  () => ({ coeffs: [1, -1], offset: 0 }),
)

/**
 * y = factor a for a Gaussian a and a constant factor.
 *
 * @concept factor-operators
 * @index structure A Gaussian variable scaled by a constant.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, timesConstant } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(2, 1))]
 * timesConstant.ep?.({ mode: 'ep', to: 0, args, params: { factor: 3 }, power: 1 }).message // => [0.6666666666666666, 0.1111111111111111]
 * ```
 */
export const timesConstant: FactorOperator = affineOperator(
  'times-constant',
  'y = factor a for a constant factor',
  [gaussianPort('y'), gaussianPort('a')],
  { factor: param.number() },
  (params) => ({ coeffs: [numberParam(params, 'factor')], offset: 0 }),
)

/**
 * y = a_1 + ... + a_n for Gaussians: the sum of an array (`n` in the parameters).
 *
 * @concept factor-operators
 * @index structure The sum of n Gaussian variables.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, sum } from 'omai-engine/message-passing'
 * const g = (m: number) => messageArg(gaussian, gaussianEta(m, 1))
 * sum.ep?.({ mode: 'ep', to: 0, args: [messageArg(gaussian, gaussian.flat), g(1), g(2), g(3)], params: { n: 3 }, power: 1 }).message // => [2, 0.3333333333333333]
 * ```
 */
export const sum: FactorOperator = affineOperator(
  'sum',
  'y = a_1 + ... + a_n for Gaussians',
  // banned-ok: non-numeric: the ports of the sum
  (params) => [gaussianPort('y'), ...Array.from({ length: numberParam(params, 'n') }, (_, i) => gaussianPort(`a${i}`))],
  { n: param.integer({ min: 1 }) },
  (params) => ({ coeffs: constantArray(numberParam(params, 'n'), 1), offset: 0 }),
)

function dimensions(families: readonly Family[]): { readonly rows: number; readonly d: number } {
  return { rows: families[0]?.dim ?? 0, d: families[1]?.dim ?? 0 }
}

/**
 * s = a . w for a Gaussian vector w and a constant vector a (a projection onto one direction: the linear predictor of a
 * regression row, or one component of a Gaussian process vector when a is a basis vector). The message to w is a
 * rank-one natural-parameter message and exact; the message to s needs the moments of w.
 *
 * @concept factor-operators
 * @index structure A Gaussian vector projected on a constant direction s = a . w.
 * @example
 * ```ts
 * import { dotConstant, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(1, 2)), messageArg(gaussian, gaussian.flat)]
 * dotConstant.ep?.({ mode: 'ep', to: 1, args: [args[0] as never, { ...(args[1] as never) }], params: { a: [1, 3] }, power: 1 }).message.length // => 6
 * ```
 */
export const dotConstant: FactorOperator = defineOperator({
  name: 'dot-constant',
  description: 's = a . w for a constant vector a',
  ports: [
    { name: 's', accepts: ['gaussian'] },
    { name: 'w', accepts: ['gaussian-mv'] },
  ],
  params: { a: param.vector() },
  check: (params, families) =>
    vectorParam(params, 'a').length === families[1]?.dim ? undefined : 'dot-constant: a must have the length of w',
  rules: {
    ep: messageRule((c): OperatorResult => dotConstantMessage(c)),
    // The message to w is linear in the cavity of s and the pushforward needs only the belief of w, so the variational
    // rule is the same function as the EP one, with the cavity of the defined output and the belief of the input.
    vmp: messageRule((c): OperatorResult => dotConstantMessage(c)),
  },
  evidence: {
    ep: (c) => {
      const w = arg(c, 1)
      // A latent output: the integral over the vector of its cavity times the output's cavity at a . w, which is the log
      // partition of the vector's message plus the (exact) message to it. The output may be flat (nothing else reads it) and
      // need not be proper on its own, only the sum must be.
      if (arg(c, 0).value === undefined) {
        const total = rawSum(etaOf(w), dotConstantMessage({ ...c, to: 1 }).message)
        return w.family.isProper(total) ? w.family.logPartition(total) : Number.NaN
      }
      if (!allProper(c.args)) return Number.NaN
      const a = vectorParam(c.params, 'a')
      const d = a.length
      const { mean, covariance } = gaussianMvMoments(etaOf(w), d)
      const s = gaussianOf(arg(c, 0))
      let m = 0
      for (let i = 0; i < d; i++) m += (a[i] ?? 0) * (mean[i] ?? 0)
      const cov = dgemv(covariance, d, d, a)
      let v = 0
      for (let i = 0; i < d; i++) v += (a[i] ?? 0) * (cov[i] ?? 0)
      return massOf(arg(c, 0)) + massOf(arg(c, 1)) + logNormal(s.mean, m, s.variance + v)
    },
    vmp: () => 0,
  },
  traits: { defines: 0 },
})

/** The message of s = a . w: to w the rank-one message of the cavity of s, to s the pushforward of the moments of w. */
function dotConstantMessage(c: OperatorContext): OperatorResult {
  const a = vectorParam(c.params, 'a')
  const d = a.length
  if (c.to === 1) {
    const s = etaOf(arg(c, 0))
    const j = toFlat(outer(mul(a, s[1] ?? 0), a))
    return { message: [...a.map((v) => (s[0] ?? 0) * v), ...j] }
  }
  const w = etaOf(arg(c, 1))
  const factor = spd(w.slice(d), d)
  if (factor === undefined) throw new ImproperMessageError('dot-constant: the precision of w is not positive definite')
  const mean = factor.solve(w.slice(0, d))
  const z = factor.solve(a)
  return { message: gaussianMessage(ddot(a, mean), ddot(a, z)) }
}

/**
 * The message of `D = (y_0, ..., y_{n-1})`: to the vector the diagonal Gaussian of the scalars' messages (their natural parameters
 * on the diagonal: the cavities under EP, the beliefs under the mean field, whose pushforward is that), to a scalar under EP the
 * marginal at its coordinate of the vector's cavity and the other scalars' cavities, and under the mean field the expected log of the vector's cavity `N(h, J)` as a
 * function of that coordinate with the others at their means, `[h_j - sum_{i != j} J_ji E[y_i], J_jj]`.
 */
function stackMessage(c: OperatorContext, variational: boolean): OperatorResult {
  const d = arg(c, 0).family.dim
  if (c.to === 0) {
    const h = zeroArray(d)
    const j = zeroArray(d * d)
    for (let i = 0; i < d; i++) {
      const eta = etaOf(arg(c, i + 1))
      h[i] = eta[0] ?? 0
      j[i * d + i] = eta[1] ?? 0
    }
    return { message: [...h, ...j] }
  }
  const k = c.to - 1
  const w = etaOf(arg(c, 0))
  if (variational) {
    let h = w[k] ?? 0
    for (let i = 0; i < d; i++) if (i !== k) h -= (w[d + k * d + i] ?? 0) * gaussianOf(arg(c, i + 1)).mean
    return { message: [h, w[d + k * d + k] ?? 0] }
  }
  // The vector's cavity times the cavities of the other scalars, which the delta carries through to it, and its marginal at `k`.
  const h = w.slice(0, d)
  const j = w.slice(d)
  for (let i = 0; i < d; i++) {
    if (i === k) continue
    const eta = etaOf(arg(c, i + 1))
    h[i] = (h[i] ?? 0) + (eta[0] ?? 0)
    j[i * d + i] = (j[i * d + i] ?? 0) + (eta[1] ?? 0)
  }
  const factor = spd(j, d)
  if (factor === undefined)
    throw new ImproperMessageError('gaussian-mv-stack: the precision of the vector is not positive definite')
  const e = unitArray(d, k)
  return { message: gaussianMessage(factor.solve(h)[k] ?? 0, factor.solve(e)[k] ?? 0) }
}

/**
 * `D = (y_0, ..., y_{n-1})`: a Gaussian vector assembled from `n` scalar Gaussian variables (`n` in the parameters), the vector being
 * the variable the factor defines. It ties the scalar variables of a vector whose coordinates are each a factor of their own (the
 * non-centred coefficients with a scale each, `y_j = z_j f(x_j)`) to one Gaussian vector variable that the likelihood reads as a
 * vector, so its message to the coefficients is dense. Under EP the message to the vector is the diagonal Gaussian of the scalars'
 * cavities and the message to a scalar the marginal at its coordinate of the vector's cavity with the other scalars' cavities (the `n` unit
 * `dot-constant` ties in one factor). Under variational message passing the vector is defined by the scalars, its belief the diagonal Gaussian of their beliefs,
 * and the message to a scalar the expected log of the vector's cavity with the other coordinates at their means.
 *
 * @concept factor-operators
 * @index structure A Gaussian vector assembled from scalar Gaussian variables D = (y_0, ..., y_n-1).
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianMv, gaussianMvStack, messageArg } from 'omai-engine/message-passing'
 * const v = gaussianMv(2)
 * const args = [messageArg(v, v.flat), messageArg(gaussian, gaussianEta(1, 2)), messageArg(gaussian, gaussianEta(3, 4))]
 * gaussianMvStack.ep?.({ mode: 'ep', to: 0, args, params: { n: 2 }, power: 1 }).message // => [0.5, 0.75, 0.5, 0, 0, 0.25]
 * ```
 */
export const gaussianMvStack: FactorOperator = defineOperator({
  name: 'gaussian-mv-stack',
  description: 'D = (y_0, ..., y_n-1): a Gaussian vector assembled from scalar Gaussian variables',
  ports: (params) => [
    { name: 'd', accepts: ['gaussian-mv'] },
    // banned-ok: non-numeric: the ports of the stack
    ...Array.from({ length: numberParam(params, 'n') }, (_, i): Port => ({ name: `y${i}`, accepts: ['gaussian'] })),
  ],
  params: { n: param.integer({ min: 1 }) },
  check: (params, families) =>
    numberParam(params, 'n') === families[0]?.dim ? undefined : 'gaussian-mv-stack: n must be the length of the vector',
  rules: {
    ep: messageRule((c): OperatorResult => stackMessage(c, false)),
    vmp: messageRule((c): OperatorResult => stackMessage(c, true)),
  },
  evidence: {
    // As the unit ties: the integral over the vector of its cavity times the scalars' cavities at its coordinates, which is the log
    // partition of the vector's cavity plus the (exact) diagonal message to it.
    ep: (c) => {
      const d = arg(c, 0)
      const total = rawSum(etaOf(d), stackMessage({ ...c, to: 0 }, false).message)
      return d.family.isProper(total) ? d.family.logPartition(total) : Number.NaN
    },
    vmp: () => 0,
  },
  traits: { defines: 0 },
})

/**
 * y = A w for a Gaussian vector w and a constant matrix A (`matrix`, row-major, with as many rows as y has entries): a
 * linear map of a vector. The message to w is exact in natural parameters (`[A' h, A' J A]`); the message to y is the
 * Gaussian pushforward of the moments of w. Under variational message passing `y` is defined by `w` (as for `dot-constant`).
 *
 * @concept factor-operators
 * @index structure A Gaussian vector under a constant matrix y = A w.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, matmulConstant, messageArg } from 'omai-engine/message-passing'
 * const y = gaussianMv(2)
 * const w = gaussianMv(2)
 * const args = [messageArg(y, y.flat), messageArg(w, gaussianMvEta([1, 2], [1, 0, 0, 1]))]
 * matmulConstant.ep?.({ mode: 'ep', to: 0, args, params: { matrix: [1, 1, 0, 2] }, power: 1 }).message.length // => 6
 * ```
 */
export const matmulConstant: FactorOperator = defineOperator({
  name: 'matmul-constant',
  description: 'y = A w for a constant matrix A',
  ports: [
    { name: 'y', accepts: ['gaussian-mv'] },
    { name: 'w', accepts: ['gaussian-mv'] },
  ],
  params: { matrix: param.vector() },
  check: (params, families) => {
    const { rows, d } = dimensions(families)
    return vectorParam(params, 'matrix').length === rows * d
      ? undefined
      : 'matmul-constant: matrix must have rows x length(w) entries'
  },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const m = vectorParam(c.params, 'matrix')
      const rows = arg(c, 0).family.dim
      const d = arg(c, 1).family.dim
      if (c.to === 1) {
        const y = etaOf(arg(c, 0))
        const h = dgemv(m, rows, d, y.slice(0, rows), true)
        const j = zeroArray(d * d)
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) {
            let s = 0
            for (let r = 0; r < rows; r++)
              for (let q = 0; q < rows; q++)
                s += (m[r * d + a] ?? 0) * (y[rows + r * rows + q] ?? 0) * (m[q * d + b] ?? 0)
            j[a * d + b] = s
          }
        return { message: [...h, ...j] }
      }
      const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 1)), d)
      const my = dgemv(m, rows, d, mean)
      const cov = zeroArray(rows * rows)
      // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
      const ac = new Float64Array(rows * d)
      for (let r = 0; r < rows; r++) {
        const row = dgemv(covariance, d, d, m.slice(r * d, (r + 1) * d))
        ac.set(row, r * d)
      }
      for (let r = 0; r < rows; r++)
        for (let q = 0; q < rows; q++) {
          let s = 0
          for (let k = 0; k < d; k++) s += (m[r * d + k] ?? 0) * (ac[q * d + k] ?? 0)
          cov[r * rows + q] = s
        }
      try {
        return { message: gaussianMvEta(Array.from(my), cov) }
      } catch (e) {
        if (e instanceof OmaiError) throw new ImproperMessageError('matmul-constant: the image of w is degenerate')
        throw e
      }
    }),
    // The mean field of a deterministic factor: the same function as the EP one, with the cavity of the defined output and the belief
    // of the input (the message to w is linear in the cavity of y, the pushforward needs the moments of w alone).
    vmp: messageRule((c): OperatorResult => (matmulConstant.ep as NonNullable<FactorOperator['ep']>)(c)),
  },
  evidence: {
    // y = A w is a point mass: the evidence is the masses of the cavities times the density of the cavity of y at the image of w,
    // N(my; A mw, Vy + A Vw A'), a vector normal whose covariance must be positive definite.
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const m = vectorParam(c.params, 'matrix')
      const rows = arg(c, 0).family.dim
      const d = arg(c, 1).family.dim
      const y = gaussianMvMoments(etaOf(arg(c, 0)), rows)
      const w = gaussianMvMoments(etaOf(arg(c, 1)), d)
      const image = dgemv(m, rows, d, w.mean)
      const spread = zeroArray(rows * rows)
      for (let r = 0; r < rows; r++)
        for (let q = 0; q < rows; q++) {
          let s = y.covariance[r * rows + q] ?? 0
          for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++) s += (m[r * d + a] ?? 0) * (w.covariance[a * d + b] ?? 0) * (m[q * d + b] ?? 0)
          spread[r * rows + q] = s
        }
      const factor = spd(spread, rows)
      if (factor === undefined) return Number.NaN
      const gap = Array.from(y.mean, (v, i) => v - (image[i] ?? 0))
      const solved = factor.solve(gap)
      let quad = 0
      for (let i = 0; i < rows; i++) quad += (gap[i] ?? 0) * (solved[i] ?? 0)
      const logDensity = -0.5 * (rows * Math.log(2 * Math.PI) + factor.logDet + quad)
      return massOf(arg(c, 0)) + massOf(arg(c, 1)) + logDensity
    },
    vmp: () => 0,
  },
  traits: { defines: 0 },
})

/**
 * y ~ N(a . w, 1 / precision) for a Gaussian vector w, a constant row a and an observed or Gaussian y: one row of a
 * linear regression with a fixed noise precision. Closed forms under EP and VMP; with an observed y the message to w
 * is the exact rank-one likelihood `[precision y a, precision a a']`.
 *
 * @concept factor-operators
 * @index structure One row of a linear regression y ~ N(a . w, 1 / precision).
 * @example
 * ```ts
 * import { gaussian, gaussianMv, linearRegressionRow, messageArg, valueArg } from 'omai-engine/message-passing'
 * const w = gaussianMv(2)
 * const args = [valueArg(gaussian, 3), messageArg(w, w.flat)]
 * linearRegressionRow.ep?.({ mode: 'ep', to: 1, args, params: { a: [1, 2], precision: 2 }, power: 1 }).message // => [6, 12, 2, 4, 4, 8]
 * ```
 */
export const linearRegressionRow: FactorOperator = defineOperator({
  name: 'linear-regression-row',
  description: 'y ~ N(a . w, 1 / precision) for a constant row a',
  ports: [
    { name: 'y', accepts: ['gaussian'], constant: true },
    { name: 'w', accepts: ['gaussian-mv'] },
  ],
  params: { a: param.vector(), precision: param.positive() },
  check: (params, families) =>
    vectorParam(params, 'a').length === families[1]?.dim
      ? undefined
      : 'linear-regression-row: a must have the length of w',
  rules: {
    ep: messageRule((c): OperatorResult => {
      const a = vectorParam(c.params, 'a')
      const d = a.length
      const tau = numberParam(c.params, 'precision')
      const y = gaussianOf(arg(c, 0))
      if (c.to === 1) {
        const effective = y.variance > 0 ? 1 / (y.variance + 1 / tau) : tau
        return { message: rowMessage(a, effective, y.mean) }
      }
      const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 1)), d)
      const cov = dgemv(covariance, d, d, a)
      let m = 0
      let v = 1 / tau
      for (let i = 0; i < d; i++) {
        m += (a[i] ?? 0) * (mean[i] ?? 0)
        v += (a[i] ?? 0) * (cov[i] ?? 0)
      }
      return { message: gaussianMessage(m, v) }
    }),
    vmp: messageRule((c): OperatorResult => {
      const a = vectorParam(c.params, 'a')
      const d = a.length
      const tau = numberParam(c.params, 'precision')
      if (c.to === 1) return { message: rowMessage(a, tau, gaussianOf(arg(c, 0)).mean) }
      const { mean } = gaussianMvMoments(etaOf(arg(c, 1)), d)
      let m = 0
      for (let i = 0; i < d; i++) m += (a[i] ?? 0) * (mean[i] ?? 0)
      return { message: [tau * m, tau] }
    }),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const a = vectorParam(c.params, 'a')
      const d = a.length
      const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 1)), d)
      const cov = dgemv(covariance, d, d, a)
      let m = 0
      let v = 1 / numberParam(c.params, 'precision')
      for (let i = 0; i < d; i++) {
        m += (a[i] ?? 0) * (mean[i] ?? 0)
        v += (a[i] ?? 0) * (cov[i] ?? 0)
      }
      const y = gaussianOf(arg(c, 0))
      return massOf(arg(c, 0)) + massOf(arg(c, 1)) + logNormal(y.mean, m, y.variance + v)
    },
    vmp: (c) => {
      const a = vectorParam(c.params, 'a')
      const d = a.length
      const tau = numberParam(c.params, 'precision')
      const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 1)), d)
      const cov = dgemv(covariance, d, d, a)
      let m = 0
      let v = 0
      for (let i = 0; i < d; i++) {
        m += (a[i] ?? 0) * (mean[i] ?? 0)
        v += (a[i] ?? 0) * (cov[i] ?? 0)
      }
      const y = gaussianOf(arg(c, 0))
      return 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (y.variance + (y.mean - m) ** 2 + v)
    },
  },
})

/** The rank-one message `[tau y a, tau a a']` of a regression row. */
function rowMessage(a: readonly number[], tau: number, y: number): number[] {
  const d = a.length
  const out = a.map((v) => tau * y * v)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) out.push(tau * (a[i] ?? 0) * (a[j] ?? 0))
  return out
}

/**
 * exp(g + h . x - x' J x / 2) over n scalar Gaussian variables: a joint Gaussian potential in information form (`h`,
 * `J` row-major, `g` the log-constant), such as a Gaussian process prior over function values or a pairwise Gaussian
 * Markov random field factor. The message to a variable is the Schur-complement marginal of the potential times the
 * messages of the others: exact, and Gaussian belief propagation when the graph is a tree or loopy.
 *
 * @concept factor-operators
 * @index structure A joint Gaussian potential over scalar variables in information form.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianJoint, messageArg } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(2, 1))]
 * gaussianJoint.ep?.({ mode: 'ep', to: 0, args, params: { J: [2, -1, -1, 2], h: [0, 0] }, power: 1 }).message.map((v) => Math.round(v * 1e6) / 1e6) // => [0.666667, 1.666667]
 * ```
 */
export const gaussianJoint: FactorOperator = defineOperator({
  name: 'gaussian-joint',
  description: 'exp(g + h x - x J x / 2) over scalar Gaussian variables',
  ports: (params) => vectorParam(params, 'h').map((_, i) => gaussianPort(`x${i}`)),
  params: { h: param.vector(), J: param.vector(), g: param.number({ optional: true }) },
  check: (params) => {
    const n = vectorParam(params, 'h').length
    return vectorParam(params, 'J').length === n * n ? undefined : 'gaussian-joint: J must be n x n for h of length n'
  },
  rules: {
    ep: messageRule((c): OperatorResult => {
      const { Jr, hr, ports, n } = reduceJoint(c)
      const t = ports.indexOf(c.to)
      const rest = ports.filter((p) => p !== c.to)
      // Add the incoming messages of the other ports to the information form.
      const jm = Float64Array.from(Jr)
      const hm = Array.from(hr)
      for (const [k, p] of ports.entries()) {
        if (p === c.to) continue
        const e = etaOf(arg(c, p))
        hm[k] = (hm[k] ?? 0) + (e[0] ?? 0)
        jm[k * n + k] = (jm[k * n + k] ?? 0) + (e[1] ?? 0)
      }
      if (rest.length === 0) return { message: [hm[t] ?? 0, jm[t * n + t] ?? 0] }
      const idx = ports.map((p, k) => (p === c.to ? -1 : k)).filter((k) => k >= 0)
      // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
      const jrr = new Float64Array(idx.length * idx.length)
      for (const [a, ka] of idx.entries())
        for (const [b, kb] of idx.entries()) jrr[a * idx.length + b] = jm[ka * n + kb] ?? 0
      const factor = spd(jrr, idx.length)
      if (factor === undefined)
        throw new ImproperMessageError('gaussian-joint: the other variables are not jointly proper')
      const jtr = idx.map((k) => jm[t * n + k] ?? 0)
      const solveH = factor.solve(idx.map((k) => hm[k] ?? 0))
      const solveJ = factor.solve(jtr)
      let h = hm[t] ?? 0
      let j = jm[t * n + t] ?? 0
      for (let a = 0; a < idx.length; a++) {
        h -= (jtr[a] ?? 0) * (solveH[a] ?? 0)
        j -= (jtr[a] ?? 0) * (solveJ[a] ?? 0)
      }
      return { message: [h, j] }
    }),
    vmp: messageRule((c): OperatorResult => {
      const J = vectorParam(c.params, 'J')
      const h = vectorParam(c.params, 'h')
      const n = h.length
      let lin = h[c.to] ?? 0
      for (let j = 0; j < n; j++) if (j !== c.to) lin -= (J[c.to * n + j] ?? 0) * gaussianOf(arg(c, j)).mean
      return { message: [lin, J[c.to * n + c.to] ?? 0] }
    }),
  },
  evidence: {
    ep: (c) => {
      // The messages need only be proper together (the joint's precision plus theirs positive definite, which `spd` below tests): a
      // cavity that is the site of a scale mixture or a mixture can have a negative precision.
      const { Jr, hr, ports, n, g } = reduceJoint(c)
      const jm = Float64Array.from(Jr)
      const hm = Array.from(hr)
      for (const [k, p] of ports.entries()) {
        const a = arg(c, p)
        if (a.eta === undefined) continue
        hm[k] = (hm[k] ?? 0) + (a.eta[0] ?? 0)
        jm[k * n + k] = (jm[k * n + k] ?? 0) + (a.eta[1] ?? 0)
      }
      const factor = spd(jm, n)
      if (factor === undefined) return Number.NaN
      const x = factor.solve(hm)
      let quad = 0
      for (let i = 0; i < n; i++) quad += (hm[i] ?? 0) * (x[i] ?? 0)
      return g + 0.5 * quad - 0.5 * factor.logDet + 0.5 * n * Math.log(2 * Math.PI)
    },
    vmp: (c) => {
      const J = vectorParam(c.params, 'J')
      const h = vectorParam(c.params, 'h')
      const n = h.length
      let value = numberParam(c.params, 'g', 0)
      for (let i = 0; i < n; i++) {
        const gi = gaussianOf(arg(c, i))
        value += (h[i] ?? 0) * gi.mean - 0.5 * (J[i * n + i] ?? 0) * gi.variance
        for (let j = 0; j < n; j++) value -= 0.5 * (J[i * n + j] ?? 0) * gi.mean * gaussianOf(arg(c, j)).mean
      }
      return value
    },
  },
})

/** The potential with its observed ports conditioned away: the information form over the ports that carry messages. */
function reduceJoint(c: OperatorContext): {
  readonly Jr: Float64Array
  readonly hr: number[]
  readonly ports: number[]
  readonly n: number
  readonly g: number
} {
  const J = vectorParam(c.params, 'J')
  const h = vectorParam(c.params, 'h')
  const total = h.length
  const ports: number[] = []
  const known: number[] = []
  for (let i = 0; i < total; i++) (arg(c, i).value === undefined ? ports : known).push(i)
  const n = ports.length
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const Jr = new Float64Array(n * n)
  const hr = ports.map((p) => h[p] ?? 0)
  let g = numberParam(c.params, 'g', 0)
  for (const [a, pa] of ports.entries()) {
    for (const [b, pb] of ports.entries()) Jr[a * n + b] = J[pa * total + pb] ?? 0
    for (const k of known) hr[a] = (hr[a] ?? 0) - (J[pa * total + k] ?? 0) * scalarOf(arg(c, k))
  }
  for (const k of known) {
    g += (h[k] ?? 0) * scalarOf(arg(c, k))
    for (const l of known) g -= 0.5 * (J[k * total + l] ?? 0) * scalarOf(arg(c, k)) * scalarOf(arg(c, l))
  }
  return { Jr, hr, ports, n, g }
}
