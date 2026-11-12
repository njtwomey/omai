/**
 * The joint site of a group of Gaussian vectors that share a population mean and a Wishart precision matrix: `x_j ~ N(mean, L^-1)` for
 * `j = 1..J` as ONE factor with the group vectors, the mean and the matrix as its ports (`mvn-wishart-group`), where the per-vector
 * factor (`mvn-wishart`) has three ports each. Given the matrix everything else is Gaussian, so the factor integrates the vectors and the mean out
 * together in closed form (`N(m_j; mean, V_j + L^-1)` for each cavity `x_j ~ N(m_j, V_j)`, then the mean against its own cavity)
 * and expectation propagation matches the Wishart to the tilted matrix by the Bartlett quadrature, each vector and the mean to their mixtures
 * over the nodes. Splitting the group into one factor per vector leaves a loop through the shared mean and the shared matrix, and with few groups
 * the mean's spread comes out too narrow; one factor has no such loop, so each marginal is matched to the moments of the exact tilt.
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, evidenceOrNaN, keyOf, memoised } from '../../core/_messages.ts'
import { numberParam, rawSum, siteMessage } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { ImproperMessageError } from '../../core/errors.ts'
import { wishart } from '../../core/families-matrix.ts'
import { gaussianMvEta } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, OperatorContext, OperatorResult, Port } from '../../core/types.ts'
import {
  BARTLETT_ORDER,
  choleskyInPlace,
  choleskyLogDet,
  choleskySolveInPlace,
  type WishartRule,
  wishartRule,
} from '../../numerics/_bartlett.ts'
import { shiftedSum } from '../../numerics/_moments.ts'
import { type GaussianSide, gaussianSide, wishartCavity } from './_sides.ts'
import { mvnWishart } from './covariance.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/** The number of vectors of the factor. */
const groupsOf = (params: OperatorContext['params']): number =>
  Math.max(1, Math.floor(numberParam(params, 'groups', 1)))

/** The ports: the group vectors, the shared mean, the precision matrix. */
const portsOf = (params: OperatorContext['params']): readonly Port[] => [
  ...Array.from(
    { length: groupsOf(params) },
    (_, j): Port => ({ name: `x${j}`, accepts: ['gaussian-mv'], constant: true }),
  ),
  { name: 'mean', accepts: ['gaussian-mv'] },
  { name: 'precision', accepts: ['wishart'] },
]

/** What the closed-form integral over the vectors and the mean reads at one matrix: the arguments as Gaussian sides. */
interface Sides {
  readonly d: number
  readonly groups: readonly GaussianSide[]
  readonly mean: GaussianSide
}

function sidesOf(c: OperatorContext): Sides {
  const groups = groupsOf(c.params)
  const d = arg(c, groups + 1).family.dim
  if (BARTLETT_ORDER[d] === undefined)
    throw new OmaiError(
      'not-implemented',
      `mvn-wishart-group: the tilted precision is a ${d} by ${d} integral; the quadrature is built for dimensions up to 3`,
    )
  const mean = gaussianSide(arg(c, groups), d)
  if (mean.point)
    throw new OmaiError(
      'value',
      'mvn-wishart-group: the mean is a Gaussian variable (a known mean is one mvn-wishart factor per vector)',
    )
  return { d, groups: Array.from({ length: groups }, (_, j) => gaussianSide(arg(c, j), d)), mean }
}

/**
 * The integral of the factor against the cavities of the vectors and the mean, as a function of the precision matrix at the nodes of a
 * quadrature: `log I(L)` and, when asked, the mean and covariance of each Gaussian under the tilted conditional given the matrix. With
 * cavities `exp(h_j x - x' W_j x / 2)` for each vector, `exp(h_0 m - m' W_0 m / 2)` for the mean and `T_j = (W_j + L)^-1`:
 * integrating `x_j` leaves `c_j + (L T_j h_j) m - m' (L - L T_j L) m / 2` with `c_j = (log |L| - log |W_j + L| + h_j' T_j h_j) / 2`, a Gaussian in
 * the mean of precision `P = W_0 + sum (L - L T_j L)` and linear term `g = h_0 + sum L T_j h_j`.
 */
function nodeEvaluator(
  s: Sides,
): (lambda: Float64Array, offset: number, logDet: number, conditionals: Float64Array | undefined) => number {
  const { d, groups, mean } = s
  const dd = d * d
  const count = groups.length
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

/** The result of one tilt, shared by every port of the factor and its evidence. */
interface Tilt {
  readonly logZ: number
  /** `[E log |L|, E L]` under the tilted matrix, the quadrature's bias in the logarithm removed. */
  readonly moments: number[]
  /** The tilted mean and covariance of each port's Gaussian (the vectors, then the mean), by `d` and `d * d` blocks. */
  readonly mean: readonly Float64Array[]
  readonly cov: readonly Float64Array[]
}

/**
 * The tilted distribution of the precision matrix on a Bartlett grid, and the moments of each Gaussian under it. The base measure is the
 * Wishart cavity times the conjugate part of the factor that the vectors' cavity means give it (`|L|^((J - 1) / 2)` and their scatter about their
 * mean: what the matrix would be told by `J` vectors with an unknown mean), so the quadrature integrates a smooth remainder; a vector whose cavity
 * says nothing yet does not enter it.
 */
function tiltOf(c: OperatorContext): Tilt {
  const s = sidesOf(c)
  const { d, groups } = s
  const dd = d * d
  const count = groups.length
  const family = wishart(d)
  const cavity = arg(c, count + 1).eta as Eta
  const known = groups.filter((x) => !x.flat)
  let eta = cavity
  let exponent = 0
  const scatter = new Float64Array(dd)
  if (known.length >= 2) {
    const mean = new Float64Array(d)
    for (const x of known)
      for (let i = 0; i < d; i++) mean[i] = (mean[i] as number) + (x.mean[i] as number) / known.length
    for (const x of known)
      for (let i = 0; i < d; i++)
        for (let k = 0; k < d; k++)
          scatter[i * d + k] =
            (scatter[i * d + k] as number) +
            ((x.mean[i] as number) - (mean[i] as number)) * ((x.mean[k] as number) - (mean[k] as number)) +
            (1 - 1 / known.length) * (x.cov[i * d + k] as number)
    exponent = 0.5 * (known.length - 1)
    eta = rawSum(cavity, [exponent, ...Array.from(scatter, (v) => -0.5 * v)])
  }
  const { nu, chol } = wishartCavity(eta, d)
  const rule: WishartRule = wishartRule(nu, chol, d)
  const evaluate = nodeEvaluator(s)
  const logw = new Float64Array(rule.count)
  for (let node = 0; node < rule.count; node++) {
    const at = node * dd
    let quad = 0
    for (let e = 0; e < dd; e++) quad += (scatter[e] as number) * (rule.lambda[at + e] as number)
    const logDet = rule.logDet[node] as number
    logw[node] =
      Math.log(rule.weight[node] as number) +
      evaluate(rule.lambda, at, logDet, undefined) -
      (exponent * logDet - 0.5 * quad)
  }
  const { top, weights, total } = shiftedSum(logw)
  if (!Number.isFinite(top))
    throw new ImproperMessageError('mvn-wishart-group: the factor has no mass under the incoming messages')
  const moments = new Array<number>(1 + dd).fill(0)
  const mean = Array.from({ length: count + 1 }, () => new Float64Array(d))
  const second = Array.from({ length: count + 1 }, () => new Float64Array(dd))
  const conditionals = new Float64Array((count + 1) * (d + dd))
  for (let node = 0; node < rule.count; node++) {
    const pk = (weights[node] as number) / total
    moments[0] = (moments[0] as number) + pk * ((rule.logDet[node] as number) - rule.logDetBias)
    for (let e = 0; e < dd; e++)
      moments[1 + e] = (moments[1 + e] as number) + pk * (rule.lambda[node * dd + e] as number)
    if (!(pk > 1e-14)) continue
    evaluate(rule.lambda, node * dd, rule.logDet[node] as number, conditionals)
    for (let port = 0; port <= count; port++) {
      if (port < count && (groups[port] as GaussianSide).point) continue
      const at = port * (d + dd)
      const mk = mean[port] as Float64Array
      const sk = second[port] as Float64Array
      for (let i = 0; i < d; i++) {
        const mi = conditionals[at + i] as number
        mk[i] = (mk[i] as number) + pk * mi
        for (let k = 0; k < d; k++)
          sk[i * d + k] =
            (sk[i * d + k] as number) +
            pk * ((conditionals[at + d + i * d + k] as number) + mi * (conditionals[at + k] as number))
      }
    }
  }
  const cov = second.map((sk, port) => {
    const mk = mean[port] as Float64Array
    return Float64Array.from(sk, (v, e) => v - (mk[Math.floor(e / d)] as number) * (mk[e % d] as number))
  })
  return { logZ: family.logPartition(eta) + top + Math.log(total), moments, mean, cov }
}

/** The tilt of the factor, kept in the engine's cache for the stretch the messages of one sweep share. */
function tiltFor(c: OperatorContext): Tilt {
  return memoised(c, `mvn-wishart-group:${keyOf(c.args.map((a) => a.value ?? a.eta))}`, () => tiltOf(c))
}

/** The message to the precision matrix or to a Gaussian port. */
function messageRuleEp(c: OperatorContext): OperatorResult {
  const count = groupsOf(c.params)
  const d = arg(c, count + 1).family.dim
  const t = tiltFor(c)
  if (c.to === count + 1) return { message: siteMessage(wishart(d), arg(c, count + 1).eta as Eta, t.moments) }
  const side = gaussianSide(arg(c, c.to), d)
  const proj = gaussianMvEta(Array.from(t.mean[c.to] as Float64Array), Array.from(t.cov[c.to] as Float64Array))
  const cavity = arg(c, c.to).eta as Eta
  return { message: proj.map((v, i) => v - (side.flat ? 0 : (cavity[i] ?? 0))) }
}

/** The per-vector factor `mvn-wishart` as it reads vector `j`: its arguments and the rule that is closed form. */
function asPair(c: OperatorContext, j: number, to: 0 | 1 | 2): OperatorContext {
  const count = groupsOf(c.params)
  return { ...c, to, args: [arg(c, j), arg(c, count), arg(c, count + 1)], params: {} }
}

/**
 * `x_j ~ N(mean, L^-1)` for `J` Gaussian vectors that share a Gaussian population mean and a Wishart precision matrix: the correlated random
 * effects of a hierarchical model with the mean and the covariance both learned, as one factor. Given the matrix the vectors and the mean are
 * jointly Gaussian, so the factor integrates them out together (`N(m_j; mean, V_j + L^-1)` for each cavity, then the mean against its cavity) and
 * expectation propagation matches the Wishart to the moments of the tilted matrix by the Bartlett quadrature (dimensions up to 3), and each vector and
 * the mean to the mixture of their conditionals over the nodes. One factor takes the place of `J` factors of `mvn-wishart`
 * that would form a loop through the shared mean and matrix (and with few groups give a mean too narrow and a matrix too wide); the cost of one tilt is shared by
 * all its ports. The variational rule is the sum of the per-vector rules; a known mean is `mvn-wishart` per vector, exact on its own.
 *
 * @concept factor-operators
 * @index structure A group of Gaussian vectors sharing a Gaussian mean and a Wishart precision matrix: one joint site.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, identityMatrix, messageArg, mvnWishartGroup, wishart, wishartEta } from 'omai-engine/message-passing'
 * const g = gaussianMv(2)
 * const x = (m: number[]) => messageArg(g, gaussianMvEta(m, [0.2, 0, 0, 0.2]))
 * const args = [x([1, 0]), x([-1, 0.5]), x([0.5, -1]), messageArg(g, gaussianMvEta([0, 0], [9, 0, 0, 9])), messageArg(wishart(2), wishartEta(4, identityMatrix(2), 2))]
 * mvnWishartGroup.ep?.({ mode: 'ep', to: 4, args, params: { groups: 3 }, power: 1 }).message.length // => 5
 * ```
 */
export const mvnWishartGroup: FactorOperator = defineOperator({
  name: 'mvn-wishart-group',
  description:
    'x_j ~ N(mean, L^-1) for J Gaussian vectors sharing a Gaussian mean and a Wishart precision matrix L: one joint site',
  ports: portsOf,
  params: { groups: param.integer({ min: 1 }) },
  check: (params, families) => {
    const dims = families.map((f) => f.dim)
    if (families.length !== groupsOf(params) + 2)
      return 'mvn-wishart-group: the vectors, the mean and the matrix are one port each'
    return dims.every((v) => v === dims[0])
      ? undefined
      : 'mvn-wishart-group: the vectors, the mean and the matrix have one dimension'
  },
  rules: {
    // Like `mvn-wishart`: a message rule, since the message to a Gaussian is taken against its cavity only when that is not flat.
    ep: messageRule(messageRuleEp, 'numeric'),
    vmp: messageRule((c): OperatorResult => {
      const count = groupsOf(c.params)
      const rule = mvnWishart.vmp as NonNullable<typeof mvnWishart.vmp>
      if (c.to < count) return rule(asPair(c, c.to, 0))
      let total: Eta | undefined
      for (let j = 0; j < count; j++) {
        const message = rule(asPair(c, j, c.to === count ? 1 : 2)).message
        total = total === undefined ? message : rawSum(total, message)
      }
      return { message: total as Eta }
    }),
  },
  evidence: {
    ep: (c) => {
      if (c.args.some((a) => a.eta !== undefined && !a.family.isProper(a.eta))) return Number.NaN
      return evidenceOrNaN(() => tiltFor(c).logZ)
    },
    vmp: (c) => {
      const count = groupsOf(c.params)
      const expectedLog = mvnWishart.expectedLog as NonNullable<typeof mvnWishart.expectedLog>
      let total = 0
      for (let j = 0; j < count; j++) total += expectedLog(asPair(c, j, 0))
      return total
    },
  },
})
