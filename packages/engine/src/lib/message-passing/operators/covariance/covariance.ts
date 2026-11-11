/**
 * The factors of a multivariate Gaussian whose covariance is learned: `x ~ N(mean, L^-1)` with the precision matrix `L` a Wishart
 * variable (`mvn-wishart`), and the joint conjugate form in which the mean is scaled by the precision, `x ~ N(mu, L^-1)` with `(mu, L)` a
 * normal-Wishart variable (`mvn-normal-wishart`). The first reduces to a closed form when the vector and the mean are
 * known (the conjugate update of the Wishart); with Gaussian cavities on the vector and the mean the factor integrates both out
 * (a function of `L` alone, `N(r; 0, V + L^-1)` for the difference `r = x - mean` with cavity moments `r ~ N(m, V)`) and expectation
 * propagation matches the Wishart to the moments of the tilted matrix by a Bartlett quadrature, and the Gaussians to the mixture of
 * their conditionals over the nodes. The variational rule is closed form throughout.
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, evidenceOrNaN } from '../../core/_messages.ts'
import { rawSum, siteMessage, vectorOf } from '../../core/args.ts'
import { ImproperMessageError } from '../../core/errors.ts'
import { normalWishart, wishart, wishartParams } from '../../core/families-matrix.ts'
import { gaussianMvEta } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { BARTLETT_ORDER, choleskyInPlace, type WishartRule, wishartRule } from '../../numerics/_bartlett.ts'
import { type Evaluator, evaluatorOf, type GaussianSide, gaussianSide, NO_ROWS, type Rows, rowsOf } from './_sides.ts'

const LOG_2PI = Math.log(2 * Math.PI)
const dimOf = (c: OperatorContext): number => arg(c, 2).family.dim

/** The number of nodes of the Wishart quadrature at dimension `d`: refuse early what it cannot do. */
function checkNumeric(d: number): void {
  if (BARTLETT_ORDER[d] === undefined)
    throw new OmaiError(
      'not-implemented',
      `mvn-wishart: a Gaussian cavity on the vector or the mean makes the tilted precision a ${d} by ${d} integral; the quadrature is built for dimensions up to 3 (an observed vector and a known mean are exact in any dimension)`,
    )
}

/** The matrix of a Wishart cavity (degrees of freedom, Cholesky factor of its scale) or an error. */
function wishartCavity(eta: Eta, d: number): { readonly nu: number; readonly chol: Float64Array } {
  const p = wishartParams(eta, d)
  if (p === undefined) throw new ImproperMessageError('a Wishart message has no density')
  const chol = Float64Array.from(p.scale)
  if (!choleskyInPlace(chol, d)) throw new ImproperMessageError('a Wishart message has no density')
  return { nu: p.nu, chol }
}

/** The Gaussian messages of the exact conjugate case: the matrix a fully known Gaussian says to its precision. */
function conjugateMessage(r: ArrayLike<number>, d: number, rows: Rows = NO_ROWS): Eta {
  const out: number[] = [0.5 * rows.count]
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      out.push(-0.5 * (rows.count * (r[i] ?? 0) * (r[j] ?? 0) + (rows.scatter[i * d + j] ?? 0)))
  return out
}

interface Tilted {
  /** log of the integral of the factor against every incoming message (unnormalised). */
  readonly logZ: number
  readonly rule: WishartRule
  /** The normalised tilted weights of the nodes. */
  readonly p: Float64Array
}

/**
 * The tilted distribution of the precision matrix on a Bartlett grid: the base measure is the Wishart cavity, or (when the difference
 * `x - mean` has a cavity much narrower than the matrix's own scale, the usual case) the cavity times the exact conjugate part
 * of the factor, which leaves a smooth correction for the quadrature, exact when the vector is known.
 */
function tiltOver(
  cavity: Eta,
  eval1: Evaluator,
  x: GaussianSide,
  mean: GaussianSide,
  d: number,
  rows: Rows = NO_ROWS,
): Tilted {
  checkNumeric(d)
  const family = wishart(d)
  if (wishartParams(cavity, d) === undefined) throw new ImproperMessageError('a Wishart message has no density')
  const k = rows.count
  const r = new Float64Array(d)
  const v = new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    r[i] = (x.mean[i] as number) - (mean.mean[i] as number)
    for (let j = 0; j < d; j++) v[i * d + j] = (x.cov[i * d + j] as number) + (mean.cov[i * d + j] as number)
  }
  // The scatter of the rows about their mean is always conjugate: |L|^((n - 1) / 2) exp(-tr(L scatter) / 2) is a Wishart message.
  const scatterPart: Eta = [0.5 * (k - 1), ...Array.from({ length: d * d }, (_, e) => -0.5 * (rows.scatter[e] ?? 0))]
  const base0 = k > 1 ? rawSum(cavity, scatterPart) : cavity
  const cav = wishartParams(base0, d)
  if (cav === undefined) throw new ImproperMessageError('a Wishart message has no density')
  // trace(n V E[L]) / d: how wide the difference's cavity is against the width the rows give it
  let tr = 0
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) tr += (v[i * d + j] as number) * (cav.mean[j * d + i] as number)
  const differencePart: Eta = [
    0.5,
    ...Array.from({ length: d * d }, (_, e) => -0.5 * k * (r[Math.floor(e / d)] as number) * (r[e % d] as number)),
  ]
  const conjugate = (k * tr) / d < 0.3 && !x.flat && !mean.flat
  const eta = conjugate ? rawSum(base0, differencePart) : base0
  const { nu, chol } = wishartCavity(eta, d)
  const rule = wishartRule(nu, chol, d)
  const logw = new Float64Array(rule.count)
  let top = Number.NEGATIVE_INFINITY
  for (let node = 0; node < rule.count; node++) {
    const at = node * d * d
    let lw = Math.log(rule.weight[node] as number) + eval1(rule.lambda, at, rule.logDet[node] as number, -1)
    // remove what is already in the base measure: |L|^((n - 1) / 2) exp(-tr(L scatter) / 2), and, in the conjugate form, |L|^(1/2) exp(-n r' L r / 2)
    let quad = 0
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) {
        const l = rule.lambda[at + i * d + j] as number
        quad += (rows.scatter[i * d + j] ?? 0) * l
        if (conjugate) quad += k * (r[i] as number) * (r[j] as number) * l
      }
    lw -= 0.5 * (k - 1 + (conjugate ? 1 : 0)) * (rule.logDet[node] as number) - 0.5 * quad
    logw[node] = lw
    if (lw > top) top = lw
  }
  if (!Number.isFinite(top))
    throw new ImproperMessageError('mvn-wishart: the factor has no mass under the incoming messages')
  const p = new Float64Array(rule.count)
  let total = 0
  for (let node = 0; node < rule.count; node++) {
    const e = Math.exp((logw[node] as number) - top)
    p[node] = e
    total += e
  }
  for (let node = 0; node < rule.count; node++) p[node] = (p[node] as number) / total
  return { logZ: family.logPartition(eta) + top + Math.log(total), rule, p }
}

/** The message of the Wishart port: the matched moments against the cavity. */
function messageToPrecision(c: OperatorContext): Eta {
  const d = dimOf(c)
  const x = gaussianSide(arg(c, 0), d)
  const mean = gaussianSide(arg(c, 1), d)
  const cavity = arg(c, 2).eta as Eta
  const family = wishart(d)
  const rows = rowsOf(c.params, d)
  if (x.point && mean.point) {
    const r = Array.from(x.mean, (v, i) => v - (mean.mean[i] as number))
    return conjugateMessage(r, d, rows)
  }
  const t = tiltOver(cavity, evaluatorOf(x, mean, d, rows), x, mean, d, rows)
  const moments = new Array<number>(1 + d * d).fill(0)
  for (let k = 0; k < t.rule.count; k++) {
    const pk = t.p[k] as number
    moments[0] = (moments[0] as number) + pk * ((t.rule.logDet[k] as number) - t.rule.logDetBias)
    for (let e = 0; e < d * d; e++)
      moments[1 + e] = (moments[1 + e] as number) + pk * (t.rule.lambda[k * d * d + e] as number)
  }
  return siteMessage(family, cavity, moments)
}

/** The message of a Gaussian port: the mixture of the conditionals over the nodes, matched, against the cavity. */
function messageToGaussian(c: OperatorContext): Eta {
  const d = dimOf(c)
  const x = gaussianSide(arg(c, 0), d)
  const mean = gaussianSide(arg(c, 1), d)
  const rows = rowsOf(c.params, d)
  const eval1 = evaluatorOf(x, mean, d, rows)
  const target = c.to as 0 | 1
  const side = target === 0 ? x : mean
  const t = tiltOver(arg(c, 2).eta as Eta, eval1, x, mean, d, rows)
  const m = new Float64Array(d)
  const s = new Float64Array(d * d)
  const nodeMean = new Float64Array(d)
  const nodeCov = new Float64Array(d * d)
  // the weights of the nodes are those of the precision's tilted distribution; the conditionals at each node are Gaussian
  for (let k = 0; k < t.rule.count; k++) {
    const pk = t.p[k] as number
    if (!(pk > 1e-14)) continue
    eval1(t.rule.lambda, k * d * d, t.rule.logDet[k] as number, target, nodeMean, nodeCov)
    for (let i = 0; i < d; i++) {
      m[i] = (m[i] as number) + pk * (nodeMean[i] as number)
      for (let j = 0; j < d; j++)
        s[i * d + j] =
          (s[i * d + j] as number) +
          pk * ((nodeCov[i * d + j] as number) + (nodeMean[i] as number) * (nodeMean[j] as number))
    }
  }
  const cov = Array.from(s, (v, e) => v - (m[Math.floor(e / d)] as number) * (m[e % d] as number))
  const proj = gaussianMvEta(Array.from(m), cov)
  const out = proj.map((v, i) => v - (side.flat ? 0 : ((arg(c, target).eta as Eta)[i] ?? 0)))
  return out
}

/** The mean and second moment of a Gaussian vector argument under its belief (a variational rule's input). */
function momentsOf(a: Arg, d: number): { readonly mean: number[]; readonly second: number[] } {
  if (a.value !== undefined) {
    const v = vectorOf(a)
    return {
      mean: Array.from(v),
      second: Array.from({ length: d * d }, (_, e) => (v[Math.floor(e / d)] as number) * (v[e % d] as number)),
    }
  }
  const mo = a.family.moments(a.eta as Eta)
  return { mean: mo.slice(0, d), second: mo.slice(d) }
}

/** `E[(x - mean)(x - mean)']` for independent beliefs. */
function expectedOuter(c: OperatorContext, d: number): { readonly mean: number[]; readonly outer: number[] } {
  const x = momentsOf(arg(c, 0), d)
  const m = momentsOf(arg(c, 1), d)
  const outer = Array.from({ length: d * d }, (_, e) => {
    const i = Math.floor(e / d)
    const j = e % d
    return (
      (x.second[e] as number) -
      (x.mean[i] as number) * (m.mean[j] as number) -
      (m.mean[i] as number) * (x.mean[j] as number) +
      (m.second[e] as number)
    )
  })
  return { mean: x.mean.map((v, i) => v - (m.mean[i] as number)), outer }
}

/**
 * `x ~ N(mean, L^-1)` for a Gaussian vector `x`, a Gaussian mean (either may be observed or constant) and a Wishart precision matrix `L`:
 * a multivariate normal whose covariance is learned (correlated random effects, a population of vectors, a Gaussian
 * likelihood with unknown noise covariance). The fully known case is the exact conjugate update of the Wishart (the matrix of the
 * residual outer product). With Gaussian cavities on `x` or `mean` expectation propagation integrates them out (`N(r; 0, V + L^-1)` for the
 * difference) and matches the Wishart to the moments of the tilted matrix, found by the Bartlett quadrature of the Wishart
 * (dimensions up to 3), and each Gaussian to the mixture of its conditionals over the matrix; a cavity with no information yet is
 * read as a very wide one. The variational rule is closed form (the expected residual outer product for the Wishart, the mean precision
 * for the Gaussians).
 *
 * The factor may stand for many rows at once (the parameters `count` and `scatter`): `x` is then the sample mean of `count` i.i.d. vectors, and
 * `scatter` their scatter about it (`sum (y_i - ybar)(y_i - ybar)'`), which is all the likelihood of the rows depends on, so a plate of
 * rows with a shared mean costs one factor whatever its size.
 *
 * @concept factor-operators
 * @index structure A multivariate Gaussian with a Wishart precision matrix: a learned covariance.
 * @example
 * ```ts
 * import { gaussianMv, identityMatrix, mvnWishart, valueArg, messageArg, wishart, wishartEta } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussianMv(2), [1, -2]), valueArg(gaussianMv(2), [0, 0]), messageArg(wishart(2), wishartEta(5, identityMatrix(2), 2))]
 * mvnWishart.ep?.({ mode: 'ep', to: 2, args, params: {}, power: 1 }).message // => [0.5, -0.5, 1, 1, -2]
 * ```
 */
export const mvnWishart: FactorOperator = {
  name: 'mvn-wishart',
  description: 'x ~ N(mean, L^-1) for Gaussian vectors x and mean and a Wishart precision matrix L',
  ports: [
    { name: 'x', accepts: ['gaussian-mv'], constant: true },
    { name: 'mean', accepts: ['gaussian-mv'], constant: true },
    { name: 'precision', accepts: ['wishart'] },
  ],
  params: { count: param.number({ min: 1, optional: true }), scatter: param.vector({ optional: true }) },
  support: { ep: 'numeric', vmp: 'exact' },
  check: (_params, families) => {
    const dims = families.map((f) => f.dim)
    return dims.every((v) => v === dims[0])
      ? undefined
      : 'mvn-wishart: the vector, the mean and the matrix have one dimension'
  },
  ep: (c): OperatorResult => ({ message: c.to === 2 ? messageToPrecision(c) : messageToGaussian(c) }),
  vmp: (c): OperatorResult => {
    const d = dimOf(c)
    const w = arg(c, 2)
    const lambda = w.family.moments(w.eta as Eta)
    const rows = rowsOf(c.params, d)
    if (c.to === 2) {
      const { outer } = expectedOuter(c, d)
      return { message: [0.5 * rows.count, ...outer.map((v, e) => -0.5 * (rows.count * v + (rows.scatter[e] ?? 0)))] }
    }
    const other = momentsOf(arg(c, 1 - c.to), d)
    const meanLambda = lambda.slice(1).map((v) => rows.count * v)
    const hs = Array.from({ length: d }, (_, i) => {
      let s = 0
      for (let j = 0; j < d; j++) s += (meanLambda[i * d + j] as number) * (other.mean[j] as number)
      return s
    })
    return { message: [...hs, ...meanLambda] }
  },
  expectedLog: (c) => {
    const d = dimOf(c)
    const w = arg(c, 2)
    const lambda = w.family.moments(w.eta as Eta)
    const { outer } = expectedOuter(c, d)
    const rows = rowsOf(c.params, d)
    let tr = 0
    for (let e = 0; e < d * d; e++)
      tr += (lambda[1 + e] as number) * (rows.count * (outer[e] as number) + (rows.scatter[e] ?? 0))
    return 0.5 * rows.count * (lambda[0] as number) - 0.5 * rows.count * d * LOG_2PI - 0.5 * tr
  },
  logEvidence: (c) => {
    const d = dimOf(c)
    const w = arg(c, 2)
    if (c.args.some((a) => a.eta !== undefined && !a.family.isProper(a.eta))) return Number.NaN
    const x = gaussianSide(arg(c, 0), d)
    const mean = gaussianSide(arg(c, 1), d)
    const family = wishart(d)
    const rows = rowsOf(c.params, d)
    if (x.point && mean.point) {
      const r = Array.from(x.mean, (v, i) => v - (mean.mean[i] as number))
      return family.logPartition(rawSum(w.eta as Eta, conjugateMessage(r, d, rows))) - 0.5 * rows.count * d * LOG_2PI
    }
    return evidenceOrNaN(() => {
      return tiltOver(w.eta as Eta, evaluatorOf(x, mean, d, rows), x, mean, d, rows).logZ
    })
  },
}

/** The normal-Wishart message that a fully observed Gaussian row sends: `(1/2, -y y' / 2, y, -1/2)` on `(log |L|, L, L mu, mu' L mu)`. */
function rowMessage(y: ArrayLike<number>, d: number): Eta {
  const out: number[] = [0.5]
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) out.push(-0.5 * (y[i] ?? 0) * (y[j] ?? 0))
  for (let i = 0; i < d; i++) out.push(y[i] ?? 0)
  out.push(-0.5)
  return out
}

/**
 * `x ~ N(mu, L^-1)` for an observed vector `x` and a normal-Wishart variable `(mu, L)`: the exact conjugate update of a multivariate Gaussian with
 * the mean and the precision matrix both unknown (the matrix version of `gaussian-normal-gamma`). The message is the same under every rule: the
 * natural parameters `(1/2, -x x' / 2, x, -1/2)` on the sufficient statistics `(log |L|, L, L mu, mu' L mu)`; the posterior after `n` rows
 * is the normal-Wishart with `beta + n`, `nu + n` and the scatter matrix added to the scale, the marginal of the mean a
 * multivariate Student t.
 *
 * @concept factor-operators
 * @index structure The conjugate update of a Gaussian vector with unknown mean and precision matrix (normal-Wishart).
 * @example
 * ```ts
 * import { mvnNormalWishart, messageArg, normalWishart, normalWishartEta, identityMatrix, valueArg, gaussianMv } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussianMv(2), [1, 2]), messageArg(normalWishart(2), normalWishartEta([0, 0], 1, 4, identityMatrix(2)))]
 * mvnNormalWishart.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message.length // => 8
 * ```
 */
export const mvnNormalWishart: FactorOperator = {
  name: 'mvn-normal-wishart',
  description: 'x ~ N(mu, L^-1) for an observed vector x and a normal-Wishart variable (mu, L)',
  ports: [
    { name: 'x', accepts: ['gaussian-mv'], observed: true, constant: true },
    { name: 'theta', accepts: ['normal-wishart'] },
  ],
  params: {},
  support: { ep: 'exact', vmp: 'exact', bp: 'exact' },
  check: (_params, families) =>
    families[0]?.dim === families[1]?.dim
      ? undefined
      : 'mvn-normal-wishart: the vector and the matrix have one dimension',
  ep: (c) => ({ message: rowMessage(vectorOf(arg(c, 0)), arg(c, 1).family.dim) }),
  vmp: (c) => ({ message: rowMessage(vectorOf(arg(c, 0)), arg(c, 1).family.dim) }),
  bp: (c) => ({ message: rowMessage(vectorOf(arg(c, 0)), arg(c, 1).family.dim) }),
  logEvidence: (c) => {
    const d = arg(c, 1).family.dim
    const a = arg(c, 1)
    const f = normalWishart(d)
    return f.logPartition(rawSum(a.eta as Eta, rowMessage(vectorOf(arg(c, 0)), d))) - 0.5 * d * LOG_2PI
  },
  expectedLog: (c) => {
    const d = arg(c, 1).family.dim
    const y = vectorOf(arg(c, 0))
    const mo = arg(c, 1).family.moments(arg(c, 1).eta as Eta)
    // E[0.5 log |L| - 0.5 (y - mu)' L (y - mu)] - (d / 2) log 2 pi
    let quad = 0
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) quad += (y[i] as number) * (mo[1 + i * d + j] as number) * (y[j] as number)
      quad -= 2 * (y[i] as number) * (mo[1 + d * d + i] as number)
    }
    quad += mo[1 + d * d + d] as number
    return 0.5 * (mo[0] as number) - 0.5 * d * LOG_2PI - 0.5 * quad
  },
}
