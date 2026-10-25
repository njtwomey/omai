/**
 * The exponential-family form p(x) = h(x) exp(eta . T(x) - A(eta)) of the distributions that have one: natural
 * parameters, sufficient statistics, log-partition function and base measure for the normal, gamma, beta, Poisson,
 * binomial, categorical and others, with the moments that follow from differentiating the log-partition.
 */

import { grad, jvp } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { cholesky, inv, solveTriangular } from '../numerics/linalg/index.ts'
import { lbeta, lbinom, lgamma, log1mexp, logIv } from '../numerics/special/index.ts'
import {
  abs,
  add,
  arctan2,
  broadcastShapes,
  broadcastTo,
  cast,
  concat,
  cos,
  diagonal,
  div,
  exp,
  expandDims,
  log,
  logsumexp,
  matmul,
  mul,
  neg,
  oneHot,
  pow,
  promoteFloat,
  reshape,
  sigmoid,
  sin,
  sliceAxis,
  softmax,
  softplus,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import { beta } from './continuous/bounded.ts'
import { exponential, gamma, inverseGamma, inverseGaussian, weibull } from './continuous/gamma.ts'
import { laplace, pareto } from './continuous/heavy-tailed.ts'
import { halfNormal, logNormal, normal } from './continuous/normal.ts'
import { bernoulli, binomial } from './discrete/binary.ts'
import { categorical, oneHotCategorical } from './discrete/categorical.ts'
import { geometric, negativeBinomial, poisson } from './discrete/counts.ts'
import type { Distribution } from './distribution.ts'
import { vonMises } from './multivariate/directional.ts'
import { multigammaln, wishart } from './multivariate/matrix.ts'
import { multivariateNormal } from './multivariate/normal.ts'
import { dirichlet, multinomial } from './multivariate/simplex.ts'

/**
 * A distribution written as an exponential family: p(x) = h(x) exp(eta . T(x) - A(eta)), with natural parameters eta
 * (one value per part, each with the batch shape plus the part's own event axes), sufficient statistics T, log
 * partition A and log base measure log h. Parameters the family holds fixed (the trials of a binomial, the location of
 * a Laplace) stay at the distribution's values.
 *
 * @concept exponential-families
 */
export interface ExponentialFamilyForm {
  readonly family: string
  readonly natural: readonly Value[]
  /** How many trailing axes each natural parameter part has beyond the batch. */
  readonly partRanks: readonly number[]
  sufficientStatistics(x: TensorLike): readonly Value[]
  /** A(eta), at the given natural parameters (default the distribution's own). */
  logPartition(eta?: readonly Value[]): Value
  logBaseMeasure(x: TensorLike): Value
  /** The mean parameters E[T(X)] = grad A(eta), by differentiation of the log partition. */
  meanParameters(): readonly Value[]
  /** The Fisher information in the natural parameters, the Hessian of A over their flattened entries: `[..., k, k]`. */
  fisherInformation(): Value
  /** The distribution of this family (with the same fixed parameters) at other natural parameters. */
  fromNatural(eta: readonly Value[]): Distribution
}

interface FormSpec {
  readonly family: string
  readonly natural: readonly Value[]
  readonly partRanks?: readonly number[]
  statistics(x: Value): readonly Value[]
  logPartition(eta: readonly Value[]): Value
  logBaseMeasure(x: Value): Value
  fromNatural(eta: readonly Value[]): Distribution
}

const part = (eta: readonly Value[], i: number): Value => eta[i] as Value

function defineForm(spec: FormSpec): ExponentialFamilyForm {
  const ranks = spec.partRanks ?? spec.natural.map(() => 0)
  const v = (x: TensorLike): Value => promoteFloat(x)[0] as Value
  const batch = spec.natural.reduce<number[]>(
    (s, p, i) => broadcastShapes(s, p.shape.slice(0, p.shape.length - (ranks[i] ?? 0))),
    [],
  )
  // The natural parameters as one flat vector per batch member, and back.
  const sizes = spec.natural.map((p, i) => p.shape.slice(p.shape.length - (ranks[i] ?? 0)).reduce((a, b) => a * b, 1))
  const eventShapes = spec.natural.map((p, i) => p.shape.slice(p.shape.length - (ranks[i] ?? 0)))
  const flatten = (eta: readonly Value[]): Value =>
    concat(
      eta.map((p, i) => reshape(broadcastTo(p, [...batch, ...(eventShapes[i] ?? [])]), [...batch, sizes[i] ?? 1])),
      -1,
    )
  const unflatten = (theta: Value): Value[] => {
    let offset = 0
    return sizes.map((size, i) => {
      const piece = sliceAxis(theta, -1, offset, offset + size)
      offset += size
      return reshape(piece, [...theta.shape.slice(0, -1), ...(eventShapes[i] ?? [])])
    })
  }
  const total = (theta: Value): Value => sum(spec.logPartition(unflatten(theta)))
  return {
    family: spec.family,
    natural: spec.natural,
    partRanks: ranks,
    sufficientStatistics: (x) => spec.statistics(v(x)),
    logPartition: (eta = spec.natural) => spec.logPartition(eta),
    logBaseMeasure: (x) => spec.logBaseMeasure(v(x)),
    meanParameters: () => unflatten(grad(total)(flatten(spec.natural)) as Value),
    fisherInformation: () => {
      const theta = flatten(spec.natural)
      const k = theta.shape[theta.shape.length - 1] ?? 0
      const gradient = (t: Value): Value => grad(total)(t) as Value
      const columns: Value[] = []
      for (let j = 0; j < k; j++) {
        const direction = add(zerosLike(theta), oneHot(promoteFloat(j)[0] as Value, k))
        columns.push(jvp(gradient, theta, direction).tangent)
      }
      return stack(columns, -1)
    },
    fromNatural: (eta) => spec.fromNatural(eta),
  }
}

const param = (dist: Distribution, name: string): Value => {
  const p = dist.params[name]
  if (p === undefined) throw new WhiteboxError('value', `exponentialFamily: ${dist.name} has no parameter ${name}`)
  return p
}
const zeroMeasure = (x: Value): Value => mul(x, 0)
const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The exponential-family form of a distribution (normal, log-normal, half-normal, exponential, gamma, chi-square,
 * Erlang, inverse gamma, inverse Gaussian, beta, Weibull with fixed shape, Pareto with fixed scale, Laplace with fixed
 * location, Bernoulli, binomial with fixed trials, Poisson, geometric, negative binomial with fixed successes,
 * categorical, one-hot categorical, multinomial, Dirichlet, multivariate normal, Wishart and von Mises), the common
 * currency of conjugacy, natural gradients and analytic divergences.
 *
 * @param dist - The distribution.
 * @returns Its natural parameters, sufficient statistics, log partition, base measure, mean parameters and Fisher
 *   information.
 * @throws WhiteboxError with code `not-implemented` for distributions that are not (or not here) exponential families.
 * @concept exponential-families
 * @example
 * ```ts
 * import { exponentialFamily, poisson } from 'whitebox-engine/dists'
 * import { toFlat } from 'whitebox-engine/tensor'
 * // For the Poisson, A(eta) = exp(eta): the Fisher information is exp(eta) = the rate.
 * toFlat(exponentialFamily(poisson(3)).fisherInformation()).map((v) => Math.round(v * 1e12) / 1e12) // => [3]
 * ```
 */
export function exponentialFamily(dist: Distribution): ExponentialFamilyForm {
  switch (dist.name) {
    case 'normal': {
      const mu = param(dist, 'loc')
      const s2 = square(param(dist, 'scale'))
      return defineForm({
        family: 'normal',
        natural: [div(mu, s2), div(-0.5, s2)],
        statistics: (x) => [x, square(x)],
        logPartition: (eta) =>
          sub(neg(div(square(part(eta, 0)), mul(part(eta, 1), 4))), mul(log(mul(part(eta, 1), -2)), 0.5)),
        logBaseMeasure: (x) => add(mul(x, 0), -0.5 * LOG_2PI),
        fromNatural: (eta) => {
          const variance = div(-0.5, part(eta, 1))
          return normal(mul(part(eta, 0), variance), sqrt(variance))
        },
      })
    }
    case 'logNormal': {
      const mu = param(dist, 'mu')
      const s2 = square(param(dist, 'sigma'))
      return defineForm({
        family: 'logNormal',
        natural: [div(mu, s2), div(-0.5, s2)],
        statistics: (x) => [log(x), square(log(x))],
        logPartition: (eta) =>
          sub(neg(div(square(part(eta, 0)), mul(part(eta, 1), 4))), mul(log(mul(part(eta, 1), -2)), 0.5)),
        logBaseMeasure: (x) => sub(-0.5 * LOG_2PI, log(x)),
        fromNatural: (eta) => {
          const variance = div(-0.5, part(eta, 1))
          return logNormal(mul(part(eta, 0), variance), sqrt(variance))
        },
      })
    }
    case 'halfNormal':
      return defineForm({
        family: 'halfNormal',
        natural: [div(-0.5, square(param(dist, 'scale')))],
        statistics: (x) => [square(x)],
        // log of int_0^inf exp(eta x^2) dx = sqrt(pi / -eta) / 2.
        logPartition: (eta) => sub(mul(log(div(Math.PI, neg(part(eta, 0)))), 0.5), Math.LN2),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => halfNormal(sqrt(div(-0.5, part(eta, 0)))),
      })
    case 'exponential':
      return defineForm({
        family: 'exponential',
        natural: [neg(param(dist, 'rate'))],
        statistics: (x) => [x],
        logPartition: (eta) => neg(log(neg(part(eta, 0)))),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => exponential(neg(part(eta, 0))),
      })
    case 'gamma':
    case 'chi2':
    case 'erlang': {
      const alpha =
        dist.name === 'gamma'
          ? param(dist, 'concentration')
          : dist.name === 'chi2'
            ? mul(param(dist, 'df'), 0.5)
            : param(dist, 'k')
      const rate =
        dist.name === 'gamma'
          ? param(dist, 'rate')
          : dist.name === 'chi2'
            ? (promoteFloat(0.5)[0] as Value)
            : param(dist, 'rate')
      return defineForm({
        family: 'gamma',
        natural: [sub(alpha, 1), neg(rate)],
        statistics: (x) => [log(x), x],
        logPartition: (eta) => sub(lgamma(add(part(eta, 0), 1)), mul(add(part(eta, 0), 1), log(neg(part(eta, 1))))),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => gamma(add(part(eta, 0), 1), neg(part(eta, 1))),
      })
    }
    case 'inverseGamma':
      return defineForm({
        family: 'inverseGamma',
        natural: [sub(neg(param(dist, 'concentration')), 1), neg(param(dist, 'scale'))],
        statistics: (x) => [log(x), div(1, x)],
        logPartition: (eta) => {
          const a = sub(neg(part(eta, 0)), 1)
          return sub(lgamma(a), mul(a, log(neg(part(eta, 1)))))
        },
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => inverseGamma(sub(neg(part(eta, 0)), 1), neg(part(eta, 1))),
      })
    case 'inverseGaussian': {
      const mu = param(dist, 'mean')
      const lambda = param(dist, 'shape')
      return defineForm({
        family: 'inverseGaussian',
        natural: [div(neg(lambda), mul(square(mu), 2)), mul(lambda, -0.5)],
        statistics: (x) => [x, div(1, x)],
        logPartition: (eta) =>
          sub(mul(sqrt(mul(part(eta, 0), part(eta, 1))), -2), mul(log(mul(part(eta, 1), -2)), 0.5)),
        logBaseMeasure: (x) => sub(-0.5 * LOG_2PI, mul(log(x), 1.5)),
        fromNatural: (eta) => {
          const lam = mul(part(eta, 1), -2)
          return inverseGaussian(sqrt(div(part(eta, 1), part(eta, 0))), lam)
        },
      })
    }
    case 'beta':
      return defineForm({
        family: 'beta',
        natural: [sub(param(dist, 'a'), 1), sub(param(dist, 'b'), 1)],
        statistics: (x) => [log(x), log(sub(1, x))],
        logPartition: (eta) => lbeta(add(part(eta, 0), 1), add(part(eta, 1), 1)),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => beta(add(part(eta, 0), 1), add(part(eta, 1), 1)),
      })
    case 'weibull': {
      const k = param(dist, 'concentration')
      return defineForm({
        family: 'weibull',
        natural: [neg(pow(param(dist, 'scale'), neg(k)))],
        statistics: (x) => [pow(x, k)],
        logPartition: (eta) => neg(log(neg(part(eta, 0)))),
        logBaseMeasure: (x) => add(log(k), mul(sub(k, 1), log(x))),
        fromNatural: (eta) => weibull(k, pow(neg(part(eta, 0)), div(-1, k))),
      })
    }
    case 'pareto': {
      const xm = param(dist, 'scale')
      return defineForm({
        family: 'pareto',
        natural: [sub(neg(param(dist, 'b')), 1)],
        statistics: (x) => [log(x)],
        logPartition: (eta) => {
          const b = sub(neg(part(eta, 0)), 1)
          return sub(neg(log(b)), mul(b, log(xm)))
        },
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => pareto(sub(neg(part(eta, 0)), 1), xm),
      })
    }
    case 'laplace': {
      const mu = param(dist, 'loc')
      return defineForm({
        family: 'laplace',
        natural: [div(-1, param(dist, 'scale'))],
        statistics: (x) => [abs(sub(x, mu))],
        logPartition: (eta) => log(div(-2, part(eta, 0))),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => laplace(mu, div(-1, part(eta, 0))),
      })
    }
    case 'bernoulli':
      return defineForm({
        family: 'bernoulli',
        natural: [param(dist, 'logits')],
        statistics: (x) => [x],
        logPartition: (eta) => softplus(part(eta, 0)),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => bernoulli({ logits: part(eta, 0) }),
      })
    case 'binomial': {
      const n = param(dist, 'n')
      const p = param(dist, 'probs')
      return defineForm({
        family: 'binomial',
        natural: [sub(log(p), log(sub(1, p)))],
        statistics: (x) => [x],
        logPartition: (eta) => mul(n, softplus(part(eta, 0))),
        logBaseMeasure: (x) => lbinom(n, x),
        fromNatural: (eta) => binomial(n, sigmoid(part(eta, 0))),
      })
    }
    case 'poisson':
      return defineForm({
        family: 'poisson',
        natural: [log(param(dist, 'rate'))],
        statistics: (x) => [x],
        logPartition: (eta) => exp(part(eta, 0)),
        logBaseMeasure: (x) => neg(lgamma(add(x, 1))),
        fromNatural: (eta) => poisson(exp(part(eta, 0))),
      })
    case 'geometric':
      return defineForm({
        family: 'geometric',
        natural: [log(sub(1, param(dist, 'probs')))],
        statistics: (x) => [x],
        // log sum_{x >= 1} e^{eta x} = eta - log(1 - e^eta).
        logPartition: (eta) => sub(part(eta, 0), log1mexp(part(eta, 0))),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) => geometric(sub(1, exp(part(eta, 0)))),
      })
    case 'negativeBinomial': {
      const r = param(dist, 'n')
      return defineForm({
        family: 'negativeBinomial',
        natural: [log(sub(1, param(dist, 'probs')))],
        statistics: (x) => [x],
        logPartition: (eta) => mul(neg(r), log1mexp(part(eta, 0))),
        logBaseMeasure: (x) => sub(sub(lgamma(add(x, r)), lgamma(add(x, 1))), lgamma(r)),
        fromNatural: (eta) => negativeBinomial(r, sub(1, exp(part(eta, 0)))),
      })
    }
    case 'categorical':
    case 'oneHotCategorical': {
      const logits = param(dist, 'logits')
      const k = logits.shape[logits.shape.length - 1] ?? 0
      const onehot = dist.name === 'oneHotCategorical'
      return defineForm({
        family: dist.name,
        natural: [logits],
        partRanks: [1],
        statistics: (x) => [onehot ? x : oneHot(cast(x, 'int32'), k)],
        logPartition: (eta) => logsumexp(part(eta, 0), -1),
        logBaseMeasure: (x) => (onehot ? sum(mul(x, 0), -1) : mul(x, 0)),
        fromNatural: (eta) =>
          onehot ? oneHotCategorical({ logits: part(eta, 0) }) : categorical({ logits: part(eta, 0) }),
      })
    }
    case 'multinomial': {
      const p = param(dist, 'probs')
      const n = dist.support.kind === 'counts' ? dist.support.total : 0
      return defineForm({
        family: 'multinomial',
        natural: [log(p)],
        partRanks: [1],
        statistics: (x) => [x],
        logPartition: (eta) => mul(logsumexp(part(eta, 0), -1), n),
        logBaseMeasure: (x) => sub(lgamma(n + 1), sum(lgamma(add(x, 1)), -1)),
        fromNatural: (eta) => multinomial(n, softmax(part(eta, 0), -1)),
      })
    }
    case 'dirichlet':
      return defineForm({
        family: 'dirichlet',
        natural: [sub(param(dist, 'concentration'), 1)],
        partRanks: [1],
        statistics: (x) => [log(x)],
        logPartition: (eta) => {
          const a = add(part(eta, 0), 1)
          return sub(sum(lgamma(a), -1), lgamma(sum(a, -1)))
        },
        logBaseMeasure: (x) => sum(mul(x, 0), -1),
        fromNatural: (eta) => dirichlet(add(part(eta, 0), 1)),
      })
    case 'multivariateNormal': {
      const mu = param(dist, 'loc')
      const l = param(dist, 'scaleTril')
      const d = mu.shape[mu.shape.length - 1] ?? 0
      const precision = inv(matmul(l, swapaxes(l, -1, -2)))
      return defineForm({
        family: 'multivariateNormal',
        natural: [squeeze(matmul(precision, expandDims(mu, -1)), -1), mul(precision, -0.5)],
        partRanks: [1, 2],
        statistics: (x) => [x, mul(expandDims(x, -1), expandDims(x, -2))],
        // A = -eta1^T eta2^{-1} eta1 / 4 - log det(-2 eta2) / 2.
        logPartition: (eta) => {
          const p2 = mul(part(eta, 1), -2)
          const lp = cholesky(p2)
          const z = squeeze(solveTriangular(lp, expandDims(part(eta, 0), -1), { lower: true }), -1)
          return sub(mul(sum(square(z), -1), 0.5), sum(log(diagonal(lp, 0, -2, -1)), -1))
        },
        logBaseMeasure: (x) => add(sum(mul(x, 0), -1), -0.5 * d * LOG_2PI),
        fromNatural: (eta) => {
          const p2 = mul(part(eta, 1), -2)
          const cov = inv(p2)
          return multivariateNormal(squeeze(matmul(cov, expandDims(part(eta, 0), -1)), -1), { covariance: cov })
        },
      })
    }
    case 'wishart': {
      const nu = param(dist, 'df')
      const l = param(dist, 'scaleTril')
      const p = l.shape[l.shape.length - 1] ?? 0
      const sigmaInv = inv(matmul(l, swapaxes(l, -1, -2)))
      return defineForm({
        family: 'wishart',
        natural: [mul(sigmaInv, -0.5), mul(sub(nu, p + 1), 0.5)],
        partRanks: [2, 0],
        statistics: (x) => {
          const lx = cholesky(x)
          return [x, mul(sum(log(diagonal(lx, 0, -2, -1)), -1), 2)]
        },
        logPartition: (eta) => {
          const halfNu = add(part(eta, 1), (p + 1) / 2)
          const m = cholesky(mul(part(eta, 0), -2))
          // log |Sigma| = -log |-2 eta1|.
          const logDetSigma = mul(sum(log(diagonal(m, 0, -2, -1)), -1), -2)
          return add(add(mul(halfNu, logDetSigma), mul(halfNu, p * Math.LN2)), multigammaln(halfNu, p))
        },
        logBaseMeasure: (x) => mul(sum(sum(x, -1), -1), 0),
        fromNatural: (eta) => wishart(add(mul(part(eta, 1), 2), p + 1), { precision: mul(part(eta, 0), -2) }),
      })
    }
    case 'vonMises': {
      const mu = param(dist, 'loc')
      const kappa = param(dist, 'concentration')
      return defineForm({
        family: 'vonMises',
        natural: [mul(kappa, cos(mu)), mul(kappa, sin(mu))],
        statistics: (x) => [cos(x), sin(x)],
        logPartition: (eta) => add(logIv(0, sqrt(add(square(part(eta, 0)), square(part(eta, 1))))), LOG_2PI),
        logBaseMeasure: zeroMeasure,
        fromNatural: (eta) =>
          vonMises(arctan2(part(eta, 1), part(eta, 0)), sqrt(add(square(part(eta, 0)), square(part(eta, 1))))),
      })
    }
    default:
      throw new WhiteboxError('not-implemented', `exponentialFamily: no exponential-family form for ${dist.name}`)
  }
}
