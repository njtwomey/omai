/**
 * The distribution constructors of the model layer: `normal(0, 5)`, `normal(theta, sigma)` and the rest, taking
 * numbers or `Var`s and returning a `DistExpr`, a distribution as data (a family name and parameter expressions with
 * their named dimensions). They read like the constructors of `omai-engine/dists` and are the same families (the table
 * in `families.ts` maps each name to the library constructor); the difference is that their parameters may be other
 * nodes of the model, and that the result serialises.
 *
 * @index function normal, gamma, bernoulli and the other families as model expressions.
 */

import { OmaiError } from '../../../base/index.ts'
import type { DistSpec, ExprSpec } from '../spec.ts'
import { type Operand, toVar } from '../var.ts'
import { distributionFamilies, type FamilyDef } from './families.ts'

const FAMILIES: Readonly<Record<string, FamilyDef>> = distributionFamilies()

/** A distribution in a model: its spec (family and parameter expressions), and the dimensions its parameters carry. */
export interface DistExpr {
  readonly spec: DistSpec
  /** The joint named dimensions of the parameters, in order of first appearance. */
  readonly dims: readonly string[]
  /** The event rank of a draw. */
  readonly eventRank: number
}

/**
 * A distribution from a family of the table and its parameters. The convenience constructors (`normal`, ...) call this.
 *
 * @param name - The family (a key of `distributionFamilies()`).
 * @param args - Parameters by name, as operands; defaults fill the ones left out.
 * @returns The distribution expression.
 * @throws OmaiError with code `value` for an unknown family or parameter, a missing parameter, or a group of
 *   alternatives (`probs`, `logits`) with none or several given.
 * @concept model-spec
 * @example
 * ```ts
 * import { family } from 'omai-engine/probprog/model/dists'
 * family('normal', { loc: 0 }).spec.params['scale'] // => { kind: 'const', value: 1 }
 * ```
 */
export function family(name: string, args: Readonly<Record<string, Operand | undefined>>): DistExpr {
  const def = FAMILIES[name]
  if (def === undefined)
    throw new OmaiError('value', `unknown distribution family '${name}' (known: ${Object.keys(FAMILIES).join(', ')})`)
  for (const k of Object.keys(args))
    if (!def.params.includes(k))
      throw new OmaiError('value', `${name}: unknown parameter '${k}' (takes ${def.params.join(', ')})`)
  const given = def.params.filter((p) => args[p] !== undefined)
  for (const group of def.oneOf) {
    const n = group.filter((p) => given.includes(p)).length
    if (n !== 1) throw new OmaiError('value', `${name}: give exactly one of ${group.join(', ')}`)
  }
  const params: Record<string, ExprSpec> = {}
  const dims: string[] = []
  for (const p of def.params) {
    const a = args[p]
    const d = def.defaults[p]
    if (a === undefined && d === undefined) {
      if (def.oneOf.some((g) => g.includes(p))) continue
      throw new OmaiError('value', `${name}: the parameter '${p}' is missing`)
    }
    const v = toVar(a === undefined ? (d as number) : a)
    params[p] = v.expr
    for (const dim of v.dims) if (!dims.includes(dim)) dims.push(dim)
  }
  return { spec: { family: name, params }, dims, eventRank: def.eventRank }
}

/** A normal distribution: mean `loc`, standard deviation `scale`. */
export function normal(loc: Operand = 0, scale: Operand = 1): DistExpr {
  return family('normal', { loc, scale })
}

/** A half-normal distribution on the positive reals. */
export function halfNormal(scale: Operand = 1): DistExpr {
  return family('halfNormal', { scale })
}

/** A log-normal distribution: `log x` is normal with mean `mu` and standard deviation `sigma`. */
export function logNormal(mu: Operand = 0, sigma: Operand = 1): DistExpr {
  return family('logNormal', { mu, sigma })
}

/** A normal distribution truncated to `[low, high]` and renormalised. */
export function truncatedNormal(loc: Operand, scale: Operand, low: Operand, high: Operand): DistExpr {
  return family('truncatedNormal', { loc, scale, low, high })
}

/** A Student t distribution with `df` degrees of freedom. */
export function studentT(df: Operand, loc: Operand = 0, scale: Operand = 1): DistExpr {
  return family('studentT', { df, loc, scale })
}

/** A Cauchy distribution. */
export function cauchy(loc: Operand = 0, scale: Operand = 1): DistExpr {
  return family('cauchy', { loc, scale })
}

/** A half-Cauchy distribution on the positive reals (the usual scale prior). */
export function halfCauchy(scale: Operand = 1): DistExpr {
  return family('halfCauchy', { scale })
}

/** A Laplace distribution. */
export function laplace(loc: Operand = 0, scale: Operand = 1): DistExpr {
  return family('laplace', { loc, scale })
}

/** An exponential distribution with rate `rate`. */
export function exponential(rate: Operand = 1): DistExpr {
  return family('exponential', { rate })
}

/** A gamma distribution with shape `concentration` and rate `rate`. */
export function gamma(concentration: Operand, rate: Operand = 1): DistExpr {
  return family('gamma', { concentration, rate })
}

/** An inverse-gamma distribution with shape `concentration` and scale `scale`. */
export function inverseGamma(concentration: Operand, scale: Operand = 1): DistExpr {
  return family('inverseGamma', { concentration, scale })
}

/** A Weibull distribution. */
export function weibull(concentration: Operand, scale: Operand = 1): DistExpr {
  return family('weibull', { concentration, scale })
}

/** A beta distribution on the unit interval. */
export function beta(a: Operand, b: Operand): DistExpr {
  return family('beta', { a, b })
}

/** A uniform distribution on `[low, high]`. */
export function uniform(low: Operand = 0, high: Operand = 1): DistExpr {
  return family('uniform', { low, high })
}

/** A Bernoulli distribution from a probability or logit. */
export function bernoulli(params: { readonly probs: Operand } | { readonly logits: Operand }): DistExpr {
  return family('bernoulli', params)
}

/** A binomial distribution of `n` trials. */
export function binomial(n: Operand, probs: Operand): DistExpr {
  return family('binomial', { n, probs })
}

/** A geometric distribution (trials up to the first success). */
export function geometric(probs: Operand): DistExpr {
  return family('geometric', { probs })
}

/**
 * The asymmetric Laplace distribution of quantile regression: `loc` is the `tau` quantile (`tau` in (0, 1), 1/2 the median) and
 * `scale` the scale of the check-loss. `y ~ asymmetricLaplace(X w, scale, tau)` is Bayesian quantile regression.
 */
export function asymmetricLaplace(loc: Operand = 0, scale: Operand = 1, tau: Operand = 0.5): DistExpr {
  return family('asymmetricLaplace', { loc, scale, tau })
}

/**
 * An ordinal outcome 0 to K-1 by the cumulative probit: `P(y <= k) = Phi(cutpoints_k - predictor)` for the increasing `cutpoints`
 * (K - 1 known values). `y ~ orderedProbit(X w, cuts)` is ordered probit regression.
 */
export function orderedProbit(predictor: Operand, cutpoints: Operand): DistExpr {
  return family('orderedProbit', { predictor, cutpoints })
}

/** An ordinal outcome 0 to K-1 by the cumulative logit (proportional odds): `P(y <= k) = sigmoid(cutpoints_k - predictor)`. */
export function orderedLogistic(predictor: Operand, cutpoints: Operand): DistExpr {
  return family('orderedLogistic', { predictor, cutpoints })
}

/** A negative binomial distribution. */
export function negativeBinomial(n: Operand, probs: Operand): DistExpr {
  return family('negativeBinomial', { n, probs })
}

/** A Poisson distribution with rate `rate`. */
export function poisson(rate: Operand): DistExpr {
  return family('poisson', { rate })
}

/** A Poisson distribution of rate `rate` conditioned on being positive (the positive part of a hurdle model). */
export function zeroTruncatedPoisson(rate: Operand): DistExpr {
  return family('zeroTruncatedPoisson', { rate })
}

/**
 * A hurdle count: zero with probability `zero`, else a zero-truncated Poisson of rate `rate`. With a beta `zero` and a rate
 * `exposure * exp(affine)` message passing reads it as a conjugate Bernoulli on the zero part and a truncated-Poisson site on the positive part.
 */
export function hurdlePoisson(rate: Operand, zero: Operand): DistExpr {
  return family('hurdlePoisson', { rate, zero })
}

/**
 * A categorical distribution over the last axis of `probs` or `logits`. With `over`, the classes are a named dimension of the
 * parameters (the logits `X w` over `row` and `class`, as a multinomial logit classifier writes them): that dimension is the
 * event axis, so the observation has one entry per row, not per row and class.
 */
export function categorical(
  params: { readonly probs: Operand } | { readonly logits: Operand },
  options: { readonly over?: string | { readonly name: string } } = {},
): DistExpr {
  const d = family('categorical', params)
  if (options.over === undefined) return d
  const over = typeof options.over === 'string' ? options.over : options.over.name
  return { ...d, dims: d.dims.filter((x) => x !== over), spec: { ...d.spec, eventDims: [over] } }
}

/**
 * A categorical outcome by the multinomial probit: the class whose utility `score_k + e_k`, with standard normal `e_k`, is the
 * largest. The `scores` carry the classes as the dimension `over`, as the logits of `categorical` do.
 */
export function categoricalProbit(
  scores: Operand,
  options: { readonly over: string | { readonly name: string } },
): DistExpr {
  const d = family('categoricalProbit', { scores })
  const over = typeof options.over === 'string' ? options.over : options.over.name
  return { ...d, dims: d.dims.filter((x) => x !== over), spec: { ...d.spec, eventDims: [over] } }
}

/** A Dirichlet distribution over the simplex; the last axis of `concentration` is the category axis. */
export function dirichlet(concentration: Operand): DistExpr {
  return family('dirichlet', { concentration })
}

/**
 * The spike-and-slab prior with the indicator summed out: a zero-mean scale mixture, `inclusion` of a normal with the wide
 * `slab` standard deviation and the rest of a normal with the narrow `spike` one. A gradient sampler can use it where the
 * indicator itself (a discrete latent) cannot be enumerated per element, and the posterior probability that a value came
 * from the slab is `inclusion N(x; 0, slab) / (inclusion N(x; 0, slab) + (1 - inclusion) N(x; 0, spike))`.
 */
export function normalSpikeSlab(inclusion: Operand, slab: Operand, spike: Operand): DistExpr {
  return family('normalSpikeSlab', { inclusion, slab, spike })
}

/**
 * A normal with outliers, a mixture of two normals around `loc`: probability `1 - outlier` of one with standard deviation
 * `scale` (the inliers) and `outlier` of one `width` times as wide. The likelihood of a regression that is robust to a
 * fraction of wild observations (the clutter problem).
 */
export function normalOutliers(loc: Operand, scale: Operand, outlier: Operand, width: Operand): DistExpr {
  return family('normalOutliers', { loc, scale, outlier, width })
}

/**
 * A mixture of two normals of one `scale` around the means `low` and `high`, with probability `weight` of the second: the likelihood
 * of an observation of a two-component mixture whose weight is unknown (the switch summed out). With a beta `weight` it is the
 * mixture-weight factor of message passing.
 */
export function normalPair(weight: Operand, low: Operand, high: Operand, scale: Operand): DistExpr {
  return family('normalPair', { weight, low, high, scale })
}

/**
 * A multivariate normal over the last axis, with a `covariance`, `precision` or `scaleTril`. With `over`, the vector's axis is a named
 * dimension of the parameters (the mean `mu` over `coef`, as a population mean of correlated random effects is written): that
 * dimension is the event axis, so a node of this distribution has one vector per index of its other dimensions. `precision` may be a
 * Wishart latent and `covariance` an inverse-Wishart one (a learned covariance matrix).
 */
export function multivariateNormal(
  loc: Operand,
  params: { readonly covariance: Operand } | { readonly precision: Operand } | { readonly scaleTril: Operand },
  options: { readonly over?: string | { readonly name: string } } = {},
): DistExpr {
  const d = family('multivariateNormal', { loc, ...params })
  if (options.over === undefined) return d
  const over = typeof options.over === 'string' ? options.over : options.over.name
  return { ...d, dims: d.dims.filter((x) => x !== over), spec: { ...d.spec, eventDims: [over] } }
}

/**
 * A Wishart distribution over `d x d` symmetric positive-definite matrices with `df` degrees of freedom and scale matrix `scale`: the
 * conjugate prior of the precision matrix of a multivariate normal (`multivariateNormal(loc, { precision: lambda })` with `lambda`
 * this latent). The value is a matrix (event rank 2), mean `df * scale`. Message passing carries it as the Wishart family.
 */
export function wishart(df: Operand, scale: Operand): DistExpr {
  return family('wishart', { df, scale })
}

/**
 * An inverse Wishart distribution over covariance matrices with `df` degrees of freedom and scale matrix `scale` (mean
 * `scale / (df - d - 1)`): the conjugate prior of the covariance of a multivariate normal
 * (`multivariateNormal(loc, { covariance: sigma })` with `sigma` this latent). Message passing carries it as the Wishart family of its inverse.
 */
export function inverseWishart(df: Operand, scale: Operand): DistExpr {
  return family('inverseWishart', { df, scale })
}
