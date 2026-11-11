/**
 * The generic numeric likelihood of a Gaussian linear predictor: a registry of log densities `log p(y | u)` of an outcome `y`
 * given the predictor `u` (the inverse link already inside), and the operators built from it. One implementation serves
 * many likelihoods: a rule is a row of the table (`poisson-log`, `negative-binomial-log`, `weibull-log`, `log-normal`,
 * `binomial-logit`, `binomial-probit`, `ordinal-probit`, `asymmetric-laplace`, `student-t`, `laplace`), and the operator
 * `likelihood-site` (a scalar predictor) or `likelihood-row` (one row of a regression on a weight vector) reads which one
 * from its parameters. EP matches the moments of the Gaussian cavity of the predictor times the density by the panelled
 * quadrature of `_tilted.ts`, laid where the density lives (its entry says where) and zoomed onto where the mass is found, so
 * a cavity much wider than the likelihood, as in the first sweeps of a run, is integrated as accurately as a narrow one.
 * The Student-t and Laplace sites of `robust.ts` are two entries of the table under their own names.
 */

import { logAddExp, OmaiError } from '../../base/index.ts'
import { lgammaOf, logPhi } from '../_scalar.ts'
import { tiltedByQuadrature, type WindowHint } from '../_tilted.ts'
import type { FactorOperator, OperatorContext, OperatorResult, Params } from '../types.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, numberParam, vectorParam } from './args.ts'
import { glmRow } from './rows.ts'

/**
 * One likelihood of the registry: the log density of an outcome given a Gaussian linear predictor `u`, with the parameters it
 * needs (known numbers or vectors of the factor) and where, as a function of `u`, it has its mass.
 */
export interface Likelihood {
  readonly name: string
  readonly description: string
  /** The numeric parameters the density needs besides `y` (each a number, except where the description says a vector). */
  readonly params: readonly string[]
  /** A reason the outcome or the parameters are not valid, or `undefined`. */
  check?(y: number, p: Params): string | undefined
  /** The log density of the outcome as a function of the predictor, with the constants of `y` and the parameters precomputed. */
  density(y: number, p: Params): (u: number) => number
  /** Where the density lives as a function of `u`, when it is narrower than a cavity may be (none: the cavity's own window). */
  locate?(y: number, p: Params): WindowHint | undefined
}

const positive = (p: Params, name: string): string | undefined =>
  numberParam(p, name) > 0 ? undefined : `${name} must be positive`

const countCheck = (y: number): string | undefined =>
  Number.isInteger(y) && y >= 0 ? undefined : 'the outcome must be a non-negative integer'

/** log Phi(b) - log Phi(a) for a < b, accurate in both tails (the upper tail is reflected into the lower one). */
function logDiffPhi(a: number, b: number): number {
  if (a === Number.NEGATIVE_INFINITY) return logPhi(b)
  if (b === Number.POSITIVE_INFINITY) return logPhi(-a)
  if (a > 0) return logDiffPhi(-b, -a)
  const hi = logPhi(b)
  return hi + Math.log1p(-Math.exp(logPhi(a) - hi))
}

/** log of the Student-t density with `df` degrees of freedom, scale `s`, centred at `u`, at the outcome `y`. */
const studentT: Likelihood = {
  name: 'student-t',
  description: 'y | u ~ Student t(df, u, scale): robust regression',
  params: ['scale', 'df'],
  check: (_y, p) => positive(p, 'scale') ?? positive(p, 'df'),
  density: (y, p) => {
    const scale = numberParam(p, 'scale')
    const df = numberParam(p, 'df')
    const c = lgammaOf((df + 1) / 2) - lgammaOf(df / 2) - 0.5 * Math.log(df * Math.PI) - Math.log(scale)
    return (u) => c - ((df + 1) / 2) * Math.log(1 + ((y - u) / scale) ** 2 / df)
  },
  locate: (y, p) => ({ centre: y, scale: numberParam(p, 'scale'), zoom: false }),
}

const laplace: Likelihood = {
  name: 'laplace',
  description: 'y | u ~ Laplace(u, scale): absolute-error regression',
  params: ['scale'],
  check: (_y, p) => positive(p, 'scale'),
  density: (y, p) => {
    const scale = numberParam(p, 'scale')
    return (u) => -Math.log(2 * scale) - Math.abs(y - u) / scale
  },
  locate: (y, p) => ({ centre: y, scale: numberParam(p, 'scale'), kinks: [y], zoom: false }),
}

const asymmetricLaplace: Likelihood = {
  name: 'asymmetric-laplace',
  description: 'y | u ~ asymmetric Laplace(u, scale, tau): quantile regression at level tau',
  params: ['scale', 'tau'],
  check: (_y, p) => {
    const tau = numberParam(p, 'tau')
    return positive(p, 'scale') ?? (tau > 0 && tau < 1 ? undefined : 'tau must be in (0, 1)')
  },
  density: (y, p) => {
    const scale = numberParam(p, 'scale')
    const tau = numberParam(p, 'tau')
    const c = Math.log(tau * (1 - tau)) - Math.log(scale)
    return (u) => {
      const r = (y - u) / scale
      return c - r * (tau - (r < 0 ? 1 : 0))
    }
  },
  locate: (y, p) => ({ centre: y, scale: numberParam(p, 'scale'), kinks: [y], zoom: false }),
}

const poissonLog: Likelihood = {
  name: 'poisson-log',
  description: 'y | u ~ Poisson(exposure exp(u)): a count with a log link',
  params: ['exposure'],
  check: (y, p) => countCheck(y) ?? positive(p, 'exposure'),
  density: (y, p) => {
    const logE = Math.log(numberParam(p, 'exposure', 1))
    const E = numberParam(p, 'exposure', 1)
    const c = y * logE - lgammaOf(y + 1)
    return (u) => c + y * u - E * Math.exp(u)
  },
  locate: (y, p) => ({
    centre: Math.log((y + 0.5) / numberParam(p, 'exposure', 1)),
    scale: 1 / Math.sqrt(y + 1),
    reach: 40,
  }),
}

const negativeBinomialLog: Likelihood = {
  name: 'negative-binomial-log',
  description:
    'y | u ~ NegBin(mean exposure exp(u), variance mean + dispersion mean^2): overdispersed counts with a log link',
  params: ['dispersion', 'exposure'],
  check: (y, p) => countCheck(y) ?? positive(p, 'dispersion') ?? positive(p, 'exposure'),
  density: (y, p) => {
    const r = 1 / numberParam(p, 'dispersion')
    const logE = Math.log(numberParam(p, 'exposure', 1))
    const logR = Math.log(r)
    const c = lgammaOf(y + r) - lgammaOf(r) - lgammaOf(y + 1) + r * logR
    // log p = c - r log(r + mu) + y (log mu - log(r + mu)) with mu = exposure exp(u).
    return (u) => {
      const logMu = logE + u
      const logDen = logAddExp(logR, logMu)
      return c - (r + y) * logDen + y * logMu
    }
  },
  locate: (y, p) => ({
    centre: Math.log((y + 0.5) / numberParam(p, 'exposure', 1)),
    scale: 1 / Math.sqrt(y + 1),
    reach: 40,
  }),
}

const weibullLog: Likelihood = {
  name: 'weibull-log',
  description:
    'y | u ~ Weibull(shape, scale exp(u)): survival or reliability time with a log link; censored = 1 makes y a right-censoring time',
  params: ['shape', 'censored'],
  check: (y, p) => (y > 0 ? undefined : 'the time must be positive') ?? positive(p, 'shape'),
  density: (y, p) => {
    const k = numberParam(p, 'shape')
    const censored = numberParam(p, 'censored', 0) !== 0
    const logY = Math.log(y)
    const yk = y ** k
    const c = censored ? 0 : Math.log(k) + (k - 1) * logY
    return (u) => c - (censored ? 0 : k * u) - yk * Math.exp(-k * u)
  },
  locate: (y, p) =>
    numberParam(p, 'censored', 0) !== 0
      ? undefined
      : { centre: Math.log(y), scale: 1 / numberParam(p, 'shape'), reach: 40 },
}

const logNormal: Likelihood = {
  name: 'log-normal',
  description:
    'y | u ~ LogNormal(u, sigma): a positive time whose log is Gaussian around the predictor; censored = 1 makes y a right-censoring time',
  params: ['sigma', 'censored'],
  check: (y, p) => (y > 0 ? undefined : 'the time must be positive') ?? positive(p, 'sigma'),
  density: (y, p) => {
    const sigma = numberParam(p, 'sigma')
    const censored = numberParam(p, 'censored', 0) !== 0
    const logY = Math.log(y)
    const c = -logY - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI)
    return (u) => (censored ? logPhi((u - logY) / sigma) : c - ((logY - u) / sigma) ** 2 / 2)
  },
  locate: (y, p) =>
    numberParam(p, 'censored', 0) !== 0
      ? undefined
      : { centre: Math.log(y), scale: numberParam(p, 'sigma'), reach: 30 },
}

/** log of the binomial coefficient. */
const logChoose = (n: number, y: number): number => lgammaOf(n + 1) - lgammaOf(y + 1) - lgammaOf(n - y + 1)

const binomialCheck = (y: number, p: Params): string | undefined => {
  const n = numberParam(p, 'n')
  return Number.isInteger(n) && n > 0 && Number.isInteger(y) && y >= 0 && y <= n
    ? undefined
    : 'the count must be an integer between 0 and the number of trials n'
}

const logSigmoid = (x: number): number => -logAddExp(0, -x)

const binomialLogit: Likelihood = {
  name: 'binomial-logit',
  description: 'y | u ~ Binomial(n, sigmoid(u)): successes in n trials with a logistic link',
  params: ['n'],
  check: binomialCheck,
  density: (y, p) => {
    const n = numberParam(p, 'n')
    const c = logChoose(n, y)
    return (u) => c + y * logSigmoid(u) + (n - y) * logSigmoid(-u)
  },
  locate: (y, p) => {
    const n = numberParam(p, 'n')
    return {
      centre: Math.log((y + 0.5) / (n - y + 0.5)),
      scale: Math.sqrt(1 / (y + 0.5) + 1 / (n - y + 0.5)),
      reach: 100,
    }
  },
}

const binomialProbit: Likelihood = {
  name: 'binomial-probit',
  description: 'y | u ~ Binomial(n, Phi(u)): successes in n trials with a probit link',
  params: ['n'],
  check: binomialCheck,
  density: (y, p) => {
    const n = numberParam(p, 'n')
    const c = logChoose(n, y)
    return (u) => c + y * logPhi(u) + (n - y) * logPhi(-u)
  },
  locate: (y, p) => {
    const n = numberParam(p, 'n')
    return { centre: 0.6 * Math.log((y + 0.5) / (n - y + 0.5)), scale: 1, reach: 100 }
  },
}

const ordinalProbit: Likelihood = {
  name: 'ordinal-probit',
  description:
    'y | u ~ ordinal probit with fixed cut points: P(y = k) = Phi(c_(k+1) - u) - Phi(c_k - u), the outcome a category 0 to K-1',
  params: ['cuts'],
  check: (y, p) => {
    const cuts = vectorParam(p, 'cuts')
    if (!(Number.isInteger(y) && y >= 0 && y <= cuts.length))
      return 'the category must be an integer from 0 to the number of cuts'
    return cuts.every((c, i) => i === 0 || c > (cuts[i - 1] as number)) ? undefined : 'the cut points must increase'
  },
  density: (y, p) => {
    const cuts = vectorParam(p, 'cuts')
    const lower = y === 0 ? Number.NEGATIVE_INFINITY : (cuts[y - 1] as number)
    const upper = y === cuts.length ? Number.POSITIVE_INFINITY : (cuts[y] as number)
    return (u) => logDiffPhi(lower - u, upper - u)
  },
  locate: (y, p) => {
    const cuts = vectorParam(p, 'cuts')
    const lower = y === 0 ? (cuts[0] as number) - 1 : (cuts[y - 1] as number)
    const upper = y === cuts.length ? (cuts[cuts.length - 1] as number) + 1 : (cuts[y] as number)
    return { centre: (lower + upper) / 2, scale: 1 + (upper - lower) / 2, reach: 100 }
  },
}

/**
 * The registry of numeric likelihoods by name: `poisson-log`, `negative-binomial-log`, `weibull-log`, `log-normal`,
 * `binomial-logit`, `binomial-probit`, `ordinal-probit`, `asymmetric-laplace`, `student-t` and `laplace`. A new object each
 * call, so a caller may add a row of its own and build operators from it with {@link likelihoodOperators}.
 *
 * @returns The likelihoods, keyed by name.
 * @concept factor-operators
 * @index table The numeric likelihoods of a Gaussian predictor (count, survival, ordinal, quantile, robust) by name.
 * @example
 * ```ts
 * import { likelihoods } from 'omai-engine/message-passing'
 * // Poisson, exposure 1, outcome 3: the log density at u = log 3 is log(3^3 e^-3 / 3!).
 * likelihoods()['poisson-log']?.density(3, { exposure: 1 })(Math.log(3)).toFixed(4) // => '-1.4959'
 * ```
 */
export function likelihoods(): Record<string, Likelihood> {
  const all: Likelihood[] = [
    poissonLog,
    negativeBinomialLog,
    weibullLog,
    logNormal,
    binomialLogit,
    binomialProbit,
    ordinalProbit,
    asymmetricLaplace,
    studentT,
    laplace,
  ]
  return Object.fromEntries(all.map((l) => [l.name, l]))
}

/** The likelihood a factor asks for, and the outcome, checked. */
function resolve(
  table: Readonly<Record<string, Likelihood>>,
  fixed: string | undefined,
  params: Params,
): { readonly likelihood: Likelihood; readonly y: number } {
  const key = fixed ?? params['likelihood']
  const likelihood = typeof key === 'string' ? table[key] : undefined
  if (likelihood === undefined)
    throw new OmaiError(
      'value',
      `likelihood-site: unknown likelihood ${String(key)} (the registry has ${Object.keys(table).join(', ')})`,
    )
  const y = numberParam(params, 'y')
  const why = likelihood.check?.(y, params)
  if (why !== undefined) throw new OmaiError('value', `${likelihood.name}: ${why}`)
  return { likelihood, y }
}

/** The tilted mass and moments of a site: the cavity of the predictor against the (powered) density. */
function tiltedSite(
  table: Readonly<Record<string, Likelihood>>,
  fixed: string | undefined,
  c: OperatorContext,
): { logMass: number; mean: number; variance: number } {
  const { likelihood, y } = resolve(table, fixed, c.params)
  const log = likelihood.density(y, c.params)
  const hint = likelihood.locate?.(y, c.params)
  const power = c.power
  return tiltedByQuadrature(gaussianOf(c.args[0] as never), power === 1 ? log : (u) => power * log(u), hint)
}

/**
 * A numeric likelihood site from a table of likelihoods: the factor `p(y | u)` of one Gaussian predictor `u`, whose EP
 * message is the moment match of the cavity times the density by quadrature. With `fixed` the operator is that likelihood
 * (the parameters are its own: `y` and what the entry lists); without it the parameter `likelihood` names the entry. It
 * supports a power (fractional and power EP).
 *
 * @param name - The operator's name.
 * @param table - The likelihoods to choose from (default {@link likelihoods}).
 * @param fixed - The entry the operator always is, or `undefined` for one chosen by a parameter.
 * @returns The operator.
 * @concept factor-operators
 * @index function A numeric EP likelihood site from a registry of log densities.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, likelihoodSite, messageArg } from 'omai-engine/message-passing'
 * const site = likelihoodSite('count-site')
 * const args = [messageArg(gaussian, gaussianEta(0, 1))]
 * const m = site.ep?.({ mode: 'ep', to: 0, args, params: { likelihood: 'poisson-log', y: 4, exposure: 1 }, power: 1 })
 * m?.message.map((v) => v.toFixed(3)) // => ['3.690', '2.765']
 * ```
 */
export function likelihoodSite(
  name: string,
  table: Readonly<Record<string, Likelihood>> = likelihoods(),
  fixed?: string,
): FactorOperator {
  const tilted = (c: OperatorContext): { logMass: number; mean: number; variance: number } =>
    tiltedSite(table, fixed, c)
  return {
    name,
    description:
      fixed === undefined ? 'y | u ~ a numeric likelihood of the registry' : (table[fixed]?.description ?? name),
    ports: [{ name: 'u', accepts: ['gaussian'] }],
    support: { ep: 'numeric' },
    readsTarget: true,
    power: true,
    check: (params) => {
      try {
        resolve(table, fixed, params)
        return undefined
      } catch (e) {
        return e instanceof Error ? e.message : String(e)
      }
    },
    ep: (c): OperatorResult => {
      const t = tilted(c)
      const site = gaussianSiteMessage(etaOf(c.args[0] as never), t.mean, t.variance)
      return { message: c.power === 1 ? site : site.map((v) => v / c.power) }
    },
    logEvidence: (c) => (allProper(c.args) ? massOf(c.args[0] as never) + tilted(c).logMass : Number.NaN),
  }
}

/**
 * The operators of a table of likelihoods: `likelihood-site` and its regression row `likelihood-row` (a row `y | a . w` on
 * a Gaussian weight vector, the rank-one lift of the site by `glmRow`), each choosing its likelihood by the parameter
 * `likelihood`.
 *
 * @param table - The likelihoods (default {@link likelihoods}).
 * @returns The site and the row operators.
 * @concept factor-operators
 * @index function The likelihood-site and likelihood-row operators of a registry of numeric likelihoods.
 * @example
 * ```ts
 * import { likelihoodOperators } from 'omai-engine/message-passing'
 * likelihoodOperators().map((o) => o.name) // => ['likelihood-site', 'likelihood-row']
 * ```
 */
export function likelihoodOperators(table: Readonly<Record<string, Likelihood>> = likelihoods()): FactorOperator[] {
  const site = likelihoodSite('likelihood-site', table)
  return [site, glmRow('likelihood-row', site, 'y | a . w ~ a numeric likelihood of the registry')]
}

/** The generic likelihood site over the library's registry. */
export const likelihoodSiteOperator: FactorOperator = likelihoodSite('likelihood-site')

/** The generic likelihood row (a regression row on a weight vector) over the library's registry. */
export const likelihoodRowOperator: FactorOperator = glmRow(
  'likelihood-row',
  likelihoodSiteOperator,
  'y | a . w ~ a numeric likelihood of the registry',
)
