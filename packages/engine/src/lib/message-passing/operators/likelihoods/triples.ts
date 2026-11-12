/**
 * The three-variable numeric sites: a likelihood `p(y | x0, x1, x2)` of an outcome over three random scalars at once, the step beyond
 * `pairs.ts`. A Student-t whose location, scale and degrees of freedom are all random, and an ordinal outcome (a cumulative probit or
 * logit) whose predictor and cut points are random, are rows of one table read by one operator, `likelihood-triple`. As in the pairs a
 * coordinate is a real value or a positive one (carried as a Gaussian on its log, projected by the compiler, or as a gamma variable), and
 * any port may be a known value (a single node of the grid). EP matches the moments of the three variables under their cavities times
 * the density on the zooming grid of `integrateTriple` (`_grid.ts`: panels per axis, a finer grid along the heavy-tailed coordinate,
 * the ordinal rows from tables of pairs of nodes, so a pass costs the pairs and not the triples), or, where a row has a closed form
 * conditional on one more variable, by a one-dimensional integral over it: the ordinal probit with Gaussian cavities is exact given its
 * latent score (`probitIntegrate`), which is also what keeps the order of the cut points. One integral is shared by the messages and the
 * evidence of a sweep.
 */

import { exp, less, log1p, neg, reshape, sub, where } from '../../../tensor/index.ts'
import { errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, gaussianOf, massOf, numberParam } from '../../core/args.ts'
import { defineOperator, messageRule, tiltedRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { namedParams, param } from '../../core/params.ts'
import type { Arg, FactorOperator, Family, OperatorContext, Params, Port, TiltedMoments } from '../../core/types.ts'
import { float64Of } from '../../numerics/_arrays.ts'
import {
  axisOf,
  type GridAxis,
  type GridIndexed,
  type GridTripleResult,
  integrateTriple,
} from '../../numerics/_grid.ts'
import { centredMoments } from '../../numerics/_moments.ts'
import { lgammaOf, logDiffPhi, logDiffSigmoid, logPhi, logSigmoidOf, millsRatio } from '../../numerics/_scalar.ts'
import { tiltGrid } from '../../numerics/_windows.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'
import type { PairPort } from './pairs.ts'

/** One density of the registry of triple likelihoods. */
export interface TripleLikelihood {
  readonly name: string
  readonly description: string
  /** The three coordinates: a real port is a Gaussian variable, a positive one a Gaussian on its log or a gamma variable. */
  readonly ports: readonly [PairPort, PairPort, PairPort]
  /** The numeric parameters besides the outcome `y`. */
  readonly params: readonly string[]
  /** A reason the outcome or the parameters are not valid, or `undefined`. */
  check?(y: number, p: Params): string | undefined
  /** The log density at the coordinates, with the constants of `y` and the parameters precomputed. */
  density(y: number, p: Params): (x0: number, x1: number, x2: number) => number
  /**
   * The same density as a function of the second and third coordinates that returns a function of the first, when what depends on
   * those two alone (a gamma function of a shape, a log scale) is worth computing once per pair of nodes of them. Absent: `density`.
   */
  column?(y: number, p: Params): (x1: number, x2: number) => (x0: number) => number
  /**
   * The integral by another route than the grid, for the cavities it can take (`undefined` for the others, and then the grid is used):
   * the log mass and the moments of each coordinate under the cavities times the density, at power one. A density that has a closed
   * form conditional on one more variable (the ordinal probit given its latent score) is a one-dimensional integral over that variable
   * and exact, not a cube of nodes.
   */
  integrate?(y: number, p: Params, args: readonly Arg[]): GridTripleResult | undefined
  /**
   * The same density by the index of the nodes of the grid, for a density made of functions of pairs of coordinates: the tables of the
   * terms are made once per pass from the nodes of each axis, so the cost of a pass is that of the pairs and not of the triples (the
   * ordinal outcome is the difference of a term of the predictor and the upper cut and one of the predictor and the lower cut). It is
   * the density of `density` at the nodes. Absent: `column` or `density`.
   */
  indexed?(y: number, p: Params): GridIndexed
  /**
   * How many times the panels of the grid each coordinate gets (default one each): the location of a heavy-tailed density has mass far
   * from its core, and needs a finer grid than the scale and the degrees of freedom, which are smooth.
   */
  readonly fineness?: readonly [number, number, number]
  /**
   * Why the density has no variational message for this outcome (an expected log that is infinite under any Gaussian belief of the
   * coordinates), or `undefined` when it has one. A function of the outcome: the ordinal rows have a message for the first and the last
   * category (one cut, a smooth density) and none for the others (two cuts, zero where they are out of order).
   */
  noVmp?(y: number, p: Params): string | undefined
}

/**
 * A Student-t outcome whose location `x0`, scale `exp(x1)` and degrees of freedom `exp(x2)` are all random: the heavy-tailed regression
 * with the noise level and the tail weight learned beside the mean.
 */
const studentTFull: TripleLikelihood = {
  name: 'student-t-full',
  description:
    'y | u, s, nu ~ Student t(exp(nu), u, exp(s)): robust noise whose location, scale and degrees of freedom are all random',
  ports: [
    { name: 'location', positive: false },
    { name: 'scale', positive: true },
    { name: 'df', positive: true },
  ],
  params: [],
  fineness: [3, 1, 1],
  density: (y) => (u, s, t) => {
    const df = Math.exp(t)
    const z = (y - u) * Math.exp(-s)
    return (
      lgammaOf((df + 1) / 2) -
      lgammaOf(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      s -
      ((df + 1) / 2) * Math.log1p((z * z) / df)
    )
  },
  column: (y) => (s, t) => {
    const df = Math.exp(t)
    const c = lgammaOf((df + 1) / 2) - lgammaOf(df / 2) - 0.5 * Math.log(df * Math.PI) - s
    const half = (df + 1) / 2
    const inverse = Math.exp(-s)
    return (u) => {
      const z = (y - u) * inverse
      return c - half * Math.log1p((z * z) / df)
    }
  },
}

/** The category and the number of categories of an ordinal row, checked. */
const ordinalCheck = (y: number, p: Params): string | undefined => {
  const categories = numberParam(p, 'categories')
  if (!(Number.isInteger(categories) && categories >= 2)) return 'categories must be an integer of at least 2'
  return Number.isInteger(y) && y >= 0 && y < categories
    ? undefined
    : 'the category must be an integer from 0 to categories - 1'
}

/** The category of an ordinal row and whether it has a lower cut and an upper one. */
function ordinalSides(y: number, p: Params): { readonly first: boolean; readonly last: boolean } {
  return { first: y === 0, last: y === numberParam(p, 'categories') - 1 }
}

/** `log Phi(sign (cut - u))` at every predictor node `u` (the fast index) and cut node (the slow one). */
function linkTable(
  link: (x: number) => number,
  predictor: Float64Array,
  cuts: Float64Array,
  sign: 1 | -1,
): Float64Array {
  const table = new Float64Array(predictor.length * cuts.length)
  for (let j = 0; j < cuts.length; j++)
    for (let i = 0; i < predictor.length; i++)
      table[j * predictor.length + i] = link(sign * ((cuts[j] as number) - (predictor[i] as number)))
  return table
}

/**
 * The cumulative probit outcome by the index of the nodes: `log (Phi(hi - u) - Phi(lo - u))` from the tables `log Phi(+-(c - u))` of the
 * two cuts, reflected into the lower tail where `lo - u` is positive (as `logDiffPhi` does), so the pass costs four tables of pairs and one
 * logarithm per node.
 */
function probitIndexed(y: number, p: Params): GridIndexed {
  const { first, last } = ordinalSides(y, p)
  return (nodes) => {
    const [u, lo, hi] = nodes as readonly [Float64Array, Float64Array, Float64Array]
    const n0 = u.length
    const upPlus = last ? undefined : linkTable(logPhi, u, hi, 1)
    const upMinus = last ? undefined : linkTable(logPhi, u, hi, -1)
    const loPlus = first ? undefined : linkTable(logPhi, u, lo, 1)
    const loMinus = first ? undefined : linkTable(logPhi, u, lo, -1)
    return (a, b, c) => {
      if (first) return (upPlus as Float64Array)[c * n0 + a] as number
      if (last) return (loMinus as Float64Array)[b * n0 + a] as number
      if (!((lo[b] as number) < (hi[c] as number))) return Number.NEGATIVE_INFINITY
      if ((lo[b] as number) - (u[a] as number) > 0) {
        const m = (loMinus as Float64Array)[b * n0 + a] as number
        return m + Math.log1p(-Math.exp(Math.min(0, ((upMinus as Float64Array)[c * n0 + a] as number) - m)))
      }
      const h = (upPlus as Float64Array)[c * n0 + a] as number
      return h + Math.log1p(-Math.exp(Math.min(0, ((loPlus as Float64Array)[b * n0 + a] as number) - h)))
    }
  }
}

/** The nodes of the grid over the latent score of the ordinal probit's exact route (the zoom finds the mass inside the first window). */
const SCORE_POINTS = 121

/**
 * The ordinal probit with Gaussian cavities for the predictor and the cut points that bound the category, by the latent score. The
 * outcome is the event `lower < u + e < upper` of the predictor plus standard normal noise `e`; given the score `z = u + e`, the
 * predictor, the lower cut and the upper cut are independent: the predictor is Gaussian with the Kalman update of `z`, the lower cut its
 * Gaussian cavity truncated below `z` and the upper cut its cavity truncated above it. So the mass is the one-dimensional integral over
 * `z` of `N(z; m_u, v_u + 1) Phi((z - m_lo) / s_lo) Phi((m_hi - z) / s_hi)` and each moment is the mixture over `z` of the conditional
 * moments, exact up to the quadrature of one smooth log-concave integrand, and the order of the cuts is enforced by the event itself.
 * `undefined` when a port that the category uses is not a Gaussian message with a density (a known value or a gamma variable), and
 * then the grid is used.
 */
function probitIntegrate(y: number, p: Params, args: readonly Arg[]): GridTripleResult | undefined {
  const { first, last } = ordinalSides(y, p)
  const used = [true, !first, !last]
  const belief = (i: number): { readonly mean: number; readonly variance: number } | undefined => {
    const a = args[i] as Arg
    if (a.value !== undefined || a.family.kind !== 'gaussian') return undefined
    return gaussianOf(a)
  }
  const [u, lo, hi] = [belief(0), used[1] ? belief(1) : undefined, used[2] ? belief(2) : undefined]
  if (u === undefined || (used[1] && lo === undefined) || (used[2] && hi === undefined)) return undefined
  const ignored = (i: number): { readonly mean: number; readonly variance: number } | undefined =>
    used[i] ? undefined : args[i]?.value !== undefined ? gaussianOf(args[i] as Arg) : belief(i)
  const unused = [undefined, ignored(1), ignored(2)]
  if ((!used[1] && unused[1] === undefined) || (!used[2] && unused[2] === undefined)) return undefined
  const sdz = Math.sqrt(u.variance + 1)
  let [wlo, whi] = [u.mean - 9 * sdz, u.mean + 9 * sdz]
  for (const c of [lo, hi]) {
    if (c === undefined) continue
    wlo = Math.min(wlo, c.mean - 9 * Math.sqrt(c.variance))
    whi = Math.max(whi, c.mean + 9 * Math.sqrt(c.variance))
  }
  const logw = (z: number): number => {
    let v = -0.5 * (Math.log(2 * Math.PI * sdz * sdz) + ((z - u.mean) / sdz) ** 2)
    if (lo !== undefined) v += logPhi((z - lo.mean) / Math.sqrt(lo.variance))
    if (hi !== undefined) v += logPhi((hi.mean - z) / Math.sqrt(hi.variance))
    return v
  }
  const { t, step, top, weights, total } = tiltGrid({ lo: wlo, hi: whi }, SCORE_POINTS, logw)
  const n = t.length
  // The conditional mean and variance of each coordinate at each score.
  const gain = u.variance / (u.variance + 1)
  const means = [new Float64Array(n), new Float64Array(n), new Float64Array(n)] as const
  const variances = [new Float64Array(n), new Float64Array(n), new Float64Array(n)] as const
  for (let i = 0; i < n; i++) {
    const z = t[i] as number
    means[0][i] = u.mean + gain * (z - u.mean)
    variances[0][i] = u.variance - gain * u.variance
    if (lo !== undefined) {
      const s = Math.sqrt(lo.variance)
      const a = (z - lo.mean) / s
      const m = millsRatio(a)
      means[1][i] = lo.mean - s * m
      variances[1][i] = Math.max(lo.variance * (1 - a * m - m * m), 0)
    }
    if (hi !== undefined) {
      const s = Math.sqrt(hi.variance)
      const b = (hi.mean - z) / s
      const m = millsRatio(b)
      means[2][i] = hi.mean + s * m
      variances[2][i] = Math.max(hi.variance * (1 - b * m - m * m), 0)
    }
  }
  const mixture = (k: number): { mean: number; variance: number } => {
    const own = [u, lo ?? unused[1], hi ?? unused[2]][k]
    if (own === undefined) return { mean: Number.NaN, variance: Number.NaN }
    if (k > 0 && !used[k]) return own
    const m = centredMoments(weights, total, means[k] as Float64Array)
    let within = 0
    for (let i = 0; i < n; i++) within += (weights[i] as number) * ((variances[k] as Float64Array)[i] as number)
    return { mean: m.mean, variance: within / total + m.variance }
  }
  const parts = [mixture(0), mixture(1), mixture(2)] as const
  return {
    logMass: top + Math.log(total * step),
    mean: [parts[0].mean, parts[1].mean, parts[2].mean],
    variance: [parts[0].variance, parts[1].variance, parts[2].variance],
    meanExp: [Number.NaN, Number.NaN, Number.NaN],
  }
}

/**
 * The cumulative logit outcome by the index of the nodes: `log sigmoid(hi - u) + log sigmoid(u - lo) + log(1 - exp(lo - hi))`, the
 * first two from tables of a cut and the predictor, the last from a table of the two cuts.
 */
function logitIndexed(y: number, p: Params): GridIndexed {
  const { first, last } = ordinalSides(y, p)
  return (nodes) => {
    const [u, lo, hi] = nodes as readonly [Float64Array, Float64Array, Float64Array]
    const n0 = u.length
    const n1 = lo.length
    const upper = last ? undefined : linkTable(logSigmoidOf, u, hi, 1)
    const lower = first ? undefined : linkTable(logSigmoidOf, u, lo, -1)
    const gap =
      first || last
        ? undefined
        : (() => {
            // log(1 - exp(d)) for the difference of every upper cut and lower cut (minus infinity when it is not negative).
            const d = sub(reshape(lo, [1, n1]), reshape(hi, [hi.length, 1]))
            return float64Of(where(less(d, 0), log1p(neg(exp(d))), Number.NEGATIVE_INFINITY))
          })()
    return (a, b, c) => {
      if (first) return (upper as Float64Array)[c * n0 + a] as number
      if (last) return (lower as Float64Array)[b * n0 + a] as number
      const g = (gap as Float64Array)[c * n1 + b] as number
      return g === Number.NEGATIVE_INFINITY
        ? g
        : ((upper as Float64Array)[c * n0 + a] as number) + ((lower as Float64Array)[b * n0 + a] as number) + g
    }
  }
}

/**
 * An ordinal outcome by a cumulative link of a random predictor `x0` with the two cut points that bound the observed category as the
 * random coordinates `x1` (the lower) and `x2` (the upper): `P(y = k) = F(c_k - u) - F(c_(k-1) - u)`. The first category has no lower
 * cut and the last no upper one, and the coordinate of the cut that is not there is ignored (the compiler gives it any known value).
 * A lower cut that is not below the upper one has no mass, so a Gaussian belief over the cuts that is not ordered is cut at the order.
 */
function ordinalCuts(
  name: string,
  what: string,
  logDiff: (a: number, b: number) => number,
  indexed: (y: number, p: Params) => GridIndexed,
  integrate?: TripleLikelihood['integrate'],
): TripleLikelihood {
  return {
    name,
    description: `y | u, lower, upper ~ ordinal ${what} with random cut points: P(y = k) = F(c_k - u) - F(c_(k-1) - u), the outcome a category 0 to categories - 1 and the two cuts around it the coordinates`,
    ports: [
      { name: 'predictor', positive: false },
      { name: 'lower cut', positive: false },
      { name: 'upper cut', positive: false },
    ],
    params: ['categories'],
    noVmp: (y, p) => {
      const { first, last } = ordinalSides(y, p)
      return first || last
        ? undefined
        : 'a category between two random cut points has a density that is zero where they are out of order, so its expected log under a Gaussian belief of the cuts is minus infinity (use expectation propagation, which cuts the cavities at the order; the first and the last category have one cut and run)'
    },
    check: ordinalCheck,
    density: (y, p) => {
      const { first, last } = ordinalSides(y, p)
      return (u, lower, upper) => {
        if (!first && !last && !(lower < upper)) return Number.NEGATIVE_INFINITY
        return logDiff(first ? Number.NEGATIVE_INFINITY : lower - u, last ? Number.POSITIVE_INFINITY : upper - u)
      }
    },
    indexed,
    ...(integrate === undefined ? {} : { integrate }),
  }
}

/**
 * The registry of triple likelihoods by name: `student-t-full`, `ordinal-probit-cuts` and `ordinal-logit-cuts`. A new object each
 * call, so a caller may add a row of its own and build the operator with {@link tripleLikelihoodOperator}.
 *
 * @returns The densities, keyed by name.
 * @concept factor-operators
 * @index table The likelihoods over three random scalars (Student-t location, scale and df; ordinal with random cut points).
 * @example
 * ```ts
 * import { tripleLikelihoods } from 'omai-engine/message-passing'
 * // The middle category of three, cuts -1 and 1, predictor 0: the mass between the cuts is 0.6827.
 * Math.exp(tripleLikelihoods()['ordinal-probit-cuts']?.density(1, { categories: 3 })(0, -1, 1) ?? 0).toFixed(4) // => '0.6827'
 * ```
 */
export function tripleLikelihoods(): Record<string, TripleLikelihood> {
  const all: TripleLikelihood[] = [
    studentTFull,
    ordinalCuts('ordinal-probit-cuts', 'probit', logDiffPhi, probitIndexed, probitIntegrate),
    ordinalCuts('ordinal-logit-cuts', 'logit', logDiffSigmoid, logitIndexed),
  ]
  return Object.fromEntries(all.map((l) => [l.name, l]))
}

/** The density a factor asks for, and the outcome, checked. */
function resolve(
  table: Readonly<Record<string, TripleLikelihood>>,
  params: Params,
): { readonly likelihood: TripleLikelihood; readonly y: number } {
  const key = params['likelihood']
  const likelihood = typeof key === 'string' ? table[key] : undefined
  if (likelihood === undefined)
    throw new ParameterError(
      `likelihood-triple: unknown likelihood ${String(key)} (the registry has ${Object.keys(table).join(', ')})`,
    )
  const y = numberParam(params, 'y')
  const why = likelihood.check?.(y, params)
  if (why !== undefined) throw new ParameterError(`${likelihood.name}: ${why}`)
  return { likelihood, y }
}

/**
 * A numeric likelihood site over three random scalars from a registry of densities: the factor `p(y | x0, x1, x2)`, the parameter
 * `likelihood` naming the row and `y` the outcome. The three ports each take a Gaussian variable (a real value, or the log of a
 * positive one) or, for a positive coordinate, a gamma variable, or a known value; the message to each is the moment match of the
 * three cavities times the density (a Gaussian site, or a gamma site matched on `E log x` and `E x`). The factor supports a power.
 *
 * @param table - The densities to choose from (default {@link tripleLikelihoods}).
 * @returns The operator, named `likelihood-triple`.
 * @concept factor-operators
 * @index function A three-variable numeric EP site from a registry of densities.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, tripleLikelihoodOperator } from 'omai-engine/message-passing'
 * const cavity = (m: number, v: number) => messageArg(gaussian, gaussianEta(m, v))
 * const args = [cavity(0.5, 1), cavity(-1, 1), cavity(1, 1)]
 * const params = { likelihood: 'ordinal-probit-cuts', y: 1, categories: 3 }
 * tripleLikelihoodOperator().tilted?.({ mode: 'ep', to: 0, args, params, power: 1 }).moments.map((v) => v.toFixed(3)) // => ['0.500', '1.250']
 * ```
 */
export function tripleLikelihoodOperator(
  table: Readonly<Record<string, TripleLikelihood>> = tripleLikelihoods(),
): FactorOperator {
  const ports: Port[] = [0, 1, 2].map((i) => ({ name: `x${i}`, accepts: ['gaussian', 'gamma'], constant: true }))
  const variational = (
    t: Readonly<Record<string, TripleLikelihood>>,
    c: OperatorContext,
  ): { readonly logf: (x: readonly number[]) => number; readonly ports: ReturnType<typeof quadraturePorts> } => {
    const { likelihood, y } = resolve(t, c.params)
    const why = likelihood.noVmp?.(y, c.params)
    if (why !== undefined) throw new ImproperMessageError(`${likelihood.name}: ${why}`)
    const density = likelihood.density(y, c.params)
    return {
      logf: (x) => density(x[0] as number, x[1] as number, x[2] as number),
      ports: quadraturePorts(c.args, { log: [true, true, true] }),
    }
  }
  const tilted = (c: OperatorContext): TiltedMoments & { readonly logMass: number } => {
    const { likelihood, y } = resolve(table, c.params)
    const density =
      likelihood.column?.(y, c.params) ??
      (() => {
        const flat = likelihood.density(y, c.params)
        return (x1: number, x2: number) => (x0: number) => flat(x0, x1, x2)
      })()
    const fineness = likelihood.fineness
    const axes = c.args.map((a, i) => {
      const axis = axisOf(a)
      return fineness === undefined ? axis : { ...axis, fineness: fineness[i] as number }
    }) as unknown as readonly [GridAxis, GridAxis, GridAxis]
    const power = c.power
    // The cavities of the three ports are the same for every port the engine asks about, so the messages and the evidence of one
    // sweep share one integral: the engine's cache keeps it under everything it depends on.
    const inputs = [c.args.map((a) => (a.value !== undefined ? a.value : a.eta)), c.params, power]
    const indexed = likelihood.indexed?.(y, c.params)
    const r = memoised(c, `likelihood-triple:${keyOf(inputs)}`, () => {
      const exact = power === 1 ? likelihood.integrate?.(y, c.params, c.args) : undefined
      if (exact !== undefined) return exact
      return integrateTriple(
        axes,
        power === 1
          ? density
          : (x1, x2) => {
              const f = density(x1, x2)
              return (x0) => power * f(x0)
            },
        indexed === undefined || power === 1
          ? indexed
          : (nodes) => {
              const f = indexed(nodes)
              return (a, b, cc) => power * f(a, b, cc)
            },
      )
    })
    const k = c.to
    const target = c.args[k]?.family as Family
    const moments =
      target.kind === 'gamma'
        ? [r.mean[k] as number, r.meanExp[k] as number]
        : [r.mean[k] as number, (r.variance[k] as number) + (r.mean[k] as number) ** 2]
    let logZ = r.logMass
    for (const a of c.args) logZ += massOf(a)
    return { moments, logZ, logMass: r.logMass }
  }
  return defineOperator({
    name: 'likelihood-triple',
    description:
      'y | x0, x1, x2 ~ a density of the registry over three random scalars (a location, a scale and degrees of freedom; a predictor and two cut points)',
    ports,
    params: (p) => {
      const key = p['likelihood']
      const entry = typeof key === 'string' ? table[key] : undefined
      return {
        likelihood: param.text({ choices: Object.keys(table) }),
        y: param.number(),
        ...(entry === undefined ? {} : namedParams(entry.params)),
      }
    },
    unsupported: (mode, params) => {
      if (mode !== 'vmp') return undefined
      const key = params['likelihood']
      const y = params['y']
      if (typeof key !== 'string' || typeof y !== 'number') return undefined
      try {
        return table[key]?.noVmp?.(y, params)
      } catch {
        // an outcome or a parameter the check refuses is its own error
        return undefined
      }
    },
    check: (params, families) => {
      try {
        const { likelihood } = resolve(table, params)
        for (const [i, port] of likelihood.ports.entries()) {
          const kind = families[i]?.spec.kind
          if (kind === undefined) continue
          if (kind !== 'gaussian' && !(port.positive && kind === 'gamma'))
            return `the ${port.name} of ${likelihood.name} is ${port.positive ? 'a Gaussian variable on its log or a gamma variable' : 'a Gaussian variable'}, not a ${kind} variable`
        }
        return undefined
      } catch (e) {
        return errorText(e)
      }
    },
    rules: {
      ep: tiltedRule((c) => {
        const t = tilted(c)
        return { moments: t.moments, logZ: t.logZ }
      }),
      // The mean field: the non-conjugate message of the density under the three beliefs.
      vmp: messageRule((c) => {
        const { logf, ports } = variational(table, c)
        return { message: numericMessage(ports, c.to, logf) }
      }, 'numeric'),
    },
    evidence: {
      ep: (c) => (allProper(c.args) ? tilted({ ...c, to: 0 }).logZ : Number.NaN),
      vmp: (c) => {
        const { logf, ports } = variational(table, c)
        return numericExpectedLog(ports, logf)
      },
    },
    traits: { readsTarget: true, power: true },
  })
}

/** The triple likelihood site over the library's registry. */
export const likelihoodTripleOperator: FactorOperator = tripleLikelihoodOperator()
