/**
 * The two-variable numeric sites: a likelihood `p(y | x0, x1)` of an outcome over two random scalars at once, the generalisation
 * of the negative binomial with a random dispersion to a registry of densities. A coordinate is a real value (a location, a
 * predictor) or a positive one carried either as a Gaussian on its log (a log-normal, half-normal or half-Cauchy parameter,
 * projected by the compiler) or as a gamma variable, so a Laplace noise whose scale is learned, a Student-t with learned degrees
 * of freedom or a learned scale, an asymmetric Laplace (quantile regression) with a learned scale, a Weibull with a learned shape and a Beta-distributed outcome with two random shapes are six rows of one table
 * read by one operator, `likelihood-pair`, and another is a few lines. EP matches the moments of both variables under their
 * cavities times the density by a panelled Gauss-Legendre grid in the two coordinates (`_grid.ts`), split at the kinks the
 * density has and zoomed onto where the mass is, so the sites cost a few thousand evaluations of the density each.
 */

import { arg, errorText, keyOf, memoised } from '../../core/_messages.ts'
import { allProper, etaOf, massOf, numberParam, vectorParam } from '../../core/args.ts'
import { defineOperator, messageRule, siteRule, tiltedRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { namedParams, param } from '../../core/params.ts'
import type { FactorOperator, Family, OperatorContext, Params, Port, TiltedMoments } from '../../core/types.ts'
import { axisOf, integratePair } from '../../numerics/_grid.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { type VectorSiteTilt, vectorSiteTilt } from '../../numerics/_vector-site.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

/** One coordinate of a pair likelihood: a real value, or a positive one (the coordinate is then its log). */
export interface PairPort {
  readonly name: string
  readonly positive: boolean
}

/** One density of the registry of pair likelihoods. */
export interface PairLikelihood {
  readonly name: string
  readonly description: string
  /** The two coordinates: a real port is a Gaussian variable, a positive one a Gaussian on its log or a gamma variable. */
  readonly ports: readonly [PairPort, PairPort]
  /** The numeric parameters besides the outcome `y`. */
  readonly params: readonly string[]
  /** A reason the outcome or the parameters are not valid, or `undefined`. */
  check?(y: number, p: Params): string | undefined
  /** The log density at the coordinates, with the constants of `y` and the parameters precomputed. */
  density(y: number, p: Params): (x0: number, x1: number) => number
  /**
   * The same density as a function of the second coordinate that returns a function of the first, when what depends on the second
   * coordinate alone (a gamma function of a shape, an exponential) is worth computing once per node of it: the grid evaluates the
   * returned function at every node of the first coordinate. Absent: `density` is used.
   */
  column?(y: number, p: Params): (x1: number) => (x0: number) => number
  /** Why the density has no variational message (an expected log that is infinite under any Gaussian or gamma belief of a coordinate), or absent. */
  readonly noVmp?: string
  /** The coordinates at which the density has a kink, for each port (panels are split there). */
  kinks?(y: number, p: Params): readonly [readonly number[], readonly number[]]
}

const positive = (p: Params, name: string): string | undefined =>
  numberParam(p, name) > 0 ? undefined : `${name} must be positive`

/** Laplace noise with a random scale `b = exp(x1)` around a random location `x0`. */
const laplaceScale: PairLikelihood = {
  name: 'laplace-scale',
  description: 'y | u, s ~ Laplace(u, exp(s)): absolute-error noise whose scale is random too',
  ports: [
    { name: 'location', positive: false },
    { name: 'scale', positive: true },
  ],
  params: [],
  density: (y) => (u, s) => -Math.LN2 - s - Math.abs(y - u) * Math.exp(-s),
  kinks: (y) => [[y], []],
}

/** The asymmetric Laplace of quantile regression at the level `tau` with a random scale `exp(x1)` around a random location `x0`. */
const asymmetricLaplaceScale: PairLikelihood = {
  name: 'asymmetric-laplace-scale',
  description: 'y | u, s ~ asymmetric Laplace(u, exp(s), tau): quantile regression whose scale is random too',
  ports: [
    { name: 'location', positive: false },
    { name: 'scale', positive: true },
  ],
  params: ['tau'],
  check: (_y, p) => {
    const tau = numberParam(p, 'tau')
    return tau > 0 && tau < 1 ? undefined : 'tau must be in (0, 1)'
  },
  density: (y, p) => {
    const tau = numberParam(p, 'tau')
    const c = Math.log(tau * (1 - tau))
    return (u, s) => {
      const r = (y - u) * Math.exp(-s)
      return c - s - r * (tau - (r < 0 ? 1 : 0))
    }
  },
  kinks: (y) => [[y], []],
}

/** Student-t noise with random degrees of freedom `df = exp(x1)` around a random location, with a known scale. */
const studentTDf: PairLikelihood = {
  name: 'student-t-df',
  description: 'y | u, nu ~ Student t(exp(nu), u, scale): robust noise whose degrees of freedom are random',
  ports: [
    { name: 'location', positive: false },
    { name: 'df', positive: true },
  ],
  params: ['scale'],
  check: (_y, p) => positive(p, 'scale'),
  density: (y, p) => {
    const scale = numberParam(p, 'scale')
    const logScale = Math.log(scale)
    return (u, t) => {
      const df = Math.exp(t)
      const z = (y - u) / scale
      return (
        lgammaOf((df + 1) / 2) -
        lgammaOf(df / 2) -
        0.5 * Math.log(df * Math.PI) -
        logScale -
        ((df + 1) / 2) * Math.log1p((z * z) / df)
      )
    }
  },
  column: (y, p) => {
    const scale = numberParam(p, 'scale')
    const logScale = Math.log(scale)
    return (t) => {
      const df = Math.exp(t)
      const c = lgammaOf((df + 1) / 2) - lgammaOf(df / 2) - 0.5 * Math.log(df * Math.PI) - logScale
      const half = (df + 1) / 2
      return (u) => {
        const z = (y - u) / scale
        return c - half * Math.log1p((z * z) / df)
      }
    }
  },
}

/** Student-t noise with a known number of degrees of freedom around a random location, whose scale `exp(x1)` is random. */
const studentTScale: PairLikelihood = {
  name: 'student-t-scale',
  description: 'y | u, s ~ Student t(df, u, exp(s)): robust noise whose scale is random',
  ports: [
    { name: 'location', positive: false },
    { name: 'scale', positive: true },
  ],
  params: ['df'],
  check: (_y, p) => positive(p, 'df'),
  density: (y, p) => {
    const df = numberParam(p, 'df')
    const c = lgammaOf((df + 1) / 2) - lgammaOf(df / 2) - 0.5 * Math.log(df * Math.PI)
    return (u, s) => {
      const z = (y - u) * Math.exp(-s)
      return c - s - ((df + 1) / 2) * Math.log1p((z * z) / df)
    }
  },
  column: (y, p) => {
    const df = numberParam(p, 'df')
    const c = lgammaOf((df + 1) / 2) - lgammaOf(df / 2) - 0.5 * Math.log(df * Math.PI)
    const half = (df + 1) / 2
    return (s) => {
      const inverse = Math.exp(-s)
      return (u) => {
        const z = (y - u) * inverse
        return c - s - half * Math.log1p((z * z) / df)
      }
    }
  },
}

/** A Weibull time with random shape `k = exp(x1)` and scale `exp(x0)`; `censored = 1` makes `y` a right-censoring time. */
const weibullShape: PairLikelihood = {
  name: 'weibull-shape',
  description:
    'y | u, k ~ Weibull(shape exp(k), scale exp(u)): a survival time whose shape is random too; censored = 1 makes y a right-censoring time',
  noVmp:
    'the survival term exp(shape (log y - log scale)) is a double exponential in a Gaussian or gamma shape, so its expected log under a mean-field belief is infinite (use expectation propagation, or fix the shape: the weibull-log likelihood)',
  ports: [
    { name: 'log scale', positive: false },
    { name: 'shape', positive: true },
  ],
  params: ['censored'],
  check: (y) => (y > 0 ? undefined : 'the time must be positive'),
  density: (y, p) => {
    const censored = numberParam(p, 'censored', 0) !== 0
    const logY = Math.log(y)
    return (u, t) => {
      const k = Math.exp(t)
      const tail = Math.exp(k * (logY - u))
      return censored ? -tail : t + (k - 1) * logY - k * u - tail
    }
  },
  column: (y, p) => {
    const censored = numberParam(p, 'censored', 0) !== 0
    const logY = Math.log(y)
    return (t) => {
      const k = Math.exp(t)
      const c = censored ? 0 : t + (k - 1) * logY
      return (u) => (censored ? -Math.exp(k * (logY - u)) : c - k * u - Math.exp(k * (logY - u)))
    }
  },
}

/** A Beta-distributed outcome in (0, 1) with two random shapes `a = exp(x0)` and `b = exp(x1)`. */
const betaShapes: PairLikelihood = {
  name: 'beta-shapes',
  description: 'y | a, b ~ Beta(exp(a), exp(b)): a proportion whose two shapes are random',
  ports: [
    { name: 'first shape', positive: true },
    { name: 'second shape', positive: true },
  ],
  params: [],
  check: (y) => (y > 0 && y < 1 ? undefined : 'the outcome must be in (0, 1)'),
  density: (y) => {
    const logY = Math.log(y)
    const log1mY = Math.log1p(-y)
    return (ta, tb) => {
      const a = Math.exp(ta)
      const b = Math.exp(tb)
      return (a - 1) * logY + (b - 1) * log1mY + lgammaOf(a + b) - lgammaOf(a) - lgammaOf(b)
    }
  },
  column: (y) => {
    const logY = Math.log(y)
    const log1mY = Math.log1p(-y)
    return (tb) => {
      const b = Math.exp(tb)
      const c = (b - 1) * log1mY - lgammaOf(b)
      return (ta) => {
        const a = Math.exp(ta)
        return (a - 1) * logY + c + lgammaOf(a + b) - lgammaOf(a)
      }
    }
  },
}

/**
 * The registry of pair likelihoods by name: `laplace-scale`, `asymmetric-laplace-scale`, `student-t-df`, `student-t-scale`, `weibull-shape` and `beta-shapes`. A new object each
 * call, so a caller may add a row of its own and build the operator with {@link pairLikelihoodOperator}.
 *
 * @returns The densities, keyed by name.
 * @concept factor-operators
 * @index table The likelihoods over two random scalars (Laplace scale, Student-t degrees of freedom or scale, Weibull shape, beta shapes).
 * @example
 * ```ts
 * import { pairLikelihoods } from 'omai-engine/message-passing'
 * // Laplace noise, outcome 1, location 0, scale exp(0) = 1: the log density is log(1/2) - 1.
 * pairLikelihoods()['laplace-scale']?.density(1, {})(0, 0).toFixed(4) // => '-1.6931'
 * ```
 */
export function pairLikelihoods(): Record<string, PairLikelihood> {
  const all: PairLikelihood[] = [
    laplaceScale,
    asymmetricLaplaceScale,
    studentTDf,
    studentTScale,
    weibullShape,
    betaShapes,
  ]
  return Object.fromEntries(all.map((l) => [l.name, l]))
}

/** The density a factor asks for, and the outcome, checked. */
function resolve(
  table: Readonly<Record<string, PairLikelihood>>,
  params: Params,
): { readonly likelihood: PairLikelihood; readonly y: number } {
  const key = params['likelihood']
  const likelihood = typeof key === 'string' ? table[key] : undefined
  if (likelihood === undefined)
    throw new ParameterError(
      `likelihood-pair: unknown likelihood ${String(key)} (the registry has ${Object.keys(table).join(', ')})`,
    )
  const y = numberParam(params, 'y')
  const why = likelihood.check?.(y, params)
  if (why !== undefined) throw new ParameterError(`${likelihood.name}: ${why}`)
  return { likelihood, y }
}

/**
 * A numeric likelihood site over two random scalars from a registry of densities: the factor `p(y | x0, x1)`, the parameter
 * `likelihood` naming the row and `y` the outcome. The two ports each take a Gaussian variable (a real value, or the log of a
 * positive one) or, for a positive coordinate, a gamma variable; the message to each is the moment match of the two cavities
 * times the density (a Gaussian site, or a gamma site matched on `E log x` and `E x`). The factor supports a power.
 *
 * @param table - The densities to choose from (default {@link pairLikelihoods}).
 * @returns The operator, named `likelihood-pair`.
 * @concept factor-operators
 * @index function A two-variable numeric EP site from a registry of densities.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, pairLikelihoodOperator } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(0, 1)), messageArg(gaussian, gaussianEta(0, 1))]
 * const params = { likelihood: 'laplace-scale', y: 1 }
 * pairLikelihoodOperator().tilted?.({ mode: 'ep', to: 0, args, params, power: 1 }).moments.map((v) => v.toFixed(3)) // => ['0.581', '0.851']
 * ```
 */
export function pairLikelihoodOperator(
  table: Readonly<Record<string, PairLikelihood>> = pairLikelihoods(),
): FactorOperator {
  const ports: Port[] = [0, 1].map((i) => ({ name: `x${i}`, accepts: ['gaussian', 'gamma'], constant: true }))
  const variational = (
    t: Readonly<Record<string, PairLikelihood>>,
    c: OperatorContext,
  ): { readonly logf: (x: readonly number[]) => number; readonly ports: ReturnType<typeof quadraturePorts> } => {
    const { likelihood, y } = resolve(t, c.params)
    if (likelihood.noVmp !== undefined) throw new ImproperMessageError(`${likelihood.name}: ${likelihood.noVmp}`)
    const density = likelihood.density(y, c.params)
    return {
      logf: (x) => density(x[0] as number, x[1] as number),
      ports: quadraturePorts(c.args, { log: [true, true], kinks: likelihood.kinks?.(y, c.params) ?? [] }),
    }
  }
  const tilted = (c: OperatorContext): TiltedMoments & { readonly logMass: number } => {
    const { likelihood, y } = resolve(table, c.params)
    const density =
      likelihood.column?.(y, c.params) ??
      (() => {
        const flat = likelihood.density(y, c.params)
        return (x1: number) => (x0: number) => flat(x0, x1)
      })()
    const kinks = likelihood.kinks?.(y, c.params) ?? [[], []]
    const axes = [axisOf(c.args[0] as never, kinks[0]), axisOf(c.args[1] as never, kinks[1])] as const
    const power = c.power
    const r = integratePair(
      axes,
      power === 1
        ? density
        : (x1) => {
            const f = density(x1)
            return (x0) => power * f(x0)
          },
    )
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
    name: 'likelihood-pair',
    description:
      'y | x0, x1 ~ a density of the registry over two random scalars (a location and a scale, a shape, a pair of shapes)',
    ports,
    params: (p) => {
      const key = p['likelihood']
      const entry = typeof key === 'string' ? table[key] : undefined
      return {
        likelihood: param.text({ choices: Object.keys(table) }),
        y: param.number(),
        ...(entry === undefined ? {} : namedParams(entry.params, { optional: ['censored'] })),
      }
    },
    unsupported: (mode, params) => {
      if (mode !== 'vmp') return undefined
      const key = params['likelihood']
      return typeof key === 'string' ? table[key]?.noVmp : undefined
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
      // The mean field: the non-conjugate message of the density under the two beliefs (a gamma belief on a positive coordinate is
      // read through its log, as the grid of EP does).
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

/** The pair likelihood site over the library's registry. */
export const likelihoodPairOperator: FactorOperator = pairLikelihoodOperator()

/**
 * A pair likelihood of two coordinates that are linear forms of one Gaussian vector: the density of the registry at `(u, t) = (a0 . w +
 * b0, a1 . w + b1)`, one port (`w`), the parameters of `likelihood-pair` and `matrix` (the two rows `a0`, `a1`, row-major `2 x D`) and
 * `offset` (`b0`, `b1`, default zero). The joint cavity of the coordinates is `N(A m + b, A V A')` with its correlation, the tilted
 * moments are an integral over it (adaptive Gauss-Hermite over its rank, which is two), and the message to `w` is the full natural-parameter
 * site, so the correlation the density puts between the coordinates (the two shapes of a beta outcome, whose posterior is a narrow ridge)
 * reaches the vector instead of two separate Gaussian messages. A coordinate is a real value or the log of a positive one, as in
 * `likelihood-pair`; a density with a kink (a Laplace) is not smooth for this rule, and the coordinates' gamma form is not a Gaussian
 * vector's, so both stay with `likelihood-pair`. The factor supports a power; there is no variational rule.
 *
 * @param table - The densities to choose from (default {@link pairLikelihoods}).
 * @returns The operator, named `likelihood-pair-row`.
 * @concept factor-operators
 * @index function A pair likelihood site on one Gaussian vector: a joint message for two coordinates that are linear forms of it.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, messageArg, pairRowOperator } from 'omai-engine/message-passing'
 * const w = gaussianMv(2)
 * const args = [messageArg(w, gaussianMvEta([0, 0], [1, 0, 0, 1]))]
 * const params = { likelihood: 'beta-shapes', y: 0.3, matrix: [1, 0, 0, 1] }
 * pairRowOperator().ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message.length // => 6
 * ```
 */
export function pairRowOperator(table: Readonly<Record<string, PairLikelihood>> = pairLikelihoods()): FactorOperator {
  const tilt = (c: OperatorContext): VectorSiteTilt => {
    const inputs = [c.args.map((a) => a.eta), c.params, c.power]
    return memoised(c, `likelihood-pair-row:${keyOf(inputs)}`, () => {
      const { likelihood, y } = resolve(table, c.params)
      const density = likelihood.density(y, c.params)
      const w = arg(c, 0)
      const offset = c.params['offset'] === undefined ? [0, 0] : vectorParam(c.params, 'offset')
      return vectorSiteTilt(w, {
        coordinates: 2,
        matrix: vectorParam(c.params, 'matrix'),
        offset,
        logf: (z) => c.power * density(z[0] as number, z[1] as number),
        name: 'likelihood-pair-row',
      })
    })
  }
  return defineOperator({
    name: 'likelihood-pair-row',
    description:
      'y | (a0 . w, a1 . w) ~ a density of the registry over two linear forms of a Gaussian vector w, with their joint message',
    ports: [{ name: 'w', accepts: ['gaussian-mv'] }],
    params: (p) => {
      const key = p['likelihood']
      const entry = typeof key === 'string' ? table[key] : undefined
      return {
        likelihood: param.text({ choices: Object.keys(table) }),
        y: param.number(),
        matrix: param.vector(),
        offset: param.vector({ optional: true }),
        ...(entry === undefined ? {} : namedParams(entry.params, { optional: ['censored'] })),
      }
    },
    check: (params, families) => {
      try {
        const { likelihood } = resolve(table, params)
        if (likelihood.kinks !== undefined)
          return `${likelihood.name} has a kink: the Gauss-Hermite rule of the row is for smooth densities (use likelihood-pair)`
        const d = families[0]?.dim ?? 0
        if (vectorParam(params, 'matrix').length !== 2 * d)
          return `likelihood-pair-row: matrix must have 2 x ${d} entries (a row per coordinate)`
        if (params['offset'] !== undefined && vectorParam(params, 'offset').length !== 2)
          return 'likelihood-pair-row: offset must have 2 entries'
        return undefined
      } catch (e) {
        return errorText(e)
      }
    },
    rules: {
      ep: siteRule((c) => {
        const cavity = etaOf(arg(c, 0))
        const t = tilt(c)
        return { natural: cavity.map((v, i) => v + (t.site[i] ?? 0)) }
      }),
    },
    evidence: {
      ep: (c) => (allProper(c.args) ? tilt(c).logMass + massOf(arg(c, 0)) : Number.NaN),
    },
    noRule: {
      vmp: 'the mean field of a vector has no Gaussian expected log of a pair density in its coordinates: compile the two coordinates as scalar variables (packScalars off) for the variational rule of likelihood-pair',
    },
    traits: { readsTarget: true, power: true },
  })
}

/** The pair likelihood row over the library's registry. */
export const likelihoodPairRowOperator: FactorOperator = pairRowOperator()
