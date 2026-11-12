/**
 * Unary functions of a Gaussian variable as an operator: `y = f(x)` for `exp`, `square`, `tanh`, `relu`, `softplus`, `abs`
 * and `sigmoid`, the factor behind a log-normal (`exp` of a Gaussian), a banana (`square`), a Bayesian neural network
 * (`tanh`, `relu`: probabilistic backpropagation) and the non-centred funnel. The output is matched by moments under the
 * input's cavity (the pushforward of the input, weighted by what the output has been told), and the message back to the input
 * is the moment match of the tilted input: one integral over `x` gives both. The rectifier has a closed form (two truncated
 * Gaussians); the pushforward of a flat output is closed form for `exp`, `square`, `relu` and `abs`; the others are the
 * panelled quadrature of `_tilted.ts`, split at the function's kinks. Under variational message passing the output is defined by the
 * input (a Gaussian with the mean and variance of the pushforward) and the message to the input is non-conjugate, by quadrature.
 *
 * The function can be a chain: the parameter `steps` lists unary steps (a registry name, or `mul`, `add`, `pow` with a constant)
 * applied one after the other, so `1 / sqrt(d exp(x) + n)` is one factor from `x` to the output rather than a chain of Gaussian
 * approximations (the compiler collapses the arithmetic on a function's output into the chain). The input can be a gamma variable
 * (a precision `alpha`: `1 / sqrt(alpha)` is the scale of a non-centred coefficient), integrated on a grid in `log alpha`, and the
 * message to it is the gamma with the tilted moments.
 */

import { sigmoidOf } from '../../../base/index.ts'
import { arg, errorText, isFlat } from '../../core/_messages.ts'
import { allProper, etaOf, gaussianMessage, gaussianOf, massOf } from '../../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { param } from '../../core/params.ts'
import { truncatedNormalMoments } from '../../core/truncated.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { centredMoments, resolvedVariance, shiftedSum } from '../../numerics/_moments.ts'
import { Phi, phi, softplusOf } from '../../numerics/_scalar.ts'
import { type TiltHint, tiltedByQuadrature } from '../../numerics/_tilted.ts'
import { definedFunctionMessage, quadraturePort } from '../kernel/variational-numeric.ts'

/** A function of the registry: its values, the kinks to split quadrature panels at, and its moments under a Gaussian where it has them in closed form. */
export interface UnaryFunction {
  readonly name: string
  readonly description: string
  readonly f: (x: number) => number
  /** Points where the function has a kink (the quadrature panels are split there). */
  readonly kinks?: readonly number[]
  /** The mean and variance of `f(x)` for `x ~ N(mean, variance)` in closed form, when the function has one. */
  readonly pushforward?: (mean: number, variance: number) => { readonly mean: number; readonly variance: number }
}

const unary = (u: UnaryFunction): UnaryFunction => u

/**
 * The registry of unary functions by name: `exp`, `square`, `tanh`, `relu`, `softplus`, `abs`, `sigmoid`, `sqrt`, `recip`. A new object each call.
 *
 * @returns The functions, keyed by name.
 * @concept factor-operators
 * @index table The unary functions of a Gaussian variable (exp, square, tanh, relu, softplus, abs, sigmoid, sqrt, recip) by name.
 * @example
 * ```ts
 * import { unaryFunctions } from 'omai-engine/message-passing'
 * // The rectified standard normal has mean 1 / sqrt(2 pi).
 * unaryFunctions()['relu']?.pushforward?.(0, 1).mean.toFixed(6) // => '0.398942'
 * ```
 */
export function unaryFunctions(): Record<string, UnaryFunction> {
  const all: UnaryFunction[] = [
    unary({
      name: 'exp',
      description: 'y = exp(x): a log-normal variable',
      f: Math.exp,
      pushforward: (m, v) => ({ mean: Math.exp(m + v / 2), variance: (Math.exp(v) - 1) * Math.exp(2 * m + v) }),
    }),
    unary({
      name: 'square',
      description: 'y = x^2: a scaled non-central chi-squared variable',
      f: (x) => x * x,
      pushforward: (m, v) => ({ mean: m * m + v, variance: 2 * v * (v + 2 * m * m) }),
    }),
    unary({ name: 'tanh', description: 'y = tanh(x): the hidden unit of a neural network', f: Math.tanh }),
    unary({
      name: 'relu',
      description: 'y = max(x, 0): the rectifier, in closed form',
      f: (x) => Math.max(x, 0),
      kinks: [0],
      pushforward: (m, v) => {
        const s = Math.sqrt(v)
        const a = m / s
        const mean = m * Phi(a) + s * phi(a)
        const second = (m * m + v) * Phi(a) + m * s * phi(a)
        return { mean, variance: Math.max(second - mean * mean, 0) }
      },
    }),
    unary({ name: 'softplus', description: 'y = log(1 + exp(x)): a smooth rectifier', f: softplusOf }),
    unary({
      name: 'abs',
      description: 'y = |x|: the absolute value',
      f: Math.abs,
      kinks: [0],
      pushforward: (m, v) => {
        const s = Math.sqrt(v)
        const mean = s * Math.sqrt(2 / Math.PI) * Math.exp(-(m * m) / (2 * v)) + m * (1 - 2 * Phi(-m / s))
        return { mean, variance: Math.max(m * m + v - mean * mean, 0) }
      },
    }),
    unary({ name: 'sigmoid', description: 'y = 1 / (1 + exp(-x)): the logistic function', f: sigmoidOf }),
    unary({ name: 'sqrt', description: 'y = sqrt(x): the square root of a positive value', f: Math.sqrt }),
    unary({ name: 'recip', description: 'y = 1 / x: the reciprocal', f: (x) => 1 / x }),
  ]
  return Object.fromEntries(all.map((u) => [u.name, u]))
}

/** One step of a chain: a registry function by name, or `mul`, `add`, `pow` with its constant. */
export type Step = readonly [string, number?]

/** The function of a chain of steps, applied in order. */
export function chainOf(table: Readonly<Record<string, UnaryFunction>>, steps: readonly Step[]): UnaryFunction {
  const fs = steps.map(([name, k]): ((x: number) => number) => {
    const c = k ?? Number.NaN
    if (name === 'mul') return (x) => x * c
    if (name === 'add') return (x) => x + c
    if (name === 'pow') return (x) => x ** c
    const fn = table[name]
    if (fn === undefined)
      throw new ParameterError(
        `apply: unknown step ${name} (the registry has ${Object.keys(table).join(', ')}, and mul, add, pow with a constant)`,
      )
    return fn.f
  })
  const first = steps.length === 1 ? table[steps[0]?.[0] ?? ''] : undefined
  return {
    name: first?.name ?? 'chain',
    description: `the composition of ${steps.map(([name, k]) => (k === undefined ? name : `${name} ${k}`)).join(', ')}`,
    f: (x) => fs.reduce((v, f) => f(v), x),
    ...(first?.kinks === undefined ? {} : { kinks: first.kinks }),
    ...(first?.pushforward === undefined ? {} : { pushforward: first.pushforward }),
  }
}

/** The function a factor applies: the chain of its parameter `steps`, else the registry function named by `fn`. */
function resolve(
  table: Readonly<Record<string, UnaryFunction>>,
  params: Readonly<Record<string, unknown>>,
): UnaryFunction {
  const steps = params['steps']
  if (Array.isArray(steps) && steps.length > 0) return chainOf(table, steps as Step[])
  const key = params['fn']
  const fn = typeof key === 'string' ? table[key] : undefined
  if (fn === undefined)
    throw new ParameterError(
      `apply: unknown function ${String(key)} (the registry has ${Object.keys(table).join(', ')})`,
    )
  return fn
}

/** What one integral over the input gives: the moments of the tilted input and of the output, and the log mass. */
interface Tilt {
  readonly logMass: number
  readonly x: { readonly mean: number; readonly variance: number }
  readonly y: { readonly mean: number; readonly variance: number }
}

const FLOOR = 1e-300

/** The rectifier against a Gaussian input N(m, v) and an output message exp(h y - J y^2 / 2), in closed form. */
function reluTilt(m: number, v: number, h: number, J: number): Tilt {
  const precision = 1 / v + J
  if (!(precision > 0))
    throw new ImproperMessageError('apply: the output message and the input make no density together')
  const v2 = 1 / precision
  const m2 = v2 * (m / v + h)
  // x <= 0: y = 0 and the output message weighs 1; x > 0: y = x, the weight a Gaussian times exp(h x - J x^2 / 2).
  const below = truncatedNormalMoments(m, v, Number.NEGATIVE_INFINITY, 0)
  const above = truncatedNormalMoments(m2, v2, 0, Number.POSITIVE_INFINITY)
  const logA = below.logMass
  const logB = 0.5 * Math.log(v2 / v) + 0.5 * ((m2 * m2) / v2 - (m * m) / v) + above.logMass
  const { weights, total, logTotal } = shiftedSum([logA, logB])
  const wA = (weights[0] as number) / total
  const wB = 1 - wA
  // The two-component mixtures, centred: the mean of the variances plus the squared distances of the means from the mixture's.
  const xMean = wA * below.mean + wB * above.mean
  const xVariance =
    wA * (below.variance + (below.mean - xMean) ** 2) + wB * (above.variance + (above.mean - xMean) ** 2)
  // y is zero below the origin and x above it.
  const yMean = wB * above.mean
  const yVariance = wA * yMean * yMean + wB * (above.variance + (above.mean - yMean) ** 2)
  return {
    logMass: logTotal,
    x: { mean: xMean, variance: resolvedVariance(xVariance, 'apply') },
    y: { mean: yMean, variance: resolvedVariance(yVariance, 'apply') },
  }
}

/** The tilted input and output of `y = f(x)` against the cavity of x and the natural parameters of the output's message. */
function tilt(fn: UnaryFunction, cavity: { mean: number; variance: number }, y: Eta, power: number): Tilt {
  const h = y[0] ?? 0
  const J = y[1] ?? 0
  if (fn.name === 'relu' && power === 1) return reluTilt(cavity.mean, cavity.variance, h, J)
  const f = fn.f
  const logf = (x: number): number => {
    const v = f(x)
    if (!Number.isFinite(v)) return Number.NEGATIVE_INFINITY
    return power * (h * v - 0.5 * J * v * v)
  }
  const hint: TiltHint | undefined = fn.kinks === undefined ? undefined : { kinks: fn.kinks }
  // Where the function is not defined (a square root of a negative value) the weight is zero: its value must not poison the moments.
  const t = tiltedByQuadrature(cavity, logf, hint, (x) => {
    const v = f(x)
    return Number.isFinite(v) ? v : 0
  })
  return {
    logMass: t.logMass,
    x: { mean: t.mean, variance: resolvedVariance(t.variance, 'apply') },
    y: { mean: t.fnMean, variance: resolvedVariance(t.fnVariance, 'apply') },
  }
}

/** The tilted input and output of `y = f(x)` for a gamma input: a grid in `log x` under the cavity, weighted by the output's message. */
function tiltGamma(
  fn: UnaryFunction,
  cavity: Arg,
  y: Eta,
  power: number,
): {
  readonly logMass: number
  readonly x: Eta
  readonly y: { readonly mean: number; readonly variance: number }
} {
  const h = y[0] ?? 0
  const J = y[1] ?? 0
  const port = quadraturePort(cavity, { points: 481 })
  const n = port.coords.length
  const values = Float64Array.from(port.coords, fn.f)
  const lw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const v = values[i] as number
    lw[i] = Number.isFinite(v)
      ? Math.log(port.weights[i] as number) + power * (h * v - 0.5 * J * v * v)
      : Number.NEGATIVE_INFINITY
  }
  const { top, weights, total: m0 } = shiftedSum(lw)
  if (!Number.isFinite(top)) throw new ImproperMessageError('apply: the factor has no mass against this cavity')
  let logX = 0
  let xMean = 0
  let yMean = 0
  for (let i = 0; i < n; i++) {
    const w = weights[i] as number
    if (w === 0) continue
    logX += w * (port.stats[i]?.[0] ?? 0)
    xMean += w * (port.stats[i]?.[1] ?? 0)
    yMean += w * (values[i] as number)
  }
  let yVar = 0
  for (let i = 0; i < n; i++) {
    const w = weights[i] as number
    if (w > 0) yVar += w * ((values[i] as number) - yMean / m0) ** 2
  }
  return {
    logMass: top + Math.log(m0),
    x: [logX / m0, xMean / m0],
    y: { mean: yMean / m0, variance: resolvedVariance(yVar / m0, 'apply') },
  }
}

/**
 * The factor `y = f(x)` for a unary function of the registry, on Gaussian variables: ports `y` and `x`, the function by the
 * parameter `fn`. The message to `y` is the moment match of the pushforward of the input's cavity, weighted by the output's
 * own cavity; the message to `x` is the moment match of the input under the same weight (nothing, while the output has been
 * told nothing). `relu` is closed form; `exp`, `square`, `relu` and `abs` push a flat output forward in closed form; the rest
 * integrate by quadrature. The moments are checked against a fine grid.
 *
 * @param name - The operator's name.
 * @param table - The functions to choose from (default {@link unaryFunctions}).
 * @returns The operator.
 * @concept factor-operators
 * @index function The factor y = f(x) of a unary function (exp, square, tanh, relu, softplus, abs, sigmoid) of a Gaussian: EP.
 * @example
 * ```ts
 * import { applyFunction, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const op = applyFunction('apply')
 * // Nothing is known about y = exp(x), x ~ N(0, 1): y is matched to the log-normal's mean and variance.
 * const args = [messageArg(gaussian, gaussian.flat), messageArg(gaussian, gaussianEta(0, 1))]
 * const m = op.ep?.({ mode: 'ep', to: 0, args, params: { fn: 'exp' }, power: 1 })
 * m?.message.map((v) => v.toFixed(4)) // => ['0.3530', '0.2141']
 * ```
 */
export function applyFunction(
  name: string,
  table: Readonly<Record<string, UnaryFunction>> = unaryFunctions(),
): FactorOperator {
  return defineOperator({
    name,
    description: 'y = f(x) for a unary function (or a chain of them) of a Gaussian or gamma variable',
    params: {
      fn: param.text({ optional: true, choices: Object.keys(table) }),
      steps: param.any({ optional: true }),
    },
    ports: [
      { name: 'y', accepts: ['gaussian'] },
      { name: 'x', accepts: ['gaussian', 'gamma'] },
    ],
    check: (params) => {
      try {
        resolve(table, params)
        return undefined
      } catch (e) {
        return errorText(e)
      }
    },
    rules: {
      ep: siteRule((c) => {
        const fn = resolve(table, c.params)
        const incoming = etaOf(arg(c, 0))
        if (arg(c, 1).family.kind === 'gamma') {
          if (c.to === 1) {
            if (isFlat(incoming)) return { flat: true }
            return { moments: tiltGamma(fn, arg(c, 1), incoming, c.power).x }
          }
          return tiltGamma(fn, arg(c, 1), incoming, c.power).y
        }
        const x = gaussianOf(arg(c, 1))
        if (c.to === 1) {
          // Nothing is known about y: the constraint says nothing about x.
          if (isFlat(incoming)) return { flat: true }
          return tilt(fn, x, incoming, c.power).x
        }
        if (isFlat(incoming) && fn.pushforward !== undefined && c.power === 1) {
          const p = fn.pushforward(x.mean, x.variance)
          return { mean: p.mean, variance: Math.max(p.variance, FLOOR) }
        }
        return tilt(fn, x, incoming, c.power).y
      }),
      // Deterministic: under the mean field `y` is defined by `x` and its belief is the Gaussian with the mean and variance of the
      // pushforward (closed form where the function has one, else quadrature), which is all a Gaussian reader of `y` sees; the message to
      // `x` is the non-conjugate message of `E[h f(x) - J f(x)^2 / 2]`, `(h, J)` the cavity of `y`. A reader that needs more than the
      // two moments of `y` (a gamma, a Poisson) sees a Gaussian approximation of it.
      vmp: messageRule((c): OperatorResult => {
        const fn = resolve(table, c.params)
        if (c.to === 0 && fn.pushforward !== undefined && arg(c, 1).family.kind === 'gaussian') {
          const x = gaussianOf(arg(c, 1))
          const p = fn.pushforward(x.mean, x.variance)
          return { message: gaussianMessage(p.mean, Math.max(p.variance, FLOOR)) }
        }
        return { message: definedFunctionMessage(c.args, c.to, (x) => fn.f(x[0] as number)) }
      }, 'numeric'),
    },
    evidence: {
      ep: (c) => {
        const incoming = etaOf(arg(c, 0))
        // An output nothing else reads (a flat cavity) integrates to one: the evidence is the mass of the input's cavity.
        if (isFlat(incoming)) return allProper([arg(c, 1)]) ? massOf(arg(c, 1)) : Number.NaN
        if (!allProper(c.args)) return Number.NaN
        const fn = resolve(table, c.params)
        if (arg(c, 1).family.kind === 'gamma')
          return massOf(arg(c, 1)) + tiltGamma(fn, arg(c, 1), incoming, c.power).logMass
        return massOf(arg(c, 1)) + tilt(fn, gaussianOf(arg(c, 1)), incoming, c.power).logMass
      },
      vmp: () => 0,
    },
    // The message to x is a match against x's own cavity; the message to y needs none (a flat y is told the pushforward).
    traits: { readsTarget: (to) => to === 1, power: true, defines: 0 },
  })
}

/**
 * The factor `y = f(x)` of the library's unary functions: the operator `apply`, the function named by its parameter `fn`.
 *
 * @concept factor-operators
 * @index structure The factor y = f(x) of a unary function of a Gaussian (exp, square, tanh, relu, softplus, abs, sigmoid).
 */
export const apply: FactorOperator = applyFunction('apply')

/** The conditional of `z` given `x` under `y = z f(x)`: the cavity of `z` times the output's message, a Gaussian in `z`, with its log mass. */
function scaledConditional(
  z: { readonly mean: number; readonly variance: number },
  y: Eta,
  u: number,
  power: number,
): { readonly logMass: number; readonly mean: number; readonly variance: number } | undefined {
  if (!Number.isFinite(u)) return undefined
  const p0 = 1 / z.variance
  const h0 = z.mean / z.variance
  const precision = p0 + power * (y[1] ?? 0) * u * u
  if (!(precision > 0)) return undefined
  const linear = h0 + power * (y[0] ?? 0) * u
  return {
    logMass: 0.5 * Math.log(p0 / precision) + 0.5 * ((linear * linear) / precision - (h0 * h0) / p0),
    mean: linear / precision,
    variance: 1 / precision,
  }
}

/** The mean and variance of an expectation over the tilted input (what `tiltedByQuadrature` returns for each extra function). */
type Extra = { readonly mean: number; readonly variance: number }

/** What one integral over `x` gives for `y = z f(x)`: the log mass, the tilted moments of `x` and the expectations that become the moments of `z` and `y`. */
interface ScaledTilt {
  readonly logMass: number
  readonly x: { readonly mean: number; readonly variance: number; readonly logMean: number }
  readonly z: { readonly mean: number; readonly variance: number }
  readonly y: { readonly mean: number; readonly variance: number }
}

function scaledTilt(fn: UnaryFunction, c: OperatorContext): ScaledTilt {
  const y = etaOf(arg(c, 0))
  const z = gaussianOf(arg(c, 1))
  const input = arg(c, 2)
  const conditionalAt = (x: number): { conditional: ReturnType<typeof scaledConditional>; u: number } => {
    const u = fn.f(x)
    return { conditional: scaledConditional(z, y, u, c.power), u }
  }
  const logf = (x: number): number => conditionalAt(x).conditional?.logMass ?? Number.NEGATIVE_INFINITY
  const expectation =
    (pick: (mean: number, variance: number, u: number) => number) =>
    (x: number): number => {
      const { conditional, u } = conditionalAt(x)
      return conditional === undefined ? 0 : pick(conditional.mean, conditional.variance, u)
    }
  const zMean = expectation((m) => m)
  const zVariance = expectation((_m, v) => v)
  const yMean = expectation((m, _v, u) => u * m)
  const yVariance = expectation((_m, v, u) => u * u * v)
  if (input.family.kind === 'gamma') {
    const port = quadraturePort(input, { points: 481 })
    const n = port.coords.length
    const lw = Float64Array.from(port.coords, (x, i) => Math.log(port.weights[i] as number) + logf(x))
    const { top, weights, total: m0 } = shiftedSum(lw)
    if (!Number.isFinite(top))
      throw new ImproperMessageError('apply-product: the factor has no mass against these cavities')
    const sums = { logX: 0, x: 0 }
    // The conditional mean and variance of z and of y at each node that carries weight.
    const nodes = { zm: new Float64Array(n), zv: new Float64Array(n), ym: new Float64Array(n), yv: new Float64Array(n) }
    for (let i = 0; i < n; i++) {
      const w = weights[i] as number
      if (w === 0) continue
      const x = port.coords[i] as number
      sums.logX += w * (port.stats[i]?.[0] ?? 0)
      sums.x += w * (port.stats[i]?.[1] ?? 0)
      nodes.zm[i] = zMean(x)
      nodes.zv[i] = zVariance(x)
      nodes.ym[i] = yMean(x)
      nodes.yv[i] = yVariance(x)
    }
    // The mixture over the nodes: the mean of the conditional variances plus the variance of the conditional means, centred.
    const mixture = (means: Float64Array, variances: Float64Array): { mean: number; variance: number } => {
      const m = centredMoments(weights, m0, means)
      let within = 0
      for (let i = 0; i < n; i++) within += (weights[i] as number) * (variances[i] as number)
      return { mean: m.mean, variance: within / m0 + m.variance }
    }
    return {
      logMass: top + Math.log(m0),
      x: { mean: sums.x / m0, variance: Number.NaN, logMean: sums.logX / m0 },
      z: mixture(nodes.zm, nodes.zv),
      y: mixture(nodes.ym, nodes.yv),
    }
  }
  const t = tiltedByQuadrature(gaussianOf(input), logf, undefined, undefined, [zMean, zVariance, yMean, yVariance])
  const [zm, zv, ym, yv] = t.extra as [Extra, Extra, Extra, Extra]
  return {
    logMass: t.logMass,
    x: { mean: t.mean, variance: resolvedVariance(t.variance, 'apply-product'), logMean: Number.NaN },
    z: { mean: zm.mean, variance: zv.mean + zm.variance },
    y: { mean: ym.mean, variance: yv.mean + ym.variance },
  }
}

/**
 * The factor `y = z f(x)`: a Gaussian `z` times a function of one other variable `x` (a Gaussian, or a gamma variable), the
 * function a registry function or a chain of steps as in {@link applyFunction}. It is the scale of a non-centred coefficient,
 * `beta = z / sqrt(precision(x))` with `z` standard normal and the precision a known map of a log smoothing parameter (or of a
 * gamma relevance): the product of `z` with the output of a unary function, fused so that nothing is approximated by a Gaussian
 * on the way. Given `x` the factor is linear in `z`, so against the cavity of `z` and the message of `y` the integral over `z`
 * is closed form and what is left is one one-dimensional integral over `x` (a panelled grid for a Gaussian, a grid in `log x` for a
 * gamma): the message to `x` is its moment match, the message to `z` is the moment match of the mixture of the conditional
 * Gaussians over it, and the message to `y` that of `u z` with `u = f(x)`.
 *
 * @param name - The operator's name.
 * @param table - The functions to choose from (default {@link unaryFunctions}).
 * @returns The operator (expectation propagation only).
 * @concept factor-operators
 * @index function The factor y = z f(x) of a Gaussian times a function of a Gaussian or gamma variable: a non-centred scale.
 * @example
 * ```ts
 * import { applyProduct, gaussian, gaussianEta, messageArg } from 'omai-engine/message-passing'
 * const op = applyProduct('apply-product')
 * // y = z / sqrt(x) with z ~ N(0, 1) and x ~ N(4, 0.01): told nothing about y, the message to y is about N(0, 1 / 4).
 * const args = [gaussian.flat, gaussianEta(0, 1), gaussianEta(4, 0.01)].map((eta) => messageArg(gaussian, eta))
 * op.ep?.({ mode: 'ep', to: 0, args, params: { steps: [['sqrt'], ['recip']] }, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.000', '4.000']
 * ```
 */
export function applyProduct(
  name: string,
  table: Readonly<Record<string, UnaryFunction>> = unaryFunctions(),
): FactorOperator {
  return defineOperator({
    name,
    description: 'y = z f(x) for a Gaussian z and a function of a Gaussian or gamma variable x',
    params: {
      fn: param.text({ optional: true, choices: Object.keys(table) }),
      steps: param.any({ optional: true }),
    },
    ports: [
      { name: 'y', accepts: ['gaussian'] },
      { name: 'z', accepts: ['gaussian'] },
      { name: 'x', accepts: ['gaussian', 'gamma'] },
    ],
    check: (params) => {
      try {
        resolve(table, params)
        return undefined
      } catch (e) {
        return errorText(e)
      }
    },
    // Expectation propagation only: the non-conjugate variational rule of a product with a scale (`z f(x)`, a funnel) is degenerate
    // (measured on the Bayesian GAM: the noise collapses to a point), so the verdict of variational message passing says so.
    rules: {
      ep: siteRule((c) => {
        const fn = resolve(table, c.params)
        const incoming = etaOf(arg(c, 0))
        if (isFlat(incoming) && c.to !== 0) return { flat: true }
        const t = scaledTilt(fn, c)
        if (c.to === 2) {
          const input = arg(c, 2)
          return input.family.kind === 'gamma' ? { moments: [t.x.logMean, t.x.mean] } : t.x
        }
        const moments = c.to === 1 ? t.z : t.y
        return { mean: moments.mean, variance: resolvedVariance(moments.variance, 'apply-product') }
      }),
    },
    evidence: {
      ep: (c) => {
        if (!allProper(c.args)) return Number.NaN
        const fn = resolve(table, c.params)
        return massOf(arg(c, 1)) + massOf(arg(c, 2)) + scaledTilt(fn, c).logMass
      },
    },
    traits: { readsTarget: (to) => to !== 0, power: true },
  })
}

/**
 * The factor `y = z f(x)` of a Gaussian times a function of a Gaussian or gamma variable: the operator `apply-product`.
 *
 * @concept factor-operators
 * @index structure The factor y = z f(x): a Gaussian times a function of a Gaussian or gamma variable (a non-centred scale).
 */
export const applyProductOperator: FactorOperator = applyProduct('apply-product')
