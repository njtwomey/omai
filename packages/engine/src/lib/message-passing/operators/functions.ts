/**
 * Unary functions of a Gaussian variable as an operator: `y = f(x)` for `exp`, `square`, `tanh`, `relu`, `softplus`, `abs`
 * and `sigmoid`, the factor behind a log-normal (`exp` of a Gaussian), a banana (`square`), a Bayesian neural network
 * (`tanh`, `relu`: probabilistic backpropagation) and the non-centred funnel. The output is matched by moments under the
 * input's cavity (the pushforward of the input, weighted by what the output has been told), and the message back to the input
 * is the moment match of the tilted input: one integral over `x` gives both. The rectifier has a closed form (two truncated
 * Gaussians); the pushforward of a flat output is closed form for `exp`, `square`, `relu` and `abs`; the others are the
 * panelled quadrature of `_tilted.ts`, split at the function's kinks. Expectation propagation only: a deterministic
 * non-affine factor has no finite mean-field message.
 */

import { logAddExp, OmaiError, sigmoidOf } from '../../base/index.ts'
import { Phi, phi } from '../_scalar.ts'
import { tiltedByQuadrature, type WindowHint } from '../_tilted.ts'
import { truncatedNormalMoments } from '../truncated.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, ImproperMessageError, massOf } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

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

const softplusOf = (x: number): number => logAddExp(0, x)

const unary = (u: UnaryFunction): UnaryFunction => u

/**
 * The registry of unary functions by name: `exp`, `square`, `tanh`, `relu`, `softplus`, `abs`, `sigmoid`. A new object each call.
 *
 * @returns The functions, keyed by name.
 * @concept factor-operators
 * @index table The unary functions of a Gaussian variable (exp, square, tanh, relu, softplus, abs, sigmoid) by name.
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
  ]
  return Object.fromEntries(all.map((u) => [u.name, u]))
}

const isFlat = (eta: Eta): boolean => (eta[0] ?? 0) === 0 && (eta[1] ?? 0) === 0

/** The function a factor applies, by its parameter `fn`. */
function resolve(
  table: Readonly<Record<string, UnaryFunction>>,
  params: Readonly<Record<string, unknown>>,
): UnaryFunction {
  const key = params['fn']
  const fn = typeof key === 'string' ? table[key] : undefined
  if (fn === undefined)
    throw new OmaiError(
      'value',
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
  const top = Math.max(logA, logB)
  const total = Math.exp(logA - top) + Math.exp(logB - top)
  const wA = Math.exp(logA - top) / total
  const wB = 1 - wA
  const xMean = wA * below.mean + wB * above.mean
  const xSecond = wA * (below.variance + below.mean ** 2) + wB * (above.variance + above.mean ** 2)
  const yMean = wB * above.mean
  const ySecond = wB * (above.variance + above.mean ** 2)
  return {
    logMass: top + Math.log(total),
    x: { mean: xMean, variance: Math.max(xSecond - xMean * xMean, FLOOR) },
    y: { mean: yMean, variance: Math.max(ySecond - yMean * yMean, FLOOR) },
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
  const hint: WindowHint | undefined =
    fn.kinks === undefined
      ? undefined
      : { centre: cavity.mean, scale: Math.sqrt(cavity.variance), reach: 9, kinks: fn.kinks }
  const t = tiltedByQuadrature(cavity, logf, hint, f)
  return {
    logMass: t.logMass,
    x: { mean: t.mean, variance: Math.max(t.variance, FLOOR) },
    y: { mean: t.fnMean, variance: Math.max(t.fnVariance, FLOOR) },
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
  return {
    name,
    description: 'y = f(x) for a unary function of a Gaussian variable',
    ports: [
      { name: 'y', accepts: ['gaussian'] },
      { name: 'x', accepts: ['gaussian'] },
    ],
    support: { ep: 'numeric' },
    // The message to x is a match against x's own cavity; the message to y needs none (a flat y is told the pushforward).
    readsTarget: (to) => to === 1,
    power: true,
    noRule: {
      vmp: 'a deterministic non-affine function has no finite mean-field message (use expectation propagation)',
    },
    check: (params) => {
      try {
        resolve(table, params)
        return undefined
      } catch (e) {
        return e instanceof Error ? e.message : String(e)
      }
    },
    ep: (c): OperatorResult => {
      const fn = resolve(table, c.params)
      const incoming = etaOf(arg(c, 0))
      const x = gaussianOf(arg(c, 1))
      if (c.to === 1) {
        // Nothing is known about y: the constraint says nothing about x.
        if (isFlat(incoming)) return { message: [0, 0] }
        const t = tilt(fn, x, incoming, c.power)
        const site = gaussianSiteMessage(etaOf(arg(c, 1)), t.x.mean, t.x.variance)
        return { message: c.power === 1 ? site : site.map((v) => v / c.power) }
      }
      if (isFlat(incoming) && fn.pushforward !== undefined && c.power === 1) {
        const p = fn.pushforward(x.mean, x.variance)
        return { message: gaussianSiteMessage(incoming, p.mean, Math.max(p.variance, FLOOR)) }
      }
      const t = tilt(fn, x, incoming, c.power)
      const site = gaussianSiteMessage(incoming, t.y.mean, t.y.variance)
      return { message: c.power === 1 ? site : site.map((v) => v / c.power) }
    },
    logEvidence: (c) => {
      const incoming = etaOf(arg(c, 0))
      // An output nothing else reads (a flat cavity) integrates to one: the evidence is the mass of the input's cavity.
      if (isFlat(incoming)) return allProper([arg(c, 1)]) ? massOf(arg(c, 1)) : Number.NaN
      if (!allProper(c.args)) return Number.NaN
      const fn = resolve(table, c.params)
      return massOf(arg(c, 1)) + tilt(fn, gaussianOf(arg(c, 1)), incoming, c.power).logMass
    },
  }
}

/**
 * The factor `y = f(x)` of the library's unary functions: the operator `apply`, the function named by its parameter `fn`.
 *
 * @concept factor-operators
 * @index structure The factor y = f(x) of a unary function of a Gaussian (exp, square, tanh, relu, softplus, abs, sigmoid).
 */
export const apply: FactorOperator = applyFunction('apply')
