/**
 * Moving between equations and transfer functions: difference and differential equations to and from transfer functions
 * (with LaTeX of each equation), and inverse Laplace and z-transforms of rational functions by partial fractions
 * (scipy.signal residue and residuez), which give the time response term by term.
 */

import { MdmlError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { type PartialFractions, residue, residuez } from '../numerics/polynomial/index.ts'
import { pad, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { choose, factorial, float, numbers, toList, trimLeading } from './_shared.ts'
import { isDiscrete, type Lti, type TransferFunction, tf, toTf } from './representations.ts'

// Equations and transforms. A difference equation a0 y[n] + a1 y[n-1] + ... = b0 x[n] + b1 x[n-1] + ... has the
// transfer function B(z^-1) / A(z^-1); a differential equation a_n y^(n) + ... + a_0 y = b_m u^(m) + ... + b_0 u has
// B(s) / A(s). Coefficients are listed highest derivative (or no delay) first, as scipy.signal's lfilter and lti.

/** Equal-length coefficient arrays: pad the shorter at the end (in delays) or the front (in powers of z). */
function equalLength(b: Value, a: Value, atEnd: boolean): [Value, Value] {
  const n = Math.max(b.shape[0] ?? 0, a.shape[0] ?? 0)
  const fit = (v: Value): Value => {
    const k = n - (v.shape[0] ?? 0)
    return k === 0 ? v : pad(v, [atEnd ? [0, k] : [k, 0]])
  }
  return [fit(b), fit(a)]
}

/**
 * The transfer function of a difference equation a0 y[n] + a1 y[n-1] + ... = b0 x[n] + b1 x[n-1] + ... (the lfilter
 * coefficients b and a): H(z) = B(z^-1) / A(z^-1), as a discrete system in powers of z (scipy's dlti from (b, a)).
 * Differentiable in the coefficients.
 *
 * @param b - Input coefficients b0, b1, ... (of x[n], x[n-1], ...).
 * @param a - Output coefficients a0, a1, ... (of y[n], y[n-1], ...), a0 != 0.
 * @param options - The sample time (default 1).
 * @returns The transfer function.
 * @concept difference-equations
 * @example
 * ```ts
 * import { differenceToTf } from 'mdml-engine/systems'
 * import { toArray } from 'mdml-engine/tensor'
 * // y[n] - 0.5 y[n-1] = x[n-1]  ->  H(z) = z^-1 / (1 - 0.5 z^-1) = 1 / (z - 0.5).
 * toArray(differenceToTf([0, 1], [1, -0.5]).num) // => [1]
 * ```
 */
export function differenceToTf(b: TensorLike, a: TensorLike, options: { readonly dt?: number } = {}): TransferFunction {
  const [bv, av] = equalLength(float(b), float(a), true)
  return tf(bv, av, { dt: options.dt ?? 1 })
}

/** A difference or differential equation: its coefficients and how it reads. */
export interface Equation {
  /** Input coefficients (of x[n], x[n-1], ... or u^(m), ..., u). */
  readonly b: Value
  /** Output coefficients (of y[n], y[n-1], ... or y^(n), ..., y), the first normalised to 1. */
  readonly a: Value
  /** The equation as LaTeX. */
  readonly latex: string
}

const fmt = (v: number): string => String(Math.round(v * 1e6) / 1e6)

function sumLatex(terms: readonly (readonly [number, string])[]): string {
  const kept = terms.filter(([c]) => c !== 0)
  if (kept.length === 0) return '0'
  return kept
    .map(([c, sym], i) => {
      const mag = Math.abs(c)
      const coef = mag === 1 && sym !== '' ? '' : fmt(mag)
      const sign = c < 0 ? (i === 0 ? '-' : ' - ') : i === 0 ? '' : ' + '
      return `${sign}${coef}${coef !== '' && sym !== '' ? ' ' : ''}${sym}`
    })
    .join('')
}

/**
 * The difference equation of a discrete transfer function: the lfilter coefficients b and a (a0 = 1, delays from
 * the positive powers of z) and its LaTeX, y[n] = sum b_k x[n-k] - sum_{k>0} a_k y[n-k].
 *
 * @param sys - A discrete system (single input and output).
 * @returns The coefficients and the equation.
 * @throws MdmlError with code `value` for a continuous system.
 * @concept difference-equations
 * @example
 * ```ts
 * import { tf, tfToDifference } from 'mdml-engine/systems'
 * tfToDifference(tf([1], [1, -0.5], { dt: 1 })).latex // => 'y[n] = 0.5 y[n-1] + x[n-1]'
 * ```
 */
export function tfToDifference(sys: Lti): Equation {
  if (!isDiscrete(sys)) throw new MdmlError('value', 'tfToDifference: the system must be discrete')
  const g = toTf(sys)
  if (g.num.shape.length !== 1) throw new MdmlError('shape', 'tfToDifference: one output is needed')
  const [b, a] = equalLength(g.num, g.den, false)
  const bn = numbers(b)
  const an = numbers(a)
  const terms: [number, string][] = [
    ...an.slice(1).map((v, k): [number, string] => [-v, `y[n-${k + 1}]`]),
    ...bn.map((v, k): [number, string] => [v, k === 0 ? 'x[n]' : `x[n-${k}]`]),
  ]
  return { b, a, latex: `y[n] = ${sumLatex(terms)}` }
}

/**
 * The transfer function of a linear differential equation a_n y^(n) + ... + a_0 y = b_m u^(m) + ... + b_0 u:
 * H(s) = B(s) / A(s). Differentiable in the coefficients.
 *
 * @param b - Input coefficients, highest derivative first.
 * @param a - Output coefficients, highest derivative first.
 * @returns The continuous transfer function.
 * @concept differential-equations
 * @example
 * ```ts
 * import { differentialToTf } from 'mdml-engine/systems'
 * import { toArray } from 'mdml-engine/tensor'
 * // y'' + 3 y' + 2 y = u.
 * toArray(differentialToTf([1], [1, 3, 2]).den) // => [1, 3, 2]
 * ```
 */
export function differentialToTf(b: TensorLike, a: TensorLike): TransferFunction {
  return tf(b, a)
}

const derivative = (symbol: string, order: number): string =>
  order === 0 ? symbol : order <= 3 ? `${symbol}${"'".repeat(order)}` : `${symbol}^{(${order})}`

/**
 * The differential equation of a continuous transfer function: coefficients (a_n = 1) and its LaTeX.
 *
 * @param sys - A continuous system (single input and output).
 * @returns The coefficients and the equation.
 * @throws MdmlError with code `value` for a discrete system.
 * @concept differential-equations
 * @example
 * ```ts
 * import { tf, tfToDifferential } from 'mdml-engine/systems'
 * tfToDifferential(tf([2], [1, 3, 2])).latex // => "y'' + 3 y' + 2 y = 2 u"
 * ```
 */
export function tfToDifferential(sys: Lti): Equation {
  if (isDiscrete(sys)) throw new MdmlError('value', 'tfToDifferential: the system must be continuous')
  const g = toTf(sys)
  if (g.num.shape.length !== 1) throw new MdmlError('shape', 'tfToDifferential: one output is needed')
  const bn = trimLeading(numbers(g.num))
  const an = numbers(g.den)
  const side = (c: readonly number[], symbol: string): string =>
    sumLatex(c.map((v, k): [number, string] => [v, derivative(symbol, c.length - 1 - k)]))
  return { b: g.num, a: g.den, latex: `${side(an, 'y')} = ${side(bn, 'u')}` }
}

/** One partial-fraction term r t^(k-1) / (k-1)! e^{pt} (or its sequence counterpart). */
export interface InverseTerm {
  readonly residue: ComplexNumber
  readonly pole: ComplexNumber
  readonly power: number
}

/** An inverse transform by partial fractions. */
export interface InverseTransform {
  readonly terms: readonly InverseTerm[]
  /** Direct terms: coefficients of delta, delta', ... (Laplace) or of delta[n], delta[n-1], ... (z). */
  readonly direct: readonly number[]
  /** The time function (t >= 0), or the sequence value at sample n. */
  readonly at: (t: number) => number
}

function termsOf(pf: PartialFractions): InverseTerm[] {
  const ps = toList({ re: toFlat(pf.poles.re), im: toFlat(pf.poles.im) })
  const rs = toList({ re: toFlat(pf.residues.re), im: toFlat(pf.residues.im) })
  return ps.map((pole, i) => {
    let power = 1
    for (let j = i - 1; j >= 0; j--) {
      const q = ps[j] as ComplexNumber
      if (q.re !== pole.re || q.im !== pole.im) break
      power++
    }
    return { residue: rs[i] ?? C.of(0), pole, power }
  })
}

/**
 * The inverse Laplace transform of a rational function by partial fractions (scipy.signal.residue):
 * b(s) / a(s) = sum r / (s - p)^k + direct, so f(t) = sum r t^(k-1) / (k-1)! e^{pt} for t >= 0 (plus impulses for the
 * direct part, reported but not evaluated). For a strictly proper system this is its impulse response.
 *
 * @param num - Numerator, highest degree first.
 * @param den - Denominator, highest degree first.
 * @returns The terms, the direct part and the time function.
 * @concept laplace-transform
 * @example
 * ```ts
 * import { inverseLaplace } from 'mdml-engine/systems'
 * // 1 / (s + 1)  ->  e^{-t}.
 * inverseLaplace([1], [1, 1]).at(1) // => 0.36787944117144233
 * ```
 */
export function inverseLaplace(num: TensorLike, den: TensorLike): InverseTransform {
  const pf = residue(numbers(float(num)), numbers(float(den)))
  const terms = termsOf(pf)
  const at = (t: number): number => {
    let acc = C.of(0)
    for (const { residue: r, pole, power } of terms) {
      const scale = t ** (power - 1) / factorial(power - 1)
      acc = C.add(acc, C.mul(C.mul(r, C.exp(C.mul(pole, t))), scale))
    }
    return acc.re
  }
  return { terms, direct: toFlat(pf.direct), at }
}

/**
 * The inverse z-transform of a rational function in z^-1 by partial fractions (scipy.signal.residuez):
 * b / a = sum r / (1 - p z^-1)^k + sum_i k_i z^-i, so x[n] = sum r C(n + k - 1, k - 1) p^n + k_n. For lfilter
 * coefficients this is the impulse response.
 *
 * @param b - Numerator coefficients of z^0, z^-1, ...
 * @param a - Denominator coefficients of z^0, z^-1, ...
 * @returns The terms, the direct part and the sequence.
 * @concept z-transform
 * @example
 * ```ts
 * import { inverseZ } from 'mdml-engine/systems'
 * // 1 / (1 - 0.5 z^-1)  ->  0.5^n.
 * inverseZ([1], [1, -0.5]).at(3) // => 0.125
 * ```
 */
export function inverseZ(b: TensorLike, a: TensorLike): InverseTransform {
  const pf = residuez(numbers(float(b)), numbers(float(a)))
  const direct = toFlat(pf.direct)
  const terms = termsOf(pf)
  const at = (n: number): number => {
    if (!Number.isInteger(n) || n < 0) return 0
    let acc = C.of(direct[n] ?? 0)
    for (const { residue: r, pole, power } of terms) {
      let pn = C.of(1)
      for (let i = 0; i < n; i++) pn = C.mul(pn, pole)
      acc = C.add(acc, C.mul(C.mul(r, pn), choose(n + power - 1, power - 1)))
    }
    return acc.re
  }
  return { terms, direct, at }
}
