/**
 * Tables of standard Laplace and z-transform pairs (a time function or sequence with its rational transform), as data
 * with evaluators and written-out forms, and `pairToTf`, which turns a pair into a transfer function whose impulse
 * response is the time function, so the tables can be checked by simulation.
 */

import { choose, factorial, polyMulHf } from './_shared.ts'
import { type TransferFunction, tf } from './representations.ts'

// Laplace and z-transform pairs as data: each pair has its time function, its rational transform and both written
// out in LaTeX. TODO: derive them with `symbolic` once that domain exists.

/** A transform pair: the time function, its transform as a rational function, and both written out. */
export interface TransformPair {
  readonly name: string
  /** The parameters the pair takes (a, omega, n), with defaults. */
  readonly params: Readonly<Record<string, number>>
  /** f(t) for t >= 0 (Laplace), or x[n] for n >= 0 (z), as LaTeX. */
  readonly time: string
  /** F(s) or X(z) as LaTeX. */
  readonly transform: string
  /** The region of convergence. */
  readonly roc: string
  /** The time function or sequence. */
  readonly f: (t: number, params?: Readonly<Record<string, number>>) => number
  /** The transform's numerator and denominator (highest power first). */
  readonly rational: (params?: Readonly<Record<string, number>>) => { readonly num: number[]; readonly den: number[] }
}

const p = <T extends Record<string, number>>(params: Readonly<Record<string, number>> | undefined, defaults: T): T =>
  ({ ...defaults, ...(params ?? {}) }) as T

/** A factor raised to a power (highest first). */
function powerOf(base: readonly number[], k: number): number[] {
  let out = [1]
  for (let i = 0; i < k; i++) out = polyMulHf(out, base)
  return out
}

/**
 * Standard Laplace transform pairs (f(t) for t >= 0 and F(s)), as data with evaluators and rational forms; with
 * {@link inverseLaplace} each rational form inverts back to its time function.
 *
 * @concept laplace-transform
 * @example
 * ```ts
 * import { laplacePairs } from 'mdml-engine/systems'
 * laplacePairs.find((q) => q.name === 'sine')?.rational({ omega: 2 }) // => { num: [2], den: [1, 0, 4] }
 * ```
 */
export const laplacePairs: readonly TransformPair[] = [
  {
    name: 'impulse',
    params: {},
    time: '\\delta(t)',
    transform: '1',
    roc: 'all s',
    f: () => 0,
    rational: () => ({ num: [1], den: [1] }),
  },
  {
    name: 'step',
    params: {},
    time: 'u(t)',
    transform: '\\frac{1}{s}',
    roc: '\\operatorname{Re} s > 0',
    f: () => 1,
    rational: () => ({ num: [1], den: [1, 0] }),
  },
  {
    name: 'power',
    params: { n: 1 },
    time: '\\frac{t^n}{n!}',
    transform: '\\frac{1}{s^{n+1}}',
    roc: '\\operatorname{Re} s > 0',
    f: (t, q) => {
      const { n } = p(q, { n: 1 })
      return t ** n / factorial(n)
    },
    rational: (q) => ({ num: [1], den: powerOf([1, 0], p(q, { n: 1 }).n + 1) }),
  },
  {
    name: 'exponential',
    params: { a: 1, n: 0 },
    time: '\\frac{t^n}{n!} e^{-a t}',
    transform: '\\frac{1}{(s + a)^{n+1}}',
    roc: '\\operatorname{Re} s > -a',
    f: (t, q) => {
      const { a, n } = p(q, { a: 1, n: 0 })
      return (t ** n / factorial(n)) * Math.exp(-a * t)
    },
    rational: (q) => {
      const { a, n } = p(q, { a: 1, n: 0 })
      return { num: [1], den: powerOf([1, a], n + 1) }
    },
  },
  {
    name: 'sine',
    params: { a: 0, omega: 1 },
    time: 'e^{-a t} \\sin \\omega t',
    transform: '\\frac{\\omega}{(s + a)^2 + \\omega^2}',
    roc: '\\operatorname{Re} s > -a',
    f: (t, q) => {
      const { a, omega } = p(q, { a: 0, omega: 1 })
      return Math.exp(-a * t) * Math.sin(omega * t)
    },
    rational: (q) => {
      const { a, omega } = p(q, { a: 0, omega: 1 })
      return { num: [omega], den: [1, 2 * a, a * a + omega * omega] }
    },
  },
  {
    name: 'cosine',
    params: { a: 0, omega: 1 },
    time: 'e^{-a t} \\cos \\omega t',
    transform: '\\frac{s + a}{(s + a)^2 + \\omega^2}',
    roc: '\\operatorname{Re} s > -a',
    f: (t, q) => {
      const { a, omega } = p(q, { a: 0, omega: 1 })
      return Math.exp(-a * t) * Math.cos(omega * t)
    },
    rational: (q) => {
      const { a, omega } = p(q, { a: 0, omega: 1 })
      return { num: [1, a], den: [1, 2 * a, a * a + omega * omega] }
    },
  },
  {
    name: 'first-order-step',
    params: { a: 1 },
    time: '1 - e^{-a t}',
    transform: '\\frac{a}{s (s + a)}',
    roc: '\\operatorname{Re} s > 0',
    f: (t, q) => 1 - Math.exp(-p(q, { a: 1 }).a * t),
    rational: (q) => {
      const { a } = p(q, { a: 1 })
      return { num: [a], den: [1, a, 0] }
    },
  },
  {
    name: 'hyperbolic-sine',
    params: { omega: 1 },
    time: '\\sinh \\omega t',
    transform: '\\frac{\\omega}{s^2 - \\omega^2}',
    roc: '\\operatorname{Re} s > |\\omega|',
    f: (t, q) => Math.sinh(p(q, { omega: 1 }).omega * t),
    rational: (q) => {
      const { omega } = p(q, { omega: 1 })
      return { num: [omega], den: [1, 0, -omega * omega] }
    },
  },
  {
    name: 'hyperbolic-cosine',
    params: { omega: 1 },
    time: '\\cosh \\omega t',
    transform: '\\frac{s}{s^2 - \\omega^2}',
    roc: '\\operatorname{Re} s > |\\omega|',
    f: (t, q) => Math.cosh(p(q, { omega: 1 }).omega * t),
    rational: (q) => {
      const { omega } = p(q, { omega: 1 })
      return { num: [1, 0], den: [1, 0, -omega * omega] }
    },
  },
]

/**
 * Standard z-transform pairs (x[n] for n >= 0 and X(z)), as data with evaluators and rational forms in powers of z;
 * with {@link inverseZ} on the coefficients in z^-1 each inverts back to its sequence.
 *
 * @concept z-transform
 * @example
 * ```ts
 * import { zTransformPairs } from 'mdml-engine/systems'
 * zTransformPairs.find((q) => q.name === 'geometric')?.f(3, { a: 0.5 }) // => 0.125
 * ```
 */
export const zTransformPairs: readonly TransformPair[] = [
  {
    name: 'impulse',
    params: {},
    time: '\\delta[n]',
    transform: '1',
    roc: 'all z',
    f: (n) => (n === 0 ? 1 : 0),
    rational: () => ({ num: [1], den: [1] }),
  },
  {
    name: 'step',
    params: {},
    time: 'u[n]',
    transform: '\\frac{z}{z - 1}',
    roc: '|z| > 1',
    f: () => 1,
    rational: () => ({ num: [1, 0], den: [1, -1] }),
  },
  {
    name: 'ramp',
    params: {},
    time: 'n',
    transform: '\\frac{z}{(z - 1)^2}',
    roc: '|z| > 1',
    f: (n) => n,
    rational: () => ({ num: [0, 1, 0], den: [1, -2, 1] }),
  },
  {
    name: 'geometric',
    params: { a: 0.5, k: 0 },
    time: '\\binom{n + k}{k} a^n',
    transform: '\\frac{z^{k+1}}{(z - a)^{k+1}}',
    roc: '|z| > |a|',
    f: (n, q) => {
      const { a, k } = p(q, { a: 0.5, k: 0 })
      return choose(n + k, k) * a ** n
    },
    rational: (q) => {
      const { a, k } = p(q, { a: 0.5, k: 0 })
      return { num: [1, ...new Array<number>(k + 1).fill(0)], den: powerOf([1, -a], k + 1) }
    },
  },
  {
    name: 'sine',
    params: { a: 1, omega: 1 },
    time: 'a^n \\sin \\omega n',
    transform: '\\frac{a z \\sin \\omega}{z^2 - 2 a z \\cos \\omega + a^2}',
    roc: '|z| > |a|',
    f: (n, q) => {
      const { a, omega } = p(q, { a: 1, omega: 1 })
      return a ** n * Math.sin(omega * n)
    },
    rational: (q) => {
      const { a, omega } = p(q, { a: 1, omega: 1 })
      return { num: [0, a * Math.sin(omega), 0], den: [1, -2 * a * Math.cos(omega), a * a] }
    },
  },
  {
    name: 'cosine',
    params: { a: 1, omega: 1 },
    time: 'a^n \\cos \\omega n',
    transform: '\\frac{z (z - a \\cos \\omega)}{z^2 - 2 a z \\cos \\omega + a^2}',
    roc: '|z| > |a|',
    f: (n, q) => {
      const { a, omega } = p(q, { a: 1, omega: 1 })
      return a ** n * Math.cos(omega * n)
    },
    rational: (q) => {
      const { a, omega } = p(q, { a: 1, omega: 1 })
      return { num: [1, -a * Math.cos(omega), 0], den: [1, -2 * a * Math.cos(omega), a * a] }
    },
  },
]

/**
 * The transfer function of a transform pair's rational form, to simulate it: the impulse response of a Laplace pair's
 * system is its time function, and of a z pair's system (with dt 1) its sequence.
 *
 * @param pair - A pair from {@link laplacePairs} or {@link zTransformPairs}.
 * @param options - Sample time (give 1 for z pairs) and the pair's parameters.
 * @returns The transfer function.
 * @example
 * ```ts
 * import { laplacePairs, pairToTf } from 'mdml-engine/systems'
 * import { toArray } from 'mdml-engine/tensor'
 * const step = laplacePairs.find((q) => q.name === 'step')
 * step === undefined ? [] : toArray(pairToTf(step).den) // => [1, 0]
 * ```
 */
export function pairToTf(
  pair: TransformPair,
  options: { readonly dt?: number; readonly params?: Readonly<Record<string, number>> } = {},
): TransferFunction {
  const { num, den } = pair.rational(options.params)
  return tf(num, den, options.dt === undefined ? {} : { dt: options.dt })
}
