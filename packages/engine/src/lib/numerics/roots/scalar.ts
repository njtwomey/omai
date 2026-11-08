/**
 * Scalar root finding, scipy.optimize style: bracketing methods (bisection, Brent's method, Ridders' method, TOMS 748)
 * and open methods (Newton, secant, Halley), each returning the root with its evaluation history for convergence
 * figures, and `scalarRootOf`, a root that is differentiable in a parameter through the implicit function theorem.
 */

import { customVjp, jvp as forward, grad } from '../../autodiff/index.ts'
import { div, item, mul, neg, primalOf, tensor, type Value } from '../../tensor/index.ts'
import {
  type BracketOptions,
  bisectAlgorithm,
  brentqAlgorithm,
  type NewtonOptions,
  newtonAlgorithm,
  type RootResult,
  ridderAlgorithm,
  solveRoot,
  toms748Algorithm,
} from './scalar-algorithms.ts'

/**
 * Bisection on a sign-changing bracket (scipy.optimize.bisect): halve the bracket until it is shorter than
 * xtol + rtol |x|.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end (f(a) and f(b) of opposite signs).
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @concept root-finding
 * @example
 * ```ts
 * import { bisect } from 'omai-engine/numerics/roots'
 * Math.abs(bisect((x) => x * x - 2, 0, 2).root - Math.SQRT2) < 1e-11 // => true
 * ```
 */
export function bisect(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  return solveRoot(bisectAlgorithm(f, a, b, options), options.maxIterations ?? 100)
}

/**
 * Brent's method (scipy.optimize.brentq): inverse quadratic interpolation and secant steps, falling back to bisection,
 * guaranteed to converge on a bracket and usually superlinear.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @concept root-finding
 * @index function Brent's bracketing root finder for a scalar function.
 * @example
 * ```ts
 * import { brentq } from 'omai-engine/numerics/roots'
 * Math.abs(brentq(Math.cos, 0, 2).root - Math.PI / 2) < 1e-12 // => true
 * ```
 */
export function brentq(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  return solveRoot(brentqAlgorithm(f, a, b, options), options.maxIterations ?? 100)
}

/**
 * Ridders' method (scipy.optimize.ridder): exponential interpolation through the bracket and its midpoint.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @example
 * ```ts
 * import { ridder } from 'omai-engine/numerics/roots'
 * Math.abs(ridder((x) => Math.exp(x) - 2, 0, 1).root - Math.LN2) < 1e-11 // => true
 * ```
 */
export function ridder(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  return solveRoot(ridderAlgorithm(f, a, b, options), options.maxIterations ?? 100)
}

/**
 * The method of Alefeld, Potra and Shi (TOMS 748; scipy.optimize.toms748) in its essentials: each iteration tries
 * inverse cubic interpolation (falling back to a Newton step on the quadratic through three points), a double-length
 * secant step, and a bisection whenever the bracket has not shrunk by half.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @example
 * ```ts
 * import { toms748 } from 'omai-engine/numerics/roots'
 * Math.abs(toms748((x) => x ** 3 - x - 1, 1, 2).root - 1.324717957244746) < 1e-12 // => true
 * ```
 */
export function toms748(f: (x: number) => number, a0: number, b0: number, options: BracketOptions = {}): RootResult {
  return solveRoot(toms748Algorithm(f, a0, b0, options), options.maxIterations ?? 100)
}

/**
 * Newton's method, or the secant method without a derivative, or Halley's method with a second derivative
 * (scipy.optimize.newton): local and fast (quadratic, superlinear and cubic convergence).
 *
 * @param f - The function.
 * @param x0 - Starting point.
 * @param options - Derivatives, a second start, tolerances and iteration cap.
 * @returns The root with its history.
 * @concept root-finding
 * @example
 * ```ts
 * import { newton } from 'omai-engine/numerics/roots'
 * Math.abs(newton((x) => x * x - 2, 1, { fprime: (x) => 2 * x }).root - Math.SQRT2) < 1e-12 // => true
 * ```
 */
export function newton(f: (x: number) => number, x0: number, options: NewtonOptions = {}): RootResult {
  return solveRoot(newtonAlgorithm(f, x0, options), options.maxIterations ?? 50)
}

/**
 * The secant method from two starting points.
 *
 * @param f - The function.
 * @param x0 - First point.
 * @param x1 - Second point.
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @example
 * ```ts
 * import { secant } from 'omai-engine/numerics/roots'
 * secant((x) => x - 3, 0, 1).root // => 3
 * ```
 */
export function secant(
  f: (x: number) => number,
  x0: number,
  x1: number,
  options: Omit<NewtonOptions, 'fprime' | 'fprime2' | 'x1'> = {},
): RootResult {
  return newton(f, x0, { ...options, x1 })
}

/**
 * Halley's method (cubically convergent with first and second derivatives).
 *
 * @param f - The function.
 * @param x0 - Starting point.
 * @param fprime - First derivative.
 * @param fprime2 - Second derivative.
 * @param options - Tolerances and iteration cap.
 * @returns The root with its history.
 * @example
 * ```ts
 * import { halley } from 'omai-engine/numerics/roots'
 * Math.abs(halley((x) => x ** 3 - 8, 3, (x) => 3 * x * x, (x) => 6 * x).root - 2) < 1e-12 // => true
 * ```
 */
export function halley(
  f: (x: number) => number,
  x0: number,
  fprime: (x: number) => number,
  fprime2: (x: number) => number,
  options: Omit<NewtonOptions, 'fprime' | 'fprime2' | 'x1'> = {},
): RootResult {
  return newton(f, x0, { ...options, fprime, fprime2 })
}

/**
 * A root that is differentiable in a parameter: solve f(x, theta) = 0 numerically (with any scalar method on the
 * primal values), then attach the implicit-function-theorem derivative dx/dtheta = -f_theta / f_x.
 *
 * @param f - The equation, differentiable in both arguments (built from operations).
 * @param theta - The parameter (scalar value, possibly traced).
 * @param solve - Finds the root of a plain scalar function (for example `(g) => brentq(g, 0, 1).root`).
 * @returns The root as a value, differentiable in theta.
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { scalarRootOf, brentq } from 'omai-engine/numerics/roots'
 * import { grad } from 'omai-engine/autodiff'
 * import { item, mul, square, sub, tensor } from 'omai-engine/tensor'
 * // x^2 = theta, so dx/dtheta = 1 / (2 sqrt(theta)) = 0.25 at theta = 4.
 * const root = (t) => scalarRootOf((x, th) => sub(square(x), th), t, (g) => brentq(g, 0, 10).root)
 * item(grad(root)(tensor(4))) // => 0.25
 * ```
 */
export function scalarRootOf(
  f: (x: Value, theta: Value) => Value,
  theta: Value,
  solve: (g: (x: number) => number) => number,
): Value {
  const fixed = primalOf(theta)
  const root = solve((x) => item(f(tensor(x), fixed)))
  const op = customVjp({
    name: 'scalarRoot',
    forward: () => tensor(root),
    vjp: (g, [th], out) => {
      const fx = grad((x: Value) => f(x, th as Value))(out)
      const fth = grad((t: Value) => f(out, t))(th as Value)
      return [neg(mul(g, div(fth as Value, fx as Value)))]
    },
    jvp: ([th], [dth], out) => {
      if (dth === null || dth === undefined) return null
      const fx = grad((x: Value) => f(x, th as Value))(out)
      return neg(div(forward((t: Value) => f(out, t), th as Value, dth).tangent, fx as Value))
    },
  })
  return op(theta)
}
