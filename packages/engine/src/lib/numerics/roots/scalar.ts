/**
 * Scalar root finding, scipy.optimize style: bracketing methods (bisection, Brent's method, Ridders' method, TOMS 748)
 * and open methods (Newton, secant, Halley), each returning the root with its evaluation history for convergence
 * figures, and `scalarRootOf`, a root that is differentiable in a parameter through the implicit function theorem.
 */

import { customVjp, jvp as forward, grad } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
import { div, item, mul, neg, primalOf, tensor, type Value } from '../../tensor/index.ts'

const EPS = 2.220446049250313e-16

/** One evaluation of a root finder, for convergence figures. */
export interface RootStep {
  readonly x: number
  readonly fx: number
}

/** Result of a scalar root finder (scipy's RootResults plus the evaluation history). */
export interface RootResult {
  readonly root: number
  readonly iterations: number
  readonly functionCalls: number
  readonly converged: boolean
  readonly flag: 'converged' | 'sign error' | 'maximum iterations' | 'derivative was zero'
  readonly history: readonly RootStep[]
}

/** Tolerances of the bracketing methods (scipy's defaults: xtol 2e-12, rtol 4 eps, maxiter 100). */
export interface BracketOptions {
  readonly xtol?: number
  readonly rtol?: number
  readonly maxIterations?: number
}

function recorder(f: (x: number) => number): { call: (x: number) => number; history: RootStep[] } {
  const history: RootStep[] = []
  return {
    history,
    call: (x) => {
      const fx = f(x)
      history.push({ x, fx })
      return fx
    },
  }
}

function result(root: number, iterations: number, history: RootStep[], flag: RootResult['flag']): RootResult {
  return { root, iterations, functionCalls: history.length, converged: flag === 'converged', flag, history }
}

function checkBracket(fa: number, fb: number, name: string): void {
  if (fa * fb > 0) throw new MdmlError('value', `${name}: f(a) and f(b) must have different signs`)
}

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
 * import { bisect } from 'mdml-engine/numerics/roots'
 * Math.abs(bisect((x) => x * x - 2, 0, 2).root - Math.SQRT2) < 1e-11 // => true
 * ```
 */
export function bisect(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  const { call, history } = recorder(f)
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  let fa = call(a)
  const fb = call(b)
  checkBracket(fa, fb, 'bisect')
  if (fa === 0) return result(a, 0, history, 'converged')
  if (fb === 0) return result(b, 0, history, 'converged')
  let lo = a
  let dm = b - a
  for (let i = 1; i <= (options.maxIterations ?? 100); i++) {
    dm *= 0.5
    const xm = lo + dm
    const fm = call(xm)
    if (fm * fa >= 0) {
      lo = xm
      fa = fm
    }
    if (fm === 0 || Math.abs(dm) < xtol + rtol * Math.abs(xm)) return result(xm, i, history, 'converged')
  }
  return result(lo + dm, options.maxIterations ?? 100, history, 'maximum iterations')
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
 * @example
 * ```ts
 * import { brentq } from 'mdml-engine/numerics/roots'
 * Math.abs(brentq(Math.cos, 0, 2).root - Math.PI / 2) < 1e-12 // => true
 * ```
 */
export function brentq(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  const { call, history } = recorder(f)
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  let xpre = a
  let xcur = b
  let fpre = call(xpre)
  let fcur = call(xcur)
  checkBracket(fpre, fcur, 'brentq')
  if (fpre === 0) return result(xpre, 0, history, 'converged')
  if (fcur === 0) return result(xcur, 0, history, 'converged')
  let xblk = 0
  let fblk = 0
  let spre = 0
  let scur = 0
  for (let i = 1; i <= (options.maxIterations ?? 100); i++) {
    if (fpre !== 0 && fcur !== 0 && Math.sign(fpre) !== Math.sign(fcur)) {
      xblk = xpre
      fblk = fpre
      spre = scur = xcur - xpre
    }
    if (Math.abs(fblk) < Math.abs(fcur)) {
      xpre = xcur
      xcur = xblk
      xblk = xpre
      fpre = fcur
      fcur = fblk
      fblk = fpre
    }
    const delta = (xtol + rtol * Math.abs(xcur)) / 2
    const sbis = (xblk - xcur) / 2
    if (fcur === 0 || Math.abs(sbis) < delta) return result(xcur, i, history, 'converged')
    if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
      let stry: number
      if (xpre === xblk) {
        stry = (-fcur * (xcur - xpre)) / (fcur - fpre)
      } else {
        const dpre = (fpre - fcur) / (xpre - xcur)
        const dblk = (fblk - fcur) / (xblk - xcur)
        stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre))
      }
      if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
        spre = scur
        scur = stry
      } else {
        spre = sbis
        scur = sbis
      }
    } else {
      spre = sbis
      scur = sbis
    }
    xpre = xcur
    fpre = fcur
    xcur += Math.abs(scur) > delta ? scur : sbis > 0 ? delta : -delta
    fcur = call(xcur)
  }
  return result(xcur, options.maxIterations ?? 100, history, 'maximum iterations')
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
 * import { ridder } from 'mdml-engine/numerics/roots'
 * Math.abs(ridder((x) => Math.exp(x) - 2, 0, 1).root - Math.LN2) < 1e-11 // => true
 * ```
 */
export function ridder(f: (x: number) => number, a: number, b: number, options: BracketOptions = {}): RootResult {
  const { call, history } = recorder(f)
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  let xa = a
  let xb = b
  let fa = call(xa)
  let fb = call(xb)
  checkBracket(fa, fb, 'ridder')
  if (fa === 0) return result(xa, 0, history, 'converged')
  if (fb === 0) return result(xb, 0, history, 'converged')
  let tol = xtol + rtol * Math.max(Math.abs(xa), Math.abs(xb))
  let xn = xa
  for (let i = 1; i <= (options.maxIterations ?? 100); i++) {
    const dm = 0.5 * (xb - xa)
    const xm = xa + dm
    const fm = call(xm)
    const dn = (Math.sign(fb - fa) * dm * fm) / Math.sqrt(fm * fm - fa * fb)
    xn = xm - Math.sign(dn) * Math.min(Math.abs(dn), Math.abs(dm) - 0.5 * tol)
    const fn = call(xn)
    if (fn * fm < 0) {
      xa = xn
      fa = fn
      xb = xm
      fb = fm
    } else if (fn * fa < 0) {
      xb = xn
      fb = fn
    } else {
      xa = xn
      fa = fn
    }
    tol = xtol + rtol * Math.abs(xn)
    if (fn === 0 || Math.abs(xb - xa) < tol) return result(xn, i, history, 'converged')
  }
  return result(xn, options.maxIterations ?? 100, history, 'maximum iterations')
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
 * import { toms748 } from 'mdml-engine/numerics/roots'
 * Math.abs(toms748((x) => x ** 3 - x - 1, 1, 2).root - 1.324717957244746) < 1e-12 // => true
 * ```
 */
export function toms748(f: (x: number) => number, a0: number, b0: number, options: BracketOptions = {}): RootResult {
  const { call, history } = recorder(f)
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  let a = Math.min(a0, b0)
  let b = Math.max(a0, b0)
  let fa = call(a)
  let fb = call(b)
  checkBracket(fa, fb, 'toms748')
  if (fa === 0) return result(a, 0, history, 'converged')
  if (fb === 0) return result(b, 0, history, 'converged')
  let d = Number.NaN
  let fd = Number.NaN
  let e = Number.NaN
  let fe = Number.NaN
  const inside = (c: number): boolean => c > a && c < b
  const tolerance = (): number => 2 * (xtol + rtol * Math.min(Math.abs(a), Math.abs(b)))
  // Shrink [a, b] around c, keeping the previous endpoint as d.
  const bracket = (c: number): boolean => {
    const fc = call(c)
    if (fc === 0) {
      a = b = c
      fa = fb = 0
      return true
    }
    if (Math.sign(fc) === Math.sign(fa)) {
      e = d
      fe = fd
      d = a
      fd = fa
      a = c
      fa = fc
    } else {
      e = d
      fe = fd
      d = b
      fd = fb
      b = c
      fb = fc
    }
    return false
  }
  const secant = (): number => a - (fa * (b - a)) / (fb - fa)
  const newtonQuadratic = (steps: number): number => {
    // P(x) = fa + B (x - a) + A (x - a)(x - b) through (a, fa), (b, fb), (d, fd).
    const B = (fb - fa) / (b - a)
    const A = ((fd - fb) / (d - b) - B) / (d - a)
    if (!Number.isFinite(A) || A === 0) return secant()
    let r = A * fa > 0 ? a : b
    for (let i = 0; i < steps; i++) r -= (fa + (B + A * (r - b)) * (r - a)) / (B + A * (2 * r - a - b))
    return inside(r) ? r : secant()
  }
  const inverseCubic = (): number => {
    // Inverse cubic interpolation through (a, b, d, e) by Aitken-Neville in the f values.
    const xs = [a, b, d, e]
    const fs = [fa, fb, fd, fe]
    const p = [...xs]
    for (let j = 1; j < 4; j++) {
      for (let i = 3; i >= j; i--)
        p[i] = ((p[i] ?? 0) * (fs[i - j] ?? 0) - (p[i - 1] ?? 0) * (fs[i] ?? 0)) / ((fs[i - j] ?? 0) - (fs[i] ?? 0))
    }
    return p[3] ?? Number.NaN
  }
  for (let n = 1; n <= (options.maxIterations ?? 100); n++) {
    const width = b - a
    for (let pass = 0; pass < 2; pass++) {
      const distinct = Number.isFinite(fe) && new Set([fa, fb, fd, fe]).size === 4
      let c = distinct ? inverseCubic() : Number.NaN
      if (!inside(c)) c = Number.isFinite(fd) ? newtonQuadratic(pass + 2) : secant()
      if (!inside(c)) c = (a + b) / 2
      if (bracket(c) || b - a < tolerance())
        return result(fa === 0 ? a : Math.abs(fa) < Math.abs(fb) ? a : b, n, history, 'converged')
    }
    // Double-length secant from the endpoint with the smaller residual.
    const [u, fu] = Math.abs(fa) < Math.abs(fb) ? [a, fa] : [b, fb]
    let c = u - (2 * fu * (b - a)) / (fb - fa)
    if (!inside(c) || Math.abs(c - u) > (b - a) / 2) c = (a + b) / 2
    if (bracket(c) || b - a < tolerance()) return result(Math.abs(fa) < Math.abs(fb) ? a : b, n, history, 'converged')
    if (b - a > 0.5 * width && bracket((a + b) / 2)) return result(a, n, history, 'converged')
    if (b - a < tolerance()) return result(Math.abs(fa) < Math.abs(fb) ? a : b, n, history, 'converged')
  }
  return result(Math.abs(fa) < Math.abs(fb) ? a : b, options.maxIterations ?? 100, history, 'maximum iterations')
}

/** Options of {@link newton}. */
export interface NewtonOptions {
  /** The derivative; without it, the secant method is used (as scipy). */
  readonly fprime?: (x: number) => number
  /** The second derivative; with fprime, Halley's method is used. */
  readonly fprime2?: (x: number) => number
  /** A second starting point for the secant method. */
  readonly x1?: number
  /** Stop when the step is below tol + rtol |x| (scipy's defaults: tol 1.48e-8, rtol 0). */
  readonly tol?: number
  readonly rtol?: number
  readonly maxIterations?: number
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
 * import { newton } from 'mdml-engine/numerics/roots'
 * Math.abs(newton((x) => x * x - 2, 1, { fprime: (x) => 2 * x }).root - Math.SQRT2) < 1e-12 // => true
 * ```
 */
export function newton(f: (x: number) => number, x0: number, options: NewtonOptions = {}): RootResult {
  const { call, history } = recorder(f)
  const tol = options.tol ?? 1.48e-8
  const rtol = options.rtol ?? 0
  const maxIterations = options.maxIterations ?? 50
  const fprime = options.fprime
  if (fprime !== undefined) {
    let p0 = x0
    for (let i = 1; i <= maxIterations; i++) {
      const fval = call(p0)
      if (fval === 0) return result(p0, i, history, 'converged')
      const d1 = fprime(p0)
      if (d1 === 0) return result(p0, i, history, 'derivative was zero')
      let step = fval / d1
      if (options.fprime2 !== undefined) {
        const adj = (step * options.fprime2(p0)) / d1 / 2
        if (Math.abs(adj) < 1) step /= 1 - adj
      }
      const p = p0 - step
      if (Math.abs(p - p0) <= tol + rtol * Math.abs(p)) return result(p, i, history, 'converged')
      p0 = p
    }
    return result(p0, maxIterations, history, 'maximum iterations')
  }
  // Secant method (scipy's choice of second point).
  let p0 = x0
  let p1 = options.x1 ?? x0 * (1 + 1e-4) + (x0 >= 0 ? 1e-4 : -1e-4)
  let q0 = call(p0)
  let q1 = call(p1)
  if (Math.abs(q1) < Math.abs(q0)) {
    ;[p0, p1, q0, q1] = [p1, p0, q1, q0]
  }
  for (let i = 1; i <= maxIterations; i++) {
    if (q1 === q0) return result((p1 + p0) / 2, i, history, p1 === p0 ? 'converged' : 'derivative was zero')
    const p =
      Math.abs(q1) > Math.abs(q0) ? ((-q0 / q1) * p1 + p0) / (1 - q0 / q1) : ((-q1 / q0) * p0 + p1) / (1 - q1 / q0)
    if (Math.abs(p - p1) <= tol + rtol * Math.abs(p)) return result(p, i, history, 'converged')
    p0 = p1
    q0 = q1
    p1 = p
    q1 = call(p1)
  }
  return result(p1, maxIterations, history, 'maximum iterations')
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
 * import { secant } from 'mdml-engine/numerics/roots'
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
 * import { halley } from 'mdml-engine/numerics/roots'
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
 * import { scalarRootOf, brentq } from 'mdml-engine/numerics/roots'
 * import { grad } from 'mdml-engine/autodiff'
 * import { item, mul, square, sub, tensor } from 'mdml-engine/tensor'
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
