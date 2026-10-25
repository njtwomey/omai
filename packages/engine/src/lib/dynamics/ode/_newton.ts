// Jacobians, LU factorisations and implicit-function corrections shared by the implicit solvers.
import { jacfwd, stopGradient } from '../../autodiff/index.ts'
import { MdmlError, minOf } from '../../base/index.ts'
import { type LuFactorisation, luFactor, luSolve, solve } from '../../numerics/linalg/index.ts'
import { isTensor, matmul, primalOf, reshape, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import { floatState, fromNumbers, numbersOf, type Tol } from './_vec.ts'
import type { VectorField } from './problem.ts'

/**
 * How an implicit solver gets df/dy: by forward-mode automatic differentiation (exact; the default), by forward finite
 * differences, or from a function returning `[n, n]` over the flattened state.
 */
export type JacobianOption = 'autodiff' | 'fd' | ((t: number, y: Value) => Value)

/** df/dy at (t, y) as a row-major n x n array of numbers. */
export function jacobianAt(f: VectorField, t: number, y: Value, jac: JacobianOption = 'autodiff'): Float64Array {
  const n = numbersOf(y).length
  const shape = y.shape
  if (typeof jac === 'function') {
    const j = numbersOf(jac(t, y))
    if (j.length !== n * n) throw new MdmlError('shape', `the Jacobian must be ${n} x ${n}`)
    return Float64Array.from(j)
  }
  if (jac === 'autodiff') {
    try {
      const g = (v: Value): Value => reshape(f(t, reshape(v, shape)), [-1])
      return Float64Array.from(primalOf(jacfwd(g)(reshape(fromNumbers(numbersOf(y), shape), [-1]))).data)
    } catch {
      // A field that cannot be differentiated falls back to differences.
    }
  }
  return fdJacobian(f, t, y)
}

/** Forward-difference Jacobian with steps sqrt(eps) max(1, |y_j|). */
export function fdJacobian(f: VectorField, t: number, y: Value): Float64Array {
  const y0 = numbersOf(y)
  const n = y0.length
  const f0 = numbersOf(f(t, fromNumbers(y0, y.shape)))
  const out = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const h = Math.sqrt(Number.EPSILON) * Math.max(1, Math.abs(y0[j] ?? 0))
    const yp = Float64Array.from(y0)
    yp[j] = (yp[j] ?? 0) + h
    const fp = numbersOf(f(t, fromNumbers(yp, y.shape)))
    for (let i = 0; i < n; i++) out[i * n + j] = ((fp[i] ?? 0) - (f0[i] ?? 0)) / h
  }
  return out
}

/** A real LU factorisation of a row-major n x n matrix. */
export function realLu(a: Float64Array, n: number): LuFactorisation {
  return luFactor(fromNumbers(a, [n, n]))
}

/** Solve with a real LU factorisation. */
export function luApply(lu: LuFactorisation, b: ArrayLike<number>): Float64Array {
  return Float64Array.from(luSolve(lu, fromNumbers(b, [b.length])).data)
}

/** A complex n x n matrix A = re + i im factorised as the real 2n x 2n system [[re, -im], [im, re]]. */
export function complexLu(re: Float64Array, im: Float64Array, n: number): LuFactorisation {
  const m = 2 * n
  const big = new Float64Array(m * m)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const r = re[i * n + j] ?? 0
      const c = im[i * n + j] ?? 0
      big[i * m + j] = r
      big[i * m + n + j] = -c
      big[(n + i) * m + j] = c
      big[(n + i) * m + n + j] = r
    }
  }
  return luFactor(fromNumbers(big, [m, m]))
}

/** Solve a complex system from {@link complexLu}: returns the real and imaginary parts. */
export function complexApply(
  lu: LuFactorisation,
  bre: ArrayLike<number>,
  bim: ArrayLike<number>,
): { re: Float64Array; im: Float64Array } {
  const n = bre.length
  const rhs = new Float64Array(2 * n)
  rhs.set(bre, 0)
  rhs.set(bim, n)
  const x = Float64Array.from(luSolve(lu, fromNumbers(rhs, [2 * n])).data)
  return { re: x.slice(0, n), im: x.slice(n) }
}

/** m @ v for a row-major n x n matrix. */
export function matVec(m: ArrayLike<number>, v: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (m[i * n + j] ?? 0) * (v[j] ?? 0)
    out[i] = s
  }
  return out
}

/** The identity as a row-major n x n array. */
export function identityNumbers(n: number): Float64Array {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = 1
  return out
}

/**
 * The implicit-function correction that makes an implicit step differentiable: given a root z* of G (found on
 * concrete numbers) and the exact derivative M = dG/dz there, return a value equal to z* whose derivative is
 * -M^{-1} dG, the implicit function theorem applied to whatever G depends on (earlier states, parameters captured by
 * the field). Untraced residuals return z* directly.
 */
export function implicitCorrection(
  zStar: ArrayLike<number>,
  shape: readonly number[],
  residual: (z: Value) => Value,
  m: () => Float64Array,
): Value {
  const z = fromNumbers(zStar, shape)
  const g = reshape(residual(z), [-1])
  if (isTensor(g)) return z
  const size = zStar.length
  const dg = sub(g, stopGradient(g))
  const delta = solve(fromNumbers(m(), [size, size]), dg)
  return sub(z, reshape(delta, shape))
}

/** A mass matrix as numbers (checked against the state size when it is first used). */
export function massNumbers(mass: TensorLike | undefined): Float64Array | undefined {
  return mass === undefined ? undefined : Float64Array.from(numbersOf(floatState(mass)))
}

export function checkMass(mass: Float64Array | undefined, n: number): Float64Array | undefined {
  if (mass !== undefined && mass.length !== n * n) throw new MdmlError('shape', `the mass matrix must be ${n} x ${n}`)
  return mass
}

export function applyMass(mass: Float64Array | undefined, v: ArrayLike<number>, n: number): Float64Array {
  return mass === undefined ? Float64Array.from(v) : matVec(mass, v, n)
}

export function isTracedState(...vs: readonly Value[]): boolean {
  return vs.some((v) => !isTensor(v))
}

/** M v for a value (the identity when there is no mass matrix). */
export function massTimes(mass: Float64Array | undefined, v: Value, n: number): Value {
  if (mass === undefined) return v
  return reshape(matmul(fromNumbers(mass, [n, n]), reshape(v, [n, 1])), v.shape)
}

/** Settings the implicit steps share. */
export interface ImplicitConfig {
  readonly rtol: Tol
  readonly atol: Tol
  readonly maxStep: number
  readonly newtonTol: number
  readonly jac: JacobianOption
  readonly mass: Float64Array | undefined
}

/** scipy's Newton tolerance for the implicit methods: max(10 eps / rtol, min(0.03, sqrt(rtol))). */
export function newtonTolerance(rtol: Tol): number {
  const r0 = typeof rtol === 'number' ? rtol : minOf(rtol)
  return Math.max((10 * Number.EPSILON) / r0, Math.min(0.03, r0 ** 0.5))
}
