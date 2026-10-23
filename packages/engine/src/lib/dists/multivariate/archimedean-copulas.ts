import { jvp } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { fixedQuad } from '../../numerics/quadrature/index.ts'
import { foldIn, gammaVariates, type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  abs,
  add,
  div,
  exp,
  expm1,
  fromStorage,
  log,
  log1p,
  mul,
  neg,
  oneHot,
  pow,
  primalOf,
  promoteFloat,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { discreteSample } from '../_sampling.ts'
import { validateParams } from '../distribution.ts'
import { coordinate } from './_coordinates.ts'
import { type Copula, defineCopula } from './copulas.ts'

/** The parts of an Archimedean copula C(u) = psi(sum_i phi(u_i)). */
interface Archimedean {
  readonly name: string
  readonly dim: number
  readonly theta: Value
  /** The generator phi. */
  phi(u: Value): Value
  /** The inverse generator psi. */
  psi(t: Value): Value
  /** A closed-form log density, when there is one for this dimension. */
  logDensity?(u: Value): Value
  /** Concrete draws `[...shape, d]` at a concrete theta. */
  draw(key: Key, theta: number, shape: readonly number[]): Float64Array
  kendallTau(): Value
}

function archimedean(a: Archimedean): Copula {
  const d = a.dim
  const cdf = (u: Value): Value => a.psi(sum(a.phi(u), -1))
  // The density as the mixed partial derivative d^d C / du_1 ... du_d, by nested forward-mode differentiation.
  const mixedPartial = (u: Value): Value => {
    let f = cdf
    for (let i = 0; i < d; i++) {
      const g = f
      const direction = oneHot(promoteFloat(i)[0] as Value, d)
      f = (v: Value): Value => jvp(g, v, mul(add(mul(v, 0), 1), direction)).tangent
    }
    return f(u)
  }
  return defineCopula({
    name: a.name,
    dim: d,
    params: { theta: a.theta },
    logDensity: (u) => (a.logDensity ? a.logDensity(u) : log(mixedPartial(u))),
    cdf,
    sample: (key, shape) =>
      discreteSample([a.theta], [...shape, d], 'float64', ([t]) =>
        fromStorage(a.draw(key, toFlat(t as Tensor)[0] ?? 1, shape), [...shape, d], 'float64'),
      ),
    kendallTau: a.kendallTau,
  })
}

/** Unit exponentials for the frailty constructions, one row of d per draw. */
function exponentials(key: Key, rows: number, d: number): Float64Array {
  const u = uniformNoise(key, [rows * d]).data
  return Float64Array.from(u, (v) => -Math.log1p(-v))
}

const rowsOf = (shape: readonly number[]): number => shape.reduce((p, q) => p * q, 1)

function checkDim(dim: number, family: string): void {
  if (!(Number.isInteger(dim) && dim >= 2)) throw new WhiteboxError('value', `${family}: dim must be at least 2`)
}

/**
 * The Clayton copula C(u) = (sum u_i^(-theta) - d + 1)^(-1/theta), theta > 0 (statsmodels' ClaytonCopula), with lower
 * tail dependence; Kendall's tau = theta / (theta + 2).
 *
 * @param theta - The parameter (positive).
 * @param dim - The dimension (default 2).
 * @returns The copula with closed-form density in any dimension and Marshall-Olkin sampling (gamma frailty).
 * @example
 * ```ts
 * import { claytonCopula } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(claytonCopula(2).kendallTau()) // => 0.5
 * ```
 */
export function claytonCopula(theta: TensorLike, dim = 2): Copula {
  checkDim(dim, 'claytonCopula')
  const [t] = promoteFloat(theta) as [Value]
  validateParams('claytonCopula', { theta: [t, 'positive'] })
  let logConstant: Value = mul(t, 0)
  for (let k = 1; k < dim; k++) logConstant = add(logConstant, log1p(mul(t, k)))
  return archimedean({
    name: 'claytonCopula',
    dim,
    theta: t,
    phi: (u) => sub(pow(u, neg(t)), 1),
    psi: (s) => pow(add(s, 1), div(-1, t)),
    logDensity: (u) =>
      sub(
        sub(logConstant, mul(add(t, 1), sum(log(u), -1))),
        mul(add(dim, div(1, t)), log(sub(sum(pow(u, neg(t)), -1), dim - 1))),
      ),
    draw: (key, th, shape) => {
      const n = rowsOf(shape)
      const v = gammaVariates(foldIn(key, 0), new Array<number>(n).fill(1 / th))
      const e = exponentials(foldIn(key, 1), n, dim)
      return e.map((x, i) => (1 + x / (v[Math.floor(i / dim)] ?? 1)) ** (-1 / th))
    },
    kendallTau: () => div(t, add(t, 2)),
  })
}

/**
 * The Gumbel (Gumbel-Hougaard) copula C(u) = exp(-(sum (-log u_i)^theta)^(1/theta)), theta >= 1 (statsmodels'
 * GumbelCopula), with upper tail dependence; Kendall's tau = 1 - 1 / theta.
 *
 * @param theta - The parameter (at least 1; 1 is independence).
 * @param dim - The dimension (default 2).
 * @returns The copula with a closed-form bivariate density (higher dimensions differentiate the CDF), and
 *   Marshall-Olkin sampling with a positive stable frailty (Kanter's representation).
 * @example
 * ```ts
 * import { gumbelCopula } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(gumbelCopula(2).kendallTau()) // => 0.5
 * ```
 */
export function gumbelCopula(theta: TensorLike, dim = 2): Copula {
  checkDim(dim, 'gumbelCopula')
  const [t] = promoteFloat(theta) as [Value]
  validateParams('gumbelCopula', { theta: [sub(t, 1), 'nonNegative'] })
  const bivariate = (u: Value): Value => {
    // log c = -A + (theta - 1) log(x y) - log(u v) + (1/theta - 2) log S + log(A + theta - 1), x = -log u.
    const x = neg(log(u))
    const s = sum(pow(x, t), -1)
    const a = pow(s, div(1, t))
    return add(
      add(sub(sub(neg(a), sum(log(u), -1)), neg(mul(sub(t, 1), sum(log(x), -1)))), mul(sub(div(1, t), 2), log(s))),
      log(add(a, sub(t, 1))),
    )
  }
  return archimedean({
    name: 'gumbelCopula',
    dim,
    theta: t,
    phi: (u) => pow(neg(log(u)), t),
    psi: (s) => exp(neg(pow(s, div(1, t)))),
    ...(dim === 2 ? { logDensity: bivariate } : {}),
    draw: (key, th, shape) => {
      const n = rowsOf(shape)
      const alpha = 1 / th
      const w = uniformNoise(foldIn(key, 0), [n]).data
      const ee = exponentials(foldIn(key, 1), n, 1)
      // Kanter: a positive stable variable with Laplace transform exp(-s^alpha).
      const v = Float64Array.from(w, (wi, i) => {
        const angle = Math.PI * (wi === 0 ? 1e-16 : wi)
        const first = Math.sin(alpha * angle) / Math.sin(angle) ** (1 / alpha)
        const second = (Math.sin((1 - alpha) * angle) / (ee[i] ?? 1)) ** ((1 - alpha) / alpha)
        return first * second
      })
      const e = exponentials(foldIn(key, 2), n, dim)
      return e.map((x, i) => Math.exp(-((x / (v[Math.floor(i / dim)] ?? 1)) ** alpha)))
    },
    kendallTau: () => sub(1, div(1, t)),
  })
}

/** Kendall's tau of the Frank copula: 1 - 4 / theta + 4 D_1(theta) / theta, D_1 the Debye function. */
function frankTau(theta: Value): Value {
  const integral = fixedQuad((s) => div(s, expm1(s)), 0, theta, 48)
  return add(sub(1, div(4, theta)), div(mul(integral, 4), square(theta)))
}

/**
 * The Frank copula C(u) = -(1/theta) log(1 + prod(e^(-theta u_i) - 1) / (e^(-theta) - 1)^(d-1)), theta nonzero
 * (statsmodels' FrankCopula), radially symmetric without tail dependence; negative theta (bivariate only) gives
 * negative dependence.
 *
 * @param theta - The parameter (nonzero; positive for d > 2).
 * @param dim - The dimension (default 2).
 * @returns The copula with a closed-form bivariate density (higher dimensions differentiate the CDF); bivariate
 *   sampling by the conditional distribution, higher dimensions by a logarithmic-series frailty.
 * @example
 * ```ts
 * import { frankCopula } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(frankCopula(3).cdf([1, 0.4])) - 0.4) < 1e-14 // => true
 * ```
 */
export function frankCopula(theta: TensorLike, dim = 2): Copula {
  checkDim(dim, 'frankCopula')
  const [t] = promoteFloat(theta) as [Value]
  const values = toFlat(primalOf(t))
  if (values.some((v) => v === 0 || !Number.isFinite(v)))
    throw new WhiteboxError('value', 'frankCopula: theta must be finite and nonzero')
  if (dim > 2 && values.some((v) => v < 0))
    throw new WhiteboxError('value', 'frankCopula: theta must be positive in more than two dimensions')
  // e^(-theta u) - 1, accurately.
  const em1 = (u: Value): Value => expm1(mul(neg(t), u))
  const bivariate = (u: Value): Value => {
    // c = theta (1 - e^-theta) e^(-theta (u + v)) / [(1 - e^-theta) - (1 - e^(-theta u))(1 - e^(-theta v))]^2.
    const one = neg(expm1(neg(t)))
    const a = em1(coordinate(u, 0))
    const b = em1(coordinate(u, 1))
    const denom = sub(one, mul(a, b))
    return sub(add(log(mul(t, one)), mul(neg(t), sum(u, -1))), mul(log(abs(denom)), 2))
  }
  return archimedean({
    name: 'frankCopula',
    dim,
    theta: t,
    phi: (u) => neg(log(div(em1(u), expm1(neg(t))))),
    psi: (s) => mul(neg(log1p(mul(exp(neg(s)), expm1(neg(t))))), div(1, t)),
    ...(dim === 2 ? { logDensity: bivariate } : {}),
    draw: (key, th, shape) => {
      const n = rowsOf(shape)
      if (dim === 2) {
        // Conditional inversion: v solves C(v | u) = w.
        const uw = uniformNoise(key, [n, 2]).data
        const out = new Float64Array(n * 2)
        for (let i = 0; i < n; i++) {
          const u = uw[2 * i] ?? 0.5
          const w = uw[2 * i + 1] ?? 0.5
          const eu = Math.exp(-th * u)
          out[2 * i] = u
          out[2 * i + 1] = -Math.log1p((w * Math.expm1(-th)) / (w + (1 - w) * eu)) / th
        }
        return out
      }
      // Marshall-Olkin with a logarithmic-series frailty: P(V = k) = p^k / (k theta), p = 1 - e^-theta.
      const p = -Math.expm1(-th)
      const u = uniformNoise(foldIn(key, 0), [n]).data
      const v = Float64Array.from(u, (ui) => {
        let k = 1
        let term = p / th
        let acc = term
        while (acc < ui && k < 1e7) {
          term *= (p * k) / (k + 1)
          acc += term
          k++
        }
        return k
      })
      const e = exponentials(foldIn(key, 1), n, dim)
      return e.map((x, i) => -Math.log1p(-p * Math.exp(-x / (v[Math.floor(i / dim)] ?? 1))) / th)
    },
    kendallTau: () => frankTau(t),
  })
}
