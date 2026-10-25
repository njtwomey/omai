/**
 * Gaussian factors in canonical (information) form, the continuous counterpart of discrete factors used by Gaussian
 * elimination and belief propagation: conversion to and from moment form, product and division (parameters add and
 * subtract), marginalisation by Schur complement, conditioning on observed values, and the factor of a linear-Gaussian
 * node.
 */

import { WhiteboxError } from '../base/index.ts'
import {
  checkSquare,
  dotVec,
  inverse,
  logdetPositive,
  type Matrix,
  matVec,
  solveVector,
  sub,
  symmetrise,
} from './_dense.ts'

/**
 * A Gaussian factor in canonical (information) form over scalar variables:
 * log phi(x) = g + h^T x - x^T K x / 2. Products, marginals and conditionals are closed-form, so Gaussian networks are
 * solved by the same elimination and message passing as discrete ones.
 *
 * @concept gaussian-factors
 */
export interface GaussianFactor {
  readonly variables: readonly string[]
  /** The precision-like matrix K (symmetric), row by row. */
  readonly K: readonly (readonly number[])[]
  /** The potential vector h. */
  readonly h: readonly number[]
  /** The log normaliser g. */
  readonly g: number
}

/** A Gaussian in moment form. */
export interface GaussianMoments {
  readonly variables: readonly string[]
  readonly mean: readonly number[]
  readonly covariance: readonly (readonly number[])[]
  /** log of the total mass of the factor it came from (0 for a normalised density). */
  readonly logMass: number
}

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * A Gaussian factor from its canonical parameters.
 *
 * @param variables - Variable names.
 * @param K - The symmetric matrix K.
 * @param h - The vector h.
 * @param g - The log normaliser (default 0).
 * @returns The factor.
 * @throws WhiteboxError with code `shape` for mismatched sizes, `value` for repeated variables.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFactor } from 'whitebox-engine/pgm'
 * gaussianFactor(['x'], [[2]], [4]).h // => [4]
 * ```
 */
export function gaussianFactor(variables: readonly string[], K: Matrix, h: readonly number[], g = 0): GaussianFactor {
  const n = variables.length
  if (new Set(variables).size !== n) throw new WhiteboxError('value', 'gaussianFactor: variables must be distinct')
  checkSquare(K, n, 'gaussianFactor')
  if (h.length !== n) throw new WhiteboxError('shape', `gaussianFactor: h needs ${n} entries`)
  return { variables: [...variables], K: symmetrise(K), h: [...h], g }
}

/**
 * The canonical form of a Gaussian density N(mean, covariance), optionally scaled by exp(logMass).
 *
 * @param variables - Variable names.
 * @param mean - The mean.
 * @param covariance - The covariance (positive definite).
 * @param logMass - log of the total mass (default 0: a normalised density).
 * @returns The factor.
 * @throws WhiteboxError with code `value` when the covariance is not positive definite.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFromMoments } from 'whitebox-engine/pgm'
 * gaussianFromMoments(['x'], [2], [[0.5]]).K // => [[2]]
 * ```
 */
export function gaussianFromMoments(
  variables: readonly string[],
  mean: readonly number[],
  covariance: Matrix,
  logMass = 0,
): GaussianFactor {
  checkSquare(covariance, variables.length, 'gaussianFromMoments')
  const K = inverse(covariance)
  const h = matVec(K, mean)
  const g =
    logMass - dotVec(mean, h) / 2 - (variables.length * LOG_2PI + logdetPositive(covariance, 'gaussianFromMoments')) / 2
  return gaussianFactor(variables, K, h, g)
}

/**
 * The moment form of a Gaussian factor with positive-definite K: mean K^{-1} h, covariance K^{-1}, and the log of its
 * total mass, g + h^T K^{-1} h / 2 + (n log 2 pi - log |K|) / 2.
 *
 * @param f - The factor.
 * @returns Mean, covariance and log mass.
 * @throws WhiteboxError with code `value` when K is not positive definite.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFactor, gaussianToMoments } from 'whitebox-engine/pgm'
 * gaussianToMoments(gaussianFactor(['x'], [[2]], [4])).mean // => [2]
 * ```
 */
export function gaussianToMoments(f: GaussianFactor): GaussianMoments {
  const covariance = symmetrise(inverse(f.K))
  const mean = matVec(covariance, f.h)
  const n = f.variables.length
  const logMass = f.g + dotVec(f.h, mean) / 2 + (n * LOG_2PI - logdetPositive(f.K, 'gaussianToMoments')) / 2
  return { variables: [...f.variables], mean, covariance, logMass }
}

/**
 * The log of the total mass of a Gaussian factor (its integral over all its variables).
 *
 * @param f - The factor (K positive definite).
 * @returns log integral of phi.
 * @example
 * ```ts
 * import { gaussianFromMoments, gaussianLogMass } from 'whitebox-engine/pgm'
 * Math.abs(gaussianLogMass(gaussianFromMoments(['x', 'y'], [1, 2], [[2, 0.5], [0.5, 1]], 3)) - 3) < 1e-12 // => true
 * ```
 */
export function gaussianLogMass(f: GaussianFactor): number {
  return f.variables.length === 0 ? f.g : gaussianToMoments(f).logMass
}

/** The factor laid out over a superset of its variables (zeros for the new ones). */
function expand(f: GaussianFactor, variables: readonly string[]): { K: number[][]; h: number[] } {
  const pos = variables.map((v) => f.variables.indexOf(v))
  const K = pos.map((i) => pos.map((j) => (i < 0 || j < 0 ? 0 : (f.K[i]?.[j] ?? 0))))
  const h = pos.map((i) => (i < 0 ? 0 : (f.h[i] ?? 0)))
  return { K, h }
}

/**
 * The product of two Gaussian factors: canonical parameters add, over the union of their variables.
 *
 * @param a - First factor.
 * @param b - Second factor.
 * @returns The product, over a's variables followed by b's new ones.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFactor, gaussianProduct } from 'whitebox-engine/pgm'
 * gaussianProduct(gaussianFactor(['x'], [[1]], [1]), gaussianFactor(['x', 'y'], [[1, 0], [0, 2]], [0, 1])).K // => [[2, 0], [0, 2]]
 * ```
 */
export function gaussianProduct(a: GaussianFactor, b: GaussianFactor): GaussianFactor {
  const variables = [...a.variables, ...b.variables.filter((v) => !a.variables.includes(v))]
  const ea = expand(a, variables)
  const eb = expand(b, variables)
  return {
    variables,
    K: ea.K.map((r, i) => r.map((v, j) => v + (eb.K[i]?.[j] ?? 0))),
    h: ea.h.map((v, i) => v + (eb.h[i] ?? 0)),
    g: a.g + b.g,
  }
}

/**
 * The quotient of two Gaussian factors (b's variables a subset of a's): canonical parameters subtract. Used by
 * HUGIN-style updates and expectation propagation.
 *
 * @param a - Numerator.
 * @param b - Denominator.
 * @returns a / b over a's variables.
 * @throws WhiteboxError with code `value` when b mentions a variable a does not.
 * @example
 * ```ts
 * import { gaussianFactor, gaussianDivide } from 'whitebox-engine/pgm'
 * gaussianDivide(gaussianFactor(['x'], [[3]], [2]), gaussianFactor(['x'], [[1]], [1])).K // => [[2]]
 * ```
 */
export function gaussianDivide(a: GaussianFactor, b: GaussianFactor): GaussianFactor {
  for (const v of b.variables)
    if (!a.variables.includes(v)) throw new WhiteboxError('value', `gaussianDivide: ${v} is not in the numerator`)
  const eb = expand(b, a.variables)
  return {
    variables: [...a.variables],
    K: a.K.map((r, i) => r.map((v, j) => v - (eb.K[i]?.[j] ?? 0))),
    h: a.h.map((v, i) => v - (eb.h[i] ?? 0)),
    g: a.g - b.g,
  }
}

/**
 * Integrate variables out of a Gaussian factor (the Schur complement): with Y the variables removed,
 * K' = K_XX - K_XY K_YY^{-1} K_YX, h' = h_X - K_XY K_YY^{-1} h_Y and
 * g' = g + (|Y| log 2 pi - log |K_YY| + h_Y^T K_YY^{-1} h_Y) / 2.
 *
 * @param f - The factor.
 * @param variables - Variables to integrate out (K_YY must be positive definite).
 * @returns The marginal factor.
 * @throws WhiteboxError with code `value` for unknown variables or a non-positive-definite K_YY.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFromMoments, gaussianMarginalise, gaussianToMoments } from 'whitebox-engine/pgm'
 * const f = gaussianFromMoments(['x', 'y'], [1, 2], [[2, 0.5], [0.5, 1]])
 * gaussianToMoments(gaussianMarginalise(f, ['y'])).covariance.map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[2]]
 * ```
 */
export function gaussianMarginalise(f: GaussianFactor, variables: readonly string[]): GaussianFactor {
  for (const v of variables)
    if (!f.variables.includes(v)) throw new WhiteboxError('value', `gaussianMarginalise: ${v} is not in the factor`)
  if (variables.length === 0) return f
  const keep = f.variables.filter((v) => !variables.includes(v))
  const x = keep.map((v) => f.variables.indexOf(v))
  const y = variables.map((v) => f.variables.indexOf(v))
  const Kyy = sub(f.K, y, y)
  const Kxy = sub(f.K, x, y)
  const hy = y.map((i) => f.h[i] ?? 0)
  const KyyInvHy = solveVector(Kyy, hy)
  // KyyInvKyx[c] is K_YY^{-1} K_{Y, x_c}; K' = K_XX - K_XY K_YY^{-1} K_YX.
  const KyyInvKyx = x.map((_, c) => solveVector(Kyy, Kxy[c] ?? []))
  const K = x.map((i, a) => x.map((j, b) => (f.K[i]?.[j] ?? 0) - dotVec(Kxy[a] ?? [], KyyInvKyx[b] ?? [])))
  const h = x.map((i, a) => (f.h[i] ?? 0) - dotVec(Kxy[a] ?? [], KyyInvHy))
  const g = f.g + (y.length * LOG_2PI - logdetPositive(Kyy, 'gaussianMarginalise') + dotVec(hy, KyyInvHy)) / 2
  return { variables: keep, K: symmetrise(K), h, g }
}

/**
 * Condition a Gaussian factor on observed values (Y = y): K' = K_XX, h' = h_X - K_XY y,
 * g' = g + h_Y^T y - y^T K_YY y / 2.
 *
 * @param f - The factor.
 * @param evidence - Observed values by variable; variables not in the factor are ignored.
 * @returns The reduced factor.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { gaussianFromMoments, gaussianReduce, gaussianToMoments } from 'whitebox-engine/pgm'
 * const f = gaussianFromMoments(['x', 'y'], [0, 0], [[1, 0.5], [0.5, 1]])
 * gaussianToMoments(gaussianReduce(f, { y: 2 })).mean.map((v) => Math.round(v * 1e12) / 1e12) // => [1]
 * ```
 */
export function gaussianReduce(f: GaussianFactor, evidence: Readonly<Record<string, number>>): GaussianFactor {
  const fixed = f.variables.filter((v) => v in evidence)
  if (fixed.length === 0) return f
  const keep = f.variables.filter((v) => !(v in evidence))
  const x = keep.map((v) => f.variables.indexOf(v))
  const y = fixed.map((v) => f.variables.indexOf(v))
  const yv = fixed.map((v) => evidence[v] ?? 0)
  const Kxy = sub(f.K, x, y)
  const Kyy = sub(f.K, y, y)
  const h = x.map((i, a) => (f.h[i] ?? 0) - dotVec(Kxy[a] ?? [], yv))
  const hy = y.map((i) => f.h[i] ?? 0)
  const g = f.g + dotVec(hy, yv) - dotVec(yv, matVec(Kyy, yv)) / 2
  return { variables: keep, K: sub(f.K, x, x), h, g }
}

/**
 * The log value of a Gaussian factor at a point.
 *
 * @param f - The factor.
 * @param x - A value for each of its variables.
 * @returns g + h^T x - x^T K x / 2.
 * @example
 * ```ts
 * import { gaussianFactor, gaussianLogValue } from 'whitebox-engine/pgm'
 * gaussianLogValue(gaussianFactor(['x'], [[2]], [1], 0.5), { x: 1 }) // => 0.5
 * ```
 */
export function gaussianLogValue(f: GaussianFactor, x: Readonly<Record<string, number>>): number {
  const v = f.variables.map((n) => x[n] ?? 0)
  return f.g + dotVec(f.h, v) - dotVec(v, matVec(f.K, v)) / 2
}

/**
 * The same Gaussian factor with its variables in another order.
 *
 * @param f - The factor.
 * @param variables - A permutation of its variables.
 * @returns The permuted factor.
 * @throws WhiteboxError with code `value` when `variables` is not a permutation.
 * @example
 * ```ts
 * import { gaussianFactor, permuteGaussian } from 'whitebox-engine/pgm'
 * permuteGaussian(gaussianFactor(['x', 'y'], [[1, 0], [0, 2]], [3, 4]), ['y', 'x']).h // => [4, 3]
 * ```
 */
export function permuteGaussian(f: GaussianFactor, variables: readonly string[]): GaussianFactor {
  if (variables.length !== f.variables.length || variables.some((v) => !f.variables.includes(v))) {
    throw new WhiteboxError('value', 'permuteGaussian: not a permutation of the scope')
  }
  const { K, h } = expand(f, variables)
  return { variables: [...variables], K, h, g: f.g }
}

/**
 * The conditional density N(x; w^T u + b, variance) of a linear-Gaussian node as a canonical factor over
 * [...parents, child]: K = [w; -1][w; -1]^T / variance, h = b [-w; 1] / variance,
 * g = -b^2 / (2 variance) - log(2 pi variance) / 2.
 *
 * @param child - The child variable.
 * @param parents - The parents u.
 * @param weights - The weights w.
 * @param intercept - The intercept b.
 * @param variance - The noise variance (positive).
 * @returns The factor.
 * @throws WhiteboxError with code `value` for a non-positive variance or mismatched weights.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { linearGaussianFactor, gaussianLogValue } from 'whitebox-engine/pgm'
 * const f = linearGaussianFactor('y', ['x'], [2], 1, 0.25)
 * Math.abs(gaussianLogValue(f, { x: 1, y: 3 }) - Math.log(1 / Math.sqrt(2 * Math.PI * 0.25))) < 1e-12 // => true
 * ```
 */
export function linearGaussianFactor(
  child: string,
  parents: readonly string[],
  weights: readonly number[],
  intercept: number,
  variance: number,
): GaussianFactor {
  if (!(variance > 0)) throw new WhiteboxError('value', `linearGaussianFactor: ${child} needs a positive variance`)
  if (weights.length !== parents.length)
    throw new WhiteboxError('shape', `linearGaussianFactor: one weight per parent of ${child}`)
  const a = [...weights.map((w) => w), -1]
  const K = a.map((x) => a.map((y) => (x * y) / variance))
  const h = a.map((x) => (-intercept * x) / variance)
  const g = -(intercept * intercept) / (2 * variance) - Math.log(2 * Math.PI * variance) / 2
  return gaussianFactor([...parents, child], K, h, g)
}
