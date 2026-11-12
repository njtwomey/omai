// Gauss-Hermite quadrature against a Gaussian message: the expectation of a function under a Gaussian and the moments
// of a Gaussian times a function, what the numeric operators (a logistic link, a product, a user's density) integrate.
import { gaussHermite, gaussLegendre } from '../../numerics/quadrature/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { shiftedSum } from './_moments.ts'

const rules = new Map<number, { readonly x: Float64Array; readonly w: Float64Array }>()

function hermite(n: number): { readonly x: Float64Array; readonly w: Float64Array } {
  let r = rules.get(n)
  if (r === undefined) {
    const g = gaussHermite(n)
    const norm = Math.sqrt(Math.PI)
    r = { x: Float64Array.from(toFlat(g.nodes)), w: Float64Array.from(toFlat(g.weights), (v) => v / norm) }
    rules.set(n, r)
  }
  return r
}

/**
 * The nodes and weights that integrate against N(mean, variance): sum_i w_i g(x_i) approximates E[g(x)].
 *
 * @param mean - The mean.
 * @param variance - The variance (non-negative; zero gives one node at the mean).
 * @param n - The number of nodes.
 * @returns Nodes and weights (summing to one).
 */
export function gaussianNodes(
  mean: number,
  variance: number,
  n = 48,
): { readonly x: Float64Array; readonly w: Float64Array } {
  if (variance <= 0) return { x: Float64Array.of(mean), w: Float64Array.of(1) }
  const { x, w } = hermite(n)
  const s = Math.sqrt(2 * variance)
  return { x: Float64Array.from(x, (v) => mean + s * v), w }
}

/** The expectation E[g(x)] for x ~ N(mean, variance). */
export function gaussianExpectation(mean: number, variance: number, g: (x: number) => number, n = 48): number {
  const { x, w } = gaussianNodes(mean, variance, n)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (w[i] ?? 0) * g(x[i] ?? 0)
  return s
}

/** The mass E[g] and the mean and variance of the density proportional to N(mean, variance) g(x). */
export function tiltedGaussian(
  mean: number,
  variance: number,
  g: (x: number) => number,
  n = 48,
): { readonly mass: number; readonly mean: number; readonly variance: number } {
  const { x, w } = gaussianNodes(mean, variance, n)
  const gv = Float64Array.from(x, (xi) => g(xi))
  let m0 = 0
  let m1 = 0
  for (let i = 0; i < x.length; i++) {
    const v = (w[i] ?? 0) * (gv[i] ?? 0)
    m0 += v
    m1 += v * (x[i] ?? 0)
  }
  const mu = m1 / m0
  let m2 = 0
  for (let i = 0; i < x.length; i++) m2 += (w[i] ?? 0) * (gv[i] ?? 0) * ((x[i] ?? 0) - mu) ** 2
  return { mass: m0, mean: mu, variance: m2 / m0 }
}

const legendre = new Map<number, { readonly x: Float64Array; readonly w: Float64Array }>()

/** The nodes and weights of the Gauss-Legendre rule of `order` points on [-1, 1], made once. */
export function legendreRule(order: number): { readonly x: Float64Array; readonly w: Float64Array } {
  let rule = legendre.get(order)
  if (rule === undefined) {
    const g = gaussLegendre(order)
    rule = { x: Float64Array.from(toFlat(g.nodes)), w: Float64Array.from(toFlat(g.weights)) }
    legendre.set(order, rule)
  }
  return rule
}

/**
 * Nodes and weights of a composite Gauss-Legendre rule over mean +- 9 standard deviations, with the Gaussian density
 * folded into the weights: for integrands that are not smooth enough for Gauss-Hermite (rational in the variable,
 * kinked), which a panelled rule resolves.
 *
 * @param mean - The mean.
 * @param variance - The variance (non-negative).
 * @param panels - The number of panels (default 24).
 * @param order - Nodes per panel (default 10).
 * @returns Nodes and weights approximating E[g(x)] = sum w g(x).
 */
export function gaussianGrid(
  mean: number,
  variance: number,
  panels = 24,
  order = 10,
): { readonly x: Float64Array; readonly w: Float64Array } {
  if (variance <= 0) return { x: Float64Array.of(mean), w: Float64Array.of(1) }
  const rule = legendreRule(order)
  const sd = Math.sqrt(variance)
  const lo = mean - 9 * sd
  const h = (18 * sd) / panels
  const x = new Float64Array(panels * order)
  const w = new Float64Array(panels * order)
  for (let p = 0; p < panels; p++)
    for (let i = 0; i < order; i++) {
      const xi = lo + p * h + ((rule.x[i] ?? 0) + 1) * 0.5 * h
      x[p * order + i] = xi
      w[p * order + i] =
        ((rule.w[i] ?? 0) * 0.5 * h * Math.exp(-0.5 * ((xi - mean) / sd) ** 2)) / (sd * Math.sqrt(2 * Math.PI))
    }
  return { x, w }
}

/** Like {@link tiltedGaussian}, on the panelled grid of {@link gaussianGrid}. */
export function tiltedGrid(
  mean: number,
  variance: number,
  g: (x: number) => number,
): { readonly mass: number; readonly mean: number; readonly variance: number } {
  const { x, w } = gaussianGrid(mean, variance)
  const gv = Float64Array.from(x, (xi) => g(xi))
  let m0 = 0
  let m1 = 0
  for (let i = 0; i < x.length; i++) {
    m0 += (w[i] ?? 0) * (gv[i] ?? 0)
    m1 += (w[i] ?? 0) * (gv[i] ?? 0) * (x[i] ?? 0)
  }
  const mu = m1 / m0
  let m2 = 0
  for (let i = 0; i < x.length; i++) m2 += (w[i] ?? 0) * (gv[i] ?? 0) * ((x[i] ?? 0) - mu) ** 2
  return { mass: m0, mean: mu, variance: m2 / m0 }
}

/**
 * Nodes and weights of a composite Gauss-Legendre rule on [lo, hi], with the Gaussian density N(mean, variance)
 * folded into the weights: the integral of g against the density over a sub-interval, for zooming in on where an
 * integrand lives inside a much wider message.
 *
 * @param mean - The mean of the density.
 * @param variance - Its variance.
 * @param lo - Lower end of the interval.
 * @param hi - Upper end.
 * @param panels - Panels (default 12).
 * @param order - Nodes per panel (default 8).
 * @returns Nodes and weights.
 */
export function gaussianGridOn(
  mean: number,
  variance: number,
  lo: number,
  hi: number,
  panels = 12,
  order = 8,
): { readonly x: Float64Array; readonly w: Float64Array } {
  const rule = legendreRule(order)
  const h = (hi - lo) / panels
  const x = new Float64Array(panels * order)
  const w = new Float64Array(panels * order)
  const sd = Math.sqrt(variance)
  for (let p = 0; p < panels; p++)
    for (let i = 0; i < order; i++) {
      const xi = lo + p * h + ((rule.x[i] ?? 0) + 1) * 0.5 * h
      x[p * order + i] = xi
      w[p * order + i] =
        ((rule.w[i] ?? 0) * 0.5 * h * Math.exp(-0.5 * ((xi - mean) / sd) ** 2)) / (sd * Math.sqrt(2 * Math.PI))
    }
  return { x, w }
}

/**
 * The log mass and the mean and variance of the density proportional to N(mean, variance) exp(logg(x)), on the
 * panelled grid of {@link gaussianGrid} with the log weights shifted by their maximum (so a factor far in the tail of the
 * message, or raised to a power, does not underflow). The rule of the powered links (`probit`, `logistic` with a power).
 *
 * @param mean - The mean of the Gaussian.
 * @param variance - Its variance (positive).
 * @param logg - The log of the function that tilts it.
 * @returns The log of E[exp(logg)] and the moments of the tilted density.
 */
export function tiltedLogGrid(
  mean: number,
  variance: number,
  logg: (x: number) => number,
): { readonly logMass: number; readonly mean: number; readonly variance: number } {
  const { x, w } = gaussianGrid(mean, variance)
  const lw = Float64Array.from(x, (xi, i) => Math.log(w[i] ?? 0) + logg(xi))
  const { weights, total: m0, logTotal } = shiftedSum(lw)
  let m1 = 0
  for (let i = 0; i < x.length; i++) m1 += (weights[i] as number) * (x[i] ?? 0)
  const mu = m1 / m0
  let m2 = 0
  for (let i = 0; i < x.length; i++) m2 += (weights[i] as number) * ((x[i] ?? 0) - mu) ** 2
  return { logMass: logTotal, mean: mu, variance: m2 / m0 }
}
