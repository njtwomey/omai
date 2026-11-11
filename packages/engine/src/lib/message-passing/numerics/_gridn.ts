// A multi-dimensional (up to three coordinates) integral of a log density against the cavities of several variables, by panelled
// Gauss-Legendre grids that are zoomed onto where the mass is: what the covariance sites that have a precision, a second precision
// and a correlation as separate variables integrate with. The two-coordinate version of the likelihood sites is `_grid.ts`; this one
// takes any number of axes (an observed or known coordinate is a single node), a coordinate being the value of a Gaussian, the log of
// a gamma variable or the logit of a beta variable, each with the Jacobian in its density.

import { OmaiError } from '../../base/index.ts'
import type { GridAxis } from './_grid.ts'
import { legendreRule } from './_quadrature.ts'
import { digammaOf, lgammaOf, softplusOf, trigammaOf } from './_scalar.ts'

/** The axis of a beta cavity `Beta(a, b)` in the coordinate `s = logit(u)`: the density of `s` is `u^a (1 - u)^b` over the beta function. */
export function betaAxis(a: number, b: number): GridAxis {
  const centre = digammaOf(a) - digammaOf(b)
  const sd = Math.sqrt(trigammaOf(a) + trigammaOf(b))
  const logNorm = lgammaOf(a) + lgammaOf(b) - lgammaOf(a + b)
  return {
    point: false,
    lo: Math.max(centre - 10 * sd, -40),
    hi: Math.min(centre + 10 * sd, 40),
    kinks: [],
    // u = 1 / (1 + e^-s): log u = -softplus(-s), log (1 - u) = -softplus(s)
    logWeight: (s) => -a * softplusOf(-s) - b * softplusOf(s) - logNorm,
  }
}

/** The result of the integral: the log mass, and over the final grid the nodes of each axis, the normalised weights and the marginal weights per axis. */
export interface GridNResult {
  readonly logMass: number
  /** The coordinates of the nodes of each axis. */
  readonly nodes: readonly Float64Array[]
  /** The normalised weight of the node `(i0, i1, i2)` at `i0 + n0 (i1 + n1 i2)` (missing axes have one node). */
  readonly weights: Float64Array
  /** The marginal weights of the nodes of each axis. */
  readonly marginals: readonly Float64Array[]
}

const COARSE = { panels: 4, order: 4 }
const FINE = { panels: 6, order: 4 }
const PASSES = 6

function panelNodes(lo: number, hi: number, shape: { panels: number; order: number }): { x: number[]; w: number[] } {
  const r = legendreRule(shape.order)
  const x: number[] = []
  const w: number[] = []
  const h = (hi - lo) / shape.panels
  for (let p = 0; p < shape.panels; p++)
    for (let i = 0; i < shape.order; i++) {
      x.push(lo + p * h + ((r.x[i] as number) + 1) * 0.5 * h)
      w.push((r.w[i] as number) * 0.5 * h)
    }
  return { x, w }
}

/**
 * The integral of `exp(logf(t))` against the cavities of up to three axes, by a coarse grid that finds the window the mass lives in
 * (each axis's window is shrunk to where its marginal weight is not negligible, until it settles) and a finer grid over it.
 *
 * @param axes - The coordinates (one to three).
 * @param logf - The log of the integrand at the coordinates `t[0], t[1], ...` (the cavities' densities are separate).
 * @returns The log mass and the weights of the final grid.
 * @throws OmaiError with code `value` when the integrand has no mass under the cavities.
 */
export function integrateAxes(axes: readonly GridAxis[], logf: (t: Float64Array) => number): GridNResult {
  const k = axes.length
  if (k < 1 || k > 3) throw new OmaiError('value', 'integrateAxes: one to three axes')
  let window: [number, number][] = axes.map((a) => [a.lo, a.hi])
  let result: GridNResult | undefined
  let fine = false
  for (let pass = 0; pass < PASSES; pass++) {
    const shape = fine || pass === PASSES - 1 ? FINE : COARSE
    const grids = axes.map((a, i) => {
      if (a.point) return { x: [a.lo], logw: [0] }
      const [lo, hi] = window[i] as [number, number]
      const g = panelNodes(lo, hi, shape)
      return { x: g.x, logw: g.x.map((xi, q) => Math.log(g.w[q] as number) + a.logWeight(xi)) }
    })
    const sizes = grids.map((g) => g.x.length)
    const n0 = sizes[0] as number
    const n1 = sizes[1] ?? 1
    const n2 = sizes[2] ?? 1
    const lw = new Float64Array(n0 * n1 * n2)
    const t = new Float64Array(k)
    let top = Number.NEGATIVE_INFINITY
    for (let c = 0; c < n2; c++)
      for (let b = 0; b < n1; b++)
        for (let a = 0; a < n0; a++) {
          t[0] = (grids[0] as { x: number[] }).x[a] as number
          if (k > 1) t[1] = (grids[1] as { x: number[] }).x[b] as number
          if (k > 2) t[2] = (grids[2] as { x: number[] }).x[c] as number
          const v =
            ((grids[0] as { logw: number[] }).logw[a] as number) +
            (k > 1 ? ((grids[1] as { logw: number[] }).logw[b] as number) : 0) +
            (k > 2 ? ((grids[2] as { logw: number[] }).logw[c] as number) : 0) +
            logf(t)
          lw[a + n0 * (b + n1 * c)] = v
          if (v > top) top = v
        }
    if (!Number.isFinite(top))
      throw new OmaiError('value', 'numeric rule: the factor has no mass under the incoming messages')
    let total = 0
    for (let i = 0; i < lw.length; i++) {
      const e = Math.exp((lw[i] as number) - top)
      lw[i] = e
      total += e
    }
    const marginals = sizes.map((n) => new Float64Array(n))
    for (let c = 0; c < n2; c++)
      for (let b = 0; b < n1; b++)
        for (let a = 0; a < n0; a++) {
          const w = (lw[a + n0 * (b + n1 * c)] as number) / total
          lw[a + n0 * (b + n1 * c)] = w
          const m0 = marginals[0] as Float64Array
          m0[a] = (m0[a] as number) + w
          if (k > 1) {
            const m1 = marginals[1] as Float64Array
            m1[b] = (m1[b] as number) + w
          }
          if (k > 2) {
            const m2 = marginals[2] as Float64Array
            m2[c] = (m2[c] as number) + w
          }
        }
    result = {
      logMass: top + Math.log(total),
      nodes: grids.map((g) => Float64Array.from(g.x)),
      weights: lw,
      marginals,
    }
    let shrunk = false
    const next = axes.map((a, i): [number, number] => {
      const [lo, hi] = window[i] as [number, number]
      if (a.point) return [lo, hi]
      const x = (grids[i] as { x: number[] }).x
      const m = marginals[i] as Float64Array
      const peak = m.reduce((p, q) => Math.max(p, q), 0)
      let first = 0
      while (first < x.length && (m[first] as number) < 1e-9 * peak) first++
      let last = x.length - 1
      while (last > first && (m[last] as number) < 1e-9 * peak) last--
      const spacing = (hi - lo) / x.length
      const nlo = Math.max(a.lo, (x[Math.max(first - 1, 0)] as number) - spacing)
      const nhi = Math.min(a.hi, (x[Math.min(last + 1, x.length - 1)] as number) + spacing)
      if (!(nhi > nlo)) return [lo, hi]
      if (nhi - nlo < 0.7 * (hi - lo)) shrunk = true
      return [nlo, nhi]
    })
    if (!shrunk) {
      if (fine || shape === FINE) break
      fine = true
      continue
    }
    window = next
  }
  return result as GridNResult
}
