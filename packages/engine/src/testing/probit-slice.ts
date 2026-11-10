// The probit regression of the probabilistic-programming vertical slice, shared by the tests of the factor-graph compiler and
// of the message-passing methods: the dataset (an intercept and two covariates, labels from known weights), the model written
// in the builder, an exact posterior on a grid, and a table that compares what several methods say with the grid. Node only;
// not published.
import { logNdtr } from '../lib/numerics/special/index.ts'
import * as d from '../lib/probprog/model/dists/index.ts'
import { defineModel, type ModelDefinition, ops } from '../lib/probprog/model/index.ts'
import type { Posterior } from '../lib/probprog/posterior.ts'
import { foldIn, key, normal as normalDraws, uniform } from '../lib/random/index.ts'
import { fromStorage, toFlat } from '../lib/tensor/index.ts'

/** Forty rows: an intercept and two covariates, labels drawn from a probit model with weights 0.3, 1.2 and -0.8. */
export function probitData(): { readonly X: number[][]; readonly y: number[]; readonly truth: number[] } {
  const k = key(2026)
  const rows = 40
  const z = toFlat(normalDraws(foldIn(k, 1), [rows, 2]))
  const u = toFlat(uniform(foldIn(k, 2), [rows]))
  const truth = [0.3, 1.2, -0.8]
  const X = Array.from({ length: rows }, (_, i) => [1, z[2 * i] as number, z[2 * i + 1] as number])
  const y = X.map((row, i) => {
    const eta = row.reduce((a, v, j) => a + v * (truth[j] as number), 0)
    return (u[i] as number) < Math.exp(toFlat(logNdtr(eta))[0] as number) ? 1 : 0
  })
  return { X, y, truth }
}

/** The probit regression in the model builder: weights over `feature`, a Bernoulli likelihood with the probit link over `row`. */
export const probitModel: ModelDefinition = defineModel('probit', (m) => {
  const n = m.range('row')
  const f = m.range('feature')
  const X = m.data('X', { dims: [n, f], role: 'covariate' })
  const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
  const w = m.latent('w', d.normal(0, 2), { dims: [f] })
  m.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
})

/** The exact posterior of the three weights: moments and the log evidence. */
export interface GridPosterior {
  readonly mean: number[]
  readonly sd: number[]
  readonly logEvidence: number
}

/**
 * The posterior mean and standard deviation of three probit weights under N(0, priorSd^2) priors, and the log evidence, on a
 * grid: a coarse pass over the prior's range, then two over the region the posterior occupies. Exact to the grid's step.
 */
export function probitGrid(X: number[][], y: number[], priorSd: number): GridPosterior {
  const logPosterior = (points: number[][]): Float64Array => {
    const zs = new Float64Array(points.length * X.length)
    points.forEach((w, p) => {
      X.forEach((row, i) => {
        const t = row.reduce((a, v, j) => a + v * (w[j] as number), 0)
        zs[p * X.length + i] = y[i] === 1 ? t : -t
      })
    })
    const logs = toFlat(logNdtr(fromStorage(zs, [zs.length], 'float64')))
    return Float64Array.from(points, (w, p) => {
      let s = -0.5 * (w.reduce((a, v) => a + v * v, 0) / priorSd ** 2) - 1.5 * Math.log(2 * Math.PI * priorSd ** 2)
      for (let i = 0; i < X.length; i++) s += logs[p * X.length + i] as number
      return s
    })
  }
  let lo = [-4 * priorSd, -4 * priorSd, -4 * priorSd]
  let hi = [4 * priorSd, 4 * priorSd, 4 * priorSd]
  let result: GridPosterior = { mean: [0, 0, 0], sd: [1, 1, 1], logEvidence: 0 }
  for (const m of [21, 31, 31]) {
    const axes = [0, 1, 2].map((q) =>
      Array.from({ length: m }, (_, i) => (lo[q] as number) + (((hi[q] as number) - (lo[q] as number)) * i) / (m - 1)),
    )
    const cell = [0, 1, 2].reduce((a, q) => a * (((hi[q] as number) - (lo[q] as number)) / (m - 1)), 1)
    const points: number[][] = []
    for (const a of axes[0] as number[])
      for (const b of axes[1] as number[]) for (const c of axes[2] as number[]) points.push([a, b, c])
    const lp = logPosterior(points)
    const top = Math.max(...lp)
    let z = 0
    const m1 = [0, 0, 0]
    const m2 = [0, 0, 0]
    lp.forEach((v, p) => {
      const wt = Math.exp(v - top)
      z += wt
      for (let q = 0; q < 3; q++) {
        const c = (points[p] as number[])[q] as number
        m1[q] = (m1[q] as number) + wt * c
        m2[q] = (m2[q] as number) + wt * c * c
      }
    })
    const mean = m1.map((v) => v / z)
    const sd = m2.map((v, q) => Math.sqrt(v / z - (mean[q] as number) ** 2))
    result = { mean, sd, logEvidence: top + Math.log(z * cell) }
    lo = mean.map((v, q) => v - 7 * (sd[q] as number))
    hi = mean.map((v, q) => v + 7 * (sd[q] as number))
  }
  return result
}

/** One row of the moments table: what a method says of one weight, against the grid. */
export interface MomentRow {
  readonly method: string
  readonly coordinate: string
  readonly mean: number
  readonly sd: number
  /** The error of the mean in posterior standard deviations of the grid answer. */
  readonly meanError: number
  /** The standard deviation over the grid's. */
  readonly sdRatio: number
  /** KL(grid || method) between the Gaussians with those moments, in nats. */
  readonly kl: number
}

/**
 * The moments of site `w` under several posteriors, compared with the grid: the error of each mean in grid standard
 * deviations, the ratio of spreads and the Gaussian KL from the grid answer. An `agree` style table for the three weights.
 */
export function momentTable(grid: GridPosterior, posteriors: Readonly<Record<string, Posterior>>): MomentRow[] {
  const rows: MomentRow[] = []
  for (const [method, p] of Object.entries(posteriors)) {
    const mean = toFlat(p.mean('w'))
    const sd = toFlat(p.stddev('w'))
    for (let q = 0; q < 3; q++) {
      const m = mean[q] as number
      const s = sd[q] as number
      const m0 = grid.mean[q] as number
      const s0 = grid.sd[q] as number
      rows.push({
        method,
        coordinate: `w[${q}]`,
        mean: m,
        sd: s,
        meanError: (m - m0) / s0,
        sdRatio: s / s0,
        kl: Math.log(s / s0) + (s0 ** 2 + (m0 - m) ** 2) / (2 * s ** 2) - 0.5,
      })
    }
  }
  return rows
}
