/**
 * The radon-style hierarchical regression: log radon in houses nested in counties, a varying intercept per county
 * (`alpha_j ~ N(mu, tau²)`) and a common floor effect, on balanced simulated data (Gelman and Hill's radon example, with the
 * counties and the sizes of the groups chosen to keep it small). The truth is exact to quadrature without any sampler: given
 * the two scales `(tau, sigma)` the model is linear-Gaussian, so the posterior of every location parameter is a closed-form
 * normal, and the posterior of the scales is the two-dimensional density `p(tau, sigma | y)` (the marginal likelihood is a
 * Gaussian one, by the matrix determinant lemma), integrated on a grid. Each location parameter's marginal is then the grid
 * mixture of those normals.
 */

import { dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type MarginalOracle, weightedOracle } from '../validation/index.ts'
import {
  type BenchmarkData,
  type BenchmarkModel,
  type BenchmarkTruth,
  defineBenchmark,
  type TruthQuality,
} from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { mixtureOracle } from './truth.ts'

const RADON = { mu: 1.5, beta: -0.6, tau: 0.4, sigma: 0.7 }
const PRIOR = { mu: 5, beta: 2, tau: 1, sigma: 1 }

function generateRadon(key: Key, size = 6): BenchmarkData {
  const counties = size
  const houses = 8
  const e = toFlat(drawNormal(foldIn(key, 0), [counties]))
  const noise = toFlat(drawNormal(foldIn(key, 1), [counties, houses]))
  const u = toFlat(drawUniform(foldIn(key, 2), [counties, houses]))
  const alpha = Array.from(e, (v) => RADON.mu + RADON.tau * v)
  const floor: number[][] = []
  const y: number[][] = []
  for (let j = 0; j < counties; j++) {
    floor.push(Array.from({ length: houses }, (_, i) => ((u[j * houses + i] ?? 1) < 0.5 ? 1 : 0)))
    y.push(floor[j]?.map((f, i) => (alpha[j] ?? 0) + RADON.beta * f + RADON.sigma * (noise[j * houses + i] ?? 0)) ?? [])
  }
  return { data: { floor, y }, generating: { ...RADON, alpha } }
}

/** What the exact posterior of a varying-intercept regression needs: the data by county and the scales of the priors. */
export interface VaryingInterceptProblem {
  /** The responses `[county][house]`. */
  readonly y: readonly (readonly number[])[]
  /** The covariate `[county][house]` (a floor indicator). */
  readonly floor: readonly (readonly number[])[]
  /** The sds of the normal priors of the common intercept and the slope. */
  readonly priorMu: number
  readonly priorBeta: number
  /** The scales of the half-normal priors of the county sd `tau` and the noise sd `sigma`. */
  readonly priorTau: number
  readonly priorSigma: number
}

/**
 * The exact posterior of the varying-intercept regression `y_ji = alpha_j + beta floor_ji + eps_ji`, `alpha_j = mu + tau
 * e_j`, with `mu ~ N(0, s_mu²)`, `beta ~ N(0, s_beta²)`, `tau ~ HalfNormal(s_tau)`, `sigma ~ HalfNormal(s_sigma)`, to quadrature
 * accuracy: for each `(log tau, log sigma)` on a grid the posterior of `(mu, beta, e)` is the normal with precision `D^-1 +
 * A'A / sigma²` (`A = [1 floor, tau Z]`), the marginal likelihood is `N(y; 0, sigma² I + A D A')` through the matrix
 * determinant lemma, and the location parameters' marginals are the grid mixtures of the conditional normals.
 *
 * @param problem - The data and the prior scales.
 * @param quality - `fast` for a coarse grid (unit tests), `full` for the committed resolution.
 * @returns The truth: marginals of `mu`, `beta`, `tau`, `sigma` and `alpha[j]`, and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { varyingInterceptTruth } from 'omai-engine/probprog/models'
 * const t = varyingInterceptTruth({ y: [[1, 1.2], [2, 2.1]], floor: [[0, 1], [0, 1]], priorMu: 5, priorBeta: 2, priorTau: 1, priorSigma: 1 }, 'fast')
 * Object.keys(t.marginals) // => ['mu', 'beta', 'tau', 'sigma', 'alpha[0]', 'alpha[1]']
 * ```
 */
export function varyingInterceptTruth(
  problem: VaryingInterceptProblem,
  quality: TruthQuality = 'full',
): BenchmarkTruth {
  const J = problem.y.length
  const m = problem.y[0]?.length ?? 0
  const n = J * m
  const p = 2 + J
  const y = problem.y.flat()
  const floor = problem.floor.flat()
  const sum = (xs: readonly number[]): number => xs.reduce((acc, v) => acc + v, 0)
  const precisionMu = 1 / problem.priorMu ** 2
  const precisionBeta = 1 / problem.priorBeta ** 2
  const logDetD = Math.log(problem.priorMu ** 2) + Math.log(problem.priorBeta ** 2)
  // The sufficient statistics of the columns [1, floor] and of the county indicators: the data enter only through them.
  const yy = sum(y.map((v) => v * v))
  const sumY = sum(y)
  const sumFloor = sum(floor)
  const sumFloorSq = sum(floor.map((v) => v * v))
  const sumFloorY = sum(y.map((v, i) => v * (floor[i] ?? 0)))
  const groupY = problem.y.map((row) => sum(row))
  const groupFloor = problem.floor.map((row) => sum(row))

  const N = quality === 'fast' ? 41 : 101
  const evalPoint = (u: number, v: number): { logPost: number; mean: Float64Array; cov: Float64Array } => {
    const tau = Math.exp(u)
    const sigma = Math.exp(v)
    const s2 = sigma * sigma
    // A'A / sigma² + D^-1 over (mu, beta, e_1..e_J), with A = [1, floor, tau Z]; and A'y / sigma².
    const lambda = new Float64Array(p * p)
    lambda[0] = precisionMu + n / s2
    lambda[1] = sumFloor / s2
    lambda[p] = sumFloor / s2
    lambda[p + 1] = precisionBeta + sumFloorSq / s2
    const rhs = new Float64Array(p)
    rhs[0] = sumY / s2
    rhs[1] = sumFloorY / s2
    for (let j = 0; j < J; j++) {
      const c = 2 + j
      const withMu = (tau * m) / s2
      const withBeta = (tau * (groupFloor[j] ?? 0)) / s2
      lambda[c] = withMu
      lambda[c * p] = withMu
      lambda[p + c] = withBeta
      lambda[c * p + 1] = withBeta
      lambda[c * p + c] = 1 + (tau * tau * m) / s2
      rhs[c] = (tau * (groupY[j] ?? 0)) / s2
    }
    const { factor, info } = dpotrf(lambda, p)
    if (info !== 0)
      return { logPost: Number.NEGATIVE_INFINITY, mean: new Float64Array(p), cov: new Float64Array(p * p) }
    let logDetLambda = 0
    for (let i = 0; i < p; i++) logDetLambda += 2 * Math.log(factor[i * p + i] ?? 1)
    const mean = dpotrs(factor, rhs, p)
    let quad = 0
    for (let i = 0; i < p; i++) quad += (rhs[i] ?? 0) * (mean[i] ?? 0)
    // y'y - (A'y)' Sigma (A'y) / sigma² = y'y - rhs' mean * sigma² ... both over sigma²: (y'y - sigma² rhs'mean) / sigma².
    const quadForm = (yy - s2 * quad) / s2
    const logLik = -0.5 * (n * Math.log(2 * Math.PI) + n * Math.log(s2) + logDetD + logDetLambda + quadForm)
    const halfNormal = (x: number, scale: number): number =>
      Math.log(2 / (scale * Math.sqrt(2 * Math.PI))) - (x * x) / (2 * scale * scale)
    const logPost = logLik + halfNormal(tau, problem.priorTau) + halfNormal(sigma, problem.priorSigma) + u + v
    const cov = new Float64Array(p * p)
    for (let c = 0; c < p; c++) {
      const e = new Float64Array(p)
      e[c] = 1
      const col = dpotrs(factor, e, p)
      for (let r = 0; r < p; r++) cov[r * p + c] = col[r] ?? 0
    }
    return { logPost, mean, cov }
  }

  const range = (lo: number, hi: number, k: number): Float64Array =>
    Float64Array.from({ length: k }, (_, i) => lo + ((hi - lo) * i) / (k - 1))
  // Coarse pass to find where the mass is.
  const coarseU = range(-9, 3, 61)
  const coarseV = range(-5, 3, 61)
  let best = Number.NEGATIVE_INFINITY
  const coarse: { u: number; v: number; lp: number }[] = []
  for (const u of coarseU)
    for (const v of coarseV) {
      const lp = evalPoint(u, v).logPost
      coarse.push({ u, v, lp })
      if (lp > best) best = lp
    }
  const live = coarse.filter((c) => c.lp > best - 40)
  const pad = 0.6
  const uLo = Math.min(...live.map((c) => c.u)) - pad
  const uHi = Math.max(...live.map((c) => c.u)) + pad
  const vLo = Math.min(...live.map((c) => c.v)) - pad
  const vHi = Math.max(...live.map((c) => c.v)) + pad
  const us = range(uLo, uHi, N)
  const vs = range(vLo, vHi, N)
  const hu = (uHi - uLo) / (N - 1)
  const hv = (vHi - vLo) / (N - 1)
  interface Cell {
    readonly tau: number
    readonly sigma: number
    readonly logw: number
    readonly mean: Float64Array
    readonly cov: Float64Array
  }
  const cells: Cell[] = []
  let maxLog = Number.NEGATIVE_INFINITY
  us.forEach((u, a) => {
    vs.forEach((v, b) => {
      const { logPost, mean, cov } = evalPoint(u, v)
      const edge = (a === 0 || a === N - 1 ? 0.5 : 1) * (b === 0 || b === N - 1 ? 0.5 : 1)
      const logw = logPost + Math.log(edge * hu * hv)
      if (logw > maxLog) maxLog = logw
      cells.push({ tau: Math.exp(u), sigma: Math.exp(v), logw, mean, cov })
    })
  })
  const weights = cells.map((c) => Math.exp(c.logw - maxLog))
  const total = weights.reduce((a, w) => a + w, 0)
  const logEvidence = maxLog + Math.log(total)
  const keep = weights.map((w, i) => (w / total > 1e-9 ? i : -1)).filter((i) => i >= 0)
  const kept = keep.map((i) => ({ cell: cells[i] as Cell, weight: (weights[i] ?? 0) / total }))
  // The marginal weights of one scale: the grid summed over the other (cells run over u outermost, v innermost).
  const margin = (axis: 0 | 1): number[] => {
    const out = new Array<number>(N).fill(0)
    weights.forEach((w, i) => {
      const k = axis === 0 ? Math.floor(i / N) : i % N
      out[k] = (out[k] ?? 0) + w / total
    })
    return out
  }
  const scalar = (name: string, f: (c: Cell) => { mean: number; variance: number }): ReturnType<typeof mixtureOracle> =>
    mixtureOracle(
      name,
      kept.map(({ cell, weight }) => {
        const g = f(cell)
        return { weight, mean: g.mean, sd: Math.sqrt(Math.max(g.variance, 1e-300)) }
      }),
    )
  const marginals: Record<string, MarginalOracle> = {
    mu: scalar('mu', (c) => ({ mean: c.mean[0] ?? 0, variance: c.cov[0] ?? 0 })),
    beta: scalar('beta', (c) => ({ mean: c.mean[1] ?? 0, variance: c.cov[p + 1] ?? 0 })),
    tau: weightedOracle('tau', us.map(Math.exp), margin(0)),
    sigma: weightedOracle('sigma', vs.map(Math.exp), margin(1)),
  }
  for (let j = 0; j < J; j++) {
    const c = 2 + j
    // alpha_j = mu + tau e_j: a linear functional of the normal (mu, beta, e) given the scales.
    marginals[`alpha[${j}]`] = scalar(`alpha[${j}]`, (cell) => ({
      mean: (cell.mean[0] ?? 0) + cell.tau * (cell.mean[c] ?? 0),
      variance:
        (cell.cov[0] ?? 0) + 2 * cell.tau * (cell.cov[c] ?? 0) + cell.tau * cell.tau * (cell.cov[c * p + c] ?? 0),
    }))
  }
  return {
    kind: 'quadrature',
    source: `grid quadrature of (log tau, log sigma) (${N} x ${N} points over the box that holds the mass), the location parameters integrated out in closed form: the posterior of (mu, beta, e) given the scales is normal, the marginal likelihood N(y; 0, sigma² I + A D A')`,
    logEvidence,
    marginals,
  }
}

const cache = new WeakMap<BenchmarkData, Map<TruthQuality, BenchmarkTruth>>()

function radonTruth(data: BenchmarkData, quality: TruthQuality): BenchmarkTruth {
  const hit = cache.get(data)?.get(quality)
  if (hit !== undefined) return hit
  const truth = varyingInterceptTruth(
    {
      y: data.data['y'] as number[][],
      floor: data.data['floor'] as number[][],
      priorMu: PRIOR.mu,
      priorBeta: PRIOR.beta,
      priorTau: PRIOR.tau,
      priorSigma: PRIOR.sigma,
    },
    quality,
  )
  const byQuality = cache.get(data) ?? new Map<TruthQuality, BenchmarkTruth>()
  byQuality.set(quality, truth)
  cache.set(data, byQuality)
  return truth
}

/**
 * Hierarchical radon regression, centred: log radon `y_ji ~ N(alpha_j + beta floor_ji, sigma²)` in six counties of eight
 * houses, `alpha_j ~ N(mu, tau²)`, `mu ~ N(0, 5²)`, `beta ~ N(0, 2²)`, `tau, sigma ~ HalfNormal(1)`. Simulated at `mu = 1.5`,
 * `beta = -0.6`, `tau = 0.4`, `sigma = 0.7`. The model is the plan's example of a Gaussian block (the county effects) that
 * could be collapsed; the truth is exact (see `varyingInterceptTruth`). NUTS agrees; the centred form is worse than the
 * non-centred one on the poorly identified `tau`, and Laplace and mean-field underestimate its spread.
 *
 * @concept posterior-validation
 * @index dataset Hierarchical radon-style regression with varying intercepts; the exact truth integrates the effects out.
 * @example
 * ```ts
 * import { radon } from 'omai-engine/probprog/models'
 * radon.bound().sizes // => { county: 6, house: 8 }
 * ```
 */
export const radon: BenchmarkModel = defineBenchmark({
  name: 'radon',
  title: 'Hierarchical radon regression',
  description:
    'Log radon in houses nested in counties: a varying intercept per county and a common floor effect, with unknown scales. A Gaussian block that could be collapsed, with an exact truth.',
  tags: ['hierarchical', 'varying-intercepts', 'linear-gaussian', 'regression', 'collapse', 'quadrature'],
  references: [
    'Gelman and Hill (2007), Data Analysis Using Regression and Multilevel/Hierarchical Models, chapters 12 and 13 (the radon example)',
    'Gelman et al., Bayesian Data Analysis, 3rd ed., chapter 15 (hierarchical linear models)',
  ],
  definition: defineModel('radon', (b) => {
    const county = b.range('county')
    const house = b.range('house')
    const floor = b.data('floor', {
      dims: [county, house],
      support: 'binary',
      role: 'covariate',
      doc: '1 for a house measured on the first floor',
    })
    const y = b.data('y', { dims: [county, house], role: 'response', doc: 'log radon' })
    const mu = b.latent('mu', d.normal(0, PRIOR.mu), { doc: 'the mean county intercept' })
    const beta = b.latent('beta', d.normal(0, PRIOR.beta), { doc: 'the floor effect' })
    const tau = b.latent('tau', d.halfNormal(PRIOR.tau), { doc: 'the sd of the county intercepts' })
    const sigma = b.latent('sigma', d.halfNormal(PRIOR.sigma), { doc: 'the sd of a house within its county' })
    // The scales are clipped to [1e-6, 1e6] where they enter a distribution. This is a modelling floor, not a guard against
    // an invalid parameter (the sampler treats that as a divergence): the centred density is unbounded as tau goes to 0, and
    // without the floor NUTS is drawn into that funnel (63 divergences against 4, and a sigma outside its tolerance).
    const alpha = b.latent('alpha', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), {
      dims: [county],
      doc: 'the intercept of each county',
    })
    b.observe('y', d.normal(alpha.add(beta.mul(floor)), ops.clip(sigma, 1e-6, 1e6)), y)
  }),
  generate: generateRadon,
  truth: radonTruth,
  plans: (scale) => [
    nutsPlan(scale, {
      note: 'a four-scale hierarchy with six groups: agrees within Monte Carlo error (tau is poorly identified, so its effective sample size is the smallest)',
      tolerances: { meanError: 0.15, ks: 0.12, sdRatio: [0.8, 1.2] },
      config: { targetAcceptance: 0.9 },
    }),
    temperedPlan(scale, {
      expect: 'report',
      particles: 1500,
      note: 'particles with MCMC moves in ten dimensions: close, noisier than NUTS',
    }),
    laplacePlan(scale, {
      expect: 'differ',
      note: 'the mode of the centred density is at tau -> 0 (the density is unbounded there): the fit is not a posterior',
    }),
    sviPlan(scale, { note: 'a mean-field normal: underestimates the spread of tau and of the county intercepts' }),
  ],
})
