/**
 * Rubin's eight schools, the standard hierarchical benchmark, in its two parametrisations: centred (`theta_j ~ N(mu,
 * tau²)`, the funnel that defeats a sampler when `tau` is small) and non-centred (`theta_j = mu + tau z_j`, no funnel). The
 * truth is exact to quadrature without any sampler: the school effects integrate out in closed form, so the posterior of
 * `(mu, tau)` is a two-dimensional density (`eight-schools-marginal`, whose likelihood is `N(y_j; mu, sigma_j² + tau²)`)
 * that `gridPosterior` integrates, and each `theta_j` given `(mu, tau)` is normal with the usual precision-weighted mean,
 * so its marginal is the grid mixture of those normals. Both parametrisations have the same posterior, so one truth
 * serves both.
 */

import { bijectors } from '../../dists/index.ts'
import { normal as drawNormal, foldIn, type Key } from '../../random/index.ts'
import { item, tensor, toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition, ops, toProgram } from '../model/index.ts'
import { gridPosterior, type MarginalOracle } from '../validation/index.ts'
import {
  type BenchmarkData,
  type BenchmarkModel,
  type BenchmarkTruth,
  defineBenchmark,
  type TruthQuality,
} from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { conditionalOracle } from './truth.ts'

/** Rubin (1981) as printed in Gelman et al., Bayesian Data Analysis, section 5.5: estimated effects of coaching on the SAT. */
const SCHOOL_Y: readonly number[] = [28, 8, -3, 7, -1, 1, 18, 12]
/** The standard errors of those estimates (taken as known). */
const SCHOOL_SIGMA: readonly number[] = [15, 10, 16, 11, 9, 11, 10, 18]

const CLASSIC: BenchmarkData = { data: { y: SCHOOL_Y, sigma: SCHOOL_SIGMA }, generating: { source: 'Rubin (1981)' } }

/** Data like the classic ones from known `mu = 4` and `tau = 3`, `size` schools, with the classic standard errors repeated. */
function generateSchools(key: Key, size = 8): BenchmarkData {
  const mu = 4
  const tau = 3
  const sigma = Array.from({ length: size }, (_, j) => SCHOOL_SIGMA[j % SCHOOL_SIGMA.length] ?? 10)
  const z = toFlat(drawNormal(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  const theta = Array.from(z, (v) => mu + tau * v)
  return {
    data: { y: theta.map((t, j) => t + (sigma[j] ?? 10) * (e[j] ?? 0)), sigma },
    generating: { mu, tau, theta },
  }
}

/**
 * The marginal of the eight schools model over the school effects: `y_j ~ N(mu, sigma_j² + tau²)` with `mu ~ N(0, 5²)` and
 * `tau ~ HalfCauchy(5)`. It has the same posterior of `(mu, tau)` as the full model (the effects are integrated out in
 * closed form), in two coordinates instead of ten, which is what makes the truth a quadrature. It is also what `collapse`
 * of the Gaussian block gives.
 *
 * @concept posterior-validation
 * @example
 * ```ts
 * import { eightSchoolsMarginal } from 'omai-engine/probprog/models'
 * eightSchoolsMarginal.spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name) // => ['mu', 'tau']
 * ```
 */
export const eightSchoolsMarginal: ModelDefinition = defineModel('eight-schools-marginal', (m) => {
  const school = m.range('school')
  const y = m.data('y', { dims: [school], role: 'response' })
  const sigma = m.data('sigma', { dims: [school], support: 'positive', role: 'covariate' })
  const mu = m.latent('mu', d.normal(0, 5))
  const tau = m.latent('tau', d.halfCauchy(5))
  m.observe('y', d.normal(mu, ops.sqrt(ops.add(sigma.square(), tau.square()))), y)
})

const cache = new WeakMap<BenchmarkData, Map<TruthQuality, BenchmarkTruth>>()

/** The truth of the eight schools data, computed once per data and quality (both parametrisations share it). */
function schoolsTruth(data: BenchmarkData, quality: TruthQuality): BenchmarkTruth {
  const hit = cache.get(data)?.get(quality)
  if (hit !== undefined) return hit
  const bound = eightSchoolsMarginal.bind(data.data)
  const grid = gridPosterior(
    toProgram(bound),
    undefined,
    quality === 'fast' ? { points: 31, maxRefinements: 0 } : { points: 61, maxRefinements: 1 },
  )
  const y = data.data['y'] as number[]
  const sigma = data.data['sigma'] as number[]
  const positive = bijectors.bijectorForSupport({ kind: 'positive' })
  const tauOf = new Map<number, number>()
  const tau = (u: number): number => {
    let t = tauOf.get(u)
    if (t === undefined) {
      t = item(positive.forward(tensor(u)))
      tauOf.set(u, t)
    }
    return t
  }
  const muIndex = grid.names.indexOf('mu')
  const tauIndex = grid.names.indexOf('tau')
  const marginals: Record<string, MarginalOracle> = { mu: grid.marginal('mu'), tau: grid.marginal('tau') }
  const effects: Record<string, MarginalOracle> = {}
  y.forEach((yj, j) => {
    const s2 = (sigma[j] ?? 1) ** 2
    effects[`theta[${j}]`] = conditionalOracle(
      `theta[${j}]`,
      grid,
      (u) => {
        const t2 = Math.min(tau(u[tauIndex] ?? 0) ** 2, 1e100)
        const mu = u[muIndex] ?? 0
        // Precision-weighted: N(mu, tau²) combined with the observation N(y_j, sigma_j²), written to be stable as tau -> 0.
        return { mean: (mu * s2 + yj * t2) / (s2 + t2), sd: Math.max(Math.sqrt((s2 * t2) / (s2 + t2)), 1e-12) }
      },
      { points: quality === 'fast' ? 120 : 300 },
    )
  })
  const truth: BenchmarkTruth = {
    kind: 'quadrature',
    source: `grid quadrature of the posterior of (mu, tau) with the school effects integrated out in closed form (${grid.points} points per axis, evidence error ${grid.evidenceError.toExponential(1)}); each theta_j is the grid mixture of its conditional normals`,
    logEvidence: grid.logEvidence,
    marginals: { ...marginals, ...effects },
  }
  const byQuality = cache.get(data) ?? new Map<TruthQuality, BenchmarkTruth>()
  byQuality.set(quality, truth)
  cache.set(data, byQuality)
  return truth
}

/**
 * Eight schools, centred: `theta_j ~ N(mu, tau²)`, `y_j ~ N(theta_j, sigma_j²)` with `mu ~ N(0, 5²)` and `tau ~
 * HalfCauchy(5)` (Rubin's data, the standard errors known). The posterior of `tau` has mass near zero where the effects
 * are tied to `mu` and the density is a funnel: NUTS diverges there and its `tau` is biased, and a mean-field guide
 * underestimates `tau`'s spread; the exact truth shows by how much.
 *
 * @concept posterior-validation
 * @index dataset Rubin's eight schools, centred: the funnel; exact quadrature truth.
 * @example
 * ```ts
 * import { eightSchoolsCentred } from 'omai-engine/probprog/models'
 * eightSchoolsCentred.bound().sizes // => { school: 8 }
 * ```
 */
export const eightSchoolsCentred: BenchmarkModel = defineBenchmark({
  name: 'eight-schools-centred',
  title: 'Eight schools, centred',
  description:
    "Rubin's eight schools with theta_j ~ N(mu, tau²): the centred hierarchy, whose funnel makes a sampler diverge when tau is small.",
  tags: ['hierarchical', 'funnel', 'divergences', 'centred', 'quadrature', 'eight-schools'],
  references: [
    'Rubin (1981), Estimation in parallel randomized experiments, Journal of Educational Statistics 6',
    'Gelman et al., Bayesian Data Analysis, 3rd ed., section 5.5',
    'Betancourt and Girolami (2015), Hamiltonian Monte Carlo for hierarchical models',
  ],
  definition: defineModel('eight-schools-centred', (m) => {
    const school = m.range('school')
    const y = m.data('y', { dims: [school], role: 'response', doc: 'the estimated coaching effects' })
    const sigma = m.data('sigma', {
      dims: [school],
      support: 'positive',
      role: 'covariate',
      doc: 'their standard errors (known)',
    })
    const mu = m.latent('mu', d.normal(0, 5), { doc: 'the mean effect' })
    const tau = m.latent('tau', d.halfCauchy(5), { doc: 'the spread of the effects' })
    // The scale is clipped to [1e-6, 1e6] where it enters the distribution. This is a modelling floor, not a guard against an
    // invalid parameter (the sampler treats that as a divergence): the centred density is unbounded as tau goes to 0, and
    // without the floor the sampler is drawn into that funnel (18 divergences against 8).
    const theta = m.latent('theta', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), {
      dims: [school],
      doc: 'the effect of each school',
    })
    m.observe('y', d.normal(theta, sigma), y)
  }),
  canonical: CLASSIC,
  generate: generateSchools,
  truth: schoolsTruth,
  plans: (scale) => [
    nutsPlan(scale, {
      expect: 'report',
      note: 'divergent transitions near tau = 0 and a tau that is too large: the funnel (the non-centred model has none)',
      config: { targetAcceptance: 0.9 },
    }),
    temperedPlan(scale, {
      expect: 'report',
      particles: 1500,
      note: 'particles with MCMC moves in the funnel: noisier tau than the non-centred model',
    }),
    laplacePlan(scale, {
      expect: 'differ',
      note: 'the mode of the centred density is at tau -> 0 (the density is unbounded there): the fit is not a posterior',
    }),
    sviPlan(scale, {
      expect: 'differ',
      note: 'a mean-field normal on log tau: underestimates the spread of tau and of every effect',
    }),
  ],
})

/**
 * Eight schools, non-centred: `theta_j = mu + tau z_j` with `z_j ~ N(0, 1)`, the same posterior as the centred model with
 * the funnel removed (the sampler sees `z`, whose posterior is not tied to `tau`). NUTS has no divergences and agrees with
 * the truth within Monte Carlo error; Laplace and mean-field are better behaved than on the centred model.
 *
 * @concept posterior-validation
 * @index dataset Rubin's eight schools, non-centred: the same posterior without the funnel; truth by quadrature.
 * @example
 * ```ts
 * import { eightSchoolsNonCentred } from 'omai-engine/probprog/models'
 * eightSchoolsNonCentred.definition?.spec.nodes.filter((n) => n.kind === 'derived').map((n) => n.name) // => ['theta']
 * ```
 */
export const eightSchoolsNonCentred: BenchmarkModel = defineBenchmark({
  name: 'eight-schools-non-centred',
  title: 'Eight schools, non-centred',
  description:
    "Rubin's eight schools with theta_j = mu + tau z_j: the same posterior as the centred hierarchy with the funnel removed.",
  tags: ['hierarchical', 'non-centred', 'reparametrisation', 'quadrature', 'eight-schools'],
  references: [
    'Rubin (1981), Estimation in parallel randomized experiments, Journal of Educational Statistics 6',
    'Gelman et al., Bayesian Data Analysis, 3rd ed., section 5.5',
    'Papaspiliopoulos, Roberts and Skoeld (2007), A general framework for the parametrization of hierarchical models',
  ],
  definition: defineModel('eight-schools-non-centred', (m) => {
    const school = m.range('school')
    const y = m.data('y', { dims: [school], role: 'response', doc: 'the estimated coaching effects' })
    const sigma = m.data('sigma', {
      dims: [school],
      support: 'positive',
      role: 'covariate',
      doc: 'their standard errors (known)',
    })
    const mu = m.latent('mu', d.normal(0, 5), { doc: 'the mean effect' })
    const tau = m.latent('tau', d.halfCauchy(5), { doc: 'the spread of the effects' })
    const z = m.latent('z', d.normal(0, 1), { dims: [school], doc: 'the standardised effect of each school' })
    const theta = m.derived('theta', mu.add(tau.mul(z)), { doc: 'the effect of each school' })
    m.observe('y', d.normal(theta, sigma), y)
  }),
  canonical: CLASSIC,
  generate: generateSchools,
  truth: schoolsTruth,
  plans: (scale) => [
    nutsPlan(scale, {
      tolerances: { meanError: 0.2, ks: 0.12, sdRatio: [0.8, 1.2] },
      note: 'no funnel: no divergences, and agreement with the truth within Monte Carlo error (tau is heavy tailed, so a looser tolerance)',
      config: { targetAcceptance: 0.9 },
    }),
    temperedPlan(scale, {
      expect: 'report',
      particles: 1500,
      note: 'particles with MCMC moves: close, with more noise than NUTS in the tail of tau',
    }),
    laplacePlan(scale, { note: 'a normal at the mode on the log tau scale: too narrow for the heavy-tailed tau' }),
    sviPlan(scale, { note: 'a mean-field normal on log tau: underestimates the spread of tau and of the effects' }),
  ],
})
