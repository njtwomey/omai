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
import { arange, exp, item, mod, mul, take, tensor, toFlat } from '../../tensor/index.ts'
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
  return generateSchoolGroups(key, {
    groups: size,
    mu: 4,
    tau: 3,
    standardErrors: toFlat(take(tensor(SCHOOL_SIGMA), mod(arange(size), SCHOOL_SIGMA.length))),
  })
}

/** What {@link generateSchoolGroups} generates from. */
export interface SchoolGroupsOptions {
  /** The number of groups (schools). */
  readonly groups: number
  /** The mean of the group effects. */
  readonly mu: number
  /** The standard deviation of the group effects around it (0: every group has the same effect). */
  readonly tau: number
  /** The standard error of each group's estimate, given (the length is the number of groups). */
  readonly standardErrors?: readonly number[]
  /** Otherwise the typical standard error (default 10), and how much it varies between groups (the sd of its log, default 0.4). */
  readonly se?: number
  readonly seSpread?: number
}

/**
 * Data of the eight schools model from known parameters: `groups` effects `theta_j ~ N(mu, tau²)`, each estimated with its
 * own known standard error `sigma_j` (`y_j ~ N(theta_j, sigma_j²)`). A group with a large standard error is one with
 * little data behind it, which is how this model expresses groups of unequal sizes: the standard errors are either given
 * or drawn log-normally around a typical one. The generating parameters (`mu`, `tau`, the true `theta`) come back as
 * the truth.
 *
 * @param key - The random key.
 * @param options - The number of groups, `mu`, `tau` and the standard errors.
 * @returns The data (`y`, `sigma`) and what generated them.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { generateSchoolGroups } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const d = generateSchoolGroups(key(1), { groups: 12, mu: 2, tau: 0, se: 5, seSpread: 0 })
 * [(d.data['y'] as number[]).length, d.generating['tau'], (d.data['sigma'] as number[])[3]] // => [12, 0, 5]
 * ```
 */
export function generateSchoolGroups(key: Key, options: SchoolGroupsOptions): BenchmarkData {
  const size = options.groups
  const { mu, tau } = options
  const z = toFlat(drawNormal(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  const w = drawNormal(foldIn(key, 2), [size])
  const sigma =
    options.standardErrors === undefined
      ? toFlat(mul(exp(mul(w, options.seSpread ?? 0.4)), options.se ?? 10))
      : [...options.standardErrors]
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

/** The ways of writing the eight schools model that {@link eightSchoolsDefinition} offers. */
export type SchoolsParameterisation = 'centred' | 'non-centred' | 'complete-pooling' | 'no-pooling'

/** The scales of the priors of {@link eightSchoolsDefinition}. */
export interface SchoolsPriors {
  /** The sd of the normal prior of the mean effect `mu` (default 5). */
  readonly muScale?: number
  /** The scale of the half-Cauchy prior of the spread `tau` (default 5). */
  readonly tauScale?: number
  /** The sd of the normal prior of each effect when nothing pools them (`no-pooling`, default 100: nearly flat, so nothing is pooled). */
  readonly effectScale?: number
}

/**
 * The eight schools model as a spec in one of four forms, for any number of groups (the size is the length of the data):
 * `centred` (`theta_j ~ N(mu, tau²)`), `non-centred` (`theta_j = mu + tau z_j`, the same posterior without the funnel),
 * `complete-pooling` (every group has the one effect `mu`: no `tau`, no `theta`) and `no-pooling` (every group has its own
 * effect `theta_j ~ N(0, effectScale²)` and nothing is shared: no `mu`, no `tau`). The two partial-pooling forms are the
 * models of `eightSchoolsCentred` and `eightSchoolsNonCentred` at the default priors.
 *
 * @param parameterisation - The form.
 * @param priors - The scales of the priors.
 * @returns The model definition, named `eight-schools-<form>`; data `y` and `sigma` over the `school` range.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { eightSchoolsDefinition } from 'omai-engine/probprog/models'
 * const latents = (p: 'centred' | 'non-centred' | 'complete-pooling' | 'no-pooling') =>
 *   eightSchoolsDefinition(p).spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name)
 * [latents('centred'), latents('non-centred'), latents('complete-pooling'), latents('no-pooling')] // => [['mu', 'tau', 'theta'], ['mu', 'tau', 'z'], ['mu'], ['theta']]
 * ```
 */
export function eightSchoolsDefinition(
  parameterisation: SchoolsParameterisation,
  priors: SchoolsPriors = {},
): ModelDefinition {
  // region schools
  const { muScale = 5, tauScale = 5, effectScale = 100 } = priors
  const definition = defineModel(`eight-schools-${parameterisation}`, (m) => {
    const school = m.range('school')
    const y = m.data('y', { dims: [school], role: 'response', doc: 'the estimated coaching effects' })
    const sigma = m.data('sigma', {
      dims: [school],
      support: 'positive',
      role: 'covariate',
      doc: 'their standard errors (known)',
    })
    // endregion
    if (parameterisation === 'no-pooling') {
      // region schools:no-pooling
      const theta = m.latent('theta', d.normal(0, effectScale), {
        dims: [school],
        doc: 'the effect of each school',
      })
      m.observe('y', d.normal(theta, sigma), y)
      // endregion
      return
    }
    // region schools:mu
    const mu = m.latent('mu', d.normal(0, muScale), { doc: 'the mean effect' })
    // endregion
    if (parameterisation === 'complete-pooling') {
      // region schools:complete-pooling
      m.observe('y', d.normal(mu, sigma), y)
      // endregion
      return
    }
    // region schools:tau
    const tau = m.latent('tau', d.halfCauchy(tauScale), { doc: 'the spread of the effects' })
    // endregion
    if (parameterisation === 'centred') {
      // region schools:centred
      // The scale is clipped to [1e-6, 1e6] where it enters the distribution. This is a modelling floor,
      // not a guard against an invalid parameter (the sampler treats that as a divergence): the centred
      // density is unbounded as tau goes to 0, and without the floor the sampler is drawn into that
      // funnel (18 divergences against 8).
      const theta = m.latent('theta', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), {
        dims: [school],
        doc: 'the effect of each school',
      })
      m.observe('y', d.normal(theta, sigma), y)
      // endregion
      return
    }
    // region schools:non-centred
    const z = m.latent('z', d.normal(0, 1), {
      dims: [school],
      doc: 'the standardised effect of each school',
    })
    const theta = m.derived('theta', mu.add(tau.mul(z)), { doc: 'the effect of each school' })
    m.observe('y', d.normal(theta, sigma), y)
    // endregion
    // region schools:end
  })
  // endregion
  return definition
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
  definition: eightSchoolsDefinition('centred'),
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
  definition: eightSchoolsDefinition('non-centred'),
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
