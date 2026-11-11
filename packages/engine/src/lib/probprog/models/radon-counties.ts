/**
 * The radon regression with counties of unequal size: log radon in houses nested in counties, the houses given as a flat
 * list with the county of each (an index), so a county may have one house or sixty. The model is written in four forms, the
 * ones Gelman and Hill compare: `pooled` (one intercept for every county), `unpooled` (an intercept of its own for each, nothing
 * shared), `partial` (varying intercepts drawn from a common distribution, centred or non-centred) and `slopes` (varying
 * intercepts and varying floor effects with a correlated prior, non-centred). A generator makes the data from a known truth
 * with a smallest county size, so the counties with little data are where partial pooling is seen to borrow most. The
 * balanced benchmark `radon` (`hierarchical.ts`) is the exactly-solvable cousin of this one.
 */

import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition, ops } from '../model/index.ts'
import type { BenchmarkData } from './benchmark.ts'

/** The four ways of pooling the counties that {@link radonCountiesDefinition} offers. */
export type RadonPooling = 'pooled' | 'unpooled' | 'partial' | 'slopes'

/** The parameterisation of the varying intercepts of the `partial` model. */
export type RadonParameterisation = 'centred' | 'non-centred'

/** The scales of the priors of {@link radonCountiesDefinition}. */
export interface RadonPriors {
  /** The sd of the normal prior of the mean intercept and of every intercept when nothing pools (default 5). */
  readonly intercept?: number
  /** The sd of the normal prior of the floor effect (default 2). */
  readonly slope?: number
  /** The scale of the half-normal priors of the county sds and of the noise sd (default 1). */
  readonly scale?: number
}

/**
 * The radon regression of houses in counties as a spec, in one of four poolings. The data are flat over the houses: `size` (the houses of each county), `county`
 * (the index of each house's county), `floor` (1 for a house measured on the first floor, 0 for a basement) and `y` (log
 * radon). The latents by form: `pooled` has `alpha`, `beta`, `sigma`; `unpooled` has `alpha` per county, `beta`, `sigma`;
 * `partial` has `mu`, `tau`, `alpha` per county (centred) or `z` per county with `alpha` derived (non-centred), `beta`, `sigma`;
 * `slopes` has `mu`, `mu_b`, `tau`, `tau_b`, a county correlation `rho` of intercept and slope (uniform on (-1, 1), which is
 * the LKJ prior of a 2 x 2 correlation matrix), `z_a` and `z_b` per county, and the derived `alpha` and `slope` per county.
 *
 * @param pooling - How the counties are pooled.
 * @param parameterisation - Centred or non-centred varying intercepts (only the `partial` form has both; `slopes` is non-centred).
 * @param priors - The scales of the priors.
 * @returns The model definition, named `radon-counties-<pooling>`.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { radonCountiesDefinition } from 'omai-engine/probprog/models'
 * const latents = (p: 'pooled' | 'unpooled' | 'partial' | 'slopes') =>
 *   radonCountiesDefinition(p).spec.nodes.filter((n) => n.kind === 'latent' && n.name !== 'sigma').map((n) => n.name)
 * [latents('pooled'), latents('partial'), latents('slopes')] // => [['alpha', 'beta'], ['mu', 'tau', 'beta', 'z'], ['mu', 'tau', 'mu_b', 'tau_b', 'rho', 'z_a', 'z_b']]
 * ```
 */
export function radonCountiesDefinition(
  pooling: RadonPooling,
  parameterisation: RadonParameterisation = 'non-centred',
  priors: RadonPriors = {},
): ModelDefinition {
  const { intercept = 5, slope = 2, scale = 1 } = priors
  return defineModel(`radon-counties-${pooling}`, (m) => {
    const county = m.range('county')
    const house = m.range('house')
    const code = m.data('county', {
      dims: [house],
      support: 'index',
      role: 'index',
      indexes: county,
      doc: 'the county of each house',
    })
    const floor = m.data('floor', {
      dims: [house],
      support: 'binary',
      role: 'covariate',
      doc: '1 for a house measured on the first floor, 0 for a basement',
    })
    // The number of houses of each county: not in the density, it gives the county range its size (index data do not), so the
    // model binds to plain data with no options and crosses a worker boundary as a configuration.
    m.data('size', { dims: [county], role: 'covariate', doc: 'the number of houses in each county' })
    const y = m.data('y', { dims: [house], role: 'response', doc: 'log radon' })
    const sigma = m.latent('sigma', d.halfNormal(scale), { doc: 'the sd of a house within its county' })
    // The scales are clipped to [1e-6, 1e6] where they enter a distribution: a modelling floor, not a guard (the sampler treats
    // an invalid parameter as a divergence); the centred density is unbounded as tau goes to 0.
    const noise = ops.clip(sigma, 1e-6, 1e6)
    if (pooling === 'pooled') {
      const alpha = m.latent('alpha', d.normal(0, intercept), { doc: 'the intercept of every county' })
      const beta = m.latent('beta', d.normal(0, slope), { doc: 'the floor effect' })
      m.observe('y', d.normal(alpha.add(beta.mul(floor)), noise), y)
      return
    }
    if (pooling === 'unpooled') {
      const alpha = m.latent('alpha', d.normal(0, intercept), { dims: [county], doc: 'the intercept of each county' })
      const beta = m.latent('beta', d.normal(0, slope), { doc: 'the floor effect' })
      m.observe('y', d.normal(alpha.at(code).add(beta.mul(floor)), noise), y)
      return
    }
    const mu = m.latent('mu', d.normal(0, intercept), { doc: 'the mean county intercept' })
    const tau = m.latent('tau', d.halfNormal(scale), { doc: 'the sd of the county intercepts' })
    if (pooling === 'partial') {
      const beta = m.latent('beta', d.normal(0, slope), { doc: 'the floor effect' })
      if (parameterisation === 'centred') {
        const alpha = m.latent('alpha', d.normal(mu, ops.clip(tau, 1e-6, 1e6)), {
          dims: [county],
          doc: 'the intercept of each county',
        })
        m.observe('y', d.normal(alpha.at(code).add(beta.mul(floor)), noise), y)
        return
      }
      const z = m.latent('z', d.normal(0, 1), { dims: [county], doc: 'the standardised intercept of each county' })
      const alpha = m.derived('alpha', mu.add(tau.mul(z)), { doc: 'the intercept of each county' })
      m.observe('y', d.normal(alpha.at(code).add(beta.mul(floor)), noise), y)
      return
    }
    const muB = m.latent('mu_b', d.normal(0, slope), { doc: 'the mean floor effect' })
    const tauB = m.latent('tau_b', d.halfNormal(scale), { doc: 'the sd of the county floor effects' })
    const rho = m.latent('rho', d.uniform(-1, 1), { doc: 'the correlation of a county intercept and floor effect' })
    const za = m.latent('z_a', d.normal(0, 1), { dims: [county], doc: 'the standardised intercept of each county' })
    const zb = m.latent('z_b', d.normal(0, 1), {
      dims: [county],
      doc: 'the part of the floor effect independent of it',
    })
    const alpha = m.derived('alpha', mu.add(tau.mul(za)), { doc: 'the intercept of each county' })
    // The Cholesky factor of the 2 x 2 correlation matrix written out: slope_j = mu_b + tau_b (rho z_a + sqrt(1 - rho^2) z_b).
    const rest = ops.sqrt(ops.clip(ops.sub(1, rho.square()), 1e-12, 1))
    const slopes = m.derived('slope', muB.add(tauB.mul(rho.mul(za).add(rest.mul(zb)))), {
      doc: 'the floor effect of each county',
    })
    m.observe('y', d.normal(alpha.at(code).add(slopes.at(code).mul(floor)), noise), y)
  })
}

/** What {@link generateRadonCounties} generates from. */
export interface RadonCountiesOptions {
  /** The number of counties. */
  readonly counties: number
  /** The fewest houses in a county (the others have more, in a long tail). */
  readonly minSize: number
  /** The sd of the county intercepts. */
  readonly tau: number
  /** The sd of the county floor effects (0: every county has the one floor effect). */
  readonly slopeSd?: number
  /** The correlation of a county's intercept and floor effect (default -0.5). */
  readonly rho?: number
  /** The most houses in a county (default 60). */
  readonly maxSize?: number
}

/** The generating values of {@link generateRadonCounties} that do not depend on the options. */
export const RADON_TRUTH: Readonly<{ mu: number; beta: number; sigma: number }> = { mu: 1.5, beta: -0.6, sigma: 0.7 }

/**
 * Radon data from a known truth with counties of unequal size: each county has `minSize` houses plus a log-normal tail
 * (a few counties have dozens), the county intercepts are `mu + tau z_j`, the floor effect is `beta` plus a county part of sd
 * `slopeSd` correlated `rho` with the intercept, and 40% of the houses are measured on the first floor. The truth
 * (`mu`, `beta`, `sigma`, `tau`, `slopeSd`, `rho`, the county `alpha` and `slope`, the sizes) comes back with the data.
 *
 * @param key - The random key.
 * @param options - The number of counties, the smallest size, and the spreads of the county effects.
 * @returns The data (`county`, `floor`, `y` flat over the houses, and `size`, the houses of each county, which sizes the `county` range) and what generated them.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { generateRadonCounties } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const g = generateRadonCounties(key(3), { counties: 10, minSize: 2, tau: 0.4 })
 * const sizes = g.generating['sizes'] as number[]
 * [sizes.length, Math.min(...sizes) >= 2, (g.data['y'] as number[]).length === sizes.reduce((a, b) => a + b, 0)] // => [10, true, true]
 * ```
 */
export function generateRadonCounties(key: Key, options: RadonCountiesOptions): BenchmarkData {
  const J = options.counties
  const { tau, minSize } = options
  const slopeSd = options.slopeSd ?? 0
  const rho = options.rho ?? -0.5
  const maxSize = options.maxSize ?? 60
  const a = toFlat(drawNormal(foldIn(key, 0), [J]))
  const b = toFlat(drawNormal(foldIn(key, 1), [J]))
  const tail = toFlat(drawNormal(foldIn(key, 2), [J]))
  const sizes = Array.from({ length: J }, (_, j) =>
    Math.min(maxSize, Math.max(minSize, minSize + Math.floor(Math.exp(1.2 + 0.9 * (tail[j] ?? 0))))),
  )
  const alpha = Array.from(a, (v) => RADON_TRUTH.mu + tau * v)
  const slope = Array.from(a, (v, j) => RADON_TRUTH.beta + slopeSd * (rho * v + Math.sqrt(1 - rho * rho) * (b[j] ?? 0)))
  const total = sizes.reduce((s, v) => s + v, 0)
  const u = toFlat(drawUniform(foldIn(key, 3), [total]))
  const e = toFlat(drawNormal(foldIn(key, 4), [total]))
  const county: number[] = []
  const floor: number[] = []
  const y: number[] = []
  let row = 0
  sizes.forEach((n, j) => {
    for (let i = 0; i < n; i++, row++) {
      const f = (u[row] ?? 1) < 0.4 ? 1 : 0
      county.push(j)
      floor.push(f)
      y.push((alpha[j] ?? 0) + (slope[j] ?? 0) * f + RADON_TRUTH.sigma * (e[row] ?? 0))
    }
  })
  return {
    data: { county, floor, y, size: sizes },
    generating: { ...RADON_TRUTH, tau, slopeSd, rho, alpha, slope, sizes },
  }
}
