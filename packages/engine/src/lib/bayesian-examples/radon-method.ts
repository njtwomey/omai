/**
 * Hierarchical radon as one inference method that a worker or a page runs by name and a few knobs: houses in counties of
 * unequal size from a generator with a known truth (the number of counties, the smallest county, the spread of the county
 * intercepts and of the floor effects), the model (pooled, unpooled, partial pooling centred or non-centred, or varying
 * slopes with a correlated prior) and which inference method (every one that runs on it, with its own knobs) fits it.
 * Beside the tape of the chosen method the contract has the paths that read the posterior at any step: the county
 * intercepts against the raw ones by county size, which is the shrinkage (`radon/estimates`), the pair plot of the
 * hyperparameters with the divergences marked (`radon/pairs`), the chain diagnostics (`radon/diagnostics`), the rows of a
 * summary (`radon/summary`) and the check against a long NUTS run of the same data (`radon/check`).
 *
 * @index estimator Hierarchical radon (pooled, unpooled, partial pooling, varying slopes) over counties of unequal size.
 */

import {
  generateRadonCounties,
  type RadonParameterisation,
  type RadonPooling,
  radonCountiesDefinition,
} from '../probprog/models/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
import { key } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { type Config, derived } from '../trace/index.ts'
import { defineExample, type Example, type ExampleProblem, exampleNumber, exampleString } from './example-method.ts'
import {
  chainsRecord,
  checkRows,
  flattenCheck,
  groupEstimates,
  type PairCoordinate,
  type PosteriorRow,
  type PosteriorRows,
  pairsRecord,
  posteriorRows,
  withDerivedSites,
} from './hierarchical-views.ts'

/** The ways of pooling the counties on offer. */
export const radonPoolings: readonly RadonPooling[] = ['partial', 'pooled', 'unpooled', 'slopes']

/** The problem a configuration stands for. */
export interface RadonProblem extends ExampleProblem {
  readonly pooling: RadonPooling
  readonly parameterisation: RadonParameterisation
  /** The number of houses in each county. */
  readonly sizes: readonly number[]
  /** The unpooled estimate of each county's intercept (a fixed-effects regression of log radon on the floor). */
  readonly raw: readonly number[]
  /** The county with the fewest houses. */
  readonly smallest: number
  /** What generated the data. */
  readonly truth: {
    readonly mu: number
    readonly beta: number
    readonly sigma: number
    readonly tau: number
    readonly slopeSd: number
    readonly rho: number
    readonly alpha: readonly number[]
  }
}

/** The knobs of the data: how many counties, how small the smallest is, and how much they and their floor effects differ. */
export const radonDataKnobs: DictSpace = dict({
  counties: int(4, 80, { default: 12, log: true, label: 'counties' }),
  minSize: int(1, 20, {
    default: 2,
    label: 'smallest county',
    description: 'the fewest houses in a county; the others have more, in a long tail',
  }),
  tau: real(0, 1.5, {
    default: 0.4,
    step: 0.05,
    label: 'true county sd',
    description: 'the sd of the county intercepts',
  }),
  slopeSd: real(0, 1, {
    default: 0.3,
    step: 0.05,
    label: 'true floor-effect sd',
    description: 'the sd of the county floor effects (0: the same in every county)',
  }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: how the counties are pooled and how the varying intercepts are written. */
export const radonModelKnobs: DictSpace = dict({
  pooling: choice([...radonPoolings], {
    default: 'partial',
    label: 'model',
    descriptions: {
      partial: 'varying intercepts drawn from a common distribution',
      pooled: 'one intercept for every county',
      unpooled: 'an intercept of its own for each county, nothing shared',
      slopes: 'varying intercepts and varying floor effects with a correlated prior (non-centred)',
    },
  }),
  parameterisation: choice(['non-centred', 'centred'], {
    default: 'non-centred',
    label: 'varying intercepts',
    descriptions: {
      'non-centred': 'alpha_j = mu + tau z_j: no funnel',
      centred: 'alpha_j ~ N(mu, tau^2): the funnel (partial pooling only)',
    },
  }),
})

/** The reference model of a form: the non-centred partial pooling for the two ways of writing it, itself otherwise. */
export function radonReferenceKnobs(config: Config): Config {
  return exampleString(config, 'pooling') === 'partial' ? { parameterisation: 'non-centred' } : {}
}

/**
 * The problem a configuration stands for: the data of unequal counties from the seed, the model, and the raw estimate of each
 * county's intercept (what no pooling gives).
 *
 * @param config - The problem's knobs (`counties`, `minSize`, `tau`, `slopeSd`, `dataSeed`, `pooling`, `parameterisation`).
 * @returns The problem.
 * @example
 * ```ts
 * import { radonProblem } from 'omai-engine/bayesian-examples'
 * const p = radonProblem({ counties: 10, minSize: 3, tau: 0.4, slopeSd: 0.3, dataSeed: 2, pooling: 'partial', parameterisation: 'non-centred' })
 * [p.sizes.length, Math.min(...p.sizes) >= 3] // => [10, true]
 * ```
 */
export function radonProblem(config: Config): RadonProblem {
  const pooling = (radonPoolings as readonly string[]).includes(exampleString(config, 'pooling'))
    ? (exampleString(config, 'pooling') as RadonPooling)
    : 'partial'
  const parameterisation = exampleString(config, 'parameterisation') === 'centred' ? 'centred' : 'non-centred'
  const g = generateRadonCounties(key(exampleNumber(config, 'dataSeed')), {
    counties: Math.round(exampleNumber(config, 'counties') || 20),
    minSize: Math.round(exampleNumber(config, 'minSize') || 1),
    tau: exampleNumber(config, 'tau'),
    slopeSd: exampleNumber(config, 'slopeSd'),
  })
  const county = g.data['county'] as number[]
  const floor = g.data['floor'] as number[]
  const y = g.data['y'] as number[]
  const sizes = g.generating['sizes'] as number[]
  // The fixed-effects estimate: the floor effect from the variation within counties, each county's intercept its own mean
  // after taking that out.
  const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)
  const byCounty = sizes.map((_, j) => county.flatMap((c, i) => (c === j ? [i] : [])))
  let num = 0
  let den = 0
  for (const rows of byCounty) {
    const fbar = mean(rows.map((i) => floor[i] as number))
    const ybar = mean(rows.map((i) => y[i] as number))
    for (const i of rows) {
      num += ((floor[i] as number) - fbar) * ((y[i] as number) - ybar)
      den += ((floor[i] as number) - fbar) ** 2
    }
  }
  const betaFE = den > 1e-9 ? num / den : 0
  const raw = byCounty.map((rows) => mean(rows.map((i) => (y[i] as number) - betaFE * (floor[i] as number))))
  const gen = g.generating
  return {
    definition: radonCountiesDefinition(pooling, parameterisation),
    data: g.data,
    pooling,
    parameterisation,
    sizes,
    raw,
    smallest: sizes.indexOf(Math.min(...sizes)),
    truth: {
      mu: gen['mu'] as number,
      beta: gen['beta'] as number,
      sigma: gen['sigma'] as number,
      tau: gen['tau'] as number,
      slopeSd: gen['slopeSd'] as number,
      rho: gen['rho'] as number,
      alpha: gen['alpha'] as number[],
    },
  }
}

const SITES = ['mu', 'mu_b', 'tau', 'tau_b', 'rho', 'sigma', 'beta', 'alpha'] as const
const COUNTY_SITES = [...SITES, 'slope']

/** Radon declared as an example: its knobs, the problem they make and the derived paths that read the posterior (`radon/summary`, `radon/estimates`, `radon/pairs`, `radon/diagnostics`, `radon/check`). */
const declared: Example<RadonProblem> = defineExample<RadonProblem>({
  name: 'radon',
  description: 'Hierarchical radon over counties of unequal size, one inference method chosen from a table',
  modelName: 'radon',
  dataKnobs: radonDataKnobs,
  modelKnobs: radonModelKnobs,
  problemKnobs: ['counties', 'minSize', 'tau', 'slopeSd', 'dataSeed', 'pooling', 'parameterisation'],
  problem: radonProblem,
  derived: ({ problem, kind, posterior: fitted }) => {
    const counties = problem.sizes.length
    // The county intercepts and slopes of the non-centred forms are derived from the
    // latents: a variational or Laplace fit has only the latents, so they are computed from draws of them.
    const posterior = (state: unknown): Posterior => {
      const post = fitted(state)
      const at = (v: number[] | undefined, j: number): number => v?.[j] ?? 0
      if (problem.pooling === 'partial' && problem.parameterisation === 'non-centred')
        return withDerivedSites(post, ['alpha'], (d) => ({
          alpha: Array.from({ length: counties }, (_, j) => at(d['mu'], 0) + at(d['tau'], 0) * at(d['z'], j)),
        }))
      if (problem.pooling === 'slopes')
        return withDerivedSites(post, ['alpha', 'slope'], (d) => {
          const rho = at(d['rho'], 0)
          const rest = Math.sqrt(Math.max(1e-12, 1 - rho * rho))
          return {
            alpha: Array.from({ length: counties }, (_, j) => at(d['mu'], 0) + at(d['tau'], 0) * at(d['z_a'], j)),
            slope: Array.from(
              { length: counties },
              (_, j) => at(d['mu_b'], 0) + at(d['tau_b'], 0) * (rho * at(d['z_a'], j) + rest * at(d['z_b'], j)),
            ),
          }
        })
      return post
    }
    const s = problem.smallest
    const t = problem.truth
    const rowsOf = (state: unknown): PosteriorRows | undefined => {
      try {
        const rows = posteriorRows(posterior(state), COUNTY_SITES)
        const one = rows?.rows.find((r) => r.name === 'alpha')
        // A pooled model has one intercept: every county is at it.
        return rows === undefined || one === undefined
          ? rows
          : {
              ...rows,
              rows: [...rows.rows, ...Array.from({ length: counties }, (_, j) => ({ ...one, name: `alpha[${j}]` }))],
            }
      } catch {
        return undefined
      }
    }
    const pairs: readonly PairCoordinate[] = [
      { label: 'mu', from: 'mu' },
      { label: 'alpha', from: 'alpha' },
      { label: 'log tau', from: 'tau', log: true },
      { label: 'mu_b', from: 'mu_b' },
      { label: 'log tau_b', from: 'tau_b', log: true },
      { label: 'rho', from: 'rho' },
      ...(problem.pooling === 'slopes' ? [] : [{ label: 'sigma', from: 'sigma' }]),
      { label: 'beta', from: 'beta' },
      { label: `alpha[${s}]`, from: `alpha[${s}]` },
    ]
    const truthOf: Record<string, number> = {
      mu: t.mu,
      alpha: t.mu,
      ...(t.tau > 0 ? { 'log tau': Math.log(t.tau) } : {}),
      mu_b: t.beta,
      ...(t.slopeSd > 0 ? { 'log tau_b': Math.log(t.slopeSd) } : {}),
      rho: t.rho,
      sigma: t.sigma,
      beta: t.beta,
      [`alpha[${s}]`]: t.alpha[s] ?? 0,
    }
    return {
      'radon/summary': derived({
        cost: 'dear',
        description:
          'the mean, spread, 90% interval, R-hat and ESS of the hyperparameters and every county intercept (a posterior rows record), with the divergences and the draws used',
        compute: (state) => rowsOf(state) ?? null,
      }),
      'radon/estimates': derived({
        cost: 'dear',
        description:
          'the intercept of each county against its raw estimate by the number of houses (a groupEstimates record): the posterior mean and interval, and how far each moved towards the mean',
        compute: (state) => {
          const r = rowsOf(state)
          if (r === undefined) return null
          const alpha = Array.from({ length: counties }, (_, j) => r.rows.find((x) => x.name === `alpha[${j}]`))
          if (!alpha.every((x) => x !== undefined)) return null
          const population = r.rows.find((x) => x.name === 'mu') ?? r.rows.find((x) => x.name === 'alpha')
          return groupEstimates({
            xLabel: 'houses in the county',
            yLabel: 'county intercept (log radon in a basement)',
            labels: Array.from({ length: counties }, (_, j) => `county ${j + 1}`),
            x: problem.sizes,
            raw: problem.raw,
            rows: alpha as PosteriorRow[],
            population: problem.pooling === 'unpooled' ? null : (population?.mean ?? null),
            dataIsLargeX: true,
            truth: t.alpha,
          })
        },
      }),
      'radon/pairs': derived({
        cost: 'dear',
        description:
          'the pair plot of the hyperparameters and the intercept of the smallest county, the divergent transitions marked (a posteriorPairs record)',
        compute: (state) => pairsRecord(posterior(state), pairs, kind, truthOf) ?? null,
      }),
      'radon/diagnostics': derived({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler for the hyperparameters (a chainDiagnostics record)',
        compute: (state) =>
          chainsRecord(
            posterior(state),
            [...SITES, `alpha[${s}]`].filter((n) => n !== 'alpha' || problem.pooling === 'pooled'),
          ) ?? null,
      }),
      'radon/check': derived({
        cost: 'dear',
        description:
          "the method's posterior against a reference run of the same data (args: `{ summary }`, the reference's `radon/summary`): the largest difference of means in the reference's sds and in Monte Carlo errors, the ratio of the spreads, and the hyperparameters on their own",
        compute: (state, args) => {
          const ref = (args as { summary?: PosteriorRows } | null | undefined)?.summary
          const mine = rowsOf(state)
          if (ref === undefined || mine === undefined) return null
          const check = checkRows(mine, ref, ['mu', 'tau', 'sigma', 'beta', 'mu_b', 'tau_b', 'rho'])
          return check === undefined ? null : flattenCheck(check)
        },
      }),
    }
  },
})

/**
 * Hierarchical radon as a declared example (knobs, problem and derived paths: `radon/summary`, `radon/estimates`,
 * `radon/pairs`, `radon/diagnostics`, `radon/check`). Its `reference` keeps the data and runs the non-centred form of the
 * partial-pooling model, so a centred run is checked against a run with no funnel.
 *
 * @concept inference-engines
 * @example
 * ```ts
 * import { radonExample } from 'omai-engine/bayesian-examples'
 * radonExample.reference({ pooling: 'partial', parameterisation: 'centred' }, { method: 'nuts' })['parameterisation'] // => 'non-centred'
 * ```
 */
export const radonExample: Example<RadonProblem> = {
  ...declared,
  reference: (config, inference, budgetMs) => ({
    ...declared.reference(config, inference, budgetMs),
    ...radonReferenceKnobs(config),
  }),
}
