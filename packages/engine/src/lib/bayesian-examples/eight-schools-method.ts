/**
 * Eight schools, partial pooling, as one inference method that a worker or a page runs by name and a few knobs: the data
 * (Rubin's classic eight, or a generator with its own number of groups, spread between them and standard errors), the model
 * (centred, non-centred, complete pooling or no pooling, with the scale of the prior of the spread) and which inference
 * method (every one that runs on it, with its own knobs) fits it. Beside the tape of the chosen method the contract has the
 * paths that read the posterior at any step: the group estimates against the raw ones (`schools/estimates`, the shrinkage),
 * the pair plot of `(mu, log tau, theta_0)` with the divergent transitions marked (`schools/pairs`), the chain diagnostics
 * (`schools/diagnostics`), the rows of a summary (`schools/summary`) and the check against a reference run
 * (`schools/check`), which a page makes a long non-centred NUTS run of the same data (`schoolsReferenceModel`).
 *
 * @index estimator Eight schools hierarchical model (centred, non-centred, pooled, unpooled): data, model and inference as knobs.
 */

import { eightSchoolsCentred, eightSchoolsDefinition, generateSchoolGroups } from '../probprog/models/index.ts'
import type { SchoolsParameterisation } from '../probprog/models/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
import { key } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real, union } from '../space/index.ts'
import { type Config, derived } from '../trace/index.ts'
import { defineExample, type Example, type ExampleProblem, exampleNumber, exampleString } from './example-method.ts'
import {
  chainsRecord,
  checkRows,
  flattenCheck,
  groupEstimates,
  type PairCoordinate,
  pairsRecord,
  type PosteriorRow,
  type PosteriorRows,
  posteriorRows,
  withDerivedSites,
} from './hierarchical-views.ts'

/** The forms of the model on offer. */
export const schoolsModels: readonly SchoolsParameterisation[] = [
  'centred',
  'non-centred',
  'complete-pooling',
  'no-pooling',
]

/** The problem a configuration stands for: the model and the data, with what generated them when it is known. */
export interface SchoolsProblem extends ExampleProblem {
  readonly parameterisation: SchoolsParameterisation
  readonly y: readonly number[]
  readonly sigma: readonly number[]
  /** The generating `mu`, `tau` and true `theta` of generated data; `undefined` for Rubin's. */
  readonly truth: { readonly mu: number; readonly tau: number; readonly theta: readonly number[] } | undefined
}

/** The knobs of the data: Rubin's classic eight schools, or generated ones with a number of groups, spread and standard errors. */
export const schoolsDataKnobs: DictSpace = dict({
  dataset: union(
    {
      classic: dict({}, { description: "Rubin's eight schools (Gelman et al., section 5.5)" }),
      generated: dict(
        {
          groups: int(3, 40, { default: 8, label: 'schools', description: 'the number of groups' }),
          tau: real(0, 20, {
            default: 3,
            step: 0.5,
            label: 'true spread tau',
            description: 'the sd of the true effects around their mean 4 (0: all the same)',
          }),
          se: real(1, 30, {
            default: 10,
            step: 1,
            label: 'standard error',
            description: 'the typical standard error of a group estimate',
          }),
          seSpread: real(0, 1.2, {
            default: 0.4,
            step: 0.1,
            label: 'standard errors differ',
            description: 'the sd of the log standard error between groups: groups with a larger one have less data behind them',
          }),
        },
        { description: 'effects drawn from the model with known mu = 4 and tau, and standard errors that differ' },
      ),
    },
    { tag: 'source', default: 'classic', label: 'data' },
  ),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: how it is written and the prior of the spread. */
export const schoolsModelKnobs: DictSpace = dict({
  model: choice([...schoolsModels], {
    default: 'centred',
    label: 'model',
    descriptions: {
      centred: 'theta_j ~ N(mu, tau^2): partial pooling, with the funnel',
      'non-centred': 'theta_j = mu + tau z_j: the same partial pooling without the funnel',
      'complete-pooling': 'every school has the one effect mu',
      'no-pooling': 'every school has its own effect and nothing is shared',
    },
  }),
  tauScale: real(0.5, 25, {
    default: 5,
    log: true,
    label: 'prior scale of tau',
    description: 'the scale of the half-Cauchy prior of the spread of the effects (ignored when nothing pools or all pool)',
  }),
})

/** The reference model of a model: the non-centred one for the two partial-pooling forms (the same posterior, no funnel), itself otherwise. */
export function schoolsReferenceModel(model: string): string {
  return model === 'centred' || model === 'non-centred' ? 'non-centred' : model
}

const dataSource = (config: Config): { source: string; knobs: Config } => {
  const given = config['dataset']
  const { source, ...knobs } = (
    typeof given === 'object' && given !== null && !Array.isArray(given) ? given : { source: 'classic' }
  ) as { source: string } & Config
  return { source, knobs }
}

/**
 * The problem a configuration stands for: the data, classic or generated from the seed, and the model with the scale of the
 * prior of `tau`.
 *
 * @param config - The problem's knobs (`dataset`, `dataSeed`, `model`, `tauScale`).
 * @returns The problem.
 * @example
 * ```ts
 * import { schoolsProblem } from 'omai-engine/bayesian-examples'
 * const p = schoolsProblem({ dataset: { source: 'generated', groups: 12, tau: 2, se: 8, seSpread: 0.3 }, dataSeed: 3, model: 'non-centred', tauScale: 5 })
 * [p.y.length, p.truth?.tau] // => [12, 2]
 * ```
 */
export function schoolsProblem(config: Config): SchoolsProblem {
  const { source, knobs } = dataSource(config)
  const model = exampleString(config, 'model')
  const parameterisation = (schoolsModels as readonly string[]).includes(model)
    ? (model as SchoolsParameterisation)
    : 'centred'
  const generated =
    source === 'generated'
      ? generateSchoolGroups(key(exampleNumber(config, 'dataSeed')), {
          groups: Math.round(exampleNumber(knobs, 'groups') || 8),
          mu: 4,
          tau: exampleNumber(knobs, 'tau'),
          se: exampleNumber(knobs, 'se') || 10,
          seSpread: exampleNumber(knobs, 'seSpread'),
        })
      : eightSchoolsCentred.data
  const data = generated.data as { y: number[]; sigma: number[] }
  const g = generated.generating
  return {
    definition: eightSchoolsDefinition(parameterisation, { tauScale: exampleNumber(config, 'tauScale') || 5 }),
    data: generated.data,
    parameterisation,
    y: data.y,
    sigma: data.sigma,
    truth:
      source === 'generated'
        ? { mu: g['mu'] as number, tau: g['tau'] as number, theta: g['theta'] as number[] }
        : undefined,
  }
}

const LATENT_SITES = ['mu', 'tau', 'theta'] as const

/** The coordinates of the pair plot: the mean, the log spread and the first school's effect (and its standardised form). */
const PAIRS: readonly PairCoordinate[] = [
  { label: 'mu', from: 'mu' },
  { label: 'log tau', from: 'tau', log: true },
  { label: 'theta[0]', from: 'theta[0]' },
  { label: 'z[0]', from: 'z[0]' },
]

/** Without `mu` and `tau` (no pooling) the first schools' effects, which are independent there. */
const UNPOOLED_PAIRS: readonly PairCoordinate[] = [0, 1, 2].map((j) => ({ label: `theta[${j}]`, from: `theta[${j}]` }))

/** Eight schools declared as an example: its knobs, the problem they make and the derived paths that read the posterior (`schools/summary`, `schools/estimates`, `schools/pairs`, `schools/diagnostics`, `schools/check`). */
const declared: Example<SchoolsProblem> = defineExample<SchoolsProblem>({
  name: 'schools',
  description: 'Rubin\'s eight schools (or generated groups) by a hierarchical model with one inference method chosen from a table',
  modelName: 'schools',
  dataKnobs: schoolsDataKnobs,
  modelKnobs: schoolsModelKnobs,
  problemKnobs: ['dataset', 'dataSeed', 'model', 'tauScale'],
  problem: schoolsProblem,
  derived: ({ problem, kind, posterior: fitted }) => {
    const groups = problem.y.length
    // The effects theta of the non-centred and complete-pooling forms are derived from the latents: a variational or Laplace fit
    // has only the latents, so the effects are computed from draws of them.
    const posterior = (state: unknown): Posterior => {
      const post = fitted(state)
      if (problem.parameterisation === 'non-centred')
        return withDerivedSites(post, ['theta'], (d) => ({
          theta: Array.from({ length: groups }, (_, j) => (d['mu']?.[0] ?? 0) + (d['tau']?.[0] ?? 0) * (d['z']?.[j] ?? 0)),
        }))
      if (problem.parameterisation === 'complete-pooling')
        return withDerivedSites(post, ['theta'], (d) => ({ theta: Array.from({ length: groups }, () => d['mu']?.[0] ?? 0) }))
      return post
    }
    const rowsOf = (state: unknown): PosteriorRows | undefined => {
      try {
        return posteriorRows(posterior(state), LATENT_SITES)
      } catch {
        return undefined
      }
    }
    const truth = problem.truth
    // Complete pooling has one parameter (every theta is mu), so no pair plot.
    const pairCoordinates =
      problem.parameterisation === 'no-pooling' ? UNPOOLED_PAIRS : problem.parameterisation === 'complete-pooling' ? [] : PAIRS
    return {
      'schools/summary': derived({
        cost: 'dear',
        description:
          'the mean, spread, 90% interval, R-hat and ESS of mu, tau and every theta (a posterior rows record), with the divergences and the draws used',
        compute: (state) => rowsOf(state) ?? null,
      }),
      'schools/estimates': derived({
        cost: 'dear',
        description:
          "the estimate of each school's effect against its raw estimate (a groupEstimates record): the posterior mean and interval, the raw y, the standard error, and how far each moved towards the mean",
        compute: (state) => {
          const r = rowsOf(state)
          if (r === undefined) return null
          const theta = Array.from({ length: groups }, (_, j) => r.rows.find((x) => x.name === `theta[${j}]`))
          if (!theta.every((x) => x !== undefined)) return null
          return groupEstimates({
            xLabel: 'standard error of the estimate',
            yLabel: 'effect of coaching',
            labels: Array.from({ length: groups }, (_, j) => `school ${j + 1}`),
            x: problem.sigma,
            raw: problem.y,
            rows: theta as PosteriorRow[],
            population: r.rows.find((x) => x.name === 'mu')?.mean ?? null,
            dataIsLargeX: false,
            truth: truth?.theta ?? null,
          })
        },
      }),
      'schools/pairs': derived({
        cost: 'dear',
        description:
          'the pair plot of mu, log tau and the first school\'s effect (and its standardised form for the non-centred model), the divergent transitions marked (a posteriorPairs record)',
        compute: (state) => {
          const truthOf =
            truth === undefined
              ? {}
              : { mu: truth.mu, ...(truth.tau > 0 ? { 'log tau': Math.log(truth.tau) } : {}), 'theta[0]': truth.theta[0] ?? 0 }
          return pairsRecord(posterior(state), pairCoordinates, kind, truthOf) ?? null
        },
      }),
      'schools/diagnostics': derived({
        cost: 'dear',
        description: 'trace, rank, R-hat and ESS of the chains of a sampler for mu, tau and the first school (a chainDiagnostics record)',
        compute: (state) => chainsRecord(posterior(state), ['mu', 'tau', 'theta[0]', 'z[0]']) ?? null,
      }),
      'schools/check': derived({
        cost: 'dear',
        description:
          "the method's posterior against a reference run of the same data (args: `{ summary }`, the reference's `schools/summary`): the largest difference of means in the reference's sds and in Monte Carlo errors, the ratio of the spreads, and mu and tau on their own",
        compute: (state, args) => {
          const ref = (args as { summary?: PosteriorRows } | null | undefined)?.summary
          const mine = rowsOf(state)
          if (ref === undefined || mine === undefined) return null
          const check = checkRows(mine, ref, ['mu', 'tau'])
          return check === undefined ? null : flattenCheck(check)
        },
      }),
    }
  },
})

/**
 * Eight schools as a declared example (knobs, problem and derived paths: `schools/summary`, `schools/estimates`, `schools/pairs`, `schools/diagnostics`, `schools/check`). Its `reference` keeps the data and runs the
 * non-centred form of a partial-pooling model, so a centred run is checked against a run with no funnel.
 *
 * @concept inference-engines
 * @example
 * ```ts
 * import { schoolsExample } from 'omai-engine/bayesian-examples'
 * schoolsExample.reference({ model: 'centred' }, { method: 'nuts' })['model'] // => 'non-centred'
 * ```
 */
export const schoolsExample: Example<SchoolsProblem> = {
  ...declared,
  reference: (config, inference, budgetMs) => ({
    ...declared.reference(config, inference, budgetMs),
    model: schoolsReferenceModel(exampleString(config, 'model')),
  }),
}
