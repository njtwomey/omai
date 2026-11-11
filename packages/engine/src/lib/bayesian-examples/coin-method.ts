/**
 * The biased coin as an example method (Beta-Bernoulli), or its counting twin (Gamma-Poisson): the data are `flips`
 * simulated from a true probability (or rate) with a seed, the model is a conjugate prior with its two numbers, and one
 * inference method of the library, with its own knobs and a budget, infers the posterior. The posterior is known in closed
 * form, so every method is checked against the exact Beta (or Gamma) posterior and its evidence. The derived paths read
 * the run: `coin/updating` (the posterior after the first `seen` flips, the chosen method refitted on that prefix and the
 * exact one beside it, with the credible intervals and the prediction of the next flip), `coin/contraction` (the exact
 * posterior's mean and interval as flips accumulate) and `coin/check` (the method's posterior against the exact one:
 * mean error, spread ratio, Kolmogorov, Wasserstein and KL, as a comparison record).
 *
 * @index estimator Biased coin (Beta-Bernoulli) or counts (Gamma-Poisson): one inference method checked against the exact posterior.
 */

import { beta, gamma } from '../dists/index.ts'
import { posteriorComparison } from '../exposition/index.ts'
import type { InferenceMethod } from '../probprog/inference/method.ts'
import { betaBernoulliModel, gammaPoissonModel } from '../probprog/models/conjugate.ts'
import type { Posterior } from '../probprog/posterior.ts'
import {
  agree,
  betaBernoulli as betaBernoulliExact,
  distributionOracle,
  type ExactPosterior,
  fromPosterior,
  gammaPoisson as gammaPoissonExact,
  type MarginalOracle,
} from '../probprog/validation/index.ts'
import { bernoulli as drawBernoulli, poisson as drawPoisson, foldIn, key } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Config, derived } from '../trace/index.ts'
import { defineExample, type Example, type ExampleProblem, exampleNumber, exampleString } from './example-method.ts'

/** What the coin is: flips of a coin (Beta-Bernoulli) or counts of events (Gamma-Poisson). */
export const coinKinds = ['coin', 'counts'] as const

/** The problem a coin configuration stands for. */
export interface CoinProblem extends ExampleProblem {
  readonly kind: 'coin' | 'counts'
  /** The observations: flips (0 or 1) or counts. */
  readonly y: readonly number[]
  /** The latent site: the probability `p` or the rate `rate`. */
  readonly site: 'p' | 'rate'
  /** The true probability or rate the data were generated from. */
  readonly truth: number
  readonly prior: { readonly a: number; readonly b: number }
}

/**
 * The problem a configuration stands for: `flips` observations simulated from the true probability (or rate) with the
 * seed (the first `n` of a longer run are the same, so a prefix is the data seen so far) and the conjugate model with
 * its prior.
 *
 * @param config - The knobs (`kind`, `flips`, `trueP`, `trueRate`, `dataSeed`, `priorA`, `priorB`).
 * @returns The problem.
 * @example
 * ```ts
 * import { coinProblem } from 'omai-engine/bayesian-examples'
 * const p = coinProblem({ kind: 'coin', flips: 8, trueP: 0.5, trueRate: 3, dataSeed: 1, priorA: 2, priorB: 2 })
 * [p.y.length, p.site] // => [8, 'p']
 * ```
 */
export function coinProblem(config: Config): CoinProblem {
  const kind = exampleString(config, 'kind') === 'counts' ? 'counts' : 'coin'
  const n = Math.max(1, Math.round(exampleNumber(config, 'flips')))
  const seed = foldIn(key(exampleNumber(config, 'dataSeed')), 0)
  const prior = { a: exampleNumber(config, 'priorA'), b: exampleNumber(config, 'priorB') }
  if (kind === 'counts') {
    const truth = exampleNumber(config, 'trueRate')
    const y = Array.from(toFlat(drawPoisson(seed, truth, [n])))
    return {
      kind,
      y,
      site: 'rate',
      truth,
      prior,
      definition: gammaPoissonModel({ shape: prior.a, rate: prior.b }),
      data: { k: y },
    }
  }
  const truth = exampleNumber(config, 'trueP')
  const y = Array.from(toFlat(drawBernoulli(seed, truth, [n])), Number)
  return { kind, y, site: 'p', truth, prior, definition: betaBernoulliModel(prior), data: { y } }
}

/** The exact posterior after the first `seen` observations. */
function exactAt(problem: CoinProblem, seen: number): ExactPosterior {
  const y = problem.y.slice(0, seen)
  return problem.kind === 'counts'
    ? gammaPoissonExact({ shape: problem.prior.a, rate: problem.prior.b, counts: y })
    : betaBernoulliExact({ a: problem.prior.a, b: problem.prior.b, y })
}

/** The knobs of the data: coin or counts, how many, the truth and the seed. */
export const coinDataKnobs: DictSpace = dict({
  kind: choice([...coinKinds], {
    default: 'coin',
    label: 'what is observed',
    descriptions: {
      coin: 'flips of a coin: heads or tails (Beta-Bernoulli)',
      counts: 'counts of events in equal intervals (Gamma-Poisson)',
    },
  }),
  flips: int(1, 500, { default: 20, log: true, label: 'flips or counts' }),
  trueP: real(0.02, 0.98, { default: 0.3, step: 0.01, label: 'true probability of heads' }),
  trueRate: real(0.2, 20, { default: 3, log: true, label: 'true rate (counts)' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the two numbers of the prior (Beta a and b, or Gamma shape and rate). */
export const coinModelKnobs: DictSpace = dict({
  priorA: real(0.1, 50, {
    default: 2,
    log: true,
    label: 'prior a',
    description: 'Beta a (prior heads), or the Gamma shape of the rate',
  }),
  priorB: real(0.1, 50, {
    default: 2,
    log: true,
    label: 'prior b',
    description: 'Beta b (prior tails), or the Gamma rate of the rate',
  }),
})

/** The knobs that make the problem. */
export const coinProblemKnobs: readonly string[] = [
  'kind',
  'flips',
  'trueP',
  'trueRate',
  'dataSeed',
  'priorA',
  'priorB',
]

const GRID_POINTS = 120
const LADDER = 60
const num = (v: unknown): number => (typeof v === 'number' ? v : Number.NaN)

/** The knobs a configuration does not use for the observation it makes (for a page to leave out). */
export function coinHiddenKnobs(config: Config): readonly string[] {
  return exampleString(config, 'kind') === 'counts' ? ['trueP'] : ['trueRate']
}

/** The draws of a posterior's site, with their mean, sd and the 95% interval. */
function summaryOf(post: Posterior, site: string): { mean: number; sd: number; lo: number; hi: number } {
  const q = toFlat(post.quantiles(site, [0.025, 0.975]))
  return { mean: num(toFlat(post.mean(site))[0]), sd: num(toFlat(post.stddev(site))[0]), lo: num(q[0]), hi: num(q[1]) }
}

const coinExample: Example<CoinProblem> = defineExample<CoinProblem>({
  name: 'coin',
  description:
    'Beta-Bernoulli coin or Gamma-Poisson counts with one inference method checked against the exact posterior',
  modelName: 'coin',
  dataKnobs: coinDataKnobs,
  modelKnobs: coinModelKnobs,
  problemKnobs: coinProblemKnobs,
  problem: coinProblem,
  derived: ({ problem, config, kind, posterior: post, fit }) => {
    const { site } = problem
    const n = problem.y.length
    const next = problem.kind === 'counts' ? 'expected next count' : 'probability the next flip is heads'
    return {
      'coin/updating': derived<unknown>({
        cost: 'dear',
        description:
          'the posterior after the first `seen` observations (args `{ seen }`): the chosen method refitted on them and the exact posterior, as a comparison record, with the means, intervals and the prediction of the next observation',
        compute: (state, args) => {
          const asked = Math.round(Number((args as { seen?: number } | null | undefined)?.seen ?? n))
          const seen = Math.max(1, Math.min(n, asked))
          const prefix = problem.y.slice(0, seen)
          const exact = exactAt(problem, seen)
          const [oracle] = exact.oracles(site)
          if (oracle === undefined) return null
          let method: Posterior
          try {
            method = seen >= n ? post(state) : fit({ ...config, flips: seen })
          } catch {
            return null
          }
          if (!method.sites.includes(site)) return null
          const approx = fromPosterior(kind, method, { sites: [site] })
          const table = agree([approx], { oracle: { [site]: oracle } })
          const m = summaryOf(method, site)
          const e = { mean: oracle.mean, sd: oracle.sd, lo: oracle.quantile(0.025), hi: oracle.quantile(0.975) }
          return {
            seen,
            total: n,
            successes: prefix.reduce((a, v) => a + v, 0),
            truth: problem.truth,
            comparison: posteriorComparison([approx], { truth: { [site]: oracle }, table, gridPoints: GRID_POINTS }),
            exact: e,
            method: m,
            predictiveLabel: next,
            exactPredictive: e.mean,
            methodPredictive: m.mean,
          } as never
        },
      }),
      'coin/contraction': derived<unknown>({
        cost: 'dear',
        description:
          'the exact posterior as observations accumulate: its mean and 95% interval after each of a ladder of counts, the running proportion of the data and the true value',
        compute: () => {
          const ns = Array.from(
            new Set([0, ...Array.from({ length: LADDER }, (_, i) => Math.max(1, Math.round(((i + 1) * n) / LADDER)))]),
          )
          const prior = priorOracle(problem)
          const pts = ns.map((k) => {
            const o = k === 0 ? undefined : exactAt(problem, k).oracles(site)[0]
            const y = problem.y.slice(0, k)
            const running = k === 0 ? Number.NaN : y.reduce((a, v) => a + v, 0) / k
            return o === undefined
              ? { n: 0, mean: prior.mean, lo: prior.quantile(0.025), hi: prior.quantile(0.975), running }
              : { n: k, mean: o.mean, lo: o.quantile(0.025), hi: o.quantile(0.975), running }
          })
          return { truth: problem.truth, points: pts, site } as never
        },
      }),
      'coin/check': derived<unknown>({
        cost: 'dear',
        description:
          'the method against the exact posterior of all the data: a comparison record (densities, ECDFs and the agreement table) and the distances (mean error in exact sds, spread ratio, Kolmogorov, Wasserstein, KL), with the exact log evidence',
        compute: (state) => {
          const method = post(state)
          const exact = exactAt(problem, n)
          const [oracle] = exact.oracles(site)
          if (oracle === undefined || !method.sites.includes(site)) return null
          const approx = fromPosterior(kind, method, { sites: [site] })
          const table = agree([approx], { oracle: { [site]: oracle } })
          const row = table.rows[0]
          if (row === undefined) return null
          return {
            comparison: posteriorComparison([approx], { truth: { [site]: oracle }, table, gridPoints: GRID_POINTS }),
            exactMean: oracle.mean,
            exactSd: oracle.sd,
            methodMean: row.mean,
            methodSd: row.sd,
            meanError: row.meanError,
            sdRatio: row.sdRatio,
            ks: row.ks,
            wasserstein: row.wasserstein,
            kl: row.kl,
            ess: row.ess,
            numDraws: row.numDraws,
            logEvidence: exact.logEvidence,
          } as never
        },
      }),
    }
  },
})

// The prior as an oracle of the same site (the posterior before any observation).
function priorOracle(p: CoinProblem): MarginalOracle {
  return p.kind === 'counts'
    ? distributionOracle('rate', gamma(p.prior.a, p.prior.b))
    : distributionOracle('p', beta(p.prior.a, p.prior.b))
}

/**
 * The biased coin (or counting) example as a method, run by name and knobs: see the file's summary. A trace `Method` whose
 * tape is the chosen inference method's own, with the derived paths `coin/updating`, `coin/contraction` and `coin/check`.
 *
 * @concept inference-engines
 * @index estimator Biased coin method: Beta-Bernoulli or Gamma-Poisson, one inference method, checked against the exact posterior.
 * @example
 * ```ts
 * import { coinMethod, coinSteps } from 'omai-engine/bayesian-examples'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(coinMethod.config), inference: { method: 'conjugate' } }
 * const s = session(coinMethod, { config, steps: coinSteps(config), mode: 'eager' })
 * const check = s.probe(s.t, 'coin/check') as { meanError: number }
 * Math.abs(check.meanError) < 0.01 // => true
 * ```
 */
export const coinMethod: InferenceMethod<unknown, Config, string> = coinExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const coinSteps: (config: Config) => number = coinExample.steps

export { coinExample }
