/**
 * Stochastic volatility as an example method: returns `r_t ~ N(0, exp(h_t))` whose log-variance `h_t` follows a latent
 * AR(1) around a mean `mu` with persistence `phi` and volatility of volatility `sigma`. The data are simulated from known
 * values of the three numbers (a length and a seed), or are the library's GARCH(1,1) returns with their true conditional
 * variance (Gaussian or Student-t innovations: a series the model is not the generator of). The model is written as a
 * `scan` in either parameterisation, centred (the states `h_t`) or non-centred (standardised states `z_t` with
 * `h_t = mu + sigma z_t`, so the sampler does not meet the funnel between `sigma` and the states), with `mu`, `phi` and
 * `sigma` inferred or known; one inference method of the library with its own knobs and a budget infers the posterior.
 * With the numbers known the states have an independent truth (a point-mass smoother on a grid, `stochasticVolatilityTruth`)
 * and the particle filter applies; with them inferred the reference is a long NUTS run of the non-centred form. The derived
 * paths read the run: `sv/fit` (the returns with the volatility band, the log-variance path with its band against the truth,
 * the width of the band against the size of the return), `sv/pairs` (the three numbers, or the log-variance at three times),
 * `sv/scores` (coverage of the truth by the band, error of the mean, the numbers against the truth, R-hat, ESS), `sv/diagnostics`,
 * `sv/draws` (what a check needs of a reference run) and `sv/check` (the method against the smoother or the reference draws).
 *
 * @index estimator Stochastic volatility: returns with an AR(1) log-variance, centred or non-centred, one inference method, checked.
 */

import { generate } from '../data/index.ts'
import { garchReturns } from '../datasets/index.ts'
import { posteriorComparison, posteriorPairs } from '../exposition/index.ts'
import type { InferenceMethod } from '../probprog/inference/method.ts'
import * as d from '../probprog/model/dists/index.ts'
import { type DataInput, defineModel, type ModelDefinition, ops } from '../probprog/model/index.ts'
import { stochasticVolatilityTruth } from '../probprog/models/series.ts'
import type { Posterior } from '../probprog/posterior.ts'
import { type Approximation, agree, type MarginalOracle, weightedOracle } from '../probprog/validation/index.ts'
import { normal as drawNormal, foldIn, key } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { quantile } from '../stats/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { type Config, derived } from '../trace/index.ts'
import {
  columnMoments,
  defineExample,
  drawRows,
  type Example,
  type ExampleProblem,
  exampleDiagnostics,
  exampleNumber,
  exampleString,
} from './example-method.ts'

/** The data sets on offer: simulated from the model, or GARCH(1,1) returns (Gaussian or Student-t innovations). */
export const volatilityDatasets = ['simulated', 'garch', 'garch-student'] as const
/** The parameterisations of the latent states. */
export const volatilityParameterisations = ['non-centred', 'centred'] as const

/** The problem a stochastic volatility configuration stands for. */
export interface VolatilityProblem extends ExampleProblem {
  /** The returns. */
  readonly y: readonly number[]
  /** The true log-variance at each step (the simulated one, or the log of the GARCH conditional variance). */
  readonly truthLogVariance: readonly number[]
  /** The generating `mu`, `phi` and `sigma` when the data are simulated from the model. */
  readonly truth: { readonly mu: number; readonly phi: number; readonly sigma: number } | null
  /** Whether the three numbers are known (and used as given) rather than inferred. */
  readonly known: boolean
  readonly centred: boolean
  /** The name of the scan's site: the log-variance `h`, or the standardised `z`. */
  readonly site: 'h' | 'z'
}

const TRUTH_POINTS = 400

/**
 * The problem a configuration stands for: the returns (simulated from `mu`, `phi` and `sigma` or from the GARCH generator,
 * the first `T` of a longer run being the same), their true log-variance, and the model with the chosen
 * parameterisation and the three numbers inferred (priors: `mu` normal, `phi` uniform on (-0.99, 0.99), `sigma` half-normal)
 * or known.
 *
 * @param config - The knobs (`dataset`, `length`, `mu`, `phi`, `sigmaH`, `dataSeed`, `parameters`, `parameterisation`, `muScale`, `sigmaScale`).
 * @returns The problem.
 * @example
 * ```ts
 * import { volatilityProblem } from 'omai-engine/bayesian-examples'
 * const p = volatilityProblem({ dataset: 'simulated', length: 50, mu: -1, phi: 0.95, sigmaH: 0.3, dataSeed: 1, parameters: 'inferred', parameterisation: 'non-centred', muScale: 2, sigmaScale: 0.5 })
 * [p.y.length, p.site] // => [50, 'z']
 * ```
 */
export function volatilityProblem(config: Config): VolatilityProblem {
  const n = Math.max(5, Math.round(exampleNumber(config, 'length')))
  const dataset = exampleString(config, 'dataset')
  const seed = key(exampleNumber(config, 'dataSeed'))
  const mu = exampleNumber(config, 'mu')
  const phi = exampleNumber(config, 'phi')
  const sigma = exampleNumber(config, 'sigmaH')
  let y: number[]
  let h: number[]
  let truth: VolatilityProblem['truth'] = null
  if (dataset === 'garch' || dataset === 'garch-student') {
    const g = generate(garchReturns, {
      knobs: { n, innovations: dataset === 'garch' ? 'gaussian' : 'student' },
      key: foldIn(seed, 0),
    })
    y = Array.from(toFlat(g.data.values))
    h = Array.from(toFlat(g.truth.latents?.['variance'] as never), (v) => Math.log(v))
  } else {
    const e = toFlat(drawNormal(foldIn(seed, 0), [n]))
    const r = toFlat(drawNormal(foldIn(seed, 1), [n]))
    const s0 = sigma / Math.sqrt(1 - phi ** 2)
    h = []
    for (let t = 0; t < n; t++)
      h.push(t === 0 ? mu + s0 * (e[0] ?? 0) : mu + phi * ((h[t - 1] ?? 0) - mu) + sigma * (e[t] ?? 0))
    y = h.map((v, t) => Math.exp(v / 2) * (r[t] ?? 0))
    truth = { mu, phi, sigma }
  }
  const known = truth !== null && exampleString(config, 'parameters') === 'known'
  const centred = exampleString(config, 'parameterisation') === 'centred'
  return {
    y,
    truthLogVariance: h,
    truth,
    known,
    centred,
    site: centred ? 'h' : 'z',
    definition: definitionOf(
      centred,
      known && truth !== null ? truth : null,
      exampleNumber(config, 'muScale'),
      exampleNumber(config, 'sigmaScale'),
    ),
    data: { y } as DataInput,
  }
}

/** The model: the log-variance as a scan, centred or whitened, the numbers latent or fixed at `known`. */
function definitionOf(
  centred: boolean,
  known: { readonly mu: number; readonly phi: number; readonly sigma: number } | null,
  muScale: number,
  sigmaScale: number,
): ModelDefinition {
  return defineModel('stochastic-volatility', (m) => {
    const y = m.data('y', { dims: ['time'], role: 'response' })
    const mu = known === null ? m.latent('mu', d.normal(0, muScale), { doc: 'the mean log-variance' }) : known.mu
    const phi =
      known === null
        ? m.latent('phi', d.uniform(-0.99, 0.99), { doc: 'the persistence of the log-variance' })
        : known.phi
    const sigma =
      known === null
        ? m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the volatility of the log-variance' })
        : known.sigma
    // The stationary sd of the AR(1): sigma / sqrt(1 - phi^2); the standardised state has sd 1 / sqrt(1 - phi^2).
    const stationary = ops.div(1, ops.sqrt(ops.sub(1, ops.square(phi))))
    if (centred)
      m.scan('h', {
        over: 'time',
        init: d.normal(mu, ops.mul(sigma, stationary)),
        step: ({ prev }) => d.normal(ops.add(mu, ops.mul(phi, ops.sub(prev, mu))), sigma),
        emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(ops.mul(state, 0.5))), y),
        doc: 'the log-variance of the returns',
      })
    else
      m.scan('z', {
        over: 'time',
        init: d.normal(0, stationary),
        step: ({ prev }) => d.normal(ops.mul(phi, prev), 1),
        emit: ({ state, observe }) =>
          observe('y', d.normal(0, ops.exp(ops.mul(ops.add(mu, ops.mul(sigma, state)), 0.5))), y),
        doc: 'the standardised log-variance: h = mu + sigma z',
      })
  })
}

/** The knobs of the data: simulated (with its three numbers) or GARCH, the length and the seed. */
export const volatilityDataKnobs: DictSpace = dict({
  dataset: choice([...volatilityDatasets], {
    default: 'simulated',
    label: 'data set',
    descriptions: {
      simulated: 'returns simulated from the model with the mean, persistence and volatility of volatility below',
      garch: "the library's GARCH(1,1) returns (volatility clustering, but not this model), the true variance as truth",
      'garch-student':
        'the same with Student-t innovations: heavy-tailed returns, which a model with normal returns reads as bursts of volatility',
    },
  }),
  length: int(20, 400, { default: 120, log: true, label: 'length T' }),
  mu: real(-3, 1, { default: -1, step: 0.1, label: 'mean log-variance mu' }),
  phi: real(0.3, 0.99, { default: 0.95, step: 0.01, label: 'persistence phi' }),
  sigmaH: real(0.05, 1, { default: 0.3, log: true, label: 'volatility of volatility sigma' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the parameterisation, whether the numbers are known, their priors. */
export const volatilityModelKnobs: DictSpace = dict({
  parameterisation: choice([...volatilityParameterisations], {
    default: 'non-centred',
    label: 'parameterisation',
    descriptions: {
      'non-centred':
        'standardised states z with h = mu + sigma z: the sampler does not meet the funnel between sigma and the states',
      centred:
        'the states h themselves: when sigma is small the states are tied tightly to their mean and a sampler struggles (switch to compare)',
    },
  }),
  parameters: choice(['inferred', 'known'], {
    default: 'inferred',
    label: 'mu, phi and sigma',
    descriptions: {
      inferred: 'inferred with the states: the posterior of the three numbers is a part of the answer',
      known:
        'taken as the true values (simulated data only): the states alone are inferred, with an independent truth and the particle filter applicable',
    },
  }),
  muScale: real(0.5, 10, {
    default: 2,
    log: true,
    label: 'prior sd of mu',
    description: 'mu has a normal prior centred at zero',
  }),
  sigmaScale: real(0.1, 3, {
    default: 0.5,
    log: true,
    label: 'prior scale of sigma',
    description: 'sigma has a half-normal prior; phi is uniform on (-0.99, 0.99)',
  }),
})

/** The knobs that make the problem. */
export const volatilityProblemKnobs: readonly string[] = [
  'dataset',
  'length',
  'mu',
  'phi',
  'sigmaH',
  'dataSeed',
  'parameterisation',
  'parameters',
  'muScale',
  'sigmaScale',
]

/** The knobs a configuration does not use (the generating numbers of GARCH data, the known switch for them). */
export function volatilityHiddenKnobs(config: Config): readonly string[] {
  return exampleString(config, 'dataset') === 'simulated' ? [] : ['mu', 'phi', 'sigmaH', 'parameters']
}

const FIT_DRAWS = 300
const PAIR_DRAWS = 400
const LATENT_DRAWS = 300

/** What a check against a reference run needs of it: its draws, one column per number and per log-variance. */
export interface VolatilityReference {
  readonly names: readonly string[]
  readonly columns: readonly (readonly number[])[]
}

/** The draws of a posterior as the three numbers and the log-variance paths. */
interface Draws {
  readonly count: number
  readonly mu: number[]
  readonly phi: number[]
  readonly sigma: number[]
  /** `h[s][t]`. */
  readonly h: number[][]
  readonly chain: number[]
  readonly index: number[]
}

/** The sites of a run: the three numbers (when inferred) and the state at each step (a scan's states are the scalar sites `h[t]` or `z[t]`). */
function sitesOf(problem: VolatilityProblem): string[] {
  const states = Array.from({ length: problem.y.length }, (_, t) => `${problem.site}[${t}]`)
  return problem.known ? states : ['mu', 'phi', 'sigma', ...states]
}

function drawsOf(problem: VolatilityProblem, post: Posterior, most: number): Draws | undefined {
  const n = problem.y.length
  const got = drawRows(post, sitesOf(problem), most)
  if (got === undefined) return undefined
  const lead = problem.known ? 0 : 3
  const truth = problem.truth
  const mu: number[] = []
  const phi: number[] = []
  const sigma: number[] = []
  const h: number[][] = []
  for (const row of got.rows) {
    const m = problem.known ? (truth?.mu ?? 0) : (row[0] ?? 0)
    const p = problem.known ? (truth?.phi ?? 0) : (row[1] ?? 0)
    const s = problem.known ? (truth?.sigma ?? 0) : (row[2] ?? 0)
    mu.push(m)
    phi.push(p)
    sigma.push(s)
    const states = row.slice(lead, lead + n)
    h.push(problem.centred ? states : states.map((z) => m + s * z))
  }
  return { count: got.rows.length, mu, phi, sigma, h, chain: got.chain, index: got.index }
}

const column = (rows: readonly (readonly number[])[], j: number): number[] => rows.map((r) => r[j] ?? 0)

const names = (n: number): string[] => Array.from({ length: n }, (_, t) => `h[${t}]`)

/** The names and columns of draws as a check or a reference sees them. */
function columnsOf(problem: VolatilityProblem, dr: Draws): { names: string[]; columns: number[][] } {
  const n = problem.y.length
  return {
    names: [...(problem.known ? [] : ['mu', 'phi', 'sigma']), ...names(n)],
    columns: [...(problem.known ? [] : [dr.mu, dr.phi, dr.sigma]), ...names(n).map((_, t) => column(dr.h, t))],
  }
}

const PAIR_TIMES = [0.2, 0.5, 0.8]

const volatilityExample: Example<VolatilityProblem> = defineExample<VolatilityProblem>({
  name: 'sv',
  description: 'Stochastic volatility (AR(1) log-variance scaling normal returns) with one inference method, checked',
  modelName: 'stochastic-volatility',
  budgetMs: 15000,
  // Each gradient runs through the unrolled recurrence (milliseconds at 30 steps, growing with T): the page opens on the
  // quick variational method, and NUTS on one short chain.
  first: 'svi',
  starting: { nuts: { numChains: 1, numWarmup: 60, numDraws: 80 } },
  dataKnobs: volatilityDataKnobs,
  modelKnobs: volatilityModelKnobs,
  problemKnobs: volatilityProblemKnobs,
  problem: volatilityProblem,
  derived: ({ problem, kind, posterior: post }) => {
    const n = problem.y.length
    return {
      'sv/fit': derived<unknown>({
        cost: 'dear',
        description:
          'the log-variance path with its 95% band, the volatility band over the returns, the truth, and the width of the band against the size of each return',
        compute: (state) => {
          const dr = drawsOf(problem, post(state), FIT_DRAWS)
          if (dr === undefined) return null
          const levels = tensor([0.025, 0.975])
          const mean: number[] = []
          const lo: number[] = []
          const hi: number[] = []
          for (let t = 0; t < n; t++) {
            const c = column(dr.h, t)
            const q = toFlat(quantile(c, levels))
            mean.push(c.reduce((a, v) => a + v, 0) / c.length)
            lo.push(q[0] ?? Number.NaN)
            hi.push(q[1] ?? Number.NaN)
          }
          return {
            t: Array.from({ length: n }, (_, i) => i),
            y: problem.y,
            truth: problem.truthLogVariance,
            mean,
            lo,
            hi,
            width: hi.map((v, t) => v - (lo[t] ?? 0)),
            draws: dr.count,
          } as never
        },
      }),
      'sv/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of mu, phi and sigma (when inferred) or of the log-variance at three times (a posteriorPairs record), with the true values marked',
        compute: (state) => {
          const dr = drawsOf(problem, post(state), PAIR_DRAWS)
          if (dr === undefined || dr.count < 3) return null
          const pick = PAIR_TIMES.map((q) => Math.min(n - 1, Math.round(q * (n - 1))))
          const coordinates = problem.known ? pick.map((t) => `h[${t}]`) : ['mu', 'phi', 'sigma']
          const cols = problem.known ? pick.map((t) => column(dr.h, t)) : [dr.mu, dr.phi, dr.sigma]
          const draws = Object.fromEntries(coordinates.map((c, j) => [c, cols[j] ?? []]))
          const truth = problem.known
            ? Object.fromEntries(pick.map((t) => [`h[${t}]`, problem.truthLogVariance[t] ?? 0]))
            : problem.truth === null
              ? undefined
              : { mu: problem.truth.mu, phi: problem.truth.phi, sigma: problem.truth.sigma }
          return posteriorPairs([{ name: kind, draws, steps: dr.index, chain: dr.chain }], {
            coordinates,
            maxPoints: 300,
            frames: 2,
            densityDraws: 300,
            ...(truth === undefined ? {} : { truth }),
          }) as never
        },
      }),
      'sv/scores': derived<unknown>({
        cost: 'dear',
        description:
          'the fraction of times the truth is inside the 95% band, the error of the mean log-variance, the mean width of the band, the three numbers against the truth, R-hat, ESS and divergences',
        compute: (state) => {
          const posterior = post(state)
          const dr = drawsOf(problem, posterior, FIT_DRAWS)
          if (dr === undefined) return null
          const levels = tensor([0.025, 0.975])
          let inside = 0
          let squared = 0
          let width = 0
          for (let t = 0; t < n; t++) {
            const c = column(dr.h, t)
            const q = toFlat(quantile(c, levels))
            const m = c.reduce((a, v) => a + v, 0) / c.length
            const truth = problem.truthLogVariance[t] ?? 0
            if (truth >= (q[0] ?? Number.NaN) && truth <= (q[1] ?? Number.NaN)) inside++
            squared += (m - truth) ** 2
            width += (q[1] ?? 0) - (q[0] ?? 0)
          }
          const moments = (v: number[]): { mean: number; sd: number } => {
            const m = columnMoments(
              v.map((x) => [x]),
              1,
            )
            return { mean: m.mean[0] ?? 0, sd: m.sd[0] ?? 0 }
          }
          const rows = posterior.summary()
          const rhat = rows.flatMap((r) => (r.rhat === undefined ? [] : [r.rhat]))
          const ess = rows.flatMap((r) => (r.ess === undefined ? [] : [r.ess]))
          const divergences = posterior.diagnostics['divergences']
          return {
            coverage: inside / n,
            rmse: Math.sqrt(squared / n),
            meanWidth: width / n,
            numbers: problem.known
              ? null
              : {
                  mu: moments(dr.mu),
                  phi: moments(dr.phi),
                  sigma: moments(dr.sigma),
                },
            truth: problem.truth,
            draws: posterior.chains === undefined ? dr.count : posterior.chains.numChains * posterior.chains.numDraws,
            maxRhat: rhat.length > 0 ? Math.max(...rhat) : null,
            minEss: ess.length > 0 ? Math.min(...ess) : null,
            divergences: typeof divergences === 'number' ? divergences : null,
          }
        },
      }),
      'sv/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler for mu, phi, sigma and a few states (a chainDiagnostics record); null for the others',
        compute: (state) => exampleDiagnostics(post(state), sitesOf(problem), (name) => name, 5) as never,
      }),
      'sv/draws': derived<unknown>({
        cost: 'dear',
        description:
          'the draws of mu, phi, sigma (when inferred) and of the log-variance at every step, one column each',
        compute: (state) => {
          const dr = drawsOf(problem, post(state), LATENT_DRAWS)
          return dr === undefined ? null : (columnsOf(problem, dr) as never)
        },
      }),
      'sv/check': derived<unknown>({
        cost: 'dear',
        description:
          "the method against the point-mass smoother (numbers known) or against reference draws (args `{ reference }`, the reference's `sv/draws`): a comparison record, the errors and the spread ratios",
        compute: (state, args) => {
          const ref = (args as { reference?: VolatilityReference | null } | null | undefined)?.reference
          const dr = drawsOf(problem, post(state), LATENT_DRAWS)
          if (dr === undefined) return null
          const { names: qn, columns } = columnsOf(problem, dr)
          const approx: Approximation = {
            name: kind,
            draws: Object.fromEntries(qn.map((q, j) => [q, Float64Array.from(columns[j] ?? [])])),
          }
          const oracle: Record<string, MarginalOracle> = {}
          if (problem.known) {
            const truth = stochasticVolatilityTruth(
              problem.y,
              TRUTH_POINTS,
              problem.truth ?? { mu: -1, phi: 0.95, sigma: 0.25 },
            )
            for (const [q, o] of Object.entries(truth.marginals)) oracle[q] = o
          } else {
            if (ref === undefined || ref === null) return null
            ref.names.forEach((q, j) => {
              const c = ref.columns[j]
              if (c !== undefined && q in approx.draws) oracle[q] = weightedOracle(q, c)
            })
          }
          const quantities = Object.keys(oracle).filter((q) => q in approx.draws)
          if (quantities.length === 0) return null
          const table = agree([approx], { oracle, quantities })
          const rows = table.rows
          const picked = PAIR_TIMES.map((q) => `h[${Math.min(n - 1, Math.round(q * (n - 1)))}]`)
          const shown = quantities.filter((q) => ['mu', 'phi', 'sigma'].includes(q) || picked.includes(q))
          return {
            comparison: posteriorComparison(
              [{ ...approx, draws: Object.fromEntries(shown.map((q) => [q, approx.draws[q] as ArrayLike<number>])) }],
              {
                truth: Object.fromEntries(shown.map((q) => [q, oracle[q] as MarginalOracle])),
                table: { ...table, quantities: shown, rows: rows.filter((r) => shown.includes(r.quantity)) },
                quantities: shown,
                gridPoints: 100,
                oracle: problem.known ? 'the point-mass smoother' : 'a long NUTS run',
              },
            ),
            maxMeanError: Math.max(...rows.map((r) => Math.abs(r.meanError))),
            medianSdRatio:
              [...rows.map((r) => r.sdRatio)].sort((a, b) => a - b)[Math.floor(rows.length / 2)] ?? Number.NaN,
            maxWasserstein: Math.max(...rows.map((r) => r.wasserstein)),
            numbers: ['mu', 'phi', 'sigma'].flatMap((q) => {
              const r = rows.find((x) => x.quantity === q)
              return r === undefined ? [] : [{ name: q, meanError: r.meanError, sdRatio: r.sdRatio }]
            }),
            against: problem.known ? 'smoother' : 'reference',
            referenceDraws: ref?.columns[0]?.length ?? 0,
          } as never
        },
      }),
    }
  },
})

/**
 * Stochastic volatility as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape is the
 * chosen inference method's own, with the derived paths `sv/fit`, `sv/pairs`, `sv/scores`, `sv/diagnostics`, `sv/draws` and
 * `sv/check`.
 *
 * @concept state-space-models
 * @index estimator Stochastic volatility method: AR(1) log-variance, centred or non-centred, one inference method.
 * @example
 * ```ts
 * import { volatilityMethod, volatilitySteps } from 'omai-engine/bayesian-examples'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(volatilityMethod.config), length: 40, inference: { method: 'laplace' } }
 * const s = session(volatilityMethod, { config, steps: volatilitySteps(config), mode: 'eager' })
 * const fit = s.probe(s.t, 'sv/fit') as { t: number[] }
 * fit.t.length // => 40
 * ```
 */
export const volatilityMethod: InferenceMethod<unknown, Config, string> = volatilityExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const volatilitySteps: (config: Config) => number = volatilityExample.steps

export { volatilityExample }
