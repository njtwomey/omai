/**
 * Bayesian linear regression as an example method: data simulated from a polynomial with known coefficients (the number of
 * rows, the noise, how the inputs are placed, a seed), a model of a polynomial of any degree with a normal prior on the
 * coefficients and the noise either known or learned, and one inference method of the library with its own knobs and a
 * budget. With the noise known the posterior of the coefficients is a normal in closed form (`bayesianLinearFit`), the
 * exact answer every method is checked against; with the noise learned there is none, and the check is against a long NUTS
 * run of the same problem (the page runs it in a worker and hands its draws to `linreg/check`). The derived paths read the
 * run: `linreg/fit` (the fitted curve with its band and sampled lines after the first `seen` rows, the method refitted on
 * them, the coefficients against the truth), `linreg/pairs` (the pair plot of the coefficients and the noise: the
 * correlation between intercept and slope), `linreg/draws` (the draws, as the reference's answer) and `linreg/check` (the
 * method against the exact posterior or the reference draws, as a comparison record with the distances).
 *
 * @index estimator Bayesian linear regression: polynomial data with known truth, one inference method, checked against the exact posterior.
 */

import { normal as normalDist } from '../dists/index.ts'
import { posteriorComparison, posteriorPairs } from '../exposition/index.ts'
import { bayesianLinearFit, designOf } from '../linear/index.ts'
import type { InferenceMethod } from '../probprog/inference/method.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel } from '../probprog/model/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
import {
  agree,
  distributionOracle,
  fromPosterior,
  type MarginalOracle,
  weightedOracle,
} from '../probprog/validation/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, key } from '../random/index.ts'
import { choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
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

/** The functions the data are drawn from: the coefficients of `1, x, x^2, x^3`. */
const TRUTH: Readonly<Record<string, readonly number[]>> = {
  line: [1, 2],
  quadratic: [1, 2, -1.5],
  cubic: [0.5, 2, -1.5, 1.2],
}
/** The truths on offer. */
export const linearTruths: readonly string[] = Object.keys(TRUTH)
/** Where the inputs lie: around zero (the coefficients are nearly uncorrelated) or away from it (intercept and slope trade off). */
const RANGE: Readonly<Record<string, readonly [number, number]>> = { centred: [-1, 1], offset: [1, 3] }
/** The placements of the inputs on offer. */
export const linearInputs: readonly string[] = Object.keys(RANGE)

/** The problem a linear regression configuration stands for. */
export interface LinearProblem extends ExampleProblem {
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** The coefficients of the generating polynomial (padded with zeros to the fitted degree). */
  readonly truth: readonly number[]
  readonly noiseSd: number
  readonly learned: boolean
  readonly degree: number
  /** The names of the coefficients: `intercept`, `slope`, `x^2`, ... */
  readonly names: readonly string[]
  readonly extent: readonly [number, number]
  readonly priorScale: number
  /** The design rows `[n][degree + 1]` of the data. */
  readonly design: readonly (readonly number[])[]
}

const monomials = (x: number, degree: number): number[] => Array.from({ length: degree + 1 }, (_, j) => x ** j)
const polynomial = (w: readonly number[], x: number): number => w.reduce((a, c, j) => a + c * x ** j, 0)

/**
 * The problem a configuration stands for: `rows` inputs placed in the range, the outputs of the polynomial with the
 * known coefficients plus noise (the first `n` rows of a longer run are the same, so a prefix is the data seen so far),
 * the polynomial design of the fitted degree and the model: a normal prior on every coefficient and the noise either
 * fixed at its true value (known) or given a half-normal prior (learned).
 *
 * @param config - The knobs (`truth`, `inputs`, `rows`, `noise`, `dataSeed`, `degree`, `priorScale`, `noiseModel`, `sigmaScale`).
 * @returns The problem.
 * @example
 * ```ts
 * import { linearRegressionProblem } from 'omai-engine/bayesian-examples'
 * const p = linearRegressionProblem({ truth: 'line', inputs: 'centred', rows: 20, noise: 0.5, dataSeed: 1, degree: 2, priorScale: 3, noiseModel: 'known', sigmaScale: 1 })
 * [p.x.length, p.names] // => [20, ['intercept', 'slope', 'x^2']]
 * ```
 */
export function linearRegressionProblem(config: Config): LinearProblem {
  const n = Math.max(2, Math.round(exampleNumber(config, 'rows')))
  const degree = Math.max(1, Math.round(exampleNumber(config, 'degree')))
  const range = RANGE[exampleString(config, 'inputs')] ?? [-1, 1]
  const generating = TRUTH[exampleString(config, 'truth')] ?? [1, 2]
  const noiseSd = exampleNumber(config, 'noise')
  const priorScale = exampleNumber(config, 'priorScale')
  const learned = exampleString(config, 'noiseModel') === 'learned'
  const sigmaScale = exampleNumber(config, 'sigmaScale')
  const seed = foldIn(key(exampleNumber(config, 'dataSeed')), 0)
  const u = toFlat(drawUniform(foldIn(seed, 0), [n]))
  const e = toFlat(drawNormal(foldIn(seed, 1), [n]))
  const x = Array.from(u, (v) => range[0] + (range[1] - range[0]) * v)
  const truth = Array.from({ length: degree + 1 }, (_, j) => generating[j] ?? 0)
  const y = x.map((v, i) => polynomial(generating, v) + noiseSd * (e[i] ?? 0))
  const design = x.map((v) => monomials(v, degree))
  const definition = defineModel('linear-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const X = m.data('X', { dims: [row, feature], role: 'covariate' })
    const yy = m.data('y', { dims: [row], role: 'response' })
    const w = m.latent('w', d.normal(0, priorScale), { dims: [feature], doc: 'the coefficient of each power of x' })
    if (learned) {
      const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise standard deviation' })
      m.observe('y', d.normal(X.matmul(w), sigma), yy)
    } else m.observe('y', d.normal(X.matmul(w), noiseSd), yy)
  })
  return {
    x,
    y,
    truth,
    noiseSd,
    learned,
    degree,
    names: ['intercept', 'slope', ...Array.from({ length: Math.max(0, degree - 1) }, (_, j) => `x^${j + 2}`)],
    extent: range as readonly [number, number],
    priorScale,
    design,
    definition,
    data: { X: design, y },
  }
}

/** The knobs of the data: the truth, where the inputs lie, how many rows, the noise and the seed. */
export const linearDataKnobs: DictSpace = dict({
  truth: choice(linearTruths, {
    default: 'line',
    label: 'true function',
    descriptions: {
      line: 'a line: intercept 1, slope 2',
      quadratic: 'a parabola: 1 + 2x - 1.5x^2',
      cubic: 'a cubic: 0.5 + 2x - 1.5x^2 + 1.2x^3',
    },
  }),
  inputs: choice(linearInputs, {
    default: 'centred',
    label: 'inputs',
    descriptions: {
      centred: 'x in [-1, 1]: the intercept and the slope are nearly uncorrelated in the posterior',
      offset: 'x in [1, 3], away from zero: intercept and slope trade off and are strongly correlated',
    },
  }),
  rows: int(2, 400, { default: 30, log: true, label: 'rows' }),
  noise: real(0.02, 3, { default: 0.5, log: true, label: 'noise sd' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the degree, the scale of the prior and whether the noise is known or learned. */
export const linearModelKnobs: DictSpace = dict({
  degree: int(1, 6, {
    default: 1,
    label: 'polynomial degree',
    description: 'the model is a polynomial of this degree',
  }),
  priorScale: real(0.1, 30, {
    default: 3,
    log: true,
    label: 'prior scale',
    description: 'standard deviation of the normal prior on each coefficient (of the powers of x)',
  }),
  noiseModel: choice(['known', 'learned'], {
    default: 'known',
    label: 'noise',
    descriptions: {
      known: 'the noise sd is known (its true value): the posterior of the coefficients is a normal in closed form',
      learned: 'the noise sd is unknown, with a half-normal prior: no closed form, the reference is a long NUTS run',
    },
  }),
  sigmaScale: real(0.1, 10, {
    default: 1,
    log: true,
    label: 'noise prior scale',
    description: 'scale of the half-normal prior on the noise sd (when it is learned)',
  }),
})

/** The knobs that make the problem. */
export const linearProblemKnobs: readonly string[] = [
  'truth',
  'inputs',
  'rows',
  'noise',
  'dataSeed',
  'degree',
  'priorScale',
  'noiseModel',
  'sigmaScale',
]

/** The knobs a configuration does not use (the noise prior of a known noise). */
export function linearHiddenKnobs(config: Config): readonly string[] {
  return exampleString(config, 'noiseModel') === 'learned' ? [] : ['sigmaScale']
}

const GRID = 60
const FIT_DRAWS = 300
const LINES = 12
const PAIR_DRAWS = 500

/** The exact posterior of the coefficients with the noise known: a normal with this mean and covariance. */
function exactOf(problem: LinearProblem, rows: number): { mean: number[]; sd: number[]; cov: number[][] } {
  const design = designOf(
    { x: problem.design.slice(0, rows) as number[][], y: problem.y.slice(0, rows) as number[] },
    'linreg',
  )
  const fit = bayesianLinearFit(design, { alpha: 1 / problem.priorScale ** 2, beta: 1 / problem.noiseSd ** 2 })
  return { mean: fit.mean, sd: fit.covariance.map((r, i) => Math.sqrt(r[i] ?? 0)), cov: fit.covariance }
}

const quantile = (sorted: readonly number[], p: number): number => {
  const i = (sorted.length - 1) * p
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return (sorted[lo] ?? 0) + ((sorted[hi] ?? 0) - (sorted[lo] ?? 0)) * (i - lo)
}

/** The draws of the coefficients (and the noise, when there is one) of a posterior, named as the page shows them. */
function namedDraws(
  post: Posterior,
  problem: LinearProblem,
  n: number,
): { rows: number[][]; names: string[]; chain: number[]; index: number[]; p: number } | undefined {
  const sites = post.sites.includes('sigma') ? ['w', 'sigma'] : ['w']
  const got = drawRows(post, sites, n)
  if (got === undefined) return undefined
  const p = problem.degree + 1
  return {
    rows: got.rows,
    names: [...problem.names, ...(sites.length > 1 ? ['sigma'] : [])],
    chain: got.chain,
    index: got.index,
    p,
  }
}

/** What a check against a reference needs of the reference: its draws, one column per coefficient (and the noise). */
export interface LinearReference {
  readonly names: readonly string[]
  readonly columns: readonly (readonly number[])[]
}

const linearExample: Example<LinearProblem> = defineExample<LinearProblem>({
  name: 'linreg',
  description:
    'Bayesian linear regression on polynomial data with one inference method checked against the exact or a reference posterior',
  modelName: 'linreg',
  dataKnobs: linearDataKnobs,
  modelKnobs: linearModelKnobs,
  problemKnobs: linearProblemKnobs,
  problem: linearRegressionProblem,
  derived: ({ problem, config, kind, posterior: post, fit }) => {
    const n = problem.x.length
    const [lo, hi] = problem.extent
    const pad = 0.08 * (hi - lo)
    const grid = Array.from({ length: GRID }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / (GRID - 1))
    const truthCurve = grid.map((v) => polynomial(problem.truth, v))
    return {
      'linreg/fit': derived<unknown>({
        cost: 'dear',
        description:
          'the fitted curve after the first `seen` rows (args `{ seen }`): the method refitted on them, its mean curve with the 95% band of the curve and of a new observation, sampled lines, the data, the true curve, and the coefficients against the truth (and against the exact posterior when the noise is known)',
        compute: (state, args) => {
          const asked = Math.round(Number((args as { seen?: number } | null | undefined)?.seen ?? n))
          const seen = Math.max(2, Math.min(n, asked))
          let method: Posterior
          try {
            method = seen >= n ? post(state) : fit({ ...config, rows: seen })
          } catch {
            return null
          }
          const got = namedDraws(method, problem, FIT_DRAWS)
          if (got === undefined) return null
          const { rows, p } = got
          const z = toFlat(drawNormal(key(7), [rows.length]))
          const at = grid.map((v) => rows.map((r) => r.slice(0, p).reduce((a, c, j) => a + c * v ** j, 0)))
          const curve = at.map((col) => [...col].sort((a, b) => a - b))
          const predictive = at.map((col) =>
            col
              .map((f, s) => f + (got.names.includes('sigma') ? (rows[s]?.[p] ?? 0) : problem.noiseSd) * (z[s] ?? 0))
              .sort((a, b) => a - b),
          )
          const { mean, sd } = columnMoments(rows, got.names.length)
          const exact = problem.learned ? null : exactOf(problem, seen)
          const stride = Math.max(1, Math.floor(rows.length / LINES))
          return {
            seen,
            total: n,
            x: grid,
            mean: at.map((col) => col.reduce((a, v) => a + v, 0) / col.length),
            lo: curve.map((c) => quantile(c, 0.025)),
            hi: curve.map((c) => quantile(c, 0.975)),
            predictiveLo: predictive.map((c) => quantile(c, 0.025)),
            predictiveHi: predictive.map((c) => quantile(c, 0.975)),
            lines: Array.from({ length: Math.min(LINES, rows.length) }, (_, s) =>
              at.map((col) => col[s * stride] ?? Number.NaN),
            ),
            truth: truthCurve,
            data: { x: problem.x.slice(0, seen), y: problem.y.slice(0, seen) },
            coefficients: got.names.map((name, j) => ({
              name,
              truth: j < p ? (problem.truth[j] ?? 0) : problem.noiseSd,
              mean: mean[j] ?? Number.NaN,
              sd: sd[j] ?? Number.NaN,
              exactMean: exact === null || j >= p ? null : (exact.mean[j] ?? null),
              exactSd: exact === null || j >= p ? null : (exact.sd[j] ?? null),
            })),
            noiseSd: problem.noiseSd,
          } as never
        },
      }),
      'linreg/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of the intercept, the slope, the next coefficients and the noise (a posteriorPairs record) with the true values marked',
        compute: (state) => {
          const got = namedDraws(post(state), problem, PAIR_DRAWS)
          if (got === undefined || got.rows.length < 3) return null
          const coordinates = got.names.slice(0, 5)
          const columns = Object.fromEntries(coordinates.map((c, j) => [c, got.rows.map((r) => r[j] ?? 0)]))
          const truth = Object.fromEntries(
            coordinates.map((c, j) => [c, j < got.p ? (problem.truth[j] ?? 0) : problem.noiseSd]),
          )
          return posteriorPairs([{ name: kind, draws: columns, steps: got.index, chain: got.chain }], {
            coordinates,
            maxPoints: 300,
            frames: 2,
            densityDraws: 300,
            truth,
          }) as never
        },
      }),
      'linreg/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler (a chainDiagnostics record); null for the others',
        compute: (state) => {
          const rename = (name: string): string => {
            const j = /^w\[(\d+)\]$/.exec(name)?.[1]
            return j === undefined ? name : (problem.names[Number(j)] ?? name)
          }
          const post0 = post(state)
          return exampleDiagnostics(post0, post0.sites.includes('sigma') ? ['w', 'sigma'] : ['w'], rename) as never
        },
      }),
      'linreg/draws': derived<unknown>({
        cost: 'dear',
        description:
          'the draws of the coefficients and the noise, one column each (what a check against a reference run needs of the reference)',
        compute: (state) => {
          const got = namedDraws(post(state), problem, 400)
          if (got === undefined) return null
          return {
            names: got.names,
            columns: got.names.map((_, j) => got.rows.map((r) => r[j] ?? 0)),
          } as never
        },
      }),
      'linreg/check': derived<unknown>({
        cost: 'dear',
        description:
          "the method against the exact posterior of the coefficients (noise known) or against reference draws (args `{ reference }`, the reference's `linreg/draws`): a comparison record and the distances for the coefficients",
        compute: (state, args) => {
          const ref = (args as { reference?: LinearReference | null } | null | undefined)?.reference
          const method = post(state)
          if (!method.sites.includes('w')) return null
          const exact = problem.learned ? undefined : exactOf(problem, n)
          if (exact === undefined && (ref === undefined || ref === null)) return null
          const approx = fromPosterior(kind, method, { sites: method.sites.includes('sigma') ? ['w', 'sigma'] : ['w'] })
          const rename = (q: string): string => {
            const j = /^w\[(\d+)\]$/.exec(q)?.[1]
            return j === undefined ? q : (problem.names[Number(j)] ?? q)
          }
          const draws = Object.fromEntries(Object.entries(approx.draws).map(([q, v]) => [rename(q), v]))
          const named = {
            ...approx,
            draws,
            ...(approx.ess === undefined
              ? {}
              : { ess: Object.fromEntries(Object.entries(approx.ess).map(([q, v]) => [rename(q), v])) }),
          }
          const oracle: Record<string, MarginalOracle> = {}
          if (exact !== undefined)
            problem.names.forEach((name, j) => {
              oracle[name] = distributionOracle(name, normalDist(exact.mean[j] ?? 0, exact.sd[j] ?? 1))
            })
          else if (ref !== undefined && ref !== null)
            ref.names.forEach((name, j) => {
              const column = ref.columns[j]
              if (column !== undefined && name in draws) oracle[name] = weightedOracle(name, column)
            })
          const quantities = Object.keys(oracle)
          if (quantities.length === 0) return null
          const table = agree([named], { oracle, quantities })
          const rows = table.rows
          return {
            comparison: posteriorComparison([named], {
              truth: oracle,
              table,
              quantities,
              gridPoints: 100,
              oracle: exact === undefined ? 'a long NUTS run' : 'the exact normal posterior',
            }),
            rows,
            maxMeanError: Math.max(...rows.map((r) => Math.abs(r.meanError))),
            medianSdRatio:
              [...rows.map((r) => r.sdRatio)].sort((a, b) => a - b)[Math.floor(rows.length / 2)] ?? Number.NaN,
            maxWasserstein: Math.max(...rows.map((r) => r.wasserstein)),
            maxKl: Math.max(...rows.map((r) => r.kl)),
            against: exact === undefined ? 'reference' : 'exact',
            logEvidence: null,
          } as never
        },
      }),
    }
  },
})

/**
 * Bayesian linear regression as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape is the
 * chosen inference method's own, with the derived paths `linreg/fit`, `linreg/pairs`, `linreg/draws` and `linreg/check`.
 *
 * @concept inference-engines
 * @index estimator Bayesian linear regression method: polynomial data, known or learned noise, one inference method.
 * @example
 * ```ts
 * import { linearRegressionMethod, linearRegressionSteps } from 'omai-engine/bayesian-examples'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(linearRegressionMethod.config), rows: 20, inference: { method: 'gaussian' } }
 * const s = session(linearRegressionMethod, { config, steps: linearRegressionSteps(config), mode: 'eager' })
 * const check = s.probe(s.t, 'linreg/check') as { maxMeanError: number }
 * check.maxMeanError < 0.01 // => true
 * ```
 */
export const linearRegressionMethod: InferenceMethod<unknown, Config, string> = linearExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const linearRegressionSteps: (config: Config) => number = linearExample.steps

export { linearExample }
