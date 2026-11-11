/**
 * Bayesian linear regression as an example method: data simulated with known coefficients (a polynomial of one input, or a
 * sparse linear model of up to twenty features with only some coefficients non-zero; the number of rows, the noise, how
 * the inputs are placed, a seed), a model with a prior on the coefficients (a normal, a spike and slab that switches each
 * coefficient on or off, ARD with a precision of its own for each coefficient, or a horseshoe) and the noise either known
 * or learned, and one inference method of the library with its own knobs and a budget. With a normal prior and the noise
 * known the posterior of the coefficients is a normal in closed form (`bayesianLinearFit`), the exact answer every method
 * is checked against; with anything else there is none, and the check is against a long NUTS run of the same problem (the
 * page runs it in a worker and hands its draws to `linreg/check`). The derived paths read the run: `linreg/fit` (the fitted
 * curve with its band and sampled lines after the first `seen` rows, the method refitted on them, the coefficients against
 * the truth), `linreg/coefficients` (each coefficient against its true value with its posterior inclusion probability or
 * relevance, the selection against the true non-zero set, and the same run with the sparsity switched off as a burn-in),
 * `linreg/pairs` (the pair plot of the coefficients and the noise), `linreg/draws` (the draws, as the reference's answer)
 * and `linreg/check` (the method against the exact posterior or the reference draws, as a comparison record with the
 * distances).
 *
 * @index estimator Bayesian linear regression with normal, spike-and-slab, ARD or horseshoe priors, one inference method.
 */

import type { InferenceMethod } from '../probprog/inference/method.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops, type Var } from '../probprog/model/index.ts'
import type { Posterior } from '../probprog/posterior.ts'
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
  exampleNumber,
  exampleString,
} from './example-method.ts'
import {
  curveBands,
  exactNormalPosterior,
  monomials,
  noiseSdOf,
  polynomialAt,
  type RegressionReference,
  type RegressionShape,
  regressionCheck,
  regressionDiagnostics,
  regressionDraws,
  regressionPairs,
  regressionReference,
} from './regression-views.ts'

/** The functions the data are drawn from: the coefficients of `1, x, x^2, x^3`; `sparse` is a linear model of many features. */
const TRUTH: Readonly<Record<string, readonly number[]>> = {
  line: [1, 2],
  quadratic: [1, 2, -1.5],
  cubic: [0.5, 2, -1.5, 1.2],
}
/** The truths on offer. */
export const linearTruths: readonly string[] = [...Object.keys(TRUTH), 'sparse']
/** The non-zero coefficients of the sparse truth, in order (the first `active` of them, cycled). */
const SPARSE_VALUES: readonly number[] = [2, -1.5, 1.2, -2, 1.6, -1, 2.4]
/** Where the inputs lie: around zero (the coefficients are nearly uncorrelated) or away from it (intercept and slope trade off). */
const RANGE: Readonly<Record<string, readonly [number, number]>> = { centred: [-1, 1], offset: [1, 3] }
/** The placements of the inputs on offer. */
export const linearInputs: readonly string[] = Object.keys(RANGE)
/** The priors on the coefficients on offer. */
export const linearPriors: readonly string[] = ['gaussian', 'spikeSlab', 'ard', 'horseshoe']

/** The settings of the sparse priors. */
export interface LinearPriorSettings {
  /** The prior probability that a coefficient is in the slab (spike and slab). */
  readonly inclusion: number
  /** The standard deviation of the spike, as a fraction of the slab's (spike and slab). */
  readonly spikeRatio: number
  /** How much of the precision is the relevance's, from 0 (a plain normal prior) to 1 (ARD alone), for ARD. */
  readonly sparsity: number
  /** The rate of the gamma prior on each coefficient's precision (ARD). */
  readonly ardRate: number
}

/** The problem a linear regression configuration stands for. */
export interface LinearProblem extends ExampleProblem {
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** The coefficients of the generating model (padded with zeros to the fitted degree). */
  readonly truth: readonly number[]
  readonly noiseSd: number
  /** Whether the noise is inferred (half-normal sd, or a gamma precision). */
  readonly learned: boolean
  /** Whether the noise is written as a gamma-distributed precision (`tau`, the textbook form message passing accepts). */
  readonly precision: boolean
  readonly degree: number
  /** The names of the coefficients: `intercept`, `slope`, `x^2`, ... (or `x1`, `x2`, ... for the sparse truth). */
  readonly names: readonly string[]
  readonly extent: readonly [number, number]
  readonly priorScale: number
  /** The design rows `[n][p]` of the data. */
  readonly design: readonly (readonly number[])[]
  /** Whether the truth is the sparse linear model (features rather than powers of one input). */
  readonly sparse: boolean
  /** The prior on the coefficients: `gaussian`, `spikeSlab`, `ard` or `horseshoe`. */
  readonly prior: string
  readonly settings: LinearPriorSettings
  /** The sum of squares of each design column (what the data say about a coefficient, in units of the noise variance). */
  readonly norms: readonly number[]
}

/**
 * The problem a configuration stands for: `rows` inputs placed in the range (or a design of independent standard normal
 * features for the sparse truth), the outputs of the generating model with the known coefficients plus noise (the first
 * `n` rows of a longer run are the same, so a prefix is the data seen so far), the design of the fitted degree and the
 * model: a prior on every coefficient (normal, spike and slab, ARD or horseshoe) and the noise either fixed at its true
 * value (known) or given a half-normal prior (learned).
 *
 * @param config - The knobs (`truth`, `inputs`, `rows`, `noise`, `dataSeed`, `features`, `active`, `degree`, `priorScale`, `noiseModel`, `sigmaScale`, `prior` and the sparse priors' own).
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
  const sparse = exampleString(config, 'truth') === 'sparse'
  const features = Math.max(1, Math.round(exampleNumber(config, 'features') || 10))
  const degree = sparse ? features - 1 : Math.max(1, Math.round(exampleNumber(config, 'degree')))
  const range = RANGE[exampleString(config, 'inputs')] ?? [-1, 1]
  const generating = sparse
    ? Array.from({ length: features }, (_, j) =>
        j < Math.round(exampleNumber(config, 'active')) ? (SPARSE_VALUES[j % SPARSE_VALUES.length] ?? 1) : 0,
      )
    : (TRUTH[exampleString(config, 'truth')] ?? [1, 2])
  const noiseSd = exampleNumber(config, 'noise')
  const priorScale = exampleNumber(config, 'priorScale')
  const noiseModel = exampleString(config, 'noiseModel')
  const gammaNoise = noiseModel === 'precision'
  const learned = noiseModel === 'learned' || gammaNoise
  const sigmaScale = exampleNumber(config, 'sigmaScale')
  const noiseShape = exampleNumber(config, 'noiseShape') || 2
  const noiseRate = exampleNumber(config, 'noiseRate') || 1
  const prior = linearPriors.includes(exampleString(config, 'prior')) ? exampleString(config, 'prior') : 'gaussian'
  const settings: LinearPriorSettings = {
    inclusion: exampleNumber(config, 'inclusion') || 0.5,
    spikeRatio: exampleNumber(config, 'spikeRatio') || 0.03,
    sparsity: 'sparsity' in config ? exampleNumber(config, 'sparsity') : 1,
    ardRate: exampleNumber(config, 'ardRate') || 0.001,
  }
  const seed = foldIn(key(exampleNumber(config, 'dataSeed')), 0)
  const e = toFlat(drawNormal(foldIn(seed, 1), [n]))
  let x: number[]
  let design: number[][]
  if (sparse) {
    const flat = toFlat(drawNormal(foldIn(seed, 2), [n * features]))
    design = Array.from({ length: n }, (_, i) =>
      Array.from({ length: features }, (_, j) => flat[i * features + j] ?? 0),
    )
    x = design.map((r) => r[0] ?? 0)
  } else {
    const u = toFlat(drawUniform(foldIn(seed, 0), [n]))
    x = Array.from(u, (v) => range[0] + (range[1] - range[0]) * v)
    design = x.map((v) => monomials(v, degree))
  }
  const truth = Array.from({ length: degree + 1 }, (_, j) => generating[j] ?? 0)
  const y = design.map((row, i) => row.reduce((a, v, j) => a + v * (truth[j] ?? 0), 0) + noiseSd * (e[i] ?? 0))
  const definition = defineModel('linear-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const X = m.data('X', { dims: [row, feature], role: 'covariate' })
    const yy = m.data('y', { dims: [row], role: 'response' })
    let w: Var
    if (prior === 'spikeSlab') {
      w = m.latent('w', d.normalSpikeSlab(settings.inclusion, priorScale, priorScale * settings.spikeRatio), {
        dims: [feature],
        doc: 'the coefficients: each in the slab (used) or the spike (off)',
      })
    } else if (prior === 'ard') {
      // Centred: the data say more about each coefficient than the prior does, so the coefficient is drawn given its own sd.
      const alpha = m.latent('alpha', d.gamma(1, settings.ardRate), {
        dims: [feature],
        doc: 'the precision of each coefficient (relevance): a large one prunes it',
      })
      if (settings.sparsity === 1) {
        // The precision idiom message passing accepts: each coefficient is a normal with scale 1 / sqrt(alpha_j).
        m.derived('precision', alpha, { doc: 'the prior precision of each coefficient' })
        w = m.latent('w', d.normal(0, ops.div(1, ops.sqrt(alpha))), { dims: [feature], doc: 'the coefficients' })
      } else {
        const precision = m.derived(
          'precision',
          alpha.mul(settings.sparsity).add((1 - settings.sparsity) / priorScale ** 2),
          { doc: 'the prior precision of each coefficient' },
        )
        w = m.latent('w', d.normal(0, precision.sqrt().pow(-1)), { dims: [feature], doc: 'the coefficients' })
      }
    } else if (prior === 'horseshoe') {
      const lambda = m.latent('lambda', d.halfCauchy(1), {
        dims: [feature],
        doc: 'the local scale of each coefficient',
      })
      const global = m.latent('globalScale', d.halfCauchy(priorScale), { doc: 'the global scale' })
      m.derived('precision', lambda.mul(global).square().pow(-1))
      w = m.latent('w', d.normal(0, lambda.mul(global)), { dims: [feature], doc: 'the coefficients' })
    } else w = m.latent('w', d.normal(0, priorScale), { dims: [feature], doc: 'the coefficient of each column' })
    if (gammaNoise) {
      const tau = m.latent('tau', d.gamma(noiseShape, noiseRate), { doc: 'the noise precision, one over its variance' })
      m.observe('y', d.normal(X.matmul(w), ops.div(1, ops.sqrt(tau))), yy)
    } else if (learned) {
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
    precision: gammaNoise,
    degree,
    names: sparse
      ? Array.from({ length: features }, (_, j) => `x${j + 1}`)
      : ['intercept', 'slope', ...Array.from({ length: Math.max(0, degree - 1) }, (_, j) => `x^${j + 2}`)],
    extent: range as readonly [number, number],
    priorScale,
    design,
    sparse,
    prior,
    settings,
    norms: truth.map((_, j) => design.reduce((a, r) => a + (r[j] ?? 0) ** 2, 0)),
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
      sparse:
        'a linear model of many independent features of which only the first few coefficients are non-zero (the sparse truth that spike and slab, ARD and the horseshoe are for)',
    },
  }),
  inputs: choice(['centred', 'offset'], {
    default: 'centred',
    label: 'inputs',
    descriptions: {
      centred: 'x in [-1, 1]: the intercept and the slope are nearly uncorrelated in the posterior',
      offset: 'x in [1, 3], away from zero: intercept and slope trade off and are strongly correlated',
    },
  }),
  features: int(2, 20, {
    default: 10,
    label: 'features',
    description: 'the number of features (columns) of the sparse truth',
  }),
  active: int(0, 20, {
    default: 3,
    label: 'non-zero coefficients',
    description: 'how many of the coefficients of the sparse truth are not zero (the first ones)',
  }),
  rows: int(2, 400, { default: 30, log: true, label: 'rows' }),
  noise: real(0.02, 3, { default: 0.5, log: true, label: 'noise sd' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the degree, the prior and its scale, and whether the noise is known or learned. */
export const linearModelKnobs: DictSpace = dict({
  degree: int(1, 6, {
    default: 1,
    label: 'polynomial degree',
    description: 'the model is a polynomial of this degree',
  }),
  prior: choice(linearPriors, {
    default: 'gaussian',
    label: 'prior on the coefficients',
    descriptions: {
      gaussian: 'every coefficient has the same normal prior: the exact posterior is a normal (noise known)',
      spikeSlab:
        'each coefficient is either in a wide slab (used) or in a narrow spike at zero (off): the posterior inclusion probability says which',
      ard: 'automatic relevance determination: each coefficient has a precision of its own with a gamma prior, and a large one switches the coefficient off',
      horseshoe:
        'a heavy-tailed shrinkage prior: a global scale and a local scale per coefficient, both half-Cauchy; large coefficients escape, small ones are pulled to zero',
    },
  }),
  priorScale: real(0.1, 30, {
    default: 3,
    log: true,
    label: 'prior scale',
    description:
      'standard deviation of the normal prior on each coefficient, the slab of spike and slab, the global scale of the horseshoe, the plain part of ARD',
  }),
  inclusion: real(0.01, 0.99, {
    default: 0.5,
    label: 'prior inclusion probability',
    description: 'the prior probability that a coefficient is in the slab',
  }),
  spikeRatio: real(0.005, 0.5, {
    default: 0.03,
    log: true,
    label: 'spike width',
    description: 'the standard deviation of the spike as a fraction of the slab',
  }),
  sparsity: real(0, 1, {
    default: 1,
    label: 'ARD sparsity',
    description:
      "how much of each coefficient's precision is its own relevance: 0 is a plain normal prior (a smooth fit with every coefficient), 1 is ARD alone; raise it from 0 to see coefficients switch off (a burn-in by hand)",
  }),
  ardRate: real(0.0001, 10, {
    default: 0.001,
    log: true,
    label: 'ARD precision rate',
    description:
      'the gamma prior of each precision has shape 1 and this rate (mean 1 / rate): a small rate allows precisions large enough to switch a coefficient off; a larger one prunes less',
  }),
  noiseModel: choice(['known', 'learned', 'precision'], {
    default: 'known',
    label: 'noise',
    descriptions: {
      known: 'the noise sd is known (its true value)',
      learned: 'the noise sd is unknown, with a half-normal prior: no closed form, the reference is a long NUTS run',
      precision:
        'the noise precision (1 / variance) is unknown, with a gamma prior (the textbook form): message passing (EP, VMP) applies; no closed form for the coefficients, the reference is a long NUTS run',
    },
  }),
  noiseShape: real(0.1, 20, {
    default: 2,
    log: true,
    label: 'precision prior shape',
    description: 'shape of the gamma prior on the noise precision',
  }),
  noiseRate: real(0.01, 20, {
    default: 1,
    log: true,
    label: 'precision prior rate',
    description: 'rate of the gamma prior on the noise precision (its mean is shape / rate)',
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
  'features',
  'active',
  'rows',
  'noise',
  'dataSeed',
  'degree',
  'prior',
  'priorScale',
  'inclusion',
  'spikeRatio',
  'sparsity',
  'ardRate',
  'noiseModel',
  'sigmaScale',
  'noiseShape',
  'noiseRate',
]

/** The knobs a configuration does not use (the sparse truth's, the noise prior of a known noise, the priors the choice leaves out). */
export function linearHiddenKnobs(config: Config): readonly string[] {
  const sparse = exampleString(config, 'truth') === 'sparse'
  const prior = exampleString(config, 'prior')
  return [
    ...(sparse ? ['inputs', 'degree'] : ['features', 'active']),
    ...(exampleString(config, 'noiseModel') === 'learned' ? [] : ['sigmaScale']),
    ...(exampleString(config, 'noiseModel') === 'precision' ? [] : ['noiseShape', 'noiseRate']),
    ...(prior === 'spikeSlab' ? [] : ['inclusion', 'spikeRatio']),
    ...(prior === 'ard' ? [] : ['sparsity', 'ardRate']),
  ]
}

const GRID = 60
const FIT_DRAWS = 300
const LINES = 12
const PAIR_DRAWS = 500
const COEFFICIENT_DRAWS = 400

/** The exact posterior of the coefficients (a normal prior, the noise known) after the first `rows` rows. */
function exactOf(problem: LinearProblem, rows: number): ReturnType<typeof exactNormalPosterior> {
  return exactNormalPosterior(
    problem.design.slice(0, rows),
    problem.y.slice(0, rows),
    problem.priorScale,
    problem.noiseSd,
  )
}

/** How the posteriors of a problem are read: the coefficients and the noise. */
function shapeOf(problem: LinearProblem): RegressionShape {
  return {
    names: problem.names,
    truth: problem.truth,
    scalars: [
      { site: 'sigma', name: 'sigma', truth: problem.noiseSd },
      { site: 'tau', name: 'tau', truth: 1 / problem.noiseSd ** 2 },
    ],
  }
}

/** A normal density at `x` with mean 0 and standard deviation `sd`, up to the constant shared by every `sd`. */
const normalDensity = (x: number, sd: number): number => Math.exp(-0.5 * (x / sd) ** 2) / sd

/** What the page shows of each coefficient: its posterior, the truth, and the score of the sparse prior. */
export interface CoefficientRecord {
  readonly names: readonly string[]
  readonly truth: readonly number[]
  readonly mean: readonly number[]
  readonly sd: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  /** The posterior inclusion probability (spike and slab) or relevance (ARD, horseshoe) of each coefficient; null for a normal prior. */
  readonly score: readonly number[] | null
  readonly scoreLabel: string | null
  /** Which coefficients are truly not zero. */
  readonly active: readonly boolean[]
  /** The selection: the coefficients scoring at least 1/2 against the true non-zero set. */
  readonly selection: {
    readonly chosen: number
    readonly trulyActive: number
    readonly found: number
    readonly falseAlarms: number
  } | null
  /** The posterior means of the same run with the sparsity switched off (the burn-in fit), for ARD; else null. */
  readonly burnin: readonly number[] | null
}

const linearExample: Example<LinearProblem> = defineExample<LinearProblem>({
  name: 'linreg',
  // The sparse priors sample a hierarchy (a precision per coefficient): their first draws take a few seconds.
  budgetMs: 8000,
  description:
    'Bayesian linear regression on polynomial or sparse data with one inference method checked against the exact or a reference posterior',
  modelName: 'linreg',
  dataKnobs: linearDataKnobs,
  modelKnobs: linearModelKnobs,
  problemKnobs: linearProblemKnobs,
  problem: linearRegressionProblem,
  derived: ({ problem, config, kind, posterior: post, fit }) => {
    const n = problem.x.length
    const shape = shapeOf(problem)
    const [lo, hi] = problem.extent
    const pad = 0.08 * (hi - lo)
    const grid = Array.from({ length: GRID }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / (GRID - 1))
    const truthCurve = grid.map((v) => polynomialAt(problem.truth, v))
    const exactApplies = !problem.learned && problem.prior === 'gaussian'
    return {
      'linreg/fit': derived<unknown>({
        cost: 'dear',
        description:
          'the fitted curve after the first `seen` rows (args `{ seen }`): the method refitted on them, its mean curve with the 95% band of the curve and of a new observation, sampled lines, the data, the true curve, and the coefficients against the truth (and against the exact posterior when there is one); null for the sparse truth',
        compute: (state, args) => {
          if (problem.sparse) return null
          const asked = Math.round(Number((args as { seen?: number } | null | undefined)?.seen ?? n))
          const seen = Math.max(2, Math.min(n, asked))
          let method: Posterior
          try {
            method = seen >= n ? post(state) : fit({ ...config, rows: seen })
          } catch {
            return null
          }
          const got = regressionDraws(method, shape, FIT_DRAWS)
          if (got === undefined) return null
          const { rows, p } = got
          const z = toFlat(drawNormal(key(7), [rows.length]))
          const bands = curveBands(
            rows,
            p,
            grid,
            (r, _s, zz) => noiseSdOf(got.names, r, problem.noiseSd) * zz,
            LINES,
            z,
          )
          const { mean, sd } = columnMoments(rows, got.names.length)
          const exact = exactApplies ? exactOf(problem, seen) : null
          return {
            seen,
            total: n,
            x: grid,
            ...bands,
            truth: truthCurve,
            data: { x: problem.x.slice(0, seen), y: problem.y.slice(0, seen) },
            coefficients: got.names.map((name, j) => ({
              name,
              truth: got.truth[j] ?? null,
              mean: mean[j] ?? Number.NaN,
              sd: sd[j] ?? Number.NaN,
              exactMean: exact === null || j >= p ? null : (exact.mean[j] ?? null),
              exactSd: exact === null || j >= p ? null : (exact.sd[j] ?? null),
            })),
            noiseSd: problem.noiseSd,
          } as never
        },
      }),
      'linreg/coefficients': derived<unknown>({
        cost: 'dear',
        description:
          'each coefficient against its true value: posterior mean, sd and 95% interval, the posterior inclusion probability (spike and slab) or relevance (ARD, horseshoe: how much of the coefficient the data determine, 1 minus its shrinkage), the selection against the true non-zero set; with args `{ burnin: true }` also the posterior means of the same run with the ARD sparsity switched off',
        compute: (state, args) => {
          const method = post(state)
          if (!method.sites.includes('w')) return null
          const p = problem.names.length
          const noiseSite = method.sites.includes('sigma') ? 'sigma' : method.sites.includes('tau') ? 'tau' : undefined
          const learned = noiseSite !== undefined
          // The prior precision of each coefficient: its own site, or the ARD precision alpha when the sparsity is full.
          const precisionSite = method.sites.includes('precision')
            ? 'precision'
            : problem.prior === 'ard' && method.sites.includes('alpha')
              ? 'alpha'
              : undefined
          const hasPrecision = precisionSite !== undefined
          const sites = [
            'w',
            ...(noiseSite === undefined ? [] : [noiseSite]),
            ...(precisionSite === undefined ? [] : [precisionSite]),
          ]
          const got = drawRows(method, sites, COEFFICIENT_DRAWS)
          if (got === undefined) return null
          const ws = got.rows.map((r) => r.slice(0, p))
          const { mean, sd } = columnMoments(ws, p)
          const sorted = Array.from({ length: p }, (_, j) => ws.map((r) => r[j] ?? 0).sort((a, b) => a - b))
          const sigmaAt = (r: readonly number[]): number =>
            noiseSite === 'sigma' ? (r[p] ?? 1) : noiseSite === 'tau' ? 1 / Math.sqrt(r[p] ?? 1) : problem.noiseSd
          let score: number[] | null = null
          let scoreLabel: string | null = null
          const { inclusion, spikeRatio } = problem.settings
          if (problem.prior === 'spikeSlab') {
            const slab = problem.priorScale
            const spike = slab * spikeRatio
            score = Array.from({ length: p }, (_, j) => {
              let a = 0
              for (const r of ws) {
                const s1 = inclusion * normalDensity(r[j] ?? 0, slab)
                const s0 = (1 - inclusion) * normalDensity(r[j] ?? 0, spike)
                a += s1 / (s1 + s0)
              }
              return a / ws.length
            })
            scoreLabel = 'posterior inclusion probability'
          } else if ((problem.prior === 'ard' || problem.prior === 'horseshoe') && hasPrecision) {
            const offset = p + (learned ? 1 : 0)
            // The typical (median) shrinkage over the draws: a coefficient that is nearly always pruned scores near 0.
            score = Array.from({ length: p }, (_, j) => {
              const gammas = got.rows
                .map((r) => {
                  const data = (problem.norms[j] ?? 0) / sigmaAt(r) ** 2
                  return data / (data + (r[offset + j] ?? 0))
                })
                .sort((a, b) => a - b)
              return gammas[Math.floor((gammas.length - 1) / 2)] ?? Number.NaN
            })
            scoreLabel = 'relevance'
          }
          const active = problem.truth.map((v) => v !== 0)
          const chosen = score === null ? [] : score.map((s, j) => (s >= 0.5 ? j : -1)).filter((j) => j >= 0)
          const wantBurnin = Boolean((args as { burnin?: boolean } | null | undefined)?.burnin)
          let burnin: number[] | null = null
          if (wantBurnin && problem.prior === 'ard' && problem.settings.sparsity > 0) {
            try {
              const base = regressionDraws(fit({ ...config, sparsity: 0 }), shapeOf(problem), COEFFICIENT_DRAWS)
              burnin = base === undefined ? null : columnMoments(base.rows, p).mean
            } catch {
              burnin = null
            }
          }
          return {
            names: problem.names,
            truth: problem.truth,
            mean,
            sd,
            lo: sorted.map((c) => c[Math.floor(0.025 * (c.length - 1))] ?? Number.NaN),
            hi: sorted.map((c) => c[Math.ceil(0.975 * (c.length - 1))] ?? Number.NaN),
            score,
            scoreLabel,
            active,
            selection:
              score === null
                ? null
                : {
                    chosen: chosen.length,
                    trulyActive: active.filter(Boolean).length,
                    found: chosen.filter((j) => active[j]).length,
                    falseAlarms: chosen.filter((j) => !active[j]).length,
                  },
            burnin,
          } satisfies CoefficientRecord as never
        },
      }),
      'linreg/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of the first five coefficients and the noise (a posteriorPairs record) with the true values marked',
        compute: (state) => {
          const got = regressionDraws(post(state), shape, PAIR_DRAWS)
          if (got === undefined || got.rows.length < 3) return null
          return regressionPairs(kind, got) as never
        },
      }),
      'linreg/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler (a chainDiagnostics record); null for the others',
        compute: (state) => regressionDiagnostics(post(state), shape) as never,
      }),
      'linreg/draws': derived<unknown>({
        cost: 'dear',
        description:
          'the draws of the coefficients and the noise, one column each, and the largest R-hat (what a check against a reference run needs of the reference)',
        compute: (state) => (regressionReference(post(state), shape) ?? null) as never,
      }),
      'linreg/check': derived<unknown>({
        cost: 'dear',
        description:
          "the method against the exact posterior of the coefficients (normal prior, noise known) or against reference draws (args `{ reference }`, the reference's `linreg/draws`): a comparison record and the distances for the coefficients",
        compute: (state, args) => {
          const ref = (args as { reference?: RegressionReference | null } | null | undefined)?.reference
          return regressionCheck(kind, post(state), shape, exactApplies ? exactOf(problem, n) : undefined, ref) as never
        },
      }),
    }
  },
})

/**
 * Bayesian linear regression as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape is the
 * chosen inference method's own, with the derived paths `linreg/fit`, `linreg/coefficients`, `linreg/pairs`,
 * `linreg/draws` and `linreg/check`.
 *
 * @concept inference-engines
 * @index estimator Linear regression method: polynomial or sparse data, sparse priors, one inference method.
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

/** What a check against a reference needs of the reference (the draws of `linreg/draws`). */
export type LinearReference = RegressionReference

export { linearExample }
