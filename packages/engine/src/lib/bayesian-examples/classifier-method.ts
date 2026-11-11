/**
 * Bayesian classification of a two-dimensional data set as one inference method that a worker or a page can run by name
 * and a few knobs: the data set and its size and noise, the link (probit or logistic), the degree of the polynomial
 * features, the scale of the prior, and which inference method (with its own knobs) fits it. Everything a model-bound
 * method closes over (`nutsMethod(bound)`) is rebuilt here from the knobs, so the configuration is all that crosses a
 * Worker boundary. Beside the tape of the chosen method the contract has derived paths that read the posterior at any
 * step: the decision surface (`classify/surface`), the pair plot of the coefficients (`classify/pairs`), the scores and
 * summary numbers (`classify/scores`), the chain diagnostics of a sampler (`classify/diagnostics`) and the method checked against
 * a reference run of the same problem, itself a session of this method with a long NUTS run (`classify/check`).
 *
 * @index estimator Bayesian probit or logistic classifier on 2-D data: one chosen inference method with its knobs.
 */

import { type ClassificationData, generate, trainTestSplit } from '../data/index.ts'
import { blobs, circles, moons, spirals, xor } from '../datasets/index.ts'
import { chainDiagnostics, decisionSurface, posteriorPairs } from '../exposition/index.ts'
import { anytimeKnobs, defineInferenceMethod, type InferenceMethod } from '../probprog/inference/method.ts'
import { type AnyInferenceMethod, inferenceMethodFor, modelRegistry } from '../probprog/inference/models.ts'
import * as d from '../probprog/model/dists/index.ts'
import { type DataInput, defineModel, type ModelDefinition, ops } from '../probprog/model/index.ts'
import {
  heldOutScores,
  type PolynomialFeatureMap,
  polynomialFeatureMap,
  probitData,
  probitPredictive,
  probitPredictiveGrid,
} from '../probprog/models/probit-classification.ts'
import type { Posterior } from '../probprog/posterior.ts'
import { posteriorChains } from '../probprog/posterior-chains.ts'
import { foldIn, type Key, key } from '../random/index.ts'
import { choice, type DictSpace, defaults, dict, int, type RealSpace, real, union } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Config, type Contract, derived } from '../trace/index.ts'

/** The two-dimensional data sets on offer. */
export const classifierDatasets = ['moons', 'circles', 'xor', 'spirals', 'blobs'] as const
/** The links on offer. */
export const classifierLinks = ['probit', 'logistic'] as const
/** A data set of {@link classifierDatasets}. */
export type ClassifierDataset = (typeof classifierDatasets)[number]

// The generator's own noise at relative noise 0 and 1, so one knob in [0, 1] means a little and a lot for every set.
const NOISE: Readonly<Record<ClassifierDataset, readonly [number, number]>> = {
  moons: [0.02, 0.8],
  circles: [0.01, 0.5],
  xor: [0.05, 1],
  spirals: [0.01, 0.3],
  blobs: [0.1, 3],
}

/** The inference methods on offer: what each is, the knobs the page shows and the tape path of its objective. */
interface KindSpec {
  /** A name of `methodKinds`. */
  readonly kind: string
  readonly description: string
  /** The method's own knobs shown, with the defaults chosen here (short runs: an answer in a second or two). */
  readonly shown: Readonly<Record<string, number | string | boolean>>
  /** Whether the method's run has a length of its own to set (`steps`), as variational inference has. */
  readonly steps?: number
  /** The tape path of the method's objective curve and what it is. */
  readonly objective?: { readonly path: string; readonly label: string }
  /** Knobs of the method fixed for every run. */
  readonly fixed?: Readonly<Record<string, number | string | boolean>>
}

const KINDS: Readonly<Record<string, KindSpec>> = {
  nuts: {
    kind: 'nuts',
    description: 'the no-U-turn sampler: draws from the posterior in chains',
    shown: { numChains: 2, numWarmup: 60, numDraws: 120 },
    objective: { path: 'mcmc/logDensity', label: 'log density of the chains' },
  },
  hmc: {
    kind: 'hmc',
    description: 'Hamiltonian Monte Carlo with a fixed number of leapfrog steps',
    shown: { numChains: 2, numWarmup: 60, numDraws: 120, numSteps: 10 },
    objective: { path: 'mcmc/logDensity', label: 'log density of the chains' },
  },
  mala: {
    kind: 'mala',
    description: 'Metropolis-adjusted Langevin: a gradient-guided random walk',
    shown: { numChains: 2, numWarmup: 100, numDraws: 250 },
    objective: { path: 'mcmc/logDensity', label: 'log density of the chains' },
  },
  svi: {
    kind: 'svi',
    description: 'stochastic variational inference: a normal fitted by following the ELBO',
    shown: { guide: 'autoNormal', learningRate: 0.05 },
    steps: 600,
    objective: { path: 'svi/elbo', label: 'ELBO' },
  },
  laplace: {
    kind: 'laplace',
    description: 'a normal at the mode with the curvature there',
    shown: {},
    objective: { path: 'laplace/logDensity', label: 'log density at the iterate' },
  },
  ep: {
    kind: 'ep',
    description: 'expectation propagation: one normal message per observation, swept to a fixed point',
    shown: { damping: 0 },
    fixed: { tol: 1e-6 },
  },
  pathfinder: {
    kind: 'pathfinder',
    description: 'a normal along the L-BFGS path with the best ELBO',
    shown: { numDraws: 400 },
  },
}

/** The inference methods of {@link classifierMethod}, in the order offered. */
export const classifierKinds: readonly string[] = Object.keys(KINDS)

/** The problem a configuration stands for: the data, the features and the model. */
export interface ClassifierProblem {
  readonly train: { readonly x: readonly (readonly number[])[]; readonly y: readonly number[] }
  readonly test: { readonly x: readonly (readonly number[])[]; readonly y: readonly number[] }
  /** The rectangle of the plane the decision surface covers. */
  readonly extent: { readonly x: readonly [number, number]; readonly y: readonly [number, number] }
  readonly map: PolynomialFeatureMap
  readonly design: readonly (readonly number[])[]
  readonly heldOut: readonly (readonly number[])[]
  readonly definition: ModelDefinition
  readonly data: DataInput
  /** The link, to compute predictive probabilities with. */
  readonly link: 'probit' | 'logistic'
}

const num = (config: Config, name: string): number => {
  const v = config[name]
  return typeof v === 'number' ? v : 0
}
const str = (config: Config, name: string): string => {
  const v = config[name]
  return typeof v === 'string' ? v : ''
}

function generator(
  id: ClassifierDataset,
  rows: number,
  noise: number,
  seed: Key,
): { readonly data: ClassificationData } {
  const [lo, hi] = NOISE[id]
  const sd = lo + (hi - lo) * noise
  const at = (knobs: Record<string, number>): { knobs: Record<string, number>; key: Key } => ({ knobs, key: seed })
  switch (id) {
    case 'moons':
      return generate(moons, at({ n: rows, noise: sd }))
    case 'circles':
      return generate(circles, at({ n: rows, noise: sd }))
    case 'xor':
      return generate(xor, at({ n: rows, noise: sd }))
    case 'spirals':
      return generate(spirals, at({ n: rows, noise: sd }))
    case 'blobs':
      return generate(blobs, at({ n: rows, nFeatures: 2, centers: 2, clusterStd: sd, boxSize: 2 }))
  }
}

const pointsOf = (data: ClassificationData): { x: number[][]; y: number[] } => {
  const x = toFlat(data.x)
  const y = toFlat(data.y)
  return {
    x: Array.from({ length: y.length }, (_, i) => [x[2 * i] ?? 0, x[2 * i + 1] ?? 0]),
    y: Array.from(y, (v) => (v > 0.5 ? 1 : 0)),
  }
}

/** The classifier as a model definition: `y ~ Bernoulli(link(X w))`, `w ~ N(0, priorScale^2)` on every weight. */
function classifierDefinition(link: 'probit' | 'logistic', priorScale: number): ModelDefinition {
  return defineModel(`${link}-classification`, (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f], role: 'covariate' })
    const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
    const w = m.latent('w', d.normal(0, priorScale), { dims: [f], doc: 'the weight of each column of the design' })
    m.observe('y', d.bernoulli(link === 'probit' ? { probs: ops.probit(X.matmul(w)) } : { logits: X.matmul(w) }), y)
  })
}

/**
 * The problem a configuration stands for: the data set generated from its seed (30% of the rows held out), the
 * polynomial features of the chosen degree and the model with the prior scale given on the weights of the unit-scale
 * columns (the library's design is divided by `weightScale`, so the prior is multiplied by it).
 *
 * @param config - The problem's knobs (`dataset`, `rows`, `noise`, `dataSeed`, `link`, `degree`, `priorScale`).
 * @returns The problem.
 * @example
 * ```ts
 * import { classifierProblem } from 'omai-engine/bayesian-examples'
 * const p = classifierProblem({ dataset: 'moons', rows: 100, noise: 0.3, dataSeed: 1, link: 'probit', degree: 2, priorScale: 1 })
 * [p.train.x.length + p.test.x.length, p.map.size] // => [100, 6]
 * ```
 */
export function classifierProblem(config: Config): ClassifierProblem {
  const seed = key(num(config, 'dataSeed'))
  const g = generator(
    str(config, 'dataset') as ClassifierDataset,
    num(config, 'rows'),
    num(config, 'noise'),
    foldIn(seed, 0),
  )
  const split = trainTestSplit(g.data, { testSize: 0.3, key: foldIn(seed, 1) })
  const train = pointsOf(split.train)
  const test = pointsOf(split.test)
  const all = [...train.x, ...test.x]
  const range = (j: number): [number, number] => {
    const v = all.map((r) => r[j] ?? 0)
    const lo = Math.min(...v)
    const hi = Math.max(...v)
    const pad = 0.12 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }
  const map = polynomialFeatureMap(train.x, Math.round(num(config, 'degree')))
  const design = map.transform(train.x)
  const link = str(config, 'link') === 'logistic' ? 'logistic' : 'probit'
  return {
    train,
    test,
    extent: { x: range(0), y: range(1) },
    map,
    design,
    heldOut: map.transform(test.x),
    definition: classifierDefinition(link, num(config, 'priorScale') * map.weightScale),
    data: probitData(design, train.y),
    link,
  }
}

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v))

// The predictive probabilities at design rows for either link (the probit one is the library's).
function predictive(
  problem: ClassifierProblem,
  draws: readonly (readonly number[])[],
  rows: readonly (readonly number[])[],
): number[] {
  if (problem.link === 'probit') return probitPredictive(draws, rows)
  const out = new Array<number>(rows.length).fill(0)
  for (const w of draws)
    rows.forEach((row, i) => {
      let s = 0
      for (let j = 0; j < w.length; j++) s += (row[j] ?? 0) * (w[j] ?? 0)
      out[i] = (out[i] ?? 0) + sigmoid(s) / draws.length
    })
  return out
}

const GRID_DRAWS = 100
const SCORE_DRAWS = 300
const PAIR_COORDINATES = 5

/** The weights of a posterior as rows `[n][p]` (at most `n`), with the chain and index of each when it is made of chains. */
function weightDraws(
  post: Posterior,
  n: number,
): { readonly rows: number[][]; readonly chain: number[]; readonly index: number[] } | undefined {
  try {
    if (!post.sites.includes('w')) return undefined
    const draws = post.draws?.['w']
    const form = post.form === 'samples' && draws !== undefined
    const tensor = form ? draws : post.sample(key(1), n)['w']
    if (tensor === undefined) return undefined
    const p = tensor.shape[1] ?? 1
    const flat = toFlat(tensor)
    const total = Math.floor(flat.length / p)
    if (total === 0) return undefined
    const perChain = form && post.chains !== undefined ? post.chains.numDraws : total
    const keep = Math.min(n, total)
    const rows: number[][] = []
    const chain: number[] = []
    const index: number[] = []
    for (let k = 0; k < keep; k++) {
      const i = keep === total ? k : Math.round((k * (total - 1)) / (keep - 1 || 1))
      rows.push(Array.from({ length: p }, (_, j) => flat[i * p + j] ?? Number.NaN))
      chain.push(Math.floor(i / perChain))
      index.push(i % perChain)
    }
    return rows.every((r) => r.every(Number.isFinite)) ? { rows, chain, index } : undefined
  } catch {
    return undefined
  }
}

const meanSd = (rows: readonly (readonly number[])[], p: number): { mean: number[]; sd: number[] } => {
  const n = rows.length
  const mean = Array.from({ length: p }, (_, j) => rows.reduce((a, r) => a + (r[j] ?? 0), 0) / n)
  const sd = Array.from({ length: p }, (_, j) =>
    Math.sqrt(rows.reduce((a, r) => a + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / Math.max(1, n - 1)),
  )
  return { mean, sd }
}

/**
 * What a check against a reference needs of the reference: the `classify/scores` and the `classify/surface` of a long run
 * of the same problem (the page runs a second session with a long NUTS run and hands over what it probes).
 */
export interface ClassifierReference {
  readonly scores: {
    readonly mean: readonly number[]
    readonly sd: readonly number[]
    readonly testAccuracy: number
    readonly testLogPredictive: number
    readonly draws: number
  }
  readonly surface: {
    readonly x: readonly number[]
    readonly probabilities: readonly (readonly (readonly number[])[])[]
  }
}

/** The knobs that make the problem (what a reference of it depends on). */
export const classifierProblemKnobs: readonly string[] = [
  'dataset',
  'rows',
  'noise',
  'dataSeed',
  'link',
  'degree',
  'priorScale',
]

const problemKey = (config: Config): string => JSON.stringify(classifierProblemKnobs.map((k) => config[k]))

function surfaceGrid(
  problem: ClassifierProblem,
  rows: readonly (readonly number[])[],
  size: number,
): { readonly xs: readonly number[]; readonly ys: readonly number[]; readonly probabilities: readonly number[] } {
  if (problem.link === 'probit') return probitPredictiveGrid(rows, problem.map, problem.extent, size)
  const axis = (lo: number, hi: number): number[] =>
    Array.from({ length: size }, (_, i) => lo + ((hi - lo) * i) / (size - 1))
  const xs = axis(problem.extent.x[0], problem.extent.x[1])
  const ys = axis(problem.extent.y[0], problem.extent.y[1])
  const pts = ys.flatMap((y) => xs.map((x) => [x, y]))
  return { xs, ys, probabilities: predictive(problem, rows, problem.map.transform(pts)) }
}

/** The union of the inference methods: each option the knobs of that method (the page shows these). */
export function classifierInferenceSpace(): ReturnType<typeof union> {
  const probe = classifierProblem(DEFAULTS)
  const registry = modelRegistry({ classifier: { definition: probe.definition, data: probe.data } })
  const options: Record<string, DictSpace> = {}
  for (const [name, spec] of Object.entries(KINDS)) {
    const inner = inferenceMethodFor(registry, spec.kind, 'classifier')
    const spaces: Record<string, DictSpace['spaces'][string]> = {}
    for (const [knob, value] of Object.entries(spec.shown)) {
      const space = inner.config.spaces[knob]
      if (space === undefined) continue
      spaces[knob] = { ...space, default: value } as never
    }
    if (spec.steps !== undefined)
      spaces['steps'] = int(50, 5000, {
        default: spec.steps,
        log: true,
        label: 'steps',
        description: 'iterations of the optimiser',
      })
    options[name] = dict(spaces, { description: spec.description })
  }
  return union(options, { default: 'nuts', label: 'inference method' })
}

const DEFAULTS: Config = {
  dataset: 'moons',
  rows: 150,
  noise: 0.3,
  dataSeed: 1,
  link: 'probit',
  degree: 2,
  priorScale: 1,
}

/** The knobs of the data (a data set and its size, noise and seed). */
export const classifierDataKnobs: DictSpace = dict({
  dataset: choice([...classifierDatasets], {
    default: 'moons',
    label: 'data set',
    descriptions: {
      moons: 'two interleaved half circles',
      circles: 'one class inside a ring of the other',
      xor: 'the classes sit in opposite quadrants',
      spirals: 'two interleaved spirals: hard for a low degree',
      blobs: 'two Gaussian blobs: linearly separable',
    },
  }),
  rows: int(40, 600, { default: 150, log: true, label: 'rows' }),
  noise: real(0, 1, {
    default: 0.3,
    step: 0.05,
    label: 'noise',
    description: 'relative to the data set: 0 the least, 1 the most',
  }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model (the link, the degree of the features, the prior scale). */
export const classifierModelKnobs: DictSpace = dict({
  link: choice([...classifierLinks], {
    default: 'probit',
    label: 'link',
    descriptions: { probit: 'the normal CDF', logistic: 'the logistic function' },
  }),
  degree: int(1, 8, {
    default: 2,
    label: 'polynomial degree',
    description: 'the features are all monomials of the two inputs up to this degree',
  }),
  priorScale: real(0.05, 20, {
    default: 1,
    log: true,
    label: 'prior scale',
    description: 'standard deviation of the normal prior on each weight (of the standardised features)',
  }),
})

const classifierInference = classifierInferenceSpace()
/** The budget knobs shown with the inference: the time budget (2 s by default) and the step budget. */
export const classifierBudgetKnobs: DictSpace = dict({
  budgetMs: { ...(anytimeKnobs.spaces['budgetMs'] as RealSpace), default: 2000, label: 'time budget' },
  budgetSteps: anytimeKnobs.spaces['budgetSteps'] as DictSpace['spaces'][string],
})

/** The knobs of {@link classifierMethod} but the budget's: data, model and the inference with its own knobs. */
export const classifierKnobs: DictSpace = dict({
  ...classifierDataKnobs.spaces,
  ...classifierModelKnobs.spaces,
  inference: classifierInference,
})

/** The inference kind and the knobs of a configuration. */
function inferenceOf(config: Config): { readonly kind: string; readonly knobs: Config } {
  const given = config['inference']
  const { kind, ...knobs } = (
    typeof given === 'object' && given !== null && !Array.isArray(given) ? given : { kind: 'nuts' }
  ) as {
    kind: string
  } & Config
  return { kind: kind in KINDS ? kind : 'nuts', knobs }
}

/**
 * How many steps a configuration's run is expected to take: what the session's `steps` should be so the run ends
 * where the method does and progress reads right (a sampler's warm-up and draws, a variational method's steps).
 *
 * @param config - The method's configuration.
 * @returns The steps.
 * @example
 * ```ts
 * import { classifierSteps } from 'omai-engine/bayesian-examples'
 * classifierSteps({ inference: { kind: 'nuts', numWarmup: 100, numDraws: 200 } }) // => 302
 * ```
 */
export function classifierSteps(config: Config): number {
  const { kind, knobs } = inferenceOf(config)
  const steps = KINDS[kind]?.steps === undefined ? undefined : num(knobs, 'steps') || KINDS[kind]?.steps
  const own =
    steps ??
    (kind === 'nuts' || kind === 'hmc' || kind === 'mala' ? num(knobs, 'numWarmup') + num(knobs, 'numDraws') + 2 : 400)
  const cap = num(config, 'budgetSteps')
  return cap > 0 ? Math.min(own, cap) : own
}

/**
 * The objective of the chosen inference method as a curve on the tape: its path and name (variational inference's ELBO,
 * the Laplace fit's log density), or `undefined` for a method with none to show (a sampler has chains, see
 * `classify/diagnostics`; expectation propagation and Pathfinder end in a few steps).
 *
 * @param config - The method's configuration.
 * @returns The path and the label.
 * @example
 * ```ts
 * import { classifierObjective } from 'omai-engine/bayesian-examples'
 * classifierObjective({ inference: { kind: 'svi' } }) // => { path: 'svi/elbo', label: 'ELBO' }
 * ```
 */
export function classifierObjective(config: Config): { readonly path: string; readonly label: string } | undefined {
  const spec = KINDS[inferenceOf(config).kind]
  return spec?.objective?.path === undefined || spec.kind === 'nuts' || spec.kind === 'hmc' || spec.kind === 'mala'
    ? undefined
    : spec.objective
}

/** A built problem with its inner method, kept for the last configuration (the session asks for it several times). */
interface Built {
  readonly problem: ClassifierProblem
  readonly inner: AnyInferenceMethod
  readonly innerConfig: Config
  readonly kind: string
}
let last: { readonly id: string; readonly built: Built } | undefined

function build(config: Config): Built {
  const { kind, knobs } = inferenceOf(config)
  const id = JSON.stringify([problemKey(config), kind, knobs])
  if (last?.id === id) return last.built
  const problem = classifierProblem(config)
  const registry = modelRegistry({ classifier: { definition: problem.definition, data: problem.data } })
  const spec = KINDS[kind] as KindSpec
  const inner = inferenceMethodFor(registry, spec.kind, 'classifier')
  const { steps: _steps, ...own } = knobs
  const innerConfig = { ...(defaults(inner.config) as Config), ...spec.fixed, ...own }
  const built = { problem, inner, innerConfig, kind }
  last = { id, built }
  return built
}

/** The state of the method is the inner method's own. */
type State = unknown

const declared: InferenceMethod<State, Config, string> = defineInferenceMethod<State, Config, string>({
  name: 'classify',
  family: 'composite',
  description:
    'Bayesian probit or logistic classification of a 2-D data set with one inference method chosen from a table',
  config: dict(classifierKnobs.spaces),
  posterior: (state, config) => {
    const b = build(config)
    return (b.inner as InferenceMethod<State, Config, string>).posterior?.(state, b.innerConfig) as Posterior
  },
  defaults: {},
  contract: (config, k): Contract<State, string> => {
    const b = build(config)
    const inner = (
      typeof b.inner.contract === 'function' ? b.inner.contract(b.innerConfig, k) : b.inner.contract
    ) as Contract<State, string>
    const post = (state: State): Posterior =>
      (b.inner as InferenceMethod<State, Config, string>).posterior?.(state, b.innerConfig) as Posterior
    const { problem } = b
    const p = problem.map.size
    const names = problem.map.names.map((n) => `w[${n === '1' ? 'bias' : n}]`)
    const draws = (state: State, n: number): ReturnType<typeof weightDraws> => {
      try {
        return weightDraws(post(state), n)
      } catch {
        return undefined
      }
    }
    return {
      ...inner,
      'classify/surface': derived<State>({
        cost: 'dear',
        description:
          'the posterior predictive probability over the plane (a decisionSurface record) with the training and held-out rows',
        compute: (state, args) => {
          const w = draws(state, GRID_DRAWS)
          if (w === undefined) return null
          const size = Math.max(5, Math.min(101, Number((args as { size?: number } | undefined)?.size ?? 41)))
          const grid = surfaceGrid(problem, w.rows, size)
          const z = (offset: (v: number) => number): number[][] =>
            Array.from({ length: size }, (_, i) =>
              Array.from({ length: size }, (_, j) => offset(grid.probabilities[i * size + j] ?? Number.NaN)),
            )
          return decisionSurface({
            features: ['x0', 'x1'],
            x: [...grid.xs],
            y: [...grid.ys],
            probabilities: [z((v) => 1 - v), z((v) => v)],
            classes: ['0', '1'],
            rows: {
              points: [...problem.train.x, ...problem.test.x],
              features: ['x0', 'x1'],
              labels: [...problem.train.y, ...problem.test.y],
              classes: ['0', '1'],
            },
          }) as never
        },
      }),
      'classify/pairs': derived<State>({
        cost: 'dear',
        description: `the pair plot of the first ${PAIR_COORDINATES} coefficients (a posteriorPairs record)`,
        compute: (state) => {
          const w = draws(state, 500)
          if (w === undefined || w.rows.length < 3) return null
          const coordinates = names.slice(0, PAIR_COORDINATES)
          const columns = Object.fromEntries(coordinates.map((c, j) => [c, w.rows.map((r) => r[j] ?? 0)]))
          return posteriorPairs([{ name: b.kind, draws: columns, steps: w.index, chain: w.chain }], {
            coordinates,
            maxPoints: 300,
            frames: 2,
            densityDraws: 300,
          }) as never
        },
      }),
      'classify/scores': derived<State>({
        cost: 'dear',
        description:
          'held-out accuracy and log predictive, the coefficients, R-hat and ESS where the method has chains',
        compute: (state) => {
          const posterior = post(state)
          const w = draws(state, SCORE_DRAWS)
          if (w === undefined) return null
          const test = heldOutScores(predictive(problem, w.rows, problem.heldOut), problem.test.y)
          const train = heldOutScores(predictive(problem, w.rows, problem.design), problem.train.y)
          const { mean, sd } = meanSd(w.rows, p)
          let rhat: number | null = null
          let ess: number | null = null
          if (posterior.form === 'samples' && posterior.chains !== undefined) {
            const rows = posterior.summary().filter((r) => r.site === 'w')
            const r = rows.flatMap((x) => (x.rhat === undefined ? [] : [x.rhat]))
            const e = rows.flatMap((x) => (x.ess === undefined ? [] : [x.ess]))
            if (r.length > 0) rhat = Math.max(...r)
            if (e.length > 0) ess = Math.min(...e)
          }
          const divergences = posterior.diagnostics['divergences']
          return {
            names,
            mean,
            sd,
            testAccuracy: test.accuracy,
            testLogPredictive: test.logPredictive,
            trainAccuracy: train.accuracy,
            trainLogPredictive: train.logPredictive,
            draws:
              posterior.chains === undefined ? w.rows.length : posterior.chains.numChains * posterior.chains.numDraws,
            maxRhat: rhat,
            minEss: ess,
            divergences: typeof divergences === 'number' ? divergences : null,
          }
        },
      }),
      'classify/diagnostics': derived<State>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler (a chainDiagnostics record); null for the others',
        compute: (state) => {
          const posterior = post(state)
          if (posterior.chains === undefined) return null
          const chains = posteriorChains(posterior, { sites: ['w'] })
          const rename = (name: string): string => names[Number(/\[(\d+)\]/.exec(name)?.[1])] ?? name
          const columns = Object.fromEntries(
            Object.entries(chains.draws)
              .slice(0, PAIR_COORDINATES)
              .map(([name, rows]) => [rename(name), rows]),
          )
          if (Object.keys(columns).length === 0) return null
          const summary = posterior
            .summary()
            .filter((r) => r.site === 'w')
            .map((r) => ({ ...r, name: rename(r.name) }))
          return chainDiagnostics(columns, {
            summary,
            ...(chains.divergent === undefined ? {} : { divergent: chains.divergent }),
          }) as never
        },
      }),
      'classify/check': derived<State>({
        cost: 'dear',
        description:
          'the method against a reference run of the same problem (args: `{ scores, surface }` of the reference): the largest mean error in reference sds, the median sd ratio, the difference of the predictive over the plane, and the reference scores',
        compute: (state, args) => {
          const ref = args as unknown as ClassifierReference | null | undefined
          const w = draws(state, GRID_DRAWS)
          const refGrid = ref?.surface.probabilities[1]
          if (ref === undefined || ref === null || w === undefined || refGrid === undefined) return null
          const { mean, sd } = meanSd(w.rows, p)
          const error = mean.map(
            (m, j) => Math.abs(m - (ref.scores.mean[j] ?? 0)) / Math.max(1e-9, ref.scores.sd[j] ?? 1),
          )
          const ratio = sd.map((s, j) => s / Math.max(1e-9, ref.scores.sd[j] ?? 1))
          const size = ref.surface.x.length
          const grid = surfaceGrid(problem, w.rows, size).probabilities
          const diff = grid.map((v, i) => Math.abs(v - (refGrid[Math.floor(i / size)]?.[i % size] ?? 0)))
          return {
            maxMeanError: Math.max(...error),
            medianSdRatio: [...ratio].sort((a, b) => a - b)[Math.floor(ratio.length / 2)] ?? Number.NaN,
            maxSurfaceDifference: Math.max(...diff),
            meanSurfaceDifference: diff.reduce((a, v) => a + v, 0) / diff.length,
            referenceAccuracy: ref.scores.testAccuracy,
            referenceLogPredictive: ref.scores.testLogPredictive,
            referenceDraws: ref.scores.draws,
          }
        },
      }),
    }
  },
  make: (config, k) => {
    const b = build(config)
    const algorithm = b.inner.make(b.innerConfig, k)
    const steps = KINDS[b.kind]?.steps === undefined ? undefined : classifierSteps(config)
    return steps === undefined
      ? algorithm
      : {
          ...algorithm,
          done: (s: State, t: number) => algorithm.done?.(s, t) ?? (t >= steps ? 'converged' : undefined),
        }
  },
}) as never

/**
 * Bayesian classification of a 2-D data set by one inference method, run by name and knobs (see the file's summary):
 * a trace `Method` whose tape is the chosen inference method's own (its objective, its chains) and whose derived paths
 * `classify/surface`, `classify/pairs`, `classify/scores`, `classify/diagnostics` and
 * `classify/check` (the method against a reference run) read the posterior at any step.
 *
 * @concept inference-engines
 * @index estimator Bayesian classifier method: data, link, degree, prior, one inference method as knobs.
 * @example
 * ```ts
 * import { classifierMethod, classifierSteps } from 'omai-engine/bayesian-examples'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(classifierMethod.config), rows: 60, inference: { kind: 'laplace' } }
 * const s = session(classifierMethod, { config, steps: classifierSteps(config), mode: 'eager' })
 * const surface = s.probe(s.t, 'classify/surface', { size: 5 }) as { kind: string }
 * surface.kind // => 'decisionSurface'
 * ```
 */
export const classifierMethod: InferenceMethod<State, Config, string> = {
  ...declared,
  config: { ...declared.config, spaces: { ...declared.config.spaces, ...classifierBudgetKnobs.spaces } },
}
