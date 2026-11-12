/**
 * The Gaussian mixture as an example method: hidden populations recovered without known assignments. The data are either a
 * generator with a known truth (a number of true components, how far apart they sit, their weights, rows, a seed) or the
 * Old Faithful waiting times; the model is the library's K-component mixture (`gaussianMixtureModel`: unknown weights with
 * a Dirichlet prior, unknown means free or ordered, one spread or one per component, a discrete label per row that the
 * samplers sum out or sample) on the standardised data; and one inference method of the library runs it. Free means have
 * K! symmetric modes (label switching): a raw mean of a free run is not a quantity a reference can be compared on, so the
 * derived paths read label-invariant quantities, those of the components sorted by their means. They are
 * `mixture/fit` (the predictive density with its band, the components, the truth), `mixture/pairs` (the means, a weight
 * and the spread, as drawn, so swapped labels show), `mixture/diagnostics`, `mixture/summary` (the sorted parameters and
 * the predictive density: what the reference run hands to the check) and `mixture/check` (a run against it).
 *
 * @index estimator Gaussian mixture example: generated or Old Faithful data, free or ordered means, label-invariant check.
 */

import { oldFaithful } from 'omai-engine/datasets'
import { posteriorPairs } from 'omai-engine/exposition'
import type { Posterior } from 'omai-engine/probprog'
import type { InferenceMethod } from 'omai-engine/probprog/inference'
import { gaussianMixtureModel } from 'omai-engine/probprog/models'
import { normal as drawNormal, uniform as drawUniform, foldIn, key } from 'omai-engine/random'
import { choice, type DictSpace, dict, int, real } from 'omai-engine/space'
import { quantile } from 'omai-engine/stats'
import {
  abs,
  add,
  any,
  arange,
  argsort,
  broadcastTo,
  cast,
  cumsum,
  div,
  exp,
  item,
  less,
  lessEqual,
  max,
  min,
  minimum,
  mul,
  ones,
  pow,
  reshape,
  sliceAxis,
  square,
  stack,
  sub,
  sum,
  take,
  takeAlongAxis,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
} from 'omai-engine/tensor'
import { type Config, derived } from 'omai-engine/trace'
import {
  defineExample,
  drawRows,
  type Example,
  type ExampleProblem,
  exampleDiagnostics,
  exampleNumber,
  exampleString,
} from './example-method.ts'
import {
  checkRows,
  flattenCheck,
  type PairCoordinate,
  type PosteriorRow,
  type PosteriorRows,
} from './hierarchical-views.ts'

/** The data on offer. */
export const mixtureDatasets: readonly string[] = ['generated', 'oldFaithful']

/** The problem a mixture configuration stands for. */
export interface MixtureProblem extends ExampleProblem {
  /** The observations, in their own units. */
  readonly y: readonly number[]
  /** What the data were standardised by (the model sees `(y - mean) / scale`). */
  readonly mean: number
  readonly scale: number
  readonly dataset: string
  /** What the values are called on a figure (`value`, `waiting time (minutes)`). */
  readonly label: string
  readonly components: number
  readonly ordered: boolean
  readonly separate: boolean
  /** What generated the data (none for the real data), in the data's units, the components in increasing order of mean. */
  readonly truth: { readonly means: readonly number[]; readonly weights: readonly number[]; readonly sd: number } | null
  /** The grid the densities are drawn on, in the data's units. */
  readonly grid: readonly number[]
}

/** The weights of `k` generated components: equal, or each half the one before. */
function generatingWeights(k: number, unequal: boolean): number[] {
  const raw = unequal ? pow(0.5, arange(k)) : ones([k])
  return toFlat(div(raw, sum(raw)))
}

/**
 * The problem a configuration stands for: the data (generated from the seed with a known truth, or the 272 Old Faithful
 * waiting times), standardised, and the mixture model bound to them.
 *
 * @param config - The knobs (`dataset`, `trueComponents`, `separation`, `weights`, `rows`, `dataSeed`, `components`, `ordering`, `spread`, `concentration`).
 * @returns The problem.
 * @example
 * ```ts
 * import { mixtureProblem } from './index.ts'
 * const p = mixtureProblem({ dataset: 'generated', trueComponents: 2, separation: 4, weights: 'equal', rows: 50, dataSeed: 1, components: 2, ordering: 'free', spread: 'shared', concentration: 1 })
 * [p.y.length, p.truth?.means] // => [50, [-2, 2]]
 * ```
 */
export function mixtureProblem(config: Config): MixtureProblem {
  const components = Math.max(1, Math.round(exampleNumber(config, 'components') || 2))
  const ordered = exampleString(config, 'ordering') !== 'free'
  const separate = exampleString(config, 'spread') === 'separate'
  const concentration = exampleNumber(config, 'concentration') || 1
  const dataset = exampleString(config, 'dataset') === 'oldFaithful' ? 'oldFaithful' : 'generated'
  let y: Value
  let truth: MixtureProblem['truth'] = null
  let label = 'value'
  if (dataset === 'oldFaithful') {
    // The waiting time is the second column of the data.
    y = take(reshape(oldFaithful().data.x, [-1, 2]), 1, 1)
    label = 'waiting time (minutes)'
  } else {
    const n = Math.max(2, Math.round(exampleNumber(config, 'rows') || 200))
    const k = Math.max(1, Math.round(exampleNumber(config, 'trueComponents') || 2))
    const separation = exampleNumber(config, 'separation') || 3
    const weights = generatingWeights(k, exampleString(config, 'weights') === 'unequal')
    const means = mul(sub(arange(k), (k - 1) / 2), separation)
    const seed = foldIn(key(exampleNumber(config, 'dataSeed')), 0)
    const u = drawUniform(foldIn(seed, 0), [n])
    const e = drawNormal(foldIn(seed, 1), [n])
    // The component of each draw: the number of cumulative weights it has passed.
    const passed = lessEqual(reshape(cumsum(weights), [1, k]), reshape(u, [n, 1]))
    const component = cast(minimum(sum(passed, 1), k - 1), 'int32')
    y = add(take(means, component), e)
    truth = { means: toFlat(means), weights, sd: 1 }
  }
  const mean = item(div(sum(y), y.shape[0] ?? 1))
  const scale = Math.sqrt(item(div(sum(square(sub(y, mean))), y.shape[0] ?? 1))) || 1
  const lo = item(min(y))
  const hi = item(max(y))
  const pad = 0.15 * (hi - lo || 1)
  const grid = toFlat(add(div(mul(arange(GRID), hi - lo + 2 * pad), GRID - 1), lo - pad))
  return {
    y: toFlat(y),
    mean,
    scale,
    dataset,
    label,
    components,
    ordered,
    separate,
    truth,
    grid,
    // With one component the ordering constrains nothing: it is dropped (a one-entry ordered vector is a plain one).
    definition: gaussianMixtureModel({
      components,
      ordered: ordered && components > 1,
      concentration,
      sharedSpread: !separate,
    }),
    data: { y: div(sub(y, mean), scale) },
  }
}

/** The knobs of the data: generated with a known truth, or the Old Faithful waiting times. */
export const mixtureDataKnobs: DictSpace = dict({
  dataset: choice([...mixtureDatasets], {
    default: 'generated',
    label: 'data',
    descriptions: {
      generated: 'normal populations with known means, weights and spread: the truth is known',
      oldFaithful: 'the waiting times (minutes) between 272 eruptions of Old Faithful: two populations, no truth',
    },
  }),
  trueComponents: int(1, 5, { default: 2, label: 'true components' }),
  separation: real(0.5, 8, {
    default: 3,
    step: 0.25,
    label: 'separation',
    description:
      'the distance between neighbouring component means, in standard deviations of a component: below 2 they overlap',
  }),
  weights: choice(['equal', 'unequal'], {
    default: 'equal',
    label: 'true weights',
    descriptions: {
      equal: 'every component the same share of the rows',
      unequal: 'each component half the share of the one before',
    },
  }),
  rows: int(20, 1000, { default: 150, log: true, label: 'rows' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the number of components, the ordering of the means, the spreads and the weight prior. */
export const mixtureModelKnobs: DictSpace = dict({
  components: int(1, 6, {
    default: 2,
    label: 'components K',
    description: 'how many normal populations the model has',
  }),
  ordering: choice(['ordered', 'free'], {
    default: 'ordered',
    label: 'means',
    descriptions: {
      ordered: 'the means are constrained to increase: one mode, labels cannot switch',
      free: 'exchangeable means: K! symmetric modes (label switching), so a raw mean has no single meaning',
    },
  }),
  spread: choice(['shared', 'separate'], {
    default: 'shared',
    label: 'spread',
    descriptions: {
      shared: 'one standard deviation for every component',
      separate: 'a standard deviation of its own for each component',
    },
  }),
  concentration: real(0.1, 10, {
    default: 1,
    log: true,
    label: 'weight prior concentration',
    description:
      'the symmetric Dirichlet prior on the weights: 1 is flat, below 1 favours a few heavy components, above 1 equal ones',
  }),
})

/** The knobs that make the problem. */
export const mixtureProblemKnobs: readonly string[] = [
  'dataset',
  'trueComponents',
  'separation',
  'weights',
  'rows',
  'dataSeed',
  'components',
  'ordering',
  'spread',
  'concentration',
]

/** The knobs a configuration does not use (the generator's, for the real data). */
export function mixtureHiddenKnobs(config: Config): readonly string[] {
  return exampleString(config, 'dataset') === 'oldFaithful' ? ['trueComponents', 'separation', 'weights', 'rows'] : []
}

const GRID = 90
const FIT_DRAWS = 300
const PAIR_DRAWS = 500

/** The draws of a mixture's parameters, in the model's standardised units: each is a row per draw and a column per component. */
interface MixtureDraws {
  readonly mu: Value
  readonly w: Value
  readonly sigma: Value
  /** The chain and the index within it of each draw. */
  readonly chain: number[]
  readonly index: number[]
}

/** The draws of the means, weights and spreads of a posterior (at most `n`), or `undefined` while it has none. */
function readDraws(post: Posterior, problem: MixtureProblem, n: number): MixtureDraws | undefined {
  const k = problem.components
  const got = drawRows(post, ['mu', 'weights', 'sigma'], n)
  if (got === undefined) return undefined
  const spreads = problem.separate ? k : 1
  if (got.rows[0]?.length !== 2 * k + spreads) return undefined
  const rows = tensor(got.rows)
  return {
    mu: sliceAxis(rows, 1, 0, k),
    w: sliceAxis(rows, 1, k, 2 * k),
    sigma: broadcastTo(sliceAxis(rows, 1, 2 * k, 2 * k + spreads), [got.rows.length, k]),
    chain: got.chain,
    index: got.index,
  }
}

/** The standard normal density at `z`. */
const phi = (z: Value): Value => div(exp(mul(mul(-0.5, z), z)), Math.sqrt(2 * Math.PI))

/** The normal density at `x` with mean `mu` and sd `sigma` (broadcast against each other). */
const density = (x: Value, mu: Value, sigma: Value): Value => div(phi(div(sub(x, mu), sigma)), sigma)

/** What the posterior says, label-invariant: the sorted components' parameters and the predictive density on the grid. */
interface MixtureReading {
  readonly rows: PosteriorRow[]
  /** The posterior mean of the predictive density, its 90% band, and the posterior mean of each sorted component's curve. */
  readonly mean: number[]
  readonly lo: number[]
  readonly hi: number[]
  readonly components: number[][]
  /** The share of draws whose means are not in increasing order as labelled (label switching, or a mode other than the ordered one). */
  readonly outOfOrder: number
  readonly draws: number
}

/** The 5% and 95% quantiles of a sample, as a tensor of two. */
const bounds = (v: Value): Value => quantile(v, tensor([0.05, 0.95]))

/** The mean and the sample standard deviation of a sample. */
const moments = (v: Value): { mean: number; sd: number } => {
  const n = v.shape[0] ?? 1
  const mean = item(div(sum(v), n))
  return { mean, sd: Math.sqrt(item(div(sum(square(sub(v, mean))), Math.max(1, n - 1)))) }
}

/** Read a posterior into its label-invariant summary, or `undefined` while it has no usable draws. */
function readMixture(post: Posterior, problem: MixtureProblem, n: number): MixtureReading | undefined {
  const d = readDraws(post, problem, n)
  const draws = d?.mu.shape[0] ?? 0
  if (d === undefined || draws < 3) return undefined
  const k = problem.components
  // Each draw's components sorted by their means, in the units of the data.
  const order = argsort(d.mu, 1)
  const mu = add(mul(takeAlongAxis(d.mu, order, 1), problem.scale), problem.mean)
  const w = takeAlongAxis(d.w, order, 1)
  const sigma = mul(takeAlongAxis(d.sigma, order, 1), problem.scale)
  const rowOf = (name: string, v: Value): PosteriorRow => {
    const [lo, hi] = toFlat(bounds(v))
    return { name, ...moments(v), lo: lo as number, hi: hi as number, ess: null, rhat: null }
  }
  const rows: PosteriorRow[] = [
    // banned-ok: non-numeric: one result row (an object) per component
    ...Array.from({ length: k }, (_, j) => rowOf(`mean ${j + 1}`, take(mu, j, 1))),
    ...Array.from({ length: k }, (_, j) => rowOf(`weight ${j + 1}`, take(w, j, 1))),
    ...Array.from({ length: problem.separate ? k : 1 }, (_, j) =>
      rowOf(problem.separate ? `spread ${j + 1}` : 'spread', take(sigma, j, 1)),
    ),
  ]
  // The weighted density of each component of each draw at each grid point: [grid, draw, component].
  const x = reshape(tensor(problem.grid as number[]), [-1, 1, 1])
  const terms = mul(reshape(w, [1, draws, k]), density(x, reshape(mu, [1, draws, k]), reshape(sigma, [1, draws, k])))
  const at = sum(terms, 2)
  // banned-ok: non-numeric: one band tensor per grid point, stacked
  const band = stack(Array.from({ length: problem.grid.length }, (_, g) => bounds(take(at, g, 0))))
  const outOfOrder = k < 2 ? 0 : item(sum(any(less(sliceAxis(d.mu, 1, 1, k), sliceAxis(d.mu, 1, 0, k - 1)), 1))) / draws
  return {
    rows,
    mean: toFlat(div(sum(at, 1), draws)),
    lo: toFlat(take(band, 0, 1)),
    hi: toFlat(take(band, 1, 1)),
    components: toArray(transpose(div(sum(terms, 1), draws))) as number[][],
    outOfOrder,
    draws,
  }
}

/** The density of the generator on a grid (null for real data). */
function truthDensity(problem: MixtureProblem): number[] | null {
  const t = problem.truth
  if (t === null) return null
  const x = reshape(tensor(problem.grid as number[]), [-1, 1])
  const terms = mul(tensor(t.weights as number[]), density(x, tensor(t.means as number[]), tensor(t.sd)))
  return toFlat(sum(terms, 1))
}

/** What the reference run hands to the check: the sorted parameters (as rows) and the predictive density. */
export interface MixtureSummary extends PosteriorRows {
  readonly density: readonly number[]
}

const declared: Example<MixtureProblem> = defineExample<MixtureProblem>({
  name: 'mixture',
  budgetMs: 8000,
  description:
    'A Gaussian mixture on generated or Old Faithful data, one inference method, checked on label-invariant quantities against a long run',
  modelName: 'mixture',
  dataKnobs: mixtureDataKnobs,
  modelKnobs: mixtureModelKnobs,
  problemKnobs: mixtureProblemKnobs,
  problem: mixtureProblem,
  derived: ({ problem, kind, posterior: post }) => {
    const k = problem.components
    const t = problem.truth
    const dx = (problem.grid[problem.grid.length - 1] as number) - (problem.grid[0] as number)
    const step = dx / (problem.grid.length - 1)
    const truth = truthDensity(problem)
    const pairNames: readonly PairCoordinate[] = [
      { label: 'mean 1', from: 'mu[0]' },
      ...(k > 1 ? [{ label: 'mean 2', from: 'mu[1]' }] : []),
      ...(k > 1 ? [{ label: 'weight 1', from: 'weights[0]' }] : []),
      { label: problem.separate ? 'spread 1' : 'spread', from: problem.separate ? 'sigma[0]' : 'sigma' },
    ]
    const summaryOf = (state: unknown, n: number): MixtureSummary | null => {
      const r = readMixture(post(state), problem, n)
      if (r === undefined) return null
      return {
        rows: r.rows,
        draws: r.draws,
        divergences: null,
        maxRhat: null,
        minEss: null,
        form: 'samples',
        density: r.mean,
      }
    }
    return {
      'mixture/fit': derived<unknown>({
        cost: 'dear',
        description:
          "the data, the posterior predictive density with its 90% band, the components (sorted by mean) and the generator's density on a grid, and the sorted components' parameters against the truth, with the share of draws whose means are out of order",
        compute: (state) => {
          const r = readMixture(post(state), problem, FIT_DRAWS)
          if (r === undefined) return null
          return {
            x: problem.grid,
            data: problem.y,
            label: problem.label,
            mean: r.mean,
            lo: r.lo,
            hi: r.hi,
            components: r.components,
            truth,
            rows: r.rows.map((row) => {
              const j = Number(/\d+/.exec(row.name)?.[0] ?? 1) - 1
              const quantity = row.name.startsWith('mean') ? 'means' : row.name.startsWith('weight') ? 'weights' : null
              const known =
                t === null || k !== t.means.length
                  ? null
                  : quantity === 'means'
                    ? (t.means[j] ?? null)
                    : quantity === 'weights'
                      ? (t.weights[j] ?? null)
                      : row.name.startsWith('spread')
                        ? t.sd
                        : null
              return { ...row, truth: known }
            }),
            outOfOrder: r.outOfOrder,
            draws: r.draws,
          } as never
        },
      }),
      'mixture/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of the first two means, the first weight and the spread as the run labelled them (a posteriorPairs record): free means show the symmetric modes and the swapped labels',
        compute: (state) => {
          const d = readDraws(post(state), problem, PAIR_DRAWS)
          if (d === undefined || (d.mu.shape[0] ?? 0) < 3) return null
          // The draws of one coordinate, in the units of the data: a means, a weights or a sigma column.
          const column = (c: PairCoordinate): number[] => {
            const m = /^(\w+)\[?(\d*)\]?$/.exec(c.from)
            const j = Number(m?.[2] || 0)
            return toFlat(
              m?.[1] === 'mu'
                ? add(mul(take(d.mu, j, 1), problem.scale), problem.mean)
                : m?.[1] === 'weights'
                  ? take(d.w, j, 1)
                  : mul(take(d.sigma, j, 1), problem.scale),
            )
          }
          const columns = Object.fromEntries(pairNames.map((c) => [c.label, column(c)]))
          const known: Record<string, number> =
            t === null || !problem.ordered || k !== t.means.length
              ? {}
              : {
                  'mean 1': t.means[0] as number,
                  ...(k > 1 ? { 'mean 2': t.means[1] as number, 'weight 1': t.weights[0] as number } : {}),
                  [problem.separate ? 'spread 1' : 'spread']: t.sd,
                }
          return posteriorPairs([{ name: kind, draws: columns, steps: d.index, chain: d.chain }], {
            coordinates: pairNames.map((c) => c.label),
            maxPoints: 300,
            frames: 2,
            densityDraws: 300,
            truth: known,
          }) as never
        },
      }),
      'mixture/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler for the means, the weights and the spread, in standardised units (a chainDiagnostics record); null for the others',
        compute: (state) => {
          const rename = (name: string): string => {
            const m = /^(\w+)\[(\d+)\]$/.exec(name)
            if (m === null) return name === 'sigma' ? 'spread' : name
            const j = Number(m[2]) + 1
            return m[1] === 'mu' ? `mean ${j}` : m[1] === 'weights' ? `weight ${j}` : `spread ${j}`
          }
          return exampleDiagnostics(post(state), ['mu', 'weights', 'sigma'], rename, 6) as never
        },
      }),
      'mixture/summary': derived<unknown>({
        cost: 'dear',
        description:
          "the sorted components' parameters (rows) and the predictive density on the grid: the label-invariant summary of a run, what the check takes of the reference",
        compute: (state) => summaryOf(state, 800) as never,
      }),
      'mixture/check': derived<unknown>({
        cost: 'dear',
        description:
          "the run against a reference run of the same data (args `{ reference }`, the reference's `mixture/summary`) on label-invariant quantities: the sorted means, weights and spread (differences in the reference's sds) and the predictive density (the total variation distance between the two)",
        compute: (state, args) => {
          const ref = (args as { reference?: MixtureSummary } | null | undefined)?.reference
          const mine = summaryOf(state, 800)
          if (ref === undefined || mine === null) return null
          const check = checkRows(
            mine,
            ref,
            // banned-ok: non-numeric: the names of the components
            Array.from({ length: k }, (_, j) => `mean ${j + 1}`),
          )
          if (check === undefined) return null
          const tv = 0.5 * step * item(sum(abs(sub(tensor(mine.density as number[]), tensor(ref.density as number[])))))
          const means = check.focus.map((f) => f.meanError)
          return {
            ...flattenCheck(check),
            densityDistance: tv,
            maxMeanDifference: means.length === 0 ? null : Math.max(...means),
          } as never
        },
      }),
    }
  },
})

/**
 * The Gaussian mixture as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape is the chosen
 * inference method's own, with the derived paths `mixture/fit`, `mixture/pairs`, `mixture/diagnostics`, `mixture/summary`
 * and `mixture/check`. Its reference keeps the data and the model but orders the means, so the long run has one mode.
 *
 * @concept inference-engines
 * @index estimator Gaussian mixture method: generated or Old Faithful data, free or ordered means, one inference method.
 * @example
 * ```ts
 * import { mixtureMethod, mixtureSteps } from './index.ts'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(mixtureMethod.config), rows: 60, budgetMs: 0 }
 * const s = session(mixtureMethod, { config, steps: mixtureSteps(config), mode: 'eager' })
 * const fit = s.probe(s.t, 'mixture/fit') as { rows: { name: string }[] }
 * fit.rows[0]?.name // => 'mean 1'
 * ```
 */
export const mixtureMethod: InferenceMethod<unknown, Config, string> = declared.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const mixtureSteps: (config: Config) => number = declared.steps

/**
 * The mixture example with its reference: the same data and model with the means ordered (one mode, whatever the run is), so
 * the long run is a run a sampler can mix on, and the check reads quantities that do not depend on the labels.
 */
export const mixtureExample: Example<MixtureProblem> = {
  ...declared,
  reference: (config, inference, budgetMs) => ({
    ...declared.reference(config, inference, budgetMs),
    ordering: 'ordered',
  }),
}
