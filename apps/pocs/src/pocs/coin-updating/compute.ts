/**
 * The coin page's computing, free of any thread: the observations of a coin (or of a Poisson counter), the ladder of
 * prefixes of them, and one inference method fitted at every rung of the ladder through the library's methods
 * (`inferenceMethodFor` on the Beta-Bernoulli or Gamma-Poisson model with the chosen prior), each rung summarised as a
 * density on a grid, a mean, a standard deviation, a 95% interval and the distance to the exact conjugate posterior (the
 * validation kernel's `agree`). A worker runs this (`worker.ts`); the tests run it on the main thread.
 */

import { inferenceMethodFor, modelRegistry } from 'omai-engine/probprog/inference'
import { betaBernoulliModel, gammaPoissonModel } from 'omai-engine/probprog/models'
import {
  agree,
  betaBernoulli as betaBernoulliExact,
  type ExactPosterior,
  fromPosterior,
  gammaPoisson as gammaPoissonExact,
  weightedOracle,
} from 'omai-engine/probprog/validation'
import { bernoulli as drawBernoulli, poisson as drawPoisson, foldIn, key } from 'omai-engine/random'
import { gaussianKde } from 'omai-engine/stats'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { JobHost, PoolRequest, PoolResult } from '../../shared/workerPool.ts'

/** The two models: a coin's bias (Beta-Bernoulli) and a counter's rate (Gamma-Poisson). */
export type Family = 'beta' | 'gamma'

/** The prior presets, named the same for both families. */
export type PresetId = 'flat' | 'jeffreys' | 'sceptical' | 'strong'

/** A prior's two parameters: Beta(a, b) for a coin, Gamma(shape a, rate b) for a counter. */
export interface Prior {
  readonly a: number
  readonly b: number
}

/** The presets of each family, with what each says. */
export const PRESETS: Readonly<
  Record<Family, Readonly<Record<PresetId, { readonly prior: Prior; readonly note: string }>>>
> = {
  beta: {
    flat: { prior: { a: 1, b: 1 }, note: 'Beta(1, 1): every bias equally likely' },
    jeffreys: { prior: { a: 0.5, b: 0.5 }, note: 'Beta(1/2, 1/2): Jeffreys, mass at the extremes' },
    sceptical: { prior: { a: 8, b: 8 }, note: 'Beta(8, 8): a fair coin is likely' },
    strong: { prior: { a: 60, b: 20 }, note: 'Beta(60, 20): firmly believes heads come 75% of the time' },
  },
  gamma: {
    flat: { prior: { a: 1, b: 0.1 }, note: 'Gamma(1, 0.1): a vague exponential, mean 10' },
    jeffreys: { prior: { a: 0.5, b: 0.1 }, note: 'Gamma(1/2, 0.1): Jeffreys shape, vague scale' },
    sceptical: { prior: { a: 10, b: 2 }, note: 'Gamma(10, 2): the rate is probably near 5' },
    strong: { prior: { a: 100, b: 10 }, note: 'Gamma(100, 10): firmly believes the rate is 10' },
  },
}

/** What the user chose. */
export interface Settings {
  readonly family: Family
  /** The true bias of the coin, or the true rate of the counter. */
  readonly truth: number
  /** The observations in all. */
  readonly n: number
  readonly preset: PresetId
  readonly seed: number
}

/** The problem a run solves: the observations, the prior, the rungs of the ladder and the grid the densities are on. */
export interface Problem {
  readonly family: Family
  readonly prior: Prior
  readonly truth: number
  readonly observations: readonly number[]
  /** The numbers of observations at which every method is fitted, ascending from 0 to all. */
  readonly ladder: readonly number[]
  /** The grid over the parameter, and the interval it spans. */
  readonly grid: readonly number[]
  readonly domain: readonly [number, number]
  readonly seed: number
}

/** The site of the parameter in each model. */
export const SITE: Readonly<Record<Family, string>> = { beta: 'p', gamma: 'rate' }

const GRID_POINTS = 241

/**
 * The numbers of observations at which to fit: 0, then about `rungs - 1` values spread evenly on a log scale up to `n`.
 *
 * @param n - The observations in all.
 * @param rungs - About how many fits (fewer when `n` is small: every integer is a rung then).
 * @returns The ascending, distinct prefix lengths, the first 0 and the last `n`.
 */
export function ladderOf(n: number, rungs = 21): number[] {
  const out = new Set<number>([0, n])
  for (let i = 1; i < rungs - 1; i++)
    out.add(Math.min(n, Math.max(1, Math.round(Math.exp((Math.log(n) * i) / (rungs - 2))))))
  return [...out].sort((a, b) => a - b)
}

/** The exact conjugate posterior after the first `n` observations. */
export function exactAt(problem: Problem, n: number): ExactPosterior {
  const y = problem.observations.slice(0, n)
  return problem.family === 'beta'
    ? betaBernoulliExact({ ...problem.prior, y })
    : gammaPoissonExact({ shape: problem.prior.a, rate: problem.prior.b, counts: y })
}

/**
 * Generates the observations from the seed and works out the ladder, the grid and the interval the densities are drawn
 * over (the unit interval for a bias; for a rate, from 0 to beyond the exact posterior at the end and the true rate).
 *
 * @param settings - The family, the truth, the number of observations, the prior preset and the seed.
 * @returns The problem.
 */
export function makeProblem(settings: Settings): Problem {
  const { family, truth, n, seed } = settings
  const k = foldIn(key(seed), 0)
  const observations = Array.from(toFlat(family === 'beta' ? drawBernoulli(k, truth, [n]) : drawPoisson(k, truth, [n])))
  const prior = PRESETS[family][settings.preset].prior
  const base = { family, prior, truth, observations, seed }
  const last = exactAt({ ...base, ladder: [], grid: [], domain: [0, 1] }, n).oracles('x')[0]
  const hi = family === 'beta' ? 1 : Math.max(2.2 * truth, (last?.quantile(0.9995) ?? 1) * 1.1)
  const lo = 0
  const eps = (hi - lo) * 0.002
  const grid = Array.from({ length: GRID_POINTS }, (_, i) => lo + eps + ((hi - lo - 2 * eps) * i) / (GRID_POINTS - 1))
  return { ...base, ladder: ladderOf(n), grid, domain: [lo, hi] }
}

/** A method on offer. */
export interface CoinMethod {
  readonly id: string
  readonly label: string
  /** A name of `methodKinds` in `omai-engine/probprog/inference` (`exact` is the conjugate posterior itself). */
  readonly kind: string
  readonly config: Readonly<Record<string, string | number | boolean>>
  /** The session's steps for one fit. */
  readonly steps: number
  readonly note: string
}

/** The methods that run on both models, the exact one first (it is always run). */
export const METHODS: readonly CoinMethod[] = [
  { id: 'exact', label: 'Exact', kind: 'exact', config: {}, steps: 0, note: 'the conjugate posterior in closed form' },
  {
    id: 'laplace',
    label: 'Laplace',
    kind: 'laplace',
    config: {},
    steps: 400,
    note: 'a normal at the mode of the logit (log) of the parameter',
  },
  {
    id: 'svi',
    label: 'SVI',
    kind: 'svi',
    config: { guide: 'autoNormal', learningRate: 0.02, numParticles: 4 },
    steps: 1500,
    note: 'a mean-field normal on the logit (log) scale, fitted by stochastic gradients',
  },
  {
    id: 'ep',
    label: 'EP',
    kind: 'ep',
    config: {},
    steps: 50,
    note: 'expectation propagation with a beta (gamma) message per observation',
  },
  {
    id: 'nuts',
    label: 'NUTS',
    kind: 'nuts',
    config: { numChains: 2, numWarmup: 150, numDraws: 250 },
    steps: 1_000_000,
    note: 'the no-U-turn sampler: 500 draws, so its density is a little noisy',
  },
  {
    id: 'importance',
    label: 'Importance',
    kind: 'importance',
    config: { numSamples: 4000, batchSize: 1000 },
    steps: 6,
    note: 'weighted draws from the prior: they thin out as the posterior narrows',
  },
  {
    id: 'smc',
    label: 'Tempered SMC',
    kind: 'tempered-smc',
    config: { numParticles: 300 },
    steps: 200,
    note: 'particles moved from the prior to the posterior',
  },
]

/** The ids shown at first, besides the exact posterior. */
export const DEFAULT_METHODS: readonly string[] = ['laplace', 'svi', 'nuts', 'importance']

/** One fit: the posterior after `n` observations, as the page draws it. */
export interface Frame {
  readonly n: number
  /** The density at each point of the problem's grid. */
  readonly density: Float32Array
  readonly mean: number
  readonly sd: number
  readonly lower: number
  readonly upper: number
  /** KL(exact || method) on the exact posterior's equal-mass bins, and the Wasserstein-1 distance in exact sds (0 for the exact). */
  readonly kl: number
  readonly w1: number
  /** The time the fit took. */
  readonly ms: number
}

/** What a worker is asked: one method on one problem. */
export interface CoinRequest extends PoolRequest {
  readonly problem: Problem
}

/** What a worker returns for one method: its frames, one per rung of the ladder it got to. */
export interface CoinResult extends PoolResult {
  readonly id: string
  readonly label: string
  readonly frames: readonly Frame[]
  readonly message: string | undefined
}

/** The ids of the methods that are not exact, for the ones the page lets the user tick. */
export const methodById = (id: string): CoinMethod => {
  const m = METHODS.find((x) => x.id === id)
  if (m === undefined) throw new Error(`the coin page has no method '${id}'`)
  return m
}

const finite = (xs: ArrayLike<number>): boolean => Array.from(xs).every(Number.isFinite)

/** One fit of one method at one rung. */
function fitOnce(spec: CoinMethod, problem: Problem, n: number): Frame {
  const t0 = performance.now()
  const { family, grid } = problem
  const site = SITE[family]
  const exact = exactAt(problem, n)
  const oracle = exact.oracles(site)[0]
  if (oracle === undefined) throw new Error('no exact oracle')
  const lower = oracle.quantile(0.025)
  const upper = oracle.quantile(0.975)
  if (spec.kind === 'exact')
    return {
      n,
      density: Float32Array.from(toFlat(exact.posterior.prob(grid) as never)),
      mean: oracle.mean,
      sd: oracle.sd,
      lower,
      upper,
      kl: 0,
      w1: 0,
      ms: performance.now() - t0,
    }
  const data = family === 'beta' ? { y: problem.observations.slice(0, n) } : { k: problem.observations.slice(0, n) }
  const definition =
    family === 'beta'
      ? betaBernoulliModel(problem.prior)
      : gammaPoissonModel({ shape: problem.prior.a, rate: problem.prior.b })
  const registry = modelRegistry({ m: { definition, data } })
  const method = inferenceMethodFor(registry, spec.kind, 'm')
  const s = session(method, { config: { ...spec.config }, seed: problem.seed, steps: spec.steps, mode: 'eager' })
  const post = method.posterior?.(s.state as never, s.config as never)
  if (post === undefined) throw new Error(`${spec.label} gives no posterior`)
  const approx = fromPosterior(spec.label, post, { key: key(problem.seed + 17), numDraws: 4000, sites: [site] })
  const row = agree([approx], { oracle: { [site]: oracle }, quantities: [site] }).rows[0]
  const draws = approx.draws[site] as ArrayLike<number>
  const q = weightedOracle(site, draws, approx.weights)
  // A fitted normal or an exact form has its density in closed form; draws are smoothed (weighted, reflected at the edges).
  let density: Float32Array
  try {
    if (post.form !== 'variational') throw new Error('draws')
    density = Float32Array.from(toFlat(post.marginal(site).prob(grid) as never))
    if (!finite(density)) throw new Error('not finite')
  } catch {
    const boundary = family === 'beta' ? { lower: 0, upper: 1 } : { lower: 0 }
    const kde = gaussianKde(Array.from(draws), {
      boundary,
      ...(approx.weights === undefined ? {} : { weights: Array.from(approx.weights) }),
    })
    density = Float32Array.from(toFlat(kde.evaluate(grid)))
  }
  return {
    n,
    density,
    mean: row?.mean ?? Number.NaN,
    sd: row?.sd ?? Number.NaN,
    lower: q.quantile(0.025),
    upper: q.quantile(0.975),
    kl: row?.kl ?? Number.NaN,
    w1: row?.wasserstein ?? Number.NaN,
    ms: performance.now() - t0,
  }
}

/**
 * Fits one method at every rung of the problem's ladder, handing the thread back between the fits so a stop is heard,
 * and returns the frames it got to.
 *
 * @param request - The method and the problem.
 * @param host - The clock, the hand-over, the stop flag and the status line.
 * @returns The method's result.
 */
export async function runCoin(request: CoinRequest, host: JobHost): Promise<CoinResult> {
  const spec = methodById(request.method)
  const { ladder } = request.problem
  const frames: Frame[] = []
  const t0 = host.now()
  let message: string | undefined
  for (const [i, n] of ladder.entries()) {
    if (host.stopped()) {
      message = `stopped after ${frames.length} of ${ladder.length} fits`
      break
    }
    host.status({ type: 'status', phase: 'running', step: i, elapsedMs: host.now() - t0 })
    try {
      frames.push(fitOnce(spec, request.problem, n))
    } catch (e) {
      message = `${spec.label} failed at n = ${n}: ${e instanceof Error ? e.message.slice(0, 80) : String(e)}`
      break
    }
    await host.yieldToPage()
  }
  host.status({ type: 'status', phase: 'finishing', step: frames.length, elapsedMs: host.now() - t0 })
  return {
    id: spec.id,
    label: spec.label,
    steps: frames.length,
    elapsedMs: host.now() - t0,
    frames,
    message,
  }
}
