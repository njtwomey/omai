import { WhiteboxError } from '../base/index.ts'
import { type Graph, transitionMatrix } from '../graph/index.ts'
import { lbinom } from '../numerics/special/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../random/index.ts'
import {
  add,
  asFloat,
  asTensor,
  exp,
  fromStorage,
  log,
  log1p,
  mul,
  neg,
  promoteFloat,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { positiveInteger, primalNumbers, traceOf, vector } from './_shared.ts'
import { simulateChain } from './markov-chains.ts'
import { type History, historyArray, type Paths, simulatePaths, type TraceOptions } from './paths.ts'

function integerSteps(steps: number, where: string): number {
  if (!(Number.isInteger(steps) && steps >= 0)) throw new WhiteboxError('value', `${where}: steps must be >= 0`)
  return steps
}

function grid(steps: number): number[] {
  return Array.from({ length: steps + 1 }, (_, i) => i)
}

/** Options of {@link simpleRandomWalk}. */
export interface WalkOptions extends TraceOptions {
  /** The probability of a step up (default 1/2; other values give a biased walk). */
  readonly p?: number
  /** The number of paths (default 1). */
  readonly paths?: number
  /** The starting point (default 0). */
  readonly start?: number
}

/**
 * The simple random walk S_n = S_0 + sum of independent steps +1 (probability p) and -1 (probability 1 - p), as a traced
 * path simulation on the times 0, 1, ..., steps.
 *
 * @param key - The key.
 * @param steps - The number of steps.
 * @param options - Up probability (biased when not 1/2), paths, start and tracing.
 * @returns Paths `[paths, steps + 1]`.
 * @concept random-walks
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { simpleRandomWalk } from 'whitebox-engine/stochastic'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(simpleRandomWalk(key(0), 10, { p: 1 }).values).at(-1) // => 10
 * ```
 */
export function simpleRandomWalk(key: Key, steps: number, options: WalkOptions = {}): Paths {
  integerSteps(steps, 'simpleRandomWalk')
  const p = options.p ?? 0.5
  if (!(p >= 0 && p <= 1)) throw new WhiteboxError('value', 'simpleRandomWalk: p must be in [0, 1]')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'simpleRandomWalk')
  return simulatePaths(
    key,
    {
      name: 'simple-random-walk',
      config: { p, paths, steps },
      times: grid(steps),
      x0: vector(new Array<number>(paths).fill(options.start ?? 0)),
      step: (x, c) => {
        const u = uniform(c.key, [paths]).data
        return add(x, vector(Array.from(u, (v) => (v < p ? 1 : -1))))
      },
    },
    traceOf(options),
  )
}

/**
 * The law of the simple random walk after n steps: P(S_n - S_0 = k) = C(n, (n + k) / 2) p^{(n+k)/2} (1 - p)^{(n-k)/2}
 * (zero unless n + k is even and |k| <= n). Differentiable in p.
 *
 * @param k - Displacements.
 * @param steps - The number of steps n.
 * @param p - The up probability (default 1/2).
 * @returns The probabilities.
 * @concept random-walks
 * @example
 * ```ts
 * import { simpleWalkPmf } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(simpleWalkPmf([-2, -1, 0, 2], 2)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.25, 0, 0.5, 0.25]
 * ```
 */
export function simpleWalkPmf(k: TensorLike, steps: number, p: TensorLike = 0.5): Value {
  integerSteps(steps, 'simpleWalkPmf')
  const [kk, pp] = promoteFloat(k, p) as [Value, Value]
  const ks = primalNumbers(kk)
  const up = ks.map((v) => (steps + v) / 2)
  const valid = ks.map((v, i) => Number.isInteger(up[i] ?? 0.5) && Math.abs(v) <= steps)
  const u = vector(
    up.map((v, i) => (valid[i] === true ? v : 0)),
    kk.shape,
  )
  const logP = add(add(lbinom(steps, u), mul(u, log(pp))), mul(sub(steps, u), log1p(neg(pp))))
  const mask = fromStorage(
    Uint8Array.from(valid, (b) => (b ? 1 : 0)),
    [...kk.shape],
    'bool',
  )
  return where(mask, exp(logP), 0)
}

/**
 * A random walk on the integer lattice Z^d: each step moves to one of the 2d neighbours uniformly (Polya's walk,
 * recurrent for d <= 2 and transient for d >= 3).
 *
 * @param key - The key.
 * @param steps - The number of steps.
 * @param options - The dimension (default 2), paths (default 1) and tracing.
 * @returns Paths `[paths, steps + 1, dim]` of lattice points.
 * @concept random-walks
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { latticeWalk } from 'whitebox-engine/stochastic'
 * latticeWalk(key(0), 5, { dim: 3, paths: 2 }).values.shape // => [2, 6, 3]
 * ```
 */
export function latticeWalk(
  key: Key,
  steps: number,
  options: { readonly dim?: number; readonly paths?: number } & TraceOptions = {},
): Paths {
  integerSteps(steps, 'latticeWalk')
  const dim = positiveInteger(options.dim ?? 2, 'dim', 'latticeWalk')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'latticeWalk')
  return simulatePaths(
    key,
    {
      name: 'lattice-walk',
      config: { dim, paths, steps },
      times: grid(steps),
      x0: fromStorage(new Float64Array(paths * dim), [paths, dim], 'float64'),
      step: (x, c) => {
        const move = randint(c.key, [paths], 0, 2 * dim).data
        const delta = new Float64Array(paths * dim)
        for (let j = 0; j < paths; j++) {
          const m = move[j] ?? 0
          delta[j * dim + (m >> 1)] = m % 2 === 0 ? 1 : -1
        }
        return add(x, fromStorage(delta, [paths, dim], 'float64'))
      },
    },
    traceOf(options),
  )
}

/**
 * A random walk on a graph: from each node, step to a neighbour with probability proportional to the edge weight (the
 * chain P = D^{-1} A of `graph.transitionMatrix`), optionally lazy. Its stationary law on a connected undirected graph
 * is proportional to the (weighted) degree.
 *
 * @param key - The key.
 * @param g - The graph.
 * @param start - The start node index.
 * @param steps - The number of steps.
 * @param options - `laziness` in [0, 1), `weighted` (default true), paths and tracing.
 * @returns Paths `[paths, steps + 1]` of node indices.
 * @concept random-walks-on-graphs
 * @example
 * ```ts
 * import { cycleGraph } from 'whitebox-engine/graph'
 * import { key } from 'whitebox-engine/random'
 * import { graphRandomWalk } from 'whitebox-engine/stochastic'
 * graphRandomWalk(key(0), cycleGraph(5), 0, 8, { paths: 3 }).values.shape // => [3, 9]
 * ```
 */
export function graphRandomWalk(
  key: Key,
  g: Graph,
  start: number,
  steps: number,
  options: { readonly laziness?: number; readonly weighted?: boolean; readonly paths?: number } & TraceOptions = {},
): Paths {
  const p = transitionMatrix(g, {
    ...(options.laziness === undefined ? {} : { laziness: options.laziness }),
    ...(options.weighted === undefined ? {} : { weighted: options.weighted }),
    dangling: 'self',
  })
  return simulateChain(key, p, start, steps, {
    ...traceOf(options),
    ...(options.paths === undefined ? {} : { paths: options.paths }),
  })
}

/**
 * The stationary law of the random walk on a connected undirected graph: pi_v = d_v / sum_u d_u (weighted degrees).
 *
 * @param g - An undirected graph without isolated nodes.
 * @param options - `weighted` (default true).
 * @returns pi `[n]`.
 * @throws WhiteboxError with code `value` for a directed graph or an isolated node.
 * @concept random-walks-on-graphs
 * @example
 * ```ts
 * import { starGraph } from 'whitebox-engine/graph'
 * import { graphWalkStationary } from 'whitebox-engine/stochastic'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(graphWalkStationary(starGraph(3))) // => [0.5, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
 * ```
 */
export function graphWalkStationary(g: Graph, options: { readonly weighted?: boolean } = {}): Tensor<'float64'> {
  if (g.directed) throw new WhiteboxError('value', 'graphWalkStationary: needs an undirected graph')
  const weighted = options.weighted ?? true
  const d = new Float64Array(g.n)
  for (const e of g.edges) {
    const w = weighted ? e.weight : 1
    d[e.source] = (d[e.source] ?? 0) + w
    if (e.target !== e.source) d[e.target] = (d[e.target] ?? 0) + w
  }
  let total = 0
  for (const v of d) {
    if (!(v > 0)) throw new WhiteboxError('value', 'graphWalkStationary: isolated nodes have no stationary mass')
    total += v
  }
  return vector(Array.from(d, (v) => v / total))
}

/** The result of {@link selfAvoidingWalks}. */
export interface SelfAvoidingWalks {
  /** The walks `[walks, steps + 1, dim]`; a trapped walk repeats its last site. */
  readonly positions: Tensor<'float64'>
  /** The number of steps each walk made before it finished or was trapped. */
  readonly lengths: Int32Array
  /** The Rosenbluth weights prod_i m_i (m_i free neighbours at step i; zero for trapped walks). */
  readonly weights: Float64Array
}

/**
 * Self-avoiding walks on Z^d grown by Rosenbluth sampling: each step moves to a uniformly chosen unvisited neighbour,
 * and the weight W = prod m_i (the numbers of choices) corrects the bias, so E[W] is the number c_n of n-step
 * self-avoiding walks and weighted averages estimate their statistics (such as the mean squared end-to-end distance,
 * which grows like n^{3/2} in two dimensions).
 *
 * @param key - The key.
 * @param steps - The walk length n.
 * @param options - The dimension (default 2), number of walks (default 100) and tracing.
 * @returns The walks, their lengths and their weights.
 * @concept self-avoiding-walks
 * @reference M. N. Rosenbluth and A. W. Rosenbluth, Monte Carlo calculation of the average extension of molecular
 *   chains, J. Chem. Phys. 23 (1955).
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { selfAvoidingWalks } from 'whitebox-engine/stochastic'
 * Array.from(selfAvoidingWalks(key(0), 2, { walks: 3 }).weights) // => [12, 12, 12]
 * ```
 */
export function selfAvoidingWalks(
  key: Key,
  steps: number,
  options: { readonly dim?: number; readonly walks?: number } & TraceOptions = {},
): SelfAvoidingWalks {
  integerSteps(steps, 'selfAvoidingWalks')
  const dim = positiveInteger(options.dim ?? 2, 'dim', 'selfAvoidingWalks')
  const walks = positiveInteger(options.walks ?? 100, 'walks', 'selfAvoidingWalks')
  const r = runAlgorithm(selfAvoidingWalkAlgorithm(dim, walks), { ...traceOf(options), key, maxSteps: steps })
  const positions = new Float64Array(walks * (steps + 1) * dim)
  const lengths = new Int32Array(walks)
  const weights = new Float64Array(walks)
  for (const [w, walk] of r.state.walks.entries()) {
    const sites = historyArray(walk.path)
    for (let i = 0; i <= steps; i++) {
      const site = sites[Math.min(i, sites.length - 1)] ?? []
      for (let d = 0; d < dim; d++) positions[(w * (steps + 1) + i) * dim + d] = site[d] ?? 0
    }
    lengths[w] = sites.length - 1
    weights[w] = walk.trapped ? 0 : walk.weight
  }
  return { positions: fromStorage(positions, [walks, steps + 1, dim], 'float64'), lengths, weights }
}

/** One growing self-avoiding walk. */
export interface GrowingWalk {
  /** The sites visited, newest first. */
  readonly path: History<readonly number[]>
  /** The visited sites as keys. */
  readonly visited: ReadonlySet<string>
  /** The Rosenbluth weight so far. */
  readonly weight: number
  /** Whether the walk has no free neighbour. */
  readonly trapped: boolean
}

/** The state of {@link selfAvoidingWalkAlgorithm}. */
export interface SelfAvoidingState {
  readonly walks: readonly GrowingWalk[]
}

/**
 * Rosenbluth growth of many self-avoiding walks as a traced {@link Algorithm}: each step extends every untrapped walk to
 * a uniformly chosen unvisited neighbour and multiplies its weight by the number of choices.
 *
 * @param dim - The lattice dimension.
 * @param walks - The number of walks.
 * @returns The algorithm.
 * @concept self-avoiding-walks
 * @example
 * ```ts
 * import { selfAvoidingWalkAlgorithm } from 'whitebox-engine/stochastic'
 * selfAvoidingWalkAlgorithm(2, 10).name // => 'rosenbluth'
 * ```
 */
export function selfAvoidingWalkAlgorithm(dim: number, walks: number): Algorithm<SelfAvoidingState> {
  positiveInteger(dim, 'dim', 'selfAvoidingWalkAlgorithm')
  positiveInteger(walks, 'walks', 'selfAvoidingWalkAlgorithm')
  const origin = new Array<number>(dim).fill(0)
  return {
    name: 'rosenbluth',
    config: { dim, walks },
    schema: channels({
      metrics: {
        alive: { description: 'walks not yet trapped' },
        weight: { description: 'mean Rosenbluth weight (estimates the number of walks)' },
      },
    }),
    init: () => ({
      walks: Array.from({ length: walks }, () => ({
        path: { item: origin, previous: undefined },
        visited: new Set([origin.join(',')]),
        weight: 1,
        trapped: false,
      })),
    }),
    step: (s, ctx) => {
      const u = uniform(ctx.key, [walks]).data
      return {
        walks: s.walks.map((w, j) => {
          if (w.trapped) return w
          const site = w.path.item
          const free: number[][] = []
          for (let m = 0; m < 2 * dim; m++) {
            const next = [...site]
            next[m >> 1] = (next[m >> 1] ?? 0) + (m % 2 === 0 ? 1 : -1)
            if (!w.visited.has(next.join(','))) free.push(next)
          }
          if (free.length === 0) return { ...w, trapped: true }
          const pick = free[Math.min(free.length - 1, Math.floor((u[j] ?? 0) * free.length))] ?? site
          const visited = new Set(w.visited)
          visited.add(pick.join(','))
          return { path: { item: pick, previous: w.path }, visited, weight: w.weight * free.length, trapped: false }
        }),
      }
    },
    observe: (s) => ({
      metrics: {
        alive: s.walks.filter((w) => !w.trapped).length,
        weight: s.walks.reduce((a, w) => a + (w.trapped ? 0 : w.weight), 0) / walks,
      },
    }),
  }
}

/**
 * The exact number c_n of n-step self-avoiding walks on Z^d by depth-first enumeration (1, 4, 12, 36, 100, 284, ... in
 * two dimensions), for small n.
 *
 * @param steps - The walk length (at most 16 in two dimensions, the enumeration being exponential).
 * @param dim - The dimension (default 2).
 * @returns c_n.
 * @throws WhiteboxError with code `value` when the enumeration would be too large.
 * @concept self-avoiding-walks
 * @example
 * ```ts
 * import { countSelfAvoidingWalks } from 'whitebox-engine/stochastic'
 * [0, 1, 2, 3, 4, 5].map((n) => countSelfAvoidingWalks(n)) // => [1, 4, 12, 36, 100, 284]
 * ```
 */
export function countSelfAvoidingWalks(steps: number, dim = 2): number {
  integerSteps(steps, 'countSelfAvoidingWalks')
  positiveInteger(dim, 'dim', 'countSelfAvoidingWalks')
  if (steps * Math.log(2 * dim - 1) > 16 * Math.log(3) + 1e-9) {
    throw new WhiteboxError('value', 'countSelfAvoidingWalks: too many walks to enumerate')
  }
  const visited = new Set<string>()
  const site = new Array<number>(dim).fill(0)
  visited.add(site.join(','))
  const count = (left: number): number => {
    if (left === 0) return 1
    let total = 0
    for (let m = 0; m < 2 * dim; m++) {
      const axis = m >> 1
      const delta = m % 2 === 0 ? 1 : -1
      site[axis] = (site[axis] ?? 0) + delta
      const id = site.join(',')
      if (!visited.has(id)) {
        visited.add(id)
        total += count(left - 1)
        visited.delete(id)
      }
      site[axis] = (site[axis] ?? 0) - delta
    }
    return total
  }
  return count(steps)
}

/** Options of {@link levyFlight}. */
export interface LevyFlightOptions extends TraceOptions {
  /** The tail index alpha of the step lengths, P(L > l) = (l / scale)^{-alpha} (default 1.5). */
  readonly alpha?: number
  /** The minimum step length (default 1). */
  readonly scale?: number
  /** The dimension (default 2). */
  readonly dim?: number
  /** The number of paths (default 1). */
  readonly paths?: number
}

/**
 * A Levy flight: a random walk in R^d with isotropic directions and Pareto step lengths, P(L > l) = (l / scale)^{-alpha}.
 * For alpha < 2 the lengths have infinite variance and the walk is superdiffusive (in the domain of attraction of an
 * alpha-stable law); rare long jumps dominate the path.
 *
 * @param key - The key.
 * @param steps - The number of steps.
 * @param options - Tail index, minimum length, dimension, paths and tracing.
 * @returns Paths `[paths, steps + 1, dim]`.
 * @concept levy-flights
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { levyFlight } from 'whitebox-engine/stochastic'
 * levyFlight(key(0), 20, { alpha: 1.2, paths: 2 }).values.shape // => [2, 21, 2]
 * ```
 */
export function levyFlight(key: Key, steps: number, options: LevyFlightOptions = {}): Paths {
  integerSteps(steps, 'levyFlight')
  const alpha = options.alpha ?? 1.5
  const scale = options.scale ?? 1
  if (!(alpha > 0 && scale > 0)) throw new WhiteboxError('value', 'levyFlight: alpha and scale must be positive')
  const dim = positiveInteger(options.dim ?? 2, 'dim', 'levyFlight')
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'levyFlight')
  return simulatePaths(
    key,
    {
      name: 'levy-flight',
      config: { alpha, scale, dim, paths, steps },
      times: grid(steps),
      x0: fromStorage(new Float64Array(paths * dim), [paths, dim], 'float64'),
      step: (x, c) => {
        const u = uniform(foldIn(c.key, 0), [paths]).data
        const z = normal(foldIn(c.key, 1), [paths, dim]).data
        const delta = new Float64Array(paths * dim)
        for (let j = 0; j < paths; j++) {
          // inverse CDF of the Pareto law, 1 - u in (0, 1]
          const length = scale * (1 - (u[j] ?? 0)) ** (-1 / alpha)
          let norm = 0
          for (let d = 0; d < dim; d++) norm += (z[j * dim + d] ?? 0) ** 2
          norm = Math.sqrt(norm) || 1
          for (let d = 0; d < dim; d++) delta[j * dim + d] = (length * (z[j * dim + d] ?? 0)) / norm
        }
        return add(x, fromStorage(delta, [paths, dim], 'float64'))
      },
    },
    traceOf(options),
  )
}

/** The closed-form answers of {@link gamblersRuin}. */
export interface GamblersRuin {
  /** P(reach n before 0 | start at i). */
  readonly winProbability: number
  /** 1 - winProbability. */
  readonly ruinProbability: number
  /** The expected number of rounds until either end. */
  readonly expectedDuration: number
}

/**
 * Gambler's ruin: a walk on {0, ..., n} with up probability p started at i and stopped at either end. With r = q / p,
 * the win probability is (1 - r^i) / (1 - r^n) (i / n for a fair game) and the expected duration
 * (i - n P(win)) / (q - p) (i (n - i) for a fair game).
 *
 * @param i - The starting fortune (0 <= i <= n).
 * @param n - The target.
 * @param p - The probability of winning a round (default 1/2).
 * @returns The win and ruin probabilities and the expected duration.
 * @throws WhiteboxError with code `value` for invalid arguments.
 * @concept gamblers-ruin
 * @example
 * ```ts
 * import { gamblersRuin } from 'whitebox-engine/stochastic'
 * gamblersRuin(3, 10) // => { winProbability: 0.3, ruinProbability: 0.7, expectedDuration: 21 }
 * ```
 */
export function gamblersRuin(i: number, n: number, p = 0.5): GamblersRuin {
  if (!(Number.isInteger(n) && n >= 1 && Number.isInteger(i) && i >= 0 && i <= n)) {
    throw new WhiteboxError('value', 'gamblersRuin: need integers 0 <= i <= n, n >= 1')
  }
  if (!(p > 0 && p < 1)) throw new WhiteboxError('value', 'gamblersRuin: p must be in (0, 1)')
  const q = 1 - p
  if (Math.abs(p - q) < 1e-12) {
    const win = i / n
    return { winProbability: win, ruinProbability: 1 - win, expectedDuration: i * (n - i) }
  }
  const r = q / p
  // (1 - r^i) / (1 - r^n), written with expm1 for accuracy near the fair game
  const win = Math.expm1(i * Math.log(r)) / Math.expm1(n * Math.log(r))
  return { winProbability: win, ruinProbability: 1 - win, expectedDuration: (i - n * win) / (q - p) }
}

/** The state of {@link gamblersRuinAlgorithm}: every gambler's fortune and how long each has played. */
export interface RuinState {
  readonly fortunes: Int32Array
  readonly rounds: Int32Array
}

/**
 * Gambler's ruin simulated for many gamblers at once as a traced {@link Algorithm}: one round per step until every
 * gambler has hit 0 or n.
 *
 * @param i - The starting fortune.
 * @param n - The target.
 * @param p - The probability of winning a round.
 * @param gamblers - The number of independent gamblers.
 * @returns The algorithm.
 * @concept gamblers-ruin
 * @example
 * ```ts
 * import { gamblersRuinAlgorithm } from 'whitebox-engine/stochastic'
 * gamblersRuinAlgorithm(2, 4, 0.5, 10).name // => 'gamblers-ruin'
 * ```
 */
export function gamblersRuinAlgorithm(i: number, n: number, p: number, gamblers: number): Algorithm<RuinState> {
  gamblersRuin(i, n, Math.min(Math.max(p, 1e-12), 1 - 1e-12))
  positiveInteger(gamblers, 'gamblers', 'gamblersRuinAlgorithm')
  const playing = (f: number): boolean => f > 0 && f < n
  return {
    name: 'gamblers-ruin',
    config: { i, n, p, gamblers },
    schema: channels({
      metrics: {
        playing: { description: 'gamblers still playing' },
        ruined: { description: 'fraction of gamblers ruined so far' },
      },
    }),
    init: () => ({ fortunes: new Int32Array(gamblers).fill(i), rounds: new Int32Array(gamblers) }),
    step: (s, ctx) => {
      const u = uniform(ctx.key, [gamblers]).data
      const fortunes = Int32Array.from(s.fortunes, (f, j) => (playing(f) ? f + ((u[j] ?? 0) < p ? 1 : -1) : f))
      const rounds = Int32Array.from(s.rounds, (r, j) => (playing(s.fortunes[j] ?? 0) ? r + 1 : r))
      return { fortunes, rounds }
    },
    done: (s) => (s.fortunes.every((f) => !playing(f)) ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        playing: s.fortunes.filter(playing).length,
        ruined: s.fortunes.filter((f) => f === 0).length / gamblers,
      },
    }),
  }
}

/**
 * Simulate gambler's ruin for many gamblers ({@link gamblersRuinAlgorithm}).
 *
 * @param key - The key.
 * @param i - The starting fortune.
 * @param n - The target.
 * @param p - The probability of winning a round (default 1/2).
 * @param options - Gamblers (default 1000), the round cap (default 100000) and tracing.
 * @returns Whether each gambler won and how many rounds each played.
 * @throws WhiteboxError with code `convergence` when some gambler is still playing at the cap.
 * @concept gamblers-ruin
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * import { simulateGamblersRuin } from 'whitebox-engine/stochastic'
 * Array.from(simulateGamblersRuin(key(0), 1, 2, 0.5, { gamblers: 3 }).rounds) // => [1, 1, 1]
 * ```
 */
export function simulateGamblersRuin(
  key: Key,
  i: number,
  n: number,
  p = 0.5,
  options: { readonly gamblers?: number; readonly maxRounds?: number } & TraceOptions = {},
): { readonly won: Uint8Array; readonly rounds: Int32Array } {
  const cap = options.maxRounds ?? 100000
  const r = runAlgorithm(gamblersRuinAlgorithm(i, n, p, options.gamblers ?? 1000), {
    ...traceOf(options),
    key,
    maxSteps: cap,
  })
  if (r.status !== 'completed' || r.state.fortunes.some((f) => f > 0 && f < n)) {
    throw new WhiteboxError('convergence', `simulateGamblersRuin: games still running after ${cap} rounds`)
  }
  return { won: Uint8Array.from(r.state.fortunes, (f) => (f === n ? 1 : 0)), rounds: r.state.rounds }
}

/**
 * The mean squared displacement E|X_n - X_0|^2 of simulated walks at every step (the diffusion signature: linear in n
 * for ordinary walks, faster for Levy flights, n^{3/2} for 2-d self-avoiding walks).
 *
 * @param paths - Paths `[paths, n]` or `[paths, n, dim]`.
 * @returns One value per time.
 * @concept random-walks
 * @example
 * ```ts
 * import { meanSquaredDisplacement } from 'whitebox-engine/stochastic'
 * Array.from(meanSquaredDisplacement({ times: Float64Array.from([0, 1, 2]), values: [[0, 1, 2], [0, -1, 0]] })) // => [0, 1, 2]
 * ```
 */
export function meanSquaredDisplacement(paths: Paths): Float64Array {
  const v = asFloat(asTensor(paths.values))
  const [p = 1, n = 0] = v.shape
  const dim = v.shape.length > 2 ? (v.shape[2] ?? 1) : 1
  const x = primalNumbers(v)
  const out = new Float64Array(n)
  for (let j = 0; j < p; j++) {
    for (let t = 0; t < n; t++) {
      let s = 0
      for (let d = 0; d < dim; d++) s += ((x[(j * n + t) * dim + d] ?? 0) - (x[j * n * dim + d] ?? 0)) ** 2
      out[t] = (out[t] ?? 0) + s / p
    }
  }
  return out
}
