/**
 * The shared shape of Markov chain Monte Carlo, in BlackJAX's style: the `MCMCKernel` contract (`init` and `step` with
 * an explicit key), warm-up `Adaptation`s, `samplerAlgorithm`, which runs a kernel as a traced algorithm recording
 * acceptance, divergences and step sizes, and `sampleChain` and `sampleChains` for one or several independent chains.
 * Also the `Chains` record of draws and statistics and helpers to thin, pool, discard warm-up and map draws back to
 * constrained parameters.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { foldIn, type Key, split as splitKey, uniform } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  channels,
  nullTracer,
  type RunHandle,
  runAlgorithm,
  startRun,
  type TracePolicy,
  type Tracer,
  type TraceSchema,
} from '../trace/index.ts'
import type { LogDensity } from './target.ts'

/** The state every Markov chain Monte Carlo kernel carries: the current point and its log density. */
export interface MCMCState {
  readonly position: Float64Array
  readonly logDensity: number
}

/**
 * What one transition did. Every kernel reports its acceptance probability and whether the proposal was taken;
 * Hamiltonian kernels add divergences, tree depth, integration steps, energy and step size.
 */
export interface TransitionInfo {
  /** The Metropolis acceptance probability of this transition (the mean over the trajectory for NUTS). */
  readonly acceptanceRate: number
  readonly accepted: boolean
  readonly divergent?: boolean
  readonly treeDepth?: number
  readonly numSteps?: number
  /** The Hamiltonian at the new state (for E-BFMI). */
  readonly energy?: number
  readonly stepSize?: number
}

/** A kernel's next state and what the transition did. */
export interface ChainTransition<S> {
  readonly state: S
  readonly info: TransitionInfo
}

/**
 * A Markov kernel in the BlackJAX shape: `init` builds a state from a position, `step` moves it with an explicit key.
 * Kernels are plain values with fixed parameters (step size, mass matrix, proposal scale); adaptation builds new
 * kernels, so a kernel never changes under a running chain.
 *
 * @concept markov-chain-monte-carlo
 */
export interface MCMCKernel<S extends MCMCState = MCMCState> {
  readonly name: string
  readonly dim: number
  readonly config: Json
  init(position: ArrayLike<number>): S
  step(key: Key, state: S): ChainTransition<S>
}

/** The tuned kernel and state after warm-up, with the tuned parameters as data. */
export interface Warmup<S extends MCMCState = MCMCState> {
  readonly kernel: MCMCKernel<S>
  readonly state: S
  readonly parameters: Json
}

/** Options of an {@link Adaptation}'s `run`. */
export interface AdaptationRunOptions {
  readonly key: Key
  readonly numWarmup: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  readonly parent?: RunHandle
  /** Called with every warm-up transition (to keep warm-up draws). */
  readonly onTransition?: (state: MCMCState, info: TransitionInfo) => void
}

/**
 * A warm-up procedure: runs from a starting position for `numWarmup` iterations, tuning a kernel (step size, mass
 * matrix, proposal covariance), and returns the tuned kernel to sample with.
 *
 * @concept markov-chain-monte-carlo
 */
export interface Adaptation<S extends MCMCState = MCMCState> {
  readonly name: string
  readonly dim: number
  run(position: Float64Array, options: AdaptationRunOptions): Warmup<S>
}

/** The state of {@link samplerAlgorithm}: the kernel state and the last transition. */
export interface SamplerStep<S extends MCMCState = MCMCState> {
  readonly state: S
  readonly info?: TransitionInfo
}

const STAT_NAMES = [
  'logDensity',
  'acceptanceRate',
  'accepted',
  'divergent',
  'treeDepth',
  'numSteps',
  'energy',
  'stepSize',
] as const

/** The names of the per-draw sampler statistics a {@link Chains} value can hold. */
export type SamplerStat = (typeof STAT_NAMES)[number]

/**
 * The trace channels of a sampler with `dim` coordinates: per-iteration log density, acceptance, divergences, tree
 * depth, integration steps, energy and step size, and the position.
 *
 * @param dim - The number of coordinates.
 * @returns The schema.
 * @concept tracing
 * @example
 * ```ts
 * import { samplerSchema } from 'omai-engine/sampling'
 * Object.keys(samplerSchema(2).metrics).includes('divergent') // => true
 * ```
 */
export function samplerSchema(dim: number): TraceSchema {
  return channels({
    metrics: {
      logDensity: { description: 'log density at the current position' },
      acceptanceRate: { description: 'Metropolis acceptance probability of the transition', direction: 'higher' },
      accepted: { description: '1 when the proposal was accepted' },
      divergent: { description: '1 when the trajectory diverged', direction: 'lower' },
      treeDepth: { description: 'depth of the NUTS tree' },
      numSteps: { description: 'number of integrator steps in the transition' },
      energy: { description: 'Hamiltonian at the new state' },
      stepSize: { description: 'integrator or proposal step size' },
    },
    tensors: { position: { role: 'state', description: 'the current position', shape: [dim] } },
  })
}

function infoMetrics(info: TransitionInfo | undefined, logDensity: number): Record<string, number> {
  const out: Record<string, number> = { logDensity }
  if (info === undefined) return out
  out['acceptanceRate'] = info.acceptanceRate
  out['accepted'] = info.accepted ? 1 : 0
  if (info.divergent !== undefined) out['divergent'] = info.divergent ? 1 : 0
  if (info.treeDepth !== undefined) out['treeDepth'] = info.treeDepth
  if (info.numSteps !== undefined) out['numSteps'] = info.numSteps
  if (info.energy !== undefined) out['energy'] = info.energy
  if (info.stepSize !== undefined) out['stepSize'] = info.stepSize
  return out
}

/**
 * A kernel as a traced {@link Algorithm}: each step is one transition with the step's key; the trace records the log
 * density, acceptance, divergences, tree depth, integration steps, energy and step size of every iteration, and the
 * position as a state tensor.
 *
 * @param kernel - The kernel.
 * @param position - The starting position.
 * @param options - A run name (default the kernel's name).
 * @returns The algorithm (run it with `runAlgorithm`).
 * @concept markov-chain-monte-carlo
 * @index algorithm Run any MCMC kernel as a steppable Algorithm.
 * @example
 * ```ts
 * import { randomWalkMetropolis, samplerAlgorithm, numericTarget } from 'omai-engine/sampling'
 * import { recorder, runAlgorithm, metricSeries } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const rec = recorder()
 * runAlgorithm(samplerAlgorithm(randomWalkMetropolis(t, { scale: 1 }), [0]), { seed: 1, maxSteps: 5, tracer: rec })
 * metricSeries(rec.run(), 'acceptanceRate').values.length // => 5
 * ```
 */
export function samplerAlgorithm<S extends MCMCState>(
  kernel: MCMCKernel<S>,
  position: ArrayLike<number>,
  options: { readonly name?: string } = {},
): Algorithm<SamplerStep<S>> {
  return {
    name: options.name ?? kernel.name,
    config: kernel.config,
    schema: samplerSchema(kernel.dim),
    init: () => ({ state: kernel.init(position) }),
    step: (s, ctx) => kernel.step(ctx.key, s.state),
    observe: (s) => ({
      metrics: infoMetrics(s.info, s.state.logDensity),
      tensors: { position: { role: 'state', value: s.state.position, shape: [kernel.dim] } },
    }),
  }
}

/** Draws and statistics of one chain, row-major `[draw, dim]` and `[draw]`. */
export interface ChainDraws {
  readonly draws: Float64Array
  readonly stats: Readonly<Partial<Record<SamplerStat, Float64Array>>>
  readonly count: number
}

/** Collects draws and statistics of one chain into flat buffers. */
function collector(
  dim: number,
  capacity: number,
): {
  readonly push: (state: MCMCState, info: TransitionInfo | undefined) => void
  readonly result: () => ChainDraws
} {
  const draws = new Float64Array(capacity * dim)
  const stats: Partial<Record<SamplerStat, Float64Array>> = {}
  let n = 0
  return {
    push: (state, info) => {
      if (n >= capacity) return
      draws.set(state.position, n * dim)
      for (const [name, value] of Object.entries(infoMetrics(info, state.logDensity))) {
        const key = name as SamplerStat
        let buffer = stats[key]
        if (buffer === undefined) {
          buffer = new Float64Array(capacity).fill(Number.NaN)
          stats[key] = buffer
        }
        buffer[n] = value
      }
      n += 1
    },
    result: () => ({ draws, stats, count: n }),
  }
}

/** Options of {@link sampleChain}. */
export interface SampleChainOptions {
  readonly key: Key
  readonly numDraws: number
  /** Iterations run before the kept draws and discarded (default 0). */
  readonly numWarmup?: number
  /** Keep every `thin`-th draw after warm-up (default 1). */
  readonly thin?: number
  /** Also return the warm-up draws (default false). */
  readonly keepWarmup?: boolean
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
  readonly parent?: RunHandle
  readonly runId?: string
}

/** The result of {@link sampleChain}. */
export interface ChainResult<S extends MCMCState = MCMCState> extends ChainDraws {
  readonly warmup?: ChainDraws
  /** The last state, to continue the chain. */
  readonly state: S
}

function checkCount(n: number, name: string, min: number): number {
  if (!Number.isInteger(n) || n < min) throw new OmaiError('value', `sampleChain: ${name} must be an integer >= ${min}`)
  return n
}

/**
 * Run one chain of a kernel from a position: `numWarmup` discarded iterations, then `numDraws` kept draws, keeping
 * every `thin`-th. The run is traced as one algorithm run ({@link samplerAlgorithm}).
 *
 * @param kernel - The kernel.
 * @param position - The starting position.
 * @param options - Key, number of draws, warm-up, thinning, whether to keep warm-up draws, tracer and policy.
 * @returns The draws `[numDraws * dim]`, per-draw statistics, optional warm-up draws and the last state.
 * @throws OmaiError with code `value` for negative or fractional counts.
 * @concept markov-chain-monte-carlo
 * @example
 * ```ts
 * import { randomWalkMetropolis, sampleChain, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const r = sampleChain(randomWalkMetropolis(t, { scale: 2 }), [0], { key: key(0), numDraws: 100, numWarmup: 50, thin: 2 })
 * r.draws.length // => 100
 * ```
 */
export function sampleChain<S extends MCMCState>(
  kernel: MCMCKernel<S>,
  position: ArrayLike<number>,
  options: SampleChainOptions,
): ChainResult<S> {
  const numDraws = checkCount(options.numDraws, 'numDraws', 0)
  const numWarmup = checkCount(options.numWarmup ?? 0, 'numWarmup', 0)
  const thin = checkCount(options.thin ?? 1, 'thin', 1)
  const kept = collector(kernel.dim, numDraws)
  const warm = options.keepWarmup === true ? collector(kernel.dim, numWarmup) : undefined
  const base = samplerAlgorithm(kernel, position)
  const collecting: Algorithm<SamplerStep<S>> = {
    ...base,
    step: (s, ctx) => {
      const next = base.step(s, ctx)
      const t = ctx.t + 1
      if (t <= numWarmup) warm?.push(next.state, next.info)
      else if ((t - numWarmup) % thin === 0) kept.push(next.state, next.info)
      return next
    },
  }
  const result = runAlgorithm(collecting, {
    key: options.key,
    maxSteps: numWarmup + numDraws * thin,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
    ...(options.runId === undefined ? {} : { runId: options.runId }),
  })
  return { ...kept.result(), ...(warm === undefined ? {} : { warmup: warm.result() }), state: result.state.state }
}

/**
 * Draws of several chains: `draws[chain, draw, dim]` on the sampler's (unconstrained) coordinates, per-draw sampler
 * statistics `[chain, draw]`, optional warm-up draws, and the tuned parameters of each chain.
 *
 * @concept markov-chain-monte-carlo
 */
export interface Chains {
  readonly numChains: number
  readonly numDraws: number
  readonly dim: number
  readonly names: readonly string[]
  readonly draws: Tensor<'float64'>
  readonly stats: Readonly<Partial<Record<SamplerStat, Tensor<'float64'>>>>
  readonly warmup?: {
    readonly draws: Tensor<'float64'>
    readonly stats: Readonly<Partial<Record<SamplerStat, Tensor<'float64'>>>>
  }
  /** The tuned parameters per chain (step size, mass matrix, ...), `{}` for kernels without warm-up adaptation. */
  readonly parameters: readonly Json[]
}

/** How chains start: one shared position, one per chain, or a function of a key and the chain index. */
export type ChainInit =
  | ArrayLike<number>
  | readonly ArrayLike<number>[]
  | ((key: Key, chain: number) => ArrayLike<number>)

/** Options of {@link sampleChains}. */
export interface SampleChainsOptions {
  readonly key: Key
  /** Number of chains (default 4). */
  readonly numChains?: number
  /** Draws kept per chain (default 1000). */
  readonly numDraws?: number
  /** Warm-up iterations per chain: adaptation for an {@link Adaptation}, discarded iterations for a kernel (default 0). */
  readonly numWarmup?: number
  readonly thin?: number
  /** Starting points (default uniform on [-2, 2]^dim per chain, as Stan). */
  readonly init?: ChainInit
  readonly keepWarmup?: boolean
  /** Coordinate names (default `x[i]`). */
  readonly names?: readonly string[]
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

function startOf(init: ChainInit | undefined, key: Key, chain: number, dim: number): Float64Array {
  let p: ArrayLike<number>
  if (init === undefined) p = uniform(key, [dim], { low: -2, high: 2 }).data
  else if (typeof init === 'function') p = init(key, chain)
  else if (init.length > 0 && typeof init[0] !== 'number') {
    const each = init as readonly ArrayLike<number>[]
    const q = each[chain]
    if (q === undefined) throw new OmaiError('value', `sampleChains: no starting point for chain ${chain}`)
    p = q
  } else p = init as ArrayLike<number>
  if (p.length !== dim)
    throw new OmaiError('shape', `sampleChains: a starting point has ${p.length} values, need ${dim}`)
  return Float64Array.from(p)
}

function isAdaptation<S extends MCMCState>(s: MCMCKernel<S> | Adaptation<S>): s is Adaptation<S> {
  return 'run' in s
}

function stackStats(per: readonly ChainDraws[], numDraws: number): Partial<Record<SamplerStat, Tensor<'float64'>>> {
  const out: Partial<Record<SamplerStat, Tensor<'float64'>>> = {}
  for (const name of STAT_NAMES) {
    if (!per.some((c) => c.stats[name] !== undefined)) continue
    const buffer = new Float64Array(per.length * numDraws).fill(Number.NaN)
    per.forEach((c, i) => {
      const s = c.stats[name]
      if (s !== undefined) buffer.set(s.subarray(0, numDraws), i * numDraws)
    })
    out[name] = fromStorage(buffer, [per.length, numDraws], 'float64')
  }
  return out
}

function stackDraws(per: readonly ChainDraws[], numDraws: number, dim: number): Tensor<'float64'> {
  const buffer = new Float64Array(per.length * numDraws * dim)
  per.forEach((c, i) => {
    buffer.set(c.draws.subarray(0, numDraws * dim), i * numDraws * dim)
  })
  return fromStorage(buffer, [per.length, numDraws, dim], 'float64')
}

/**
 * Run several chains with independent keys (split from `key`), as NumPyro's `MCMC(num_chains=...)` and BlackJAX's
 * vmapped kernels: each chain starts from its own point, warms up (adapting when given an {@link Adaptation}, else
 * discarding the kernel's first iterations) and keeps `numDraws` draws. The whole run is traced as a parent run with
 * one child run per chain phase.
 *
 * @param sampler - A kernel (fixed parameters) or an adaptation (tunes a kernel during warm-up).
 * @param options - Key, chains, draws, warm-up, thinning, starting points, names, tracer and policy.
 * @returns The chains.
 * @throws OmaiError with code `value` for invalid counts, and `shape` for starting points of the wrong length.
 * @concept markov-chain-monte-carlo
 * @index function Run several chains of any MCMC kernel.
 * @example
 * ```ts
 * import { randomWalkMetropolis, sampleChains, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const c = sampleChains(randomWalkMetropolis(t, { scale: 2 }), { key: key(0), numChains: 2, numDraws: 50, numWarmup: 20 })
 * c.draws.shape // => [2, 50, 1]
 * ```
 */
export function sampleChains<S extends MCMCState>(
  sampler: MCMCKernel<S> | Adaptation<S>,
  options: SampleChainsOptions,
): Chains {
  const numChains = checkCount(options.numChains ?? 4, 'numChains', 1)
  const numDraws = checkCount(options.numDraws ?? 1000, 'numDraws', 0)
  const numWarmup = checkCount(options.numWarmup ?? 0, 'numWarmup', 0)
  const dim = sampler.dim
  const names = options.names ?? Array.from({ length: dim }, (_, i) => `x[${i}]`)
  if (names.length !== dim) throw new OmaiError('value', `sampleChains: ${names.length} names for ${dim} coordinates`)
  const run = startRun(options.tracer ?? nullTracer, {
    name: 'sampleChains',
    seed: null,
    config: { sampler: sampler.name, numChains, numDraws, numWarmup, thin: options.thin ?? 1 },
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const keys = splitKey(options.key, numChains)
  const kept: ChainDraws[] = []
  const warm: ChainDraws[] = []
  const parameters: Json[] = []
  keys.forEach((k, chain) => {
    const start = startOf(options.init, foldIn(k, 0), chain, dim)
    let kernel: MCMCKernel<S>
    let state: S | undefined
    let warmCollect: ReturnType<typeof collector> | undefined
    if (isAdaptation(sampler)) {
      warmCollect = options.keepWarmup === true ? collector(dim, numWarmup) : undefined
      const w = sampler.run(start, {
        key: foldIn(k, 1),
        numWarmup,
        parent: run,
        ...(warmCollect === undefined ? {} : { onTransition: warmCollect.push }),
      })
      kernel = w.kernel
      state = w.state
      parameters.push(w.parameters)
    } else {
      kernel = sampler
      parameters.push({})
    }
    const r = sampleChain(kernel, state?.position ?? start, {
      key: foldIn(k, 2),
      numDraws,
      numWarmup: isAdaptation(sampler) ? 0 : numWarmup,
      ...(options.thin === undefined ? {} : { thin: options.thin }),
      keepWarmup: options.keepWarmup === true && !isAdaptation(sampler),
      parent: run,
      runId: `${run.runId}/chain-${chain}`,
    })
    kept.push(r)
    const w = warmCollect?.result() ?? r.warmup
    if (w !== undefined) warm.push(w)
  })
  run.end('completed', numChains)
  return {
    numChains,
    numDraws,
    dim,
    names,
    draws: stackDraws(kept, numDraws, dim),
    stats: stackStats(kept, numDraws),
    ...(warm.length === numChains
      ? { warmup: { draws: stackDraws(warm, numWarmup, dim), stats: stackStats(warm, numWarmup) } }
      : {}),
    parameters,
  }
}

function drawsOf(chains: Chains): Float64Array {
  return chains.draws.data
}

/**
 * The draws of one coordinate as `[chain, draw]`, the input shape of every diagnostic.
 *
 * @param chains - The chains.
 * @param which - The coordinate's index or name.
 * @returns A `[chain, draw]` tensor.
 * @throws OmaiError with code `value` for an unknown coordinate.
 * @example
 * ```ts
 * import { chainsFromDraws, parameterDraws } from 'omai-engine/sampling'
 * import { toArray } from 'omai-engine/tensor'
 * const c = chainsFromDraws([[[1, 10], [2, 20]]], { names: ['a', 'b'] })
 * toArray(parameterDraws(c, 'b')) // => [[10, 20]]
 * ```
 */
export function parameterDraws(chains: Chains, which: number | string): Tensor<'float64'> {
  const i = typeof which === 'number' ? which : chains.names.indexOf(which)
  if (!(Number.isInteger(i) && i >= 0 && i < chains.dim)) {
    throw new OmaiError('value', `parameterDraws: unknown coordinate ${String(which)}`)
  }
  const { numChains, numDraws, dim } = chains
  const x = drawsOf(chains)
  const out = new Float64Array(numChains * numDraws)
  for (let c = 0; c < numChains; c++)
    for (let d = 0; d < numDraws; d++) out[c * numDraws + d] = x[(c * numDraws + d) * dim + i] ?? Number.NaN
  return fromStorage(out, [numChains, numDraws], 'float64')
}

/**
 * All draws pooled across chains, `[chain * draw, dim]` (chain-major), for posterior means and plots.
 *
 * @param chains - The chains.
 * @returns The pooled draws.
 * @example
 * ```ts
 * import { chainsFromDraws, flattenChains } from 'omai-engine/sampling'
 * flattenChains(chainsFromDraws([[[1], [2]], [[3], [4]]])).shape // => [4, 1]
 * ```
 */
export function flattenChains(chains: Chains): Tensor<'float64'> {
  return fromStorage(drawsOf(chains).slice(), [chains.numChains * chains.numDraws, chains.dim], 'float64')
}

function selectDraws(chains: Chains, keep: readonly number[]): Chains {
  const { numChains, numDraws, dim } = chains
  const n = keep.length
  const x = drawsOf(chains)
  const draws = new Float64Array(numChains * n * dim)
  for (let c = 0; c < numChains; c++)
    keep.forEach((d, j) => {
      draws.set(x.subarray((c * numDraws + d) * dim, (c * numDraws + d + 1) * dim), (c * n + j) * dim)
    })
  const stats: Partial<Record<SamplerStat, Tensor<'float64'>>> = {}
  for (const [name, t] of Object.entries(chains.stats)) {
    const s = new Float64Array(numChains * n)
    for (let c = 0; c < numChains; c++)
      keep.forEach((d, j) => {
        s[c * n + j] = t.data[c * numDraws + d] ?? Number.NaN
      })
    stats[name as SamplerStat] = fromStorage(s, [numChains, n], 'float64')
  }
  const { warmup: _, ...rest } = chains
  return { ...rest, numDraws: n, draws: fromStorage(draws, [numChains, n, dim], 'float64'), stats }
}

/**
 * Keep every `factor`-th draw of each chain (ArviZ `thin`), starting from the first.
 *
 * @param chains - The chains.
 * @param factor - The thinning factor (a positive integer).
 * @returns The thinned chains (warm-up draws dropped).
 * @throws OmaiError with code `value` for a factor that is not a positive integer.
 * @example
 * ```ts
 * import { chainsFromDraws, thin } from 'omai-engine/sampling'
 * thin(chainsFromDraws([[[1], [2], [3], [4], [5]]]), 2).numDraws // => 3
 * ```
 */
export function thin(chains: Chains, factor: number): Chains {
  if (!(Number.isInteger(factor) && factor >= 1))
    throw new OmaiError('value', 'thin: factor must be a positive integer')
  const keep: number[] = []
  for (let d = 0; d < chains.numDraws; d += factor) keep.push(d)
  return selectDraws(chains, keep)
}

/**
 * Drop the first `n` draws of every chain: warm-up handling for chains whose draws include burn-in (draws from
 * elsewhere, or a kernel run without `numWarmup`).
 *
 * @param chains - The chains.
 * @param n - Draws to drop per chain.
 * @returns The remaining draws.
 * @throws OmaiError with code `value` unless 0 <= n <= numDraws.
 * @example
 * ```ts
 * import { chainsFromDraws, discardWarmup } from 'omai-engine/sampling'
 * discardWarmup(chainsFromDraws([[[1], [2], [3]]]), 2).numDraws // => 1
 * ```
 */
export function discardWarmup(chains: Chains, n: number): Chains {
  if (!(Number.isInteger(n) && n >= 0 && n <= chains.numDraws)) {
    throw new OmaiError('value', 'discardWarmup: n must be an integer in [0, numDraws]')
  }
  return selectDraws(
    chains,
    Array.from({ length: chains.numDraws - n }, (_, i) => i + n),
  )
}

/**
 * Build {@link Chains} from draws computed elsewhere (another sampler, a file): `[chain, draw, dim]`, or
 * `[chain, draw]` for one coordinate.
 *
 * @param draws - The draws.
 * @param options - Coordinate names and per-draw statistics `[chain, draw]`.
 * @returns The chains.
 * @throws OmaiError with code `shape` unless the draws have two or three axes.
 * @example
 * ```ts
 * import { chainsFromDraws } from 'omai-engine/sampling'
 * chainsFromDraws([[1, 2, 3], [4, 5, 6]]).draws.shape // => [2, 3, 1]
 * ```
 */
export function chainsFromDraws(
  draws: TensorLike,
  options: {
    readonly names?: readonly string[]
    readonly stats?: Readonly<Partial<Record<SamplerStat, TensorLike>>>
  } = {},
): Chains {
  const x = asTensor(draws)
  if (x.shape.length !== 2 && x.shape.length !== 3) {
    throw new OmaiError('shape', 'chainsFromDraws: draws must be [chain, draw] or [chain, draw, dim]')
  }
  const numChains = x.shape[0] ?? 0
  const numDraws = x.shape[1] ?? 0
  const dim = x.shape[2] ?? 1
  const data = Float64Array.from((x as Tensor).data as ArrayLike<number>)
  const names = options.names ?? Array.from({ length: dim }, (_, i) => `x[${i}]`)
  if (names.length !== dim)
    throw new OmaiError('value', `chainsFromDraws: ${names.length} names for ${dim} coordinates`)
  const stats: Partial<Record<SamplerStat, Tensor<'float64'>>> = {}
  for (const [name, v] of Object.entries(options.stats ?? {})) {
    const t = asTensor(v as TensorLike) as Tensor
    if (t.shape[0] !== numChains || t.shape[1] !== numDraws) {
      throw new OmaiError('shape', `chainsFromDraws: statistic ${name} must be [chain, draw]`)
    }
    stats[name as SamplerStat] = fromStorage(
      Float64Array.from(t.data as ArrayLike<number>),
      [numChains, numDraws],
      'float64',
    )
  }
  return {
    numChains,
    numDraws,
    dim,
    names,
    draws: fromStorage(data, [numChains, numDraws, dim], 'float64'),
    stats,
    parameters: Array.from({ length: numChains }, () => ({})),
  }
}

/**
 * Map every draw through a target's `constrain` (unconstrained coordinates back to the model's parameters), as
 * NumPyro's `get_samples` does after sampling on the reals. Statistics and parameters carry over.
 *
 * @param chains - Chains on the target's unconstrained coordinates.
 * @param target - The target that defines the map.
 * @param options - Names of the constrained coordinates (default the target's names when the dimension is unchanged).
 * @returns The constrained chains.
 * @example
 * ```ts
 * import { chainsFromDraws, constrainChains, targetFromDistribution } from 'omai-engine/sampling'
 * import { exponential } from 'omai-engine/dists'
 * import { toArray } from 'omai-engine/tensor'
 * const c = constrainChains(chainsFromDraws([[[0], [Math.log(2)]]]), targetFromDistribution(exponential(1)))
 * toArray(c.draws) // => [[[1], [2]]]
 * ```
 */
export function constrainChains(
  chains: Chains,
  target: LogDensity,
  options: { readonly names?: readonly string[] } = {},
): Chains {
  const { numChains, numDraws, dim } = chains
  const x = drawsOf(chains)
  const rows: Float64Array[] = []
  for (let i = 0; i < numChains * numDraws; i++) rows.push(target.constrain(x.subarray(i * dim, (i + 1) * dim)))
  const out = rows[0]?.length ?? dim
  const draws = new Float64Array(numChains * numDraws * out)
  rows.forEach((r, i) => {
    draws.set(r, i * out)
  })
  const names = options.names ?? (out === target.dim ? target.names : Array.from({ length: out }, (_, i) => `y[${i}]`))
  const { warmup: _, ...rest } = chains
  return { ...rest, dim: out, names, draws: fromStorage(draws, [numChains, numDraws, out], 'float64') }
}
