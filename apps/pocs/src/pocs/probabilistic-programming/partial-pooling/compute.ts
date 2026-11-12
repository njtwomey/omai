/**
 * The partial pooling page's computing, free of any thread: groups of very different sizes (their means with the
 * standard errors the sizes give), the three classical estimates of each group (alone, one common mean, and the
 * hierarchical posterior) and the NUTS run behind the third, recorded as timestamped draws and as the group intervals
 * at positions along the run. The model is the library's eight schools, in both parametrisations: a group of n
 * observations with a known spread has the mean as a sufficient statistic with standard error sd / sqrt(n), which is
 * exactly the schools' `y_j` and `sigma_j`, so the benchmark models are used as they are. A worker runs `runPooling`
 * (`worker.ts`); the tests and a page without Workers run it on the main thread. The library has no clock, so the
 * caller gives one.
 */

import { posteriorChains } from 'omai-engine/probprog'
import { nutsMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { eightSchoolsCentred, eightSchoolsNonCentred } from 'omai-engine/probprog/models'
import { normal as drawNormal, foldIn, key } from 'omai-engine/random'
import { quantile } from 'omai-engine/stats'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { JobHost } from '../../../shared/workerPool.ts'

/** The two parametrisations of the hierarchy. */
export const FORMS = ['centred', 'non-centred'] as const
export type Form = (typeof FORMS)[number]

/** What the viewer sets. */
export interface Settings {
  readonly groups: number
  /** The size of the smallest group; the largest has `SIZE_RATIO` times as many. */
  readonly smallest: number
  /** The spread of the group effects around their common mean (the generating tau). */
  readonly tau: number
  readonly seed: number
}

/** What generated the data besides the spread: the mean of the group effects and the spread of one observation in a group. */
export const GENERATING = { mu: 1.5, sd: 1 } as const
/** The largest group is this many times the smallest. */
export const SIZE_RATIO = 20

/** The groups: sizes, observed means with their standard errors, and the effects the means were drawn around. */
export interface GroupData {
  readonly settings: Settings
  readonly size: readonly number[]
  readonly mean: readonly number[]
  readonly se: readonly number[]
  readonly truth: readonly number[]
}

/**
 * Groups of sizes spread geometrically from `smallest` to `SIZE_RATIO` times it, each with an effect drawn around the
 * common mean and a mean observed around that effect with the standard error its size gives.
 *
 * @param settings - The number of groups, the smallest size and the seed.
 * @returns The data.
 */
export function makeData(settings: Settings): GroupData {
  const { groups, smallest, tau, seed } = settings
  const k = key(seed)
  const z = toFlat(drawNormal(foldIn(k, 0), [groups]))
  const e = toFlat(drawNormal(foldIn(k, 1), [groups]))
  const size = Array.from({ length: groups }, (_, j) =>
    Math.round(smallest * SIZE_RATIO ** (groups === 1 ? 0 : j / (groups - 1))),
  )
  const se = size.map((n) => GENERATING.sd / Math.sqrt(n))
  const truth = Array.from(z, (v) => GENERATING.mu + tau * v)
  return { settings, size, se, truth, mean: truth.map((t, j) => t + (se[j] ?? 1) * (e[j] ?? 0)) }
}

/** An estimate with a 90% interval. */
export interface Estimate {
  readonly mean: number
  readonly lower: number
  readonly upper: number
}

const Z90 = 1.6449

/**
 * The two estimates that need no sampler: each group by its own mean (no pooling) and every group by the precision
 * weighted grand mean (complete pooling).
 *
 * @param data - The groups.
 * @returns The estimate of each group alone, and the one common estimate.
 */
export function classicalEstimates(data: GroupData): { alone: Estimate[]; common: Estimate } {
  const weight = data.se.map((s) => 1 / (s * s))
  const total = weight.reduce((a, b) => a + b, 0)
  const mean = weight.reduce((a, w, j) => a + w * (data.mean[j] ?? 0), 0) / total
  const half = Z90 / Math.sqrt(total)
  return {
    alone: data.mean.map((m, j) => ({
      mean: m,
      lower: m - Z90 * (data.se[j] ?? 0),
      upper: m + Z90 * (data.se[j] ?? 0),
    })),
    common: { mean, lower: mean - half, upper: mean + half },
  }
}

/** One run of NUTS: the form, the data, the chains and their length. */
export interface PoolingRequest {
  readonly method: Form
  readonly data: GroupData
  readonly seed: number
  readonly chains: number
  readonly warmup: number
  readonly draws: number
}

/** The hierarchical estimates of every group from the draws up to `step`. */
export interface PoolingFrame {
  readonly step: number
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
}

/** What a run records: every draw with the iteration it was made at, and the group intervals along the run. */
export interface PoolingResult {
  readonly method: Form
  /** Steps run, the start and the warm-up included (all of them unless the run was stopped). */
  readonly steps: number
  readonly elapsedMs: number
  /** The steps before the first draw: the start of the chains and their warm-up. */
  readonly warmup: number
  readonly chains: number
  /** Draws per chain. */
  readonly drawsPerChain: number
  /** Per draw (chains one after another): the iteration it was made at, the chain, whether it diverged. */
  readonly at: readonly number[]
  readonly chain: readonly number[]
  readonly divergent: readonly number[]
  /** Per draw: mu, log tau, and the effect of each group (`theta[j][draw]`). */
  readonly mu: readonly number[]
  readonly logTau: readonly number[]
  readonly theta: readonly (readonly number[])[]
  readonly frames: readonly PoolingFrame[]
}

/** The most positions of the run the group intervals are made at. */
export const MAX_FRAMES = 60

/**
 * Runs NUTS on one form of the hierarchy and records it.
 *
 * @param request - The form, the data and the size of the run.
 * @param host - The clock, the hand-over, the stop flag and the status line.
 * @returns The recorded run (what had been drawn, if it was stopped).
 */
export async function runPooling(request: PoolingRequest, host: JobHost): Promise<PoolingResult> {
  const { data, chains, warmup, draws } = request
  const model = request.method === 'centred' ? eightSchoolsCentred : eightSchoolsNonCentred
  const method = nutsMethod(model.bound({ data: { y: data.mean, sigma: data.se }, generating: {} }))
  // One step to start the chains, `warmup` of adaptation, then a draw per chain per step.
  const lead = warmup + 1
  const total = lead + draws
  const config = { numChains: chains, numWarmup: warmup, numDraws: draws, targetAcceptance: 0.8, maxTreeDepth: 6 }
  const s = session(method, { config, seed: request.seed, steps: total, mode: 'live' })
  const t0 = host.now()
  const run = await runAnytimeAsync(s, {
    budget: { steps: total },
    clock: host.now,
    sliceMs: 40,
    yieldToPage: host.yieldToPage,
    shouldStop: () => {
      host.status({ type: 'status', phase: 'running', step: s.t, elapsedMs: host.now() - t0 })
      return host.stopped()
    },
  })
  host.status({ type: 'status', phase: 'finishing', step: s.t, elapsedMs: host.now() - t0 })
  const posterior = method.posterior?.(s.state as never, s.config as never)
  const c = posterior === undefined ? { draws: {} } : posteriorChains(posterior, { sites: ['mu', 'tau', 'theta'] })
  const mu = c.draws['mu'] ?? []
  const perChain = mu[0]?.length ?? 0
  const flat = (rows: readonly (readonly number[])[] | undefined): number[] => (rows ?? []).flatMap((r) => [...r])
  const groups = data.size.length
  const theta = Array.from({ length: groups }, (_, j) => flat(c.draws[`theta[${j}]`]))
  const at = Array.from({ length: mu.length * perChain }, (_, k) => lead + (k % perChain) + 1)
  const frames: PoolingFrame[] = []
  const first = lead + 2
  const last = lead + perChain
  for (let f = 0; perChain >= 2 && f < MAX_FRAMES; f++) {
    const step = Math.round(first + ((last - first) * f) / (MAX_FRAMES - 1))
    if (f > 0 && step === frames[frames.length - 1]?.step) continue
    const rows = at.flatMap((a, i) => (a <= step ? [i] : []))
    const bounds = theta.map((col) =>
      toFlat(
        quantile(
          rows.map((i) => col[i] ?? 0),
          [0.05, 0.5, 0.95],
        ),
      ),
    )
    frames.push({
      step,
      mean: theta.map((col) => rows.reduce((a, i) => a + (col[i] ?? 0), 0) / rows.length),
      lower: bounds.map((b) => b[0] ?? Number.NaN),
      upper: bounds.map((b) => b[2] ?? Number.NaN),
    })
  }
  return {
    method: request.method,
    steps: run.t,
    elapsedMs: run.elapsedMs,
    warmup: lead,
    chains,
    drawsPerChain: perChain,
    at,
    chain: Array.from({ length: mu.length * perChain }, (_, k) => Math.floor(k / Math.max(1, perChain))),
    divergent: flat(c.divergent).map((v) => (v > 0 ? 1 : 0)),
    mu: flat(mu),
    logTau: flat(c.draws['tau']).map((v) => Math.log(Math.max(v, 1e-300))),
    theta,
    frames,
  }
}

/**
 * The frame of the run at an iteration: the last one made by then.
 *
 * @param result - The run.
 * @param step - The playhead, in iterations.
 * @returns The frame, or undefined while there are no draws yet.
 */
export function frameAt(result: PoolingResult, step: number): PoolingFrame | undefined {
  let found: PoolingFrame | undefined
  for (const f of result.frames) if (f.step <= step) found = f
  return found
}

/**
 * How many draws had been made by an iteration and how many of them diverged.
 *
 * @param result - The run.
 * @param step - The playhead, in iterations.
 * @returns The counts.
 */
export function countsAt(result: PoolingResult, step: number): { draws: number; divergent: number } {
  let draws = 0
  let divergent = 0
  result.at.forEach((a, i) => {
    if (a > step) return
    draws++
    divergent += result.divergent[i] ?? 0
  })
  return { draws, divergent }
}
