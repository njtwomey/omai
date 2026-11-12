/**
 * The mixture page's computing, free of any thread: the Old Faithful waiting times (standardised), the library's
 * K-component Gaussian mixture (`gaussianMixtureModel`, with the means ordered or free) bound to them, one inference method
 * run for a time budget, and its draws recorded as plain data with the time each was available: a sampler's chains laid
 * out by draw, a variational fit redrawn at each snapshot. Every draw is a row of the means, the weights and the spread in
 * minutes. A worker runs this (`worker.ts`); the test and a page without Workers run it on the main thread.
 */

import { oldFaithful } from 'omai-engine/datasets'
import { type Posterior, posteriorChains } from 'omai-engine/probprog'
import { inferenceMethodFor, modelRegistry, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { gaussianMixtureModel } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { JobHost } from '../../../shared/workerPool.ts'

/** How a method's draws are organised: chains that grow, or an approximation redrawn at each snapshot. */
export type Organisation = 'chain' | 'iterate'

/** A method on offer. */
export interface MixtureMethod {
  readonly id: string
  readonly label: string
  /** A name of `methodKinds` in `omai-engine/probprog/inference`. */
  readonly kind: string
  readonly organisation: Organisation
  /** The knobs for a mixture of `components` normals. */
  readonly config: (components: number) => Readonly<Record<string, string | number | boolean>>
  readonly note: string
}

/** The methods that run on the mixture (those with discrete latents summed out or sampled: the tempered ones refuse it). */
export const METHODS: readonly MixtureMethod[] = [
  {
    id: 'nuts',
    label: 'NUTS',
    kind: 'nuts',
    organisation: 'chain',
    config: () => ({ numChains: 4, numWarmup: 120, numDraws: 600, algorithm: 'nuts', maxTreeDepth: 7 }),
    note: 'four chains, the labels summed out',
  },
  {
    id: 'gibbs',
    label: 'Gibbs',
    kind: 'gibbs',
    organisation: 'chain',
    config: () => ({ numWarmup: 100, numDraws: 100_000 }),
    note: 'four chains of sweeps, the labels sampled',
  },
  {
    id: 'svi',
    label: 'SVI mean-field',
    kind: 'svi',
    organisation: 'iterate',
    config: () => ({ guide: 'autoNormal', learningRate: 0.02 }),
    note: 'one normal per coordinate: it cannot hold two modes',
  },
  {
    id: 'svi-mixture',
    label: 'SVI mixture guide',
    kind: 'svi-mixture',
    organisation: 'iterate',
    config: (k) => ({ components: k === 2 ? 2 : 6 }),
    note: 'a mixture of normals as the guide: it can sit on several modes',
  },
]

/** The Gibbs chains and the knobs a request does not set. */
const GIBBS_CHAINS = 4
/** The most draws kept of a sampler, over all its chains. */
const MAX_POOL = 800
/** The draws taken from a variational fit at each snapshot, and the most snapshots kept. */
const FIT_DRAWS = 60
const MAX_FRAMES = 60

/** One job: a method on a mixture of `components`, with the means ordered or not, for a budget. `method` names the job. */
export interface MixtureRequest {
  /** The job's id: the method and the labelling. */
  readonly method: string
  readonly kind: string
  readonly ordered: boolean
  readonly components: number
  readonly budgetMs: number
  readonly seed: number
}

/** The id of a job. */
export const jobId = (kind: string, ordered: boolean): string => `${kind}-${ordered ? 'ordered' : 'free'}`

/** The draws of a run: row `i` is `values[i * p .. (i + 1) * p]`, available from `at[i]` until `until[i]` (milliseconds). */
export interface Pool {
  readonly p: number
  readonly values: Float64Array
  readonly at: Float64Array
  readonly until: Float64Array
  readonly chain: Int32Array
}

/** One coordinate of the final posterior. */
export interface SummaryRow {
  readonly name: string
  readonly mean: number
  readonly sd: number
}

/** What a job returns. */
export interface MixtureResult {
  readonly method: string
  readonly label: string
  readonly kind: string
  readonly ordered: boolean
  readonly organisation: Organisation
  readonly components: number
  readonly steps: number
  readonly elapsedMs: number
  /** `budget`, `completed`, `stopped` or `failed`. */
  readonly ending: string
  readonly message: string | undefined
  /** The names of a row's coordinates: the means, the weights, the spread. */
  readonly names: readonly string[]
  readonly pool: Pool
  /** The time of each snapshot of a variational fit (empty for a sampler). */
  readonly frames: readonly number[]
  readonly chains: number
}

/** The waiting times of the 272 Old Faithful eruptions (minutes), and their mean and standard deviation. */
export function waitingTimes(): { readonly minutes: number[]; readonly mean: number; readonly sd: number } {
  const x = toFlat(oldFaithful().data.x)
  const minutes = Array.from({ length: x.length / 2 }, (_, i) => x[2 * i + 1] as number)
  const mean = minutes.reduce((a, b) => a + b, 0) / minutes.length
  const sd = Math.sqrt(minutes.reduce((a, b) => a + (b - mean) ** 2, 0) / minutes.length)
  return { minutes, mean, sd }
}

/** The names of a row: the means, the weights, the shared spread. */
export const namesOf = (components: number): string[] => [
  ...Array.from({ length: components }, (_, k) => `mean ${k + 1}`),
  ...Array.from({ length: components }, (_, k) => `weight ${k + 1}`),
  'spread',
]

/** `max` indices spread evenly over `count` items, the first and the last included. */
const evenly = (count: number, max: number): number[] =>
  count <= max
    ? Array.from({ length: count }, (_, i) => i)
    : Array.from({ length: max }, (_, i) => Math.round((i * (count - 1)) / (max - 1)))

interface Draws {
  /** `[coordinate][chain][draw]`, in the model's standardised units. */
  readonly series: number[][][]
}

/** The draws of a posterior by coordinate and chain: Markov chains when it has them, else one chain of its draws. */
function drawsOf(post: Posterior, components: number): Draws | undefined {
  const sites = ['mu', 'weights', 'sigma']
  if (post.chains !== undefined) {
    const { draws } = posteriorChains(post, { sites })
    const take = (name: string): number[][] | undefined => draws[name]
    const series = [
      ...Array.from({ length: components }, (_, k) => take(`mu[${k}]`)),
      ...Array.from({ length: components }, (_, k) => take(`weights[${k}]`)),
      take('sigma'),
    ]
    return series.every((s) => s !== undefined) ? { series: series as number[][][] } : undefined
  }
  const d = post.draws
  if (d === undefined || d['mu'] === undefined) return undefined
  const column = (site: string, size: number, j: number): number[] => {
    const flat = toFlat(d[site] as never)
    return Array.from({ length: Math.floor(flat.length / size) }, (_, i) => flat[i * size + j] as number)
  }
  const series = [
    ...Array.from({ length: components }, (_, k) => [column('mu', components, k)]),
    ...Array.from({ length: components }, (_, k) => [column('weights', components, k)]),
    [column('sigma', 1, 0)],
  ]
  return { series }
}

/** Standardised units back to minutes: the means and the spread scale, the weights stay. */
const toMinutes = (j: number, components: number, v: number, mean: number, sd: number): number =>
  j < components ? v * sd + mean : j < 2 * components ? v : v * sd

/**
 * Runs one job and returns everything the page draws.
 *
 * @param request - The method, whether the means are ordered, the number of components, the budget and the seed.
 * @param host - The clock, the hand-over, the stop flag and the status line.
 * @returns The draws with the time each was available.
 */
export async function runMixture(request: MixtureRequest, host: JobHost): Promise<MixtureResult> {
  const spec = METHODS.find((m) => m.kind === request.kind) as MixtureMethod
  const { components } = request
  const { minutes, mean, sd } = waitingTimes()
  const y = minutes.map((v) => (v - mean) / sd)
  const registry = modelRegistry({
    mixture: { definition: gaussianMixtureModel({ components, ordered: request.ordered }), data: { y } },
  })
  const method = inferenceMethodFor(registry, spec.kind, 'mixture')
  const config = { ...spec.config(components) }
  const names = namesOf(components)
  const p = names.length
  const t0 = host.now()
  const elapsed = (): number => host.now() - t0
  const base = {
    method: request.method,
    label: `${spec.label}, ${request.ordered ? 'ordered' : 'unordered'}`,
    kind: spec.kind,
    ordered: request.ordered,
    organisation: spec.organisation,
    components,
    names,
  }
  const status = (step: number, phase: 'running' | 'finishing'): void =>
    host.status({ type: 'status', phase, step, elapsedMs: elapsed() })
  const sessionOf = (seed: number) =>
    session(method, { config: config as never, seed, steps: 1_000_000_000, mode: 'live' })

  if (spec.organisation === 'iterate') {
    const s = sessionOf(request.seed)
    const snapshots: { readonly step: number; readonly ms: number; readonly post: Posterior | undefined }[] = []
    const run = await runAnytimeAsync(s, {
      budget: { ms: request.budgetMs },
      snapshot: { everyMs: request.budgetMs / (2 * MAX_FRAMES) },
      clock: host.now,
      sliceMs: 40,
      yieldToPage: host.yieldToPage,
      shouldStop: () => {
        status(s.t, 'running')
        return host.stopped()
      },
      onSnapshot: (x) => {
        let post: Posterior | undefined
        try {
          post = method.posterior?.(x.state as never, s.config as never)
        } catch {
          post = undefined
        }
        snapshots.push({ step: x.t, ms: x.elapsedMs, post })
      },
    })
    status(run.t, 'finishing')
    const usable = snapshots.filter((x) => x.post?.sites.includes('mu') === true)
    const kept = evenly(usable.length, MAX_FRAMES).map((i) => usable[i] as (typeof usable)[number])
    const total = kept.length * FIT_DRAWS
    const values = new Float64Array(total * p)
    const at = new Float64Array(total)
    const until = new Float64Array(total)
    for (const [f, x] of kept.entries()) {
      const drawn = (x.post as Posterior).sample(key(1000 + f), FIT_DRAWS)
      const mu = toFlat(drawn['mu'] as never)
      const w = toFlat(drawn['weights'] as never)
      const sigma = toFlat(drawn['sigma'] as never)
      for (let i = 0; i < FIT_DRAWS; i++) {
        const row = f * FIT_DRAWS + i
        for (let j = 0; j < p; j++) {
          const raw =
            j < components ? mu[i * components + j] : j < 2 * components ? w[i * components + j - components] : sigma[i]
          values[row * p + j] = toMinutes(j, components, raw ?? Number.NaN, mean, sd)
        }
        at[row] = x.ms
        until[row] = kept[f + 1]?.ms ?? Number.POSITIVE_INFINITY
      }
    }
    return {
      ...base,
      steps: run.t,
      elapsedMs: run.elapsedMs,
      ending:
        usable.length === 0
          ? 'failed'
          : run.reason === 'cancelled'
            ? 'stopped'
            : run.reason === 'budget'
              ? 'budget'
              : 'completed',
      message: usable.length === 0 ? 'no snapshot of the fit was usable' : undefined,
      pool: { p, values, at, until, chain: new Int32Array(total) },
      frames: kept.map((x) => x.ms),
      chains: 1,
    }
  }

  // A sampler: its chains, with the number of draws each had made at each snapshot, so each draw can be given its time.
  const chainCount = spec.kind === 'gibbs' ? GIBBS_CHAINS : (config['numChains'] as number)
  const sessions =
    spec.kind === 'gibbs'
      ? Array.from({ length: chainCount }, (_, c) => sessionOf(request.seed + 1000 * c))
      : [sessionOf(request.seed)]
  const marks: { readonly ms: number; readonly count: number }[] = [{ ms: 0, count: 0 }]
  const counted = (): number =>
    spec.kind === 'gibbs'
      ? Math.min(...sessions.map((s) => (s.state as { kept: unknown[] }).kept.length))
      : Math.min(
          ...((sessions[0] as (typeof sessions)[number]).state as { chains: { kept: unknown[] }[] }).chains.map(
            (c) => c.kept.length,
          ),
        )
  let steps = 0
  let ending = 'budget'
  if (spec.kind === 'gibbs') {
    // The sweeps of the chains are interleaved so that they run side by side, as the chains of NUTS do.
    for (;;) {
      if (host.stopped()) {
        ending = 'stopped'
        break
      }
      if (elapsed() >= request.budgetMs) break
      for (const s of sessions) s.advance({ steps: 3 })
      steps += 3
      marks.push({ ms: elapsed(), count: counted() })
      status(steps, 'running')
      if (steps % 30 === 0) await host.yieldToPage()
    }
  } else {
    const s = sessions[0] as (typeof sessions)[number]
    const run = await runAnytimeAsync(s, {
      budget: { ms: request.budgetMs },
      snapshot: { everyMs: request.budgetMs / 100 },
      clock: host.now,
      sliceMs: 40,
      yieldToPage: host.yieldToPage,
      shouldStop: () => {
        status(s.t, 'running')
        return host.stopped()
      },
      onSnapshot: (x) => {
        marks.push({ ms: x.elapsedMs, count: counted() })
      },
    })
    steps = run.t
    ending = run.reason === 'cancelled' ? 'stopped' : run.reason === 'budget' ? 'budget' : 'completed'
  }
  status(steps, 'finishing')
  marks.push({ ms: elapsed(), count: counted() })
  const empty = (message: string): MixtureResult => ({
    ...base,
    steps,
    elapsedMs: elapsed(),
    ending,
    message,
    pool: {
      p,
      values: new Float64Array(0),
      at: new Float64Array(0),
      until: new Float64Array(0),
      chain: new Int32Array(0),
    },
    frames: [],
    chains: chainCount,
  })
  // Each chain's draws: the posterior of each Gibbs session (one chain), or the one posterior of NUTS (all of its chains).
  const perChain: number[][][] = []
  for (const s of sessions) {
    let post: Posterior | undefined
    try {
      post = method.posterior?.(s.state as never, s.config as never)
    } catch {
      post = undefined
    }
    const d = post === undefined ? undefined : drawsOf(post, components)
    if (d === undefined) continue
    for (let c = 0; c < (d.series[0]?.length ?? 0); c++)
      perChain.push(d.series.map((coordinate) => coordinate[c] as number[]))
  }
  const numDraws = Math.min(...perChain.map((c) => c[0]?.length ?? 0))
  if (perChain.length === 0 || !(numDraws >= 1))
    return empty('no draws yet: the budget ended during warm-up; a longer budget or fewer components')
  const timeOf = (i: number): number => {
    // The draw `i` of a chain exists from the first snapshot that counts more than `i` draws: interpolated between the two around it.
    for (let m = 1; m < marks.length; m++) {
      const hi = marks[m] as (typeof marks)[number]
      const lo = marks[m - 1] as (typeof marks)[number]
      if (hi.count >= i + 1)
        return hi.count === lo.count ? hi.ms : lo.ms + ((hi.ms - lo.ms) * (i + 1 - lo.count)) / (hi.count - lo.count)
    }
    return elapsed()
  }
  const keep = evenly(numDraws, Math.max(2, Math.floor(MAX_POOL / perChain.length)))
  const total = keep.length * perChain.length
  const values = new Float64Array(total * p)
  const at = new Float64Array(total)
  const chain = new Int32Array(total)
  let row = 0
  for (const [c, draws] of perChain.entries())
    for (const i of keep) {
      for (let j = 0; j < p; j++) values[row * p + j] = toMinutes(j, components, draws[j]?.[i] ?? Number.NaN, mean, sd)
      at[row] = timeOf(i)
      chain[row] = c
      row++
    }
  return {
    ...base,
    steps,
    elapsedMs: elapsed(),
    ending,
    message: undefined,
    pool: { p, values, at, until: new Float64Array(total).fill(Number.POSITIVE_INFINITY), chain },
    frames: [],
    chains: perChain.length,
  }
}
