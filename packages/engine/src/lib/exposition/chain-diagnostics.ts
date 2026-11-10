/**
 * The exposition record of the diagnostics of Markov chains (`chainDiagnostics`): for each quantity (a site, or a
 * coordinate of a vector site), the draws of every chain for trace plots, the rank histograms of each chain for rank
 * plots (uniform when the chains mix), the split R-hat and effective sample size where the method reports them, and the
 * divergent transitions. `chainsFromTape` reads the draws of a sampler's tape (`mcmc/position`, `mcmc/divergent`, with
 * `mcmc/stage` to leave out the warm-up) and `chainDiagnostics` makes the record from draws and the rows of a
 * posterior summary (`<method>/snapshot`'s `summary`), so a run in progress has its charts from the draws so far.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'

/** One quantity: its draws by chain, what the method reports of them, and the rank histogram of each chain. */
export interface ChainQuantity {
  /** The quantity's name: a site (`mu`) or a coordinate (`w[2]`). */
  readonly name: string
  /** The site it belongs to (`w` for `w[2]`). */
  readonly site: string
  /** The draws: `chains[c][i]` is the `i`-th draw of chain `c` (after warm-up, thinned to the record's limit). */
  readonly chains: readonly (readonly number[])[]
  readonly mean: number
  readonly sd: number
  /** Rank-normalised split R-hat and bulk effective sample size, from the method's summary when it gave them. */
  readonly rhat?: number
  readonly ess?: number
  /** `counts[c][b]`: how many draws of chain `c` have a pooled rank in the `b`-th of `bins` equal parts; flat when the chains agree. */
  readonly ranks: { readonly bins: number; readonly counts: readonly (readonly number[])[] }
}

/** Diagnostics of the chains of a sampler. */
export interface ChainDiagnostics {
  readonly kind: 'chainDiagnostics'
  readonly quantities: readonly ChainQuantity[]
  /** Divergent transitions: the chain and the draw (index into `chains[c]`). */
  readonly divergences: readonly { readonly chain: number; readonly draw: number }[]
}

/**
 * The rank histogram of each chain: the draws of all chains pooled and ranked (ties by their mean rank), the ranks cut
 * into `bins` equal parts, and each chain's draws counted per part. A chain that sits low or high, or wanders, shows as
 * a slope or a hump; chains that explore the same distribution are flat (Vehtari et al. 2021).
 *
 * @param chains - The draws by chain.
 * @param bins - The number of parts (default 20).
 * @returns The counts per chain and part.
 * @example
 * ```ts
 * import { rankCounts } from 'omai-engine/exposition'
 * rankCounts([[1, 3, 5, 7], [2, 4, 6, 8]], 2) // => [[2, 2], [2, 2]]
 * rankCounts([[1, 2, 3, 4], [5, 6, 7, 8]], 2) // => [[4, 0], [0, 4]]
 * ```
 */
export function rankCounts(chains: readonly (readonly number[])[], bins = 20): number[][] {
  const all = chains.flatMap((c, ci) => c.map((v, i) => ({ v, ci, i })))
  const order = all.map((_, k) => k).sort((a, b) => (all[a]?.v ?? 0) - (all[b]?.v ?? 0))
  const rank = new Array<number>(all.length).fill(0)
  for (let k = 0; k < order.length; ) {
    let end = k
    while (end + 1 < order.length && all[order[end + 1] as number]?.v === all[order[k] as number]?.v) end++
    for (let j = k; j <= end; j++) rank[order[j] as number] = (k + end) / 2
    k = end + 1
  }
  const counts = chains.map(() => new Array<number>(bins).fill(0))
  for (const [k, d] of all.entries()) {
    const b = Math.min(bins - 1, Math.floor((((rank[k] as number) + 0.5) / Math.max(1, all.length)) * bins))
    ;(counts[d.ci] as number[])[b] = ((counts[d.ci] as number[])[b] ?? 0) + 1
  }
  return counts
}

/** Options of {@link chainDiagnostics}. */
export interface ChainDiagnosticsOptions {
  /** The rows of a posterior summary (`site`, `name`, `rhat`, `ess`) to take R-hat and ESS from. */
  readonly summary?: readonly unknown[]
  /** Divergent transitions: `divergent[c][i]` is 1 where draw `i` of chain `c` diverged. */
  readonly divergent?: readonly (readonly number[])[]
  /** Parts of the rank histograms (default 20). */
  readonly bins?: number
  /** The most draws per chain kept for the trace plots (default 500, thinned evenly); the ranks use them all. */
  readonly maxDraws?: number
}

const siteOf = (name: string): string => name.replace(/\[.*\]$/, '')

/**
 * The diagnostics record of the chains of a sampler, from the draws of each quantity by chain.
 *
 * @param draws - Quantity name to its draws by chain (`draws['mu'][c][i]`).
 * @param options - A summary to take R-hat and ESS from, the divergent transitions, the number of bins and the trace length.
 * @returns The record.
 * @throws OmaiError `shape` when chains of one quantity differ in length or a quantity has no draws.
 * @index function Trace, rank and R-hat/ESS data of Markov chains, from draws and a posterior summary.
 * @example
 * ```ts
 * import { chainDiagnostics } from 'omai-engine/exposition'
 * const r = chainDiagnostics({ mu: [[1, 2, 3, 4], [2, 3, 4, 5]] }, { bins: 2, summary: [{ site: 'mu', name: 'mu', rhat: 1.01, ess: 300 }] })
 * [r.quantities[0]?.rhat, r.quantities[0]?.mean] // => [1.01, 3]
 * ```
 */
export function chainDiagnostics(
  draws: Readonly<Record<string, readonly (readonly number[])[]>>,
  options: ChainDiagnosticsOptions = {},
): ChainDiagnostics {
  const bins = options.bins ?? 20
  const maxDraws = options.maxDraws ?? 500
  const rows = new Map<string, { rhat?: number; ess?: number }>()
  for (const r of options.summary ?? []) {
    if (typeof r !== 'object' || r === null) continue
    const x = r as Record<string, unknown>
    if (typeof x['name'] !== 'string') continue
    rows.set(x['name'], {
      ...(typeof x['rhat'] === 'number' && Number.isFinite(x['rhat']) ? { rhat: x['rhat'] } : {}),
      ...(typeof x['ess'] === 'number' && Number.isFinite(x['ess']) ? { ess: x['ess'] } : {}),
    })
  }
  const quantities = Object.entries(draws).map(([name, chains]): ChainQuantity => {
    const n = chains[0]?.length ?? 0
    if (chains.length === 0 || n === 0) throw new OmaiError('shape', `chainDiagnostics: ${name} has no draws`)
    for (const [c, row] of chains.entries())
      if (row.length !== n)
        throw new OmaiError('shape', `chainDiagnostics: chain ${c} of ${name} has ${row.length} draws, expected ${n}`)
    const flat = chains.flat()
    const mean = flat.reduce((s, v) => s + v, 0) / flat.length
    const sd = Math.sqrt(flat.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, flat.length - 1))
    const thin = (row: readonly number[]): number[] =>
      row.length <= maxDraws
        ? [...row]
        : Array.from({ length: maxDraws }, (_, i) => row[Math.floor((i * (row.length - 1)) / (maxDraws - 1))] as number)
    return {
      name,
      site: siteOf(name),
      chains: chains.map(thin),
      mean,
      sd,
      ...rows.get(name),
      ranks: { bins, counts: rankCounts(chains, bins) },
    }
  })
  const divergences = (options.divergent ?? []).flatMap((row, chain) =>
    row.flatMap((v, draw) => (v > 0 ? [{ chain, draw }] : [])),
  )
  return seal({ kind: 'chainDiagnostics', quantities, divergences })
}

/** A tape event, as much of it as the readers need (a `Run`'s events are these). */
interface TensorEventLike {
  readonly type: string
  readonly config?: unknown
  readonly step?: number
  readonly name?: string
  readonly value?: number
  readonly shape?: readonly number[]
  readonly values?: readonly number[]
}

/** Options of {@link chainsFromTape}. */
export interface ChainsFromTapeOptions {
  /** The tensor path holding every chain's position, `chains x dim` (default `mcmc/position`). */
  readonly position?: string
  /** The path of the divergence flags, `chains` (default `mcmc/divergent`). */
  readonly divergent?: string
  /** The path of the stage scalar: draws while it is 2 are kept, the warm-up left out (default `mcmc/stage`); a tape without it keeps all. */
  readonly stage?: string
  /**
   * The number of chains, for a position tensor whose shape does not say (a one-dimensional model's is `[chains]`, not
   * `[chains, 1]`). Default the run's own `numChains` knob, else the first axis of a two-dimensional shape, else one.
   */
  readonly chains?: number
  /** Names of the coordinates (default `<last part of the path>[j]`). */
  readonly names?: readonly string[]
}

/**
 * The draws of a sampler's tape: every chain's position at each step as `chains x dim` tensors, regrouped by coordinate
 * and chain, the warm-up steps (stage 1) left out. A tape in progress gives the draws so far. The positions are on the
 * unconstrained scale the sampler moves on; the constrained draws of a site come from the posterior (`Posterior.draws`),
 * which `chainDiagnostics` takes as well.
 *
 * @param run - The tape (finished or not).
 * @param options - The paths and the coordinates' names.
 * @returns The draws by coordinate and chain, and the divergence flags by chain and draw.
 * @example
 * ```ts
 * import { chainsFromTape } from 'omai-engine/exposition'
 * const t = (step: number, a: number, b: number) => ({ type: 'tensor', step, name: 'mcmc/position', shape: [2, 1], values: [a, b] })
 * chainsFromTape({ events: [t(1, 0, 10), t(2, 1, 11)] }).draws // => { 'position[0]': [[0, 1], [10, 11]] }
 * ```
 */
export function chainsFromTape(
  run: { readonly events: readonly TensorEventLike[] },
  options: ChainsFromTapeOptions = {},
): { readonly draws: Record<string, number[][]>; readonly divergent: number[][] } {
  const position = options.position ?? 'mcmc/position'
  const divergentPath = options.divergent ?? 'mcmc/divergent'
  const stagePath = options.stage ?? 'mcmc/stage'
  const stage = new Map<number, number>()
  for (const e of run.events)
    if (e.type === 'metric' && e.name === stagePath && e.value !== undefined) stage.set(e.step ?? 0, e.value)
  const kept = (step: number): boolean => stage.size === 0 || stage.get(step) === 2
  const configured = run.events.find((e) => e.type === 'run')?.config
  const knob =
    typeof configured === 'object' && configured !== null
      ? (configured as Record<string, unknown>)['numChains']
      : undefined
  const given = options.chains ?? (typeof knob === 'number' ? knob : undefined)
  const base = position.split('/').at(-1) ?? position
  const draws: Record<string, number[][]> = {}
  const flags: number[][] = []
  for (const e of run.events) {
    if (e.type !== 'tensor' || e.values === undefined || !kept(e.step ?? 0)) continue
    const shape = e.shape ?? []
    if (e.name === position) {
      const chains = shape.length === 2 ? (shape[0] as number) : (given ?? 1)
      const dim = e.values.length / chains
      for (let j = 0; j < dim; j++) {
        const name = options.names?.[j] ?? `${base}[${j}]`
        const rows = draws[name] ?? Array.from({ length: chains }, (): number[] => [])
        draws[name] = rows
        for (let c = 0; c < chains; c++) rows[c]?.push(e.values[c * dim + j] as number)
      }
    } else if (e.name === divergentPath) {
      for (const [c, v] of e.values.entries()) {
        const row = flags[c] ?? []
        row.push(v)
        flags[c] = row
      }
    }
  }
  return { draws, divergent: flags }
}
