/**
 * The exposition record of the diagnostics of Markov chains (`chainDiagnostics`): for each quantity (a site, or a
 * coordinate of a vector site), the draws of every chain for trace plots, the rank histograms of each chain for rank
 * plots (uniform when the chains mix), the split R-hat and effective sample size where the method reports them, and the
 * divergent transitions, and the data of the other standard charts: the ECDF of each chain's pooled ranks minus the
 * uniform CDF with the simultaneous band, the autocorrelation of each chain, the effective sample size by quantile and
 * as the draws accumulate (given, as `exposition` cannot compute it: `chainEssCurves` in `omai-engine/sampling`), and
 * the energy of the Hamiltonian with each chain's E-BFMI. `chainsFromTape` reads the draws of a sampler's tape
 * (`mcmc/position`, `mcmc/divergent`, `mcmc/energy`, with `mcmc/stage` to leave out the warm-up) and `chainDiagnostics`
 * makes the record from draws and the rows of a posterior summary (`<method>/snapshot`'s `summary`), so a run in
 * progress has its charts from the draws so far. The autocorrelation and E-BFMI are computed here, by the formulas of
 * `omai-engine/sampling` (`autocorrelation`, `bfmi`, which the tests compare them with), because `exposition` may not
 * import it.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'
import { ecdfBand, uniformGrid } from './ecdf-band.ts'
import { type CellMark, type MarkedTable, markedTable } from './tables.ts'

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
  /** The autocorrelation of each chain at lags 0..`lags`, and their mean over the chains (from all of the draws, not the thinned trace). */
  readonly acf?: {
    readonly lags: number
    readonly chains: readonly (readonly number[])[]
    readonly mean: readonly number[]
  }
  /** `ecdfDiff[c][j]`: the ECDF of chain `c`'s pooled ranks at the record's `ecdf.z[j]` minus `z[j]`; near zero, inside the band, when the chains agree. */
  readonly ecdfDiff?: readonly (readonly number[])[]
  /** The effective sample size by quantile and as the draws accumulate, when given. */
  readonly essCurves?: EssCurves
  /**
   * The value of the quantity at each divergent transition, in the order of the record's `divergences` (the first
   * {@link MAX_DIVERGENT_VALUES} of them): the points a trace or pair plot marks, whatever the thinning of `chains`.
   */
  readonly atDivergences?: readonly number[]
}

/** How many divergent transitions carry the value of each quantity in `atDivergences`. */
export const MAX_DIVERGENT_VALUES = 1000

/**
 * The effective sample size of a quantity as curves, ArviZ's `plot_ess` kinds `quantile` and `evolution`: the ESS of the
 * indicator of each quantile (where in the distribution the chains explore badly), and the bulk and tail ESS as more
 * draws are used (growing in line with the draws when the chains are healthy). `chainEssCurves` in `omai-engine/sampling` makes them.
 */
export interface EssCurves {
  readonly quantile: { readonly probs: readonly number[]; readonly ess: readonly number[] }
  /** `draws[i]` draws of each chain gave the ESS `bulk[i]` and `tail[i]`. */
  readonly evolution: {
    readonly draws: readonly number[]
    readonly bulk: readonly number[]
    readonly tail: readonly number[]
  }
}

/**
 * The energy of the Hamiltonian of the chains: the marginal energy distribution and the distribution of its change
 * between draws, as densities on common bins, and each chain's E-BFMI (Betancourt 2016). When the two distributions
 * differ a lot, or a chain's E-BFMI is under 0.3, the momentum resampling cannot explore the energy levels of the
 * target: reparametrise.
 */
export interface ChainEnergy {
  /** E-BFMI of each chain: the mean squared change of the energy over its variance. */
  readonly bfmi: readonly number[]
  /** The bin edges (`bins + 1` of them), and the densities of the energy minus its chain's mean and of its change between draws. */
  readonly edges: readonly number[]
  readonly marginal: readonly number[]
  readonly transition: readonly number[]
}

/** Diagnostics of the chains of a sampler. */
export interface ChainDiagnostics {
  readonly kind: 'chainDiagnostics'
  readonly quantities: readonly ChainQuantity[]
  /** Divergent transitions: the chain and the draw (index into `chains[c]`). */
  readonly divergences: readonly { readonly chain: number; readonly draw: number }[]
  /**
   * The grid of the ECDF-difference plots and the simultaneous band around zero (the ECDF minus the grid, with
   * probability `level` for every chain at every point when the chains sample the same distribution), made of `draws`
   * values per chain.
   */
  readonly ecdf?: {
    readonly z: readonly number[]
    readonly lower: readonly number[]
    readonly upper: readonly number[]
    readonly level: number
    readonly draws: number
  }
  readonly energy?: ChainEnergy
}

/**
 * The pooled rank of every draw: the draws of all chains ranked together (ties by their mean rank, from 0), kept by
 * chain and position.
 *
 * @param chains - The draws by chain.
 * @returns `ranks[c][i]` is the rank of draw `i` of chain `c` among all the draws.
 * @example
 * ```ts
 * import { pooledRanks } from 'omai-engine/exposition'
 * pooledRanks([[3, 1], [2, 2]]) // => [[3, 0], [1.5, 1.5]]
 * ```
 */
export function pooledRanks(chains: readonly (readonly number[])[]): number[][] {
  const all = chains.flatMap((c, ci) => c.map((v, i) => ({ v, ci, i })))
  const order = all.map((_, k) => k).sort((a, b) => (all[a]?.v ?? 0) - (all[b]?.v ?? 0))
  const rank = new Array<number>(all.length).fill(0)
  for (let k = 0; k < order.length; ) {
    let end = k
    while (end + 1 < order.length && all[order[end + 1] as number]?.v === all[order[k] as number]?.v) end++
    for (let j = k; j <= end; j++) rank[order[j] as number] = (k + end) / 2
    k = end + 1
  }
  const out = chains.map((c) => new Array<number>(c.length).fill(0))
  for (const [k, d] of all.entries()) (out[d.ci] as number[])[d.i] = rank[k] as number
  return out
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
  const total = Math.max(
    1,
    chains.reduce((s, c) => s + c.length, 0),
  )
  return pooledRanks(chains).map((ranks) => {
    const counts = new Array<number>(bins).fill(0)
    for (const r of ranks) {
      const b = Math.min(bins - 1, Math.floor(((r + 0.5) / total) * bins))
      counts[b] = (counts[b] ?? 0) + 1
    }
    return counts
  })
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
  /** The longest lag of the autocorrelation (default 40, at most half the draws); `0` leaves it out. */
  readonly acfLags?: number
  /** The points of the ECDF-difference grid (default 40); `0` leaves the ECDF out. */
  readonly ecdfPoints?: number
  /** The probability the simultaneous band of the ECDF difference holds (default 0.95). */
  readonly ecdfLevel?: number
  /** The most draws per chain the ECDF is made of (default 2000, thinned evenly), which bounds the cost of its band. */
  readonly ecdfDraws?: number
  /** The energy of each chain at each draw (`energy[c][i]`, the `mcmc/energy` of the tape): makes the record's `energy`. */
  readonly energy?: readonly (readonly number[])[]
  /** The ESS curves by quantity name, from `chainEssCurves` in `omai-engine/sampling`. */
  readonly essCurves?: Readonly<Record<string, EssCurves>>
}

/** The autocorrelation of draws at lags 0..maxLag, with the sum over the whole chain in the autocovariance (as `autocorrelation` of `omai-engine/sampling`); a constant chain is 1 at lag 0 and 0 after. */
function autocorrelationOf(x: readonly number[], maxLag: number): number[] {
  const n = x.length
  const mean = x.reduce((s, v) => s + v, 0) / n
  const centred = x.map((v) => v - mean)
  const lag = (k: number): number => {
    let acc = 0
    for (let t = 0; t + k < n; t++) acc += (centred[t] as number) * (centred[t + k] as number)
    return acc / n
  }
  const g0 = lag(0)
  return Array.from({ length: maxLag + 1 }, (_, k) => (k === 0 ? 1 : g0 === 0 ? 0 : lag(k) / g0))
}

/** The energy record: E-BFMI per chain and the two densities on common bins. */
function energyOf(energy: readonly (readonly number[])[], bins = 30): ChainEnergy {
  const marginal: number[] = []
  const transition: number[] = []
  const bfmi = energy.map((e): number => {
    const n = e.length
    const mean = e.reduce((s, v) => s + v, 0) / Math.max(1, n)
    let moved = 0
    let spread = 0
    for (let i = 0; i < n; i++) {
      marginal.push((e[i] as number) - mean)
      spread += ((e[i] as number) - mean) ** 2
      if (i > 0) {
        const d = (e[i] as number) - (e[i - 1] as number)
        transition.push(d)
        moved += d * d
      }
    }
    return n < 2 || spread === 0 ? Number.NaN : moved / (n - 1) / (spread / (n - 1))
  })
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of [...marginal, ...transition]) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!(hi > lo)) {
    lo -= 0.5
    hi += 0.5
  }
  const width = (hi - lo) / bins
  const density = (xs: readonly number[]): number[] => {
    const c = new Array<number>(bins).fill(0)
    for (const v of xs) {
      const b = Math.min(bins - 1, Math.floor((v - lo) / width))
      c[b] = (c[b] ?? 0) + 1
    }
    return c.map((k) => k / Math.max(1, xs.length) / width)
  }
  return {
    // A chain with no spread (a constant energy) has no E-BFMI: it is reported as zero, the worst value.
    bfmi: bfmi.map((v) => (Number.isFinite(v) ? v : 0)),
    edges: Array.from({ length: bins + 1 }, (_, b) => lo + b * width),
    marginal: density(marginal),
    transition: density(transition),
  }
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
  const divergences = (options.divergent ?? []).flatMap((row, chain) =>
    row.flatMap((v, draw) => (v > 0 ? [{ chain, draw }] : [])),
  )
  const acfLags = options.acfLags ?? 40
  const ecdfPoints = options.ecdfPoints ?? 40
  const ecdfDraws = options.ecdfDraws ?? 2000
  const grid = ecdfPoints > 0 ? uniformGrid(ecdfPoints) : undefined
  let ecdfDrawn = 0
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
    const maxLag = Math.min(acfLags, Math.floor(n / 2))
    const acf = maxLag >= 1 ? chains.map((c) => autocorrelationOf(c, maxLag)) : undefined
    let ecdfDiff: number[][] | undefined
    if (grid !== undefined && n >= 4) {
      const pooled = pooledRanks(chains)
      const total = n * chains.length
      const keep = Math.min(n, ecdfDraws)
      ecdfDiff = pooled.map((ranks) => {
        const u = Array.from(
          { length: keep },
          (_, i) => ((ranks[Math.round((i * (n - 1)) / Math.max(1, keep - 1))] as number) + 0.5) / total,
        )
        u.sort((a, b) => a - b)
        let at = 0
        return grid.map((z) => {
          while (at < u.length && (u[at] as number) <= z) at++
          return at / keep - z
        })
      })
      ecdfDrawn = keep
    }
    const curves = options.essCurves?.[name]
    const atDivergences = divergences
      .slice(0, MAX_DIVERGENT_VALUES)
      .map((d) => chains[d.chain]?.[d.draw])
      .filter((v): v is number => v !== undefined)
    return {
      name,
      site: siteOf(name),
      chains: chains.map(thin),
      mean,
      sd,
      ...rows.get(name),
      ranks: { bins, counts: rankCounts(chains, bins) },
      ...(acf === undefined
        ? {}
        : {
            acf: {
              lags: maxLag,
              chains: acf,
              mean: Array.from(
                { length: maxLag + 1 },
                (_, k) => acf.reduce((s, c) => s + (c[k] as number), 0) / acf.length,
              ),
            },
          }),
      ...(ecdfDiff === undefined ? {} : { ecdfDiff }),
      ...(curves === undefined ? {} : { essCurves: curves }),
      ...(atDivergences.length === 0 || atDivergences.length !== Math.min(divergences.length, MAX_DIVERGENT_VALUES)
        ? {}
        : { atDivergences }),
    }
  })
  // Every quantity has the same number of draws per chain, so one band serves them all.
  const band = grid !== undefined && ecdfDrawn > 0 ? ecdfBand(ecdfDrawn, grid, options.ecdfLevel ?? 0.95) : undefined
  return seal({
    kind: 'chainDiagnostics',
    quantities,
    divergences,
    ...(band === undefined
      ? {}
      : {
          ecdf: {
            z: band.z,
            lower: band.lower.map((v, j) => v - (band.z[j] as number)),
            upper: band.upper.map((v, j) => v - (band.z[j] as number)),
            level: band.level,
            draws: ecdfDrawn,
          },
        }),
    ...(options.energy === undefined || options.energy.length === 0 ? {} : { energy: energyOf(options.energy) }),
  })
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
  /** The path of the energy of each chain, `chains` (default `mcmc/energy`). */
  readonly energy?: string
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
 * @returns The draws by coordinate and chain, the divergence flags by chain and draw, and the energy by chain and draw.
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
): { readonly draws: Record<string, number[][]>; readonly divergent: number[][]; readonly energy: number[][] } {
  const position = options.position ?? 'mcmc/position'
  const divergentPath = options.divergent ?? 'mcmc/divergent'
  const energyPath = options.energy ?? 'mcmc/energy'
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
  const energies: number[][] = []
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
    } else if (e.name === energyPath) {
      for (const [c, v] of e.values.entries()) {
        const row = energies[c] ?? []
        row.push(v)
        energies[c] = row
      }
    }
  }
  return { draws, divergent: flags, energy: energies }
}

/**
 * The summary of chains as a table: one row per quantity with its mean, sd, R-hat and ESS, the R-hat above 1.01 marked as
 * a warning and above 1.05 as bad, the ESS under 400 as a warning, and the divergences in the caption. The table is the
 * library's `markedTable`, so its view is the existing one.
 *
 * @param record - The diagnostics.
 * @returns The table.
 * @example
 * ```ts
 * import { chainDiagnostics, chainTable } from 'omai-engine/exposition'
 * const r = chainDiagnostics({ mu: [[1, 2, 3], [2, 3, 4]] }, { summary: [{ site: 'mu', name: 'mu', rhat: 1.08, ess: 90 }] })
 * chainTable(r).marks?.map((m) => [m.col, m.status]) // => [[3, 'bad'], [4, 'warn']]
 * ```
 */
export function chainTable(record: ChainDiagnostics): MarkedTable {
  const marks: CellMark[] = []
  const cells = record.quantities.map((q, row) => {
    if (q.rhat !== undefined && q.rhat > 1.01)
      marks.push({
        row,
        col: 3,
        status: q.rhat > 1.05 ? 'bad' : 'warn',
        note: 'R-hat above 1.01: the chains have not mixed',
      })
    if (q.ess !== undefined && q.ess < 400)
      marks.push({ row, col: 4, status: 'warn', note: 'effective sample size under 400' })
    return [q.name, q.mean, q.sd, q.rhat ?? null, q.ess ?? null]
  })
  const n = record.divergences.length
  return markedTable({
    columns: ['quantity', 'mean', 'sd', 'R-hat', 'ESS'],
    cells,
    marks,
    digits: 4,
    caption: n === 0 ? 'no divergent transitions' : `${n} divergent transition${n === 1 ? '' : 's'}`,
  })
}
