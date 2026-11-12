/**
 * The draws of a sampler's posterior regrouped by chain: each scalar coordinate (`mu`, `theta[3]`) as one row of draws per
 * chain, with the divergent flags and the energy of every chain beside them. A `Posterior` of Markov chains keeps its
 * draws of a site laid end to end (chain after chain), which is what the summary and the validation kernel want; the
 * diagnostics of chains (trace and rank plots, R-hat, the energy plot, the pair plot with its divergent transitions) want
 * them apart, as `chainDiagnostics` of `omai-engine/exposition` takes them.
 */

import { asTensor, reshape, sliceAxis, type TensorLike, take, toArray, toFlat } from '../tensor/index.ts'
import type { Posterior } from './posterior.ts'

/** The chains of a posterior by coordinate. */
export interface PosteriorChains {
  /** Draws by scalar coordinate and chain: `draws['theta[0]'][c][i]`, on the scale of the site (constrained). */
  readonly draws: Readonly<Record<string, number[][]>>
  /** 1 where a transition diverged: `divergent[c][i]` (absent for a sampler that does not report it). */
  readonly divergent?: number[][]
  /** The energy of each chain at each draw (absent for a sampler that does not report it). */
  readonly energy?: number[][]
}

/** Options of {@link posteriorChains}. */
export interface PosteriorChainsOptions {
  /** The sites to take (default every site with draws). */
  readonly sites?: readonly string[]
}

const rowsOf = (flat: TensorLike, chains: number, draws: number): number[][] =>
  toArray(reshape(asTensor(flat), [chains, draws])) as number[][]

/**
 * The draws of a posterior of Markov chains by chain, with the divergent flags and the energy.
 *
 * @param posterior - A posterior with `chains` (NUTS, HMC and the other kernel methods give one).
 * @param options - The sites to take.
 * @returns The chains by coordinate; empty `draws` when the posterior has no chains or no draws yet.
 * @concept markov-chain-monte-carlo
 * @index function MCMC posterior draws by chain and coordinate, with divergences and energy.
 * @example
 * ```ts
 * import { posteriorChains, observe, sample, mcmc } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const p = mcmc(model, undefined, { key: key(0), numChains: 2, numWarmup: 40, numDraws: 25 })
 * const c = posteriorChains(p)
 * [Object.keys(c.draws), c.draws['mu']?.length, c.draws['mu']?.[0]?.length, c.divergent?.length] // => [['mu'], 2, 25, 2]
 * ```
 */
export function posteriorChains(posterior: Posterior, options: PosteriorChainsOptions = {}): PosteriorChains {
  const chains = posterior.chains
  if (chains === undefined || posterior.draws === undefined || chains.numDraws < 1) return { draws: {} }
  const draws: Record<string, number[][]> = {}
  for (const [site, value] of Object.entries(posterior.draws)) {
    if (options.sites !== undefined && !options.sites.includes(site)) continue
    const total = chains.numChains * chains.numDraws
    const size = Math.floor(toFlat(value).length / total)
    // One row per draw of every chain, one column per coordinate of the site.
    const columns = reshape(sliceAxis(reshape(value, [-1]), 0, 0, total * size), [total, size])
    for (let j = 0; j < size; j++) {
      const name = size === 1 ? site : `${site}[${j}]`
      draws[name] = rowsOf(take(columns, j, 1), chains.numChains, chains.numDraws)
    }
  }
  const stat = (name: 'divergent' | 'energy'): number[][] | undefined => {
    const t = chains.stats[name]
    return t === undefined ? undefined : rowsOf(t, chains.numChains, chains.numDraws)
  }
  const divergent = stat('divergent')
  const energy = stat('energy')
  return { draws, ...(divergent === undefined ? {} : { divergent }), ...(energy === undefined ? {} : { energy }) }
}
