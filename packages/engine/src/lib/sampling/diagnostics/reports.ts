/**
 * Posterior summaries and sampler health checks, as ArviZ's `summary` and `diagnose`: a per-coordinate table of mean,
 * sd, credible interval, effective sample sizes, R-hat and standard errors, the energy fraction of missing information
 * (E-BFMI), counts of divergent transitions and of NUTS trees that hit the depth limit, and an overall verdict.
 */

import { WhiteboxError } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { type Chains, parameterDraws } from '../chains.ts'
import { chainOf, drawsOf } from './_draws.ts'
import { ess, mcse, rhat } from './convergence.ts'
import { eti, hdi } from './intervals.ts'

/** One row of {@link summary}: ArviZ's `summary` columns for one coordinate. */
export interface SummaryRow {
  readonly name: string
  readonly mean: number
  /** The standard deviation of the pooled draws (ddof 1). */
  readonly sd: number
  /** The credible interval's bounds. */
  readonly low: number
  readonly high: number
  readonly essBulk: number
  readonly essTail: number
  readonly rhat: number
  readonly mcseMean: number
  readonly mcseSd: number
}

/** Options of {@link summary}. */
export interface SummaryOptions {
  /** The interval's probability (default 0.89, ArviZ's default). */
  readonly prob?: number
  /** Equal-tailed (default, as ArviZ) or highest-density interval. */
  readonly interval?: 'eti' | 'hdi'
  /**
   * The tail probability of the tail ESS (default 0.05: the q05 and q95 indicators, as Stan and Vehtari et al.; ArviZ
   * 1.x's `summary` uses its interval default 0.89 here, i.e. q11 and q89).
   */
  readonly tailProb?: number
}

function pooledMoments(x: Float64Array): { readonly mean: number; readonly sd: number } {
  let m = 0
  for (const v of x) m += v
  m /= x.length
  let s = 0
  for (const v of x) s += (v - m) ** 2
  return { mean: m, sd: x.length < 2 ? Number.NaN : Math.sqrt(s / (x.length - 1)) }
}

/**
 * The posterior summary of every coordinate (ArviZ `summary`, kind `all`): mean, sd, a credible interval, bulk and
 * tail ESS, rank-normalised split R-hat, and the Monte Carlo standard errors of the mean and the sd.
 *
 * @param chains - Chains, or draws `[chain, draw]` of one coordinate.
 * @param options - The interval's probability and kind, and the tail probability of the tail ESS.
 * @returns One row per coordinate.
 * @concept mcmc-diagnostics
 * @example
 * ```ts
 * import { summary } from 'whitebox-engine/sampling'
 * const [row] = summary([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
 * row?.mean // => 3.5
 * ```
 */
export function summary(chains: Chains | TensorLike, options: SummaryOptions = {}): SummaryRow[] {
  const prob = options.prob ?? 0.89
  const interval = options.interval ?? 'eti'
  const rows: { readonly name: string; readonly draws: TensorLike }[] = isChains(chains)
    ? chains.names.map((name, i) => ({ name, draws: parameterDraws(chains, i) }))
    : [{ name: 'x', draws: chains }]
  return rows.map(({ name, draws }) => {
    const d = drawsOf(draws, 'summary')
    const { mean, sd } = pooledMoments(d.data)
    const [low, high] = interval === 'hdi' ? hdi(d.data, prob) : eti(d.data, prob)
    return {
      name,
      mean,
      sd,
      low,
      high,
      essBulk: ess(draws),
      essTail: ess(draws, { method: 'tail', prob: options.tailProb ?? 0.05 }),
      rhat: rhat(draws),
      mcseMean: mcse(draws),
      mcseSd: mcse(draws, { method: 'sd' }),
    }
  })
}

function isChains(x: Chains | TensorLike): x is Chains {
  return typeof x === 'object' && x !== null && 'numChains' in x && 'draws' in x && 'names' in x
}

/**
 * The energy Bayesian fraction of missing information per chain (ArviZ `bfmi`, Betancourt 2016): the mean squared
 * change of the energy between iterations over its variance. Values below 0.3 mean the momentum resampling cannot move
 * the chain through the energy levels of the target (heavy tails, a funnel).
 *
 * @param energy - The Hamiltonian at each draw, `[chain, draw]` or `[draw]` (the `energy` statistic of HMC and NUTS
 *   chains).
 * @returns E-BFMI per chain.
 * @concept energy-bfmi
 * @reference M. Betancourt, Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte Carlo (2016).
 * @example
 * ```ts
 * import { bfmi } from 'whitebox-engine/sampling'
 * import { toArray } from 'whitebox-engine/tensor'
 * // Energies that alternate move freely: E-BFMI well above 1.
 * (toArray(bfmi([[1, -1, 1, -1, 1, -1]])) as number[])[0] > 1 // => true
 * ```
 */
export function bfmi(energy: TensorLike): Tensor<'float64'> {
  const d = drawsOf(energy, 'bfmi')
  const out = new Float64Array(d.chains)
  for (let c = 0; c < d.chains; c++) {
    const e = chainOf(d, c)
    let num = 0
    for (let i = 1; i < e.length; i++) num += ((e[i] ?? 0) - (e[i - 1] ?? 0)) ** 2
    num /= e.length - 1
    out[c] = num / pooledMoments(e).sd ** 2
  }
  return fromStorage(out, [d.chains], 'float64')
}

/** Divergent transitions of HMC or NUTS chains. */
export interface DivergenceReport {
  readonly count: number
  readonly total: number
  /** Percentage of transitions that diverged. */
  readonly percent: number
  readonly perChain: readonly number[]
  /** `[chain, draw]` of every divergent transition. */
  readonly locations: readonly (readonly [number, number])[]
}

function statOf(chains: Chains, name: 'divergent' | 'treeDepth' | 'energy', who: string): Tensor<'float64'> {
  const s = chains.stats[name]
  if (s === undefined) throw new WhiteboxError('value', `${who}: the chains have no ${name} statistic`)
  return s
}

/**
 * Count the divergent transitions of Hamiltonian chains (the `divergent` statistic), per chain and in all, with their
 * positions, as ArviZ's `diagnose` reports them. Divergences mean the integrator met curvature too strong for its step
 * size, and the draws near them are biased.
 *
 * @param chains - Chains with a `divergent` statistic (HMC, NUTS, Riemannian HMC).
 * @returns The counts, the percentage and the `[chain, draw]` locations.
 * @throws WhiteboxError with code `value` when the chains carry no divergence statistic.
 * @concept divergent-transitions
 * @example
 * ```ts
 * import { chainsFromDraws, divergenceReport } from 'whitebox-engine/sampling'
 * const c = chainsFromDraws([[1, 2, 3]], { stats: { divergent: [[0, 1, 0]] } })
 * divergenceReport(c).locations // => [[0, 1]]
 * ```
 */
export function divergenceReport(chains: Chains): DivergenceReport {
  const s = statOf(chains, 'divergent', 'divergenceReport').data
  const perChain: number[] = []
  const locations: [number, number][] = []
  for (let c = 0; c < chains.numChains; c++) {
    let n = 0
    for (let d = 0; d < chains.numDraws; d++) {
      if ((s[c * chains.numDraws + d] ?? 0) > 0) {
        n += 1
        locations.push([c, d])
      }
    }
    perChain.push(n)
  }
  const count = locations.length
  const total = chains.numChains * chains.numDraws
  return { count, total, percent: (100 * count) / total, perChain, locations }
}

/** Transitions of NUTS chains that hit the tree-depth limit. */
export interface TreeDepthReport {
  readonly maxTreeDepth: number
  /** Transitions whose tree reached the limit. */
  readonly count: number
  readonly total: number
  readonly percent: number
  readonly perChain: readonly number[]
  /** The mean tree depth per chain. */
  readonly meanDepth: readonly number[]
}

/**
 * Count the NUTS transitions whose tree reached the depth limit (the `treeDepth` statistic): such trajectories stop
 * before the U-turn, so the sampler explores slowly; raise the limit or reparameterise.
 *
 * @param chains - Chains with a `treeDepth` statistic (NUTS).
 * @param maxTreeDepth - The sampler's limit (default 10, as NUTS).
 * @returns The counts, the percentage and the mean depth per chain.
 * @throws WhiteboxError with code `value` when the chains carry no tree-depth statistic.
 * @concept nuts
 * @example
 * ```ts
 * import { chainsFromDraws, treeDepthReport } from 'whitebox-engine/sampling'
 * const c = chainsFromDraws([[1, 2, 3, 4]], { stats: { treeDepth: [[2, 3, 3, 1]] } })
 * treeDepthReport(c, 3).count // => 2
 * ```
 */
export function treeDepthReport(chains: Chains, maxTreeDepth = 10): TreeDepthReport {
  const s = statOf(chains, 'treeDepth', 'treeDepthReport').data
  const perChain: number[] = []
  const meanDepth: number[] = []
  for (let c = 0; c < chains.numChains; c++) {
    let n = 0
    let sum = 0
    for (let d = 0; d < chains.numDraws; d++) {
      const depth = s[c * chains.numDraws + d] ?? 0
      sum += depth
      if (depth >= maxTreeDepth) n += 1
    }
    perChain.push(n)
    meanDepth.push(sum / chains.numDraws)
  }
  const count = perChain.reduce((a, b) => a + b, 0)
  const total = chains.numChains * chains.numDraws
  return { maxTreeDepth, count, total, percent: (100 * count) / total, perChain, meanDepth }
}

/** Options of {@link diagnose}. */
export interface DiagnoseOptions {
  /** The largest acceptable R-hat (default 1.01). */
  readonly rhatMax?: number
  /** The smallest acceptable min(bulk, tail) ESS (default 100 per chain). */
  readonly essThreshold?: number
  /** The smallest acceptable E-BFMI (default 0.3). */
  readonly bfmiThreshold?: number
  /** The NUTS tree-depth limit (default 10). */
  readonly maxTreeDepth?: number
}

/** The verdict of {@link diagnose}. */
export interface Diagnosis {
  readonly hasErrors: boolean
  readonly divergences?: DivergenceReport
  readonly treeDepth?: TreeDepthReport
  readonly bfmi?: readonly number[]
  /** Coordinates with R-hat above the limit, and those with too few effective draws. */
  readonly highRhat: readonly string[]
  readonly lowEss: readonly string[]
  /** Human-readable findings, one per check. */
  readonly messages: readonly string[]
}

/**
 * Run the standard checks on chains, as ArviZ's `diagnose`: divergences, tree depth, E-BFMI (when the chains carry
 * those statistics), R-hat below `rhatMax` and min(bulk, tail) ESS above `essThreshold` for every coordinate.
 *
 * @param chains - The chains.
 * @param options - Thresholds.
 * @returns Whether any check failed, the reports, and a message per check.
 * @concept mcmc-diagnostics
 * @example
 * ```ts
 * import { chainsFromDraws, diagnose } from 'whitebox-engine/sampling'
 * // Two chains in different places fail the R-hat check.
 * const c = chainsFromDraws([[0, 0.1, -0.1, 0.05, 0, -0.05], [5, 5.1, 4.9, 5.05, 5, 4.95]])
 * diagnose(c).highRhat // => ['x[0]']
 * ```
 */
export function diagnose(chains: Chains, options: DiagnoseOptions = {}): Diagnosis {
  const rhatMax = options.rhatMax ?? 1.01
  const essThreshold = options.essThreshold ?? 100 * chains.numChains
  const bfmiThreshold = options.bfmiThreshold ?? 0.3
  const messages: string[] = []
  let hasErrors = false
  let divergences: DivergenceReport | undefined
  let tree: TreeDepthReport | undefined
  let energy: number[] | undefined
  if (chains.stats.divergent !== undefined) {
    divergences = divergenceReport(chains)
    if (divergences.count > 0) {
      hasErrors = true
      messages.push(
        `${divergences.count} of ${divergences.total} (${divergences.percent.toFixed(2)}%) transitions diverged; increase the target acceptance or reparameterise.`,
      )
    } else messages.push('No divergent transitions.')
  }
  if (chains.stats.treeDepth !== undefined) {
    tree = treeDepthReport(chains, options.maxTreeDepth ?? 10)
    if (tree.count > 0) {
      hasErrors = true
      messages.push(`${tree.count} of ${tree.total} transitions hit the maximum tree depth ${tree.maxTreeDepth}.`)
    } else messages.push('Tree depth satisfactory for all transitions.')
  }
  if (chains.stats.energy !== undefined) {
    energy = Array.from(bfmi(statOf(chains, 'energy', 'diagnose')).data)
    const low = energy.flatMap((v, c) => (v < bfmiThreshold ? [c] : []))
    if (low.length > 0) {
      hasErrors = true
      messages.push(`E-BFMI below ${bfmiThreshold} in chains ${low.join(', ')}.`)
    } else messages.push('E-BFMI satisfactory for all chains.')
  }
  const highRhat: string[] = []
  const lowEss: string[] = []
  chains.names.forEach((name, i) => {
    const x = parameterDraws(chains, i)
    if (!(rhat(x) <= rhatMax)) highRhat.push(name)
    if (!(Math.min(ess(x), ess(x, { method: 'tail' })) >= essThreshold)) lowEss.push(name)
  })
  if (highRhat.length > 0) {
    hasErrors = true
    messages.push(`R-hat above ${rhatMax} for ${highRhat.join(', ')}.`)
  } else messages.push('R-hat satisfactory for all coordinates.')
  if (lowEss.length > 0) {
    hasErrors = true
    messages.push(`Effective sample size below ${essThreshold} for ${lowEss.join(', ')}.`)
  } else messages.push('Effective sample size satisfactory for all coordinates.')
  return {
    hasErrors,
    ...(divergences === undefined ? {} : { divergences }),
    ...(tree === undefined ? {} : { treeDepth: tree }),
    ...(energy === undefined ? {} : { bfmi: energy }),
    highRhat,
    lowEss,
    messages,
  }
}
