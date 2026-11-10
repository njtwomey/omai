/**
 * The runner of the periodic validation: a `Campaign` is a named check (a calibration run, an agreement table, a
 * Geweke test) with a budget that scales, and `runCampaigns` runs a chosen set and returns one report with, per
 * campaign, the verdict, the markdown table, the seeds and the budgets, ready to be written under
 * `docs/probprog/results/` by the script that owns a clock and the file system (the library has neither).
 */

import { foldIn, type Key, key as makeKey } from '../../random/index.ts'

/** What a campaign is given. */
export interface CampaignContext {
  /** The campaign's own key (derived from the run key by the campaign's position). */
  readonly key: Key
  /** The budget multiplier: replications and iterations are multiplied by it (1 is the committed budget). */
  readonly scale: number
}

/** What a campaign returns. */
export interface CampaignOutcome {
  /** Whether every expectation of the campaign held (a method that must fail did fail, one that must pass did). */
  readonly passed: boolean
  /** One line saying what happened. */
  readonly verdict: string
  /** The table of results, in markdown. */
  readonly markdown: string
  /** The budgets used (replications, draws, iterations), by name. */
  readonly budget: Readonly<Record<string, number>>
}

/** A named validation check. */
export interface Campaign {
  readonly name: string
  readonly description: string
  run(context: CampaignContext): CampaignOutcome
}

/** A campaign's outcome with its place in the run. */
export interface CampaignRun extends CampaignOutcome {
  readonly name: string
  readonly description: string
  /** The seed of the run's key. */
  readonly seed: number
  readonly scale: number
  /** Seconds the campaign took, when the run was given a clock. */
  readonly seconds?: number
}

/** Options of {@link runCampaigns}. */
export interface RunCampaignsOptions {
  /** The seed of the run's key (default 0); campaign `i` runs under `foldIn(key(seed), i)`. */
  readonly seed?: number
  /** The budget multiplier (default 1). */
  readonly scale?: number
  /** Run only campaigns whose name starts with one of these prefixes. */
  readonly only?: readonly string[]
  /** A clock in seconds, injected because the library reads none. */
  readonly clock?: () => number
}

/**
 * Run campaigns and collect their outcomes: each under its own derived key, so adding or skipping a campaign does not
 * change another's draws.
 *
 * @param campaigns - The campaigns, in a fixed order.
 * @param options - Seed, budget multiplier, a name filter and a clock.
 * @returns One record per campaign that ran.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { runCampaigns } from 'omai-engine/probprog/validation'
 * const ok = { name: 'always', description: 'passes', run: () => ({ passed: true, verdict: 'fine', markdown: '', budget: {} }) }
 * runCampaigns([ok]).map((r) => r.passed) // => [true]
 * ```
 */
export function runCampaigns(campaigns: readonly Campaign[], options: RunCampaignsOptions = {}): CampaignRun[] {
  const seed = options.seed ?? 0
  const scale = options.scale ?? 1
  const root = makeKey(seed)
  const out: CampaignRun[] = []
  campaigns.forEach((c, i) => {
    if (options.only !== undefined && !options.only.some((p) => c.name.startsWith(p))) return
    const t0 = options.clock?.()
    const outcome = c.run({ key: foldIn(root, i), scale })
    const t1 = options.clock?.()
    out.push({
      ...outcome,
      name: c.name,
      description: c.description,
      seed,
      scale,
      ...(t0 !== undefined && t1 !== undefined ? { seconds: t1 - t0 } : {}),
    })
  })
  return out
}

/**
 * One campaign's result as a markdown document: the verdict, the table, the budgets and seeds, with the commit and
 * date the script supplies.
 *
 * @param run - The campaign's record.
 * @param header - The date and commit of the run (supplied by the caller: the library knows neither).
 * @returns The markdown.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { campaignMarkdown } from 'omai-engine/probprog/validation'
 * const run = { name: 'x', description: 'd', passed: true, verdict: 'fine', markdown: '| a |', budget: { reps: 5 }, seed: 0, scale: 1 }
 * campaignMarkdown(run, { date: '2026-11-10', commit: 'abc' }).split('\n')[0] // => '# x'
 * ```
 */
export function campaignMarkdown(run: CampaignRun, header: { readonly date: string; readonly commit: string }): string {
  const budget = Object.entries(run.budget)
    .map(([k, v]) => `${k} ${v}`)
    .join(', ')
  return [
    `# ${run.name}`,
    '',
    run.description,
    '',
    `Run on ${header.date} at commit ${header.commit}; seed ${run.seed}, budget scale ${run.scale} (${budget})${run.seconds === undefined ? '' : `, ${run.seconds.toFixed(1)} s`}.`,
    '',
    `**${run.passed ? 'Passed' : 'FAILED'}**: ${run.verdict}`,
    '',
    run.markdown,
    '',
  ].join('\n')
}
