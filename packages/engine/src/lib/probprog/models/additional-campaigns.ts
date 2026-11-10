/**
 * The periodic campaigns of the benchmark models added after the first suite: `additionalCampaigns` returns their
 * `benchmark/<model>` agreement campaigns (the same as `benchmarkCampaigns` makes) and the clutter problem's EP table,
 * `benchmark/clutter-ep`, which runs expectation propagation with numeric sites under three integration rules against the
 * exact posterior, beside the Laplace fit and the mean-field guide (Minka's 2001 comparison). The campaigns are meant to come
 * after every earlier campaign, so that a campaign's random key, which follows from its position, does not move.
 */

import type { Campaign, CampaignOutcome } from '../validation/runner.ts'
import { additionalBenchmarkModels } from './additional.ts'
import { benchmarkAgreement } from './agreement.ts'
import { benchmarkCampaigns } from './campaigns.ts'
import { type ClutterRule, clutterEp, clutterProblem } from './clutter.ts'

const fix = (v: number, digits = 3): string => (Number.isFinite(v) ? v.toFixed(digits) : String(v))

/**
 * The clutter problem's comparison of approximations: the exact posterior of `theta` (two modes) by quadrature, with its mean, spread
 * and dominant mode; expectation propagation with the numeric sites integrated by the engine's panelled quadrature, a 32-point
 * Gauss-Hermite rule and the Laplace rule; and the Laplace fit, mean-field guide and NUTS of the benchmark's plans. It is a report:
 * it passes when EP converges under every rule and, with the two integrating rules, lands on the dominant mode (within one of its
 * own standard deviations), which is what moment matching of the factors does when the posterior has two modes; with the Laplace
 * rule, which follows the mode of each tilted site, it may settle on the other. The exact mean and spread are those of both.
 *
 * @returns The campaign, named `benchmark/clutter-ep`.
 * @concept numerical-projections
 * @example
 * ```ts
 * import { clutterEpCampaign } from 'omai-engine/probprog/models'
 * clutterEpCampaign().name // => 'benchmark/clutter-ep'
 * ```
 */
export function clutterEpCampaign(): Campaign {
  return {
    name: 'benchmark/clutter-ep',
    description:
      "Minka's clutter problem: EP with numeric sites, the Laplace fit and the mean-field guide against the exact posterior.",
    run: ({ key, scale }): CampaignOutcome => {
      const y = clutterProblem.data.data['y'] as number[]
      const truth = clutterProblem.truth()
      const exact = truth.marginals['theta']
      if (exact === undefined) throw new Error('the clutter truth has no theta')
      const rows: string[] = []
      // The dominant mode of the exact posterior: where the mass in a window of 0.2 is largest.
      let mode = 0
      let best = -1
      for (let x = -10; x <= 15; x += 0.05) {
        const m = exact.cdf(x + 0.1) - exact.cdf(x - 0.1)
        if (m > best) {
          best = m
          mode = x
        }
      }
      let passed = true
      for (const rule of ['quadrature', 'gauss-hermite', 'laplace'] as ClutterRule[]) {
        const r = clutterEp(y, rule)
        const err = (r.mean - exact.mean) / exact.sd
        const ratio = r.sd / exact.sd
        if (!r.converged || (rule !== 'laplace' && Math.abs(r.mean - mode) > r.sd)) passed = false
        rows.push(
          `| EP, ${rule} rule | ${fix(r.mean)} | ${fix(r.sd)} | ${fix(err)} | ${fix(ratio)} | ${r.converged ? 'converged' : 'not converged'} |`,
        )
      }
      const a = benchmarkAgreement(clutterProblem, { key, scale, truth, only: ['laplace', 'svi', 'nuts'] })
      for (const row of a.table.rows)
        rows.push(
          `| ${row.method} | ${fix(row.mean)} | ${fix(row.sd)} | ${fix(row.meanError)} | ${fix(row.sdRatio)} | reported |`,
        )
      const markdown = [
        `**The clutter problem** (n = ${y.length}): ${truth.source}.`,
        '',
        `Exact posterior of theta: mean ${fix(exact.mean)}, sd ${fix(exact.sd)}, dominant mode near ${fix(mode, 1)}.`,
        '',
        '| method | mean | sd | mean error (exact sd) | sd ratio | run |',
        '| --- | --- | --- | --- | --- | --- |',
        ...rows,
        '',
        'EP with numeric sites integrates each clutter factor against its cavity by the rule given and projects the tilted distribution to a Gaussian; with two modes it settles on one (the integrating rules on the dominant one, the Laplace rule on whichever its sites lead to), as the Laplace fit and the mean-field guide (single normals at one mode) do; the exact mean sits between the modes.',
      ].join('\n')
      return {
        passed,
        verdict: passed
          ? 'EP converges under every rule; the integrating rules land on the dominant mode'
          : 'EP did not converge, or an integrating rule missed the dominant mode',
        markdown,
        budget: { scalePercent: Math.round(scale * 100) },
      }
    },
  }
}

/**
 * The campaigns of the additional benchmark models, in order: one `benchmark/<model>` each, then `benchmark/clutter-ep`.
 *
 * @returns The campaigns.
 * @concept posterior-validation
 * @index function The periodic campaigns of the later benchmark models, plus the clutter problem's EP comparison.
 * @example
 * ```ts
 * import { additionalCampaigns } from 'omai-engine/probprog/models'
 * additionalCampaigns().map((c) => c.name).slice(-2) // => ['benchmark/clutter-problem', 'benchmark/clutter-ep']
 * ```
 */
export function additionalCampaigns(): Campaign[] {
  return [...benchmarkCampaigns(additionalBenchmarkModels()), clutterEpCampaign()]
}
