/**
 * The cross-engine agreement of the benchmark suite as campaigns of the periodic validation: one `benchmark/<model>`
 * campaign per model that runs the model's planned methods under modest budgets, compares each with the model's truth with
 * the validation kernel's `agree`, and writes the table with the truth's source, which methods apply (their own `check`s),
 * how each run ended and what to expect of it. A campaign passes when every method that must agree does and every method
 * expected to differ differs. `scripts/probprog-validate.ts` takes these with `validationCampaigns()`; none of it runs in
 * `make verify` or `make check`.
 */

import { agreementMarkdown } from '../validation/agree.ts'
import type { Campaign, CampaignOutcome } from '../validation/runner.ts'
import { type BenchmarkAgreement, benchmarkAgreement } from './agreement.ts'
import { type BenchmarkModel, type MethodVerdict, methodVerdicts } from './benchmark.ts'
import { benchmarkModels } from './suite.ts'

const cell = (v: unknown): string => String(v).replaceAll('|', '\\|').replaceAll('\n', ' ')

/** How a run went: the status, the steps, and the diagnostics a reader of the table looks for. */
function runLine(a: BenchmarkAgreement, index: number): string {
  const run = a.runs[index] as BenchmarkAgreement['runs'][number]
  const d = run.posterior.diagnostics
  const parts: string[] = []
  if (typeof d['divergences'] === 'number') parts.push(`${d['divergences']} divergences`)
  const rhat = d['highRhat']
  if (Array.isArray(rhat) && rhat.length > 0)
    parts.push(`R-hat high: ${rhat.slice(0, 3).join(', ')}${rhat.length > 3 ? ', ...' : ''}`)
  const ess = d['lowEss']
  if (Array.isArray(ess) && ess.length > 0)
    parts.push(`ESS low: ${ess.slice(0, 3).join(', ')}${ess.length > 3 ? ', ...' : ''}`)
  if (typeof d['ess'] === 'number') parts.push(`weights ESS ${Math.round(d['ess'])}`)
  const expect =
    run.plan.expect === 'agree' ? 'must agree' : run.plan.expect === 'differ' ? 'expected to differ' : 'reported'
  return `| ${cell(run.plan.kind)} | ${expect} | ${cell(run.status)} (${run.steps} steps) | ${cell(parts.join('; ') || '-')} | ${cell(run.plan.note)} |`
}

/**
 * The campaign's table for a model: what it is, where the truth comes from, which methods apply, how the planned runs
 * ended and the agreement of each with the truth.
 *
 * @param model - The benchmark model.
 * @param a - Its agreement.
 * @param verdicts - The verdict of every method kind's `check` on it.
 * @returns The markdown.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { benchmarkAgreement, benchmarkMarkdown, betaBernoulli, methodVerdicts } from 'omai-engine/probprog/models'
 * const text = benchmarkMarkdown(betaBernoulli, benchmarkAgreement(betaBernoulli, { only: ['conjugate'] }), methodVerdicts(betaBernoulli))
 * text.split('\n')[0] // => '**Beta-Bernoulli coin** (beta-bernoulli): conjugate, exact, discrete-data, one-dimensional, unit-interval'
 * ```
 */
export function benchmarkMarkdown(
  model: BenchmarkModel,
  a: BenchmarkAgreement,
  verdicts: readonly MethodVerdict[],
): string {
  const applies = verdicts.filter((v) => v.verdict.applies).map((v) => v.kind)
  const refused = verdicts.filter((v) => !v.verdict.applies)
  const lines = [
    `**${model.title}** (${model.name}): ${model.tags.join(', ')}`,
    '',
    model.description,
    '',
    `Truth (${a.truth.kind}): ${a.truth.source}.${a.truth.logEvidence === undefined ? '' : ` Log evidence ${a.truth.logEvidence.toFixed(4)}.`}`,
    '',
    `Methods whose own \`check\` says they apply${model.target === 'density' ? ' to the compiled program (a target with no observed node is refused at the spec level)' : ''}: ${applies.join(', ') || 'none'}. Not applicable: ${refused.map((v) => `${v.kind} (${v.verdict.reason})`).join('; ') || 'none'}.`,
    '',
    '| method | expectation | run | diagnostics | why |',
    '| --- | --- | --- | --- | --- |',
    ...a.runs.map((_, i) => runLine(a, i)),
    '',
    agreementMarkdown(a.table),
  ]
  const findings = [
    ...a.failures.map((f) => `${f.row.method} breaks ${f.failed.join(', ')} on ${f.row.quantity}, and must agree`),
    ...a.surprises,
    ...a.skipped
      .filter((s) => !a.surprises.some((x) => x.startsWith(s.kind)))
      .map((s) => `${s.kind} did not run: ${s.reason}`),
    ...a.notes,
  ]
  if (findings.length > 0) lines.push('', 'Notes:', '', ...findings.map((f) => `- ${f}`))
  return lines.join('\n')
}

/**
 * The benchmark campaigns, one per model, named `benchmark/<model>`, in the order of `benchmarkModels()` (campaign `i`
 * runs under the key the runner derives from its position, so adding a model at the end changes no other table).
 *
 * @param models - The models (default the suite).
 * @returns The campaigns.
 * @concept posterior-validation
 * @index function The cross-engine agreement of every benchmark model as periodic validation campaigns.
 * @example
 * ```ts
 * import { benchmarkCampaigns } from 'omai-engine/probprog/models'
 * benchmarkCampaigns()[0]?.name // => 'benchmark/beta-bernoulli'
 * ```
 */
export function benchmarkCampaigns(models: readonly BenchmarkModel[] = benchmarkModels()): Campaign[] {
  return models.map((model) => ({
    name: `benchmark/${model.name}`,
    description: `${model.title}: the planned inference methods against the exact, quadrature or analytic truth.`,
    run: ({ key, scale }): CampaignOutcome => {
      const a = benchmarkAgreement(model, { key, scale })
      const verdicts = methodVerdicts(model)
      const must = a.runs.filter((r) => r.plan.expect === 'agree').length
      const differ = a.runs.filter((r) => r.plan.expect === 'differ').length
      return {
        passed: a.passed,
        verdict: a.passed
          ? `${must} method(s) agree with the ${a.truth.kind} truth${differ === 0 ? '' : `, ${differ} differ as expected`}; ${a.runs.length - must - differ} reported`
          : `${a.failures.length} row(s) of methods that must agree are outside their tolerances${a.surprises.length === 0 ? '' : `; ${a.surprises.length} surprise(s)`}`,
        markdown: benchmarkMarkdown(model, a, verdicts),
        budget: { plans: a.runs.length, rows: a.table.rows.length, scalePercent: Math.round(scale * 100) },
      }
    },
  }))
}
