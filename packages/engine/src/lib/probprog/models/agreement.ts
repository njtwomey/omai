/**
 * Running a benchmark model's plans and comparing them with its truth: `runPlan` makes the method of a plan on the
 * model, runs it as a session of the trace library and reads the posterior of its last state; `benchmarkAgreement` does
 * that for every plan (or those of some kinds), tabulates each against the model's truth with the validation kernel's
 * `agree`, and says which rows break their tolerances. A method whose `check` says it does not apply is skipped with
 * its reason; nothing here chooses a method.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { session } from '../../trace/index.ts'
import { methodKinds } from '../inference/index.ts'
import type { Posterior } from '../posterior.ts'
import {
  type AgreementRow,
  type AgreementTable,
  agree,
  agreementWithin,
  fromPosterior,
  type MarginalOracle,
} from '../validation/index.ts'
import type { BenchmarkData, BenchmarkModel, BenchmarkTruth, MethodPlan, TruthQuality } from './benchmark.ts'

/** The result of running one plan. */
export interface PlanRun {
  readonly plan: MethodPlan
  readonly posterior: Posterior
  /** The session's status at the end (`completed`, `converged`, ...). */
  readonly status: string
  /** The steps taken. */
  readonly steps: number
}

/**
 * Run one plan on a model: make the method of the plan's kind on the model, run a session of it and read the posterior of
 * its last state.
 *
 * @param model - The benchmark model.
 * @param plan - The method, knobs and steps.
 * @param options - The data (default the canonical data) and the key of the run, or a seed (default 0).
 * @returns The posterior, the status and the steps taken.
 * @throws OmaiError with code `value` for a kind that is not in `methodKinds`, and whatever the method throws when it does
 *   not apply (its verdict's reason).
 * @concept inference-engines
 * @example
 * ```ts
 * import { betaBernoulli, runPlan } from 'omai-engine/probprog/models'
 * const plan = { kind: 'conjugate', config: {}, steps: 3, expect: 'agree', note: 'exact' } as const
 * runPlan(betaBernoulli, plan).posterior.engine // => 'conjugate'
 * ```
 */
export function runPlan(
  model: BenchmarkModel,
  plan: MethodPlan,
  options: { readonly data?: BenchmarkData; readonly seed?: number; readonly key?: Key } = {},
): PlanRun {
  const make = methodKinds[plan.kind]
  if (make === undefined)
    throw new OmaiError(
      'value',
      `no inference method of kind '${plan.kind}' (known: ${Object.keys(methodKinds).join(', ')})`,
    )
  const method = make(model.methodModel(options.data), {})
  const s = session(method, {
    config: plan.config,
    steps: plan.steps,
    mode: 'eager',
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
  })
  const posterior = method.posterior?.(s.state, s.config)
  if (posterior === undefined) throw new OmaiError('value', `the ${plan.kind} method gives no posterior`)
  return { plan, posterior, status: s.status ?? 'running', steps: s.t }
}

/**
 * A posterior whose summary reads: one that lists a site it holds no draws of (the discrete label of a mixture, for a
 * Laplace or variational fit) cannot be summarised, so it is given an empty summary and `agree` falls back to Kish's
 * effective sample size from the weights.
 */
function summarisable(p: Posterior): Posterior {
  try {
    p.summary()
    return p
  } catch {
    return { ...p, summary: () => [] }
  }
}

/** What a model's plans did: one table against the truth, the runs and which `agree` rows broke their tolerances. */
export interface BenchmarkAgreement {
  readonly model: string
  readonly truth: BenchmarkTruth
  /** One row per method and quantity, against the truth's marginals. */
  readonly table: AgreementTable
  /** The runs, in plan order (a plan that was skipped has none). */
  readonly runs: readonly PlanRun[]
  /** What could not be compared: quantities a method's posterior does not hold (a deterministic site, for instance). */
  readonly notes: readonly string[]
  /** Plans not run, with the reason (a method's own verdict, or the filter). */
  readonly skipped: readonly { readonly kind: string; readonly reason: string }[]
  /** Rows of `agree` plans outside their tolerances, with the measures that failed. */
  readonly failures: readonly { readonly row: AgreementRow; readonly failed: readonly string[] }[]
  /** Plans that were expected to differ from the truth and did not, or that could not be compared at all. */
  readonly surprises: readonly string[]
  /** Whether every `agree` plan held and every `differ` plan differed. */
  readonly passed: boolean
}

/** Options of {@link benchmarkAgreement}. */
export interface BenchmarkAgreementOptions {
  /** The data (default the canonical data). */
  readonly data?: BenchmarkData
  /** Seed of the runs (default 0); plan `i` runs under a key derived from it. */
  readonly seed?: number
  /** The root key of the runs, in place of the seed (a campaign's own key). */
  readonly key?: Key
  /** Multiplier of the plans' budgets (default 1). */
  readonly scale?: number
  /** Only the plans of these kinds. */
  readonly only?: readonly string[]
  /** A truth already computed (a quadrature can take seconds): default `model.truth(data)`. */
  readonly truth?: BenchmarkTruth
  /** Truth quality when no `truth` is given (default `full`). */
  readonly truthQuality?: TruthQuality
  /** Draws made for a posterior that stores none (default 4000). */
  readonly numDraws?: number
}

/**
 * Run the plans of a benchmark model and compare each with the truth: the cross-engine agreement table of the model.
 * A plan whose method says it does not apply is skipped with that reason; the table has a row per method and quantity
 * with the mean's error in posterior sds, the sd ratio, Kolmogorov and Wasserstein distances and the KL to the truth.
 *
 * @param model - The benchmark model.
 * @param options - Data, seed, budget scale, a kind filter, a precomputed truth.
 * @returns The table, the runs, what was skipped and which `agree` rows are outside their tolerances.
 * @concept posterior-validation
 * @index function Cross-engine agreement of a benchmark model: every planned method against the truth.
 * @example
 * ```ts
 * import { benchmarkAgreement, betaBernoulli } from 'omai-engine/probprog/models'
 * const a = benchmarkAgreement(betaBernoulli, { only: ['conjugate'] })
 * [a.passed, a.table.rows.map((r) => r.method)] // => [true, ['conjugate']]
 * ```
 */
export function benchmarkAgreement(model: BenchmarkModel, options: BenchmarkAgreementOptions = {}): BenchmarkAgreement {
  const truth = options.truth ?? model.truth(options.data, options.truthQuality)
  const root = options.key ?? makeKey(options.seed ?? 0)
  const runs: PlanRun[] = []
  const skipped: { kind: string; reason: string }[] = []
  const rows: AgreementRow[] = []
  const failures: { row: AgreementRow; failed: readonly string[] }[] = []
  const notes: string[] = []
  const surprises: string[] = []
  const names = Object.keys(truth.marginals)
  model.plans(options.scale ?? 1).forEach((plan, i) => {
    if (options.only !== undefined && !options.only.includes(plan.kind)) return
    let run: PlanRun
    try {
      run = runPlan(model, plan, {
        ...(options.data === undefined ? {} : { data: options.data }),
        key: foldIn(root, i),
      })
    } catch (e) {
      const reason = e instanceof Error ? e.message : String(e)
      skipped.push({ kind: plan.kind, reason })
      if (plan.expect === 'agree') surprises.push(`${plan.kind} was expected to agree and did not run: ${reason}`)
      return
    }
    runs.push(run)
    const sampled = fromPosterior(plan.kind, summarisable(run.posterior), {
      key: foldIn(root, 100 + i),
      ...(options.numDraws === undefined ? {} : { numDraws: options.numDraws }),
    })
    const approximation = { ...sampled, draws: { ...sampled.draws, ...(model.summarise?.(sampled.draws) ?? {}) } }
    const wanted = (plan.quantities ?? names).filter((q) => truth.marginals[q] !== undefined)
    const quantities = wanted.filter((q) => approximation.draws[q] !== undefined)
    if (quantities.length < wanted.length)
      notes.push(
        `${plan.kind}: no draws of ${wanted.filter((q) => !quantities.includes(q)).join(', ')} (its posterior does not hold those sites)`,
      )
    const oracle = Object.fromEntries(quantities.map((q) => [q, truth.marginals[q] as MarginalOracle]))
    const table = agree([approximation], { oracle, quantities: Object.keys(oracle), quantilePoints: 400 })
    rows.push(...table.rows)
    if (plan.expect === 'agree') failures.push(...agreementWithin(table, plan.tolerances ?? {}))
    // A method that failed, or that gave nothing to compare, has differed from the truth in the most complete way.
    if (
      plan.expect === 'differ' &&
      run.status !== 'failed' &&
      table.rows.length > 0 &&
      agreementWithin(table, plan.tolerances ?? {}).length === 0
    )
      surprises.push(`${plan.kind} was expected to differ from the truth (${plan.note}) and agrees with it`)
  })
  return {
    model: model.name,
    truth,
    table: { oracle: `${truth.kind}: ${truth.source}`, quantities: names, rows },
    runs,
    notes,
    skipped,
    failures,
    surprises,
    passed: failures.length === 0 && surprises.length === 0,
  }
}
