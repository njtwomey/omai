import { modelRunFromTape } from 'omai-engine/exposition'
import { activeSetScore, recoverySummary } from 'omai-engine/probprog/ep'
import { toFactorGraph, toStructure } from 'omai-engine/probprog/model'
import { type ZeroInflatedPosterior, zeroInflatedPoisson } from 'omai-engine/probprog/validation'
import { type ProgramCheck, sig } from '../../../../shared/ep/program-check.ts'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'
import { defaultSetup, infer, parameters, refusedForms, simulate, type ZipSetup } from './program.ts'

export type { ZipSetup }
export { defaultSetup }

/** One simulation, run by EP. */
export interface ZipRun {
  readonly setup: ZipSetup
  readonly data: ReturnType<typeof simulate>
  readonly fit: Awaited<ReturnType<typeof infer>>
  readonly ms: number
  readonly found: ReturnType<typeof parameters>
  /** The compiler's reason for each way of writing the model that it refuses. */
  readonly refusals: readonly { readonly form: string; readonly reason: string }[]
}

/** The messages of the run as a record for the player. */
export const recordOf = (run: ZipRun) =>
  modelRunFromTape(
    toStructure(run.fit.bound, { view: 'factor' }),
    { run: run.fit.run.run() },
    { maxSteps: 120, maxMessages: 6 },
  )

/** Simulates a setup and runs EP on the zero-inflated model. */
export async function runZip(setup: ZipSetup, hooks: EpHooks = {}): Promise<ZipRun> {
  const data = simulate(setup)
  const t0 = performance.now()
  const fit = await infer(data, 0.5, hooks)
  const ms = performance.now() - t0
  const refusals = Object.entries(refusedForms).map(([form, model]) => {
    const issues = toFactorGraph(model.bind({ y: data.y.slice(0, 6) } as never)).issues
    return { form, reason: issues.length === 0 ? 'accepted' : issues.map((i) => i.reason).join('; ') }
  })
  return { setup, data, fit, ms, found: parameters(fit.posterior), refusals }
}

/** The exact posterior of the same model (the structural zeros summed out, a refined grid over the three parameters). */
export const exactZip = (data: {
  readonly x: readonly number[]
  readonly y: readonly number[]
}): ZeroInflatedPosterior => zeroInflatedPoisson({ x: data.x, y: data.y, inflation: { a: 1, b: 1 }, coefficientSd: 2 })

/** The indices of the zero counts: the ones a structural zero can explain. */
export const zeroRows = (run: ZipRun): number[] => run.data.y.flatMap((v, i) => (v === 0 ? [i] : []))

/** The recovery check of a run. */
export function checkZip(run: ZipRun, exact: ZeroInflatedPosterior): ProgramCheck {
  const { setup, found, data } = run
  const names = ['inflation', 'intercept', 'slope'] as const
  const truth = [setup.inflation, setup.intercept, setup.slope]
  const ep = names.map((k) => found[k])
  const reference = [exact.pi, exact.b0, exact.b1]
  const rows: ProgramCheck['rows'][number][] = []
  const truthErr = names.map((_k, j) => Math.abs(((ep[j]?.mean ?? 0) - (truth[j] ?? 0)) / (ep[j]?.sd ?? 1)))
  rows.push({
    label: 'the true inflation, intercept and slope against EP',
    value: names.map((k, j) => `${k} ${sig(truthErr[j] ?? Number.NaN, 2)} sd`).join(', '),
    tolerance: 'each within 3 posterior sds (99%)',
    pass: truthErr.every((e) => e <= 3),
  })
  const summary = recoverySummary({
    truth,
    mean: ep.map((e) => e?.mean ?? 0),
    sd: ep.map((e) => e?.sd ?? 0),
    lower: ep.map((e) => e?.lower ?? 0),
    upper: ep.map((e) => e?.upper ?? 0),
    reference: { mean: reference.map((r) => r.mean), sd: reference.map((r) => r.sd) },
  })
  rows.push(
    {
      label: 'EP against the exact posterior: means of the three parameters',
      value: `largest error ${sig(summary.reference?.maxMeanZ ?? Number.NaN, 2)} exact sds`,
      tolerance: 'at most 0.25 sd',
      pass: (summary.reference?.maxMeanZ ?? Number.POSITIVE_INFINITY) <= 0.25,
    },
    {
      label: 'EP against the exact posterior: sds of the three parameters',
      value: `ratio EP / exact from ${sig(summary.reference?.sdRatio[0] ?? Number.NaN, 3)} to ${sig(summary.reference?.sdRatio[1] ?? Number.NaN, 3)}`,
      tolerance: 'between 0.8 and 1.25',
      pass: (summary.reference?.sdRatio[0] ?? 0) >= 0.8 && (summary.reference?.sdRatio[1] ?? 9) <= 1.25,
    },
  )
  const zeros = zeroRows(run)
  const epZ = zeros.map((i) => found.structural[i] ?? 0)
  const exactZ = zeros.map((i) => exact.structural[i] ?? 0)
  const flags = zeros.map((i) => data.structural[i] === true)
  const gap = Math.max(...epZ.map((p, k) => Math.abs(p - (exactZ[k] ?? 0))))
  const ours = activeSetScore(epZ, flags).auc
  const best = activeSetScore(exactZ, flags).auc
  rows.push(
    {
      label: 'P(structural zero) of each zero count, EP against exact',
      value: `largest difference ${sig(gap, 2)} over ${zeros.length} zeros`,
      tolerance: 'at most 0.05',
      pass: gap <= 0.05,
    },
    {
      label: 'zeros that are structural, ranked by EP’s probability (AUC), against the exact posterior’s own ranking',
      value: `${sig(ours, 3)} against ${sig(best, 3)}`,
      tolerance: 'within 0.03 of the exact ranking',
      pass: ours >= best - 0.03,
    },
  )
  return {
    rows,
    notes: [
      'The two states of each observation are the categories of a latent with Dirichlet(1, 1) weights: the second weight is the inflation, with the uniform Beta(1, 1) prior. A Bernoulli switch with a Beta probability is refused by the compiler; it has to be a two-category Dirichlet.',
      ...run.refusals.map((r) => `Refused: ${r.form}: "${r.reason}".`),
      'How the rate compiles: it is a known exposure chosen by the state (1 for a count, 0 for a structural zero) times exp(eta), which the compiler turns into a gate over the Poisson likelihood of the count, with a table that rules out a structural zero for a positive count. The rule is tested against the grid posterior in gates.test.ts.',
    ],
  }
}
