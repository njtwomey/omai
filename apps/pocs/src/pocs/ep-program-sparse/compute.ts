import { activeSetScore, recoverySummary } from 'omai-engine/bayesian-examples'
import { modelRunFromTape } from 'omai-engine/exposition'
import { toFactorGraph, toStructure } from 'omai-engine/probprog/model'
import { type SpikeSlabPosterior, spikeSlabRegression } from 'omai-engine/probprog/validation'
import { toFlat } from 'omai-engine/tensor'
import { type ProgramCheck, sig } from '../../shared/ep/program-check.ts'
import {
  coefficients,
  defaultSetup,
  infer,
  type SparseSetup,
  simulate,
  sparseMixtureModel,
  sparseModel,
  studentTUnknownScale,
} from './program.ts'

export type { SparseSetup }
export { defaultSetup }

/** One model run by EP and read against the truth. */
export interface Fit {
  readonly bound: ReturnType<typeof infer>['bound']
  readonly run: ReturnType<typeof infer>['run']
  readonly ms: number
  readonly coef: ReturnType<typeof coefficients>
  /** The posterior mean of the noise precision (Gaussian likelihood only). */
  readonly precision: number | undefined
}

/** Everything the page and the test read of one setup. */
export interface SparseRun {
  readonly setup: SparseSetup
  readonly data: ReturnType<typeof simulate>
  readonly active: readonly boolean[]
  readonly gaussian: Fit
  readonly student: Fit
  /** The closed-form mixture prior (`mixture-site`), Gaussian likelihood: the coefficients' mean and sd. */
  readonly mixture: { readonly mean: readonly number[]; readonly sd: readonly number[] }
  /** The compiler's verdict on a Student-t likelihood with an unknown scale. */
  readonly refusal: string | undefined
}

function fit(model: ReturnType<typeof sparseModel>, data: ReturnType<typeof simulate>): Fit {
  const t0 = performance.now()
  const { bound, run, posterior } = infer(model, data)
  const ms = performance.now() - t0
  const tau = posterior.sites.includes('tau') ? Number(toFlat(posterior.mean('tau'))[0]) : undefined
  return { bound, run, ms, coef: coefficients(posterior), precision: tau }
}

/** The messages of a run as a record for the player. */
export const recordOf = (f: Fit) =>
  modelRunFromTape(toStructure(f.bound, { view: 'factor' }), { run: f.run.run() }, { maxSteps: 120, maxMessages: 6 })

/** Simulates a setup and runs EP on the models. */
export function runSparse(setup: SparseSetup): SparseRun {
  const data = simulate(setup)
  const gaussian = fit(sparseModel('gaussian', setup.noise), data)
  const student = fit(sparseModel('student-t', setup.noise), data)
  const { posterior } = infer(sparseMixtureModel() as never, data)
  const mixture = {
    mean: Array.from(toFlat(posterior.mean('w'))),
    sd: Array.from(toFlat(posterior.stddev('w'))),
  }
  const refused = toFactorGraph(studentTUnknownScale().bind({ X: data.X, y: data.y }))
  return {
    setup,
    data,
    active: data.truth.map((v) => v !== 0),
    gaussian,
    student,
    mixture,
    refusal: refused.issues.length === 0 ? undefined : refused.issues.map((i) => i.reason).join('; '),
  }
}

/** The largest number of coefficients whose subsets are enumerated for the exact reference. */
export const MAX_EXACT = 14

/** The exact posterior of the Gaussian-likelihood model (all subsets, the precision on a grid), when the coefficients are few enough. */
export function exactSparse(run: SparseRun): SpikeSlabPosterior | undefined {
  if (run.setup.features > MAX_EXACT) return undefined
  return spikeSlabRegression({
    X: run.data.X,
    y: run.data.y,
    inclusion: 0.2,
    slab: 3,
    spike: 0.05,
    noise: { shape: 2, rate: 0.5 },
  })
}

/** The recovery check of a run. */
export function checkSparse(run: SparseRun, exact: SpikeSlabPosterior | undefined): ProgramCheck {
  const { setup, data, active, gaussian, student } = run
  const found = (f: Fit): ReturnType<typeof activeSetScore> => activeSetScore(f.coef.pip, active)
  const k = active.filter(Boolean).length
  const rows: ProgramCheck['rows'][number][] = []
  const clean = setup.outliers === 0
  const truthRows = (name: string, f: Fit, gating: boolean): void => {
    const s = found(f)
    const r = recoverySummary({ truth: data.truth, ...f.coef })
    rows.push(
      {
        label: `${name}: active set (P(slab) above 0.5)`,
        value: `found ${s.found} of ${k}, ${s.falsePositives} false positive${s.falsePositives === 1 ? '' : 's'}, AUC ${sig(s.auc, 3)}`,
        tolerance: 'all found, at most 1 false positive',
        pass: s.missed === 0 && s.falsePositives <= 1,
        gating,
      },
      {
        label: `${name}: the true coefficients inside the 90% intervals`,
        value: `${sig(100 * r.coverage, 3)}%, rmse ${sig(r.rmse, 2)}`,
        tolerance: 'at least 75% (nominal 90%)',
        pass: r.coverage >= 0.75,
        gating,
      },
    )
  }
  truthRows('Student-t, known scale', student, true)
  truthRows(
    clean ? 'Gaussian, unknown precision' : 'Gaussian, unknown precision (misspecified by the outliers)',
    gaussian,
    clean,
  )
  if (gaussian.precision !== undefined) {
    const sd = 1 / Math.sqrt(gaussian.precision)
    rows.push({
      label: 'Gaussian: the noise sd EP infers against the true inlier noise',
      value: `${sig(sd, 3)} against ${setup.noise}`,
      tolerance: 'within 25%',
      pass: Math.abs(sd / setup.noise - 1) <= 0.25,
      gating: clean,
    })
  }
  if (exact === undefined) {
    return {
      rows,
      notes: [
        `There are ${setup.features} coefficients, so the exact posterior (2^${setup.features} subsets) is not enumerated here (at most ${MAX_EXACT}); the check is against the truth only.`,
        ...notesOf(run),
      ],
    }
  }
  const ref = recoverySummary({
    truth: data.truth,
    mean: gaussian.coef.mean,
    sd: gaussian.coef.sd,
    lower: gaussian.coef.lower,
    upper: gaussian.coef.upper,
    reference: { mean: exact.mean, sd: exact.sd },
  })
  const pipGap = Math.max(...gaussian.coef.pip.map((p, j) => Math.abs(p - (exact.inclusion[j] as number))))
  const mixtureGap = Math.max(...run.mixture.mean.map((m, j) => Math.abs(m - (gaussian.coef.mean[j] ?? 0))))
  rows.push(
    {
      label: 'Gaussian: EP against the exact posterior, P(slab) of every coefficient',
      value: `largest difference ${sig(pipGap, 2)}`,
      tolerance: 'at most 0.1',
      pass: pipGap <= 0.1,
    },
    {
      label: 'Gaussian: EP against the exact posterior, coefficient means',
      value: `largest error ${sig(ref.reference?.maxMeanZ ?? Number.NaN, 2)} exact sds`,
      tolerance: 'at most 0.5 sd',
      pass: (ref.reference?.maxMeanZ ?? Number.POSITIVE_INFINITY) <= 0.5,
    },
    {
      label: 'Gaussian: EP against the exact posterior, coefficient sds',
      value: `ratio EP / exact from ${sig(ref.reference?.sdRatio[0] ?? Number.NaN, 2)} to ${sig(ref.reference?.sdRatio[1] ?? Number.NaN, 2)}`,
      tolerance: 'between 0.6 and 1.5',
      pass: (ref.reference?.sdRatio[0] ?? 0) >= 0.6 && (ref.reference?.sdRatio[1] ?? 9) <= 1.5,
    },
    {
      label: 'Gaussian: the noise precision, EP against exact',
      value: `${sig(gaussian.precision ?? Number.NaN, 3)} against ${sig(exact.precision, 3)}`,
      tolerance: 'within 10%',
      pass: Math.abs((gaussian.precision ?? 0) / exact.precision - 1) <= 0.1,
    },
    {
      label: 'The prior written as the closed-form mixture (d.normalSpikeSlab) against the explicit gates',
      value: `largest difference of a coefficient mean ${sig(mixtureGap, 2)}`,
      tolerance: 'at most 0.1',
      pass: mixtureGap <= 0.1,
    },
  )
  return { rows, notes: notesOf(run) }
}

function notesOf(run: SparseRun): string[] {
  const out = [
    'EP has no exact reference for the Student-t model (its likelihood is not Gaussian, so the subsets cannot be summed in closed form): it is held to the truth. The Gaussian model is held to the exact posterior as well.',
  ]
  if (run.refusal !== undefined)
    out.push(
      `An unknown scale for the Student-t likelihood (a gamma prior on it) is refused by the compiler: "${run.refusal}". The noise level is therefore known to the Student-t model (the true inlier sd), while the Gaussian model infers it. This is a gap of the library, not of the program.`,
    )
  return out
}
