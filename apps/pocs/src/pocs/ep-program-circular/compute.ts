import { logSumExpOf } from 'omai-engine/base'
import { type ModelRun, modelRunFromTape } from 'omai-engine/exposition'
import type { BoundModel } from 'omai-engine/probprog/model'
import { toStructure } from 'omai-engine/probprog/model'
import { foldIn, key, uniform } from 'omai-engine/random'
import { periodicCopies, threeCopiesBound, wrappedNormalLogPdf } from 'omai-engine/stats'
import { toFlat } from 'omai-engine/tensor'
import { type ProgramCheck, sig } from '../../shared/ep/program-check.ts'
import { type Estimate, recovery } from '../../shared/ep/recovery.ts'
import type { EpHooks } from '../../shared/ep/sweeps.ts'
import { dataOf, infer, model, PERIOD, simulate, TRUTHS, type TruthName, type Variant } from './program.ts'

/** What the page changes. */
export interface CircularSetup {
  readonly seed: number
  readonly truth: TruthName
  /** How many components the model has (1 to 3). */
  readonly components: number
  readonly draws: number
  readonly variant: Variant
}

export const defaultSetup: CircularSetup = { seed: 1, truth: 'one', components: 1, draws: 200, variant: 'data' }

/** The parameters EP found: the means, the shared sd and the weights, each with its 90% interval. */
export interface Found {
  readonly means: readonly Estimate[]
  readonly sd: Estimate
  readonly weights: readonly Estimate[]
}

/** The estimates of a posterior over the three sites: `quantiles` runs over (5%, 50%, 95%). */
export function foundOf(posterior: Awaited<ReturnType<typeof infer>>['posterior'], components: number): Found {
  const probs = [0.05, 0.5, 0.95]
  const row = (site: string, k: number): number[] =>
    Array.from(toFlat(posterior.quantiles(site, probs))).slice(3 * k, 3 * k + 3)
  const mean = Array.from(toFlat(posterior.mean('mean')))
  const spread = Array.from(toFlat(posterior.stddev('mean')))
  const means = mean.map((m, k): Estimate => {
    const q = row('mean', k)
    return { mean: m, sd: spread[k] as number, lower: q[0] as number, upper: q[2] as number }
  })
  // The precision's quantiles reversed are the sd's.
  const [t0, t1, t2] = row('tau', 0) as [number, number, number]
  const sdMean = 1 / Math.sqrt(Number(toFlat(posterior.mean('tau'))[0]))
  const sd: Estimate = {
    mean: sdMean,
    sd: (1 / Math.sqrt(t0) - 1 / Math.sqrt(t2)) / 3.29,
    lower: 1 / Math.sqrt(t2),
    upper: 1 / Math.sqrt(t0),
  }
  void t1
  if (components === 1) return { means, sd, weights: [{ mean: 1, sd: 0, lower: 1, upper: 1 }] }
  const w = Array.from(toFlat(posterior.mean('weight')))
  const ws = Array.from(toFlat(posterior.stddev('weight')))
  const weights = w.map((m, k): Estimate => {
    const q = row('weight', k)
    return { mean: m, sd: ws[k] as number, lower: q[0] as number, upper: q[2] as number }
  })
  return { means, sd, weights }
}

/**
 * One fit: the clock times, how EP ended and what it found. Plain data, so a worker can send it to the page: the bound model
 * (for the diagram) is rebuilt on the page with {@link boundOf}, and the messages of the run are the `record`.
 */
export interface CircularRun {
  readonly setup: CircularSetup
  readonly hours: readonly number[]
  /** How EP ended (`converged` or `stopped`; undefined for a run stopped by the page) and the sweeps it took. */
  readonly fit: { readonly run: { readonly status: string | undefined; readonly t: number } }
  readonly ms: number
  readonly found: Found
  /** For each reading, the probability that it is read as its early or late copy (the reading wrapped round midnight). */
  readonly wrapped: readonly number[]
  /** The messages of the run as a record for the player. */
  readonly record: ModelRun | undefined
}

/** The bound model of a run, for its diagram: the same model and data `infer` binds. */
export const boundOf = (run: CircularRun): BoundModel =>
  model(run.setup.variant, run.setup.components).bind(dataOf(run.setup.variant, run.setup.components, run.hours))

/**
 * Simulates the setup's clock times and fits its variant by EP. The hooks report each sweep and can stop the fit, which then
 * keeps the sweeps done.
 */
export async function runCircular(
  setup: CircularSetup,
  hours?: readonly number[],
  hooks: EpHooks = {},
): Promise<CircularRun> {
  const data = hours ?? simulate(setup.seed, setup.truth, setup.draws)
  const t0 = performance.now()
  const fit = await infer(setup.variant, setup.components, data, hooks)
  const ms = performance.now() - t0
  const found = foundOf(fit.posterior, setup.components)
  const record = modelRunFromTape(
    toStructure(fit.bound, { view: 'factor' }),
    { run: fit.run.run() },
    { maxSteps: 120, maxMessages: 6 },
  )
  return {
    setup,
    hours: data,
    fit: { run: { status: fit.run.status, t: fit.run.t } },
    ms,
    found,
    wrapped: wrappedShare(setup, fit.posterior, data.length),
    record,
  }
}

/** For each reading, the probability that EP reads it as an early or a late copy (a shift of a period), for the two models with copies. */
function wrappedShare(
  setup: CircularSetup,
  posterior: Awaited<ReturnType<typeof infer>>['posterior'],
  rows: number,
): number[] {
  if (setup.variant !== 'means' && setup.variant !== 'data') return []
  const marginal = posterior.marginal('copy')
  const share = new Array<number>(rows).fill(0)
  for (let s = 0; s < 3 * setup.components; s++) {
    if (s % 3 === 1) continue
    toFlat(marginal.prob(s)).forEach((p, i) => {
      share[i] = (share[i] as number) + p
    })
  }
  return share
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const logNormal = (x: number, mu: number, s: number): number => -0.5 * ((x - mu) / s) ** 2 - Math.log(s) - LOG_SQRT_2PI

/** The parameters of a mixture of normals on the clock: weights, means and one sd. */
export interface Mixture {
  readonly weights: readonly number[]
  readonly means: readonly number[]
  readonly sd: number
}

export const mixtureOf = (f: Found): Mixture => ({
  weights: f.weights.map((w) => w.mean),
  means: f.means.map((m) => m.mean),
  sd: f.sd.mean,
})

/** The truth as a mixture (with each component's own scale), for its exact wrapped density. */
export const truthMixture = (
  truth: TruthName,
): { weights: readonly number[]; means: readonly number[]; scales: readonly number[] } => TRUTHS[truth]

/** The exact wrapped density of the truth at `x` (ten periods each way: an error far below anything drawn). */
export const truthLogDensity = (truth: TruthName, x: number): number => {
  const t = TRUTHS[truth]
  return logSumExpOf(
    t.weights.map((w, k) => Math.log(w) + wrappedNormalLogPdf(x, t.means[k] as number, t.scales[k] as number, PERIOD)),
  )
}

/**
 * What a fit says the density of a clock time is: the wrapped mixture for the two models that wrap, the plain mixture of
 * normals for the two that do not (they know nothing of midnight).
 */
export function fitLogDensity(variant: Variant, mix: Mixture, x: number): number {
  const wrap = variant === 'means' || variant === 'data'
  return logSumExpOf(
    mix.weights.map(
      (w, k) =>
        Math.log(w) +
        (wrap
          ? wrappedNormalLogPdf(x, mix.means[k] as number, mix.sd, PERIOD, 1)
          : logNormal(x, mix.means[k] as number, mix.sd)),
    ),
  )
}

/** Fresh readings from the truth, for the held-out score (the same simulator, a seed far from the training one). */
export const heldOutHours = (truth: TruthName, seed: number): number[] => simulate(seed + 7919, truth, 3000)

/** How far, in nats per reading, the fit's held-out log density is below the truth's own: zero would be the truth. */
export function heldOutLoss(variant: Variant, truth: TruthName, mix: Mixture, held: readonly number[]): number {
  let gap = 0
  for (const x of held) gap += truthLogDensity(truth, x) - fitLogDensity(variant, mix, x)
  return gap / held.length
}

/** The circular distance between two clock times. */
export const circularGap = (a: number, b: number): number => {
  const r = Math.abs(a - b) % PERIOD
  return Math.min(r, PERIOD - r)
}

/** For each truth component, the fitted component whose mean is nearest on the circle (and so the fit's means, weights, in truth order). */
export function matchToTruth(truth: TruthName, found: Found): readonly (number | undefined)[] {
  const t = TRUTHS[truth]
  const used = new Set<number>()
  return t.means.map((m) => {
    let best: number | undefined
    found.means.forEach((f, k) => {
      if (used.has(k)) return
      if (best === undefined || circularGap(f.mean, m) < circularGap((found.means[best] as Estimate).mean, m)) best = k
    })
    if (best !== undefined) used.add(best)
    return best
  })
}

/** The truth's scale as one number a shared sd can be held to: the weighted root mean square of the scales. */
export const truthScale = (truth: TruthName): number => {
  const t = TRUTHS[truth]
  return Math.sqrt(t.weights.reduce((s, w, k) => s + w * (t.scales[k] as number) ** 2, 0))
}

/** The log likelihood of the readings under the same wrapped mixture, three ways: shift the means, shift the data, extra rows. */
export interface Losses {
  readonly shiftMeans: number
  readonly shiftData: number
  readonly extraRows: number
}

/**
 * The total log likelihood of the readings under a mixture, computed the three ways of the post, at given parameters: the
 * three copies of each component's mean (the data fixed), the three copies of each reading (one normal fixed), and the copies
 * scored as separate readings (a sum of the logs rather than the log of the sum).
 */
export function lossesAt(hours: readonly number[], mix: Mixture): Losses {
  const third = Math.log(1 / 3)
  let shiftMeans = 0
  let shiftData = 0
  let extraRows = 0
  for (const x of hours) {
    shiftMeans += logSumExpOf(
      mix.weights.flatMap((w, k) =>
        [-1, 0, 1].map((c) => Math.log(w) + third + logNormal(x, (mix.means[k] as number) + c * PERIOD, mix.sd)),
      ),
    )
    shiftData += logSumExpOf(
      mix.weights.flatMap((w, k) =>
        periodicCopies(x, PERIOD).map((copy) => Math.log(w) + third + logNormal(copy, mix.means[k] as number, mix.sd)),
      ),
    )
    for (const copy of periodicCopies(x, PERIOD))
      extraRows += logSumExpOf(mix.weights.map((w, k) => Math.log(w) + logNormal(copy, mix.means[k] as number, mix.sd)))
  }
  return { shiftMeans, shiftData, extraRows }
}

/** Mixtures with random means, sds and weights (a fixed seed): the points where the two losses are compared. */
export function randomMixtures(count: number, components: number, seed = 5): Mixture[] {
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 1), [count * (2 * components + 1)])))
  return Array.from({ length: count }, (_, i) => {
    const at = (j: number): number => u[i * (2 * components + 1) + j] as number
    const raw = Array.from({ length: components }, (_, k) => 0.2 + at(k))
    const total = raw.reduce((a, b) => a + b, 0)
    return {
      weights: raw.map((r) => r / total),
      means: Array.from({ length: components }, (_, k) => PERIOD * at(components + k)),
      sd: 1 + 5 * at(2 * components),
    }
  })
}

/** The largest difference between the two ways, over random parameters, in nats (the post measured 9e-16 per loss). */
export function lossGap(hours: readonly number[], components: number): number {
  let worst = 0
  for (const mix of randomMixtures(6, components)) {
    const l = lossesAt(hours, mix)
    worst = Math.max(worst, Math.abs(l.shiftMeans - l.shiftData))
  }
  return worst
}

/** The relative error the three-copy density is allowed at this sd (the post's bound). */
export const copiesError = (sd: number): number => threeCopiesBound(sd, PERIOD)

/** How EP's posteriors of the two forms compare, and how long each took. */
export interface Equivalence {
  readonly means: CircularRun
  readonly data: CircularRun
  /** Largest gap between the two fits' component means, in hours. */
  readonly meanGap: number
  readonly sdGap: number
  readonly weightGap: number
  readonly lossGapAtFit: number
  readonly lossGapRandom: number
}

/** Fits both forms to the same readings and compares them. */
export async function equivalence(
  setup: CircularSetup,
  current?: CircularRun,
  hooks: (stage: string) => EpHooks = () => ({}),
): Promise<Equivalence> {
  const hours = current?.hours
  const other = (variant: Variant): CircularRun | Promise<CircularRun> =>
    current !== undefined && current.setup.variant === variant
      ? current
      : runCircular(
          { ...setup, variant },
          hours,
          hooks(`EP, ${variant === 'means' ? 'shift the means' : 'shift the data'}`),
        )
  const means = await other('means')
  const data = await other('data')
  const meanGap = Math.max(
    ...means.found.means.map((m, k) => circularGap(m.mean, (data.found.means[k] as Estimate).mean)),
  )
  const wGap = Math.max(
    ...means.found.weights.map((w, k) => Math.abs(w.mean - (data.found.weights[k] as Estimate).mean)),
  )
  const at = mixtureOf(means.found)
  const l = lossesAt(means.hours, at)
  return {
    means,
    data,
    meanGap,
    sdGap: Math.abs(means.found.sd.mean - data.found.sd.mean),
    weightGap: wGap,
    lossGapAtFit: Math.abs(l.shiftMeans - l.shiftData),
    lossGapRandom: lossGap(means.hours, setup.components),
  }
}

/** The check of a run against the truth: what the post says each model should and should not do. */
export function checkCircular(run: CircularRun, held: readonly number[], eq?: Equivalence): ProgramCheck {
  const { setup, found } = run
  const truth = TRUTHS[setup.truth]
  const right = setup.variant === 'means' || setup.variant === 'data'
  const sameSize = setup.components === truth.means.length
  const matched = matchToTruth(setup.truth, found)
  const rows: ProgramCheck['rows'][number][] = []
  rows.push({
    label: 'EP converged',
    value: `${run.fit.run.status ?? 'stopped'} after ${run.fit.run.t} sweeps`,
    tolerance: 'converged',
    pass: run.fit.run.status === 'converged',
  })
  const mix = mixtureOf(found)
  const loss = heldOutLoss(setup.variant, setup.truth, mix, held)
  rows.push({
    label: 'held-out log density, below the truth’s own (nats per reading)',
    value: sig(loss, 2),
    tolerance: right
      ? `at most ${truth.means.length === 1 ? '0.03' : '0.1 (one shared sd for two scales costs some)'}`
      : 'shown: this model ignores midnight or counts it three times',
    pass: loss <= (truth.means.length === 1 ? 0.03 : 0.1),
    gating: right && sameSize,
  })
  if (sameSize) {
    const errors = truth.means.map((m, k) => {
      const f = found.means[matched[k] as number] as Estimate
      return circularGap(f.mean, m)
    })
    rows.push({
      label: 'the true means against EP, round the circle (hours)',
      value: truth.means.map((m, k) => `${m} h: off by ${sig(errors[k] as number, 2)} h`).join(', '),
      tolerance: 'each within 1.5 hours',
      pass: errors.every((e) => e <= 1.5),
      gating: right,
    })
    const scale = truthScale(setup.truth)
    rows.push({
      label: 'the shared sd against the truth’s root mean square scale (hours)',
      value: `${sig(found.sd.mean, 3)} against ${sig(scale, 3)}`,
      tolerance: 'within 25%',
      pass: Math.abs(found.sd.mean - scale) / scale <= 0.25,
      gating: right,
    })
    if (truth.weights.length > 1) {
      const wErr = truth.weights.map((w, k) => Math.abs((found.weights[matched[k] as number] as Estimate).mean - w))
      rows.push({
        label: 'the weights against the truth',
        value: truth.weights.map((w, k) => `${w}: off by ${sig(wErr[k] as number, 2)}`).join(', '),
        tolerance: 'each within 0.12',
        pass: wErr.every((e) => e <= 0.12),
        gating: right,
      })
    }
    const r = recovery(
      truth.means.map((_, k) => found.means[matched[k] as number] as Estimate),
      [...truth.means],
    )
    rows.push({
      label: 'the true means inside EP’s 90% intervals; the errors in posterior sds (rms)',
      value: `${sig(100 * r.coverage, 3)}% inside; ${sig(r.zrms, 2)} sds`,
      tolerance: 'shown: EP is overconfident on a mixture',
      pass: r.coverage >= 0.8 && r.zrms <= 1.6,
      gating: false,
    })
  }
  if (eq !== undefined) {
    rows.push(
      {
        label:
          'shift the means against shift the data: the log likelihood, at the fitted parameters and at random ones (nats)',
        value: `${sig(eq.lossGapAtFit, 2)} and ${sig(eq.lossGapRandom, 2)}`,
        tolerance: 'at most 1e-9',
        pass: eq.lossGapAtFit <= 1e-9 && eq.lossGapRandom <= 1e-9,
      },
      {
        label: 'shift the means against shift the data: the EP fits (largest mean gap in hours, sd gap, weight gap)',
        value: `${sig(eq.meanGap, 2)} h, ${sig(eq.sdGap, 2)} h, ${sig(eq.weightGap, 2)}`,
        tolerance:
          'means within 0.5 h, the sd and the weights within 0.25: the arithmetic is the same, EP’s projections are not quite',
        pass: eq.meanGap <= 0.5 && eq.sdGap <= 0.25 && eq.weightGap <= 0.25,
      },
    )
  }
  return {
    rows,
    notes: [
      'The weights, the component of each reading and the copy of each reading (early, on time or late) are discrete latents; the three copies of a component are equally likely, so a reading near midnight is read as the early or the late copy and the one normal is not pulled across the day.',
      'One sd is shared by every component: a random sd per component, chosen by the same discrete variable as the mean, is a gate over likelihoods with a random scale, which the compiler maps only with a known scale. With the sd chosen separately from the mean EP lost which component both came from and one sd ran off to infinity (measured). With two components of different scales the shared sd is a compromise.',
    ],
  }
}

/** One seed of the calibration: does EP's 90% interval cover the truth, for each parameter of the matched fit. */
export interface CalibrationRow {
  readonly seed: number
  readonly covered: readonly boolean[]
  readonly converged: boolean
  readonly zrms: number
}

/** The parameter names a calibration reports, in order: the truth's means, then (with two components) the weight of the first, then the sd. */
export function parameterNames(truth: TruthName): string[] {
  const t = TRUTHS[truth]
  return [...t.means.map((m) => `mean ${m} h`), ...(t.weights.length > 1 ? ['weight of the first'] : []), 'sd']
}

/** Many seeds, one setup: the coverage of the truth by EP's 90% intervals, parameter by parameter. */
export async function calibrate(
  setup: CircularSetup,
  seeds: readonly number[],
  onSeed?: (row: CalibrationRow) => void,
  hooks: (seed: number) => EpHooks = () => ({}),
  stopped: () => boolean = () => false,
): Promise<CalibrationRow[]> {
  const t = TRUTHS[setup.truth]
  const out: CalibrationRow[] = []
  for (const seed of seeds) {
    const run = await runCircular({ ...setup, seed }, undefined, hooks(seed))
    if (stopped()) break
    const matched = matchToTruth(setup.truth, run.found)
    const est: Estimate[] = [
      ...t.means.map((_, k) => run.found.means[matched[k] as number] as Estimate),
      ...(t.weights.length > 1 ? [run.found.weights[matched[0] as number] as Estimate] : []),
      run.found.sd,
    ]
    const truthValues = [...t.means, ...(t.weights.length > 1 ? [t.weights[0] as number] : []), truthScale(setup.truth)]
    const r = recovery(est, truthValues)
    const row: CalibrationRow = {
      seed,
      covered: est.map((e, i) => (truthValues[i] as number) >= e.lower && (truthValues[i] as number) <= e.upper),
      converged: run.fit.run.status === 'converged',
      zrms: r.zrms,
    }
    out.push(row)
    onSeed?.(row)
  }
  return out
}
