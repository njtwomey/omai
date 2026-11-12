/**
 * The recovery of the extended EP atoms. An extended atom is the twin of a basic one in which the numbers the basic atom
 * fixes (a noise scale, a precision, the strength of a prior) are unknown, with priors, and the data are simulated from a
 * stated truth with a seed. `atomRecovery` reads one run: for each quantity EP's mean, sd and 90% interval beside the true
 * value and whether it is inside. `atomCalibration` repeats it over many seeds, which EP's speed allows, and reports per
 * parameter the share of runs whose interval holds the truth (about 0.9 if the posterior is honest), the error of the mean in
 * posterior sds, and the ratio of EP's spread to the actual error: a simulation-based calibration of the atom, whose verdict
 * is checked against what the atom promises (`recovery`) and is never tuned to hide a poor parameter.
 */

import { OmaiError } from '../../base/index.ts'
import { betaincinv, gammaincinv, ndtri } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import type { AtomProblem, AtomRecoverySpec, AtomSupport, Belief, EpAtom } from './atom-types.ts'
import { atomRun } from './ep-atoms.ts'

/**
 * The credible interval of a belief on its support: a normal on the line, a gamma on the positive axis, a beta on the unit
 * interval, each matched to the belief's mean and standard deviation (the families EP carries for those).
 *
 * @param support - Where the quantity lives.
 * @param mean - The belief's mean.
 * @param sd - The belief's standard deviation.
 * @param level - The mass inside (default 0.9).
 * @returns The lower and upper ends.
 * @concept expectation-propagation
 * @index function The equal-tailed credible interval of an EP belief on its support.
 * @example
 * ```ts
 * import { beliefCredibleInterval } from 'omai-engine/probprog/ep'
 * const [lo, hi] = beliefCredibleInterval('real', 0, 1)
 * [lo.toFixed(3), hi.toFixed(3)] // => ['-1.645', '1.645']
 * ```
 */
export function beliefCredibleInterval(
  support: AtomSupport,
  mean: number,
  sd: number,
  level = 0.9,
): readonly [number, number] {
  const lo = (1 - level) / 2
  const hi = 1 - lo
  if (support === 'positive') {
    const shape = (mean / sd) ** 2
    const rate = mean / sd ** 2
    return [item(gammaincinv(shape, lo)) / rate, item(gammaincinv(shape, hi)) / rate]
  }
  if (support === 'unit') {
    const common = (mean * (1 - mean)) / sd ** 2 - 1
    const a = mean * common
    const b = (1 - mean) * common
    return [item(betaincinv(a, b, lo)), item(betaincinv(a, b, hi))]
  }
  const z = item(ndtri(hi))
  return [mean - z * sd, mean + z * sd]
}

/** One parameter of one run: EP's belief beside the truth. */
export interface RecoveryRow {
  readonly name: string
  readonly label: string
  readonly truth: number
  readonly mean: number
  readonly sd: number
  readonly lower: number
  readonly upper: number
  /** The error of the mean in posterior sds. */
  readonly z: number
  readonly covered: boolean
}

/**
 * EP's beliefs of a problem's quantities beside the true values they carry: the mean, sd, 90% interval, the error in
 * posterior sds and whether the truth is inside.
 *
 * @param problem - The problem (its quantities carry `truth`).
 * @param beliefs - EP's beliefs.
 * @param level - The interval's mass (default 0.9).
 * @returns A row per quantity that has a truth and a proper belief.
 * @concept expectation-propagation
 * @index function EP's belief of each parameter beside the truth it was simulated from: interval and coverage.
 * @example
 * ```ts
 * import { atomRecovery } from 'omai-engine/probprog/ep'
 * const q = { name: 'mu', label: 'mu', support: 'real' as const, truth: 1 }
 * const rows = atomRecovery({ quantities: [q] } as never, { mu: { mean: 1.2, sd: 0.5, proper: true } })
 * [rows[0]?.covered, rows[0]?.z.toFixed(1)] // => [true, '0.4']
 * ```
 */
export function atomRecovery(
  problem: Pick<AtomProblem, 'quantities'>,
  beliefs: Readonly<Record<string, Belief>>,
  level = 0.9,
): readonly RecoveryRow[] {
  const rows: RecoveryRow[] = []
  for (const q of problem.quantities) {
    const b = beliefs[q.name]
    if (q.truth === undefined || b === undefined || !b.proper) continue
    const [lower, upper] = beliefCredibleInterval(q.support, b.mean, b.sd, level)
    rows.push({
      name: q.name,
      label: q.label,
      truth: q.truth,
      mean: b.mean,
      sd: b.sd,
      lower,
      upper,
      z: (b.mean - q.truth) / b.sd,
      covered: q.truth >= lower && q.truth <= upper,
    })
  }
  return rows
}

/** One parameter over all the seeds. */
export interface CalibrationRow {
  /** The quantity's name, or the pool's label for a group of them. */
  readonly key: string
  readonly label: string
  /** How many (run, member) pairs went in. */
  readonly count: number
  /** The share of them whose 90% interval holds the truth. */
  readonly coverage: number
  /** The mean of the error in posterior sds (a bias shows here). */
  readonly meanZ: number
  /** The root mean square of the error in posterior sds (near 1 when the posterior sd is honest). */
  readonly zrms: number
  /** The root mean square error of the mean, in the parameter's units. */
  readonly rmse: number
  /** The root mean square of EP's posterior sd, in the parameter's units: compare with `rmse`. */
  readonly postSd: number
  /** The mean true value over the seeds (the stated truth, or the mean of the draws of it). */
  readonly truth: number
  readonly pass: boolean
}

/** The calibration of an extended atom over many seeds. */
export interface AtomCalibration {
  readonly seeds: number
  /** The runs EP completed with proper beliefs. */
  readonly ran: number
  readonly rows: readonly CalibrationRow[]
  readonly spec: AtomRecoverySpec
  /** Every parameter has coverage and error within the atom's promise, and every run completed. */
  readonly pass: boolean
}

/** One seed of a calibration: what EP recovered of each parameter (the rows of `atomRecovery`, with their pool), or none if EP failed. */
export interface CalibrationSample {
  readonly seed: number
  readonly rows: readonly (RecoveryRow & { readonly pool?: string })[]
}

/**
 * One seed of an extended atom's calibration: simulate with the seed, run EP lightly, and read each parameter's belief beside
 * its truth. A page calls this seed by seed to fill its calibration table as it goes.
 *
 * @param atom - An extended atom.
 * @param seed - The seed of the simulated data.
 * @param knobs - Knob values over the defaults (the seed is set by `seed`).
 * @returns The sample (no rows when EP was refused or failed).
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { calibrationSample, epAtom } from 'omai-engine/probprog/ep'
 * calibrationSample(epAtom('gaussian-learned'), 3).rows.map((r) => r.name) // => ['mu', 'sigma2']
 * ```
 */
export function calibrationSample(atom: EpAtom, seed: number, knobs: Config = {}): CalibrationSample {
  const problem = atom.problem({ ...atom.defaults, ...knobs, seed })
  const run = atomRun(problem, atom.ep ?? {}, { light: true })
  if (run.status !== 'ran') return { seed, rows: [] }
  const pool = new Map(problem.quantities.map((q) => [q.name, q.pool]))
  return {
    seed,
    rows: atomRecovery(problem, run.beliefs).map((r) => {
      const group = pool.get(r.name)
      return group === undefined ? r : { ...r, pool: group }
    }),
  }
}

/**
 * The calibration table of a set of samples: per parameter (members of a pool together) the share of runs whose 90% interval
 * holds the truth, the bias and root mean square of the error in posterior sds, and the actual error beside EP's own spread,
 * with the verdict against what the atom promises.
 *
 * @param spec - The atom's promise.
 * @param samples - The samples, one per seed.
 * @returns The calibration.
 * @concept expectation-propagation
 * @index function The calibration table of an extended EP atom from its seeds: coverage, bias and error per parameter.
 * @example
 * ```ts
 * import { calibrationTable } from 'omai-engine/probprog/ep'
 * const row = { name: 'mu', label: 'mu', truth: 1, mean: 1.1, sd: 0.2, lower: 0.7, upper: 1.5, z: 0.5, covered: true }
 * const c = calibrationTable({ seeds: 1, coverage: 0.8, zrms: 1.5, reason: '' }, [{ seed: 1, rows: [row] }])
 * c.rows[0]?.coverage // => 1
 * ```
 */
export function calibrationTable(spec: AtomRecoverySpec, samples: readonly CalibrationSample[]): AtomCalibration {
  type Acc = { label: string; z: number[]; sd: number[]; err: number[]; hit: number; truth: number[] }
  const acc = new Map<string, Acc>()
  let ran = 0
  for (const sample of samples) {
    if (sample.rows.length === 0) continue
    ran++
    for (const r of sample.rows) {
      const key = r.pool ?? r.name
      const a = acc.get(key) ?? { label: r.pool ?? r.label, z: [], sd: [], err: [], hit: 0, truth: [] }
      a.z.push(r.z)
      a.sd.push(r.sd)
      a.err.push(r.mean - r.truth)
      a.truth.push(r.truth)
      if (r.covered) a.hit++
      acc.set(key, a)
    }
  }
  const mean = (xs: readonly number[]): number => xs.reduce((p, c) => p + c, 0) / xs.length
  const rows = [...acc.entries()].map(([key, a]): CalibrationRow => {
    const coverage = a.hit / a.z.length
    const zrms = Math.sqrt(mean(a.z.map((v) => v * v)))
    return {
      key,
      label: a.label,
      count: a.z.length,
      coverage,
      meanZ: mean(a.z),
      zrms,
      rmse: Math.sqrt(mean(a.err.map((v) => v * v))),
      postSd: Math.sqrt(mean(a.sd.map((v) => v * v))),
      truth: mean(a.truth),
      pass: coverage >= spec.coverage && zrms <= spec.zrms,
    }
  })
  return {
    seeds: samples.length,
    ran,
    rows,
    spec,
    pass: ran === samples.length && rows.length > 0 && rows.every((r) => r.pass),
  }
}

/**
 * Repeat an extended atom over many seeds and summarise how EP recovers the truth: per parameter (groups of members pooled),
 * the share of runs whose 90% interval holds the true value, the bias and the root mean square of the error in posterior sds,
 * and the actual error beside EP's own spread. EP runs in milliseconds, so the table is a simulation-based calibration the page
 * can show.
 *
 * @param atom - An extended atom (it has `recovery`, a `seed` knob and quantities with truths).
 * @param options - The number of seeds (default the atom's), the first seed, and knob values over the defaults.
 * @returns The table and the verdict against the atom's promise.
 * @throws OmaiError with code `value` when the atom is not extended.
 * @concept expectation-propagation
 * @index function Calibration of an extended EP atom: interval coverage of the truth over many simulated seeds.
 * @example
 * ```ts
 * import { atomCalibration, epAtom } from 'omai-engine/probprog/ep'
 * const c = atomCalibration(epAtom('gaussian-learned'), { seeds: 10 })
 * c.rows.map((r) => r.key) // => ['mu', 'sigma2']
 * ```
 */
export function atomCalibration(
  atom: EpAtom,
  options: { readonly seeds?: number; readonly firstSeed?: number; readonly knobs?: Config } = {},
): AtomCalibration {
  const spec = atom.recovery
  if (spec === undefined) throw new OmaiError('value', `atomCalibration: '${atom.name}' is not an extended atom`)
  const seeds = options.seeds ?? spec.seeds
  const first = options.firstSeed ?? 1
  // banned-ok: non-numeric: one calibration record per seed
  const samples = Array.from({ length: seeds }, (_, s) => calibrationSample(atom, first + s, options.knobs ?? {}))
  return calibrationTable(spec, samples)
}
