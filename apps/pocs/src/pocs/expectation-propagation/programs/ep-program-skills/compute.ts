import { modelRunFromTape } from 'omai-engine/exposition'
import { trueskillUpdate } from 'omai-engine/pgm'
import { recoverySummary } from 'omai-engine/probprog/ep'
import { nutsMethod } from 'omai-engine/probprog/inference'
import { toStructure } from 'omai-engine/probprog/model'
import { spearman } from 'omai-engine/stats'
import { item, sum, tensor } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import { type ProgramCheck, sig } from '../../../../shared/ep/program-check.ts'
import { type EpHooks, stepSession } from '../../../../shared/ep/sweeps.ts'
import { defaultSetup, infer, type SkillsSetup, simulate, skillsOf, tournamentModel } from './program.ts'

export type { SkillsSetup }
export { defaultSetup }

/** One simulation, run by EP. */
export interface SkillsRun {
  readonly setup: SkillsSetup
  readonly data: ReturnType<typeof simulate>
  readonly fit: Awaited<ReturnType<typeof infer>>
  readonly ms: number
  readonly found: ReturnType<typeof skillsOf>
  /** The ratings of TrueSkill in one pass over the games (the library's `pgm` update), for comparison. */
  readonly trueskill: readonly number[]
}

/** The long reference run: NUTS on the same model. */
export interface SkillsReference {
  readonly mean: readonly number[]
  readonly sd: readonly number[]
  readonly tau: { readonly mean: number; readonly sd: number }
  readonly ms: number
}

/** The messages of the run as a record for the player. */
export const recordOf = (run: SkillsRun) =>
  modelRunFromTape(
    toStructure(run.fit.bound, { view: 'factor' }),
    { run: run.fit.run.run() },
    { maxSteps: 120, maxMessages: 6 },
  )

/** TrueSkill by one pass over the games in order (the prior is the spread of the truth, the performance noise 1, no draws, no drift). */
function oneTrueskillPass(data: ReturnType<typeof simulate>, spread: number): number[] {
  // banned-ok: non-numeric: one rating record per player
  const ratings = Array.from({ length: data.players }, () => ({ mu: 0, sigma: spread }))
  const env = { mu: 0, sigma: spread, beta: 1, tau: 0, drawProbability: 1e-9 }
  data.a.forEach((a, g) => {
    const b = data.b[g] ?? 0
    const first = data.win[g] === 1
    const [w, l] = trueskillUpdate(
      [ratings[first ? a : b] ?? { mu: 0, sigma: spread }],
      [ratings[first ? b : a] ?? { mu: 0, sigma: spread }],
      false,
      env,
    )
    ratings[first ? a : b] = w?.[0] ?? { mu: 0, sigma: spread }
    ratings[first ? b : a] = l?.[0] ?? { mu: 0, sigma: spread }
  })
  return ratings.map((r) => r.mu)
}

/** Simulates a setup and runs EP on the tournament. */
export async function runSkills(setup: SkillsSetup, hooks: EpHooks = {}): Promise<SkillsRun> {
  const data = simulate(setup)
  const t0 = performance.now()
  const fit = await infer(data, hooks)
  const ms = performance.now() - t0
  return { setup, data, fit, ms, found: skillsOf(fit.posterior), trueskill: oneTrueskillPass(data, setup.spread) }
}

/**
 * The long NUTS run of the same model (two chains), as the reference EP is checked against. The hooks report each iteration
 * and can stop it; a run stopped before it has draws gives undefined.
 */
export async function referenceSkills(
  data: ReturnType<typeof simulate>,
  hooks: EpHooks = {},
): Promise<SkillsReference | undefined> {
  const t0 = performance.now()
  const bound = tournamentModel.bind({ a: data.a, b: data.b, win: data.win }, { sizes: { player: data.players } })
  const method = nutsMethod(bound)
  const s = session(method, {
    config: { numChains: 2, numWarmup: 300, numDraws: 600 },
    seed: 1,
    steps: 5000,
    mode: 'live',
  })
  await stepSession(s, 300 + 600 + 2, hooks)
  const posterior = method.posterior?.(s.state, s.config)
  if (posterior === undefined || (posterior.draws?.['skill']?.shape[0] ?? 0) < 4) return undefined
  const found = skillsOf(posterior)
  return { mean: found.mean, sd: found.sd, tau: found.tau, ms: performance.now() - t0 }
}

/** The share of the true skills inside the 90% intervals of a run's means and sds. */
export const coverageOf = (
  truth: readonly number[],
  run: { readonly mean: readonly number[]; readonly sd: readonly number[] },
): number => truth.filter((t, i) => Math.abs(t - (run.mean[i] ?? 0)) <= 1.645 * (run.sd[i] ?? 0)).length / truth.length

const rho = (x: readonly number[], y: readonly number[]): number => Number(spearman([...x], [...y]).statistic)

/** The recovery check of a run. */
export function checkSkills(run: SkillsRun, reference: SkillsReference): ProgramCheck {
  const { found, data } = run
  const r = recoverySummary({
    truth: data.skills,
    mean: found.mean,
    sd: found.sd,
    lower: found.lower,
    upper: found.upper,
    reference,
  })
  const epRho = rho(found.mean, data.skills)
  const nutsRho = rho(reference.mean, data.skills)
  const sdRatio = found.sd.map((s, i) => s / (reference.sd[i] ?? 1))
  const meanSd = item(sum(tensor(sdRatio))) / sdRatio.length
  const rows: ProgramCheck['rows'][number][] = [
    {
      label: 'the true skills against EP\u2019s 90% intervals',
      value: `${sig(100 * r.coverage, 3)}% of ${data.players} players inside (the long run\u2019s own intervals: ${sig(100 * coverageOf(data.skills, reference), 3)}%), largest error ${sig(r.maxZ, 2)} posterior sds`,
      tolerance: 'every error within 3.5 sds (the coverage is pooled over seeds in the test)',
      pass: r.maxZ <= 3.5,
    },
    {
      label: 'the true ordering of the players against EP’s (Spearman)',
      value: `${sig(epRho, 3)}; the long reference run: ${sig(nutsRho, 3)}`,
      tolerance: 'within 0.05 of the reference run’s',
      pass: epRho >= nutsRho - 0.05,
    },
    {
      label: 'EP against the long NUTS run: the posterior means of the skills',
      value: `largest error ${sig(r.reference?.maxMeanZ ?? Number.NaN, 2)} reference sds`,
      tolerance: 'at most 0.5 sd',
      pass: (r.reference?.maxMeanZ ?? Number.POSITIVE_INFINITY) <= 0.5,
    },
    {
      label: 'EP against the long NUTS run: the posterior sds of the skills',
      value: `ratio EP / NUTS from ${sig(r.reference?.sdRatio[0] ?? Number.NaN, 3)} to ${sig(r.reference?.sdRatio[1] ?? Number.NaN, 3)}, on average ${sig(meanSd, 3)}`,
      tolerance: 'between 0.7 and 1.3',
      pass: (r.reference?.sdRatio[0] ?? 0) >= 0.7 && (r.reference?.sdRatio[1] ?? 9) <= 1.3,
    },
    {
      label: 'the precision of the skills (the hierarchical gamma), EP against NUTS',
      value: `${sig(found.tau.mean, 3)} ± ${sig(found.tau.sd, 2)} against ${sig(reference.tau.mean, 3)} ± ${sig(reference.tau.sd, 2)} (true ${sig(1 / run.setup.spread ** 2, 3)})`,
      tolerance: 'means within 0.5 NUTS sd',
      pass: Math.abs(found.tau.mean - reference.tau.mean) <= 0.5 * reference.tau.sd,
    },
    {
      label: 'EP’s ordering against one pass of TrueSkill (the library’s pgm update, the true spread as its prior)',
      value: `Spearman ${sig(rho(found.mean, run.trueskill), 3)}`,
      tolerance: 'information only',
      pass: true,
      gating: false,
    },
  ]
  return {
    rows,
    notes: [
      `EP’s skill sds are on average ${sig(100 * meanSd, 3)}% of the NUTS run’s: ${meanSd < 0.95 ? 'EP is a little overconfident here (the skills of players who met are negatively correlated, which one Gaussian per skill does not keep)' : 'EP matches the spread'}. The reference run took ${sig(reference.ms / 1000, 2)} s in a worker.`,
      'The draw margin of TrueSkill (a win, a draw or a loss on the skill difference) is not part of this program: it needs an ordinal likelihood on a difference of skills. The likelihood registry has `ordinal-probit`, but no distribution of the model language compiles to it, so the model is written with wins and losses only.',
      'The skills are identified through their differences and by their prior, centred at zero: a tournament says who is better, and by how much, but the prior sets where zero is.',
    ],
  }
}
