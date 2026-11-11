// The calibration of the program over many seeds, shared by the page and the tests (not part of the program shown): for each seed
// the data are simulated from the truth, EP and the Gibbs sampler are run on them, and the belief about each parameter is held
// against the true value. A run is `agrees` when EP's regime probabilities are within 0.05 of the sampler's on average (it found
// the same reading of the data); a run is `barely` when some regime occurs in fewer than 6% of the steps.
import { type AtomQuantity, atomRecovery, type Belief, type CalibrationSample } from 'omai-engine/bayesian-examples'
import { infer, RATES, simulate, transitions } from './program.ts'
import { gibbsReference, type Reference } from './reference.ts'

export const NAMES = ['low', 'middle', 'high']

/** The parameters checked, with the truth of the simulation (`stay` the probability of staying in a regime). */
export function quantities(stay: number): AtomQuantity[] {
  const A = transitions(stay)
  const out: AtomQuantity[] = NAMES.map((name, k) => ({
    name: `rate[${k}]`,
    label: `rate of the ${name} regime`,
    support: 'positive',
    truth: RATES[k] as number,
  }))
  for (let i = 0; i < 3; i++)
    out.push({
      name: `A[${4 * i}]`,
      label: `stay probability, ${NAMES[i]} regime`,
      support: 'unit',
      truth: (A[i] as number[])[i] as number,
    })
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      if (i !== j)
        out.push({
          name: `A[${3 * i + j}]`,
          label: `${NAMES[i]} to ${NAMES[j]}`,
          support: 'unit',
          truth: (A[i] as number[])[j] as number,
          pool: 'the six probabilities of moving from one regime to another',
        })
  return out
}

/** One seed of the calibration. */
export interface SeedRun {
  readonly seed: number
  readonly status: string
  /** The steps spent in each regime. */
  readonly visits: readonly number[]
  readonly barely: boolean
  /** The mean over steps of the largest gap between EP's and the sampler's regime probabilities. */
  readonly gap: number
  readonly agrees: boolean
  /** The share of steps whose most probable regime is the true one. */
  readonly accuracy: number
  /** EP's and the sampler's belief about each parameter beside the truth. */
  readonly ep: CalibrationSample
  readonly gibbs: CalibrationSample
}

/** The share of the steps below which a regime counts as barely visited. */
export const BARELY = 0.06

/** The largest tolerated mean gap between EP's and the sampler's regime probabilities for EP to have found the same reading. */
export const AGREES = 0.05

/**
 * What EP and the sampler recovered of one simulated data set: the regimes and every parameter beside its truth.
 *
 * @param seed - The seed the data were simulated with.
 * @param sim - The simulation (the true regimes, the counts, the stay probability).
 * @param ep - EP's run (`infer`).
 * @param gibbs - The sampler's result (`gibbsReference`).
 * @returns The comparison.
 */
export function assess(
  seed: number,
  sim: ReturnType<typeof simulate>,
  ep: ReturnType<typeof infer>,
  gibbs: Reference,
): SeedRun {
  const steps = sim.y.length
  const qs = quantities(sim.stay)
  const epBeliefs: Record<string, Belief> = {}
  for (const r of ep.summary) epBeliefs[r.name] = { mean: r.mean, sd: r.sd, proper: Number.isFinite(r.sd) && r.sd > 0 }
  const gibbsBeliefs: Record<string, Belief> = {}
  for (let k = 0; k < 3; k++)
    gibbsBeliefs[`rate[${k}]`] = { mean: gibbs.rateMean[k] as number, sd: gibbs.rateSd[k] as number, proper: true }
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      gibbsBeliefs[`A[${3 * i + j}]`] = {
        mean: (gibbs.matrixMean[i] as number[])[j] as number,
        sd: (gibbs.matrixSd[i] as number[])[j] as number,
        proper: true,
      }
  const visits = [0, 1, 2].map((k) => sim.z.filter((v) => v === k).length)
  const gap =
    ep.regimes.reduce(
      (s, p, t) => s + Math.max(...p.map((v, k) => Math.abs(v - ((gibbs.regimes[t] as number[])[k] as number)))),
      0,
    ) / steps
  const accuracy = ep.regimes.filter((p, t) => p.indexOf(Math.max(...p)) === sim.z[t]).length / steps
  return {
    seed,
    status: String(ep.run.status),
    visits,
    barely: visits.some((v) => v < BARELY * steps),
    gap,
    agrees: gap < AGREES,
    accuracy,
    ep: { seed, rows: ep.run.status === 'converged' ? atomRecovery({ quantities: qs }, epBeliefs) : [] },
    gibbs: { seed, rows: atomRecovery({ quantities: qs }, gibbsBeliefs) },
  }
}

/**
 * One seed: simulate, run EP and the sampler, read each parameter's belief beside its truth.
 *
 * @param seed - The seed of the simulated data.
 * @param options - The stay probability and the number of steps of the simulation, the sampler's sweeps, EP's damping.
 * @returns What each method recovered.
 */
export function runSeed(
  seed: number,
  options: { stay?: number; steps?: number; sweeps?: number; damping?: number } = {},
): SeedRun {
  const { stay = 0.93, steps = 100, sweeps = 1500, damping = 0.5 } = options
  const sim = simulate(seed, stay, steps)
  const ep = infer(sim.y, damping)
  const gibbs = gibbsReference(sim.y, { sweeps, burn: Math.round(sweeps / 7) })
  return assess(seed, sim, ep, gibbs)
}
