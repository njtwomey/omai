import { scan } from 'omai-engine/nn'
import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, poisson, split, uniform } from 'omai-engine/random'
import {
  arange,
  cast,
  concat,
  cumsum,
  equal,
  floor,
  lessEqual,
  minimum,
  mul,
  reshape,
  sliceAxis,
  stack,
  sum,
  take,
  tensor,
  toArray,
  toFlat,
  type Value,
  where,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

// The truth the data are simulated from: the rate of each regime, and, for a regime left, how the move splits.
export const RATES = [3, 9, 22]
export const SPLIT = [
  [0, 0.7, 0.3],
  [0.5, 0, 0.5],
  [0.3, 0.7, 0],
]
// The model's beliefs. Rates: gamma priors with means 4, 12 and 30 (a low, a middle and a high regime). Each row of the
// transition matrix: a Dirichlet that leans towards staying (mean stay 0.67) and is open about where a move goes.
export const PRIOR = {
  shape: [2, 2, 2],
  rate: [0.5, 1 / 6, 1 / 15],
  rows: [
    [4, 1, 1],
    [1, 4, 1],
    [1, 1, 4],
  ],
}

/** The transition matrix the data are simulated from: stay with probability `stay`, else move by `SPLIT`. */
export const transitions = (stay: number): number[][] => toArray(transitionMatrix(stay)) as number[][]

/** The same matrix as a tensor. */
const transitionMatrix = (stay: number): Value =>
  where(equal(reshape(arange(3), [3, 1]), reshape(arange(3), [1, 3])), stay, mul(tensor(SPLIT), 1 - stay))

// region simulate
/**
 * A sticky chain over three regimes with its own transition matrix, then a Poisson count at each regime's rate.
 */
export function simulate(seed: number, stay = 0.93, steps = 150) {
  const [kz, ky] = split(key(seed), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
  const A = transitionMatrix(stay)
  const u = uniform(kz, [steps])
  const start = cast(floor(mul(take(u, 0), 3)), 'int32')
  // The next regime: the first whose running probability, from this one, passes the draw.
  const path = scan(
    (prev, draw) => {
      const passed = lessEqual(cumsum(take(A, prev)), draw)
      const next = cast(minimum(sum(passed), 2), 'int32')
      return [next, next]
    },
    start,
    sliceAxis(u, 0, 1, steps),
  )
  const z = toFlat(concat([reshape(start, [1]), path.ys]))
  const y = z.map((s, t) => toFlat(poisson(foldIn(ky, t), RATES[s] as number, [1]))[0] ?? 0)
  return { z, y, stay, A: toArray(A) as number[][] }
}
// endregion

// region model
/**
 * z: a Markov chain over three regimes whose transition matrix A is unknown (a Dirichlet per row).
 * Each regime has an unknown rate (gamma prior); the count at t is Poisson(rate[z_t]).
 */
export function model() {
  return defineModel('poisson-changepoint-learned', (b) => {
    const y = b.data('y', { dims: ['time'], support: 'count' })
    const regime = b.range('regime', { size: 3 })
    const shape = b.const(PRIOR.shape, ['regime'])
    const rateScale = b.const(PRIOR.rate, ['regime'])
    const rowPrior = b.const(PRIOR.rows, ['regime'])
    // The unknown rate of each regime.
    const rate = b.latent('rate', d.gamma(shape, rateScale), { dims: [regime] })
    // The unknown transition probabilities: row i is where the chain goes from regime i.
    const A = b.latent('A', d.dirichlet(rowPrior), { dims: [regime] })
    b.scan('z', {
      over: 'time',
      init: d.categorical({ probs: [1 / 3, 1 / 3, 1 / 3] }),
      step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
      emit: ({ state, observe }) => observe('y', d.poisson(ops.at(rate, state)), y),
    })
  })
}
// endregion

// region infer
/**
 * Expectation propagation on the factor graph: the regime per step, the rates and the transition matrix. Damped by half:
 * undamped, EP sometimes settles on the labelling in which the low and middle regimes are swapped.
 */
export async function infer(y: readonly number[], damping = 0.5, hooks: EpHooks = {}) {
  const bound = model().bind({ y })
  const ep = epSpecMethod(bound)
  const config = { tol: 1e-9, damping }
  const cap = 300
  const run = session(ep, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each (the bar), and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  const post = ep.posterior?.(run.state, run.config)
  if (post === undefined) throw new Error('EP gave no posterior')
  // The probability of each regime at each step: one row per step.
  const steps = Array.from({ length: y.length }, (_, t) => post.marginal(`z[${t}]`).prob(arange(3)))
  const regimes = toArray(stack(steps)) as number[][]
  const summary = post.summary({ prob: 0.9 })
  const rates = summary.filter((r) => r.site === 'rate')
  return { bound, run, regimes, rates, summary }
}
// endregion
