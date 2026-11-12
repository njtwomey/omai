/**
 * The computation of 'Counts with too many zeros': a zero-inflated Poisson regression written as a probabilistic program, run by
 * expectation propagation. Each count has a switch with two states, a count or a structural zero. The model learns the chance
 * that a switch is structural and the two coefficients of the Poisson rate, and gives each zero a chance of being structural. The
 * model is run on the first few counts, then on more, to see the belief about the chance sharpen; on all of them it is also run
 * with a plain Poisson model that has no switch, and its messages from the gates to the switches are read off the tape. The
 * library's exact posterior of the same model, which sums the switches out and integrates the three parameters on a grid, comes
 * back beside the run. The result is plain data and a pure function of the seed (which draws the counts), so it runs in the
 * story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { zeroInflatedPoisson } from 'omai-engine/probprog/validation'
import { session } from 'omai-engine/trace'
// endregion

import { beta } from 'omai-engine/dists'
import type { TapeLike } from 'omai-engine/exposition'
import type { BoundModel } from 'omai-engine/probprog/model'
import { abs, arange, item, linspace, max, sub, type Tensor, toFlat, type Value } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { CHECKPOINTS, ROWS, type Series, series } from './data.ts'

// region models
// every count is a structural zero or a Poisson count, and a switch says which
export const zeroInflated = defineModel('zero-inflated-poisson', (m) => {
  const row = m.range('row')
  const state = m.range('state')
  const x = m.data('x', { dims: [row] })
  const y = m.data('y', { dims: [row] })
  // is the Poisson on in each state: yes for a count, no for a structural zero
  const on = m.data('on', { dims: [state] })
  // the chances of the two states: a count first, then a structural zero
  const w = m.latent('w', d.dirichlet([1, 1]))
  // the state of each count
  const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] })
  const b0 = m.latent('b0', d.normal(0, 2))
  const b1 = m.latent('b1', d.normal(0, 2))
  const rate = ops.exp(ops.add(b0, ops.mul(b1, x)))
  // a switch that is off sets the rate to zero, so the count is zero
  m.observe('y', d.poisson(ops.mul(ops.at(on, z), rate)), y)
})

// the same counts with no switch: every zero is an ordinary Poisson zero
export const plainPoisson = defineModel('poisson-regression', (m) => {
  const row = m.range('row')
  const x = m.data('x', { dims: [row] })
  const y = m.data('y', { dims: [row] })
  const b0 = m.latent('b0', d.normal(0, 2))
  const b1 = m.latent('b1', d.normal(0, 2))
  m.observe('y', d.poisson(ops.exp(ops.add(b0, ops.mul(b1, x)))), y)
})
// endregion

/** A belief about one number: its mean and its spread. */
export interface Moments {
  readonly mean: number
  readonly sd: number
}

/** The belief about the chance that a switch is structural after some number of counts. */
export interface Belief extends Moments {
  /** How many counts it has seen. */
  readonly rows: number
  /** The range that holds nine tenths of the belief. */
  readonly lower: number
  readonly upper: number
  /** The belief's density at each value of {@link Result.grid}. */
  readonly density: readonly number[]
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The counts, in the order they came, and the covariate of each. */
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** The truth: 1 for a count whose switch was on. */
  readonly structural: readonly number[]
  /** For each count, the chance EP gives to its switch being structural (zero for a count above zero). */
  readonly chance: readonly number[]
  /** For each zero, the log-odds of structural against count in the message the gate sends to its switch (0 for other counts). */
  readonly gate: readonly number[]
  /** For each count, the same log-odds in the message the switch's own factor sends from the belief about the chance. */
  readonly prior: readonly number[]
  /** The values the beliefs about the chance are drawn at. */
  readonly grid: readonly number[]
  /** The belief about the chance after each number of counts in `CHECKPOINTS`, the last on all of them. */
  readonly beliefs: readonly Belief[]
  /** The intercept and slope of the Poisson rate on all counts, with the switch and without it. */
  readonly withSwitch: { readonly b0: Moments; readonly b1: Moments }
  readonly withoutSwitch: { readonly b0: Moments; readonly b1: Moments }
  /** The library's exact posterior of the model with the switch, and the largest gap to it in a zero's chance. */
  readonly exact: {
    readonly pi: Moments
    readonly b0: Moments
    readonly b1: Moments
    readonly chanceGap: number
  }
  /** How many sweeps of messages the run on all the counts took, and whether they settled. */
  readonly sweeps: number
  readonly settled: boolean
}

/** How much to do. */
export interface Size {
  /** How many counts there are, and the numbers of them the model is run on (ending with all of them). */
  readonly rows: number
  readonly checkpoints: readonly number[]
}

/** The size of the real run. */
export const DEFAULT_SIZE: Size = { rows: ROWS, checkpoints: CHECKPOINTS }

/** The sweeps and the damping EP is run with. */
const EP = { steps: 300, config: { tol: 1e-6, damping: 0.5 } } as const

// region infer
// run expectation propagation on a bound model and read its beliefs
function infer(bound: BoundModel, logs: 'all' | 'defaults') {
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: every factor sends messages, every variable multiplies them
  const run = session(route.method, { ...EP, mode: 'eager', logs })
  const beliefs = route.method.posterior?.(run.state, run.config)
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  return { run, beliefs }
}
// endregion

// region fit
// the model with the switch, on the first `rows` counts
function withSwitch(counts: Series, rows: number, logs: 'all' | 'defaults') {
  // the model is told the covariate, the counts, and which state has the Poisson on
  const bound = zeroInflated.bind({ x: counts.x.slice(0, rows), y: counts.y.slice(0, rows), on: [1, 0] })
  return infer(bound, logs)
}

// the model with no switch, on all the counts
function withoutSwitch(counts: Series) {
  return infer(plainPoisson.bind({ x: counts.x, y: counts.y }), 'defaults')
}
// endregion

/**
 * The log-odds of structural against count in the messages one kind of factor sent to the switches in the last sweep, by count.
 *
 * @param tape - The tape of a run with every message logged.
 * @param sweep - The sweep to read.
 * @param rows - How many counts there are.
 * @param operator - `gate` for the factor of the count, `discrete` for the factor that ties the switch to its chances.
 * @returns One log-odds per count: 0 where that factor sent no message.
 */
export function messageOdds(tape: TapeLike, sweep: number, rows: number, operator: 'gate' | 'discrete'): number[] {
  const odds = new Map<number, number>()
  for (const e of tape.events) {
    if (e.kind !== 'mp/message' || e.step !== sweep || e.to === undefined || !e.to.startsWith('z[')) continue
    const sent = e.payload as { readonly operator: string; readonly eta: readonly number[] }
    // a message to a variable with two states is its two log-weights: structural minus count is the log-odds
    if (sent.operator === operator)
      odds.set(Number(e.to.slice(2, -1)), (sent.eta[1] as number) - (sent.eta[0] as number))
  }
  return toFlat(arange(rows)).map((i) => odds.get(i) ?? 0)
}

// region beliefs
// the belief about the chance of a structural zero, from the belief about the states
function chanceBelief(rows: number, beliefs: ReturnType<typeof infer>['beliefs'], grid: Tensor): Belief {
  const [count = 1, zero = 1] = toFlat(beliefs.marginal('w').params['concentration'] as Value)
  const curve = beta(zero, count)
  // the quantiles come as the 5% of each state, then the 95% of each
  const [, lower = 0, , upper = 1] = toFlat(beliefs.quantiles('w', [0.05, 0.95]))
  return {
    rows,
    mean: zero / (zero + count),
    sd: toFlat(beliefs.stddev('w'))[1] ?? Number.NaN,
    lower,
    upper,
    density: toFlat(curve.prob(grid)),
  }
}
// endregion

/** One intercept and slope belief: the means and spreads of `b0` and `b1`. */
function coefficients(beliefs: ReturnType<typeof infer>['beliefs']): { readonly b0: Moments; readonly b1: Moments } {
  const one = (site: string): Moments => ({
    mean: toFlat(beliefs.mean(site))[0] as number,
    sd: toFlat(beliefs.stddev(site))[0] as number,
  })
  return { b0: one('b0'), b1: one('b1') }
}

/**
 * The run of the story: the model with the switch on the first few counts and then on all of them, the model without the switch
 * on all of them, and the library's exact posterior to put beside the last run.
 *
 * @param seed - Draws the counts.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do (the real size unless a test asks for less).
 * @returns The result.
 */
export function run(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  const counts = series(seed, size.rows)
  const rows = counts.y.length
  const grid = linspace(0.002, 0.998, 250)
  const fitFirst = (n: number): ReturnType<typeof withSwitch> => {
    say(`Passing messages on the first ${n} counts`)
    return withSwitch(counts, n, n === rows ? 'all' : 'defaults')
  }
  // region prefixes
  // the model on the first few counts, then on more, to watch the belief about the chance sharpen
  const fits = size.checkpoints.map(fitFirst)
  // endregion
  const final = fits.at(-1)
  if (final === undefined) throw new Error('no counts to fit')
  say('Passing messages with no switch')
  const plain = withoutSwitch(counts)
  say('Working out the exact posterior')
  // region exact
  // the exact posterior: switches summed out, the three parameters integrated on a grid
  const exact = zeroInflatedPoisson({ x: counts.x, y: counts.y, inflation: { a: 1, b: 1 }, coefficientSd: 2 })
  // endregion
  // region messages
  // the chance that each switch is structural
  const chance = toFlat(final.beliefs.mean('z'))
  // the log-odds of what the gate and the factor of the chance sent each switch
  const tape = final.run.run()
  const gate = messageOdds(tape, final.run.t, rows, 'gate')
  const prior = messageOdds(tape, final.run.t, rows, 'discrete')
  // endregion
  return {
    x: counts.x,
    y: counts.y,
    structural: counts.structural,
    chance,
    gate,
    prior,
    grid: toFlat(grid),
    beliefs: fits.map((f, i) => chanceBelief(size.checkpoints[i] as number, f.beliefs, grid)),
    withSwitch: coefficients(final.beliefs),
    withoutSwitch: coefficients(plain.beliefs),
    exact: {
      pi: exact.pi,
      b0: exact.b0,
      b1: exact.b1,
      chanceGap: item(max(abs(sub(chance, exact.structural)))),
    },
    sweeps: final.run.t,
    settled: final.run.status === 'converged',
  }
}
