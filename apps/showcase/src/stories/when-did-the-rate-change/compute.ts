/**
 * The computation of 'When did the rate change?': a count series is given a model with one switch (a rate before it, a rate
 * after it, and a hidden phase for each year that can only move from before to after), and expectation propagation turns the
 * counts into the chance that each year is already in the new phase. Differencing those chances gives the posterior over the
 * year of the switch. The same model is run on the real coal-mining counts and on two made-up series, one with an abrupt change
 * and one with a gentle one, to show that the sharper the change the sharper the date. It is a pure function of the seed (which
 * draws the made-up series) and returns plain data, so it runs in the story's worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, poissonEach } from 'omai-engine/random'
import {
  add,
  arange,
  concat,
  div,
  less,
  maximum,
  mul,
  sliceAxis,
  sub,
  sum,
  tensor,
  toFlat,
  where,
  zeros,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
// endregion

import type { Say } from '../../kit/protocol.ts'
import { coalCounts, SIMULATED } from './data.ts'

/** A rate (events per year) with the range that holds nine tenths of the belief about it. */
export interface Interval {
  readonly mean: number
  readonly lo: number
  readonly hi: number
}

/** What the model concludes about one series. */
export interface Detection {
  /** The counts the model saw. */
  readonly counts: readonly number[]
  /** For each year, the chance that the switch happened in that year (the rate is new from then on). */
  readonly when: readonly number[]
  /** For each year, the chance that the rate has already switched by then. */
  readonly switched: readonly number[]
  /** The rate expected in each year: the old rate before the switch, the new one after, weighted by the chance of each. */
  readonly expected: readonly number[]
  /** The rate before the switch and the rate after it. */
  readonly before: Interval
  readonly after: Interval
  /** How many sweeps of message passing it took, and whether they settled. */
  readonly sweeps: number
  readonly settled: boolean
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The real series. */
  readonly coal: Detection
  /** The made-up series, drawn from the seed, with the index where their rate really changed. */
  readonly abrupt: Detection
  readonly gentle: Detection
  /** The index of the year the made-up series really changed in. */
  readonly truth: number
}

/**
 * The model's reading of one count series: write the switch model, pass messages until the beliefs settle, and read the
 * chance of each year being the one the rate changed in.
 *
 * @param counts - The count in each year.
 * @returns What the model concludes.
 */
export function detect(counts: readonly number[]): Detection {
  const years = counts.length
  // region model
  // each year there is a small chance, one in a hundred or so, that the rate changes for good
  const chance = 1 / years
  const switchpoint = defineModel('switchpoint', (m) => {
    const events = m.data('events', { dims: ['year'], support: 'count' })
    const phase = m.range('phase', { size: 2 })
    // the phase of a year is "before" or "after"; it can move from before to after and never back
    const moves = m.const(
      [
        [1 - chance, chance],
        [1e-6, 1 - 1e-6],
      ],
      ['from'],
    )
    // the rate of each phase: broad, with a mean of two events a year
    const shape = m.const([2, 2], ['phase'])
    const rate = m.latent('rate', d.gamma(shape, 1), { dims: [phase] })
    m.scan('regime', {
      over: 'year',
      init: d.categorical({ probs: [1 - 1e-6, 1e-6] }),
      step: ({ prev }) => d.categorical({ probs: ops.at(moves, prev) }),
      emit: ({ state, observe }) => observe('events', d.poisson(ops.at(rate, state)), events),
    })
  })
  // endregion
  // region infer
  const bound = switchpoint.bind({ events: counts })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep of messages along the years and back
  const run = session(route.method, { steps: 200, mode: 'eager', config: { tol: 1e-6 } })
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  // region reading
  const rows = beliefs.summary({ prob: 0.9 })
  // the mean of a 0/1 phase is the chance that the year is in the "after" phase
  const switched = tensor(rows.filter((r) => r.site.startsWith('regime[')).map((r) => r.mean))
  // the switch happened in year t when the rate had switched by t but not by the year before
  const yearBefore = concat([zeros([1]), sliceAxis(switched, 0, 0, years - 1)])
  const wobbly = sub(switched, yearBefore)
  // rounding can leave a tiny negative chance: clip it, and make the chances add up to one
  const when = div(maximum(wobbly, 0), sum(maximum(wobbly, 0)))
  const [old, fresh] = rows.filter((r) => r.site === 'rate')
  // endregion
  if (old === undefined || fresh === undefined) throw new Error('the method returned no rates')
  // the rate to expect in each year: the old rate or the new one, by the chance of each
  const expected = add(mul(sub(1, switched), old.mean), mul(switched, fresh.mean))
  return {
    counts,
    when: toFlat(when),
    switched: toFlat(switched),
    expected: toFlat(expected),
    before: { mean: old.mean, lo: old.lower, hi: old.upper },
    after: { mean: fresh.mean, lo: fresh.lower, hi: fresh.upper },
    sweeps: run.t,
    settled: run.status === 'converged',
  }
}

/** A made-up series: the counts of `years` years whose rate is `before` up to the index `at` and `after` from then on. */
function simulate(
  seed: number,
  salt: number,
  years: number,
  rates: { readonly before: number; readonly after: number },
): number[] {
  const perYear = where(less(arange(years), SIMULATED.at), rates.before, rates.after)
  return toFlat(poissonEach(perYear, (t) => foldIn(foldIn(key(seed), salt), t)))
}

/**
 * The run of the story: the model on the real series and on the two made-up ones.
 *
 * @param seed - Draws the made-up series (the real one does not change with it).
 * @param say - Says what is happening, for the play button.
 * @returns The result.
 */
export function run(seed: number, say: Say = () => {}): Result {
  const real = coalCounts()
  say('Passing messages along the years')
  const coal = detect(real)
  say('Making two made-up series')
  const abrupt = detect(simulate(seed, 0, real.length, SIMULATED.abrupt))
  say('Finding the switch in them')
  const gentle = detect(simulate(seed, 1, real.length, SIMULATED.gentle))
  return { coal, abrupt, gentle, truth: SIMULATED.at }
}
