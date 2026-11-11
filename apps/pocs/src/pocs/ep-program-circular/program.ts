import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { periodicCopies, wrapToPeriod } from 'omai-engine/stats'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../shared/ep/sweeps.ts'

/** One day, in hours. */
export const PERIOD = 24

/** The two generating densities of the post: a normal that straddles midnight, and a mixture of two. */
export const TRUTHS = {
  one: { weights: [1], means: [23], scales: [3] },
  two: { weights: [0.6, 0.4], means: [22, 9], scales: [2.5, 2] },
} as const
export type TruthName = keyof typeof TRUTHS

/** How the circle is handled: not at all, by shifting the means (the paper), by shifting the data (the post), or by extra rows (wrong). */
export type Variant = 'ignore' | 'means' | 'data' | 'rows'

// region simulate
/** Clock times in [0, 24): draws from the truth's mixture of normals, wrapped round the day. */
export function simulate(seed: number, truth: TruthName, draws: number): number[] {
  const { weights, means, scales } = TRUTHS[truth]
  const k = key(seed)
  const u = Array.from(toFlat(uniform(foldIn(k, 1), [draws])))
  const e = Array.from(toFlat(normal(foldIn(k, 2), [draws])))
  return u.map((ui, i) => {
    // Which component this draw comes from.
    const c = weights.findIndex((_, j) => ui < weights.slice(0, j + 1).reduce((a, b) => a + b, 0))
    const x = (means[c] as number) + (scales[c] as number) * (e[i] as number)
    return wrapToPeriod(x, PERIOD)
  })
}
// endregion

// region model
/**
 * A mixture of `components` normals for clock times, one sd shared by all (a gamma precision).
 *
 * ignore: a plain Gaussian mixture, 23:50 and 00:10 are 23 h 40 apart.
 * means: the paper. Each reading may come from any of three copies of its component, a period early,
 *   on time or late (the mean is shifted), the central mean held inside [0, 24).
 * data: the post. The same, with the shift put on the reading instead of the mean.
 * rows: wrong. The three copies of every reading are three separate readings.
 */
export function model(variant: Variant, components: number) {
  return defineModel(`circular-${variant}`, (m) => {
    const row = m.range('row')
    const comp = m.range('comp', { size: components })
    // States 3k, 3k + 1, 3k + 2 of j are the early, on-time and late copy of component k.
    const state = m.range('state', { size: 3 * components })
    const compOf = m.data('compOf', { dims: [state] })
    const shiftOf = m.data('shiftOf', { dims: [state] })
    const hours = m.data('hours', { dims: [row] })
    // Where each component starts, spread evenly round the day.
    const start = m.data('start', { dims: [comp] })
    // Except for the plain mixture, the central mean is held inside one period.
    const held = variant === 'ignore' ? { dims: [comp] } : { dims: [comp], constraint: { lower: 0, upper: PERIOD } }
    const mean = m.latent('mean', d.normal(start, 8), held)
    // The shared precision: sd 3 hours, broadly.
    const tau = m.latent('tau', d.gamma(2, 18))
    const sd = ops.div(1, ops.sqrt(tau))
    // The component of each reading, with Dirichlet weights (a single component needs neither).
    const first = m.data('first', { dims: [row] })
    const weight = components > 1 ? m.latent('weight', d.dirichlet(new Array<number>(components).fill(1))) : undefined
    const z = weight === undefined ? first : m.latent('z', d.categorical({ probs: weight }), { dims: [row] })
    if (variant === 'ignore' || variant === 'rows') {
      m.observe('hours', d.normal(ops.at(mean, z), sd), hours)
      return
    }
    // The copy: given the component, the three copies are equally likely.
    const copies = m.const(rowsOfCopies(components), ['comp'])
    const probs = weight === undefined ? [1 / 3, 1 / 3, 1 / 3] : ops.at(copies, z)
    const j = m.latent('copy', d.categorical({ probs }), { dims: [row] })
    if (variant === 'means') {
      const shifted = ops.add(ops.at(mean, compOf), shiftOf)
      m.observe('hours', d.normal(ops.at(shifted, j), sd), hours)
    } else {
      // The one normal stays; the reading is copied a period early and late, and the copy picks which is scored.
      const copy = ops.sub(hours, ops.at(shiftOf, j))
      m.observe('hours', d.normal(ops.at(ops.at(mean, compOf), j), sd), copy)
    }
  })
}
// endregion

/** The probabilities of the three copies of each component: row k is 1/3 on states 3k..3k+2 and 0 elsewhere. */
const rowsOfCopies = (components: number): number[][] =>
  Array.from({ length: components }, (_, k) =>
    Array.from({ length: 3 * components }, (_, s) => (Math.floor(s / 3) === k ? 1 / 3 : 0)),
  )

/** What the model reads: the clock times (tripled for the extra rows) and the tables of the copies. */
export function dataOf(variant: Variant, components: number, hours: readonly number[]) {
  const states = 3 * components
  return {
    hours: variant === 'rows' ? hours.flatMap((x) => periodicCopies(x, PERIOD)) : [...hours],
    start: Array.from({ length: components }, (_, k) => (PERIOD * (k + 0.5)) / components),
    compOf: Array.from({ length: states }, (_, s) => Math.floor(s / 3)),
    first: new Array<number>(hours.length * (variant === 'rows' ? 3 : 1)).fill(0),
    shiftOf: Array.from({ length: states }, (_, s) => ((s % 3) - 1) * PERIOD),
  }
}

// region infer
/** Expectation propagation on the factor graph: the means, the sd, the weights and the copy of each reading. */
export async function infer(variant: Variant, components: number, hours: readonly number[], hooks: EpHooks = {}) {
  const bound = model(variant, components).bind(dataOf(variant, components, hours))
  const ep = epSpecMethod(bound)
  const config = { tol: 1e-4, damping: 0.5 }
  const cap = 150
  const run = session(ep, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each (the bar), and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  const posterior = ep.posterior?.(run.state, run.config)
  if (posterior === undefined) throw new Error('EP gave no posterior')
  return { bound, run, posterior }
}
// endregion
