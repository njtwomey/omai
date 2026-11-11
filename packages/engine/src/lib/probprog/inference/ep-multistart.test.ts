// EP from several starts on a small regime-switching model: two regimes with Poisson counts, only one of them visited. The exact
// posterior is an enumeration of the 2^T assignments (the rates are gamma given an assignment, so each assignment's evidence and
// each rate's posterior are closed forms), against which one EP run (which commits to one reading of the empty regime) and the
// evidence-weighted mixture of EP's fixed points are held.
import { describe, expect, it } from 'vitest'
import { lgamma } from '../../numerics/special/index.ts'
import { item, toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { epMultiStartMethod } from './ep-multistart.ts'
import { analysisOf } from './model-input.ts'
import { methodKinds } from './models.ts'
import { epSpecMethod } from './mp.ts'

const STAY = 0.9
const SHAPE = [2, 2]
const RATE = [0.5, 1 / 12]
/** Counts from the first regime's neighbourhood only: the second regime (prior mean 24) has no data. */
const Y = [3, 4, 2, 5, 3, 4, 3, 2, 4, 3, 5, 3, 4, 3]

const spec = defineModel('two-regimes', (b) => {
  const y = b.data('y', { dims: ['time'], support: 'count' })
  const regime = b.range('regime', { size: 2 })
  const A = b.const(
    [
      [STAY, 1 - STAY],
      [1 - STAY, STAY],
    ],
    ['from'],
  )
  const rate = b.latent('rate', d.gamma(b.const(SHAPE, ['regime']), b.const(RATE, ['regime'])), { dims: [regime] })
  b.scan('z', {
    over: 'time',
    init: d.categorical({ probs: [0.5, 0.5] }),
    step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
    emit: ({ state, observe }) => observe('y', d.poisson(ops.at(rate, state)), y),
  })
})

const lg = (x: number): number => item(lgamma(x))

/** The exact posterior mean and sd of each rate: a mixture over the assignments of the gamma posteriors given them. */
function exact(): { mean: number[]; sd: number[]; logEvidence: number } {
  const T = Y.length
  const logs: number[] = []
  const stats: { a: number[]; b: number[] }[] = []
  const logFact = Y.reduce((s, v) => s + lg(v + 1), 0)
  for (let code = 0; code < 2 ** T; code++) {
    const z = Array.from({ length: T }, (_, t) => (code >> t) & 1)
    let lp = Math.log(0.5)
    for (let t = 1; t < T; t++) lp += Math.log(z[t] === z[t - 1] ? STAY : 1 - STAY)
    const a: number[] = []
    const bb: number[] = []
    for (const k of [0, 1]) {
      const idx = z.map((v, t) => (v === k ? t : -1)).filter((t) => t >= 0)
      const s = idx.reduce((acc, t) => acc + (Y[t] ?? 0), 0)
      const n = idx.length
      const [sh, rt] = [SHAPE[k] as number, RATE[k] as number]
      lp += sh * Math.log(rt) - lg(sh) + lg(sh + s) - (sh + s) * Math.log(rt + n)
      a.push(sh + s)
      bb.push(rt + n)
    }
    logs.push(lp - logFact)
    stats.push({ a, b: bb })
  }
  const top = Math.max(...logs)
  const total = logs.reduce((s, l) => s + Math.exp(l - top), 0)
  const mean = [0, 1].map((k) =>
    stats.reduce(
      (s, st, i) => s + (Math.exp((logs[i] as number) - top) / total) * ((st.a[k] as number) / (st.b[k] as number)),
      0,
    ),
  )
  const second = [0, 1].map((k) =>
    stats.reduce(
      (s, st, i) =>
        s +
        (Math.exp((logs[i] as number) - top) / total) *
          (((st.a[k] as number) * ((st.a[k] as number) + 1)) / (st.b[k] as number) ** 2),
      0,
    ),
  )
  return { mean, sd: second.map((v, k) => Math.sqrt(v - (mean[k] as number) ** 2)), logEvidence: top + Math.log(total) }
}

describe('epMultiStartMethod on a regime model with an empty regime', () => {
  const want = exact()
  const bound = spec.bind({ y: Y })

  it('is a kind of methodKinds, and refuses a model with no discrete variable', () => {
    expect(Object.keys(methodKinds)).toContain('ep-multistart')
    const nodiscrete = defineModel('m', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }))
    }).bind({ y: [1, 2] })
    expect(epMultiStartMethod(nodiscrete).check?.(analysisOf(nodiscrete)).applies).toBe(false)
    expect(epMultiStartMethod(bound).check?.(analysisOf(bound)).applies).toBe(true)
  })

  it('finds several fixed points, and their mixture puts the empty regime where the exact posterior has it where one run does not', () => {
    const single = epSpecMethod(bound)
    const one = session(single, { steps: 300, mode: 'eager', config: { tol: 1e-9 } })
    const base =
      single
        .posterior?.(one.state, one.config)
        ?.summary()
        .filter((r) => r.site === 'rate') ?? []
    const method = epMultiStartMethod(bound)
    const run = session(method, { steps: 40, mode: 'eager', config: { starts: 30 } })
    expect(run.status).toBe('completed')
    const post = method.posterior?.(run.state, run.config)
    expect(Number(post?.diagnostics['modes'])).toBeGreaterThan(1)
    const mixed = post?.summary().filter((r) => r.site === 'rate') ?? []
    // The regime nobody visits: its posterior is close to its prior (mean 24, sd 17): the exact mean is 17.7 and sd 16.9.
    expect(want.sd[1]).toBeGreaterThan(10)
    const meanError = (rows: typeof mixed): number =>
      Math.abs((rows[1]?.mean ?? 0) - (want.mean[1] as number)) / (want.sd[1] as number)
    expect(meanError(base)).toBeGreaterThan(0.2)
    expect(meanError(mixed)).toBeLessThan(0.05)
    expect(Math.abs((mixed[1]?.sd ?? 0) / (want.sd[1] as number) - 1)).toBeLessThan(0.05)
    // The visited regime is where the data say, with a spread that is too small (the exact one counts the steps that may belong to the other regime): the mixture does not make it worse.
    expect(Math.abs((mixed[0]?.mean ?? 0) - (want.mean[0] as number)) / (want.sd[0] as number)).toBeLessThan(0.2)
    expect(mixed[0]?.sd ?? 0).toBeGreaterThanOrEqual((base[0]?.sd ?? 0) - 1e-9)
  })

  it('a snapshot is the mixture of the modes found so far, and the tape counts starts and modes', () => {
    const method = epMultiStartMethod(bound)
    const run = session(method, { steps: 40, mode: 'eager', config: { starts: 12 } })
    const early = method.posterior?.(run.stateAt(2), run.config)
    expect(early?.status).toBe('stopped')
    const late = method.posterior?.(run.state, run.config)
    expect(late?.logEvidence?.method).toBe('ep-multistart')
    expect(Number.isFinite(late?.logEvidence?.value)).toBe(true)
    expect(toFlat(late?.mean('rate') ?? ([] as never)).length).toBe(2)
  })
})
