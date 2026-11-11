import { type DictSpace, defaults, type UnionSpace } from 'omai-engine/space'
import { session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { coinExample, coinMethod, coinProblem, coinSteps } from './coin-method.ts'

const options = (coinMethod.config.spaces['inference'] as UnionSpace).options
const configOf = (kind: string, extra: Record<string, unknown> = {}, over: Record<string, unknown> = {}) => ({
  ...(defaults(coinMethod.config) as Record<string, unknown>),
  ...over,
  inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Record<string, unknown>), ...extra },
})
const run = (config: Record<string, unknown>) =>
  session(coinMethod, { config: config as never, steps: coinSteps(config as never), mode: 'eager' })

describe('coinMethod', () => {
  it('generates the same flips with a prefix property, for both observations', () => {
    const long = coinProblem({ kind: 'coin', flips: 30, trueP: 0.4, trueRate: 3, dataSeed: 5, priorA: 1, priorB: 1 })
    const short = coinProblem({ kind: 'coin', flips: 10, trueP: 0.4, trueRate: 3, dataSeed: 5, priorA: 1, priorB: 1 })
    expect(long.y.slice(0, 10)).toEqual(short.y)
    const counts = coinProblem({
      kind: 'counts',
      flips: 12,
      trueP: 0.4,
      trueRate: 3,
      dataSeed: 5,
      priorA: 2,
      priorB: 1,
    })
    expect(counts.site).toBe('rate')
    expect(counts.y.length).toBe(12)
  })

  it('offers the conjugate and the sampling methods and gives the library reason for the ones that do not apply', () => {
    const offer = coinExample.offer(defaults(coinMethod.config) as never)
    for (const kind of ['conjugate', 'nuts', 'laplace', 'svi', 'importance']) expect(offer.applicable).toContain(kind)
    expect(Object.keys(offer.unavailable).length).toBeGreaterThan(0)
  })

  it('the conjugate method is the exact posterior: the check finds no error', () => {
    const s = run(configOf('conjugate'))
    const check = s.probe(s.t, 'coin/check') as {
      meanError: number
      sdRatio: number
      kl: number
      comparison: { kind: string }
    }
    expect(check.comparison.kind).toBe('posteriorComparison')
    expect(Math.abs(check.meanError)).toBeLessThan(0.1)
    expect(check.sdRatio).toBeGreaterThan(0.9)
    expect(check.sdRatio).toBeLessThan(1.1)
  })

  it('NUTS agrees with the exact posterior', () => {
    const nuts = run(configOf('nuts', { numChains: 2, numWarmup: 100, numDraws: 300 }, { flips: 15 }))
    const c = nuts.probe(nuts.t, 'coin/check') as { meanError: number; wasserstein: number }
    expect(Math.abs(c.meanError)).toBeLessThan(0.3)
    expect(c.wasserstein).toBeLessThan(0.3)
  })

  it('refits the method on the first flips seen and the posterior contracts as they accumulate', () => {
    const s = run(configOf('conjugate', {}, { flips: 40 }))
    const early = s.probe(s.t, 'coin/updating', { seen: 5 }) as {
      seen: number
      exact: { sd: number }
      method: { sd: number }
    }
    const late = s.probe(s.t, 'coin/updating', { seen: 40 }) as { seen: number; exact: { sd: number } }
    expect(early.seen).toBe(5)
    expect(late.exact.sd).toBeLessThan(early.exact.sd)
    const curve = s.probe(s.t, 'coin/contraction') as { points: { n: number; mean: number; lo: number; hi: number }[] }
    expect(curve.points[0]?.n).toBe(0)
    const first = curve.points[1]
    const last = curve.points.at(-1)
    expect((last?.hi ?? 0) - (last?.lo ?? 0)).toBeLessThan((first?.hi ?? 1) - (first?.lo ?? 0))
  })

  it('works on counts too', () => {
    const s = run(configOf('conjugate', {}, { kind: 'counts', flips: 25 }))
    const check = s.probe(s.t, 'coin/check') as { exactMean: number; methodMean: number }
    expect(check.methodMean).toBeCloseTo(check.exactMean, 1)
  })
})
