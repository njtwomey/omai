import { describe, expect, it } from 'vitest'
import { uniform } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { recorder } from '../trace/index.ts'
import { erlangB, erlangC, littlesLaw, mg1, mm1, mmc, simulateQueue } from './queues.ts'

describe('queue formulas', () => {
  it('M/M/c reduces to M/M/1 and the state probabilities sum to one', () => {
    const a = mm1(3, 5)
    const b = mmc(3, 5, 1)
    expect(b.L).toBeCloseTo(a.L, 12)
    expect(b.Wq).toBeCloseTo(a.Wq, 12)
    expect(b.p0).toBeCloseTo(a.p0, 12)
    const c = mmc(5, 2, 4)
    let total = 0
    for (let n = 0; n < 400; n++) total += c.stateProbability(n)
    expect(total).toBeCloseTo(1, 10)
    expect(c.L).toBeCloseTo(c.Lq + 5 / 2, 12)
    expect(a.sojournCdf(a.W)).toBeCloseTo(1 - Math.exp(-1), 12)
    expect(c.waitCdf(0)).toBeCloseTo(1 - c.waitProbability, 12)
  })

  it('M/G/1 recovers M/M/1 for exponential service and halves the wait for deterministic service', () => {
    const exp = mg1(2, { mean: 0.25, variance: 0.0625 })
    expect(exp.Lq).toBeCloseTo(mm1(2, 4).Lq, 12)
    expect(mg1(2, { mean: 0.25, variance: 0 }).Lq).toBeCloseTo(exp.Lq / 2, 12)
    expect(mg1(0.5, uniform(0, 2)).utilisation).toBeCloseTo(0.5, 12)
  })

  it('rejects unstable queues', () => {
    expect(() => mm1(3, 3)).toThrow()
    expect(() => mmc(9, 1, 8)).toThrow()
    expect(() => erlangC(2, 2)).toThrow()
    expect(erlangB(3, 0)).toBe(0)
  })
})

describe('queue simulation', () => {
  it('an M/M/1 simulation approaches the steady state, and Little holds on the sample path', () => {
    const rec = recorder()
    const sim = simulateQueue(key(1), { interarrival: 2, service: 4 }, 20000, { tracer: rec })
    const exact = mm1(2, 4)
    expect(Math.abs(sim.W - exact.W)).toBeLessThan(0.06)
    expect(Math.abs(sim.L - exact.L)).toBeLessThan(0.12)
    const l = littlesLaw(sim)
    expect(l.L).toBeCloseTo(l.lambdaW, 9)
    expect(l.Lq).toBeCloseTo(l.lambdaWq, 9)
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'sojourn')).toBe(true)
  })

  it('an M/M/3 simulation matches Erlang C', () => {
    const sim = simulateQueue(key(2), { interarrival: 5, service: 2, servers: 3 }, 20000)
    const exact = mmc(5, 2, 3)
    expect(Math.abs(sim.Wq - exact.Wq)).toBeLessThan(0.15 * exact.Wq + 0.02)
    const waited = Array.from(sim.waits).filter((w) => w > 0).length / sim.waits.length
    expect(Math.abs(waited - exact.waitProbability)).toBeLessThan(0.05)
  })

  it('an M/D/1 simulation matches Pollaczek-Khinchine', () => {
    const sim = simulateQueue(key(3), { interarrival: 0.6, service: uniform(1, 1 + 1e-9) }, 20000)
    expect(Math.abs(sim.Wq - mg1(0.6, { mean: 1, variance: 0 }).Wq)).toBeLessThan(0.15)
    expect(() => simulateQueue(key(0), { interarrival: -1, service: 1 }, 3)).toThrow()
  })
})
