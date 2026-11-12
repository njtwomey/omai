import { lgamma } from 'omai-engine/numerics/special'
import { item } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { detect, run } from './compute.ts'
import { coalCounts, SIMULATED } from './data.ts'
import { readResult, shortestWindow } from './reading.ts'

const lg = (x: number): number => item(lgamma(x))

/**
 * The exact posterior over the first year of the new rate, by summing over every possible year: the prior is the model's (a
 * switch chance of one in `years` each year, the rates gamma(2, 1)) and each side's marginal likelihood is the closed form of
 * the gamma-Poisson model. Returns the chance that the switch has happened by each year.
 */
function exactSwitched(counts: readonly number[]): number[] {
  const n = counts.length
  const h = 1 / n
  const total = (a: number, b: number): number => counts.slice(a, b).reduce((x, y) => x + y, 0)
  // log of the marginal likelihood of `s` events in `m` years under a gamma(2, 1) rate (the factorials are common to all)
  const marginal = (s: number, m: number): number => -lg(2) + lg(2 + s) - (2 + s) * Math.log(1 + m)
  const log = counts.map((_, i) => {
    const k = i + 1
    const prior = k < n ? (k - 1) * Math.log(1 - h) + Math.log(h) : (n - 1) * Math.log(1 - h)
    return prior + marginal(total(0, k), k) + (k < n ? marginal(total(k, n), n - k) : marginal(0, 0))
  })
  const top = Math.max(...log)
  const w = log.map((v) => Math.exp(v - top))
  const z = w.reduce((a, b) => a + b, 0)
  let acc = 0
  // the chance that the switch has happened in year t is the chance of a first year of the new rate up to t
  return w.map((v) => (acc += v / z))
}

describe('the when-did-the-rate-change computation', () => {
  it('has the 112 yearly counts of coal-mining disasters, 191 in all', () => {
    const c = coalCounts()
    expect(c).toHaveLength(112)
    expect(c.reduce((a, b) => a + b, 0)).toBe(191)
  })

  it('finds the switch in the coal-mining counts, and message passing agrees with the sum over every year', () => {
    const counts = coalCounts()
    const d = detect(counts)
    expect(d.settled).toBe(true)
    expect(d.when).toHaveLength(112)
    expect(d.when.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(d.when.every((p) => p >= 0)).toBe(true)
    expect(d.before.mean).toBeGreaterThan(d.after.mean)
    expect(d.before.lo).toBeLessThan(d.before.mean)
    expect(d.before.hi).toBeGreaterThan(d.before.mean)
    const exact = exactSwitched(counts)
    // the switch at index k means "switched by t" for t >= k, so the exact chance by t is exact[t - 1]
    const gap = d.switched.slice(1).map((p, i) => Math.abs(p - (exact[i] as number)))
    expect(Math.max(...gap)).toBeLessThan(0.03)
    // the window of the classic analysis: around 1890
    const w = shortestWindow(d.when)
    expect(w.from).toBeGreaterThan(30)
    expect(w.to).toBeLessThan(50)
  })

  it('returns plain data on the full run, sharper for the abrupt series, and survives structuredClone', () => {
    const r = run(1)
    expect(r.abrupt.when).toHaveLength(112)
    expect(r.gentle.when).toHaveLength(112)
    expect(r.truth).toBe(SIMULATED.at)
    expect(structuredClone(r)).toEqual(r)
    const reading = readResult(r)
    expect(reading.abruptSharper).toBe(true)
    // the made-up series differ with the seed and the real one does not
    const other = run(2)
    expect(other.coal).toEqual(r.coal)
    expect(other.abrupt.counts).not.toEqual(r.abrupt.counts)
  }, 60000)

  it('finds the shortest window of consecutive years that holds a share of the chance', () => {
    const w = shortestWindow([0.05, 0.1, 0.5, 0.3, 0.05], 0.79)
    expect([w.from, w.to, w.years]).toEqual([2, 3, 2])
  })
})
