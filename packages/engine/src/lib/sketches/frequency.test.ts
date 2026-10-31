/**
 * Tests for the frequency sketches on skewed streams: Count-Min never underestimates and exceeds epsilon N with
 * probability at most delta, the conservative update dominates, tables merge by addition, the Count sketch is unbiased
 * with L2 error, and Misra-Gries and Space-Saving keep their counter guarantees (including after a merge).
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key, uniform } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  countMinAlgorithm,
  countMinSketch,
  countSketch,
  misraGries,
  spaceSavingAlgorithm,
  spaceSavingTopK,
} from './index.ts'

/** A stream of `n` draws from a Zipf(1) law over `support` items, as strings. */
function zipf(n: number, support: number, seed: number): string[] {
  const weights = Array.from({ length: support }, (_, i) => 1 / (i + 1))
  const total = weights.reduce((a, b) => a + b, 0)
  const cdf: number[] = []
  weights.reduce((acc, w) => {
    cdf.push((acc + w) / total)
    return acc + w
  }, 0)
  return Array.from(uniform(key(seed), [n]).data, (u) => {
    let i = cdf.findIndex((c) => c >= u)
    if (i < 0) i = support - 1
    return `w${i}`
  })
}

function exact(stream: readonly string[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const x of stream) m.set(x, (m.get(x) ?? 0) + 1)
  return m
}

describe('countMinSketch', () => {
  const stream = zipf(20_000, 2000, 1)
  const truth = exact(stream)

  it('never underestimates and breaks epsilon N for at most a delta fraction of items', () => {
    const cms = countMinSketch({ epsilon: 0.005, delta: 0.01, seed: 2 })
    for (const x of stream) cms.add(x)
    expect(cms.totalCount).toBe(20_000)
    let bad = 0
    for (const [item, count] of truth) {
      const est = cms.estimate(item)
      expect(est).toBeGreaterThanOrEqual(count)
      if (est - count > 0.005 * 20_000) bad += 1
    }
    expect(bad / truth.size).toBeLessThanOrEqual(0.01)
    expect(cms.width).toBe(Math.ceil(Math.E / 0.005))
    expect(cms.depth).toBe(5)
  })

  it('has tail error that falls with width', () => {
    const maxError = (width: number): number => {
      const cms = countMinSketch({ width, depth: 4, seed: 9 })
      for (const x of stream) cms.add(x)
      return Math.max(...Array.from(truth, ([item, c]) => cms.estimate(item) - c))
    }
    expect(maxError(2000)).toBeLessThan(maxError(100))
  })

  it('conservative update is never worse than plain and never below the truth', () => {
    const plain = countMinSketch({ width: 200, depth: 4, seed: 3 })
    const cons = countMinSketch({ width: 200, depth: 4, seed: 3, conservative: true })
    for (const x of stream) {
      plain.add(x)
      cons.add(x)
    }
    let strictly = 0
    for (const [item, c] of truth) {
      expect(cons.estimate(item)).toBeGreaterThanOrEqual(c)
      expect(cons.estimate(item)).toBeLessThanOrEqual(plain.estimate(item))
      if (cons.estimate(item) < plain.estimate(item)) strictly += 1
    }
    expect(strictly).toBeGreaterThan(0)
    expect(() => cons.merge(plain)).toThrow(/conservative/)
  })

  it('merges to the sketch of the concatenated stream (property)', () => {
    fc.assert(
      fc.property(fc.array(fc.nat(50), { maxLength: 100 }), fc.array(fc.nat(50), { maxLength: 100 }), (xs, ys) => {
        const a = countMinSketch({ width: 32, depth: 3 })
        const b = countMinSketch({ width: 32, depth: 3 })
        const whole = countMinSketch({ width: 32, depth: 3 })
        for (const x of xs) {
          a.add(x)
          whole.add(x)
        }
        for (const y of ys) {
          b.add(y)
          whole.add(y)
        }
        const merged = a.merge(b)
        return (
          Array.from(merged.table).join() === Array.from(whole.table).join() && merged.totalCount === whole.totalCount
        )
      }),
      { numRuns: RUNS },
    )
  })

  it('rejects negative counts', () => {
    expect(() => countMinSketch().add('a', -1)).toThrow(/non-negative/)
  })

  it('the stepped algorithm keeps the maximum overestimate under its bound and serialises', () => {
    const items = stream.slice(0, 2000)
    const alg = countMinAlgorithm({ items, width: 272, depth: 5, seed: 1 })
    const run = runAlgorithm(alg, { maxSteps: 3000 })
    expect(run.state.seen).toBe(2000)
    expect(run.state.maxOverestimate).toBeLessThanOrEqual(run.state.bound)
    expect(run.state.truth['s:w0']).toBe(items.filter((x) => x === 'w0').length)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(again?.maxOverestimate).toBe(run.state.maxOverestimate)
  })
})

describe('countSketch', () => {
  const stream = zipf(20_000, 1000, 5)
  const truth = exact(stream)
  const l2 = Math.sqrt(Array.from(truth.values()).reduce((a, c) => a + c * c, 0))

  it('estimates within epsilon times the L2 norm for nearly every item', () => {
    const cs = countSketch({ epsilon: 0.05, delta: 0.01, seed: 4 })
    for (const x of stream) cs.add(x)
    let bad = 0
    for (const [item, c] of truth) if (Math.abs(cs.estimate(item) - c) > 0.05 * l2) bad += 1
    expect(bad / truth.size).toBeLessThanOrEqual(0.02)
  })

  it('is unbiased: the error averaged over seeds is within its standard error', () => {
    const target = 'w10'
    const errors = Array.from({ length: 60 }, (_, s) => {
      const cs = countSketch({ width: 64, depth: 1, seed: s })
      for (const x of stream) cs.add(x)
      return cs.estimate(target) - (truth.get(target) ?? 0)
    })
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length
    const sd = Math.sqrt(errors.reduce((a, b) => a + (b - mean) ** 2, 0) / (errors.length - 1))
    expect(Math.abs(mean)).toBeLessThan(4 * (sd / Math.sqrt(errors.length)))
    expect(sd).toBeGreaterThan(0)
  })

  it('handles negative updates and merges by addition', () => {
    const a = countSketch({ width: 128, depth: 5, seed: 1 })
    const b = countSketch({ width: 128, depth: 5, seed: 1 })
    a.add('x', 10)
    b.add('x', -4)
    expect(a.merge(b).estimate('x')).toBe(6)
  })

  it('estimates the second moment from the squared counters', () => {
    const cs = countSketch({ width: 512, depth: 5, seed: 8 })
    for (const x of stream) cs.add(x)
    expect(Math.abs(cs.secondMoment() / l2 ** 2 - 1)).toBeLessThan(0.1)
  })
})

describe('spaceSavingTopK', () => {
  const stream = zipf(20_000, 1000, 6)
  const truth = exact(stream)

  it('bounds every counter and keeps every item above N / k', () => {
    const k = 50
    const ss = spaceSavingTopK(k)
    for (const x of stream) ss.add(x)
    const entries = ss.entries()
    expect(entries.reduce((a, e) => a + e.count, 0)).toBe(20_000)
    for (const e of entries) {
      const t = truth.get(String(e.item)) ?? 0
      expect(e.count).toBeGreaterThanOrEqual(t)
      expect(e.count - e.error).toBeLessThanOrEqual(t)
      expect(e.error).toBeLessThanOrEqual(ss.guaranteedBound)
    }
    const monitored = new Set(entries.map((e) => e.item))
    for (const [item, c] of truth) if (c > 20_000 / k) expect(monitored.has(item)).toBe(true)
    expect(ss.heavyHitters(0.02).length).toBeGreaterThanOrEqual(ss.guaranteedHeavyHitters(0.02).length)
    for (const e of ss.guaranteedHeavyHitters(0.02))
      expect(truth.get(String(e.item)) ?? 0).toBeGreaterThan(0.02 * 20_000)
  })

  it('stepped: the largest counter error stays under N / k', () => {
    const run = runAlgorithm(spaceSavingAlgorithm({ items: stream.slice(0, 3000), k: 20 }), { maxSteps: 5000 })
    expect(run.state.maxError).toBeLessThanOrEqual(run.state.bound)
    expect(run.state.entries.length).toBe(20)
  })

  it('rejects a bad k and non-positive counts', () => {
    expect(() => spaceSavingTopK(0)).toThrow(/k must/)
    expect(() => spaceSavingTopK(3).add('a', 0)).toThrow(/positive/)
  })
})

describe('misraGries', () => {
  const stream = zipf(20_000, 1000, 7)
  const truth = exact(stream)

  it('underestimates by at most N / (k + 1) and keeps every item above it', () => {
    const k = 40
    const mg = misraGries(k)
    for (const x of stream) mg.add(x)
    expect(mg.counters().length).toBeLessThanOrEqual(k)
    for (const [item, c] of truth) {
      const est = mg.estimate(item)
      expect(est).toBeLessThanOrEqual(c)
      expect(est).toBeGreaterThanOrEqual(c - mg.guaranteedBound)
      if (c > mg.guaranteedBound) expect(est).toBeGreaterThan(0)
    }
  })

  it('keeps the bound after merging summaries of two halves', () => {
    const k = 30
    const a = misraGries(k)
    const b = misraGries(k)
    for (const x of stream.slice(0, 10_000)) a.add(x)
    for (const x of stream.slice(10_000)) b.add(x)
    const merged = a.merge(b)
    expect(merged.counters().length).toBeLessThanOrEqual(k)
    for (const [item, c] of truth) {
      expect(merged.estimate(item)).toBeLessThanOrEqual(c)
      expect(merged.estimate(item)).toBeGreaterThanOrEqual(c - merged.guaranteedBound)
    }
  })

  it('finds a strict majority with one counter (property)', () => {
    fc.assert(
      fc.property(fc.nat(30), fc.array(fc.nat(5), { maxLength: 60 }), (extra, rest) => {
        const stream2 = [...new Array<number>(rest.length + extra + 1).fill(99), ...rest]
        const mg = misraGries(1)
        for (const x of stream2) mg.add(x)
        return mg.counters()[0]?.item === 99
      }),
      { numRuns: RUNS },
    )
  })
})
