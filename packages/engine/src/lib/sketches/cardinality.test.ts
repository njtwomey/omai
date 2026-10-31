/**
 * Tests for HyperLogLog: accuracy against the published standard error across the whole range (empty, tiny, linear
 * counting, intermediate, large), absence of bias over many seeds, merge as register-wise maximum, idempotence under
 * repeats, and the stepped algorithm.
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { hllMerge, hyperLogLog, hyperLogLogAlgorithm, hyperLogLogEstimate } from './index.ts'

const filled = (n: number, seed: number, precision = 10, from = 0): ReturnType<typeof hyperLogLog> => {
  const hll = hyperLogLog({ precision, seed })
  for (let i = 0; i < n; i++) hll.add(`user-${from + i}`)
  return hll
}

describe('hyperLogLog', () => {
  it('is exact-ish on an empty and a tiny sketch', () => {
    expect(hyperLogLog().count()).toBe(0)
    expect(hyperLogLog().count('classic')).toBe(0)
    expect(filled(1, 0).count()).toBeCloseTo(1, 1)
    expect(Math.abs(filled(10, 0).count() - 10)).toBeLessThan(1)
  })

  it('stays within four standard errors from linear counting to saturation of the registers', () => {
    const m = 1 << 10
    const se = 1.04 / Math.sqrt(m)
    for (const n of [20, 100, 500, 1500, 2600, 5000, 20_000, 100_000]) {
      const hll = filled(n, 7)
      expect(Math.abs(hll.count() / n - 1)).toBeLessThan(4 * se)
    }
  })

  it('has no visible bias averaged over seeds, in the range where the classical estimator is biased', () => {
    const n = 3000
    const seeds = Array.from({ length: 60 }, (_, s) => s)
    const errors = seeds.map((s) => filled(n, s).count() / n - 1)
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length
    const sd = Math.sqrt(errors.reduce((a, b) => a + (b - mean) ** 2, 0) / (errors.length - 1))
    expect(Math.abs(mean)).toBeLessThan(4 * (sd / Math.sqrt(errors.length)))
    // the spread is the published 1.04 / sqrt(m), within sampling error of 60 seeds
    expect(sd).toBeGreaterThan(0.6 * (1.04 / Math.sqrt(1024)))
    expect(sd).toBeLessThan(1.5 * (1.04 / Math.sqrt(1024)))
  })

  it('the error shrinks as 1 / sqrt(m) with the precision', () => {
    const rms = (precision: number): number => {
      const errs = Array.from({ length: 30 }, (_, s) => filled(4000, s, precision).count() / 4000 - 1)
      return Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length)
    }
    expect(rms(12)).toBeLessThan(rms(8) * 0.5)
  })

  it('repeats change nothing and merging is the sketch of the union (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 500 }), { maxLength: 200 }),
        fc.array(fc.integer({ min: 0, max: 500 }), { maxLength: 200 }),
        (xs, ys) => {
          const a = hyperLogLog({ precision: 8 })
          const b = hyperLogLog({ precision: 8 })
          const both = hyperLogLog({ precision: 8 })
          for (const x of xs) {
            a.add(x)
            a.add(x)
            both.add(x)
          }
          for (const y of ys) {
            b.add(y)
            both.add(y)
          }
          const merged = hllMerge(a, b)
          return (
            Array.from(merged.registers).join() === Array.from(both.registers).join() && merged.count() === both.count()
          )
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('refuses to merge sketches of different shape', () => {
    expect(() => hllMerge(hyperLogLog({ precision: 8 }), hyperLogLog({ precision: 9 }))).toThrow(/precision/)
    expect(() => hyperLogLog({ precision: 3 })).toThrow(/precision/)
  })

  it('takes its seed from a key reproducibly', () => {
    const a = hyperLogLog({ seed: key(4) })
    const b = hyperLogLog({ seed: key(4) })
    for (let i = 0; i < 100; i++) {
      a.add(i)
      b.add(i)
    }
    expect(Array.from(a.registers)).toEqual(Array.from(b.registers))
  })

  it('the classical estimator matches the formula on a hand-built register file', () => {
    const regs = Uint8Array.from([1, 1, ...new Array(14).fill(0)])
    // zeros present and raw <= 2.5 m: linear counting m ln(m / zeros)
    expect(hyperLogLogEstimate(regs, 'classic')).toBeCloseTo(16 * Math.log(16 / 14), 12)
  })
})

describe('hyperLogLogAlgorithm', () => {
  it('tracks the true cardinality of a stream with repeats and serialises', () => {
    const items = Array.from({ length: 3000 }, (_, i) => `item-${(i * 7) % 1200}`)
    const alg = hyperLogLogAlgorithm({ items, precision: 10 })
    const run = runAlgorithm(alg, { maxSteps: 5000 })
    expect(run.steps).toBe(3000)
    expect(run.state.trueCardinality).toBe(1200)
    expect(run.state.relativeError).toBeLessThan(0.13)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(Array.from(again?.registers ?? [])).toEqual(Array.from(run.state.registers))
  })
})
