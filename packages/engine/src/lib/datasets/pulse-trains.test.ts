import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { convolutionalSynthesis } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { datasetGenerators } from './catalogue.ts'
import { pulseTrain } from './pulse-trains.ts'

describe('pulseTrain', () => {
  const g = generate(pulseTrain, {
    key: key(3),
    knobs: { n: 300, pulses: 3, length: 10, occurrences: 4, noise: 0.05 },
  })
  const extra = g.truth.extra as {
    filters: number[][]
    codes: number[][]
    positions: number[][]
    amplitudes: number[][]
    clean: number[]
  }

  it('is a one-channel series with the truth of its generative model', () => {
    expect(g.data.values.shape).toEqual([300, 1])
    expect(extra.filters.length).toBe(3)
    expect(extra.codes.map((c) => c.length)).toEqual([291, 291, 291])
    for (const f of extra.filters) expect(Math.hypot(...f)).toBeCloseTo(1, 12)
  })

  it('has the clean signal equal to the filters convolved with the planted sparse maps', () => {
    const made = toFlat(convolutionalSynthesis(extra.filters, extra.codes))
    for (const [n, v] of extra.clean.entries()) expect(made[n]).toBeCloseTo(v, 12)
    for (const k of [0, 1, 2]) {
      expect(extra.codes[k]?.filter((v) => v !== 0).length).toBe(4)
      for (const [j, p] of (extra.positions[k] ?? []).entries())
        expect(extra.codes[k]?.[p]).toBe(extra.amplitudes[k]?.[j])
    }
  })

  it('places no two copies on top of each other, with sizes from 0.5 to 2 and both signs', () => {
    const starts = extra.positions.flat().sort((a, b) => a - b)
    for (let i = 1; i < starts.length; i++)
      expect((starts[i] as number) - (starts[i - 1] as number)).toBeGreaterThanOrEqual(10)
    const sizes = extra.amplitudes.flat()
    for (const a of sizes) expect(Math.abs(a)).toBeGreaterThanOrEqual(0.5)
    for (const a of sizes) expect(Math.abs(a)).toBeLessThanOrEqual(2)
    expect(sizes.some((a) => a < 0) && sizes.some((a) => a > 0)).toBe(true)
  })

  it('adds noise of the stated size and is reproducible from the key', () => {
    const y = toFlat(g.data.values)
    const noise = y.map((v, n) => v - (extra.clean[n] ?? 0))
    const sd = Math.sqrt(noise.reduce((a, b) => a + b * b, 0) / noise.length)
    expect(sd).toBeGreaterThan(0.03)
    expect(sd).toBeLessThan(0.07)
    const again = generate(pulseTrain, {
      key: key(3),
      knobs: { n: 300, pulses: 3, length: 10, occurrences: 4, noise: 0.05 },
    })
    expect(Array.from(toFlat(again.data.values))).toEqual(Array.from(y))
    expect(
      generate(pulseTrain, { key: key(4), knobs: { n: 300, pulses: 3, length: 10, occurrences: 4 } }).data.values,
    ).not.toEqual(g.data.values)
  })

  it('places as many copies as fit when the signal is short', () => {
    const short = generate(pulseTrain, { key: key(1), knobs: { n: 100, pulses: 2, length: 10, occurrences: 50 } })
    const positions = short.truth.extra?.['positions'] as number[][]
    expect(positions.map((p) => p.length)).toEqual([2, 2])
  })

  it('refuses a signal too short for its copies, and is in the catalogue', () => {
    expect(() => generate(pulseTrain, { knobs: { n: 100, pulses: 4, length: 20, occurrences: 5 } })).toThrow(/at least/)
    expect(datasetGenerators['pulseTrain']).toBe(pulseTrain)
  })
})
