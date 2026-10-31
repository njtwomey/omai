import { describe, expect, it } from 'vitest'
import { evalfr, stabilityMargins, tf } from '../../systems/index.ts'
import { item, toFlat } from '../../tensor/index.ts'
import { lagCompensator, lagDesign, leadCompensator, leadDesign, leadLag } from './compensators.ts'

const at = (sys: ReturnType<typeof tf>, w: number): { gain: number; phase: number } => {
  const z = evalfr(sys, { re: 0, im: w })
  return { gain: Math.hypot(item(z.re), item(z.im)), phase: (Math.atan2(item(z.im), item(z.re)) * 180) / Math.PI }
}

describe('lead and lag compensators', () => {
  it('adds the requested phase at the centre, the most it adds anywhere, with gain sqrt(alpha) there', () => {
    for (const phase of [10, 30, 55, 70]) {
      const c = leadCompensator({ phase, centre: 7 })
      expect(at(c.system, 7).phase).toBeCloseTo(phase, 9)
      expect(at(c.system, 7).gain).toBeCloseTo(Math.sqrt(c.ratio), 12)
      for (const w of [3, 5, 6.5, 7.5, 10, 20]) expect(at(c.system, w).phase).toBeLessThan(phase + 1e-9)
      expect(Math.sqrt(c.zero * c.pole)).toBeCloseTo(7, 12)
      expect(c.pole / c.zero).toBeCloseTo(c.ratio, 12)
    }
  })

  it('has unit DC gain by default, unit gain at the centre when asked, and gain alpha at high frequency', () => {
    const dc = leadCompensator({ phase: 40, centre: 3 })
    expect(at(dc.system, 1e-6).gain).toBeCloseTo(1, 6)
    expect(at(dc.system, 1e6).gain).toBeCloseTo(dc.ratio, 4)
    expect(at(leadCompensator({ phase: 40, centre: 3, unitGain: 'centre' }).system, 3).gain).toBeCloseTo(1, 12)
  })

  it('multiplies the DC gain by the ratio and leaves the high-frequency gain alone, at a bounded phase cost', () => {
    const c = lagCompensator({ zero: 0.5, ratio: 10 })
    expect(at(c.system, 1e-6).gain).toBeCloseTo(10, 5)
    expect(at(c.system, 1e6).gain).toBeCloseTo(1, 5)
    const worst = Math.min(...Array.from({ length: 400 }, (_, i) => at(c.system, 0.01 * 1.02 ** i).phase))
    expect(worst).toBeCloseTo(-c.maxPhase, 2)
    expect(at(lagCompensator({ zero: 0.5, ratio: 10, unitGain: 'dc' }).system, 1e-6).gain).toBeCloseTo(1, 6)
  })

  it('rejects impossible compensators', () => {
    expect(() => leadCompensator({ phase: 0, centre: 1 })).toThrow()
    expect(() => leadCompensator({ phase: 90, centre: 1 })).toThrow()
    expect(() => leadCompensator({ phase: 30, centre: -1 })).toThrow()
    expect(() => lagCompensator({ zero: 1, ratio: 1 })).toThrow()
    expect(() => leadLag()).toThrow()
  })

  it('composes in series', () => {
    const lead = leadCompensator({ phase: 30, centre: 5 })
    const lag = lagCompensator({ zero: 0.5, ratio: 4 })
    const both = leadLag(lead, lag)
    expect(toFlat(both.den).length).toBe(3)
    expect(at(both, 2).gain).toBeCloseTo(at(lead.system, 2).gain * at(lag.system, 2).gain, 10)
  })
})

describe('the design rules', () => {
  it('puts the lead’s centre where the plant has gain 1 / sqrt(alpha) and lands near the margin asked', () => {
    const plant = tf([4], [1, 2, 0])
    expect(stabilityMargins(plant).pm).toBeCloseTo(51.827, 2)
    const d = leadDesign(plant, { phaseMargin: 60 })
    expect(at(plant, d.centre).gain).toBeCloseTo(1 / Math.sqrt(d.ratio), 8)
    // The new crossover is the centre, where the lead gives its full phase.
    expect(d.after.wgc).toBeCloseTo(d.centre, 6)
    expect(d.after.wgc).toBeGreaterThan(d.before.wgc)
    expect(d.after.pm).toBeGreaterThan(57)
    expect(d.after.pm).toBeLessThan(63)
    // More allowance, more margin.
    expect(leadDesign(plant, { phaseMargin: 60, allowance: 12 }).after.pm).toBeGreaterThan(d.after.pm)
  })

  it('refuses a plant that already has the margin, has no crossover, or needs more than one lead', () => {
    expect(() => leadDesign(tf([1], [1, 2, 0]), { phaseMargin: 40, allowance: -30 })).toThrow()
    expect(() => leadDesign(tf([0.1], [1, 1]), { phaseMargin: 60 })).toThrow()
    expect(() => leadDesign(tf([4], [1, 2, 0]), { phaseMargin: 170 })).toThrow()
  })

  it('raises the velocity constant by the ratio with a lag a decade below the crossover, costing a few degrees', () => {
    const plant = tf([2], [1, 3, 2, 0])
    const d = lagDesign(plant, { ratio: 10 })
    expect(d.zero).toBeCloseTo(d.before.wgc / 10, 12)
    expect(d.phaseAtCrossover).toBeGreaterThan(-6)
    expect(d.phaseAtCrossover).toBeLessThan(-1)
    expect(d.before.pm - d.after.pm).toBeLessThan(8)
    expect(d.after.wgc).toBeCloseTo(d.before.wgc, 0)
    // Kv = lim s G(s): the lag multiplies it by the ratio.
    const kv = (sys: ReturnType<typeof tf>): number => at(sys, 1e-6).gain * 1e-6
    expect(kv(d.loop as ReturnType<typeof tf>) / kv(plant)).toBeCloseTo(10, 3)
    expect(() => lagDesign(tf([0.1], [1, 1]), { ratio: 5 })).toThrow()
  })
})
