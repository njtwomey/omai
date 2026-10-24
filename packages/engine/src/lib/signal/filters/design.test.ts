import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../../testing/check.ts'
import { absComplex, tensor, toFlat } from '../../tensor/index.ts'
import * as S from '../index.ts'

const gainAt = (f: S.FilterDesign, w: number): number =>
  toFlat(absComplex(S.frequencyResponse(f, [w], { fs: 2 }).h))[0] ?? 0

describe('IIR and FIR designs', () => {
  it('meets the specifications their order functions promise', () => {
    const spec: S.FilterSpec = { wp: 0.2, ws: 0.3, gpass: 1, gstop: 40 }
    for (const ftype of ['butter', 'cheby1', 'cheby2', 'ellip'] as const) {
      const check = S.checkSpecification(S.iirdesign(spec, { ftype, output: 'sos' }), spec)
      expect(check.meets).toBe(true)
    }
    const band: S.FilterSpec = { wp: [0.2, 0.4], ws: [0.1, 0.5], gpass: 0.5, gstop: 50, fs: 2 }
    expect(S.checkSpecification(S.iirdesign(band, { output: 'sos' }), band).meets).toBe(true)
    const stop: S.FilterSpec = { wp: [0.1, 0.6], ws: [0.2, 0.5], gpass: 1, gstop: 30 }
    expect(S.checkSpecification(S.iirdesign(stop, { ftype: 'cheby1', output: 'sos' }), stop).meets).toBe(true)
  })

  it('places the -3 dB point of a Butterworth filter at its cutoff and has unit DC gain', () => {
    const f = S.butter(5, 0.3)
    expect(gainAt(f, 0)).toBeCloseTo(1, 12)
    expect(gainAt(f, 0.3)).toBeCloseTo(Math.SQRT1_2, 10)
    const hp = S.butter(5, 0.3, { btype: 'highpass', output: 'zpk' })
    expect(gainAt(hp, 0.999)).toBeCloseTo(1, 4)
  })

  it('designs notch and peak filters with unit gain where they should', () => {
    const notch = S.iirnotch(50, 10, 1000)
    expect(toFlat(absComplex(S.frequencyResponse(notch, [50]).h))[0]).toBeLessThan(1e-12)
    const peak = S.iirpeak(50, 10, 1000)
    expect(toFlat(absComplex(S.frequencyResponse(peak, [50]).h))[0]).toBeCloseTo(1, 12)
  })

  it('makes equiripple and windowed FIR filters with the right gains', () => {
    const h = S.remez(41, [0, 0.15, 0.2, 0.5], [1, 0])
    const r = S.frequencyResponse({ b: h }, [0, 0.05, 0.3, 0.45], { fs: 1 })
    const mag = toFlat(absComplex(r.h))
    expect(mag[0]).toBeCloseTo(1, 1)
    expect(mag[2]).toBeLessThan(0.05)
    const lin = S.firwin(31, 0.4)
    const gd = toFlat(S.groupDelay(lin, [1], 8).delay)
    for (const v of gd) expect(v).toBeCloseTo(15, 8)
    const pd = toFlat(S.phaseDelay(lin, [1], [0.01, 0.1, 0.3], { fs: 2 }).delay)
    for (const v of pd) expect(v).toBeCloseTo(15, 8)
    // A minimum-phase version keeps (the square root of) the magnitude and moves the energy forward.
    const mp = toFlat(S.minimumPhase(lin))
    const energy = (xs: number[], k: number): number => xs.slice(0, k).reduce((a, v) => a + v * v, 0)
    expect(energy(mp, 4) / energy(mp, mp.length)).toBeGreaterThan(energy(toFlat(lin), 4) / energy(toFlat(lin), 31))
  })

  it('rejects invalid designs', () => {
    expect(() => S.butter(2, 1.2)).toThrow()
    expect(() => S.butter(2, [0.4, 0.2], { btype: 'bandpass' })).toThrow()
    expect(() => S.cheby1(2, Number.NaN, 0.2, { btype: 'bandpass' })).toThrow()
    expect(() => S.firwin(10, 0.3, { passZero: false })).toThrow()
    expect(() => S.iircomb(300, 10, { fs: 1000 })).toThrow()
    expect(() => S.kaiserord(5, 0.1)).toThrow()
  })

  it('differentiates frequency responses through the coefficients', () => {
    expectConformant((s) => absComplex(S.sosfreqz(s, 4).h), [tensor([[0.3, 0.2, 0.1, 1, -0.4, 0.1]])])
  })
})
