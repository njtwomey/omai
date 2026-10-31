import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { evalfr } from './frequency.ts'
import { pade, ss, tf } from './representations.ts'
import { step, stepInfo, stepMeasures } from './responses.ts'

const grid = (end: number, n: number): number[] => Array.from({ length: n + 1 }, (_, i) => (i * end) / n)

describe('stepInfo', () => {
  it('matches the closed forms of a second-order system: overshoot, peak time, and the final value', () => {
    for (const zeta of [0.1, 0.3, 0.6]) {
      const wn = 2
      const sys = tf([wn * wn], [1, 2 * zeta * wn, wn * wn])
      const m = stepInfo(sys, { t: grid(40, 4000) })
      const wd = wn * Math.sqrt(1 - zeta * zeta)
      expect(m.overshoot).toBeCloseTo(100 * Math.exp((-Math.PI * zeta) / Math.sqrt(1 - zeta * zeta)), 1)
      expect(m.peakTime).toBeCloseTo(Math.PI / wd, 1)
      expect(m.peak).toBeCloseTo(1 + m.overshoot / 100, 6)
      expect(m.steadyStateValue).toBeCloseTo(1, 12)
      expect(m.undershoot).toBe(0)
      // The settling time is inside the 2% envelope exp(-zeta wn t) / sqrt(1 - zeta^2) <= 0.02, loosely.
      expect(m.settlingTime).toBeLessThan((Math.log(50) + 0.5 * Math.log(1 / (1 - zeta * zeta))) / (zeta * wn) + 1 / wd)
      expect(m.settlingTime).toBeGreaterThan((Math.log(50 * 0.9) / (zeta * wn)) * 0.5)
    }
  })

  it('has the first-order lag rise in ln(9) tau and settle in ln(50) tau, without overshoot', () => {
    const tau = 0.5
    const m = stepInfo(tf([1], [tau, 1]), { t: grid(5, 5000) })
    expect(m.riseTime).toBeCloseTo(Math.log(9) * tau, 2)
    expect(m.settlingTime).toBeCloseTo(Math.log(50) * tau, 2)
    expect(m.overshoot).toBe(0)
    expect(m.settlingMin).toBeCloseTo(0.9, 2)
    expect(m.settlingMax).toBe(1)
  })

  it('reads an undershoot off a non-minimum-phase zero and the sign off a negative gain', () => {
    const nmp = stepInfo(tf([-1, 1], [1, 3, 2]), { t: grid(20, 2000) })
    expect(nmp.undershoot).toBeGreaterThan(5)
    const negative = stepInfo(tf([-3], [1, 2, 5]), { t: grid(20, 2000) })
    expect(negative.steadyStateValue).toBeCloseTo(-0.6, 12)
    expect(negative.overshoot).toBeGreaterThan(0)
    expect(negative.riseTime).toBeGreaterThan(0)
  })

  it('takes the band and the rise limits as options and the default time grid when none is given', () => {
    const sys = tf([4], [1, 0.8, 4])
    const t = grid(20, 2000)
    expect(stepInfo(sys, { t, settlingThreshold: 0.05 }).settlingTime).toBeLessThan(stepInfo(sys, { t }).settlingTime)
    expect(stepInfo(sys, { t, riseLimits: [0.2, 0.8] }).riseTime).toBeLessThan(stepInfo(sys, { t }).riseTime)
    expect(stepInfo(sys).overshoot).toBeGreaterThan(30)
    expect(() => stepInfo(tf([1], [1, 1]), { t })).not.toThrow()
  })

  it('reports NaN for a response that never reaches the rise limit or never settles, and refuses MIMO', () => {
    const slow = stepMeasures([0, 1, 2], [0, 0.01, 0.02], { final: 1 })
    expect(slow.riseTime).toBeNaN()
    expect(slow.steadyStateValue).toBe(1)
    const ringing = stepMeasures(
      grid(10, 100),
      grid(10, 100).map((t) => 1 + Math.sin(3 * t)),
      { final: 1 },
    )
    expect(ringing.settlingTime).toBeNaN()
    expect(stepMeasures([0, 1], [0, 1], { final: Number.POSITIVE_INFINITY }).overshoot).toBeNaN()
    expect(() => stepMeasures([0, 1], [0], {})).toThrow()
    expect(() =>
      stepInfo(
        ss(
          [
            [-1, 0],
            [0, -2],
          ],
          [
            [1, 0],
            [0, 1],
          ],
          [
            [1, 0],
            [0, 1],
          ],
        ),
      ),
    ).toThrow(/single-input/)
  })

  it('agrees with measures taken from the step response itself', () => {
    const sys = tf([4], [1, 0.8, 4])
    const r = step(sys, { n: 2000 })
    const direct = stepMeasures(toFlat(r.t), toFlat(r.y), { final: 1 })
    expect(stepInfo(sys, { n: 2000 }).peakTime).toBe(direct.peakTime)
  })
})

describe('pade', () => {
  it('matches the series of the delay through degree order + numOrder', () => {
    const T = 2
    const n = 2
    const { num, den } = pade(T, n)
    // num(s) - e^{-sT} den(s) = O(s^(2n + 1)): evaluate at small s.
    const p = (c: number[], s: number): number => c.reduce((acc, a) => acc * s + a, 0)
    const error = (s: number): number => Math.abs(p(toFlat(num), s) - Math.exp(-s * T) * p(toFlat(den), s))
    const ratio = error(0.2) / error(0.1)
    expect(ratio).toBeGreaterThan(2 ** (2 * n + 1) * 0.85)
    expect(ratio).toBeLessThan(2 ** (2 * n + 1) * 1.15)
  })

  it('is all-pass with the phase of the delay at low frequency, and has right-half-plane zeros', () => {
    const T = 1.3
    const sys = pade(T, 3)
    for (const w of [0.1, 0.5, 1]) {
      const { re, im } = evalfr(sys, { re: 0, im: w } as never)
      expect(Math.hypot(toFlat(re)[0] as number, toFlat(im)[0] as number)).toBeCloseTo(1, 12)
      expect(Math.atan2(toFlat(im)[0] as number, toFlat(re)[0] as number)).toBeCloseTo(-w * T, 3)
    }
    const num = toFlat(sys.num)
    // Alternating signs of the numerator: every zero is in the right half-plane (the mirror of the poles).
    for (const [k, c] of num.entries()) expect(Math.sign(c)).toBe(k % 2 === 0 ? -1 : 1)
  })

  it('undershoots at first, as a delay approximated by a non-minimum-phase system must', () => {
    const r = step(pade(1, 2), { t: grid(5, 500) })
    expect(Math.min(...toFlat(r.y))).toBeLessThan(-0.1)
    // A strictly proper one has no such jump.
    const proper = pade(1, 3, -2)
    expect(toFlat(proper.num).length).toBe(2)
    expect(toFlat(proper.den).length).toBe(4)
  })

  it('is the identity for no delay and rejects bad arguments', () => {
    expect(toFlat(pade(0, 3).num)).toEqual([1])
    expect(() => pade(-1, 2)).toThrow()
    expect(() => pade(1, -1)).toThrow()
    expect(() => pade(1, 2, 3)).toThrow()
  })
})
