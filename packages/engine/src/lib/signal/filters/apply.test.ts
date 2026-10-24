import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { expectConformant } from '../../../testing/check.ts'
import { grad, vmap } from '../../autodiff/index.ts'
import { sos as makeSos, tf, zpk } from '../../systems/index.ts'
import { stack, sum, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import * as S from '../index.ts'

const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7])

describe('lfilter derivatives', () => {
  it('passes the conformance check in the signal, coefficients and initial state', () => {
    const b = tensor([0.4, 0.2, -0.1])
    const a = tensor([1.1, -0.5, 0.2])
    expectConformant((bb, aa, xx) => S.lfilter(bb, aa, xx), [b, a, x])
    expectConformant((bb, aa, xx, zi) => S.lfilter(bb, aa, xx, { zi }).y, [b, a, x, tensor([0.3, -0.2])])
    expectConformant((bb, aa, xx, zi) => S.lfilter(bb, aa, xx, { zi }).zf, [b, a, x, tensor([0.3, -0.2])])
    expectConformant((bb, aa, xx, zi) => S.lfilter(bb, aa, xx, { zi }).zf, [b, a, tensor([1, 2]), tensor([0.3, -0.2])])
  })

  it('has second derivatives', () => {
    const f = (a: Value): Value => sum(S.lfilter([1], a, x))
    expectConformant((a) => grad(f)(a), [tensor([1, -0.4])])
  })

  it('vectorises over signals', () => {
    const rows = tensor([
      [1, 0, 0, 0],
      [0, 1, 2, 3],
    ])
    const batched = vmap((r: Value) => S.lfilter([1, 1], [1, -0.5], r))(rows) as Value
    const looped = stack([S.lfilter([1, 1], [1, -0.5], [1, 0, 0, 0]), S.lfilter([1, 1], [1, -0.5], [0, 1, 2, 3])])
    expect(toArray(batched)).toEqual(toArray(looped))
  })

  it('differentiates the composed filters', () => {
    const sos = tensor([
      [0.2, 0.4, 0.2, 1, -0.3, 0.1],
      [1, -1, 0, 1, -0.2, 0],
    ])
    expectConformant((s, xx) => S.sosfilt(s, xx), [sos, x])
    expectConformant((bb, aa, xx) => S.filtfilt(bb, aa, xx, { padlen: 3 }), [tensor([0.5, 0.5]), tensor([1, -0.3]), x])
    expectConformant((bb, xx) => S.filtfilt(bb, [1, -0.3], xx, { method: 'gust' }), [tensor([0.5, 0.5]), x])
    expectConformant((xx) => S.sosfiltfilt(sos, xx, { padlen: 2 }), [x])
    expectConformant((xx) => S.savgolFilter(xx, 5, 2), [x])
    expectConformant((xx) => S.medfilt(xx, 3), [x])
    expectConformant((bb, aa) => S.lfilterZi(bb, aa), [tensor([0.5, 0.2]), tensor([1, -0.3])])
  })
})

describe('filter identities', () => {
  it('splits a signal: filtering in two pieces with states equals filtering at once', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 6, maxLength: 12 }),
        fc.integer({ min: 1, max: 5 }),
        (xs, cut) => {
          const b = [0.3, 0.2, 0.1]
          const a = [1, -0.4, 0.1]
          const whole = toFlat(S.lfilter(b, a, xs))
          const first = S.lfilter(b, a, xs.slice(0, cut), { zi: [0, 0] })
          const second = S.lfilter(b, a, xs.slice(cut), { zi: first.zf })
          const joined = [...toFlat(first.y), ...toFlat(second.y)]
          joined.forEach((v, i) => {
            expect(v).toBeCloseTo(whole[i] ?? 0, 10)
          })
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('agrees across representations and with lfiltic', () => {
    const b = [0.2, 0.3]
    const a = [1, -0.5]
    const xs = [1, 2, -1, 0.5, 3]
    const direct = toFlat(S.lfilter(b, a, xs))
    const viaSystem = toFlat(S.applyFilter(tf([0.2, 0.3], [1, -0.5], { dt: 1 }), xs))
    for (const [i, v] of viaSystem.entries()) expect(v).toBeCloseTo(direct[i] ?? 0, 12)
    const viaZpk = toFlat(S.applyFilter(zpk([-1.5], [0.5], 0.2, { dt: 1 }), xs))
    for (const [i, v] of viaZpk.entries()) expect(v).toBeCloseTo(direct[i] ?? 0, 12)
    const viaSos = toFlat(S.applyFilter(makeSos([[0.2, 0.3, 0, 1, -0.5, 0]], { dt: 1 }), xs))
    for (const [i, v] of viaSos.entries()) expect(v).toBeCloseTo(direct[i] ?? 0, 12)
    // Continue a filtering from its past outputs and inputs.
    const head = toFlat(S.lfilter(b, a, xs.slice(0, 3)))
    const zi = S.lfiltic(b, a, [head[2] ?? 0], [xs[2] ?? 0])
    const tail = toFlat(S.lfilter(b, a, xs.slice(3), { zi }).y)
    for (const [i, v] of tail.entries()) expect(v).toBeCloseTo(direct[3 + i] ?? 0, 12)
    expect(() => S.lfilter([1], [0, 1], xs)).toThrow()
    expect(() => S.applyFilter(tf([1], [1, 1]), xs)).toThrow()
    expect(() => S.filtfilt(b, a, [1, 2, 3])).toThrow()
  })
})
