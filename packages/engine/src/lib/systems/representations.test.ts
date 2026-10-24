import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { isWhiteboxError } from '../base/index.ts'
import { type Tensor, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const close = (a: readonly number[], b: readonly number[], tol = 1e-10): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries())
    expect(Math.abs(v - (b[i] ?? Number.NaN))).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(v)))
}

const a3: Tensor = tensor([
  [-1, 0.5, 0.2],
  [0.3, -2, 1],
  [0, -0.4, -1.5],
])
const b32: Tensor = tensor([
  [1, 0],
  [0.5, -1],
  [0, 2],
])
const c23: Tensor = tensor([
  [1, 0, -1],
  [0.2, 1, 0.4],
])
const d22: Tensor = tensor([
  [0, 0.1],
  [0.3, 0],
])

describe('constructors and conversions', () => {
  it('build systems of each kind and report their dimensions', () => {
    expect(S.dimensions(S.ss(a3, b32, c23, d22))).toEqual({ inputs: 2, outputs: 2, states: 3 })
    expect(S.dimensions(S.zpk([-1], [-2, -3], 1))).toEqual({ inputs: 1, outputs: 1, states: 2 })
    expect(S.dimensions(S.sos([[1, 0, 0, 1, 0.5, 0.1]], { dt: 1 }))).toEqual({ inputs: 1, outputs: 1, states: 2 })
    expect(
      S.dimensions(
        S.tf(
          [
            [1, 2],
            [0, 1],
          ],
          [1, 3],
        ),
      ),
    ).toEqual({ inputs: 1, outputs: 2, states: 1 })
    expect(S.isDiscrete(S.ss(a3, b32, c23, d22, { dt: 0.2 }))).toBe(true)
    expect(() => S.ss(a3, b32, b32)).toThrow()
    expect(() => S.tf([1], [1], { dt: -1 })).toThrow()
    expect(() => S.sos([[1, 2, 3]])).toThrow()
  })

  it('reject an improper transfer function and a zero denominator', () => {
    expect(() => S.tf2ss([1, 2, 3], [1, 2])).toThrow()
    try {
      S.normalize([1], [0, 0])
    } catch (e) {
      expect(isWhiteboxError(e, 'value')).toBe(true)
    }
  })

  it('round-trip tf -> ss -> tf and tf -> zpk -> tf, and keep the response through sections', () => {
    const num = [0.5, -1, 2]
    const den = [2, 1, 3, 0.4]
    const ref = S.normalize(num, den)
    const g = S.tf(num, den)
    for (const back of [S.toTf(S.toSs(g)), S.toTf(S.toZpk(g))]) {
      const pad = toFlat(back.num as Tensor)
      const want = toFlat(ref.num as Tensor)
      close(pad.slice(pad.length - want.length), want, 1e-9)
      close(toFlat(back.den as Tensor), toFlat(ref.den as Tensor), 1e-9)
    }
    const gd = S.tf(num, den, { dt: 1 })
    const w = [0.1, 0.7, 2]
    const want = S.freqresp(gd, w).h
    for (const sys of [S.toSos(gd), S.toTf(S.toSos(gd)), S.toZpk(S.toSos(gd)), S.toSs(S.toSos(gd))]) {
      const h = S.freqresp(sys, w).h
      close(toFlat(h.re as Tensor), toFlat(want.re as Tensor), 1e-9)
      close(toFlat(h.im as Tensor), toFlat(want.im as Tensor), 1e-9)
    }
  })

  it('give every channel of a multivariable system', () => {
    const s = S.ss(a3, b32, c23, d22)
    const g = S.toTf(s, { input: 1 })
    expect(g.num.shape).toEqual([2, 4])
    const z = S.toZpk(s, { input: 1, output: 1 })
    expect(z.poles.re.length).toBe(3)
  })

  it('find minimal realisations', () => {
    // An uncontrollable and an unobservable mode out of three.
    const s = S.ss(
      [
        [-1, 0, 0],
        [0, -2, 0],
        [0, 0, -3],
      ],
      [[1], [0], [1]],
      [[1, 1, 0]],
    )
    const m = S.minreal(s) as S.StateSpace
    expect(S.dimensions(m).states).toBe(1)
    close(toFlat(S.toTf(m).den as Tensor), [1, 1])
    const z = S.minreal(S.zpk([-1, -4], [-1, -2, -3], 2)) as S.ZerosPolesGain
    expect([...z.poles.re].sort((x, y) => x - y)).toEqual([-3, -2])
    const sec = S.minreal(S.toSos(S.zpk([0.5], [0.5, 0.2], 1, { dt: 1 }))) as S.SecondOrderSections
    expect(S.toZpk(sec).poles.re.filter((p) => Math.abs(p - 0.5) < 1e-9)).toEqual([])
    expect(toArray(S.minreal(S.tf([0], [1, 2])).den as Tensor)).toEqual([1])
  })
})

describe('derivatives', () => {
  it('ss2tf is differentiable in A, B, C and D', () => {
    const den = (a: Value): Value => S.ss2tf(a, b32, c23, d22, 1).den
    const num = (a: Value, b: Value, c: Value, d: Value): Value => S.ss2tf(a, b, c, d, 1).num
    expect(checkPrimitive(den, [a3], { vmap: false }).failures).toEqual([])
    expect(checkPrimitive(num, [a3, b32, c23, d22], { vmap: false }).failures).toEqual([])
  })

  it('tf2ss and sos2tf are differentiable in the coefficients', () => {
    const c = (n: Value, d: Value): Value => S.tf2ss(n, d).c
    expect(checkPrimitive(c, [tensor([1, -0.5, 2]), tensor([2, 0.3, 1, 0.4])], { vmap: false }).failures).toEqual([])
    const den = (s: Value): Value => S.sos2tf(s).den
    expect(
      checkPrimitive(
        den,
        [
          tensor([
            [1, 0.2, 0, 1, -0.5, 0.1],
            [1, 1, 0.5, 1, 0.3, 0.2],
          ]),
        ],
        { vmap: false },
      ).failures,
    ).toEqual([])
  })
})
