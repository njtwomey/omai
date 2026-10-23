import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { absComplex, angle, cAdd, cDiv, cExp, cis, cMul, complex, conj, imag, polar, real } from './complex.ts'
import { item, tensor } from './tensor.ts'

const z = fc.record({
  re: fc.double({ min: -10, max: 10, noNaN: true }),
  im: fc.double({ min: -10, max: 10, noNaN: true }),
})
const c = (v: { re: number; im: number }): ReturnType<typeof complex> => complex(tensor(v.re), tensor(v.im))

describe('complex pairs', () => {
  it('agree with the arithmetic of complex numbers', () => {
    fc.assert(
      fc.property(z, z, (a, b) => {
        const p = cMul(c(a), c(b))
        expect(item(real(p))).toBeCloseTo(a.re * b.re - a.im * b.im, 10)
        expect(item(imag(p))).toBeCloseTo(a.re * b.im + a.im * b.re, 10)
        const s = cAdd(c(a), c(b))
        expect(item(real(s))).toBeCloseTo(a.re + b.re, 12)
        if (Math.hypot(b.re, b.im) > 1e-3) {
          const back = cMul(cDiv(c(a), c(b)), c(b))
          expect(item(real(back))).toBeCloseTo(a.re, 8)
          expect(item(imag(back))).toBeCloseTo(a.im, 8)
        }
        // z conj(z) = |z|^2.
        expect(item(real(cMul(c(a), conj(c(a)))))).toBeCloseTo(item(absComplex(c(a))) ** 2, 9)
      }),
    )
  })

  it('round-trip through polar form', () => {
    fc.assert(
      fc.property(z, (a) => {
        fc.pre(Math.hypot(a.re, a.im) > 1e-6)
        const back = polar(absComplex(c(a)), angle(c(a)))
        expect(item(real(back))).toBeCloseTo(a.re, 10)
        expect(item(imag(back))).toBeCloseTo(a.im, 10)
      }),
    )
    // Euler: exp(i pi) = -1.
    expect(item(real(cExp(complex(0, Math.PI))))).toBeCloseTo(-1, 15)
    expect(item(real(cis(Math.PI)))).toBeCloseTo(-1, 15)
  })

  it('are differentiable', () => {
    expect(
      checkPrimitive(
        (re, im) => absComplex({ re, im }),
        [
          [3, -1],
          [4, 2],
        ],
      ).failures,
    ).toEqual([])
    expect(
      checkPrimitive(
        (re, im) => angle({ re, im }),
        [
          [3, -1],
          [4, 2],
        ],
      ).failures,
    ).toEqual([])
    expect(
      checkPrimitive(
        (re, im) => real(cDiv({ re, im }, { re: im, im: re })),
        [
          [3, -1],
          [4, 2],
        ],
      ).failures,
    ).toEqual([])
  })
})
