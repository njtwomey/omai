import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, stack, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const a2: Tensor = tensor([
  [-1, 0.5],
  [-0.3, -2],
])
const b22: Tensor = tensor([
  [1, 0],
  [0.4, 1],
])
const c12: Tensor = tensor([[1, -1]])
const w = [0.3, 1, 4]

describe('frequency responses', () => {
  it('agree across representations, continuous and discrete', () => {
    const g = S.tf([1, 0.2, 3], [1, 2, 3, 1])
    const gd = S.cont2discrete(g, 0.1)
    for (const base of [g, gd]) {
      const want = S.freqresp(base, w).h
      for (const sys of [S.toSs(base), S.toZpk(base)]) {
        const h = S.freqresp(sys, w).h
        for (const [i, v] of toFlat(h.re as Tensor).entries())
          expect(v).toBeCloseTo(toFlat(want.re as Tensor)[i] ?? 0, 10)
        for (const [i, v] of toFlat(h.im as Tensor).entries())
          expect(v).toBeCloseTo(toFlat(want.im as Tensor)[i] ?? 0, 10)
      }
    }
  })

  it('give multivariable responses with the frequency first', () => {
    const h = S.freqresp(
      S.ss(
        a2,
        b22,
        tensor([
          [1, 0],
          [0.5, 2],
          [0, 1],
        ]),
      ),
      w,
    ).h
    expect(h.re.shape).toEqual([3, 3, 2])
    expect(S.evalfr(S.ss(a2, b22, c12), { re: 0, im: 1 }).re.shape).toEqual([1, 2])
  })

  it('handle integrators: infinite dc gain, a pole at the origin indented for nyquist', () => {
    expect(item(S.dcgain(S.tf([1], [1, 0])))).toBe(Number.POSITIVE_INFINITY)
    expect(item(S.dcgain(S.ss([[0]], [[1]], [[1]])) as Tensor)).toBe(Number.POSITIVE_INFINITY)
    expect(S.nyquist(S.tf([1], [1, 1, 0])).count).toBe(0)
    expect(S.nyquist(S.tf([5], [1, 2, 2, 0])).count).toBe(2)
  })

  it('bode and nichols give the same data; the phase is unwrapped past -180', () => {
    const g = S.tf([1], [1, 1, 1, 0.2])
    const b = S.bode(g, [0.1, 1, 10, 100])
    expect(toFlat(S.nichols(g, [0.1, 1, 10, 100]).phase as Tensor)).toEqual(toFlat(b.phase as Tensor))
    const phase = toFlat(b.phase as Tensor)
    expect(phase[3]).toBeLessThan(-260)
  })

  it('reject multivariable systems where a scalar response is needed', () => {
    expect(() => S.bode(S.ss(a2, b22, c12), w)).toThrow()
  })
})

describe('frequency responses are differentiable', () => {
  const parts = (h: { re: Value; im: Value }): Value => stack([h.re, h.im], 0)
  it('in transfer-function coefficients and state-space matrices', () => {
    const tfOf = (num: Value, den: Value): Value => parts(S.freqresp(S.tf(num, den), w).h)
    expect(checkPrimitive(tfOf, [tensor([1, 0.5]), tensor([1, 2, 3])], { vmap: false }).failures).toEqual([])
    const ssOf = (a: Value, b: Value): Value => parts(S.freqresp(S.ss(a, b, c12), w).h)
    expect(checkPrimitive(ssOf, [a2, b22], { vmap: false }).failures).toEqual([])
    const disc = (a: Value): Value => parts(S.freqresp(S.ss(a, b22, c12, undefined, { dt: 0.5 }), w).h)
    expect(
      checkPrimitive(
        disc,
        [
          tensor([
            [0.5, 0.1],
            [-0.2, 0.3],
          ]),
        ],
        { vmap: false },
      ).failures,
    ).toEqual([])
  })

  it('through bode magnitude and phase and freqz', () => {
    const bodeOf = (den: Value): Value => {
      const r = S.bode(S.tf([2, 1], den), w)
      return stack([r.mag, r.phase], 0)
    }
    expect(checkPrimitive(bodeOf, [tensor([1, 0.8, 2])], { vmap: false }).failures).toEqual([])
    const z = (b: Value, a: Value): Value => parts(S.freqz(b, a, 8).h)
    expect(checkPrimitive(z, [tensor([0.2, 0.3, 0.2]), tensor([1, -0.5])], { vmap: false }).failures).toEqual([])
    const fir = (b: Value): Value => parts(S.freqz(b, [1], 8).h)
    expect(checkPrimitive(fir, [tensor([0.2, 0.3, 0.2])], { vmap: false }).failures).toEqual([])
  })
})
