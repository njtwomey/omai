import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { eigvals } from '../numerics/linalg/index.ts'
import { concat, item, reshape, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const a2: Tensor = tensor([
  [-1, 0.5],
  [-0.3, -2],
])
const b2: Tensor = tensor([[1], [0.4]])
const c2: Tensor = tensor([[1, -1]])
const d1: Tensor = tensor([[0.2]])

const flatAll = (s: S.StateSpace): Value =>
  concat([reshape(s.a, [-1]), reshape(s.b, [-1]), reshape(s.c, [-1]), reshape(s.d, [-1])], 0)

describe('cont2discrete', () => {
  it('maps poles as the method says: e^{sT} for zoh, the bilinear map for Tustin', () => {
    const s = S.ss(a2, b2, c2, d1)
    const cont = eigvals(a2)
    const zoh = eigvals(S.cont2discrete(s, 0.1).a)
    const tustin = eigvals(S.cont2discrete(s, 0.1, { method: 'tustin' }).a)
    const sortRe = (x: Tensor): number[] => [...toFlat(x)].sort((p, q) => p - q)
    const cs = sortRe(cont.re)
    for (const [i, z] of sortRe(zoh.re).entries()) expect(z).toBeCloseTo(Math.exp((cs[i] ?? 0) * 0.1), 12)
    for (const [i, z] of sortRe(tustin.re).entries()) {
      const p = cs[i] ?? 0
      expect(z).toBeCloseTo((1 + p * 0.05) / (1 - p * 0.05), 12)
    }
  })

  it('keep the dc gain for zoh, foh, tustin and matched', () => {
    const g = S.tf([1, 2], [1, 3, 2.5])
    const k = item(S.dcgain(g))
    for (const method of ['zoh', 'foh', 'tustin', 'matched'] as const) {
      expect(item(S.dcgain(S.cont2discrete(g, 0.2, { method })))).toBeCloseTo(k, 10)
    }
  })

  it('prewarping matches the response exactly at the prewarp frequency', () => {
    const g = S.tf([1], [1, 0.4, 4])
    const w0 = 2
    const gd = S.cont2discrete(g, 0.3, { method: 'tustin', prewarpFrequency: w0 })
    const hc = S.freqresp(g, [w0]).h
    const hd = S.freqresp(gd, [w0]).h
    expect(toFlat(hd.re as Tensor)[0]).toBeCloseTo(toFlat(hc.re as Tensor)[0] ?? 0, 10)
    expect(toFlat(hd.im as Tensor)[0]).toBeCloseTo(toFlat(hc.im as Tensor)[0] ?? 0, 10)
  })

  it('reject discrete inputs, bad steps, missing alpha, proper systems for impulse and integrators for matched', () => {
    const s = S.ss(a2, b2, c2, d1)
    expect(() => S.cont2discrete(S.cont2discrete(s, 0.1), 0.1)).toThrow()
    expect(() => S.cont2discrete(s, 0)).toThrow()
    expect(() => S.cont2discrete(s, 0.1, { method: 'gbt' })).toThrow()
    expect(() => S.cont2discrete(s, 0.1, { method: 'gbt', alpha: 2 })).toThrow()
    expect(() => S.cont2discrete(s, 0.1, { method: 'impulse' })).toThrow()
    expect(() => S.cont2discrete(S.tf([1], [1, 0]), 0.1, { method: 'matched' })).toThrow()
  })

  it('keeps the representation and the matrices form', () => {
    expect(S.cont2discrete(S.zpk([], [-1], 1), 0.1).kind).toBe('zpk')
    expect(S.cont2discrete(S.toSos(S.tf([1], [1, 1, 1])), 0.1).kind).toBe('sos')
    const m = S.c2dMatrices(
      [
        [0, 1],
        [0, 0],
      ],
      [[0], [1]],
      [[1, 0]],
      undefined,
      0.5,
    )
    expect(toFlat(m.a as Tensor)).toEqual([1, 0.5, 0, 1])
  })

  it('Tustin maps stable systems to Schur-stable ones and back-difference too (property)', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.integer({ min: -300, max: 300 }).map((k) => k / 100),
          { minLength: 4, maxLength: 4 },
        ),
        (v) => {
          const a = tensor([
            [(v[0] ?? 0) - 3.5, v[1] ?? 0],
            [-(v[1] ?? 0), (v[2] ?? 0) - 3.5],
          ])
          const s = S.ss(a, b2, c2)
          for (const method of ['tustin', 'backward_diff', 'zoh'] as const) {
            const { re, im } = eigvals(S.cont2discrete(s, 0.1 + Math.abs(v[3] ?? 0), { method }).a)
            for (const [i, r] of toFlat(re).entries()) expect(Math.hypot(r, toFlat(im)[i] ?? 0)).toBeLessThan(1)
          }
        },
      ),
      { numRuns: 30 },
    )
  })
})

describe('cont2discrete is differentiable', () => {
  for (const method of ['zoh', 'foh', 'tustin', 'euler', 'gbt'] as const) {
    it(method, () => {
      const f = (a: Value, b: Value, c: Value): Value =>
        flatAll(S.cont2discrete(S.ss(a, b, c, d1), 0.2, { method, ...(method === 'gbt' ? { alpha: 0.3 } : {}) }))
      expect(checkPrimitive(f, [a2, b2, c2], { vmap: false }).failures).toEqual([])
    })
  }
  it('impulse, and through a transfer function', () => {
    const f = (a: Value): Value => flatAll(S.cont2discrete(S.ss(a, b2, c2), 0.2, { method: 'impulse' }))
    expect(checkPrimitive(f, [a2], { vmap: false }).failures).toEqual([])
    const g = (den: Value): Value => S.cont2discrete(S.tf([1, 1], den), 0.1).den
    expect(checkPrimitive(g, [tensor([1, 2, 3])], { vmap: false }).failures).toEqual([])
  })
})
