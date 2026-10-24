import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const a2: Tensor = tensor([
  [-0.3, 1],
  [-1, -0.3],
])
const b2: Tensor = tensor([[0], [1]])
const c2: Tensor = tensor([[1, 0.5]])

describe('system norms', () => {
  it('H2 by Parseval: the energy of the impulse response', () => {
    const s = S.ss(a2, b2, c2)
    const t = Array.from({ length: 4001 }, (_, k) => 0.01 * k)
    const y = toFlat(S.impulse(s, { t }).y as Tensor)
    // Trapezoid rule on y^2.
    const energy = y.reduce((acc, v, i) => acc + (i === 0 || i === y.length - 1 ? 0.5 : 1) * v * v, 0) * 0.01
    expect(item(S.h2Norm(s))).toBeCloseTo(Math.sqrt(energy), 4)
    expect(item(S.h2Norm(S.ss(a2, b2, c2, [[1]])))).toBe(Number.POSITIVE_INFINITY)
    expect(item(S.h2Norm(S.ss([[1]], [[1]], [[1]])))).toBe(Number.POSITIVE_INFINITY)
  })

  it('H-infinity is the peak of a dense frequency sweep, continuous and discrete', () => {
    for (const s of [S.ss(a2, b2, c2), S.cont2discrete(S.ss(a2, b2, c2, [[0.2]]), 0.3)]) {
      const { norm, frequency } = S.hinfNorm(s)
      const top = S.isDiscrete(s) ? Math.PI / 0.3 : 10
      const w = Array.from({ length: 2000 }, (_, k) => (top * (k + 0.5)) / 2000)
      const h = S.freqresp(s, w).h
      const re = toFlat(h.re as Tensor)
      const im = toFlat(h.im as Tensor)
      const sweep = re.reduce((acc, r, i) => Math.max(acc, Math.hypot(r, im[i] ?? 0)), 0)
      expect(item(norm)).toBeGreaterThanOrEqual(sweep * (1 - 1e-12))
      expect(item(norm)).toBeLessThan(sweep * 1.001)
      expect(frequency).toBeGreaterThan(0)
    }
    expect(item(S.systemNorm(S.tf([2], [1, 1]), 2))).toBeCloseTo(Math.sqrt(2), 12)
    expect(item(S.hinfNorm(S.tf([1], [1, -1])).norm)).toBe(Number.POSITIVE_INFINITY)
  })

  it('are differentiable in the matrices', () => {
    const h2 = (a: Value, c: Value): Value => S.h2Norm(S.ss(a, b2, c))
    expect(checkPrimitive(h2, [a2, c2], { vmap: false }).failures).toEqual([])
    const hinf = (c: Value): Value => S.hinfNorm(S.ss(a2, b2, c)).norm
    expect(checkPrimitive(hinf, [c2], { vmap: false, rtol: 1e-4 }).failures).toEqual([])
  })
})
