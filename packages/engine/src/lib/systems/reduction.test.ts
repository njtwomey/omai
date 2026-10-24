import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, sub, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const a4: Tensor = tensor([
  [-1, 0.4, 0, 0.1],
  [0, -2, 0.5, 0],
  [0.2, 0, -5, 1],
  [0, 0.1, 0, -9],
])
const b4: Tensor = tensor([[1], [0.5], [1], [0.2]])
const c4: Tensor = tensor([[1, -1, 0.5, 0.3]])
const sys = S.ss(a4, b4, c4, [[0.1]])

describe('balanced realisation and truncation', () => {
  it('balance both Gramians to diag(hsv)', () => {
    const { sys: balanced, hsv } = S.balancedRealization(sys)
    const h = toFlat(hsv as Tensor)
    for (const kind of ['c', 'o'] as const) {
      const w = toFlat(S.gram(balanced, kind) as Tensor)
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) expect(w[4 * i + j]).toBeCloseTo(i === j ? (h[i] ?? 0) : 0, 9)
      }
    }
    expect(h).toEqual([...h].sort((x, y) => y - x))
  })

  it('respect the error bound: |G - Gr|_inf <= 2 (sum of the discarded values)', () => {
    const h = toFlat(S.hsvd(sys) as Tensor)
    for (const order of [1, 2, 3]) {
      const reduced = S.balred(sys, order)
      expect(S.dimensions(reduced).states).toBe(order)
      const error = S.parallel(sys, S.series(reduced, -1))
      const bound = 2 * h.slice(order).reduce((s, v) => s + v, 0)
      expect(item(S.hinfNorm(error).norm)).toBeLessThanOrEqual(bound * (1 + 1e-8))
      // matchdc keeps the dc gain exactly.
      const dc = S.balred(sys, order, { method: 'matchdc' })
      expect(item(S.dcgain(dc))).toBeCloseTo(item(S.dcgain(sys)), 10)
    }
  })

  it('keep unstable parts and work in discrete time', () => {
    const unstable = S.ss(
      [
        [0.5, 0, 0],
        [0, -1, 0],
        [0, 0, -20],
      ],
      [[1], [1], [1]],
      [[1, 1, 1]],
    )
    const r = S.balred(unstable, 2)
    expect(S.poles(r).re.some((p) => Math.abs(p - 0.5) < 1e-9)).toBe(true)
    expect(() => S.balred(unstable, 0)).toThrow()
    const d = S.cont2discrete(sys, 0.1)
    const rd = S.balred(d, 2)
    expect(S.isDiscrete(rd)).toBe(true)
    expect(item(S.dcgain(S.balred(d, 2, { method: 'matchdc' })))).toBeCloseTo(item(S.dcgain(d)), 9)
    expect(() =>
      S.balancedRealization(
        S.ss(
          [
            [-1, 0],
            [0, -2],
          ],
          [[1], [0]],
          [[1, 1]],
        ),
      ),
    ).toThrow()
  })

  it('Hankel singular values are differentiable', () => {
    const f = (b: Value): Value => S.hsvd(S.ss(a4, b, c4))
    expect(checkPrimitive(f, [b4], { vmap: false, rtol: 1e-4 }).failures).toEqual([])
    expect(item(sum(sub(S.hsvd(sys), S.hsvd(S.balancedRealization(sys).sys))))).toBeCloseTo(0, 9)
  })
})
