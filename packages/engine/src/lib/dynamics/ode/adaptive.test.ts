import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { mul, neg, sliceAxis, stack, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  adaptiveRk,
  BOGACKI_SHAMPINE,
  CASH_KARP,
  DOP853_TABLEAU,
  DORMAND_PRINCE,
  type EmbeddedTableau,
  embeddedPair,
  FEHLBERG,
  HEUN_EULER,
  rk45,
  TSITOURAS,
} from './adaptive.ts'
import { orderConditions } from './convergence.ts'
import { HEUN } from './explicit.ts'
import { solveIvp } from './solve-ivp.ts'

const PAIRS: readonly EmbeddedTableau[] = [HEUN_EULER, BOGACKI_SHAMPINE, FEHLBERG, CASH_KARP, DORMAND_PRINCE, TSITOURAS]

// The embedded solution as a method of s + 1 stages, the last being f at the new point (first same as last).
function embeddedMethod(t: EmbeddedTableau): { a: number[][]; b: number[]; c: number[] } {
  const s = t.b.length
  const a = [...t.a.map((row) => [...row, ...new Array<number>(s + 1 - row.length).fill(0)]), [...t.b, 0]]
  const b = [...t.b.map((bi, i) => bi - (t.e[i] ?? 0)), -(t.e[s] ?? 0)]
  return { a, b, c: [...t.c, 1] }
}

describe('embedded Runge-Kutta pairs', () => {
  for (const t of [...PAIRS, DOP853_TABLEAU]) {
    it(`${t.name} propagates order ${t.order}`, () => {
      expect(orderConditions(t, t.order + 1, 1e-9).order).toBe(t.order)
    })
  }
  for (const t of PAIRS) {
    it(`${t.name} estimates with an embedded method of order ${t.errorOrder}`, () => {
      const embeddedOrder = orderConditions(embeddedMethod(t), t.errorOrder + 2, 1e-9).order
      expect(Math.min(embeddedOrder, t.order)).toBe(t.errorOrder)
      expect(Math.abs(t.e.reduce((x, y) => x + y, 0))).toBeLessThan(1e-14)
    })
  }

  it('dense output passes through both ends of every step', () => {
    for (const t of [BOGACKI_SHAMPINE, DORMAND_PRINCE, DOP853_TABLEAU, CASH_KARP]) {
      const s = adaptiveRk(t, { rtol: 1e-6 })
      let st = s.init((_t, y) => neg(y), 0, tensor([1, 2]), 1)
      st = s.step(st)
      const d = s.dense(st)
      expect(toFlat(d.at(st.tOld))).toEqual(toFlat(st.yOld))
      const end = toFlat(d.at(st.t))
      toFlat(st.y).forEach((v, i) => {
        expect(Math.abs((end[i] ?? 0) - v)).toBeLessThan(1e-12)
      })
    }
  })

  it('the error falls with the tolerance (property)', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.2, max: 3, noNaN: true }), (k) => {
        const f = (_t: number, y: Value): Value => mul(y, -k)
        const loose = solveIvp(f, [0, 2], [1], { method: 'RK45', rtol: 1e-4, atol: 1e-8, tEval: [2] })
        const tight = solveIvp(f, [0, 2], [1], { method: 'RK45', rtol: 1e-9, atol: 1e-13, tEval: [2] })
        const exact = Math.exp(-2 * k)
        const e1 = Math.abs((toFlat(loose.y)[0] ?? 0) - exact)
        const e2 = Math.abs((toFlat(tight.y)[0] ?? 0) - exact)
        return e2 <= e1 + 1e-15 && e2 < 1e-7
      }),
      { numRuns: 15 },
    )
  })

  it('the PI and PID controllers integrate with fewer rejections on an oscillatory problem', () => {
    const f = (_t: number, y: Value): Value => {
      const x = sum(sliceAxis(y, 0, 0, 1))
      const v = sum(sliceAxis(y, 0, 1, 2))
      return stack([v, mul(x, -400)])
    }
    const run = (controller: 'integral' | 'pi'): { rejected: number; y: number } => {
      const sol = solveIvp(f, [0, 3], [1, 0], {
        method: adaptiveRk(DORMAND_PRINCE, { rtol: 1e-6, atol: 1e-9, controller }),
      })
      return { rejected: sol.nRejected, y: toFlat(sol.ys.at(-1) as Value)[0] ?? 0 }
    }
    const i = run('integral')
    const p = run('pi')
    expect(Math.abs(p.y - Math.cos(60))).toBeLessThan(1e-3)
    expect(p.rejected).toBeLessThanOrEqual(i.rejected)
    const pid = solveIvp(f, [0, 3], [1, 0], {
      method: adaptiveRk(TSITOURAS, { rtol: 1e-6, atol: 1e-9, controller: { beta1: 0.4, beta2: 0.2, beta3: 0.05 } }),
    })
    expect(pid.success).toBe(true)
  })

  it('records rejected attempts and fails after too many', () => {
    const s = rk45({ firstStep: 1, rtol: 1e-10, atol: 1e-12 })
    const st = s.step(s.init((_t, y) => mul(y, -30), 0, tensor([1]), 2))
    expect(st.attempts.some((a) => !a.accepted)).toBe(true)
    const capped = rk45({ firstStep: 1, rtol: 1e-12, atol: 1e-14, maxAttempts: 1 })
    expect(capped.step(capped.init((_t, y) => mul(y, -30), 0, tensor([1]), 2)).status).toBe('failed')
  })

  it('embeddedPair builds the error weights', () => {
    expect(embeddedPair(HEUN, [1, 0], 1).e).toEqual([-0.5, 0.5, 0])
  })
})
