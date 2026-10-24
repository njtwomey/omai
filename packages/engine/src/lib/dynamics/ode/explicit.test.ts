import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../../base/index.ts'
import { neg, tensor, toFlat } from '../../tensor/index.ts'
import { orderConditions, stepperOrder } from './convergence.ts'
import { butcherTableau, EULER, explicitRk, HEUN, KUTTA3, MIDPOINT, RALSTON, RK4, RK38, rkStages } from './explicit.ts'

const TABLEAUS = [EULER, HEUN, MIDPOINT, RALSTON, KUTTA3, RK4, RK38]

describe('explicit Runge-Kutta', () => {
  for (const t of TABLEAUS) {
    it(`${t.name} satisfies the order conditions of order ${t.order} exactly`, () => {
      expect(orderConditions(t, 6).order).toBe(t.order)
    })
    it(`${t.name} converges at order ${t.order}`, () => {
      const study = stepperOrder(
        (dt) => explicitRk(t, { dt }),
        (_t, y) => neg(y),
        [0, 1],
        [1],
        [Math.exp(-1)],
        [0.1, 0.05, 0.025],
      )
      expect(Math.abs(study.order - t.order)).toBeLessThan(0.25)
    })
  }

  it('rejects implicit or inconsistent tableaus', () => {
    expect(() => butcherTableau('bad', [[0.5]], [1], [0.5], 1)).toThrow()
    try {
      butcherTableau('bad', [[], [0.3]], [0.5, 0.5], [0, 1], 2)
    } catch (e) {
      expect(isWhiteboxError(e, 'value')).toBe(true)
    }
  })

  it('reuses the first stage when it is given', () => {
    let calls = 0
    rkStages(
      (_t, y) => {
        calls += 1
        return y
      },
      0,
      tensor([1]),
      0.1,
      RK4,
      tensor([1]),
    )
    expect(calls).toBe(3)
  })

  it('lands exactly on the bound and fails on non-finite states', () => {
    const s = explicitRk(EULER, { dt: 0.3 })
    let st = s.init((_t, y) => y, 0, tensor([1]), 1)
    while (st.status === 'running') st = s.step(st)
    expect(st.t).toBe(1)
    const blow = explicitRk(EULER, { dt: 1 })
    let b = blow.init((_t, y) => tensor([Number.POSITIVE_INFINITY * (toFlat(y)[0] ?? 0)]), 0, tensor([1]), 3)
    b = blow.step(b)
    expect(b.status).toBe('failed')
    expect(() => explicitRk(EULER, { dt: 0 })).toThrow()
  })
})
