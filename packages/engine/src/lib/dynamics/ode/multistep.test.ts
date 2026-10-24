import { describe, expect, it } from 'vitest'
import { mul, neg, sliceAxis, stack, sum, toFlat, type Value } from '../../tensor/index.ts'
import { boundaryLocus, stepperOrder } from './convergence.ts'
import {
  ADAMS_BASHFORTH,
  ADAMS_ERROR_COEFFICIENTS,
  ADAMS_MOULTON,
  adams,
  adamsBashforth,
  adamsMoulton,
  nordsieckAdamsL,
  pece,
} from './multistep.ts'
import { solveIvp } from './solve-ivp.ts'

const decay = (_t: number, y: Value): Value => neg(y)

describe('linear multistep methods', () => {
  for (const order of [1, 2, 3, 4, 5] as const) {
    it(`Adams-Bashforth, Adams-Moulton and PECE of order ${order} converge at that order`, () => {
      for (const make of [adamsBashforth, adamsMoulton, pece]) {
        const study = stepperOrder((dt) => make({ dt, order }), decay, [0, 2], [1], [Math.exp(-2)], [0.04, 0.02, 0.01])
        expect(Math.abs(study.order - order)).toBeLessThan(0.35)
      }
    })
  }

  it('weights sum to one (consistency)', () => {
    for (const w of [...ADAMS_BASHFORTH, ...ADAMS_MOULTON])
      expect(Math.abs(w.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-14)
  })

  it('Nordsieck vectors and error coefficients', () => {
    expect(nordsieckAdamsL(1)).toEqual([1, 1])
    const l3 = nordsieckAdamsL(3)
    expect(l3[0]).toBeCloseTo(5 / 12, 14)
    expect(l3[1]).toBeCloseTo(1, 14)
    ;[1, -0.5, -1 / 12, -1 / 24, -19 / 720].forEach((g, i) => {
      expect(ADAMS_ERROR_COEFFICIENTS[i]).toBeCloseTo(g, 14)
    })
    expect(() => nordsieckAdamsL(13)).toThrow()
  })

  it('variable-order Adams is accurate and raises its order', () => {
    const f = (_t: number, y: Value): Value => {
      const x = sum(sliceAxis(y, 0, 0, 1))
      const v = sum(sliceAxis(y, 0, 1, 2))
      return stack([v, neg(x)])
    }
    const s = adams({ rtol: 1e-10, atol: 1e-12 })
    let st = s.init(f, 0, toTensor([1, 0]), 10)
    let maxOrder = 1
    while (st.status === 'running') {
      st = s.step(st)
      maxOrder = Math.max(maxOrder, st.order)
    }
    expect(maxOrder).toBeGreaterThan(4)
    expect(Math.abs((toFlat(st.y)[0] ?? 0) - Math.cos(10))).toBeLessThan(1e-7)
  })

  it('the boundary locus of AB2 crosses the real axis at -1', () => {
    // y_{n+2} - y_{n+1} = h (3/2 f_{n+1} - 1/2 f_n)
    const locus = boundaryLocus([0, -1, 1], [-0.5, 1.5, 0], 2)
    expect(locus.re[1]).toBeCloseTo(-1, 12)
  })

  it('a growing problem still terminates with a status', () => {
    const sol = solveIvp((_t, y) => mul(y, y), [0, 2], [1], { method: 'Adams', maxSteps: 3000 })
    expect(sol.success).toBe(false)
  })
})

function toTensor(x: number[]): Value {
  return stack(x.map((v) => sum(mul(stack([v]), 1))))
}
