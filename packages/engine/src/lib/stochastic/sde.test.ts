import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { SDEBlock } from '../nn/blocks/index.ts'
import { Linear } from '../nn/index.ts'
import { key, normal } from '../random/index.ts'
import {
  add,
  div,
  exp,
  expandDims,
  mul,
  neg,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import { brownianIncrements } from './brownian.ts'
import { timeGrid, valuesAt } from './paths.ts'
import {
  convertCalculus,
  probabilityFlow,
  reverseSde,
  type Sde,
  sdeint,
  sdeSolver,
  sdeStep,
  strongConvergence,
  weakConvergence,
} from './sde.ts'

const MU = 0.5
const SIGMA = 0.8
const gbm: Sde = { drift: (_t, y) => mul(y, MU), diffusion: (_t, y) => mul(y, SIGMA) }
const gbmExact = (dw: Value): Value => mul(exp(mul(sum(dw, 1), SIGMA)), Math.exp(MU - SIGMA ** 2 / 2))

describe('sde schemes given the Brownian increments', () => {
  const t = timeGrid(1, 4)
  const dw = brownianIncrements(key(1), t, { paths: 1, shape: [1] })
  const w = toFlat(dw)
  const h = 0.25

  it('Euler-Maruyama and Milstein follow their recurrences on geometric Brownian motion', () => {
    const euler = toFlat(sdeint(key(0), gbm, [[1]], t, { increments: dw }).values)
    const milstein = toFlat(sdeint(key(0), gbm, [[1]], t, { increments: dw, method: 'milstein' }).values)
    let e = 1
    let m = 1
    for (let i = 0; i < 4; i++) {
      const d = w[i] ?? 0
      e *= 1 + MU * h + SIGMA * d
      m *= 1 + MU * h + SIGMA * d + 0.5 * SIGMA ** 2 * (d * d - h)
      expect(euler[i + 1]).toBeCloseTo(e, 13)
      expect(milstein[i + 1]).toBeCloseTo(m, 13)
    }
  })

  it("Platen's SRK follows its derivative-free recurrence", () => {
    const srk = toFlat(sdeint(key(0), gbm, [[1]], t, { increments: dw, method: 'srk' }).values)
    let y = 1
    for (let i = 0; i < 4; i++) {
      const d = w[i] ?? 0
      const support = y + MU * y * h + SIGMA * y * Math.sqrt(h)
      y += MU * y * h + SIGMA * y * d + ((SIGMA * support - SIGMA * y) * (d * d - h)) / (2 * Math.sqrt(h))
      expect(srk[i + 1]).toBeCloseTo(y, 13)
    }
  })

  it('Heun follows its recurrence on a Stratonovich equation', () => {
    const strat: Sde = { ...gbm, calculus: 'stratonovich' }
    const heun = toFlat(sdeint(key(0), strat, [[1]], t, { increments: dw, method: 'heun' }).values)
    let y = 1
    for (let i = 0; i < 4; i++) {
      const d = w[i] ?? 0
      const p = y + MU * y * h + SIGMA * y * d
      y += 0.5 * MU * (y + p) * h + 0.5 * SIGMA * (y + p) * d
      expect(heun[i + 1]).toBeCloseTo(y, 13)
    }
  })

  it('substeps consume the increments of the finer grid', () => {
    const fine = brownianIncrements(key(2), timeGrid(1, 8), { paths: 1, shape: [1] })
    const a = sdeint(key(0), gbm, [[1]], timeGrid(1, 4), { increments: fine, substeps: 2 })
    const b = sdeint(key(0), gbm, [[1]], timeGrid(1, 8), { increments: fine })
    expect(toFlat(valuesAt(a, -1))[0]).toBeCloseTo(toFlat(valuesAt(b, -1))[0] ?? 0, 13)
  })

  it('scalar and general noise agree when they describe the same equation', () => {
    const inc = brownianIncrements(key(3), t, { paths: 2 })
    const scalar: Sde = { drift: (_t, y) => neg(y), diffusion: (_t, y) => mul(y, 0.3), noise: 'scalar' }
    const general: Sde = {
      drift: (_t, y) => neg(y),
      diffusion: (_t, y) => expandDims(mul(y, 0.3), -1),
      noise: 'general',
      noiseDim: 1,
    }
    const y0 = [
      [1, 2],
      [3, 4],
    ]
    const a = sdeint(key(0), scalar, y0, t, { increments: inc })
    const b = sdeint(key(0), general, y0, t, { increments: expandDims(inc, -1) })
    expect(a.values.shape).toEqual([2, 5, 2])
    expect(toFlat(a.values)).toEqual(toFlat(b.values))
    // one Brownian motion drives both coordinates: their ratio is constant under linear coefficients
    const end = toArray(valuesAt(a, -1)) as number[][]
    expect((end[0]?.[1] ?? 0) / (end[0]?.[0] ?? 1)).toBeCloseTo(2, 12)
  })

  it('rejects schemes that do not fit the equation', () => {
    expect(() => sdeint(key(0), gbm, [[1]], t, { method: 'heun' })).toThrow()
    expect(() => sdeint(key(0), { ...gbm, calculus: 'stratonovich' }, [[1]], t, { method: 'euler' })).toThrow()
    expect(() => sdeint(key(0), { ...gbm, noise: 'general', noiseDim: 1 }, [[1]], t, { method: 'milstein' })).toThrow()
    expect(() => sdeint(key(0), gbm, [[1]], t, { increments: zeros([1, 3, 1]) })).toThrow()
    expect(() => sdeint(key(0), gbm, 1, t)).toThrow()
  })

  it('takes one documented step', () => {
    expect(toArray(sdeStep(gbm, 'euler', 0, tensor([1]), 0.1, tensor([0.5])))).toEqual([1 + 0.05 + 0.4])
  })
})

describe('convergence studies', () => {
  it('measure strong order 1/2 for Euler-Maruyama and 1 for Milstein and SRK', () => {
    const steps = [8, 16, 32, 64]
    const em = strongConvergence(key(4), gbm, [1], 1, gbmExact, { steps, paths: 300 })
    const mil = strongConvergence(key(4), gbm, [1], 1, gbmExact, { steps, paths: 300, method: 'milstein' })
    const srk = strongConvergence(key(4), gbm, [1], 1, gbmExact, { steps, paths: 300, method: 'srk' })
    expect(em.order).toBeGreaterThan(0.35)
    expect(em.order).toBeLessThan(0.7)
    expect(mil.order).toBeGreaterThan(0.85)
    expect(mil.order).toBeLessThan(1.2)
    expect(srk.order).toBeGreaterThan(0.85)
    expect((mil.errors[3]?.value ?? 1) < (em.errors[3]?.value ?? 0)).toBe(true)
  })

  it('measures weak order 1 for Euler-Maruyama', () => {
    const drifty: Sde = { drift: (_t, y) => y, diffusion: (_t, y) => mul(y, 0.1) }
    const w = weakConvergence(key(5), drifty, [1], 1, (y) => sum(y, 1), Math.E, {
      steps: [4, 8, 16, 32],
      paths: 4000,
    })
    expect(w.order).toBeGreaterThan(0.75)
    expect(w.order).toBeLessThan(1.25)
  })

  it('validates its step counts', () => {
    expect(() => strongConvergence(key(0), gbm, [1], 1, gbmExact, { steps: [8] })).toThrow()
    expect(() => strongConvergence(key(0), gbm, [1], 1, gbmExact, { steps: [3, 8] })).toThrow()
  })
})

describe('pathwise derivatives and tracing', () => {
  it('differentiates solutions in the initial state and coefficients', () => {
    const t = timeGrid(1, 6)
    for (const method of ['euler', 'milstein', 'srk'] as const) {
      const f = (y0: Value, s: Value): Value =>
        valuesAt(
          sdeint(key(6), { drift: (_t, y) => neg(y), diffusion: (_t, y) => mul(sqrt(add(square(y), 1)), s) }, y0, t, {
            method,
          }),
          -1,
        )
      expectConformant(f, [tensor([[0.5], [-1.2]]), tensor(0.4)], { rtol: 1e-4 })
    }
  })

  it('records one step per grid interval', () => {
    const rec = recorder()
    sdeint(key(7), gbm, [[1]], timeGrid(1, 5), { tracer: rec })
    expect(rec.run().events.filter((e) => e.type === 'metric' && e.name === 't').length).toBe(6)
  })
})

describe('the nn SdeSolver contract', () => {
  it('drives an SDEBlock', () => {
    const block = new SDEBlock(new Linear(2, 2, { key: key(0) }), new Linear(2, 2, { key: key(1) }), {
      solver: sdeSolver('milstein', { substeps: 4 }),
      times: [0, 0.5, 1],
      trajectory: true,
    })
    expect(block.forward([[1, 2]], { key: key(2) }).shape).toEqual([3, 1, 2])
    const solve = sdeSolver('euler')
    expect(
      solve(
        (_t, y) => neg(y),
        (_t, y) => mul(y, 0),
        tensor(2),
        [0, 1],
        key(0),
      )[1]?.shape,
    ).toEqual([])
  })
})

describe('reverse-time SDEs', () => {
  // Ornstein-Uhlenbeck dX = -X dt + sqrt(2) dW from N(m0, s0^2): p_t = N(m0 e^-t, s0^2 e^-2t + 1 - e^-2t).
  const m0 = 2
  const s0 = 0.5
  const ou: Sde = { drift: (_t, y) => neg(y), diffusion: (_t, y) => add(zeros(y.shape), Math.SQRT2) }
  const moments = (t: number): [number, number] => [
    m0 * Math.exp(-t),
    s0 ** 2 * Math.exp(-2 * t) + 1 - Math.exp(-2 * t),
  ]
  const score = (t: number, y: Value): Value => {
    const [m, v] = moments(t)
    return div(sub(m, y), v)
  }

  it('maps samples of p_T back to p_0', () => {
    const [mT, vT] = moments(1)
    const start = normal(key(8), [4000, 1], { loc: mT, scale: Math.sqrt(vT) })
    const back = sdeint(key(9), reverseSde(ou, score, 1), start, timeGrid(1, 200))
    const x = toFlat(valuesAt(back, -1))
    expect(Math.abs(meanOf(x) - m0)).toBeLessThan(4 * Math.sqrt(s0 ** 2 / 4000) + 0.03)
    expect(Math.abs(varianceOf(x) - s0 ** 2)).toBeLessThan(0.05)
  })

  it('transports samples by the probability-flow ODE', () => {
    const [mT, vT] = moments(1)
    const start = normal(key(10), [2000, 1], { loc: mT, scale: Math.sqrt(vT) })
    const flow = probabilityFlow(ou, score)
    const ode: Sde = { drift: (s, y) => neg(flow(1 - s, y)), diffusion: (_t, y) => zeros(y.shape) }
    const x = toFlat(valuesAt(sdeint(key(0), ode, start, timeGrid(1, 400)), -1))
    expect(Math.abs(meanOf(x) - m0)).toBeLessThan(0.05)
    expect(Math.abs(varianceOf(x) - s0 ** 2)).toBeLessThan(0.05)
  })

  it('converts between Ito and Stratonovich forms', () => {
    const strat = convertCalculus(gbm)
    expect(strat.calculus).toBe('stratonovich')
    expect(toArray(strat.drift(0, tensor([2])))).toEqual([2 * (MU - SIGMA ** 2 / 2)])
    expect(toArray(convertCalculus(strat).drift(0, tensor([2])))).toEqual([2 * MU])
    const dw = brownianIncrements(key(11), timeGrid(1, 400), { paths: 50, shape: [1] })
    const heun = sdeint(key(0), strat, add(zeros([50, 1]), 1), timeGrid(1, 400), { increments: dw })
    const err = toFlat(sub(valuesAt(heun, -1), gbmExact(dw))).map(Math.abs)
    expect(meanOf(err)).toBeLessThan(0.02)
  })
})
