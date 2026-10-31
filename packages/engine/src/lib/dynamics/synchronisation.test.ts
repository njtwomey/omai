import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import { flowAlgorithm } from './ode/flow.ts'
import {
  consensusField,
  kuramotoAlgorithm,
  kuramotoCriticalCoupling,
  kuramotoField,
  kuramotoFrequencies,
  kuramotoTheoreticalOrder,
  orderParameter,
} from './synchronisation.ts'

describe('the Kuramoto field', () => {
  it('pulls each oscillator towards the others by K / n times the sum of the sines', () => {
    const f = kuramotoField({ omega: [0.5, -0.2, 0.1], K: 1.5 })
    const theta = [0.3, 1.1, -0.8]
    const out = toFlat(f(0, tensor(theta)))
    for (let i = 0; i < 3; i++) {
      const pull = theta.reduce((a, tj) => a + Math.sin(tj - (theta[i] as number)), 0)
      expect(out[i]).toBeCloseTo(([0.5, -0.2, 0.1][i] as number) + (1.5 / 3) * pull, 14)
    }
  })

  it('takes the complete graph with the mean-field scaling to the all-to-all field, and respects a missing edge', () => {
    const n = 4
    const complete = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 0 : 1)))
    const omega = [0.1, 0.2, -0.3, 0.4]
    const theta = tensor([0.2, 0.9, -1.4, 2.0])
    const a = toFlat(kuramotoField({ omega, K: 2 })(0, theta))
    const b = toFlat(kuramotoField({ omega, K: 2, adjacency: complete, normalisation: 'n' })(0, theta))
    for (let i = 0; i < n; i++) expect(a[i]).toBeCloseTo(b[i] as number, 14)
    // Remove every edge of oscillator 0: it turns at its natural frequency.
    const isolated = complete.map((row, i) => row.map((v, j) => (i === 0 || j === 0 ? 0 : v)))
    expect(toFlat(kuramotoField({ omega, K: 2, adjacency: isolated })(0, theta))[0]).toBeCloseTo(0.1, 14)
    expect(() =>
      kuramotoField({
        omega,
        K: 1,
        adjacency: [
          [0, 1],
          [1, 0],
        ],
      }),
    ).toThrow()
  })

  it('locks two oscillators when the frequency gap is below K, at the phase lag asin(gap / K)', () => {
    const gap = 0.6
    const K = 1
    const alg = flowAlgorithm(kuramotoField({ omega: [gap / 2, -gap / 2], K }), [0, 1], { dt: 0.02, steps: 3000 })
    const y = runAlgorithm(alg, { maxSteps: 3000 }).state.y
    // theta_1' - theta_2' = gap - K sin(theta_1 - theta_2) with the mean-field scaling K / 2 on each: gap = K sin(delta)
    expect(Math.sin((y[0] as number) - (y[1] as number))).toBeCloseTo(gap / K, 6)
    // Above the lock threshold they drift.
    const drift = flowAlgorithm(kuramotoField({ omega: [0.6, -0.6], K }), [0, 1], { dt: 0.02, steps: 3000 })
    const z = runAlgorithm(drift, { maxSteps: 3000 }).state.y
    expect(Math.abs((z[0] as number) - (z[1] as number))).toBeGreaterThan(10)
  })
})

describe('the order parameter and the populations', () => {
  it('is 1 for equal phases, 0 for evenly spread phases, and the mean phase otherwise', () => {
    expect(orderParameter([0.7, 0.7, 0.7]).r).toBeCloseTo(1, 14)
    expect(orderParameter([0.7, 0.7, 0.7]).psi).toBeCloseTo(0.7, 14)
    expect(orderParameter([0, (2 * Math.PI) / 3, (4 * Math.PI) / 3]).r).toBeLessThan(1e-15)
  })

  it('builds symmetric ordered quantile populations, or draws from a key', () => {
    const w = kuramotoFrequencies(101, 'gaussian', 2)
    expect(Math.abs(w.reduce((a, b) => a + b, 0))).toBeLessThan(1e-9)
    expect(w).toEqual([...w].sort((a, b) => a - b))
    expect(w[50]).toBeCloseTo(0, 12)
    const draws = kuramotoFrequencies(5000, 'gaussian', 2, key(3))
    const sd = Math.sqrt(draws.reduce((a, b) => a + b * b, 0) / draws.length)
    expect(sd).toBeCloseTo(2, 0)
    expect(kuramotoFrequencies(4, 'lorentzian', 1, key(1))).toEqual(kuramotoFrequencies(4, 'lorentzian', 1, key(1)))
    expect(() => kuramotoFrequencies(0, 'gaussian', 1)).toThrow()
  })

  it('has the critical coupling 2 gamma (Lorentzian) and sigma sqrt(8 / pi) (Gaussian)', () => {
    expect(kuramotoCriticalCoupling('lorentzian', 0.7)).toBeCloseTo(1.4, 14)
    expect(kuramotoCriticalCoupling('gaussian', 2)).toBeCloseTo(2 * Math.sqrt(8 / Math.PI), 12)
  })

  it('has the stationary order parameter zero below K_c, sqrt(1 - K_c / K) for the Lorentzian, and a root of the self-consistency condition for the Gaussian', () => {
    expect(kuramotoTheoreticalOrder(1.9, 'lorentzian', 1)).toBe(0)
    expect(kuramotoTheoreticalOrder(8, 'lorentzian', 1)).toBeCloseTo(Math.sqrt(0.75), 14)
    const Kc = kuramotoCriticalCoupling('gaussian', 1)
    expect(kuramotoTheoreticalOrder(0.9 * Kc, 'gaussian', 1)).toBe(0)
    const r = kuramotoTheoreticalOrder(2 * Kc, 'gaussian', 1)
    expect(r).toBeGreaterThan(0.5)
    expect(r).toBeLessThan(1)
    // Just above K_c the order parameter rises like the square root of the excess (the classic 16 / (pi K_c^3 g''...) law).
    const small = kuramotoTheoreticalOrder(1.01 * Kc, 'gaussian', 1)
    const smaller = kuramotoTheoreticalOrder(1.0025 * Kc, 'gaussian', 1)
    expect(small / smaller).toBeCloseTo(2, 0)
  })
})

describe('the Kuramoto simulation', () => {
  const n = 400
  const omega = kuramotoFrequencies(n, 'gaussian', 1)
  const start = Array.from({ length: n }, (_, i) => ((i * 2.399963229728653) % (2 * Math.PI)) - Math.PI)
  const orderAt = (K: number, steps: number): { order: number[]; locked: number[] } => {
    const rec = recorder()
    runAlgorithm(kuramotoAlgorithm({ omega, K }, start, { dt: 0.05, steps }), { maxSteps: steps, tracer: rec })
    return {
      order: metricSeries(rec.run(), 'oscillators/order').values,
      locked: metricSeries(rec.run(), 'oscillators/locked').values,
    }
  }

  it('stays incoherent below the critical coupling and synchronises above it, at the theoretical order parameter', () => {
    const Kc = kuramotoCriticalCoupling('gaussian', 1)
    const below = orderAt(0.5 * Kc, 1600).order
    expect(Math.max(...below.slice(-200))).toBeLessThan(0.15)
    const K = 2 * Kc
    const above = orderAt(K, 1600)
    const r = above.order.slice(-100).reduce((a, b) => a + b, 0) / 100
    expect(Math.abs(r - kuramotoTheoreticalOrder(K, 'gaussian', 1))).toBeLessThan(0.03)
    // The locked share is the share of frequencies within K r of the centre: P(|w| < K r) for the Gaussian.
    const share = above.locked.at(-1) as number
    const expected = omega.filter((w) => Math.abs(w) < K * r).length / n
    expect(Math.abs(share - expected)).toBeLessThan(0.05)
    expect(share).toBeGreaterThan(0.5)
  })

  it('logs every path of its contract and resumes from a serialised state', () => {
    const alg = kuramotoAlgorithm({ omega: [0.3, -0.1, 0.2], K: 1 }, [0, 1, 2], { dt: 0.05, steps: 20 })
    const rec = recorder()
    const straight = runAlgorithm(alg, { maxSteps: 50, tracer: rec }).state
    for (const name of ['oscillators/t', 'oscillators/order', 'oscillators/meanPhase', 'oscillators/locked']) {
      expect(metricSeries(rec.run(), name).values.length).toBe(20)
    }
    const half = runAlgorithm(alg, { maxSteps: 7 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.y)).toEqual(Array.from(straight.y))
  })
})

describe('consensus', () => {
  it('moves each node towards its neighbours and conserves the mean', () => {
    const n = 5
    const path = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (Math.abs(i - j) === 1 ? 1 : 0)),
    )
    const f = consensusField(path)
    expect(toFlat(f(0, tensor([1, 0, 0, 0, 0])))).toEqual([-1, 1, 0, 0, 0])
    const x0 = [3, -1, 4, 1, 5]
    const alg = flowAlgorithm(f, x0, { dt: 0.02, steps: 4000 })
    const y = runAlgorithm(alg, { maxSteps: 4000 }).state.y
    const mean = x0.reduce((a, b) => a + b, 0) / n
    for (const v of y) expect(v).toBeCloseTo(mean, 6)
  })

  it('agrees at the rate of the algebraic connectivity: the path graph’s 2 (1 - cos(pi / n))', () => {
    const n = 6
    const path = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (Math.abs(i - j) === 1 ? 1 : 0)),
    )
    const lambda2 = 2 * (1 - Math.cos(Math.PI / n))
    const x0 = Array.from({ length: n }, (_, i) => Math.cos(((i + 0.5) * Math.PI) / n))
    const spread = (steps: number): number => {
      const y = runAlgorithm(flowAlgorithm(consensusField(path), x0, { dt: 0.01, steps }), { maxSteps: steps }).state.y
      return Math.max(...y) - Math.min(...y)
    }
    // The start is the Fiedler vector, so the spread decays as exp(-lambda2 t) from the first.
    const rate = Math.log(spread(100) / spread(300)) / 2
    expect(rate).toBeCloseTo(lambda2, 5)
  })
})
