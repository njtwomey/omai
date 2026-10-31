import { describe, expect, it } from 'vitest'
import { solve } from '../numerics/linalg/index.ts'
import { key, normal } from '../random/index.ts'
import { metricSeries, policies, recorder, runAlgorithm, tensorSeries } from '../trace/index.ts'
import { arx, arxRegression, rlsAlgorithm, rlsUpdate, simulateArx } from './identification.ts'
import { fromNumbers, numbersOf } from './ode/_vec.ts'

const input = (n: number, seed: number): number[] => Array.from(normal(key(seed), [n]).data)

describe('recursive least squares', () => {
  const plant = { a: [-1.2, 0.5], b: [1, 0.4] }
  const u = input(400, 1)

  it('recovers the parameters of a noise-free ARX system exactly, and its estimate equals the batch fit', () => {
    const { y } = simulateArx(plant, u)
    const reg = arxRegression(y, u, { na: 2, nb: 2 })
    const { state } = runAlgorithm(rlsAlgorithm(reg, { initialVariance: 1e8 }), { maxSteps: 1000 })
    const batch = arx(y, u, { na: 2, nb: 2 })
    const truth = [...plant.a, ...plant.b]
    for (const [i, v] of state.theta.entries()) expect(v).toBeCloseTo(truth[i] as number, 6)
    const fitted = [...batch.a, ...batch.b]
    for (const [i, v] of state.theta.entries()) expect(v).toBeCloseTo(fitted[i] as number, 6)
  })

  it('has the estimate after t samples the weighted regularised least squares of the prior and the data, for any forgetting factor', () => {
    const { y } = simulateArx(plant, u, { noise: 0.1, key: key(2) })
    const reg = arxRegression(y, u, { na: 2, nb: 2 })
    const t = 60
    const delta = 5
    const d = 4
    for (const lambda of [1, 0.98, 0.9, 0.8]) {
      let s = rlsAlgorithm(reg, { initialVariance: delta, forgetting: lambda }).init(null as never)
      for (let k = 0; k < t; k++) s = rlsUpdate(s, reg.phi[k] as number[], reg.y[k] as number, lambda)
      // (lambda^t I / delta + sum lambda^(t-1-k) phi phi')^-1 sum lambda^(t-1-k) phi y
      const A = new Float64Array(d * d)
      const b = new Float64Array(d)
      for (let i = 0; i < d; i++) A[i * d + i] = lambda ** t / delta
      for (let k = 0; k < t; k++) {
        const w = lambda ** (t - 1 - k)
        const phi = reg.phi[k] as number[]
        for (let i = 0; i < d; i++) {
          b[i] = (b[i] as number) + w * (phi[i] as number) * (reg.y[k] as number)
          for (let j = 0; j < d; j++)
            A[i * d + j] = (A[i * d + j] as number) + w * (phi[i] as number) * (phi[j] as number)
        }
      }
      const direct = numbersOf(solve(fromNumbers(A, [d, d]), fromNumbers(b, [d])))
      for (const [i, v] of s.theta.entries()) expect(v).toBeCloseTo(direct[i] as number, 8)
      // P stays symmetric.
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) expect(s.covariance[i * d + j]).toBeCloseTo(s.covariance[j * d + i] as number, 12)
    }
  })

  it('converges under noise at the rate of the data: the error falls like 1 / sqrt(n) and the covariance shrinks', () => {
    const long = input(2000, 5)
    const { y } = simulateArx(plant, long, { noise: 0.2, key: key(6) })
    const rec = recorder()
    runAlgorithm(rlsAlgorithm(arxRegression(y, long, { na: 2, nb: 2 })), {
      maxSteps: 3000,
      tracer: rec,
      policy: policies.teaching,
    })
    const traces = tensorSeries(rec.run(), 'fit/parameters')
    const truth = [...plant.a, ...plant.b]
    const errAt = (k: number): number =>
      Math.hypot(...(traces[k]?.values ?? []).map((v, i) => v - (truth[i] as number)))
    expect(errAt(1990)).toBeLessThan(errAt(30))
    expect(errAt(1990)).toBeLessThan(0.05)
    const trace = metricSeries(rec.run(), 'fit/covarianceTrace').values
    expect(trace.at(-1)).toBeLessThan((trace[50] as number) / 10)
    expect(metricSeries(rec.run(), 'fit/error').values.length).toBe(traces.length)
  })

  it('follows a change in the plant when it forgets, and a forgetting-free estimate does not', () => {
    const change = { at: 300, to: { a: [-0.6, 0.1], b: [2, -0.5] } }
    const { y } = simulateArx(plant, u, { change })
    const reg = arxRegression(y, u, { na: 2, nb: 2 })
    const target = [...change.to.a, ...change.to.b]
    const finalError = (forgetting: number): number => {
      const { state } = runAlgorithm(rlsAlgorithm(reg, { forgetting }), { maxSteps: 1000 })
      return Math.hypot(...Array.from(state.theta).map((v, i) => v - (target[i] as number)))
    }
    expect(finalError(0.9)).toBeLessThan(0.05)
    expect(finalError(1)).toBeGreaterThan(0.1)
  })

  it('serialises mid-run and resumes to the same estimate', () => {
    const { y } = simulateArx(plant, u, { noise: 0.1, key: key(8) })
    const alg = rlsAlgorithm(arxRegression(y, u, { na: 2, nb: 2 }), { forgetting: 0.99 })
    const straight = runAlgorithm(alg, { maxSteps: 500 }).state
    const half = runAlgorithm(alg, { maxSteps: 120 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.t) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.theta)).toEqual(Array.from(straight.theta))
  })

  it('builds the ARX regression and refuses bad input', () => {
    expect(arxRegression([1, 2, 3, 4], [5, 6, 7, 8], { na: 1, nb: 1 })).toEqual({
      phi: [
        [-1, 5],
        [-2, 6],
        [-3, 7],
      ],
      y: [2, 3, 4],
    })
    expect(() => arxRegression([1, 2], [1], { na: 1, nb: 1 })).toThrow()
    expect(() => rlsAlgorithm({ phi: [[1]], y: [1] }, { forgetting: 1.5 })).toThrow()
    expect(() => rlsAlgorithm({ phi: [[1], [1, 2]], y: [1, 2] })).toThrow()
  })
})
