import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { item, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { dareContract, solveDare, solveDareAlgorithm } from './equations.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const a = [
  [1, 0.1],
  [0, 1],
]
const b = [[0], [0.1]]
const q = [
  [1, 0],
  [0, 1],
]
const r = [[1]]

describe('solveDareAlgorithm', () => {
  it('doubles until H stops changing, with quadratically falling changes', () => {
    const run = runAlgorithm(solveDareAlgorithm(a, b, q, r), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(run.state.change).toBeLessThanOrEqual(1e-14)
    expect(run.steps).toBe(run.state.iterations)
    const x = toFlat(solveDare(a, b, q, r))
    // X solves the Riccati equation: X = A'XA - A'XB (R + B'XB)^-1 B'XA + Q.
    const xm = [
      [x[0] ?? 0, x[1] ?? 0],
      [x[2] ?? 0, x[3] ?? 0],
    ]
    const at = (m: number[][]): number[][] => m[0]?.map((_, j) => m.map((row) => row[j] ?? 0)) ?? []
    const mul = (p: number[][], s: number[][]): number[][] =>
      p.map((row) => s[0]?.map((_, j) => row.reduce((t, v, k) => t + v * (s[k]?.[j] ?? 0), 0)) ?? [])
    const axa = mul(at(a), mul(xm, a))
    const bxa = mul(at(b), mul(xm, a))
    const gain = (r[0]?.[0] ?? 0) + (mul(at(b), mul(xm, b))[0]?.[0] ?? 0)
    const corr = mul(at(bxa), bxa).map((row) => row.map((v) => v / gain))
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        expect(xm[i]?.[j]).toBeCloseTo((axa[i]?.[j] ?? 0) - (corr[i]?.[j] ?? 0) + (q[i]?.[j] ?? 0), 9)
  })

  it('stays differentiable through the wrapper and agrees with the scalar solution', () => {
    expect(item(solveDare([[1]], [[1]], [[1]], [[1]]))).toBeCloseTo((1 + Math.sqrt(5)) / 2, 12)
    const dx = grad((qq: Value) => sum(solveDare([[1]], [[1]], qq, [[1]])))(tensor([[1]]))
    expect(toFlat(dx as Value)[0]).toBeGreaterThan(0)
  })

  it('serialises, resumes, fails on an unstabilisable pair, and logs', () => {
    const alg = solveDareAlgorithm(a, b, q, r)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state.iterations).toBe(whole.state.iterations)
    for (const [i, v] of toFlat(rest.state.hk).entries()) expect(v).toBeCloseTo(toFlat(whole.state.hk)[i] ?? 0, 10)
    expect(runAlgorithm(solveDareAlgorithm(a, b, q, r, { maxIterations: 1 }), { maxSteps: 100 }).status).toBe('failed')
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 100, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const residual = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'dare/residual' ? [e.value] : []))
    expect(residual).toHaveLength(whole.steps)
    expect(residual.at(-1) ?? 1).toBeLessThanOrEqual(1e-14)
    expect(Object.keys(dareContract)).toHaveLength(3)
  })
})
