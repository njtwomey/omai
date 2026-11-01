import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { splineContract, univariateSpline, univariateSplineAlgorithm } from './smoothing.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const xs = Array.from({ length: 40 }, (_, i) => i / 6)
const ys = xs.map((x, i) => Math.sin(x) + 0.02 * Math.sin(37 * i))

describe('univariateSplineAlgorithm', () => {
  it('inserts one knot per step while the residual exceeds s, as univariateSpline does', () => {
    const run = runAlgorithm(univariateSplineAlgorithm(xs, ys, { s: 0.002 }), { maxSteps: 60 })
    expect(run.status).toBe('converged')
    expect(run.state.total).toBeLessThanOrEqual(0.002)
    expect(run.steps).toBe(run.state.interior.length)
    const f = univariateSpline(xs, ys, { s: 0.002 })
    expect(f.interiorKnots).toEqual(run.state.interior)
    expect(f.residual).toBe(run.state.total)
    // Knots are sorted data points, strictly inside the range.
    expect([...run.state.interior].sort((p, q) => p - q)).toEqual(run.state.interior)
    for (const t of run.state.interior) expect(xs.includes(t)).toBe(true)
    const fitted = toFlat(f.spline.evaluate(xs))
    expect(fitted.reduce((sum, v, i) => sum + (v - (ys[i] ?? 0)) ** 2, 0)).toBeCloseTo(f.residual, 10)
  })

  it('has a residual that never rises as knots are added', () => {
    const rec = recorder()
    runAlgorithm(univariateSplineAlgorithm(xs, ys, { s: 1e-6 }), {
      maxSteps: 60,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const residual = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'spline/residual' ? [e.value] : []))
    expect(residual.length).toBeGreaterThan(3)
    for (let i = 1; i < residual.length; i++)
      expect(residual[i] ?? 0).toBeLessThanOrEqual((residual[i - 1] ?? 0) + 1e-12)
    expect(Object.keys(splineContract)).toHaveLength(3)
  })

  it('interpolates with s = 0 and resumes from JSON', () => {
    const pts = xs.slice(0, 12)
    const vals = ys.slice(0, 12)
    const alg = univariateSplineAlgorithm(pts, vals, { s: 0 })
    const whole = runAlgorithm(alg, { maxSteps: 60 })
    expect(whole.state.interpolated).toBe(true)
    expect(whole.state.total).toBeLessThan(1e-12)
    const part = runAlgorithm(alg, { maxSteps: 3 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 60,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 3, serialised: true },
    })
    expect(rest.state.interior).toEqual(whole.state.interior)
    expect(rest.state.total).toBeCloseTo(whole.state.total, 12)
    expect(() => univariateSplineAlgorithm([0, 1, 2], [0, 1, 0])).toThrow(/degree/)
  })
})
